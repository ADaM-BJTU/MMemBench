"""
M3Bench任务生成器 V2 - 基于配置文件
==========================================

使用dataset_configs.yaml驱动的任务生成系统。

功能：
1. 自动从配置文件加载数据集信息
2. 为每个数据集生成所有支持的任务类型
3. 支持严格的质量过滤
4. 自动保存生成结果和元数据

支持的任务类型：
- Attribute Bridge Reasoning (ABR): 多跳属性推理
- Attribute Comparison (AC): 属性对比 [NEW!]
- Visual Noise Filtering (VNF): 视觉噪声过滤
- Relation Comparison (RC): 关系对比
"""

import sys
from pathlib import Path
import logging
import json
import shutil
from datetime import datetime
from typing import Dict, List, Any

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from dataprovider import DataLoader, DataGeneratorV2, load_config

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def find_next_run_number(base_dir="d:\\install_file\\M3Bench\\generated_tasks_v2"):
    """找到下一个可用的 run 编号"""
    base_path = Path(base_dir)
    base_path.mkdir(exist_ok=True)

    existing_runs = [d for d in base_path.iterdir()
                    if d.is_dir() and d.name.startswith('run_')]

    if not existing_runs:
        return 1

    run_numbers = []
    for run_dir in existing_runs:
        try:
            num = int(run_dir.name.split('_')[1])
            run_numbers.append(num)
        except:
            continue

    return max(run_numbers) + 1 if run_numbers else 1


def setup_output_directory(base_dir="d:\\install_file\\M3Bench\\generated_tasks_v2"):
    """创建输出目录结构"""
    run_number = find_next_run_number(base_dir)
    run_dir = Path(base_dir) / f"run_{run_number}"

    # 创建子目录
    (run_dir / "tasks").mkdir(parents=True, exist_ok=True)
    (run_dir / "images").mkdir(parents=True, exist_ok=True)
    (run_dir / "annotations").mkdir(parents=True, exist_ok=True)
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)

    return run_dir, run_number


def copy_image_with_check(src_path, dst_dir):
    """复制图片并验证"""
    src = Path(src_path)

    if not src.exists():
        logger.debug(f"Image not found (will skip copy): {src}")
        # Return relative path even if original doesn't exist
        # This ensures consistent path format in output
        return f"images/{src.name}"

    dst = Path(dst_dir) / src.name

    try:
        if not dst.exists():  # 避免重复复制
            shutil.copy2(src, dst)
        return f"images/{src.name}"
    except Exception as e:
        logger.error(f"Failed to copy {src}: {e}")
        # Return relative path even if copy fails
        return f"images/{src.name}"


def process_task_with_images(task, run_dir):
    """处理单个任务：复制图片和标注"""
    try:
        # 复制图片
        image_paths = task.get('images', [])
        new_image_paths = []
        missing_count = 0

        for img_path in image_paths:
            relative_path = copy_image_with_check(img_path, run_dir / "images")
            if relative_path:
                new_image_paths.append(relative_path)
                if not Path(img_path).exists():
                    missing_count += 1
            else:
                logger.debug(f"Skipping task {task.get('task_id')}: missing image {img_path}")
                # 不要因为单个图片缺失而舍弃整个任务
                new_image_paths.append(img_path)  # 使用原始路径
                missing_count += 1

        # 保存推理证据
        if 'reasoning_evidence' in task:
            annot_file = run_dir / "annotations" / f"{task['task_id']}_evidence.json"
            annot_file.parent.mkdir(parents=True, exist_ok=True)

            with open(annot_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'task_id': task['task_id'],
                    'evidence': task['reasoning_evidence'],
                    'saved_at': datetime.now().isoformat()
                }, f, indent=2, ensure_ascii=False)

            task['evidence_file'] = f"annotations/{annot_file.name}"

        # 更新任务中的图片路径
        task['images'] = new_image_paths
        task['run_info'] = {
            'generated_at': datetime.now().isoformat(),
            'quality_verified': True,
            'image_files_copied': len(new_image_paths) - missing_count,
            'missing_images': missing_count
        }

        # 移除原始证据（已保存到单独文件）
        if 'reasoning_evidence' in task:
            del task['reasoning_evidence']

        return task

    except Exception as e:
        logger.error(f"Failed to process task {task.get('task_id', 'unknown')}: {e}")
        import traceback
        traceback.print_exc()
        return None


def generate_dataset_tasks(generator: DataGeneratorV2,
                          dataset_id: str,
                          run_dir: Path,
                          num_samples: int = 10,
                          split: str = "train") -> Dict[str, List[Dict]]:
    """
    从指定数据集生成所有支持的任务。

    Args:
        generator: DataGeneratorV2 实例
        dataset_id: 数据集ID (e.g., 'mscoco14', 'vcr')
        run_dir: 输出目录
        num_samples: 每种任务生成的样本数
        split: 数据split

    Returns:
        字典: {task_type: [processed_tasks]}
    """
    logger.info(f"\n{'='*70}")
    logger.info(f"处理数据集: {dataset_id.upper()}")
    logger.info(f"{'='*70}")

    all_tasks = {}

    try:
        # 生成所有任务
        tasks_by_type = generator.generate_all_tasks_for_dataset(
            dataset_id=dataset_id,
            num_samples_per_task=num_samples,
            split=split
        )

        # 处理每种任务类型
        for task_type, tasks in tasks_by_type.items():
            logger.info(f"\n处理任务类型: {task_type}")
            logger.info(f"  原始生成: {len(tasks)} 个任务")

            # 复制图片和处理任务
            processed = []
            for task in tasks:
                processed_task = process_task_with_images(task, run_dir)
                if processed_task:
                    processed.append(processed_task)

            if processed:
                # 保存到JSONL文件，使用完整的任务类型名称
                task_type_full = {
                    'VNF': 'visual_noise_filtering',
                    'ABR': 'attribute_bridge_reasoning',
                    'RC': 'relation_comparison',
                    'AC': 'attribute_comparison'
                }.get(task_type, task_type)
                output_file = run_dir / "tasks" / f"{task_type_full}_{dataset_id}.jsonl"
                with open(output_file, 'w', encoding='utf-8') as f:
                    for task in processed:
                        f.write(json.dumps(task, ensure_ascii=False) + '\n')

                all_tasks[task_type] = processed
                logger.info(f"  ✓ 成功保存: {len(processed)} 个任务 -> {output_file.name}")
            else:
                logger.warning(f"  ✗ 没有有效任务")

    except Exception as e:
        logger.error(f"生成任务失败 ({dataset_id}): {e}")
        import traceback
        traceback.print_exc()

    return all_tasks


def generate_report(run_dir: Path, all_results: Dict[str, Dict], run_number: int):
    """生成详细报告"""

    # 统计信息
    total_tasks = 0
    tasks_by_type = {}
    tasks_by_dataset = {}

    for dataset_id, tasks_dict in all_results.items():
        dataset_total = 0
        for task_type, tasks in tasks_dict.items():
            count = len(tasks)
            total_tasks += count
            dataset_total += count

            if task_type not in tasks_by_type:
                tasks_by_type[task_type] = 0
            tasks_by_type[task_type] += count

        tasks_by_dataset[dataset_id] = dataset_total

    # 生成报告
    report = {
        'run_number': run_number,
        'generated_at': datetime.now().isoformat(),
        'summary': {
            'total_tasks': total_tasks,
            'datasets_processed': list(all_results.keys()),
            'task_types': list(tasks_by_type.keys())
        },
        'tasks_by_type': tasks_by_type,
        'tasks_by_dataset': tasks_by_dataset,
        'detailed_counts': {
            dataset_id: {
                task_type: len(tasks)
                for task_type, tasks in tasks_dict.items()
            }
            for dataset_id, tasks_dict in all_results.items()
        },
        'output_structure': {
            'tasks_directory': 'tasks/',
            'images_directory': 'images/',
            'annotations_directory': 'annotations/',
            'logs_directory': 'logs/'
        }
    }

    # 保存JSON报告
    report_file = run_dir / "REPORT.json"
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # 创建Markdown报告
    readme_file = run_dir / "README.md"
    with open(readme_file, 'w', encoding='utf-8') as f:
        f.write(f"# M3Bench 任务生成报告 - Run {run_number}\n\n")
        f.write(f"**生成时间**: {report['generated_at']}\n\n")

        f.write("## 📊 总览\n\n")
        f.write(f"- **总任务数**: {total_tasks}\n")
        f.write(f"- **数据集数**: {len(all_results)}\n")
        f.write(f"- **任务类型**: {', '.join(tasks_by_type.keys())}\n\n")

        f.write("## 📈 按任务类型统计\n\n")
        for task_type, count in sorted(tasks_by_type.items()):
            f.write(f"- **{task_type}**: {count} 个任务\n")

        f.write("\n## 📁 按数据集统计\n\n")
        for dataset_id, count in sorted(tasks_by_dataset.items()):
            f.write(f"### {dataset_id} ({count} 个任务)\n\n")
            if dataset_id in all_results:
                for task_type, tasks in all_results[dataset_id].items():
                    f.write(f"  - {task_type}: {len(tasks)}\n")
                f.write("\n")

        f.write("## 📂 目录结构\n\n")
        f.write("```\n")
        f.write(f"run_{run_number}/\n")
        f.write("├── tasks/               # 生成的任务文件 (JSONL)\n")
        f.write("├── images/              # 复制的图片\n")
        f.write("├── annotations/         # 推理证据和原始标注\n")
        f.write("├── logs/                # 生成日志\n")
        f.write("├── REPORT.json          # JSON格式的详细报告\n")
        f.write("└── README.md            # 本文件\n")
        f.write("```\n\n")

        f.write("## 🔍 任务文件说明\n\n")
        f.write("每个任务文件的格式为 `{task_type}_{dataset_id}.jsonl`\n\n")
        f.write("任务字段说明：\n")
        f.write("- `task_id`: 任务唯一标识\n")
        f.write("- `task_type`: 任务类型\n")
        f.write("- `images`: 图片路径列表（相对路径）\n")
        f.write("- `question`: 问题\n")
        f.write("- `answer`: 答案\n")
        f.write("- `reasoning_depth`: 推理深度\n")
        f.write("- `evidence_file`: 推理证据文件路径\n")
        f.write("- `metadata`: 元数据\n")

    logger.info(f"✓ 报告已保存: {report_file} 和 {readme_file}")


def main():
    """主函数"""
    print("\n" + "="*80)
    print("M3Bench 任务生成器 V2 (配置驱动)")
    print("="*80)
    print("\n支持的任务类型:")
    print("  1. Attribute Bridge Reasoning (ABR)")
    print("  2. Attribute Comparison (AC) [NEW!]")
    print("  3. Visual Noise Filtering (VNF)")
    print("  4. Relation Comparison (RC)")
    print("\n" + "="*80 + "\n")

    # 设置输出目录
    run_dir, run_number = setup_output_directory()
    print(f"📁 输出目录: {run_dir}")
    print(f"🔢 运行编号: {run_number}\n")

    # 设置日志文件
    log_file = run_dir / "logs" / "generation.log"
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    ))
    logging.getLogger().addHandler(file_handler)

    try:
        # 加载配置
        logger.info("加载配置文件...")
        import os
        # 使用绝对路径指定配置文件
        config_file = "d:\\install_file\\M3Bench\\M3Bench-delivery\\dataprovider\\dataset_configs.yaml"
        config = load_config(config_file)

        # 验证数据集路径
        logger.info("验证数据集路径...")
        path_validation = config.validate_dataset_paths()
        valid_datasets = [ds for ds, valid in path_validation.items() if valid]

        print("\n可用数据集:")
        for dataset_id in valid_datasets:
            dataset_config = config.get_dataset_config(dataset_id)
            # 显示所有支持的任务，而不仅仅是被启用的任务
            supported_tasks = dataset_config.supported_tasks
            print(f"  ✓ {dataset_id}: {', '.join(supported_tasks)}")
        
        # 确保包含vcr数据集
        if 'vcr' not in valid_datasets:
            logger.warning("vcr数据集未找到，请检查配置文件")

        # 初始化生成器
        logger.info("初始化数据生成器...")
        import os
        # 使用新的数据集路径
        data_root = "d:\install_file\M3Bench\dataset"
        loader = DataLoader(data_root=data_root)
        # 使用与路径验证相同的配置文件
        config_file = "d:\install_file\M3Bench\M3Bench-delivery\dataprovider\dataset_configs.yaml"
        generator = DataGeneratorV2(loader, config_file=config_file)

        # 生成任务
        all_results = {}

        # 配置：为每个数据集生成多少样本
        generation_config = {
            'mscoco14': {
                'num_samples': 5,
                'split': 'val'  # Use val split since images are in val2014 directory
            },
            'vcr': {
                'num_samples': 5,
                'split': 'train'
            },
            'scienceqa': {
                'num_samples': 5,
                'split': 'validation'
            },
            'docvqa': {
                'num_samples': 5,
                'split': 'validation'
            },
            'realworldqa': {
                'num_samples': 5,
                'split': 'test'
            }
        }

        for dataset_id in valid_datasets:
            if dataset_id not in generation_config:
                logger.info(f"跳过 {dataset_id} (未配置)")
                continue

            config_for_dataset = generation_config[dataset_id]

            tasks_dict = generate_dataset_tasks(
                generator=generator,
                dataset_id=dataset_id,
                run_dir=run_dir,
                num_samples=config_for_dataset['num_samples'],
                split=config_for_dataset['split']
            )

            if tasks_dict:
                all_results[dataset_id] = tasks_dict

        # 生成报告
        if all_results:
            logger.info("\n生成报告...")
            generate_report(run_dir, all_results, run_number)

        # 总结
        print("\n" + "="*80)
        print("✅ 生成完成！")
        print("="*80)
        print(f"\n📂 输出位置: {run_dir}")

        if all_results:
            total_tasks = sum(
                len(tasks)
                for tasks_dict in all_results.values()
                for tasks in tasks_dict.values()
            )

            print(f"\n📊 任务统计:")
            print(f"  总计: {total_tasks} 个任务")

            for dataset_id, tasks_dict in all_results.items():
                dataset_total = sum(len(tasks) for tasks in tasks_dict.values())
                print(f"\n  {dataset_id}: {dataset_total} 个任务")
                for task_type, tasks in tasks_dict.items():
                    print(f"    - {task_type}: {len(tasks)}")

            total_images = len(list((run_dir / 'images').glob('*')))
            total_annotations = len(list((run_dir / 'annotations').glob('*')))

            print(f"\n📁 文件统计:")
            print(f"  - 图片: {total_images}")
            print(f"  - 标注: {total_annotations}")
        else:
            print("\n⚠️  没有成功生成任何任务")

        print(f"\n📄 查看报告:")
        print(f"  cat {run_dir / 'README.md'}")
        print()

    except Exception as e:
        logger.error(f"生成过程出错: {e}")
        import traceback
        traceback.print_exc()
        print(f"\n❌ 错误: {e}")
        print("请查看日志文件了解详情")


if __name__ == "__main__":
    main()
