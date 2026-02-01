"""
运行多模态模型实验
=================

这是一个简单的包装脚本，用于运行 gpt-4o 多模态模型实验。
基于 run_batch_test.py，但指定了 target_model 为 gpt-4o。
"""

import sys
import json
import logging
from pathlib import Path
from datetime import datetime

# Add project root
sys.path.insert(0, str(Path(__file__).parent))

from src.simulator import (
    BatchTaskSimulator,
    BatchConfig,
    LLMClient,
    Evaluator,
    EvaluationMode
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_tasks_from_files(task_files, max_tasks=None):
    """从JSONL文件加载任务"""
    all_tasks = []

    for file_pattern in task_files:
        # 处理glob模式
        if '*' in file_pattern:
            matching_files = list(Path().glob(file_pattern))
        else:
            matching_files = [Path(file_pattern)]

        for file_path in matching_files:
            if not file_path.exists():
                logger.warning(f"文件不存在: {file_path}")
                continue

            logger.info(f"加载任务: {file_path}")

            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    if line.strip():
                        try:
                            task = json.loads(line)
                            all_tasks.append(task)
                        except json.JSONDecodeError as e:
                            logger.warning(f"跳过无效JSON ({file_path}:{line_num}): {e}")
                            continue

    if max_tasks and len(all_tasks) > max_tasks:
        all_tasks = all_tasks[:max_tasks]

    logger.info(f"共加载 {len(all_tasks)} 个任务")
    return all_tasks


def split_into_batches(tasks, batch_size):
    """将任务分成批次"""
    batches = []
    for i in range(0, len(tasks), batch_size):
        batches.append(tasks[i:i + batch_size])
    return batches


def main():
    """主函数"""
    print("\n" + "="*80)
    print("M3Bench 多模态模型实验 (GPT-4o)")
    print("="*80 + "\n")

    # 配置参数
    task_files = ["generated_tasks_v2/run_18/tasks/*.jsonl"]
    tasks_per_batch = 3  # Restored to 3 tasks per batch
    num_batches = 3      # Restored to 3 batches
    output_dir = Path("test_output/final_mm")

    # 创建输出目录
    output_dir.mkdir(parents=True, exist_ok=True)

    # 加载任务
    tasks = load_tasks_from_files(task_files, max_tasks=50)

    if not tasks:
        logger.error("没有加载到任何任务!")
        print("\n错误: 没有找到任务文件")
        return 1

    # 分批
    batches = split_into_batches(tasks, tasks_per_batch)
    batches = batches[:num_batches]

    print(f"任务总数: {len(tasks)}")
    print(f"批次数: {len(batches)}")
    print(f"每批任务数: {tasks_per_batch}")
    print(f"目标模型: gpt-4o")
    print()

    # 初始化组件 - 指定 gpt-4o 作为 target_model
    llm_client = LLMClient(
        core_model="gemini-3-pro-image-preview",
        target_model="gpt-4o",  # 多模态模型
        weak_model="claude-3-5-haiku-20241022-c"
    )
    evaluator = Evaluator(mode=EvaluationMode.STRESS_TEST)

    # 配置批处理
    config = BatchConfig(
        max_turns_per_session=50,
        min_turns_per_task=8,
        max_turns_per_task=15,
        transition_style='natural',
        enable_cross_task_memory_test=True,
        cross_task_memory_interval=3
    )

    batch_simulator = BatchTaskSimulator(
        llm_client=llm_client,
        evaluator=evaluator,
        config=config,
        verbose=True
    )

    # 运行批处理
    all_results = []
    start_time = datetime.now()

    for batch_idx, batch in enumerate(batches):
        logger.info(f"\n{'='*60}")
        logger.info(f"运行批次 {batch_idx + 1}/{len(batches)}")
        logger.info(f"任务: {[t.get('task_id', 'unknown') for t in batch]}")
        logger.info(f"{'='*60}")

        try:
            result = batch_simulator.run_batch(batch)
            all_results.append(result)

            # 保存当前批次结果
            batch_output_file = output_dir / f"batch_results_{batch_idx + 1}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(batch_output_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)

            logger.info(f"批次 {batch_idx + 1} 完成，结果已保存到: {batch_output_file}")

        except Exception as e:
            logger.error(f"批次 {batch_idx + 1} 运行失败: {e}")
            import traceback
            traceback.print_exc()
            continue

    # 汇总统计
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    print("\n" + "="*80)
    print("实验完成")
    print("="*80)
    print(f"总批次数: {len(batches)}")
    print(f"成功批次数: {len(all_results)}")
    print(f"总耗时: {duration:.1f} 秒")
    print(f"结果保存在: {output_dir}")
    print()

    # 保存最终汇总
    summary = {
        "timestamp": datetime.now().isoformat(),
        "model": "gpt-4o",
        "total_batches": len(batches),
        "successful_batches": len(all_results),
        "duration_seconds": duration,
        "output_directory": str(output_dir)
    }

    summary_file = output_dir / "experiment_summary.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    logger.info(f"汇总报告已保存到: {summary_file}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
