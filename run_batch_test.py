"""
批处理测试运行脚本
==================

用于运行批处理任务测试，在单个session中测试多个任务，支持长上下文测试。

用法:
    # 基本用法
    python run_batch_test.py --tasks-per-batch 4 --num-batches 5

    # 指定任务文件
    python run_batch_test.py --task-files "generated_tasks_v2/run_*/tasks/*.jsonl"

    # 详细输出
    python run_batch_test.py --tasks-per-batch 3 --num-batches 2 --verbose
"""

import sys
import json
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

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


def load_tasks_from_files(task_files: List[str], max_tasks: int = None) -> List[Dict]:
    """从JSONL文件加载任务

    Args:
        task_files: 任务文件路径模式列表
        max_tasks: 最大加载任务数

    Returns:
        任务字典列表
    """
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


def split_into_batches(tasks: List[Dict], batch_size: int) -> List[List[Dict]]:
    """将任务分成批次

    Args:
        tasks: 任务列表
        batch_size: 每批任务数

    Returns:
        批次列表
    """
    batches = []
    for i in range(0, len(tasks), batch_size):
        batches.append(tasks[i:i + batch_size])
    return batches


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="M3Bench 批处理任务测试",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 使用默认任务文件运行批处理测试
  python run_batch_test.py

  # 自定义批次大小和数量
  python run_batch_test.py --tasks-per-batch 4 --num-batches 5

  # 指定任务文件
  python run_batch_test.py --task-files generated_tasks_v2/run_1/tasks/*.jsonl

  # 调整轮数限制
  python run_batch_test.py --max-turns-per-session 60 --min-turns-per-task 10
        """
    )

    # 任务来源
    parser.add_argument(
        '--task-files', nargs='+',
        default=['generated_tasks_v2/run_*/tasks/*.jsonl'],
        help='任务文件路径模式 (默认: generated_tasks_v2/run_*/tasks/*.jsonl)'
    )
    parser.add_argument(
        '--max-tasks', type=int, default=50,
        help='最大加载任务数 (默认: 50)'
    )

    # 批处理配置
    parser.add_argument(
        '--tasks-per-batch', type=int, default=4,
        help='每批任务数 (默认: 4)'
    )
    parser.add_argument(
        '--num-batches', type=int, default=5,
        help='运行批次数 (默认: 5)'
    )
    parser.add_argument(
        '--max-turns-per-session', type=int, default=50,
        help='每session最大轮数 (默认: 50)'
    )
    parser.add_argument(
        '--min-turns-per-task', type=int, default=8,
        help='每任务最小轮数 (默认: 8)'
    )
    parser.add_argument(
        '--max-turns-per-task', type=int, default=15,
        help='每任务最大轮数 (默认: 15)'
    )
    parser.add_argument(
        '--transition-style', choices=['natural', 'abrupt', 'contextual'],
        default='natural',
        help='任务过渡风格 (默认: natural)'
    )

    # 跨任务记忆测试
    parser.add_argument(
        '--enable-memory-test', action='store_true', default=True,
        help='启用跨任务记忆测试 (默认: 启用)'
    )
    parser.add_argument(
        '--disable-memory-test', action='store_true',
        help='禁用跨任务记忆测试'
    )
    parser.add_argument(
        '--memory-test-interval', type=int, default=3,
        help='跨任务记忆测试间隔 (默认: 每3个任务)'
    )

    # 模型配置
    parser.add_argument(
        '--model', type=str, default=None,
        help='指定target_model (如: gemini-3-flash-preview-nothinking, gpt-4o)'
    )
    parser.add_argument(
        '--core-model', type=str, default=None,
        help='指定core_model (默认: gemini-3-pro-image-preview)'
    )

    # 输出
    parser.add_argument(
        '--output-dir', default='batch_test_results',
        help='输出目录 (默认: batch_test_results)'
    )
    parser.add_argument(
        '--verbose', '-v', action='store_true',
        help='详细输出'
    )

    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()

    print("\n" + "="*80)
    print("M3Bench 批处理任务测试")
    print("="*80 + "\n")

    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 加载任务
    tasks = load_tasks_from_files(args.task_files, args.max_tasks)

    if not tasks:
        logger.error("没有加载到任何任务!")
        print("\n错误: 没有找到任务文件")
        print("请先运行 generate_all_tasks_v2.py 生成任务")
        return 1

    # 分批
    batches = split_into_batches(tasks, args.tasks_per_batch)
    batches = batches[:args.num_batches]

    print(f"任务总数: {len(tasks)}")
    print(f"批次数: {len(batches)}")
    print(f"每批任务数: {args.tasks_per_batch}")
    print(f"每session最大轮数: {args.max_turns_per_session}")
    if args.model:
        print(f"Target Model: {args.model}")
    if args.core_model:
        print(f"Core Model: {args.core_model}")
    print()

    # 初始化组件 - 支持自定义模型
    llm_kwargs = {}
    if args.model:
        llm_kwargs['target_model'] = args.model
    if args.core_model:
        llm_kwargs['core_model'] = args.core_model
    llm_client = LLMClient(**llm_kwargs)
    evaluator = Evaluator(mode=EvaluationMode.STRESS_TEST)

    # 配置批处理
    enable_memory_test = args.enable_memory_test and not args.disable_memory_test
    config = BatchConfig(
        max_turns_per_session=args.max_turns_per_session,
        min_turns_per_task=args.min_turns_per_task,
        max_turns_per_task=args.max_turns_per_task,
        transition_style=args.transition_style,
        enable_cross_task_memory_test=enable_memory_test,
        cross_task_memory_interval=args.memory_test_interval
    )

    batch_simulator = BatchTaskSimulator(
        llm_client=llm_client,
        evaluator=evaluator,
        config=config,
        verbose=args.verbose
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
            all_results.append(result.to_dict())

            logger.info(f"批次 {batch_idx + 1} 完成:")
            logger.info(f"  总轮数: {result.total_turns}")
            logger.info(f"  完成任务: {result.tasks_completed}/{result.tasks_attempted}")
            logger.info(f"  聚合分数: {result.aggregate_scores}")

        except Exception as e:
            logger.error(f"批次 {batch_idx + 1} 失败: {e}")
            import traceback
            traceback.print_exc()

    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    # 保存结果
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = output_dir / f"batch_results_{timestamp}.json"

    summary = {
        'total_batches': len(all_results),
        'total_turns': sum(r.get('total_turns', 0) for r in all_results),
        'total_tasks_completed': sum(r.get('tasks_completed', 0) for r in all_results),
        'total_tasks_attempted': sum(r.get('tasks_attempted', 0) for r in all_results),
        'average_turns_per_batch': (
            sum(r.get('total_turns', 0) for r in all_results) / len(all_results)
            if all_results else 0
        ),
        'duration_seconds': duration
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            'run_time': timestamp,
            'config': {
                'tasks_per_batch': args.tasks_per_batch,
                'num_batches': len(batches),
                'max_turns_per_session': args.max_turns_per_session,
                'min_turns_per_task': args.min_turns_per_task,
                'max_turns_per_task': args.max_turns_per_task,
                'transition_style': args.transition_style,
                'enable_memory_test': enable_memory_test
            },
            'results': all_results,
            'summary': summary
        }, f, indent=2, ensure_ascii=False)

    logger.info(f"\n结果已保存: {output_file}")

    # 打印总结
    print("\n" + "="*60)
    print("批处理测试完成")
    print("="*60)
    print(f"总批次: {summary['total_batches']}")
    print(f"总轮数: {summary['total_turns']}")
    print(f"完成任务: {summary['total_tasks_completed']}/{summary['total_tasks_attempted']}")
    print(f"平均轮数/批: {summary['average_turns_per_batch']:.1f}")
    print(f"总耗时: {summary['duration_seconds']:.1f}s")
    print(f"\n结果文件: {output_file}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
