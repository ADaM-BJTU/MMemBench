#!/usr/bin/env python3
"""
将batch_results格式转换为run_log格式

Batch Results格式 (来自BatchTaskSimulator):
{
    "batch_id": "...",
    "task_results": [
        {"task_id": "...", "scores": {...}, ...}
    ],
    ...
}

Run Log格式 (用于Human Annotation):
[
    {"event": "task_start", "task_id": "...", ...},
    {"event": "core_model_decision", "turn": 0, "message_to_model": "...", ...},
    {"event": "target_model_response", "turn": 0, "target_response": "...", ...},
    ...
]

用法:
    python tools/convert_batch_results_to_runlog.py [batch_results_file] [output_dir]

示例:
    python tools/convert_batch_results_to_runlog.py batch_test_results/batch_results_20260126_102200.json simulator_test_log/converted/
"""

import json
import sys
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional

# Task 1D: Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def convert_batch_result_to_events(batch_result: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    将单个batch结果转换为事件流

    Args:
        batch_result: batch结果字典

    Returns:
        事件列表，符合run_log格式
    """
    events = []
    timestamp = batch_result.get('start_time', datetime.now().isoformat())

    # 添加batch_start事件
    events.append({
        "event": "batch_start",
        "batch_id": batch_result.get('batch_id', 'unknown'),
        "timestamp": timestamp
    })

    # 处理每个任务
    task_results = batch_result.get('task_results', [])

    for task_result in task_results:
        task_id = task_result.get('task_id', 'unknown')
        task_type = task_result.get('task_type', 'unknown')

        # task_start事件
        events.append({
            "event": "task_start",
            "task_id": task_id,
            "task_type": task_type,
            "question": task_result.get('question', ''),
            "expected_answer": task_result.get('expected_answer', ''),
            "images": task_result.get('images', []),
            "timestamp": timestamp
        })

        # 如果有详细的对话日志 (turns字段)，添加turn事件
        turns = task_result.get('turns', [])
        if turns:
            # Task 1D: Log when using real turn data
            logger.info(f"Task {task_id}: Using {len(turns)} turns from conversation history")
            for turn in turns:
                turn_num = turn.get('turn', 0)

                # user_query (core_model_decision)
                events.append({
                    "event": "core_model_decision",
                    "turn": turn_num,
                    "action": turn.get('action', 'guidance'),
                    "message_to_model": turn.get('query', turn.get('message', '')),
                    "reasoning": turn.get('reasoning', ''),
                    "task_progress": turn.get('task_progress', 'incomplete'),
                    "timestamp": turn.get('timestamp', timestamp)
                })

                # vlm_response (target_model_response)
                events.append({
                    "event": "target_model_response",
                    "turn": turn_num,
                    "target_response": turn.get('response', turn.get('vlm_response', '')),
                    "images_sent": turn.get('images_sent', []),
                    "evaluation": turn.get('evaluation', {}),
                    "timestamp": turn.get('timestamp', timestamp)
                })
        else:
            # 如果没有详细对话，创建占位事件
            # Task 1D: Warning when using placeholders
            turns_used = task_result.get('turns_used', 1)
            logger.warning(f"Task {task_id}: No turn details available, using {turns_used} placeholder events")
            for t in range(turns_used):
                events.append({
                    "event": "core_model_decision",
                    "turn": t,
                    "action": "guidance",
                    "message_to_model": f"[Turn {t} query - detailed log not available]",
                    "task_progress": "incomplete" if t < turns_used - 1 else "complete",
                    "timestamp": timestamp
                })

                events.append({
                    "event": "target_model_response",
                    "turn": t,
                    "target_response": f"[Turn {t} response - detailed log not available]",
                    "images_sent": task_result.get('images', []) if t == 0 else [],
                    "evaluation": {},
                    "timestamp": timestamp
                })

        # task_end事件
        events.append({
            "event": "task_end",
            "task_id": task_id,
            "completed": task_result.get('completed', False),
            "scores": task_result.get('scores', {}),
            "turns_used": task_result.get('turns_used', 0),
            "timestamp": timestamp
        })

    # batch_end事件
    events.append({
        "event": "batch_end",
        "batch_id": batch_result.get('batch_id', 'unknown'),
        "tasks_completed": batch_result.get('tasks_completed', 0),
        "tasks_attempted": batch_result.get('tasks_attempted', 0),
        "total_turns": batch_result.get('total_turns', 0),
        "timestamp": batch_result.get('end_time', timestamp)
    })

    return events


def convert_batch_file(input_file: str, output_dir: str) -> List[str]:
    """
    转换整个batch结果文件

    Args:
        input_file: 输入的batch_results JSON文件路径
        output_dir: 输出目录

    Returns:
        生成的run_log文件路径列表
    """
    input_path = Path(input_file)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Loading batch results from: {input_file}")

    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 判断数据结构
    # 可能是单个batch结果或结果列表
    if isinstance(data, list):
        batch_results = data
    elif 'results' in data:
        batch_results = data['results']
    elif 'batch_id' in data:
        batch_results = [data]
    else:
        print(f"Unknown data format in {input_file}")
        return []

    output_files = []

    for i, batch_result in enumerate(batch_results):
        events = convert_batch_result_to_events(batch_result)

        batch_id = batch_result.get('batch_id', f'batch_{i+1}')
        safe_batch_id = batch_id.replace('/', '_').replace('\\', '_')

        output_file = output_path / f"run_log_{safe_batch_id}.json"

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(events, f, indent=2, ensure_ascii=False)

        print(f"  Converted batch {i+1} -> {output_file.name} ({len(events)} events)")
        output_files.append(str(output_file))

    return output_files


def convert_batch_directory(batch_dir: str, output_dir: str) -> List[str]:
    """
    转换目录下所有batch结果文件

    Args:
        batch_dir: batch_test_results目录
        output_dir: 输出目录

    Returns:
        生成的run_log文件路径列表
    """
    batch_path = Path(batch_dir)
    all_output_files = []

    for batch_file in batch_path.glob("batch_results_*.json"):
        output_files = convert_batch_file(str(batch_file), output_dir)
        all_output_files.extend(output_files)

    return all_output_files


def print_conversion_summary(output_files: List[str]):
    """打印转换摘要"""
    print("\n" + "="*60)
    print("Conversion Summary")
    print("="*60)
    print(f"Total files converted: {len(output_files)}")

    if output_files:
        # 统计事件
        total_events = 0
        total_tasks = 0
        total_turns = 0

        for f in output_files:
            with open(f, 'r', encoding='utf-8') as fp:
                events = json.load(fp)
                total_events += len(events)
                total_tasks += sum(1 for e in events if e.get('event') == 'task_start')
                total_turns += sum(1 for e in events if e.get('event') == 'core_model_decision')

        print(f"Total events: {total_events}")
        print(f"Total tasks: {total_tasks}")
        print(f"Total turns: {total_turns}")

    print("\nOutput files:")
    for f in output_files[:10]:
        print(f"  - {f}")
    if len(output_files) > 10:
        print(f"  ... and {len(output_files) - 10} more")


def main():
    """主函数"""
    # 默认路径
    default_input = "batch_test_results"
    default_output = "simulator_test_log/converted"

    # 解析命令行参数
    if len(sys.argv) >= 2:
        input_path = sys.argv[1]
    else:
        input_path = default_input

    if len(sys.argv) >= 3:
        output_dir = sys.argv[2]
    else:
        output_dir = default_output

    print("="*60)
    print("Batch Results to Run Log Converter")
    print("="*60)
    print(f"Input: {input_path}")
    print(f"Output: {output_dir}")
    print()

    input_path_obj = Path(input_path)

    if input_path_obj.is_file():
        # 单个文件
        output_files = convert_batch_file(input_path, output_dir)
    elif input_path_obj.is_dir():
        # 目录
        output_files = convert_batch_directory(input_path, output_dir)
    else:
        print(f"Error: {input_path} does not exist")
        return

    print_conversion_summary(output_files)

    print("\nNext steps:")
    print("1. Run extract_annotation_samples.py to extract samples from converted logs")
    print("2. Or use the converted logs directly with Human Annotation tools")


if __name__ == "__main__":
    main()
