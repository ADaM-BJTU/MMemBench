#!/usr/bin/env python3
"""
从simulator_test_log中抽取样本用于人类标注
"""

import json
import random
from pathlib import Path
from typing import List, Dict, Any
from collections import defaultdict


def load_all_logs(log_dir: str) -> List[Dict]:
    """加载所有日志文件（支持多种格式）"""
    log_path = Path(log_dir)
    all_samples = []

    # 加载标准 run_log 文件
    for log_file in log_path.glob("run_log_*.json"):
        print(f"  Loading {log_file.name}...")
        with open(log_file, 'r', encoding='utf-8') as f:
            events = json.load(f)
            samples = extract_samples_from_log(events)
            all_samples.extend(samples)

    # 加载batch_run子目录中的日志
    for batch_dir in log_path.glob("batch_run_*"):
        if batch_dir.is_dir():
            print(f"  Loading batch logs from {batch_dir.name}/...")
            for log_file in batch_dir.glob("run_log_*.json"):
                with open(log_file, 'r', encoding='utf-8') as f:
                    events = json.load(f)
                    samples = extract_samples_from_log(events)
                    all_samples.extend(samples)

    # 加载converted子目录中的日志
    converted_dir = log_path / "converted"
    if converted_dir.exists():
        print(f"  Loading converted logs from converted/...")
        for log_file in converted_dir.glob("run_log_*.json"):
            with open(log_file, 'r', encoding='utf-8') as f:
                events = json.load(f)
                samples = extract_samples_from_log(events)
                all_samples.extend(samples)

    # 尝试加载 batch_test_results 目录（直接解析batch格式）
    batch_results_dir = Path("batch_test_results")
    if batch_results_dir.exists():
        print(f"  Loading batch_test_results/...")
        for batch_file in batch_results_dir.glob("batch_results_*.json"):
            samples = extract_samples_from_batch_result(batch_file)
            all_samples.extend(samples)

    return all_samples


def extract_samples_from_batch_result(batch_file: Path) -> List[Dict]:
    """从batch_results文件直接提取样本"""
    samples = []

    try:
        with open(batch_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 判断数据结构
        if isinstance(data, list):
            batch_results = data
        elif 'results' in data:
            batch_results = data['results']
        elif 'batch_id' in data:
            batch_results = [data]
        else:
            return samples

        for batch_result in batch_results:
            task_results = batch_result.get('task_results', [])

            for task_result in task_results:
                task_id = task_result.get('task_id', 'unknown')
                task_type = task_result.get('task_type', 'unknown')

                # 如果有详细的turns信息
                turns = task_result.get('turns', [])
                if turns:
                    for turn in turns:
                        sample = {
                            "sample_id": f"{task_id}_turn_{turn.get('turn', 0)}",
                            "task_id": task_id,
                            "task_type": task_type,
                            "turn_number": turn.get('turn', 0),
                            "action_type": turn.get('action', 'unknown'),
                            "user_message": turn.get('query', turn.get('message', '')),
                            "vlm_response": turn.get('response', ''),
                            "expected_answer": task_result.get('expected_answer', ''),
                            "images_sent": turn.get('images_sent', []),
                            "auto_evaluation": turn.get('evaluation', {}),
                            "timestamp": turn.get('timestamp', '')
                        }
                        samples.append(sample)
                else:
                    # 没有详细turns，创建摘要样本
                    sample = {
                        "sample_id": f"{task_id}_summary",
                        "task_id": task_id,
                        "task_type": task_type,
                        "turn_number": 0,
                        "action_type": "summary",
                        "user_message": task_result.get('question', ''),
                        "vlm_response": "[Detailed response not available]",
                        "expected_answer": task_result.get('expected_answer', ''),
                        "images_sent": task_result.get('images', []),
                        "auto_evaluation": task_result.get('scores', {}),
                        "timestamp": ""
                    }
                    samples.append(sample)

    except Exception as e:
        print(f"  Warning: Failed to load {batch_file.name}: {e}")

    return samples


def extract_samples_from_log(events: List[Dict]) -> List[Dict]:
    """从事件日志中提取样本"""
    samples = []
    task_info = {}

    for i, event in enumerate(events):
        if event.get("event") == "task_start":
            task_info = {
                "task_id": event.get("task_id", "unknown"),
                "task_type": event.get("task_type", "unknown"),
                "question": event.get("question", ""),
                "expected_answer": event.get("expected_answer", ""),
                "images": event.get("images", [])
            }

        elif event.get("event") == "target_model_response":
            # 找到对应的core_model_decision
            turn = event.get("turn", 0)
            user_message = ""
            action = ""

            # 向前查找core_model_decision
            for prev_event in reversed(events[:i]):
                if prev_event.get("event") == "core_model_decision" and prev_event.get("turn") == turn:
                    user_message = prev_event.get("message_to_model", "")
                    action = prev_event.get("action", "")
                    break

            if not task_info:
                continue

            sample = {
                "sample_id": f"{task_info['task_id']}_turn_{turn}",
                "task_id": task_info["task_id"],
                "task_type": task_info["task_type"],
                "turn_number": turn,
                "action_type": action,
                "user_message": user_message,
                "vlm_response": event.get("target_response", ""),
                "expected_answer": task_info["expected_answer"],
                "images_sent": event.get("images_sent", []),
                "auto_evaluation": event.get("evaluation", {}),
                "timestamp": event.get("timestamp", "")
            }
            samples.append(sample)

    return samples


def prioritize_samples(samples: List[Dict]) -> List[Dict]:
    """按优先级排序样本"""
    priority_map = {
        "mislead": 10,
        "mislead_subtle": 10,
        "memory_injection": 9,
        "cross_image_confusion": 9,
        "consistency_check": 8,
        "distraction": 5,
        "redundancy": 5,
        "follow_up": 3,
        "guidance": 2,
        "filler": 0
    }

    for sample in samples:
        action = sample.get("action_type", "")
        sample["priority"] = priority_map.get(action, 1)

    return sorted(samples, key=lambda x: x["priority"], reverse=True)


def stratified_sample(samples: List[Dict], n: int = 50) -> List[Dict]:
    """分层抽样"""
    if not samples:
        return []

    # 按任务类型分组
    by_task_type = defaultdict(list)
    for s in samples:
        by_task_type[s["task_type"]].append(s)

    if not by_task_type:
        return samples[:n]

    # 每种任务类型至少10个（如果有的话）
    selected = []
    per_type = max(10, n // len(by_task_type))

    for task_type, task_samples in by_task_type.items():
        # 优先选择高优先级样本
        task_samples = prioritize_samples(task_samples)
        selected.extend(task_samples[:per_type])

    # 如果不够，补充高优先级样本
    if len(selected) < n:
        remaining = [s for s in samples if s not in selected]
        remaining = prioritize_samples(remaining)
        selected.extend(remaining[:n - len(selected)])

    return selected[:n]


def export_to_jsonl(samples: List[Dict], output_file: str):
    """导出为JSONL格式"""
    with open(output_file, 'w', encoding='utf-8') as f:
        for sample in samples:
            # 添加空的human_annotation字段
            sample["human_annotation"] = {
                "correctness": None,
                "reasoning_completeness": None,
                "resists_misleading": None,
                "context_consistency": None,
                "overall_quality": None,
                "comments": ""
            }
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')


def export_to_excel(samples: List[Dict], output_file: str):
    """导出为Excel格式（可选）"""
    try:
        import pandas as pd

        rows = []
        for sample in samples:
            rows.append({
                "sample_id": sample["sample_id"],
                "task_type": sample["task_type"],
                "turn": sample["turn_number"],
                "action": sample["action_type"],
                "question": sample["user_message"][:100] + "..." if len(sample["user_message"]) > 100 else sample["user_message"],
                "response": sample["vlm_response"][:200] + "..." if len(sample["vlm_response"]) > 200 else sample["vlm_response"],
                "expected": sample["expected_answer"],
                "auto_score": sample.get("auto_evaluation", {}).get("score", "N/A"),
                # 标注列
                "correctness_1_5": "",
                "reasoning_1_5": "",
                "resists_mislead_Y_N_NA": "",
                "consistency_Y_N_NA": "",
                "overall_1_5": "",
                "comments": ""
            })

        df = pd.DataFrame(rows)
        df.to_excel(output_file, index=False)
        print(f"Excel template exported to: {output_file}")
    except ImportError:
        print("pandas not installed, skipping Excel export")
        print("To enable Excel export, run: pip install pandas openpyxl")


def main():
    # 配置
    log_dir = "simulator_test_log"
    output_dir = Path("human_annotations")
    output_dir.mkdir(exist_ok=True)

    num_samples = 50  # 可调整

    print(f"Loading logs from {log_dir}...")
    all_samples = load_all_logs(log_dir)
    print(f"Found {len(all_samples)} total samples")

    if len(all_samples) == 0:
        print("No samples found! Please run the simulator first to generate logs.")
        return

    print(f"\nSelecting {num_samples} samples using stratified sampling...")
    selected = stratified_sample(all_samples, n=num_samples)
    print(f"Selected {len(selected)} samples")

    # 统计
    action_counts = defaultdict(int)
    task_counts = defaultdict(int)
    for s in selected:
        action_counts[s["action_type"]] += 1
        task_counts[s["task_type"]] += 1

    print("\nSample distribution:")
    print("By action type:")
    for action, count in sorted(action_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {action}: {count}")
    print("\nBy task type:")
    for task, count in task_counts.items():
        print(f"  {task}: {count}")

    # 导出
    jsonl_file = output_dir / "annotation_samples.jsonl"
    export_to_jsonl(selected, str(jsonl_file))
    print(f"\nSamples exported to: {jsonl_file}")

    # 可选：导出Excel
    excel_file = output_dir / "annotation_template.xlsx"
    export_to_excel(selected, str(excel_file))

    print("\nNext steps:")
    print("1. Review annotation_samples.jsonl")
    print("2. Annotate using Excel or directly edit JSONL")
    print("3. Save annotated results as annotation_results.jsonl")
    print("4. Run validate_evaluation.py to analyze results")


if __name__ == "__main__":
    main()
