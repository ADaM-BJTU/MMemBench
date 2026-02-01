"""
测试BatchTaskSimulator生成真实VLM对话日志
===========================================

验证任务A的修改:
1. BatchTaskSimulator调用StrategicSimulator
2. 生成真实的对话内容
3. 保存标准格式的run_log
"""

import json
from pathlib import Path
from src.simulator import BatchTaskSimulator, BatchConfig, LLMClient, Evaluator

def load_sample_tasks(num_tasks=3):
    """加载示例任务"""
    # 从generated_tasks_v2/run_18/tasks加载任务
    task_dir = Path("generated_tasks_v2/run_18/tasks")

    if not task_dir.exists():
        print(f"警告: 任务目录不存在: {task_dir}")
        return create_mock_tasks(num_tasks)

    tasks = []
    for task_file in task_dir.glob("*.jsonl"):
        with open(task_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    task = json.loads(line)
                    tasks.append(task)
                    if len(tasks) >= num_tasks:
                        break
        if len(tasks) >= num_tasks:
            break

    if not tasks:
        print("警告: 未找到任务文件，使用模拟任务")
        return create_mock_tasks(num_tasks)

    return tasks[:num_tasks]

def create_mock_tasks(num_tasks=3):
    """创建模拟任务用于测试"""
    tasks = []
    for i in range(num_tasks):
        tasks.append({
            "task_id": f"test_task_{i}",
            "task_type": "attribute_comparison",
            "question": f"Test question {i}: Which image has the red car?",
            "answer": f"Image {i % 3 + 1}",
            "images": [
                f"test_image_{i}_1.jpg",
                f"test_image_{i}_2.jpg",
                f"test_image_{i}_3.jpg"
            ]
        })
    return tasks

def test_batch_with_real_logs():
    """测试批处理并验证日志生成"""
    print("="*60)
    print("测试BatchTaskSimulator生成真实VLM对话日志")
    print("="*60)

    # 加载任务
    print("\n1. 加载测试任务...")
    tasks = load_sample_tasks(num_tasks=3)
    print(f"   加载了 {len(tasks)} 个任务")
    for i, task in enumerate(tasks):
        print(f"   - 任务 {i+1}: {task.get('task_id', 'unknown')} ({task.get('task_type', 'unknown')})")

    # 配置
    print("\n2. 配置BatchTaskSimulator...")
    config = BatchConfig(
        max_turns_per_session=50,
        min_turns_per_task=3,  # 减少轮数以加快测试
        max_turns_per_task=8,
        transition_style="natural",
        enable_cross_task_memory_test=False  # 暂时禁用以简化测试
    )
    print(f"   max_turns_per_session: {config.max_turns_per_session}")
    print(f"   min_turns_per_task: {config.min_turns_per_task}")
    print(f"   max_turns_per_task: {config.max_turns_per_task}")

    # 创建模拟器
    print("\n3. 创建模拟器...")
    # 注意: 这里使用默认的LLMClient，实际环境需要配置API
    llm_client = LLMClient()  # Mock client
    evaluator = Evaluator()

    batch_sim = BatchTaskSimulator(
        llm_client=llm_client,
        evaluator=evaluator,
        config=config,
        verbose=True
    )
    print("   模拟器创建成功")

    # 运行批处理
    print("\n4. 运行批处理...")
    print("-"*60)

    try:
        result = batch_sim.run_batch(tasks)

        print("\n" + "="*60)
        print("5. 批处理完成!")
        print("="*60)
        print(f"\n统计信息:")
        print(f"  - 尝试任务数: {result.tasks_attempted}")
        print(f"  - 完成任务数: {result.tasks_completed}")
        print(f"  - 总轮数: {result.total_turns}")
        print(f"  - 过渡次数: {len(result.transitions)}")

        # 检查日志目录
        print("\n6. 检查生成的日志...")
        log_dir = batch_sim.get_batch_log_dir()

        if log_dir:
            log_path = Path(log_dir)
            print(f"   日志目录: {log_path}")

            if log_path.exists():
                log_files = list(log_path.glob("run_log_*.json"))
                print(f"   生成的日志文件数: {len(log_files)}")

                # 检查第一个日志文件
                if log_files:
                    first_log = log_files[0]
                    print(f"\n   检查日志文件: {first_log.name}")

                    with open(first_log, 'r', encoding='utf-8') as f:
                        log_data = json.load(f)

                    print(f"   - 事件数: {len(log_data)}")

                    # 统计事件类型
                    event_types = {}
                    for event in log_data:
                        event_type = event.get('event', 'unknown')
                        event_types[event_type] = event_types.get(event_type, 0) + 1

                    print(f"   - 事件类型分布: {event_types}")

                    # 检查是否有对话内容
                    has_queries = any(e.get('event') == 'core_model_decision' for e in log_data)
                    has_responses = any(e.get('event') == 'target_model_response' for e in log_data)

                    print(f"\n   验证结果:")
                    print(f"   - 包含用户query: {'✓' if has_queries else '✗'}")
                    print(f"   - 包含VLM响应: {'✓' if has_responses else '✗'}")

                    if has_queries and has_responses:
                        print(f"\n   ✓ 日志格式正确，包含真实对话内容!")
                    else:
                        print(f"\n   ✗ 日志缺少对话内容，可能仍在使用mock模式")

                    # 显示前几个事件
                    print(f"\n   前3个事件预览:")
                    for i, event in enumerate(log_data[:3]):
                        print(f"   [{i+1}] {event.get('event', 'unknown')}")
                        if event.get('event') == 'core_model_decision':
                            msg = event.get('message_to_model', '')[:50]
                            print(f"       Query: {msg}...")
                        elif event.get('event') == 'target_model_response':
                            resp = event.get('target_response', '')[:50]
                            print(f"       Response: {resp}...")

                else:
                    print("   ✗ 未生成日志文件")
            else:
                print(f"   ✗ 日志目录不存在")
        else:
            print("   ✗ 未设置日志目录")

        print("\n" + "="*60)
        print("测试完成!")
        print("="*60)

        return result

    except Exception as e:
        print(f"\n✗ 批处理失败: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    test_batch_with_real_logs()
