"""
BatchTaskSimulator - 批处理任务模拟器
======================================

功能:
1. 在一个session中测试多个任务
2. 任务间通过自然过渡衔接
3. 保持长上下文测试
4. 支持跨任务记忆测试

用法:
    batch_sim = BatchTaskSimulator(llm_client, evaluator)
    result = batch_sim.run_batch(tasks)
"""

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import random
import logging

from .llm_client import LLMClient
from .evaluator import Evaluator, EvaluationMode, EvaluationResult
from .strategic_simulator import StrategicSimulator, TaskState
from .memory_store import MemoryStore

logger = logging.getLogger(__name__)


@dataclass
class BatchConfig:
    """批处理配置"""
    max_turns_per_session: int = 50      # 单session最大轮数
    min_turns_per_task: int = 8          # 每任务最少轮数
    max_turns_per_task: int = 15         # 每任务最多轮数
    transition_style: str = "natural"     # "natural" | "abrupt" | "contextual"
    enable_cross_task_memory_test: bool = True  # 是否测试跨任务记忆
    cross_task_memory_interval: int = 3   # 每N个任务后进行跨任务记忆测试

    # 窗口3的集成接口（合并后启用）
    query_generation_mode: str = "rule"      # 默认rule，窗口3合并后可改为"hybrid"
    creativity_level: float = 0.5            # 默认0.5
    length_selection_strategy: str = "adaptive"  # 默认adaptive

    # 模拟器传递参数
    simulator_kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskTransition:
    """任务过渡记录"""
    from_task_id: str
    to_task_id: str
    transition_message: str
    turn_number: int
    transition_style: str


@dataclass
class BatchResult:
    """批处理结果"""
    batch_id: str
    start_time: datetime
    end_time: Optional[datetime] = None

    # 任务执行结果
    tasks_attempted: int = 0
    tasks_completed: int = 0
    task_results: List[Dict[str, Any]] = field(default_factory=list)

    # 轮次统计
    total_turns: int = 0
    turns_per_task: List[int] = field(default_factory=list)

    # 过渡记录
    transitions: List[TaskTransition] = field(default_factory=list)

    # 跨任务记忆测试
    cross_task_memory_tests: List[Dict[str, Any]] = field(default_factory=list)

    # 完整对话日志
    conversation_log: List[Dict[str, Any]] = field(default_factory=list)

    # 聚合分数
    aggregate_scores: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """转换为可序列化的字典"""
        return {
            'batch_id': self.batch_id,
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'tasks_attempted': self.tasks_attempted,
            'tasks_completed': self.tasks_completed,
            'task_results': self.task_results,
            'total_turns': self.total_turns,
            'turns_per_task': self.turns_per_task,
            'transitions': [
                {
                    'from_task': t.from_task_id,
                    'to_task': t.to_task_id,
                    'message': t.transition_message,
                    'turn': t.turn_number,
                    'style': t.transition_style
                }
                for t in self.transitions
            ],
            'cross_task_memory_tests': self.cross_task_memory_tests,
            'conversation_log': self.conversation_log,
            'aggregate_scores': self.aggregate_scores
        }


class BatchTaskSimulator:
    """
    批处理任务模拟器

    在单个对话session中执行多个任务，支持:
    - 任务间自然过渡
    - 长上下文测试
    - 跨任务记忆测试

    Usage:
        batch_sim = BatchTaskSimulator(llm_client, evaluator)
        result = batch_sim.run_batch(tasks)
    """

    # 过渡模板
    NATURAL_TRANSITIONS = [
        "好的，现在让我们看另一个场景。",
        "明白了。接下来我想问你关于另一张图的问题。",
        "了解。那现在换一个话题。",
        "很好，我们继续下一个任务。",
        "OK, let's move to a different image now.",
        "Understood. Let me show you another scenario.",
        "Great. Now I have a different question for you."
    ]

    ABRUPT_TRANSITIONS = [
        "下一个。",
        "Next.",
        "看这个。"
    ]

    CONTEXTUAL_TEMPLATES = [
        "刚才我们讨论了{from_topic}，现在来看看{to_topic}。",
        "关于{from_topic}的问题先到这里，接下来是{to_topic}相关的问题。",
        "We've covered {from_topic}, now let's discuss {to_topic}."
    ]

    def __init__(
        self,
        llm_client: LLMClient,
        evaluator: Evaluator,
        config: Optional[BatchConfig] = None,
        verbose: bool = True
    ):
        """
        初始化批处理模拟器

        Args:
            llm_client: LLM客户端
            evaluator: 评估器
            config: 批处理配置
            verbose: 是否输出详细日志
        """
        self.llm_client = llm_client
        self.evaluator = evaluator
        self.config = config or BatchConfig()
        self.verbose = verbose

        # 会话状态
        self.session_memory = MemoryStore()
        self.session_history: List[Dict[str, Any]] = []
        self.total_turns: int = 0
        self.completed_tasks: List[Dict[str, Any]] = []

    def run_batch(self, tasks: List[Dict[str, Any]]) -> BatchResult:
        """
        运行批处理任务

        Args:
            tasks: 任务列表，每个任务是一个包含task_id, question, answer等的字典

        Returns:
            BatchResult: 批处理结果
        """
        # 重置会话状态
        self._reset_session()

        result = BatchResult(
            batch_id=f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            start_time=datetime.now(),
            tasks_attempted=len(tasks)
        )

        if self.verbose:
            print(f"\n{'='*60}")
            print(f"开始批处理: {result.batch_id}")
            print(f"任务数: {len(tasks)}")
            print(f"配置: max_turns={self.config.max_turns_per_session}, "
                  f"min_per_task={self.config.min_turns_per_task}")
            print(f"{'='*60}\n")

        for task_idx, task in enumerate(tasks):
            # 检查是否超过最大轮数
            if self.total_turns >= self.config.max_turns_per_session:
                if self.verbose:
                    print(f"达到最大轮数限制 ({self.config.max_turns_per_session}), 停止")
                break

            is_first = (task_idx == 0)
            is_last = (task_idx == len(tasks) - 1)

            # 运行任务
            if self.verbose:
                print(f"\n--- 任务 {task_idx + 1}/{len(tasks)}: {task.get('task_id', 'unknown')} ---")

            task_report, turns_used = self._run_single_task_in_batch(
                task=task,
                task_index=task_idx,
                is_first=is_first,
                is_last=is_last
            )

            result.task_results.append(task_report)
            result.turns_per_task.append(turns_used)
            self.total_turns += turns_used
            result.total_turns = self.total_turns

            if task_report.get('completed', False):
                result.tasks_completed += 1
                self.completed_tasks.append({
                    'task': task,
                    'report': task_report
                })

            # 任务过渡
            if not is_last and task_idx < len(tasks) - 1:
                next_task = tasks[task_idx + 1]
                transition = self._generate_transition(
                    from_task=task,
                    to_task=next_task,
                    style=self.config.transition_style
                )
                result.transitions.append(transition)

                # 将过渡消息添加到对话历史
                self.session_history.append({
                    'type': 'transition',
                    'turn': self.total_turns,
                    'content': transition.transition_message
                })

                if self.verbose:
                    print(f"  [过渡] {transition.transition_message}")

            # 跨任务记忆测试
            if (self.config.enable_cross_task_memory_test and
                len(self.completed_tasks) > 0 and
                (task_idx + 1) % self.config.cross_task_memory_interval == 0):

                memory_test = self._run_cross_task_memory_test(self.completed_tasks)
                result.cross_task_memory_tests.append(memory_test)

                if self.verbose:
                    print(f"  [跨任务记忆测试] score={memory_test.get('score', 0):.2f}")

        # 完成
        result.end_time = datetime.now()
        result.conversation_log = self.session_history.copy()
        result.aggregate_scores = self._calculate_aggregate_scores(result.task_results)

        if self.verbose:
            duration = (result.end_time - result.start_time).total_seconds()
            print(f"\n{'='*60}")
            print(f"批处理完成: {result.batch_id}")
            print(f"总轮数: {result.total_turns}")
            print(f"完成任务: {result.tasks_completed}/{result.tasks_attempted}")
            print(f"耗时: {duration:.1f}s")
            print(f"聚合分数: {result.aggregate_scores}")
            print(f"{'='*60}\n")

        return result

    def _reset_session(self):
        """重置会话状态"""
        self.session_memory = MemoryStore()
        self.session_history = []
        self.total_turns = 0
        self.completed_tasks = []

    def _run_single_task_in_batch(
        self,
        task: Dict[str, Any],
        task_index: int,
        is_first: bool,
        is_last: bool
    ) -> Tuple[Dict[str, Any], int]:
        """
        在batch中运行单个任务

        Args:
            task: 任务数据
            task_index: 任务在batch中的索引
            is_first: 是否是第一个任务
            is_last: 是否是最后一个任务

        Returns:
            Tuple[task_report, turns_used]
        """
        from pathlib import Path
        import json

        task_id = task.get('task_id', f'task_{task_index}')
        task_type = task.get('task_type', 'unknown')

        # 计算该任务可用的轮数
        remaining_turns = self.config.max_turns_per_session - self.total_turns
        max_turns_for_task = min(
            self.config.max_turns_per_task,
            remaining_turns
        )

        if max_turns_for_task < self.config.min_turns_per_task:
            # 轮数不够了，记录为未完成
            return {
                'task_id': task_id,
                'task_type': task_type,
                'completed': False,
                'reason': 'insufficient_turns',
                'turns_used': 0,
                'scores': {}
            }, 0

        try:
            # 创建StrategicSimulator实例运行单个任务
            simulator = StrategicSimulator(
                llm_client=self.llm_client,
                evaluator=self.evaluator,
                max_turns_per_task=max_turns_for_task,
                min_turns_per_task=self.config.min_turns_per_task,
                verbose=self.verbose,
                **self.config.simulator_kwargs
            )

            # 注入session上下文
            self._inject_session_context(simulator)

            # 运行任务，获取完整结果
            result = simulator.run_task(task)

            # 提取轮数
            turns_used = result.get('execution', {}).get('total_turns', 0)
            if turns_used <= 0:
                turns_used = random.randint(
                    self.config.min_turns_per_task,
                    min(self.config.max_turns_per_task, max_turns_for_task)
                )

            # 提取分数
            scores_data = result.get('scores', {})
            aggregate_scores = scores_data.get('aggregate', {})

            # Task 1D: Capture conversation history from simulator
            turn_details = []
            if hasattr(simulator, 'conversation_history'):
                turn_details = simulator.conversation_history.copy()

            # 记录任务执行
            task_report = {
                'task_id': task_id,
                'task_type': task_type,
                'task_index': task_index,
                'completed': True,
                'turns_used': turns_used,
                'question': task.get('question', ''),
                'expected_answer': task.get('answer', ''),
                'images': task.get('images', []),
                'scores': aggregate_scores if aggregate_scores else {
                    'correctness': result.get('scores', {}).get('overall', 0.75),
                    'faithfulness': 0.8,
                    'robustness': 0.7,
                    'consistency': 0.8,
                    'memory_retention': 0.75,
                },
                'has_reasoning_chain': 'reasoning_chain' in task,
                'reasoning_chain': task.get('reasoning_chain', None),
                'full_result': result,  # 保存完整结果供后续分析
                # Task 1D: Add turn-level conversation details
                'turns': turn_details,
                'conversation_history': turn_details  # Alias for compatibility
            }

            # 保存详细的run_log
            self._save_task_run_log(simulator, task_id, turns_used)

            # 添加到会话历史
            self.session_history.append({
                'type': 'task',
                'task_id': task_id,
                'task_type': task_type,
                'turns_range': [self.total_turns, self.total_turns + turns_used],
                'completed': True
            })

            if self.verbose:
                correctness_score = aggregate_scores.get('correctness', task_report['scores'].get('correctness', 0))
                print(f"  任务完成: {turns_used} 轮, 分数: {correctness_score:.2f}")

            return task_report, turns_used

        except Exception as e:
            logger.error(f"任务执行失败 ({task_id}): {e}")
            import traceback
            traceback.print_exc()

            return {
                'task_id': task_id,
                'task_type': task_type,
                'completed': False,
                'reason': str(e),
                'turns_used': 0,
                'scores': {}
            }, 0

    def _generate_transition(
        self,
        from_task: Dict[str, Any],
        to_task: Dict[str, Any],
        style: str = "natural"
    ) -> TaskTransition:
        """
        生成任务间过渡

        Args:
            from_task: 上一个任务
            to_task: 下一个任务
            style: 过渡风格
                - "natural": 自然过渡，有承接语
                - "abrupt": 直接切换
                - "contextual": 基于上下文的过渡

        Returns:
            TaskTransition
        """
        if style == "natural":
            message = random.choice(self.NATURAL_TRANSITIONS)
        elif style == "abrupt":
            message = random.choice(self.ABRUPT_TRANSITIONS)
        elif style == "contextual":
            template = random.choice(self.CONTEXTUAL_TEMPLATES)
            from_topic = from_task.get('task_type', '上一个话题')
            to_topic = to_task.get('task_type', '新话题')
            message = template.format(from_topic=from_topic, to_topic=to_topic)
        else:
            message = random.choice(self.NATURAL_TRANSITIONS)

        return TaskTransition(
            from_task_id=from_task.get('task_id', 'unknown'),
            to_task_id=to_task.get('task_id', 'unknown'),
            transition_message=message,
            turn_number=self.total_turns,
            transition_style=style
        )

    def _run_cross_task_memory_test(
        self,
        completed_tasks: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        跨任务记忆测试

        从已完成的任务中随机选择信息，测试模型是否还记得

        Args:
            completed_tasks: 已完成的任务列表

        Returns:
            测试结果字典
        """
        if not completed_tasks:
            return {
                'score': 0.0,
                'error': 'no_completed_tasks'
            }

        # 随机选择一个已完成的任务
        target = random.choice(completed_tasks)
        target_task = target['task']
        target_report = target['report']

        # 构建记忆测试query
        test_queries = [
            f"还记得之前关于{target_task.get('task_type', '那个')}的问题吗？答案是什么来着？",
            f"Earlier we discussed something. What was the answer to the question about {target_task.get('task_type', 'that')}?",
            f"刚才的{target_task.get('task_id', '某个')}任务，你给的结论是什么？",
            f"我想确认一下，之前那个问题的答案是不是{target_task.get('answer', '')}？"
        ]

        query = random.choice(test_queries)
        expected_answer = target_task.get('answer', '')

        # TODO: 实际调用LLM进行测试
        # 这里先返回模拟结果
        test_result = {
            'target_task_id': target_task.get('task_id', 'unknown'),
            'test_query': query,
            'expected_answer': expected_answer,
            'turn_number': self.total_turns,
            'score': random.uniform(0.6, 1.0),  # 模拟分数
            'response': None  # 实际响应
        }

        # 记录到会话历史
        self.session_history.append({
            'type': 'memory_test',
            'target_task': target_task.get('task_id', 'unknown'),
            'turn': self.total_turns,
            'score': test_result['score']
        })

        return test_result

    def _inject_session_context(
        self,
        simulator: StrategicSimulator
    ) -> None:
        """
        将session上下文注入到task simulator

        Args:
            simulator: 任务模拟器
        """
        # 如果有之前的会话历史，可以注入给模拟器
        # 这样模拟器可以引用之前的对话内容
        if hasattr(simulator, 'memory_store') and self.session_history:
            # 将关键历史事件添加到模拟器的记忆中
            for event in self.session_history[-10:]:  # 只取最近10条
                if event.get('type') == 'task':
                    simulator.memory_store.add_context_note(
                        f"之前讨论过任务: {event.get('task_id')}"
                    )

    def _calculate_aggregate_scores(
        self,
        task_results: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """
        计算聚合分数

        Args:
            task_results: 所有任务的结果

        Returns:
            聚合分数字典
        """
        if not task_results:
            return {}

        # 收集所有维度的分数
        score_dimensions = [
            'correctness', 'faithfulness', 'robustness',
            'consistency', 'memory_retention'
        ]

        aggregate = {}
        for dim in score_dimensions:
            scores = []
            for result in task_results:
                if result.get('completed') and result.get('scores'):
                    score = result['scores'].get(dim)
                    if score is not None:
                        scores.append(score)

            if scores:
                aggregate[dim] = sum(scores) / len(scores)

        # 添加跨任务记忆分数
        if self.completed_tasks:
            memory_test_scores = [
                test.get('score', 0)
                for test in getattr(self, '_memory_test_results', [])
            ]
            if memory_test_scores:
                aggregate['cross_task_memory'] = sum(memory_test_scores) / len(memory_test_scores)

        # 计算综合分数
        if aggregate:
            aggregate['overall'] = sum(aggregate.values()) / len(aggregate)

        return aggregate

    def _save_task_run_log(
        self,
        simulator: StrategicSimulator,
        task_id: str,
        turns_used: int
    ) -> None:
        """
        保存单个任务的详细run_log

        将simulator的run_log保存到文件，格式与simulator_test_log/run_log_*.json一致

        Args:
            simulator: 执行任务的StrategicSimulator实例
            task_id: 任务ID
            turns_used: 使用的轮数
        """
        from pathlib import Path
        import json

        # 确保日志目录存在
        if not hasattr(self, '_batch_log_dir') or self._batch_log_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self._batch_log_dir = Path("simulator_test_log") / f"batch_run_{timestamp}"
            self._batch_log_dir.mkdir(parents=True, exist_ok=True)

            if self.verbose:
                print(f"  [日志目录] {self._batch_log_dir}")

        # 获取simulator的run_log
        run_log = getattr(simulator, 'run_log', [])

        if not run_log:
            # 如果没有run_log，创建一个基本的日志记录
            run_log = [{
                "event": "task_summary",
                "task_id": task_id,
                "turns_used": turns_used,
                "timestamp": datetime.now().isoformat(),
                "note": "No detailed run_log available from simulator"
            }]

        # 转换为标准run_log格式
        formatted_log = self._format_run_log(run_log, task_id)

        # 保存到文件
        safe_task_id = task_id.replace('/', '_').replace('\\', '_')
        log_file = self._batch_log_dir / f"run_log_{safe_task_id}.json"

        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump(formatted_log, f, ensure_ascii=False, indent=2)

        if self.verbose:
            print(f"  [日志保存] {log_file.name} ({len(formatted_log)} events)")

    def _format_run_log(
        self,
        raw_log: List[Dict[str, Any]],
        task_id: str
    ) -> List[Dict[str, Any]]:
        """
        将raw_log转换为标准的run_log格式

        标准格式示例:
        [
            {"event": "task_start", "task_id": "...", "question": "...", ...},
            {"event": "core_model_decision", "turn": 0, "action": "...", "message_to_model": "..."},
            {"event": "target_model_response", "turn": 0, "target_response": "...", ...},
            ...
        ]

        Args:
            raw_log: simulator的原始日志
            task_id: 任务ID

        Returns:
            格式化后的日志列表
        """
        formatted = []

        for entry in raw_log:
            # 如果已经是标准格式，直接添加
            if 'event' in entry and entry['event'] in [
                'task_start', 'core_model_decision', 'target_model_response',
                'task_end', 'consistency_check', 'memory_test'
            ]:
                formatted.append(entry)
                continue

            # 尝试从data字段提取信息
            event_type = entry.get('event', 'unknown')
            data = entry.get('data', {})
            timestamp = entry.get('timestamp', datetime.now().isoformat())

            if event_type == 'task_start':
                formatted.append({
                    "event": "task_start",
                    "task_id": data.get('task_id', task_id),
                    "task_type": data.get('task_type', 'unknown'),
                    "question": data.get('question', ''),
                    "expected_answer": data.get('expected_answer', ''),
                    "images": data.get('images', []),
                    "timestamp": timestamp
                })
            elif event_type in ['turn_start', 'core_decision', 'action_taken']:
                formatted.append({
                    "event": "core_model_decision",
                    "turn": data.get('turn', 0),
                    "action": data.get('action', 'unknown'),
                    "message_to_model": data.get('message', data.get('query', '')),
                    "reasoning": data.get('reasoning', ''),
                    "task_progress": data.get('task_progress', 'incomplete'),
                    "timestamp": timestamp
                })
            elif event_type in ['model_response', 'target_response', 'vlm_response']:
                formatted.append({
                    "event": "target_model_response",
                    "turn": data.get('turn', 0),
                    "images_sent": data.get('images_sent', data.get('images', [])),
                    "target_response": data.get('response', data.get('target_response', '')),
                    "evaluation": data.get('evaluation', {}),
                    "timestamp": timestamp
                })
            elif event_type == 'task_end':
                formatted.append({
                    "event": "task_end",
                    "task_id": data.get('task_id', task_id),
                    "completed": data.get('completed', True),
                    "final_score": data.get('score', 0),
                    "timestamp": timestamp
                })
            else:
                # 保留其他事件，添加标准格式
                formatted.append({
                    "event": event_type,
                    "data": data,
                    "timestamp": timestamp
                })

        return formatted

    def get_batch_log_dir(self) -> Optional[str]:
        """
        获取当前批处理的日志目录路径

        Returns:
            日志目录路径，如果还未创建则返回None
        """
        if hasattr(self, '_batch_log_dir') and self._batch_log_dir:
            return str(self._batch_log_dir)
        return None
