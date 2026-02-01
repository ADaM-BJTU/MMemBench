"""
Reasoning Chain Builder
=======================

将任务的推理链转换为可执行的多轮对话序列。

功能:
1. 将rationale拆解为多轮验证
2. 生成引导模型逐步推理的问题
3. 跟踪推理链的完成度
4. 支持窗口3的LLMQueryGenerator注入
"""

import random
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class ChainQuery:
    """推理链中的单个查询"""
    turn: int
    step_index: int
    step_type: str  # "initial" | "intermediate" | "final"
    query: str
    expected_info: str  # 期望模型提供的信息
    depends_on: List[str]  # 依赖的前置信息
    validation_hints: List[str]  # 用于验证回答的提示词


@dataclass
class ChainExecutionState:
    """推理链执行状态"""
    total_steps: int
    current_step: int = 0
    steps_passed: List[bool] = field(default_factory=list)
    collected_facts: List[str] = field(default_factory=list)
    chain_broken: bool = False
    break_reason: Optional[str] = None


class ReasoningChainBuilder:
    """
    推理链构建器

    将任务的推理链转换为可执行的对话序列。

    Usage:
        builder = ReasoningChainBuilder(task)
        queries = builder.build_chain_queries()

        for query in queries:
            response = vlm.generate(query.query)
            passed = builder.validate_response(response, query)
    """

    # 初始问题模板
    INITIAL_TEMPLATES = [
        "让我们一步步分析这个问题。首先，{hint}是什么样的？",
        "To answer this question, let's start with: {hint}?",
        "第一步，请观察{hint}。",
        "让我们从{hint}开始分析。",
        "First, let's look at {hint}. What do you see?",
        "观察图片中的{hint}，告诉我你看到了什么。"
    ]

    # 中间推理模板
    INTERMEDIATE_TEMPLATES = [
        "你刚才提到{previous_fact}。基于这个，{next_question}？",
        "根据{previous_fact}，现在{next_question}？",
        "知道了{previous_fact}之后，{next_question}？",
        "Based on {previous_fact}, {next_question}?",
        "Given that {previous_fact}, can you tell me {next_question}?",
        "Considering {previous_fact}, what about {next_question}?"
    ]

    # 最终验证模板
    FINAL_TEMPLATES = [
        "综合以上分析: {chain_summary}。所以，{original_question}",
        "根据推理链 {chain_summary}，最终结论是？",
        "Putting it together: {chain_summary}. Therefore, {original_question}",
        "我们分析了{chain_summary}。最终，{original_question}",
        "Based on our analysis ({chain_summary}), what is the answer to: {original_question}",
        "总结一下：{chain_summary}。那么{original_question}"
    ]

    def __init__(
        self,
        task: Dict[str, Any],
        query_generator: Optional[Any] = None  # ← 窗口3注入点
    ):
        """
        初始化推理链构建器

        Args:
            task: 任务数据，必须包含 'reasoning_chain' 字段
            query_generator: 可选的LLM查询生成器（窗口3集成）
        """
        self.task = task
        self.reasoning_chain = task.get('reasoning_chain', {})
        self.steps = self.reasoning_chain.get('steps', [])
        self.original_question = task.get('question', '')
        self.expected_answer = task.get('answer', '')

        # 窗口3集成: LLM驱动的查询生成
        self.query_generator = query_generator

        # 执行状态
        self.state = ChainExecutionState(total_steps=len(self.steps))

    def build_chain_queries(self) -> List[ChainQuery]:
        """
        构建推理链查询序列

        Returns:
            ChainQuery列表，每个对应一轮对话
        """
        if not self.steps:
            # 如果没有明确的步骤，使用简单的单步验证
            return [self._build_simple_query()]

        queries = []

        for i, step in enumerate(self.steps):
            step_type = self._determine_step_type(i)

            if step_type == "initial":
                query = self._build_initial_query(step)
            elif step_type == "final":
                query = self._build_final_query(step, queries)
            else:
                query = self._build_intermediate_query(step, queries)

            queries.append(query)

        return queries

    def _determine_step_type(self, index: int) -> str:
        """确定步骤类型"""
        if index == 0:
            return "initial"
        elif index == len(self.steps) - 1:
            return "final"
        else:
            return "intermediate"

    def _build_initial_query(self, step: Dict) -> ChainQuery:
        """构建初始查询"""
        step_content = step.get('content', '')
        hint = self._extract_hint(step_content)

        # 如果有LLM查询生成器，使用它
        if self.query_generator:
            query_text = self._generate_with_llm(
                step_type="initial",
                step_content=step_content,
                previous_queries=[]
            )
        else:
            template = random.choice(self.INITIAL_TEMPLATES)
            query_text = template.format(hint=hint)

        return ChainQuery(
            turn=1,
            step_index=0,
            step_type="initial",
            query=query_text,
            expected_info=step_content,
            depends_on=[],
            validation_hints=step.get('expected_hints', [])
        )

    def _build_intermediate_query(
        self,
        step: Dict,
        previous_queries: List[ChainQuery]
    ) -> ChainQuery:
        """构建中间推理查询"""
        step_content = step.get('content', '')
        step_index = step.get('step_index', len(previous_queries))

        # 获取前一步的信息
        if previous_queries:
            previous_fact = self._simplify(previous_queries[-1].expected_info)
        else:
            previous_fact = "之前的观察"

        next_question = self._to_question(step_content)

        # 如果有LLM查询生成器，使用它
        if self.query_generator:
            query_text = self._generate_with_llm(
                step_type="intermediate",
                step_content=step_content,
                previous_queries=previous_queries
            )
        else:
            template = random.choice(self.INTERMEDIATE_TEMPLATES)
            query_text = template.format(
                previous_fact=previous_fact,
                next_question=next_question
            )

        return ChainQuery(
            turn=len(previous_queries) + 1,
            step_index=step_index,
            step_type="intermediate",
            query=query_text,
            expected_info=step_content,
            depends_on=[pq.expected_info for pq in previous_queries[-2:]],
            validation_hints=step.get('expected_hints', [])
        )

    def _build_final_query(
        self,
        step: Dict,
        previous_queries: List[ChainQuery]
    ) -> ChainQuery:
        """构建最终验证查询"""
        # 构建推理链摘要
        chain_summary = " → ".join(
            self._simplify(q.expected_info)
            for q in previous_queries[-3:]  # 最多取最近3步
        )

        # 如果有LLM查询生成器，使用它
        if self.query_generator:
            query_text = self._generate_with_llm(
                step_type="final",
                step_content=step.get('content', ''),
                previous_queries=previous_queries
            )
        else:
            template = random.choice(self.FINAL_TEMPLATES)
            query_text = template.format(
                chain_summary=chain_summary,
                original_question=self.original_question
            )

        return ChainQuery(
            turn=len(previous_queries) + 1,
            step_index=len(self.steps) - 1,
            step_type="final",
            query=query_text,
            expected_info=self.expected_answer,
            depends_on=[q.expected_info for q in previous_queries],
            validation_hints=step.get('expected_hints', []) + [self.expected_answer]
        )

    def _build_simple_query(self) -> ChainQuery:
        """构建简单的单步查询"""
        return ChainQuery(
            turn=1,
            step_index=0,
            step_type="final",
            query=self.original_question,
            expected_info=self.expected_answer,
            depends_on=[],
            validation_hints=[self.expected_answer]
        )

    def _generate_with_llm(
        self,
        step_type: str,
        step_content: str,
        previous_queries: List[ChainQuery]
    ) -> str:
        """使用LLM生成更自然的查询（窗口3集成）"""
        if not self.query_generator:
            # 降级到模板生成
            return self._fallback_generate(step_type, step_content, previous_queries)

        try:
            # 调用窗口3的LLMQueryGenerator
            context = {
                'step_type': step_type,
                'step_content': step_content,
                'previous_queries': [q.query for q in previous_queries[-3:]],
                'original_question': self.original_question,
                'expected_answer': self.expected_answer
            }
            return self.query_generator.generate_reasoning_query(context)
        except Exception as e:
            logger.debug(f"LLM query generation failed: {e}, falling back to template")
            return self._fallback_generate(step_type, step_content, previous_queries)

    def _fallback_generate(
        self,
        step_type: str,
        step_content: str,
        previous_queries: List[ChainQuery]
    ) -> str:
        """降级模板生成"""
        hint = self._extract_hint(step_content)

        if step_type == "initial":
            template = random.choice(self.INITIAL_TEMPLATES)
            return template.format(hint=hint)
        elif step_type == "final":
            chain_summary = " → ".join(
                self._simplify(q.expected_info) for q in previous_queries[-3:]
            )
            template = random.choice(self.FINAL_TEMPLATES)
            return template.format(
                chain_summary=chain_summary,
                original_question=self.original_question
            )
        else:
            previous_fact = self._simplify(previous_queries[-1].expected_info) if previous_queries else ""
            template = random.choice(self.INTERMEDIATE_TEMPLATES)
            return template.format(
                previous_fact=previous_fact,
                next_question=self._to_question(step_content)
            )

    def validate_response(
        self,
        response: str,
        query: ChainQuery
    ) -> Tuple[bool, float, str]:
        """
        验证VLM的回答

        Args:
            response: VLM的回答
            query: 对应的查询

        Returns:
            Tuple[passed, score, reasoning]
        """
        response_lower = response.lower()

        # 检查关键词匹配
        hints_found = 0
        total_hints = len(query.validation_hints)

        for hint in query.validation_hints:
            if hint.lower() in response_lower:
                hints_found += 1

        # 计算分数
        if total_hints > 0:
            score = hints_found / total_hints
        else:
            # 无明确提示时，使用语义相似度或假设通过
            score = 0.6

        # 判断是否通过
        passed = score >= 0.5

        # 构建评估理由
        if passed:
            reasoning = f"Found {hints_found}/{total_hints} expected elements"
        else:
            missing = [h for h in query.validation_hints if h.lower() not in response_lower]
            reasoning = f"Missing elements: {missing[:3]}"

        # 更新状态
        self.state.current_step = query.step_index + 1
        self.state.steps_passed.append(passed)

        if passed:
            self.state.collected_facts.append(response[:200])
        else:
            self.state.chain_broken = True
            self.state.break_reason = reasoning

        return passed, score, reasoning

    def get_chain_progress(self) -> Dict[str, Any]:
        """获取推理链进度"""
        return {
            'total_steps': self.state.total_steps,
            'current_step': self.state.current_step,
            'steps_passed': self.state.steps_passed,
            'pass_rate': sum(self.state.steps_passed) / len(self.state.steps_passed) if self.state.steps_passed else 0,
            'chain_broken': self.state.chain_broken,
            'break_reason': self.state.break_reason,
            'collected_facts_count': len(self.state.collected_facts)
        }

    def is_chain_complete(self) -> bool:
        """检查推理链是否完成"""
        return (
            self.state.current_step >= self.state.total_steps and
            not self.state.chain_broken
        )

    def reset_state(self):
        """重置执行状态"""
        self.state = ChainExecutionState(total_steps=len(self.steps))

    # ========== 辅助方法 ==========

    def _extract_hint(self, text: str) -> str:
        """从文本中提取提示"""
        import re

        # 优先提取物体引用
        obj_refs = re.findall(r'\[([^\]]+)\]', text)
        if obj_refs:
            return obj_refs[0]

        # 取前30个字符或第一个分句
        if '.' in text:
            return text.split('.')[0][:30]
        return text[:30]

    def _simplify(self, text: str) -> str:
        """简化文本"""
        if '.' in text:
            return text.split('.')[0][:40]
        return text[:40]

    def _to_question(self, statement: str) -> str:
        """将陈述转换为问题"""
        statement = statement.strip().rstrip('.')

        # 简单转换
        if statement.startswith(('The ', 'A ', 'An ')):
            return f"what about {statement.lower()}"
        elif statement.startswith('他') or statement.startswith('她'):
            return f"{statement}是怎样的"
        else:
            return f"{statement}?"


class ChainExecutionManager:
    """
    推理链执行管理器

    负责协调ReasoningChainBuilder和模拟器/评估器的交互。
    """

    def __init__(
        self,
        task: Dict[str, Any],
        query_generator: Optional[Any] = None
    ):
        self.task = task
        self.builder = ReasoningChainBuilder(task, query_generator)
        self.queries = self.builder.build_chain_queries()
        self.responses: List[Dict[str, Any]] = []

    def get_next_query(self) -> Optional[ChainQuery]:
        """获取下一个查询"""
        current_step = self.builder.state.current_step
        if current_step < len(self.queries):
            return self.queries[current_step]
        return None

    def process_response(
        self,
        response: str,
        query: ChainQuery
    ) -> Dict[str, Any]:
        """处理VLM响应"""
        passed, score, reasoning = self.builder.validate_response(response, query)

        result = {
            'step': query.step_index,
            'step_type': query.step_type,
            'query': query.query,
            'response': response,
            'passed': passed,
            'score': score,
            'reasoning': reasoning,
            'chain_progress': self.builder.get_chain_progress()
        }

        self.responses.append(result)
        return result

    def is_complete(self) -> bool:
        """检查是否完成"""
        return (
            self.builder.state.current_step >= len(self.queries) or
            self.builder.state.chain_broken
        )

    def get_final_report(self) -> Dict[str, Any]:
        """获取最终报告"""
        progress = self.builder.get_chain_progress()

        return {
            'task_id': self.task.get('task_id', ''),
            'total_steps': len(self.queries),
            'steps_completed': self.builder.state.current_step,
            'chain_complete': self.builder.is_chain_complete(),
            'chain_broken': progress['chain_broken'],
            'break_reason': progress['break_reason'],
            'pass_rate': progress['pass_rate'],
            'step_results': self.responses
        }
