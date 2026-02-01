"""
LLM驱动的Query生成器

使用LLM生成更灵活、多样的用户查询，同时保留规则生成器作为fallback。
"""

import random
from typing import Dict, List, Any, Optional, Literal

from .query_generator import QueryGenerator
from .prompt_templates import PromptTemplates
from .action_space import ACTION_DEFINITIONS


class LLMQueryGenerator:
    """
    LLM驱动的Query生成器

    特性:
    1. 使用LLM生成自然、多样的query
    2. 支持创造性级别调节
    3. 保留规则生成器作为fallback
    4. 记录生成来源（LLM/Rule）
    """

    def __init__(
        self,
        llm_client: Any,
        fallback_generator: Optional[QueryGenerator] = None,
        default_creativity: float = 0.5,
        max_retries: int = 2
    ):
        """
        Args:
            llm_client: LLM客户端实例
            fallback_generator: 规则生成器（作为fallback）
            default_creativity: 默认创造性级别 (0.0-1.0)
            max_retries: LLM生成失败时的最大重试次数
        """
        self.llm_client = llm_client
        self.fallback_generator = fallback_generator or QueryGenerator()
        self.default_creativity = default_creativity
        self.max_retries = max_retries
        self.templates = PromptTemplates()

        # 统计信息
        self.generation_stats = {
            "llm_success": 0,
            "llm_failed": 0,
            "fallback_used": 0
        }

    def generate(
        self,
        action_type: str,
        entities: Dict[str, Any],
        length: str = "medium",
        vlm_response: str = "",
        history: Optional[List[Dict]] = None,
        context: Optional[Dict[str, Any]] = None,
        creativity_level: Optional[float] = None,
        task_info: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        生成用户查询

        Args:
            action_type: 动作类型
            entities: 提取的entities
            length: 长度控制
            vlm_response: VLM的原始响应
            history: 对话历史
            context: 额外上下文
            creativity_level: 创造性级别 (0.0-1.0)
            task_info: 任务信息

        Returns:
            {
                "query": str,              # 生成的query
                "source": str,             # "llm" 或 "rule"
                "creativity_used": float,  # 实际使用的创造性级别
                "length_suffix_added": bool,  # 是否添加了长度后缀
                "generation_attempts": int    # 尝试次数
            }
        """
        history = history or []
        context = context or {}
        task_info = task_info or {}
        creativity = creativity_level if creativity_level is not None else self.default_creativity

        result = {
            "query": "",
            "source": "unknown",
            "creativity_used": creativity,
            "length_suffix_added": False,
            "generation_attempts": 0
        }

        # 根据创造性级别决定生成方式
        if creativity < 0.3:
            # 低创造性：直接使用规则
            query = self.fallback_generator.generate(
                action_type=action_type,
                entities=entities,
                length=length,
                vlm_response=vlm_response,
                history=history,
                context=context
            )
            result["query"] = query
            result["source"] = "rule"
            result["generation_attempts"] = 1
            result["length_suffix_added"] = self._check_length_suffix(query, length)
            self.generation_stats["fallback_used"] += 1
        else:
            # 中高创造性：尝试LLM生成
            for attempt in range(self.max_retries):
                result["generation_attempts"] = attempt + 1

                try:
                    query = self._generate_with_llm(
                        action_type=action_type,
                        entities=entities,
                        vlm_response=vlm_response,
                        task_info=task_info,
                        history=history,
                        creativity=creativity
                    )

                    if query and len(query) >= 5:
                        # 添加长度控制后缀
                        query, suffix_added = self._add_length_suffix(query, length)
                        result["query"] = query
                        result["source"] = "llm"
                        result["length_suffix_added"] = suffix_added
                        self.generation_stats["llm_success"] += 1
                        break

                except Exception as e:
                    print(f"LLM generation attempt {attempt + 1} failed: {e}")
                    continue

            # 所有尝试失败，使用fallback
            if not result["query"]:
                query = self.fallback_generator.generate(
                    action_type=action_type,
                    entities=entities,
                    length=length,
                    vlm_response=vlm_response,
                    history=history,
                    context=context
                )
                result["query"] = query
                result["source"] = "rule_fallback"
                result["length_suffix_added"] = self._check_length_suffix(query, length)
                self.generation_stats["llm_failed"] += 1
                self.generation_stats["fallback_used"] += 1

        return result

    def _generate_with_llm(
        self,
        action_type: str,
        entities: Dict[str, Any],
        vlm_response: str,
        task_info: Dict,
        history: List[Dict],
        creativity: float
    ) -> str:
        """使用LLM生成query"""

        # 构建LLM prompt
        llm_prompt = self._build_llm_prompt(
            action_type=action_type,
            entities=entities,
            vlm_response=vlm_response,
            task_info=task_info,
            history=history
        )

        # 调用LLM
        response = self.llm_client.generate(
            prompt=llm_prompt,
            temperature=creativity,
            max_tokens=150
        )

        # 解析输出
        query = self._parse_llm_output(response)

        return query

    def _build_llm_prompt(
        self,
        action_type: str,
        entities: Dict[str, Any],
        vlm_response: str,
        task_info: Dict,
        history: List[Dict]
    ) -> str:
        """构建给LLM的prompt"""

        # 获取动作定义 (ActionDefinition是dataclass)
        action_def = ACTION_DEFINITIONS.get(action_type)

        # 从ActionDefinition对象获取属性
        if action_def:
            action_purpose = action_def.purpose
            action_description = action_def.description
        else:
            action_purpose = "测试模型能力"
            action_description = ""

        # 格式化entities
        objects_str = ", ".join(entities.get("objects", [])[:5]) or "无"
        attributes_str = self._format_attributes(entities.get("attributes", {}))
        regions_str = ", ".join(entities.get("regions", [])[:3]) or "无"

        # 格式化历史
        history_str = self._format_history(history[-3:])

        prompt = f"""你是一个专业的VLM测试员，正在测试视觉语言模型的能力。

**当前动作类型**: {action_type}
**动作目的**: {action_purpose}
**动作描述**: {action_description}

**任务上下文**:
- 原始问题: {task_info.get('question', '未知')}
- 预期答案: {task_info.get('answer', '未知')[:100] if task_info.get('answer') else '未知'}

**VLM最近的回复**:
"{vlm_response[:300]}..."

**从回复中提取的实体**:
- 物体: {objects_str}
- 属性: {attributes_str}
- 区域: {regions_str}

**最近的对话历史**:
{history_str}

---

**你的任务**: 生成一个自然、口语化的用户查询，实现"{action_type}"动作的目的。

要求:
1. 语言自然，像真实用户一样交流
2. 合理使用提取的实体
3. 符合动作的目的
4. 简洁明了（1-2句话）
5. 避免重复之前的问法

直接输出查询内容，不要任何解释或前缀："""

        return prompt

    def _format_attributes(self, attributes: Dict) -> str:
        """格式化属性字典"""
        if not attributes:
            return "无"

        parts = []
        for entity, attrs in list(attributes.items())[:3]:
            if isinstance(attrs, dict):
                attr_strs = [f"{k}={v}" for k, v in list(attrs.items())[:2]]
                parts.append(f"{entity}({', '.join(attr_strs)})")

        return "; ".join(parts) if parts else "无"

    def _format_history(self, history: List[Dict]) -> str:
        """格式化对话历史"""
        if not history:
            return "无历史记录"

        lines = []
        for turn in history:
            query = turn.get("user_query", "")[:50]
            response = turn.get("vlm_response", "")[:50]
            lines.append(f"- User: {query}...")
            lines.append(f"- VLM: {response}...")

        return "\n".join(lines)

    def _parse_llm_output(self, response: str) -> str:
        """解析LLM输出"""
        if not response:
            return ""

        # 清理输出
        query = response.strip()

        # 移除可能的引号
        if query.startswith('"') and query.endswith('"'):
            query = query[1:-1]
        if query.startswith("'") and query.endswith("'"):
            query = query[1:-1]

        # 移除可能的前缀
        prefixes = ["查询:", "Query:", "用户:", "User:"]
        for prefix in prefixes:
            if query.startswith(prefix):
                query = query[len(prefix):].strip()

        return query

    def _add_length_suffix(self, query: str, length: str) -> tuple:
        """添加长度控制后缀"""
        suffix = self.templates.LENGTH_CONTROL_SUFFIX.get(length, "")

        if suffix and suffix not in query:
            return query + " " + suffix, True

        return query, False

    def _check_length_suffix(self, query: str, length: str) -> bool:
        """检查query是否包含长度控制后缀"""
        suffix = self.templates.LENGTH_CONTROL_SUFFIX.get(length, "")
        return suffix in query if suffix else False

    def get_stats(self) -> Dict[str, Any]:
        """获取生成统计信息"""
        total = sum(self.generation_stats.values())
        return {
            **self.generation_stats,
            "total_generations": total,
            "llm_success_rate": (
                self.generation_stats["llm_success"] / total
                if total > 0 else 0
            )
        }

    def reset_stats(self):
        """重置统计信息"""
        self.generation_stats = {
            "llm_success": 0,
            "llm_failed": 0,
            "fallback_used": 0
        }


# ========== 混合模式生成器 ==========

class HybridQueryGenerator:
    """
    混合模式Query生成器

    根据配置在LLM和规则生成器之间切换，
    支持概率混合模式。
    """

    def __init__(
        self,
        llm_client: Optional[Any] = None,
        mode: Literal["rule", "llm", "hybrid"] = "hybrid",
        llm_probability: float = 0.7,
        min_creativity: float = 0.3,
        max_creativity: float = 0.9
    ):
        """
        Args:
            llm_client: LLM客户端
            mode: 生成模式
            llm_probability: hybrid模式下使用LLM的概率
            min_creativity: 最小创造性
            max_creativity: 最大创造性
        """
        self.mode = mode
        self.llm_probability = llm_probability
        self.min_creativity = min_creativity
        self.max_creativity = max_creativity

        # 初始化生成器
        self.rule_generator = QueryGenerator()
        self.llm_generator = None

        if llm_client and mode in ["llm", "hybrid"]:
            self.llm_generator = LLMQueryGenerator(
                llm_client=llm_client,
                fallback_generator=self.rule_generator
            )

    def generate(
        self,
        action_type: str,
        entities: Dict[str, Any],
        length: str = "medium",
        vlm_response: str = "",
        history: Optional[List[Dict]] = None,
        context: Optional[Dict[str, Any]] = None,
        task_info: Optional[Dict] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        生成query

        Returns:
            包含query和元信息的字典
        """
        history = history or []
        context = context or {}
        task_info = task_info or {}

        # 决定使用哪种生成器
        use_llm = False

        if self.mode == "llm" and self.llm_generator:
            use_llm = True
        elif self.mode == "hybrid" and self.llm_generator:
            use_llm = random.random() < self.llm_probability

        # 生成
        if use_llm:
            # 随机创造性级别
            creativity = random.uniform(self.min_creativity, self.max_creativity)

            result = self.llm_generator.generate(
                action_type=action_type,
                entities=entities,
                length=length,
                vlm_response=vlm_response,
                history=history,
                context=context,
                creativity_level=creativity,
                task_info=task_info
            )
        else:
            # 使用规则生成
            query = self.rule_generator.generate(
                action_type=action_type,
                entities=entities,
                length=length,
                vlm_response=vlm_response,
                history=history,
                context=context
            )

            # 检查长度后缀是否已添加
            templates = PromptTemplates()
            suffix = templates.LENGTH_CONTROL_SUFFIX.get(length, "")
            length_suffix_added = suffix in query if suffix else False

            result = {
                "query": query,
                "source": "rule",
                "creativity_used": 0.0,
                "length_suffix_added": length_suffix_added,
                "generation_attempts": 1
            }

        result["mode"] = self.mode
        return result

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        if self.llm_generator:
            return self.llm_generator.get_stats()
        return {
            "llm_success": 0,
            "llm_failed": 0,
            "fallback_used": 0,
            "total_generations": 0,
            "llm_success_rate": 0
        }

    def reset_stats(self):
        """重置统计信息"""
        if self.llm_generator:
            self.llm_generator.reset_stats()
