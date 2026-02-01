"""
用户模拟器 - 模拟用户与VLM的交互

新架构 (重构版):
- EntityExtractor: 统一提取entities
- ActionSelector: 选择动作
- QueryGenerator: 生成query
- LLMQueryGenerator/HybridQueryGenerator: LLM增强的query生成 (窗口3新增)

优势:
- 解耦合，每个模块可独立优化
- 灵活性高，易于实验不同策略
- 可测试性强
- 支持LLM驱动的query生成 (窗口3)
"""

from typing import Dict, List, Any, Optional, Literal

from .entity_extractor import EntityExtractor
from .action_selector import ActionSelector
from .query_generator import QueryGenerator
from .llm_query_generator import LLMQueryGenerator, HybridQueryGenerator
from .prompt_templates import PromptTemplates


class UserSimulator:
    """
    用户模拟器（重构版）

    核心流程:
    1. VLM响应 -> EntityExtractor -> 提取entities
    2. entities + 历史 -> ActionSelector -> 选择动作
    3. 动作 + entities -> QueryGenerator -> 生成query
    """

    def __init__(
        self,
        task: Dict[str, Any],
        extraction_mode: str = "simple",
        action_strategy: str = "rule_based",
        allowed_actions: Optional[List[str]] = None,
        verbose: bool = False,
        # === 窗口3新增参数 ===
        query_generation_mode: str = "rule",  # "rule", "llm", "hybrid"
        llm_client: Optional[Any] = None,
        creativity_level: float = 0.5,
        llm_probability: float = 0.7
    ):
        """
        Args:
            task: 任务信息（包含question、image_path等）
            extraction_mode: entity提取模式 (simple/ner/llm)
            action_strategy: 动作选择策略 (rule_based/random/weighted)
            allowed_actions: 允许的动作列表
            verbose: 是否打印详细信息
            query_generation_mode: Query生成模式 ("rule"/"llm"/"hybrid") - 窗口3新增
            llm_client: LLM客户端（用于llm/hybrid模式）- 窗口3新增
            creativity_level: LLM创造性级别 (0.0-1.0) - 窗口3新增
            llm_probability: hybrid模式下LLM使用概率 - 窗口3新增
        """
        self.task = task
        self.verbose = verbose
        self.query_generation_mode = query_generation_mode

        # 初始化三个核心组件
        self.entity_extractor = EntityExtractor(extraction_mode=extraction_mode)
        self.action_selector = ActionSelector(
            strategy=action_strategy,
            allowed_actions=allowed_actions
        )

        # === 窗口3新增: 根据模式初始化Query Generator ===
        if query_generation_mode == "rule":
            self.query_generator = QueryGenerator()
            self._use_enhanced_generator = False
        else:
            self.query_generator = HybridQueryGenerator(
                llm_client=llm_client,
                mode=query_generation_mode,
                llm_probability=llm_probability,
                min_creativity=max(0.3, creativity_level - 0.2),
                max_creativity=min(1.0, creativity_level + 0.2)
            )
            self._use_enhanced_generator = True

        # 用于检查长度后缀的模板
        self._templates = PromptTemplates()

        # 状态管理
        self.history = []
        self.current_turn = 0
        self.current_entities = {}

    def generate_initial_query(self) -> str:
        """
        生成初始query

        Returns:
            初始query（直接使用任务的question）
        """
        return self.task.get('question', '请描述这张图。')

    def step(
        self,
        vlm_response: str,
        length: str = "medium",
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        执行一步交互

        Args:
            vlm_response: VLM的响应
            length: 查询长度控制
            context: 额外上下文

        Returns:
            {
                "turn": 当前轮次,
                "action": 选择的动作,
                "entities": 提取的entities,
                "user_query": 生成的query,
                "vlm_response": VLM响应,
                # === 窗口3新增字段 ===
                "query_length_control": 使用的长度控制,
                "query_word_count": query词数,
                "query_generation_source": 生成来源,
                "query_creativity_used": 使用的创造性级别,
                "length_suffix_applied": 是否添加了长度后缀
            }
        """
        context = context or {}

        # Step 1: 提取entities
        if self.verbose:
            print(f"\n[Turn {self.current_turn}] 提取entities...")

        entities = self.entity_extractor.extract(
            vlm_response=vlm_response,
            context={
                "task": self.task,
                "history": self.history,
                **context
            }
        )

        self.current_entities = entities

        if self.verbose:
            print(f"  提取到: objects={entities.get('objects', [])[:3]}, "
                  f"regions={entities.get('regions', [])[:2]}")

        # Step 2: 选择动作
        if self.verbose:
            print(f"  选择动作...")

        action = self.action_selector.select(
            turn=self.current_turn,
            history=self.history,
            entities=entities,
            task_info=self.task,
            context=context
        )

        if self.verbose:
            print(f"  动作: {action}")

        # Step 3: 生成query (窗口3增强版)
        if self.verbose:
            print(f"  生成query...")

        if self._use_enhanced_generator:
            # 使用增强生成器
            generation_result = self.query_generator.generate(
                action_type=action,
                entities=entities,
                length=length,
                vlm_response=vlm_response,
                history=self.history,
                context=context,
                task_info=self.task
            )
            user_query = generation_result["query"]
            query_source = generation_result["source"]
            creativity_used = generation_result["creativity_used"]
            length_suffix_applied = generation_result["length_suffix_added"]
        else:
            # 使用基础生成器
            user_query = self.query_generator.generate(
                action_type=action,
                entities=entities,
                length=length,
                vlm_response=vlm_response,
                history=self.history,
                context=context
            )
            query_source = "rule"
            creativity_used = 0.0
            length_suffix_applied = self._check_length_suffix(user_query, length)

        if self.verbose:
            print(f"  Query ({query_source}, len={length}): {user_query[:80]}...")

        # 记录历史 (窗口3增强版)
        step_info = {
            "turn": self.current_turn,
            "action": action,
            "entities": entities,
            "user_query": user_query,
            "vlm_response": vlm_response,
            # === 窗口3新增字段 ===
            "query_length_control": length,
            "query_word_count": len(user_query.split()),
            "query_generation_source": query_source,
            "query_creativity_used": creativity_used,
            "length_suffix_applied": length_suffix_applied
        }

        self.history.append(step_info)
        self.current_turn += 1

        return step_info

    def _check_length_suffix(self, query: str, length: str) -> bool:
        """检查query是否包含长度控制后缀"""
        suffix = self._templates.LENGTH_CONTROL_SUFFIX.get(length, "")
        return suffix in query if suffix else False

    def get_history(self) -> List[Dict[str, Any]]:
        """获取对话历史"""
        return self.history.copy()

    def get_current_entities(self) -> Dict[str, Any]:
        """获取当前的entities"""
        return self.current_entities.copy()

    def reset(self):
        """重置状态（用于新任务）"""
        self.history = []
        self.current_turn = 0
        self.current_entities = {}

    # ========== 配置接口 ==========

    def set_extraction_mode(self, mode: str):
        """动态切换entity提取模式"""
        self.entity_extractor.extraction_mode = mode

    def set_action_strategy(self, strategy: str):
        """动态切换动作选择策略"""
        self.action_selector.strategy = strategy

    def set_allowed_actions(self, actions: List[str]):
        """动态设置允许的动作"""
        self.action_selector.allowed_actions = actions

    def set_action_weight(self, action: str, weight: float):
        """调整动作权重（用于加权策略）"""
        self.action_selector.set_action_weight(action, weight)

    # ========== 统计接口 ==========

    def get_action_distribution(self) -> Dict[str, int]:
        """获取动作分布统计"""
        distribution = {}
        for step in self.history:
            action = step.get("action", "unknown")
            distribution[action] = distribution.get(action, 0) + 1
        return distribution

    def get_entity_coverage(self) -> Dict[str, int]:
        """获取entity类型覆盖统计"""
        coverage = {}
        for step in self.history:
            entities = step.get("entities", {})
            for entity_type, entity_list in entities.items():
                if isinstance(entity_list, (list, dict)) and len(entity_list) > 0:
                    coverage[entity_type] = coverage.get(entity_type, 0) + 1
        return coverage

    # ========== 窗口3新增: Query生成统计 ==========

    def get_query_generation_stats(self) -> Dict[str, Any]:
        """
        获取Query生成统计

        Returns:
            {
                "total_queries": 总query数,
                "length_distribution": 长度分布统计,
                "source_distribution": 来源分布统计,
                "length_suffix_applied_rate": 长度后缀应用率,
                "llm_generator_stats": LLM生成器统计（如果使用）
            }
        """
        if self._use_enhanced_generator and hasattr(self.query_generator, 'get_stats'):
            llm_stats = self.query_generator.get_stats()
        else:
            llm_stats = {}

        # 从历史中统计
        length_stats = {"short": 0, "medium": 0, "long": 0}
        source_stats = {"rule": 0, "llm": 0, "rule_fallback": 0}
        suffix_applied_count = 0
        word_counts = []

        for step in self.history:
            length = step.get("query_length_control", "medium")
            length_stats[length] = length_stats.get(length, 0) + 1

            source = step.get("query_generation_source", "rule")
            source_stats[source] = source_stats.get(source, 0) + 1

            if step.get("length_suffix_applied", False):
                suffix_applied_count += 1

            word_counts.append(step.get("query_word_count", 0))

        avg_word_count = sum(word_counts) / len(word_counts) if word_counts else 0

        return {
            "total_queries": len(self.history),
            "length_distribution": length_stats,
            "source_distribution": source_stats,
            "length_suffix_applied_rate": (
                suffix_applied_count / len(self.history)
                if self.history else 0
            ),
            "average_word_count": avg_word_count,
            "llm_generator_stats": llm_stats
        }

    def get_length_control_summary(self) -> Dict[str, Any]:
        """
        获取长度控制摘要

        Returns:
            长度控制的详细统计信息
        """
        if not self.history:
            return {"message": "No data available"}

        length_word_counts = {"short": [], "medium": [], "long": []}

        for step in self.history:
            length = step.get("query_length_control", "medium")
            word_count = step.get("query_word_count", 0)
            if length in length_word_counts:
                length_word_counts[length].append(word_count)

        # 计算每个长度的平均词数
        avg_word_counts = {}
        for length, counts in length_word_counts.items():
            if counts:
                avg_word_counts[length] = sum(counts) / len(counts)
            else:
                avg_word_counts[length] = 0

        return {
            "total_turns": len(self.history),
            "avg_word_count_by_length": avg_word_counts,
            "query_generation_mode": self.query_generation_mode,
            "uses_enhanced_generator": self._use_enhanced_generator
        }


# 示例用法
if __name__ == "__main__":
    print("=" * 60)
    print("测试UserSimulator (重构版)")
    print("=" * 60)

    # 创建任务
    task = {
        "task_id": "demo_001",
        "question": "图中发生了什么？",
        "image_path": "demo.jpg"
    }

    # 初始化UserSimulator
    simulator = UserSimulator(
        task=task,
        extraction_mode="simple",
        action_strategy="rule_based",
        verbose=True
    )

    # 初始query
    print("\n" + "=" * 60)
    print("初始Query")
    print("=" * 60)
    initial_query = simulator.generate_initial_query()
    print(f"User: {initial_query}")

    # 模拟VLM响应
    mock_responses = [
        "图中有一个人拿着一把黑色的伞站在街道上。",
        "这把伞是打开的，人穿着红色的衣服。",
        "天空看起来是阴沉的，可能在下雨。地面是湿的。",
        "周围还有几栋建筑物和一些树木。",
        "人站在道路的左边，背景中有一辆车经过。"
    ]

    # 进行5轮交互
    for i, vlm_response in enumerate(mock_responses):
        print("\n" + "=" * 60)
        print(f"轮次 {i+1}")
        print("=" * 60)
        print(f"VLM: {vlm_response}")

        step_info = simulator.step(
            vlm_response=vlm_response,
            length="medium"
        )

    # 统计信息
    print("\n" + "=" * 60)
    print("统计信息")
    print("=" * 60)

    print(f"\n动作分布:")
    for action, count in simulator.get_action_distribution().items():
        print(f"  {action}: {count}")

    print(f"\nEntity覆盖:")
    for entity_type, count in simulator.get_entity_coverage().items():
        print(f"  {entity_type}: {count}")

    print("\n✓ 测试完成!")
