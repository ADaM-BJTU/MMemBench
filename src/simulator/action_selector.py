"""
动作选择器 - 根据对话状态选择下一个动作

设计理念:
1. 与entity提取解耦
2. 支持多种选择策略（规则、概率、学习等）
3. 可独立测试和优化
"""

import random
from typing import Dict, List, Any, Optional, Literal


class ActionSelector:
    """
    动作选择器

    支持的策略:
    - "rule_based": 基于规则（轮次、任务类型等）
    - "random": 随机选择
    - "weighted": 加权随机
    - "learned": 基于学习的策略（未来扩展）
    """

    # 定义所有可用的动作类型及其能力层级
    ACTION_TYPES = {
        # 推理层 (Reasoning)
        "follow_up": {
            "level": "reasoning",
            "description": "追问，引导多跳推理"
        },
        "logic_skip": {
            "level": "reasoning",
            "description": "要求跳过推理，测试模型是否拒绝"
        },
        "negation": {
            "level": "reasoning",
            "description": "否定，纠正错误推理"
        },

        # 聚合层 (Aggregation)
        "guidance": {
            "level": "aggregation",
            "description": "引导，提示关键证据"
        },

        # 上下文管理层 (Context Management)
        "mislead": {
            "level": "context_management",
            "description": "误导，注入错误信息"
        },
        "update": {
            "level": "context_management",
            "description": "更新，改变对象状态"
        },
        "distraction": {
            "level": "context_management",
            "description": "干扰，注入无关信息"
        },
        "redundancy": {
            "level": "context_management",
            "description": "冗余，重复已有信息"
        },
        "fine_grained": {
            "level": "context_management",
            "description": "细粒度请求，要求精确定位"
        },

        # 记忆层 (Memory)
        "memory_injection": {
            "level": "memory",
            "description": "记忆注入，测试视觉记忆完整性"
        },
        "consistency_check": {
            "level": "memory",
            "description": "一致性检查，验证核心记忆"
        },
        "cross_image_confusion": {
            "level": "memory",
            "description": "跨图混淆，测试多图物体区分能力"
        },

        # 控制层
        "next_task": {
            "level": "control",
            "description": "切换到下一个任务"
        }
    }

    def __init__(
        self,
        strategy: Literal["rule_based", "random", "weighted"] = "rule_based",
        allowed_actions: Optional[List[str]] = None
    ):
        """
        Args:
            strategy: 选择策略
            allowed_actions: 允许的动作列表，None表示允许所有动作
        """
        self.strategy = strategy

        # 设置允许的动作
        if allowed_actions is None:
            self.allowed_actions = list(self.ACTION_TYPES.keys())
        else:
            # 验证动作是否有效
            invalid = set(allowed_actions) - set(self.ACTION_TYPES.keys())
            if invalid:
                raise ValueError(f"Invalid actions: {invalid}")
            self.allowed_actions = allowed_actions

        # 加权策略的权重（可调整）
        self.action_weights = self._init_action_weights()

    def _init_action_weights(self) -> Dict[str, float]:
        """初始化动作权重（用于加权随机策略）"""
        return {
            "follow_up": 0.15,           # 追问
            "logic_skip": 0.08,          # 逻辑跳跃
            "negation": 0.1,             # 否定
            "guidance": 0.15,            # 引导
            "mislead": 0.1,              # 误导
            "update": 0.08,              # 更新
            "distraction": 0.08,         # 干扰
            "redundancy": 0.06,          # 冗余
            "fine_grained": 0.1,         # 细粒度
            "memory_injection": 0.08,    # 记忆注入
            "consistency_check": 0.06,   # 一致性检查
            "cross_image_confusion": 0.06,  # 跨图混淆
        }

    def select(
        self,
        turn: int,
        history: List[Dict[str, Any]],
        entities: Dict[str, Any],
        task_info: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        选择下一个动作

        Args:
            turn: 当前轮次（从0开始）
            history: 对话历史
            entities: 当前提取的entities
            task_info: 任务信息
            context: 额外上下文

        Returns:
            选中的动作类型
        """
        task_info = task_info or {}
        context = context or {}

        if self.strategy == "rule_based":
            return self._select_rule_based(turn, history, entities, task_info)
        elif self.strategy == "random":
            return self._select_random()
        elif self.strategy == "weighted":
            return self._select_weighted(turn, history, entities, task_info)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

    def _select_rule_based(
        self,
        turn: int,
        history: List[Dict[str, Any]],
        entities: Dict[str, Any],
        task_info: Dict[str, Any]
    ) -> str:
        """
        基于规则选择动作

        规则设计（已放宽以增加动作多样性）:
        - 前期(0-2轮): 引入基础和部分高级动作
        - 中期(3-5轮): 引入更多高级动作
        - 后期(6+轮): 使用所有类型的动作
        """
        # 获取允许的动作
        candidates = []

        # 阶段1: 前期 - 建立基础理解，同时引入多样性
        if turn < 3:
            candidates = ["follow_up", "guidance", "fine_grained", "logic_skip"]

        # 阶段2: 中期 - 推理挑战和高级动作
        elif turn < 6:
            candidates = [
                "follow_up", "negation", "mislead", "guidance",
                "memory_injection", "consistency_check", "fine_grained",
                "distraction"
            ]

        # 阶段3: 后期 - 上下文管理和所有高级能力
        else:
            candidates = [
                "update", "distraction", "redundancy", "fine_grained",
                "memory_injection", "cross_image_confusion",
                "consistency_check", "negation"
            ]

        # 过滤出允许的动作
        candidates = [a for a in candidates if a in self.allowed_actions]

        # 如果没有候选，使用所有允许的动作
        if not candidates:
            candidates = self.allowed_actions

        # 避免连续重复同一动作
        if len(history) > 0 and "action" in history[-1]:
            last_action = history[-1]["action"]
            if len(candidates) > 1 and last_action in candidates:
                # 尽量选择不同的动作
                other_candidates = [a for a in candidates if a != last_action]
                if other_candidates:
                    candidates = other_candidates

        return random.choice(candidates)

    def _select_random(self) -> str:
        """随机选择动作"""
        return random.choice(self.allowed_actions)

    def _select_weighted(
        self,
        turn: int,
        history: List[Dict[str, Any]],
        entities: Dict[str, Any],
        task_info: Dict[str, Any]
    ) -> str:
        """
        加权随机选择

        根据轮次动态调整权重
        """
        # 复制基础权重
        weights = self.action_weights.copy()

        # 根据轮次调整权重
        if turn <= 2:
            # 前期提高follow_up和guidance权重
            weights["follow_up"] = weights.get("follow_up", 0.2) * 2
            weights["guidance"] = weights.get("guidance", 0.2) * 2
        elif turn <= 5:
            # 中期提高推理类动作权重
            weights["negation"] = weights.get("negation", 0.1) * 1.5
            weights["mislead"] = weights.get("mislead", 0.1) * 1.5
        else:
            # 后期提高上下文管理类动作权重
            weights["update"] = weights.get("update", 0.1) * 2
            weights["fine_grained"] = weights.get("fine_grained", 0.1) * 2

        # 过滤允许的动作
        allowed_weights = {
            a: weights.get(a, 0.1)
            for a in self.allowed_actions
        }

        # 归一化权重
        total = sum(allowed_weights.values())
        if total == 0:
            return random.choice(self.allowed_actions)

        normalized = {a: w/total for a, w in allowed_weights.items()}

        # 加权随机选择
        actions = list(normalized.keys())
        weights_list = [normalized[a] for a in actions]

        return random.choices(actions, weights=weights_list)[0]

    def get_action_info(self, action_type: str) -> Dict[str, Any]:
        """获取动作的元信息"""
        if action_type not in self.ACTION_TYPES:
            raise ValueError(f"Unknown action type: {action_type}")

        return self.ACTION_TYPES[action_type].copy()

    def get_actions_by_level(self, level: str) -> List[str]:
        """获取指定能力层级的所有动作"""
        return [
            action for action, info in self.ACTION_TYPES.items()
            if info["level"] == level
        ]

    def set_action_weight(self, action_type: str, weight: float):
        """动态调整动作权重（用于加权策略）"""
        if action_type not in self.ACTION_TYPES:
            raise ValueError(f"Unknown action type: {action_type}")
        self.action_weights[action_type] = weight


# 示例用法
if __name__ == "__main__":
    # 测试不同策略
    print("=" * 60)
    print("测试ActionSelector")
    print("=" * 60)

    # 1. 规则策略
    print("\n1. 规则策略 (rule_based):")
    selector = ActionSelector(strategy="rule_based")

    for turn in range(10):
        action = selector.select(
            turn=turn,
            history=[],
            entities={"objects": ["人", "伞"]},
            task_info={}
        )
        print(f"   Turn {turn}: {action} ({selector.get_action_info(action)['level']})")

    # 2. 加权策略
    print("\n2. 加权策略 (weighted):")
    selector = ActionSelector(strategy="weighted")

    action_counts = {}
    for _ in range(100):
        action = selector.select(
            turn=3,
            history=[],
            entities={},
            task_info={}
        )
        action_counts[action] = action_counts.get(action, 0) + 1

    print("   动作分布 (100次采样):")
    for action, count in sorted(action_counts.items(), key=lambda x: -x[1]):
        print(f"   {action}: {count}%")

    # 3. 限制动作类型
    print("\n3. 限制动作类型:")
    selector = ActionSelector(
        strategy="rule_based",
        allowed_actions=["follow_up", "guidance", "update"]
    )

    for turn in range(10):
        action = selector.select(turn=turn, history=[], entities={})
        print(f"   Turn {turn}: {action}")
