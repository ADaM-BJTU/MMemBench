"""
动态长度选择器

根据对话状态、任务阶段、难度级别等动态选择query长度。
"""

import random
from typing import List, Dict, Any, Optional
from enum import Enum


class LengthSelectionStrategy(Enum):
    """长度选择策略"""
    FIXED = "fixed"           # 固定长度
    RANDOM = "random"         # 随机选择
    PHASE_BASED = "phase_based"   # 基于阶段
    DIFFICULTY_BASED = "difficulty_based"  # 基于难度
    ADAPTIVE = "adaptive"     # 自适应


class DynamicLengthSelector:
    """
    动态长度选择器

    根据多种因素动态选择query长度:
    - 对话阶段 (grounding/stress_test/final)
    - 难度级别
    - 历史响应质量
    - 任务类型
    """

    def __init__(
        self,
        strategy: LengthSelectionStrategy = LengthSelectionStrategy.ADAPTIVE,
        default_length: str = "medium"
    ):
        self.strategy = strategy
        self.default_length = default_length

        # 各阶段的长度分布配置
        self.phase_distributions = {
            "grounding": {
                "short": 0.2,
                "medium": 0.6,
                "long": 0.2
            },
            "noise_injection": {
                "short": 0.4,
                "medium": 0.4,
                "long": 0.2
            },
            "stress_test": {
                "short": 0.5,
                "medium": 0.3,
                "long": 0.2
            },
            "final": {
                "short": 0.1,
                "medium": 0.3,
                "long": 0.6
            },
            "final_evaluation": {
                "short": 0.1,
                "medium": 0.3,
                "long": 0.6
            },
            "entity_grounding": {
                "short": 0.2,
                "medium": 0.6,
                "long": 0.2
            },
            "chain_navigation": {
                "short": 0.3,
                "medium": 0.5,
                "long": 0.2
            },
            "chain_verification": {
                "short": 0.2,
                "medium": 0.4,
                "long": 0.4
            },
            "noise_during_reasoning": {
                "short": 0.4,
                "medium": 0.4,
                "long": 0.2
            },
            "final_answer": {
                "short": 0.1,
                "medium": 0.3,
                "long": 0.6
            },
            "target_presentation": {
                "short": 0.2,
                "medium": 0.6,
                "long": 0.2
            },
            "memory_attack": {
                "short": 0.4,
                "medium": 0.4,
                "long": 0.2
            },
            "final_test": {
                "short": 0.1,
                "medium": 0.3,
                "long": 0.6
            }
        }

        # 难度级别对长度的影响
        self.difficulty_modifiers = {
            1: {"short": -0.1, "medium": 0.0, "long": 0.1},   # 简单: 偏向详细
            2: {"short": 0.0, "medium": 0.0, "long": 0.0},    # 中等: 均衡
            3: {"short": 0.1, "medium": 0.0, "long": -0.1},   # 困难: 偏向简洁
            4: {"short": 0.2, "medium": 0.0, "long": -0.2}    # 极难: 更偏向简洁
        }

    def select(
        self,
        phase: Optional[str] = None,
        difficulty: int = 2,
        turn: int = 0,
        history: Optional[List[Dict]] = None,
        task_type: Optional[str] = None,
        context: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        选择query长度

        Args:
            phase: 当前阶段
            difficulty: 难度级别 (1-4)
            turn: 当前轮次
            history: 对话历史
            task_type: 任务类型
            context: 额外上下文

        Returns:
            {
                "length": str,          # 选择的长度
                "reason": str,          # 选择原因
                "probabilities": Dict   # 各长度的概率
            }
        """
        history = history or []
        context = context or {}

        if self.strategy == LengthSelectionStrategy.FIXED:
            return self._select_fixed()

        elif self.strategy == LengthSelectionStrategy.RANDOM:
            return self._select_random()

        elif self.strategy == LengthSelectionStrategy.PHASE_BASED:
            return self._select_by_phase(phase)

        elif self.strategy == LengthSelectionStrategy.DIFFICULTY_BASED:
            return self._select_by_difficulty(difficulty)

        elif self.strategy == LengthSelectionStrategy.ADAPTIVE:
            return self._select_adaptive(
                phase=phase,
                difficulty=difficulty,
                turn=turn,
                history=history,
                task_type=task_type
            )

        return self._select_fixed()

    def _select_fixed(self) -> Dict[str, Any]:
        """固定长度"""
        return {
            "length": self.default_length,
            "reason": "fixed_strategy",
            "probabilities": {self.default_length: 1.0}
        }

    def _select_random(self) -> Dict[str, Any]:
        """随机选择"""
        lengths = ["short", "medium", "long"]
        selected = random.choice(lengths)
        return {
            "length": selected,
            "reason": "random_selection",
            "probabilities": {l: 1/3 for l in lengths}
        }

    def _select_by_phase(self, phase: Optional[str]) -> Dict[str, Any]:
        """基于阶段选择"""
        if phase is None:
            phase = "grounding"

        distribution = self.phase_distributions.get(
            phase,
            self.phase_distributions["grounding"]
        )

        selected = self._sample_from_distribution(distribution)

        return {
            "length": selected,
            "reason": f"phase_based_{phase}",
            "probabilities": distribution
        }

    def _select_by_difficulty(self, difficulty: int) -> Dict[str, Any]:
        """基于难度选择"""
        base_distribution = {"short": 0.33, "medium": 0.34, "long": 0.33}

        modifiers = self.difficulty_modifiers.get(
            difficulty,
            self.difficulty_modifiers[2]
        )

        adjusted = {
            length: max(0, base_distribution[length] + modifiers[length])
            for length in base_distribution
        }

        # 归一化
        total = sum(adjusted.values())
        adjusted = {k: v/total for k, v in adjusted.items()}

        selected = self._sample_from_distribution(adjusted)

        return {
            "length": selected,
            "reason": f"difficulty_based_level_{difficulty}",
            "probabilities": adjusted
        }

    def _select_adaptive(
        self,
        phase: Optional[str],
        difficulty: int,
        turn: int,
        history: List[Dict],
        task_type: Optional[str]
    ) -> Dict[str, Any]:
        """自适应选择"""

        # 从阶段分布开始
        phase = phase or "grounding"
        base_dist = self.phase_distributions.get(
            phase,
            self.phase_distributions["grounding"]
        ).copy()

        # 应用难度修正
        modifiers = self.difficulty_modifiers.get(difficulty, self.difficulty_modifiers[2])
        for length in base_dist:
            base_dist[length] = max(0, base_dist[length] + modifiers[length])

        # 基于历史调整
        if history:
            # 如果最近3轮都是同一长度，增加其他长度的概率
            recent_lengths = [
                h.get("query_length_control", "medium")
                for h in history[-3:]
            ]
            if len(set(recent_lengths)) == 1 and len(recent_lengths) == 3:
                dominant = recent_lengths[0]
                # 减少主导长度，增加其他长度
                for length in base_dist:
                    if length == dominant:
                        base_dist[length] *= 0.5
                    else:
                        base_dist[length] *= 1.25

        # 基于任务类型调整
        if task_type:
            if "reasoning" in task_type.lower():
                # 推理任务偏向详细
                base_dist["long"] *= 1.2
            elif "comparison" in task_type.lower():
                # 比较任务偏向中等
                base_dist["medium"] *= 1.2
            elif "bridge" in task_type.lower():
                # 桥式推理偏向中长
                base_dist["medium"] *= 1.1
                base_dist["long"] *= 1.1

        # 归一化
        total = sum(base_dist.values())
        if total > 0:
            base_dist = {k: v/total for k, v in base_dist.items()}
        else:
            base_dist = {"short": 0.33, "medium": 0.34, "long": 0.33}

        selected = self._sample_from_distribution(base_dist)

        return {
            "length": selected,
            "reason": f"adaptive_phase={phase}_diff={difficulty}_turn={turn}",
            "probabilities": base_dist
        }

    def _sample_from_distribution(self, distribution: Dict[str, float]) -> str:
        """从分布中采样"""
        lengths = list(distribution.keys())
        weights = list(distribution.values())
        return random.choices(lengths, weights=weights, k=1)[0]

    def set_phase_distribution(self, phase: str, distribution: Dict[str, float]):
        """设置特定阶段的长度分布"""
        # 验证分布
        total = sum(distribution.values())
        if abs(total - 1.0) > 0.01:
            # 归一化
            distribution = {k: v/total for k, v in distribution.items()}
        self.phase_distributions[phase] = distribution

    def get_phase_distribution(self, phase: str) -> Dict[str, float]:
        """获取特定阶段的长度分布"""
        return self.phase_distributions.get(
            phase,
            self.phase_distributions["grounding"]
        )


# ========== 预设配置 ==========

def get_length_selector_presets() -> Dict[str, DynamicLengthSelector]:
    """获取预设的长度选择器配置"""
    return {
        "default": DynamicLengthSelector(
            strategy=LengthSelectionStrategy.ADAPTIVE
        ),
        "stress_test": DynamicLengthSelector(
            strategy=LengthSelectionStrategy.PHASE_BASED
        ),
        "quick": DynamicLengthSelector(
            strategy=LengthSelectionStrategy.FIXED,
            default_length="short"
        ),
        "detailed": DynamicLengthSelector(
            strategy=LengthSelectionStrategy.FIXED,
            default_length="long"
        ),
        "random": DynamicLengthSelector(
            strategy=LengthSelectionStrategy.RANDOM
        ),
        "difficulty_adaptive": DynamicLengthSelector(
            strategy=LengthSelectionStrategy.DIFFICULTY_BASED
        )
    }


def create_length_selector(
    strategy: str = "adaptive",
    default_length: str = "medium",
    **kwargs
) -> DynamicLengthSelector:
    """
    工厂函数：创建长度选择器

    Args:
        strategy: 策略名称 ("fixed", "random", "phase_based", "difficulty_based", "adaptive")
        default_length: 默认长度
        **kwargs: 其他配置

    Returns:
        DynamicLengthSelector实例
    """
    strategy_map = {
        "fixed": LengthSelectionStrategy.FIXED,
        "random": LengthSelectionStrategy.RANDOM,
        "phase_based": LengthSelectionStrategy.PHASE_BASED,
        "difficulty_based": LengthSelectionStrategy.DIFFICULTY_BASED,
        "adaptive": LengthSelectionStrategy.ADAPTIVE
    }

    strategy_enum = strategy_map.get(strategy.lower(), LengthSelectionStrategy.ADAPTIVE)

    return DynamicLengthSelector(
        strategy=strategy_enum,
        default_length=default_length
    )
