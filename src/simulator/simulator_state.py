"""
Simulator State - 状态机数据结构
================================

统一管理 Simulator 的状态信息，包括：
- 任务信息
- 阶段控制
- 评估结果
- 控制信号
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from enum import Enum


class Phase(Enum):
    """对话阶段"""
    GROUNDING = "grounding"           # 建立基础理解
    NOISE_INJECTION = "noise_injection"  # 注入噪声/干扰
    STRESS_TEST = "stress_test"        # 压力测试
    FINAL = "final"                    # 最终评估


@dataclass
class ResponseEvaluation:
    """
    对待测 VLM 回复的评估

    多维度评估，重点关注推理能力和状态追踪稳定性
    """
    # 推理能力 (1-5)
    reasoning_quality: int = 0
    reasoning_notes: str = ""

    # 记忆/状态追踪稳定性 (1-5)
    memory_stability: int = 0
    memory_notes: str = ""

    # 置信度 (1-5): 模型是否表现出合理的不确定性
    confidence_calibration: int = 0
    confidence_notes: str = ""

    # 整体评估
    overall_notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "reasoning_quality": self.reasoning_quality,
            "reasoning_notes": self.reasoning_notes,
            "memory_stability": self.memory_stability,
            "memory_notes": self.memory_notes,
            "confidence_calibration": self.confidence_calibration,
            "confidence_notes": self.confidence_notes,
            "overall_notes": self.overall_notes
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ResponseEvaluation":
        return cls(
            reasoning_quality=data.get("reasoning_quality", 0),
            reasoning_notes=data.get("reasoning_notes", ""),
            memory_stability=data.get("memory_stability", 0),
            memory_notes=data.get("memory_notes", ""),
            confidence_calibration=data.get("confidence_calibration", 0),
            confidence_notes=data.get("confidence_notes", ""),
            overall_notes=data.get("overall_notes", "")
        )


@dataclass
class SimulatorState:
    """
    Simulator 状态机

    包含任务 D 类信息：
    - 任务 ID 和状态
    - 任务核心 rationale 和涉及的实体
    - 当前轮次对待测 VLM 回复质量的评估
    - phase、当前对话轮数等
    """

    # === 任务标识 ===
    task_id: str
    task_type: str

    # === 阶段控制 ===
    phase: Phase = Phase.GROUNDING
    current_turn: int = 0
    max_turns: int = 10

    # === 任务核心信息 ===
    rationale: str = ""               # 任务核心推理链描述
    key_entities: List[str] = field(default_factory=list)  # 关键实体

    # === 评估信息 ===
    last_response_evaluation: Optional[ResponseEvaluation] = None
    cumulative_scores: Dict[str, float] = field(default_factory=dict)
    evaluation_history: List[ResponseEvaluation] = field(default_factory=list)

    # === 动作历史 ===
    action_history: List[str] = field(default_factory=list)

    # === 控制信号 ===
    should_end: bool = False
    end_reason: Optional[str] = None
    task_progress: str = "incomplete"  # incomplete / partial / complete / failed

    def advance_turn(self):
        """推进到下一轮"""
        self.current_turn += 1
        if self.current_turn >= self.max_turns:
            self.should_end = True
            self.end_reason = "max_turns_reached"

    def update_phase(self):
        """根据轮次自动更新阶段"""
        progress_ratio = self.current_turn / self.max_turns

        if progress_ratio < 0.25:
            self.phase = Phase.GROUNDING
        elif progress_ratio < 0.5:
            self.phase = Phase.NOISE_INJECTION
        elif progress_ratio < 0.8:
            self.phase = Phase.STRESS_TEST
        else:
            self.phase = Phase.FINAL

    def record_action(self, action: str):
        """记录动作"""
        self.action_history.append(action)

    def record_evaluation(self, evaluation: ResponseEvaluation):
        """记录评估"""
        self.last_response_evaluation = evaluation
        self.evaluation_history.append(evaluation)

        # 更新累积分数
        if evaluation.reasoning_quality > 0:
            scores = self.cumulative_scores
            n = len(self.evaluation_history)

            # 增量平均
            for key in ["reasoning_quality", "memory_stability", "confidence_calibration"]:
                val = getattr(evaluation, key, 0)
                if val > 0:
                    old_avg = scores.get(key, 0)
                    scores[key] = old_avg + (val - old_avg) / n

    def get_recent_actions(self, n: int = 3) -> List[str]:
        """获取最近 n 个动作"""
        return self.action_history[-n:] if self.action_history else []

    def mark_complete(self, status: str = "complete", reason: str = ""):
        """标记任务完成"""
        self.should_end = True
        self.task_progress = status
        self.end_reason = reason or f"task_{status}"

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典（用于日志）"""
        return {
            "task_id": self.task_id,
            "task_type": self.task_type,
            "phase": self.phase.value,
            "current_turn": self.current_turn,
            "max_turns": self.max_turns,
            "rationale": self.rationale,
            "key_entities": self.key_entities,
            "last_evaluation": self.last_response_evaluation.to_dict() if self.last_response_evaluation else None,
            "cumulative_scores": self.cumulative_scores,
            "action_history": self.action_history,
            "should_end": self.should_end,
            "end_reason": self.end_reason,
            "task_progress": self.task_progress
        }

    def get_context_summary(self) -> str:
        """获取状态摘要（用于 Context Builder）"""
        eval_summary = ""
        if self.last_response_evaluation:
            eval_summary = (
                f"上轮评估: 推理={self.last_response_evaluation.reasoning_quality}/5, "
                f"记忆稳定性={self.last_response_evaluation.memory_stability}/5"
            )

        return f"""- 任务ID: {self.task_id}
- 任务类型: {self.task_type}
- 当前阶段: {self.phase.value}
- 当前轮次: {self.current_turn}/{self.max_turns}
- 任务核心: {self.rationale}
- 关键实体: {', '.join(self.key_entities) if self.key_entities else 'N/A'}
- 最近动作: {', '.join(self.get_recent_actions())}
- {eval_summary}"""


# 测试
if __name__ == "__main__":
    # 创建状态
    state = SimulatorState(
        task_id="task_001",
        task_type="attribute_comparison",
        rationale="比较两张图中人的数量",
        key_entities=["person", "count"],
        max_turns=8
    )

    print("初始状态:")
    print(state.get_context_summary())

    # 模拟几轮
    for i in range(4):
        state.advance_turn()
        state.update_phase()
        state.record_action(["guidance", "follow_up", "mislead", "fine_grained"][i])

        eval_result = ResponseEvaluation(
            reasoning_quality=3 + i % 2,
            reasoning_notes="推理清晰" if i % 2 else "推理略有跳跃",
            memory_stability=4,
            memory_notes="记忆保持良好"
        )
        state.record_evaluation(eval_result)

    print("\n4轮后状态:")
    print(state.get_context_summary())
    print(f"\n累积分数: {state.cumulative_scores}")
