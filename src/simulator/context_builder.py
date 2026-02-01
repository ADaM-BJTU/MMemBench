"""
Context Builder - 统一构建核心模型的上下文
==========================================

打包 6 类信息:
A: Entity 提取结果
B: 动作空间、模板、参数
C: 上下文历史摘要
D: 状态机信息
E: Ground Truth / 标注
F: 图片描述
"""

from typing import Dict, List, Any, Optional
from .simulator_state import SimulatorState, Phase
from .action_space import ACTION_DEFINITIONS, ActionCategory


class ContextBuilder:
    """
    统一构建给核心模型的上下文

    设计理念:
    - 把所有信息打包成一个结构化的 prompt
    - 让核心模型基于完整上下文生成自然的对话
    - 不需要 creativity level，直接让 LLM 发挥
    """

    # 核心 prompt 模板
    CORE_PROMPT_TEMPLATE = """你是一个用户模拟器，正在测试一个多模态视觉语言模型（VLM）的能力。
你的目标是通过多轮对话，全面测试模型的推理能力和状态追踪稳定性。

## A. 待测模型刚才的回复

{vlm_response}

## B. 从回复中提取的实体

{entities_summary}

## C. 可用的动作类型

当前阶段 [{phase}] 推荐的动作:
{recommended_actions}

所有可用动作:
{all_actions_summary}

## D. 任务状态

{state_summary}

## E. 标注信息

- 原始问题: {question}
- 正确答案: {answer}
- 评估重点: {evaluation_focus}

## F. 图片信息

{image_descriptions}

## G. 对话历史摘要

{history_summary}

---

## 你的任务

根据当前阶段 [{phase}] 和任务目标，生成一个自然的用户问题或陈述。

### 阶段指导

{phase_guidance}

### 要求

1. **语言自然**: 像真实用户的对话，不要机械化
2. **推进任务**: 每轮对话都应该有明确目的，测试模型某方面能力
3. **利用实体**: 适当使用从回复中提取的实体
4. **评估回复**: 对模型上一轮回复的质量给出评估

### 输出格式 (JSON)

```json
{{
    "action": "选择的动作类型（从可用动作中选择）",
    "query": "发给待测模型的自然语言问题/陈述（1-3句话）",
    "reasoning": "为什么选择这个动作和问题（简要说明）",
    "evaluation": {{
        "reasoning_quality": 1-5,
        "reasoning_notes": "对推理能力的评估说明",
        "memory_stability": 1-5,
        "memory_notes": "对记忆/状态追踪的评估说明",
        "confidence_calibration": 1-5,
        "confidence_notes": "模型是否表现出合理的不确定性",
        "overall_notes": "其他观察"
    }},
    "task_progress": "incomplete/partial/complete/failed",
    "key_observations": ["从回复中注意到的关键信息1", "关键信息2"]
}}
```
"""

    # 各阶段的指导
    PHASE_GUIDANCE = {
        Phase.GROUNDING: """
**Grounding 阶段**: 建立基础理解
- 使用 guidance、follow_up、fine_grained 动作
- 引导模型观察图片关键区域
- 确认模型对基本事实的理解是否正确
- 不要急于挑战，先建立 baseline
""",
        Phase.NOISE_INJECTION: """
**Noise Injection 阶段**: 注入噪声测试鲁棒性
- 使用 mislead、distraction、redundancy 动作
- 尝试用错误信息误导模型
- 插入无关话题分散注意力
- 观察模型是否能保持原有正确认知
""",
        Phase.STRESS_TEST: """
**Stress Test 阶段**: 压力测试
- 使用 mislead_subtle、memory_injection、inconsistency_injection 动作
- 进行更隐蔽的误导尝试
- 测试跨图记忆混淆
- 注入虚假记忆，看模型是否会接受
""",
        Phase.FINAL: """
**Final 阶段**: 最终评估
- 使用 fine_grained、consistency_check 动作
- 回到核心问题，获取最终答案
- 验证模型经过干扰后是否还能给出正确答案
- 这是最关键的评估轮次
"""
    }

    def __init__(self):
        """初始化 Context Builder"""
        pass

    def build(
        self,
        vlm_response: str,
        entities: Dict[str, Any],
        history_summary: str,
        state: SimulatorState,
        ground_truth: Dict[str, Any],
        image_descriptions: List[str]
    ) -> str:
        """
        构建完整的核心模型 prompt

        Args:
            vlm_response: 待测 VLM 的最新回复 (A 的来源)
            entities: Entity 提取结果 (A)
            history_summary: 上下文历史摘要 (C)
            state: 状态机信息 (D)
            ground_truth: Ground Truth 标注 (E)
            image_descriptions: 图片描述列表 (F)

        Returns:
            完整的 prompt 字符串
        """
        # A: 格式化实体
        entities_summary = self._format_entities(entities)

        # B: 格式化动作空间
        recommended_actions = self._get_recommended_actions(state.phase)
        all_actions_summary = self._format_all_actions()

        # D: 状态摘要
        state_summary = state.get_context_summary()

        # E: 标注信息
        question = ground_truth.get("question", "N/A")
        answer = ground_truth.get("answer", "N/A")
        evaluation_focus = self._get_evaluation_focus(state.task_type)

        # F: 图片描述
        image_desc_text = self._format_image_descriptions(image_descriptions)

        # 阶段指导
        phase_guidance = self.PHASE_GUIDANCE.get(state.phase, "")

        # 组装 prompt
        prompt = self.CORE_PROMPT_TEMPLATE.format(
            vlm_response=vlm_response[:1000] if vlm_response else "(无回复)",
            entities_summary=entities_summary,
            phase=state.phase.value,
            recommended_actions=recommended_actions,
            all_actions_summary=all_actions_summary,
            state_summary=state_summary,
            question=question,
            answer=answer,
            evaluation_focus=evaluation_focus,
            image_descriptions=image_desc_text,
            history_summary=history_summary if history_summary else "(对话开始)",
            phase_guidance=phase_guidance
        )

        return prompt

    def _format_entities(self, entities: Dict[str, Any]) -> str:
        """格式化实体提取结果"""
        if not entities:
            return "(未提取到实体)"

        lines = []

        # 物体
        objects = entities.get("objects", [])
        if objects:
            lines.append(f"- 物体: {', '.join(objects[:10])}")

        # 属性
        attributes = entities.get("attributes", [])
        if attributes:
            lines.append(f"- 属性: {', '.join(attributes[:10])}")

        # 区域
        regions = entities.get("regions", [])
        if regions:
            lines.append(f"- 区域: {', '.join(regions[:5])}")

        # 数量
        counts = entities.get("counts", {})
        if counts:
            count_strs = [f"{k}: {v}" for k, v in list(counts.items())[:5]]
            lines.append(f"- 数量: {', '.join(count_strs)}")

        # 关系
        relations = entities.get("relations", [])
        if relations:
            lines.append(f"- 关系: {', '.join(relations[:5])}")

        return "\n".join(lines) if lines else "(未提取到实体)"

    def _get_recommended_actions(self, phase: Phase) -> str:
        """根据阶段获取推荐动作"""
        phase_actions = {
            Phase.GROUNDING: ["guidance", "follow_up", "fine_grained"],
            Phase.NOISE_INJECTION: ["mislead", "distraction", "redundancy"],
            Phase.STRESS_TEST: ["mislead_subtle", "memory_injection", "inconsistency_injection", "cross_image_confusion"],
            Phase.FINAL: ["fine_grained", "consistency_check"]
        }

        actions = phase_actions.get(phase, ["follow_up"])
        lines = []

        for action_name in actions:
            action_def = ACTION_DEFINITIONS.get(action_name)
            if action_def:
                lines.append(f"- **{action_name}**: {action_def.purpose}")

        return "\n".join(lines)

    def _format_all_actions(self) -> str:
        """格式化所有可用动作（简略版）"""
        # 按类别分组
        by_category = {}
        for name, action_def in ACTION_DEFINITIONS.items():
            cat = action_def.category.value
            if cat not in by_category:
                by_category[cat] = []
            by_category[cat].append(f"{name}")

        lines = []
        for cat, actions in by_category.items():
            lines.append(f"- {cat}: {', '.join(actions)}")

        return "\n".join(lines)

    def _get_evaluation_focus(self, task_type: str) -> str:
        """根据任务类型获取评估重点"""
        focus_map = {
            "attribute_comparison": "比较准确性、数量计数、属性识别",
            "visual_noise_filtering": "噪声过滤能力、注意力保持",
            "attribute_bridge_reasoning": "多跳推理、推理链完整性",
            "spatial_reasoning": "空间关系理解、位置描述准确性",
            "temporal_reasoning": "时序理解、状态变化追踪"
        }
        return focus_map.get(task_type, "推理能力和状态追踪稳定性")

    def _format_image_descriptions(self, descriptions: List[str]) -> str:
        """格式化图片描述"""
        if not descriptions:
            return "(无图片描述)"

        lines = []
        for i, desc in enumerate(descriptions):
            lines.append(f"Image {i}: {desc[:200]}...")

        return "\n".join(lines)

    def build_initial_prompt(
        self,
        state: SimulatorState,
        ground_truth: Dict[str, Any],
        image_descriptions: List[str]
    ) -> str:
        """
        构建初始轮次的 prompt（没有 VLM 回复）

        用于对话的第一轮
        """
        return self.build(
            vlm_response="(这是对话的开始，待测模型尚未回复)",
            entities={},
            history_summary="",
            state=state,
            ground_truth=ground_truth,
            image_descriptions=image_descriptions
        )


# 测试
if __name__ == "__main__":
    from .simulator_state import SimulatorState, Phase

    # 创建测试数据
    state = SimulatorState(
        task_id="task_001",
        task_type="attribute_comparison",
        phase=Phase.GROUNDING,
        current_turn=2,
        max_turns=8,
        rationale="比较两张图中人的数量，判断哪张图人更多",
        key_entities=["person", "count", "image"]
    )

    entities = {
        "objects": ["person", "umbrella", "car"],
        "attributes": ["red", "standing", "walking"],
        "counts": {"person": 3, "umbrella": 1}
    }

    ground_truth = {
        "question": "哪张图中的人更多？",
        "answer": "Image 0 中有5个人，Image 1 中有3个人，所以 Image 0 人更多"
    }

    image_descriptions = [
        "Image 0: 一个繁忙的街道场景，有多人行走",
        "Image 1: 一个相对安静的公园场景"
    ]

    # 构建 context
    builder = ContextBuilder()
    prompt = builder.build(
        vlm_response="我看到图片中有几个人在街道上行走，其中一个人拿着红色雨伞。",
        entities=entities,
        history_summary="Turn 1: 用户询问图片中有什么。模型回复描述了基本场景。",
        state=state,
        ground_truth=ground_truth,
        image_descriptions=image_descriptions
    )

    print("=" * 60)
    print("Generated Context Prompt:")
    print("=" * 60)
    print(prompt)
