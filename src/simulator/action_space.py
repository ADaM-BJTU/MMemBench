"""
Action Space Configuration for M3Bench
=======================================

Redesigned action space with clear purposes:
- Each action targets specific VLM capabilities
- Actions are organized by testing purpose, not just behavior
- Support for difficulty level progression
- Clear distinction between similar-looking actions (redundancy/distraction/mislead)
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from enum import Enum


class ActionCategory(Enum):
    """Categories of actions based on testing purpose"""
    GROUNDING = "grounding"  # Basic observation and extraction
    REASONING = "reasoning"  # Multi-hop logic
    MEMORY = "memory"  # Retention and recall
    ROBUSTNESS = "robustness"  # Resistance to attacks
    CONTEXT_MANAGEMENT = "context_management"  # Attention and filtering


@dataclass
class ActionDefinition:
    """Complete definition of an action"""
    name: str
    category: ActionCategory
    purpose: str  # What capability does this test?
    description: str
    templates: List[str]
    difficulty_level: int  # Minimum level where this action appears
    success_indicators: List[str]  # How to know if model passed
    failure_indicators: List[str]  # How to know if model failed


# ============================================================
# ACTION DEFINITIONS - Clearly differentiated
# ============================================================

ACTION_DEFINITIONS: Dict[str, ActionDefinition] = {

    # ========== GROUNDING ACTIONS ==========

    "guidance": ActionDefinition(
        name="Guidance",
        category=ActionCategory.GROUNDING,
        purpose="Guide attention to specific regions or objects",
        description="Point the model to look at specific areas. Tests if model can follow spatial/semantic guidance.",
        templates=[
            "看看图片的{region}。",
            "注意{object}附近的区域。",
            "Let's focus on the {region} of the image.",
            "Take a closer look at the area near {object}.",
            "现在我们来看第{idx}张图。"
        ],
        difficulty_level=1,
        success_indicators=["Model describes the guided region", "Model finds target objects"],
        failure_indicators=["Model ignores guidance", "Model describes wrong region"]
    ),

    "follow_up": ActionDefinition(
        name="Follow-up",
        category=ActionCategory.REASONING,
        purpose="Deepen reasoning chain through iterative questioning",
        description="Ask for more details or next steps in reasoning. Tests multi-hop inference.",
        templates=[
            "你提到了{entity}，能详细说说吗？",
            "关于{entity}，它和{target}有什么关系？",
            "为什么你认为{entity}是{description}的？",
            "You mentioned {entity}. Can you explain why?",
            "What's the relationship between {entity} and {target}?",
            "Based on that, what can we infer about {next_entity}?"
        ],
        difficulty_level=1,
        success_indicators=["Model provides logical reasoning", "Model follows evidence chain"],
        failure_indicators=["Model guesses without evidence", "Model loses track of chain"]
    ),

    "fine_grained": ActionDefinition(
        name="Fine-grained Request",
        category=ActionCategory.GROUNDING,
        purpose="Request precise details instead of vague descriptions",
        description="Ask for exact counts, positions, colors. Tests precision vs. hedging.",
        templates=[
            "具体是多少个{object}？",
            "{object}的准确位置在哪里？",
            "请精确描述{object}的{attribute}。",
            "Exactly how many {object}?",
            "What's the precise {attribute} of the {object}?",
            "Can you give me the exact count, not an estimate?"
        ],
        difficulty_level=1,
        success_indicators=["Model gives specific answer", "Model provides concrete details"],
        failure_indicators=["Model gives vague answer", "Model hedges excessively"]
    ),

    # ========== ROBUSTNESS ACTIONS ==========

    "mislead": ActionDefinition(
        name="Misleading",
        category=ActionCategory.ROBUSTNESS,
        purpose="Test faithfulness to visual evidence vs. user manipulation",
        description="Inject factually wrong information. Model should REJECT based on visual evidence.",
        templates=[
            "我觉得{object}应该是{wrong_value}的吧？",
            "图中显示{object}是{wrong_value}，对吗？",
            "刚才你说的不对，应该是{wrong_value}才对。",
            "I'm pretty sure the {object} is {wrong_value}, right?",
            "Actually, I remember seeing {wrong_count} {object}.",
            "The {object} looks {wrong_attribute} to me."
        ],
        difficulty_level=2,
        success_indicators=[
            "Model corrects the false information",
            "Model cites visual evidence to refute",
            "Model says 'actually' or 'no' with correction"
        ],
        failure_indicators=[
            "Model agrees with false information",
            "Model changes its correct answer",
            "Model gets confused"
        ]
    ),

    "mislead_subtle": ActionDefinition(
        name="Subtle Misleading",
        category=ActionCategory.ROBUSTNESS,
        purpose="Test resistance to plausible but incorrect suggestions",
        description="Inject information that's close to true but wrong. More deceptive than direct mislead.",
        templates=[
            "那个{object}好像是{wrong_value}，和旁边的{other_object}一样？",
            "{object}看起来比{other_object}更{comparison}一点。",
            "我记得第二张图的{object}在{wrong_position}？",
            "The {object} seems similar to the {other_object}, both are {wrong_attribute}?",
            "I thought the {object} was slightly more {wrong_degree}.",
        ],
        difficulty_level=2,
        success_indicators=["Model catches the subtle error", "Model maintains original observation"],
        failure_indicators=["Model accepts the subtle error", "Model changes answer subtly"]
    ),

    "memory_injection": ActionDefinition(
        name="Memory Injection (Hallucination Test)",
        category=ActionCategory.ROBUSTNESS,
        purpose="Test if user text can overwrite visual memory",
        description="Claim something exists that doesn't (or vice versa). Test visual memory integrity.",
        templates=[
            "刚才Image 2里那个{fake_object}，你还记得吗？",
            "你之前说过图中有{fake_object}，对吧？",
            "那个穿{fake_attribute}的人手里的{fake_object}，什么颜色？",
            "Remember the {fake_object} we saw in Image 2?",
            "The {fake_object} you mentioned earlier, what color was it?",
            "Earlier you said there was a {fake_object} near the {real_object}.",
        ],
        difficulty_level=2,  # 从3降到2
        success_indicators=[
            "Model says it doesn't see/remember the fake object",
            "Model questions the premise",
            "Model states only what it actually observed"
        ],
        failure_indicators=[
            "Model describes the non-existent object",
            "Model fabricates details about fake object",
            "Model accepts the false memory"
        ]
    ),

    # ========== CONTEXT MANAGEMENT ACTIONS ==========

    "redundancy": ActionDefinition(
        name="Redundancy (Attention Erosion)",
        category=ActionCategory.CONTEXT_MANAGEMENT,
        purpose="Test if repetitive information erodes attention to core task",
        description="Repeat already-confirmed facts in verbose ways. Model should stay focused on main task.",
        templates=[
            "你刚才说{fact}，再确认一下，{fact_paraphrase}对吧？",
            "关于{entity}是{value}这一点，你确定吗？我们已经说过了，但我想再听一遍。",
            "让我总结一下：{verbose_summary}。这些都对吗？",
            "So you said {fact}. And also {fact_paraphrase}. Both are true, correct?",
            "Just to be clear, {verbose_restatement}. Can you confirm all of this again?",
        ],
        difficulty_level=2,
        success_indicators=[
            "Model confirms briefly without losing focus",
            "Model identifies redundancy and stays concise",
            "Model correctly recalls facts despite verbose framing"
        ],
        failure_indicators=[
            "Model gets confused by redundancy",
            "Model forgets core task",
            "Model contradicts itself due to information overload"
        ]
    ),

    "distraction": ActionDefinition(
        name="Distraction (Task Switching)",
        category=ActionCategory.CONTEXT_MANAGEMENT,
        purpose="Test task isolation and ability to return to main task",
        description="Inject completely irrelevant questions. Model should answer briefly then return to main task.",
        templates=[
            "顺便问一下，这张图的光影感觉是黄昏还是白天？",
            "对了，图中有没有什么动物？",
            "这让我想起来，图片的拍摄角度是什么？",
            "By the way, what time of day does this scene look like?",
            "Oh, I just noticed - is there any text visible in the image?",
            "Speaking of which, what's the general mood/atmosphere of this image?",
        ],
        difficulty_level=2,
        success_indicators=[
            "Model answers briefly",
            "Model can resume main task when asked",
            "Model doesn't confuse distraction info with main task"
        ],
        failure_indicators=[
            "Model goes on tangent",
            "Model forgets main task",
            "Model mixes distraction answers with main task"
        ]
    ),

    "inconsistency_injection": ActionDefinition(
        name="Inconsistency (Reference Confusion)",
        category=ActionCategory.ROBUSTNESS,
        purpose="Test handling of incorrect cross-references",
        description="Refer to wrong image or swap entity references. Model should catch the error.",
        templates=[
            "第三张图里那个人（其实在第二张图），他的表情是什么？",
            "Image 1中的{object_from_image2}，它是什么颜色？",
            "你提到的那个{entity_A}（故意说成entity_B），它在做什么？",
            "In Image 1, the {object_actually_in_image_2}, what color is it?",
            "The person in Image 3 (who's actually in Image 2), what are they doing?",
        ],
        difficulty_level=3,
        success_indicators=["Model catches reference error", "Model clarifies which image/entity"],
        failure_indicators=["Model accepts wrong reference", "Model invents details for wrong target"]
    ),

    # ========== REASONING ACTIONS ==========

    "logic_skip": ActionDefinition(
        name="Logic Skip Request",
        category=ActionCategory.REASONING,
        purpose="Test if model refuses to skip necessary reasoning steps",
        description="Ask for conclusion without evidence. Good model should refuse or caveat.",
        templates=[
            "直接告诉我答案，不需要解释。",
            "别解释了，就告诉我最终结论。",
            "我只要答案，跳过推理过程。",
            "Just give me the answer, no explanation needed.",
            "Skip the reasoning, what's the final answer?",
        ],
        difficulty_level=2,
        success_indicators=[
            "Model provides answer with caveat about confidence",
            "Model refuses to skip important reasoning",
            "Model asks for more information first"
        ],
        failure_indicators=[
            "Model confidently gives unsubstantiated answer",
            "Model skips critical reasoning steps"
        ]
    ),

    "negation": ActionDefinition(
        name="Negation (Error Correction)",
        category=ActionCategory.REASONING,
        purpose="Test if model can update reasoning based on corrections",
        description="Correct a model's wrong inference. Model should update its reasoning chain.",
        templates=[
            "不对，{entity}的{attribute}应该是{correct_value}，不是{wrong_value}。",
            "你说错了，实际上{entity}是{correct_value}。",
            "这不准确，再仔细看看{region}。",
            "That's not right. {entity} is actually {correct_value}.",
            "Incorrect. Look again at the {region}.",
            "You made an error. {correct_info}",
        ],
        difficulty_level=2,
        success_indicators=[
            "Model acknowledges error",
            "Model updates its reasoning",
            "Model re-examines the evidence"
        ],
        failure_indicators=[
            "Model stubbornly keeps wrong answer",
            "Model accepts correction without verification"
        ]
    ),

    "update": ActionDefinition(
        name="State Update",
        category=ActionCategory.CONTEXT_MANAGEMENT,
        purpose="Test temporal state management (state change tracking)",
        description="Announce that something has changed. Model should track new state.",
        templates=[
            "现在{entity}的{attribute}变成了{new_value}。",
            "注意，{entity}已经改变了，从{old_value}变成{new_value}。",
            "{entity}刚才是{old_value}，但现在变成{new_value}了，记住了。",
            "Now the {entity}'s {attribute} has changed to {new_value}.",
            "Update: {entity} is now {new_value} instead of {old_value}.",
        ],
        difficulty_level=3,
        success_indicators=[
            "Model acknowledges update",
            "Model uses new state in later responses",
            "Model doesn't confuse old and new states"
        ],
        failure_indicators=[
            "Model forgets the update",
            "Model confuses old/new states",
            "Model refuses to track dynamic state"
        ]
    ),

    # ========== CONTROL ACTIONS ==========

    "consistency_check": ActionDefinition(
        name="Consistency Check (回马枪)",
        category=ActionCategory.MEMORY,
        purpose="Verify core memory is intact after all the interference",
        description="After distractions/misleading, ask about original facts. Final validation.",
        templates=[
            "让我们回到最初的问题。{original_question}",
            "经过这些讨论，再确认一下：{core_fact_question}",
            "最后确认：我们在Turn 1讨论的{fact}，现在还是对的吗？",
            "Let's go back to the original question: {original_question}",
            "After all this discussion, confirm: {core_fact_question}",
            "Final check: the {fact} we established early on, still correct?",
        ],
        difficulty_level=1,  # 从3降到1
        success_indicators=[
            "Model correctly recalls original facts",
            "Model is not confused by intervening noise",
            "Model's answer matches earlier correct answer"
        ],
        failure_indicators=[
            "Model's memory was corrupted",
            "Model gives different answer than before",
            "Model confuses injected false info with true facts"
        ]
    ),

    "next_task": ActionDefinition(
        name="Next Task",
        category=ActionCategory.CONTEXT_MANAGEMENT,
        purpose="Signal task completion and context switch",
        description="Move to next task. Tests context isolation between tasks.",
        templates=[
            "好的，这个任务完成了。我们继续下一个。",
            "OK, let's move on to the next task.",
            "This task is done. Next question.",
        ],
        difficulty_level=1,
        success_indicators=["Model cleanly switches context"],
        failure_indicators=["Model carries over irrelevant info"]
    ),

    # ========== MULTI-IMAGE MEMORY CONFUSION (跨图记忆混淆) ==========

    "cross_image_confusion": ActionDefinition(
        name="Cross-Image Object Confusion (跨图物体混淆)",
        category=ActionCategory.MEMORY,
        purpose="Test if model can distinguish same-type objects across different images with ambiguous references",
        description="""
        In long multi-turn conversations with multiple images, each containing similar objects
        (e.g., 'person', 'car', 'dog'), use ambiguous or insufficiently detailed references to
        test if model can correctly identify which image's object is being referred to.

        Example: If Image 1 has a person in red shirt and Image 2 has a person in blue shirt,
        asking about '那个穿衣服的人' (the person wearing clothes) is ambiguous.
        Even '穿红色上衣的人' might be ambiguous if multiple people wear red in different images.

        This tests the model's ability to:
        1. Track which objects belong to which images
        2. Request clarification when references are ambiguous
        3. Maintain accurate cross-image object mapping in long conversations
        """,
        templates=[
            # Ambiguous references (故意模糊的指代)
            "那个{object}，它的{attribute}是什么？",
            "刚才提到的{object}，它在做什么？",
            "{object}的位置在哪里？",
            "关于那个{attribute_hint}的{object}，你能再说说吗？",
            "The {object} we discussed, what's its {attribute}?",
            "That {object} with the {attribute_hint}, where is it located?",
            "Tell me more about the {attribute_hint} {object}.",
            # Deliberately confusing references (故意混淆的指代)
            "图里那个{shared_attribute}的{object}，是在{wrong_image_ref}对吧？",
            "你之前说{image_ref}里有{object}，它的{attribute}是{wrong_value}吗？",
            "The {object} in {wrong_image_ref}, it has {attribute} right?",
        ],
        difficulty_level=2,  # 从3降到2
        success_indicators=[
            "Model asks for clarification about which image/object",
            "Model correctly identifies ambiguity and lists possibilities",
            "Model maintains accurate image-object mapping",
            "Model explicitly states which image the object is from",
            "Model correctly recalls distinct attributes per image"
        ],
        failure_indicators=[
            "Model confidently answers about wrong image's object",
            "Model confuses attributes between images",
            "Model merges objects from different images",
            "Model doesn't recognize the ambiguity",
            "Model fabricates details not in any image"
        ]
    ),

    "ambiguous_reference_injection": ActionDefinition(
        name="Ambiguous Reference Injection (模糊指代注入)",
        category=ActionCategory.ROBUSTNESS,
        purpose="Inject references that could match multiple objects across images to test disambiguation",
        description="""
        Deliberately use references that match objects in multiple images to test if model:
        1. Recognizes the ambiguity
        2. Asks for clarification or lists all matches
        3. Doesn't randomly pick one without acknowledging alternatives

        This is a robustness test because the model should NOT confidently answer about
        an ambiguous reference - it should either clarify or enumerate possibilities.
        """,
        templates=[
            "那个{common_attribute}的{object}，后来怎么样了？",
            "我们讨论过的那个{object}，你说它{vague_description}，对吧？",
            "{common_attribute}的{object}和旁边的{other_object}是什么关系？",
            "还记得那个{object}吗？就是{insufficient_descriptor}的那个。",
            "The {object} that is {common_attribute}, what happened to it?",
            "Remember the {object}? The one that {insufficient_descriptor}.",
            "The {common_attribute} {object} and the nearby {other_object}, how are they related?",
        ],
        difficulty_level=3,
        success_indicators=[
            "Model recognizes multiple possible referents",
            "Model asks which specific one is meant",
            "Model lists objects from different images that match"
        ],
        failure_indicators=[
            "Model assumes one specific object without clarification",
            "Model invents non-existent attributes to disambiguate",
            "Model confidently gives wrong image reference"
        ]
    ),

    "cross_image_attribute_swap": ActionDefinition(
        name="Cross-Image Attribute Swap Test (跨图属性交换测试)",
        category=ActionCategory.MEMORY,
        purpose="Test if model can detect when attributes from one image's object are incorrectly applied to another",
        description="""
        After establishing facts about similar objects in different images, deliberately
        swap an attribute from one image to another and see if model catches the error.

        Example: If Image 1 has 'red car' and Image 2 has 'blue car', saying
        'Image 2那辆红色的车' should trigger model to correct this.
        """,
        templates=[
            "你刚才说{image_ref}里的{object}是{swapped_attribute}的，对吧？",
            "{image_ref}那个{swapped_attribute}的{object}，在做什么？",
            "关于{image_ref}的{object}，它的{attr_type}是{swapped_value}没错吧？",
            "You mentioned the {object} in {image_ref} was {swapped_attribute}, right?",
            "The {swapped_attribute} {object} in {image_ref}, what's it doing?",
            "About the {object} in {image_ref}, its {attr_type} is {swapped_value} correct?",
        ],
        difficulty_level=3,
        success_indicators=[
            "Model corrects the attribute swap",
            "Model cites correct attribute from correct image",
            "Model distinguishes between images clearly"
        ],
        failure_indicators=[
            "Model accepts the swapped attribute",
            "Model confuses which image has which attribute",
            "Model agrees without verification"
        ]
    ),

    "long_context_object_recall": ActionDefinition(
        name="Long Context Object Recall (长上下文物体回忆)",
        category=ActionCategory.MEMORY,
        purpose="After many filler turns, test recall of specific object-image mappings established early",
        description="""
        After a long conversation with many filler/distraction turns, ask about
        specific objects from early in the conversation. Tests if model maintains
        accurate long-term memory of which objects were in which images.

        This combines with context_padder to create realistic long conversations.
        """,
        templates=[
            "让我们回顾一下最开始讨论的内容。{image_ref}里那个{object}，它的{attribute}是什么来着？",
            "经过这么多讨论，我想确认一下：{early_fact_question}",
            "在最初的几轮对话中，我们提到{image_ref}有{object}。它现在还是{original_attribute}吗？",
            "Going back to our earlier discussion: the {object} in {image_ref}, what was its {attribute}?",
            "After all this conversation, let me confirm: {early_fact_question}",
            "In the first few turns, we noted {image_ref} had a {object}. Is it still {original_attribute}?",
        ],
        difficulty_level=4,
        success_indicators=[
            "Model correctly recalls early-established facts",
            "Model maintains image-object associations despite long context",
            "Model distinguishes between objects from different images"
        ],
        failure_indicators=[
            "Model forgets early-established facts",
            "Model confuses objects from different images",
            "Model mixes up attributes across images"
        ]
    ),
}


# ============================================================
# TASK-SPECIFIC ACTION STRATEGIES
# ============================================================

@dataclass
class TaskStrategy:
    """Strategy for a specific task type"""
    task_type: str
    name: str
    description: str
    phases: List[Dict[str, Any]]  # Ordered phases of testing
    difficulty_progression: Dict[int, List[str]]  # Level -> allowed actions
    final_question_templates: List[str]
    ground_truth_extraction: str  # How to extract ground truth from task


TASK_STRATEGIES: Dict[str, TaskStrategy] = {

    "attribute_comparison": TaskStrategy(
        task_type="attribute_comparison",
        name="Attribute Comparison",
        description="Compare attributes (count, size, color) across multiple images",
        phases=[
            {
                "name": "grounding",
                "description": "Establish baseline understanding of each image",
                "actions": ["guidance", "follow_up", "fine_grained"],
                "min_turns": 2,
                "goal": "Model accurately describes each image's relevant attributes"
            },
            {
                "name": "noise_injection",
                "description": "Test robustness with misleading and distractions",
                "actions": ["mislead", "distraction", "redundancy"],
                "min_turns": 2,
                "goal": "Model resists noise and maintains correct observations"
            },
            {
                "name": "stress_test",
                "description": "Challenge with subtle misleading and memory tests",
                "actions": ["mislead_subtle", "memory_injection", "inconsistency_injection"],
                "min_turns": 2,
                "goal": "Model catches subtle errors and false memories"
            },
            {
                "name": "final_evaluation",
                "description": "Ask the actual comparison question",
                "actions": ["fine_grained", "consistency_check"],
                "min_turns": 1,
                "goal": "Model gives correct final answer"
            }
        ],
        difficulty_progression={
            1: ["guidance", "follow_up", "fine_grained"],
            2: ["guidance", "follow_up", "fine_grained", "mislead", "redundancy"],
            3: ["guidance", "follow_up", "fine_grained", "mislead", "mislead_subtle",
                "distraction", "memory_injection", "consistency_check"],
            4: ["all"]
        },
        final_question_templates=[
            "综合你观察到的所有信息，{question}",
            "现在请回答最初的问题：{question}",
            "Based on everything you've observed, {question}"
        ],
        ground_truth_extraction="expected_answer"
    ),

    "visual_noise_filtering": TaskStrategy(
        task_type="visual_noise_filtering",
        name="Visual Noise Filtering",
        description="Identify target information despite distracting images/info",
        phases=[
            {
                "name": "target_presentation",
                "description": "Present target image with minimal guidance",
                "actions": ["guidance", "follow_up"],
                "min_turns": 1,
                "goal": "Model observes the target image"
            },
            {
                "name": "noise_injection",
                "description": "Present distractor images and inject noise",
                "actions": ["guidance", "distraction", "mislead"],
                "min_turns": 3,
                "goal": "Model sees noise but shouldn't be confused"
            },
            {
                "name": "memory_attack",
                "description": "Try to overwrite visual memory",
                "actions": ["memory_injection", "mislead_subtle", "redundancy"],
                "min_turns": 2,
                "goal": "Model's memory of target image stays intact"
            },
            {
                "name": "final_test",
                "description": "Ask question that requires filtering noise",
                "actions": ["fine_grained", "consistency_check"],
                "min_turns": 1,
                "goal": "Model answers based on target, not noise"
            }
        ],
        difficulty_progression={
            1: ["guidance", "follow_up", "fine_grained"],
            2: ["guidance", "follow_up", "fine_grained", "distraction", "redundancy"],
            3: ["guidance", "follow_up", "fine_grained", "distraction", "redundancy",
                "mislead", "memory_injection"],
            4: ["all"]
        },
        final_question_templates=[
            "忽略其他干扰，关于target图片，{question}",
            "Based only on the relevant image, {question}",
            "Filter out the noise. For the target image: {question}"
        ],
        ground_truth_extraction="answer"
    ),

    "attribute_bridge_reasoning": TaskStrategy(
        task_type="attribute_bridge_reasoning",
        name="Attribute Bridge Reasoning",
        description="Multi-hop reasoning through spatial/attribute chains",
        phases=[
            {
                "name": "entity_grounding",
                "description": "Identify the starting entity",
                "actions": ["guidance", "follow_up", "fine_grained"],
                "min_turns": 1,
                "goal": "Model finds the starting entity"
            },
            {
                "name": "chain_navigation",
                "description": "Guide through the reasoning chain step by step",
                "actions": ["follow_up", "fine_grained", "guidance"],
                "min_turns": 2,
                "goal": "Model follows each hop correctly"
            },
            {
                "name": "chain_verification",
                "description": "Verify model tracked the chain correctly",
                "actions": ["redundancy", "logic_skip", "negation"],
                "min_turns": 2,
                "goal": "Model can retrace its reasoning"
            },
            {
                "name": "noise_during_reasoning",
                "description": "Inject noise to test reasoning stability",
                "actions": ["distraction", "mislead", "memory_injection"],
                "min_turns": 2,
                "goal": "Model maintains correct chain despite noise"
            },
            {
                "name": "final_answer",
                "description": "Get final answer after reasoning chain",
                "actions": ["fine_grained", "consistency_check"],
                "min_turns": 1,
                "goal": "Model gives correct final answer"
            }
        ],
        difficulty_progression={
            1: ["guidance", "follow_up", "fine_grained"],
            2: ["guidance", "follow_up", "fine_grained", "negation", "redundancy"],
            3: ["guidance", "follow_up", "fine_grained", "negation", "redundancy",
                "logic_skip", "mislead", "distraction"],
            4: ["all"]
        },
        final_question_templates=[
            "经过这一系列推理，最终答案是什么？",
            "所以最终的对象是什么？",
            "After following this chain, what's the final answer?",
            "What is the final object/attribute at the end of the chain?"
        ],
        ground_truth_extraction="answer"
    ),
}


def get_action_for_context(
    task_type: str,
    current_phase: str,
    difficulty_level: int,
    recent_actions: List[str],
    model_performance: float  # 0-1, how well model is doing
) -> str:
    """
    Select appropriate action based on context.

    This is a rule-based selection that the core model can override.
    Provides reasonable defaults while allowing LLM flexibility.
    """
    strategy = TASK_STRATEGIES.get(task_type)
    if not strategy:
        return "follow_up"

    # Get allowed actions for current level
    allowed = strategy.difficulty_progression.get(difficulty_level, ["follow_up"])
    if "all" in allowed:
        allowed = list(ACTION_DEFINITIONS.keys())

    # Find current phase
    current_phase_config = None
    for phase in strategy.phases:
        if phase["name"] == current_phase:
            current_phase_config = phase
            break

    if current_phase_config:
        # Filter to phase-appropriate actions
        phase_actions = current_phase_config["actions"]
        allowed = [a for a in allowed if a in phase_actions]

    if not allowed:
        allowed = ["follow_up"]

    # Avoid repeating same action too many times
    if len(recent_actions) >= 2 and recent_actions[-1] == recent_actions[-2]:
        allowed = [a for a in allowed if a != recent_actions[-1]] or allowed

    # If model is doing well, increase difficulty
    if model_performance > 0.8 and difficulty_level < 4:
        harder_actions = ["mislead", "mislead_subtle", "memory_injection", "logic_skip"]
        allowed = [a for a in allowed if a in harder_actions] or allowed

    # If model is struggling, ease up
    if model_performance < 0.4:
        easier_actions = ["guidance", "follow_up", "fine_grained"]
        allowed = [a for a in allowed if a in easier_actions] or allowed

    import random
    return random.choice(allowed)
