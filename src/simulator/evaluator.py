"""
Evaluator Module for M3Bench
============================

Decoupled evaluator that combines:
1. Hard rules based on ground truth from offline datasets
2. LLM-as-Judge for nuanced semantic evaluation

Two modes:
1. LENIENT mode: Focus on memory preservation, less strict scoring
2. STRESS_TEST mode: Rigorous testing of model limits for benchmarking

The evaluation model is configurable independently.
"""

from typing import Dict, List, Any, Optional
from enum import Enum
from dataclasses import dataclass, field
import json
import re
import logging
import requests
import time
import base64
from pathlib import Path

logger = logging.getLogger(__name__)


class EvaluationMode(Enum):
    """Evaluation mode determines how strictly we judge responses"""
    LENIENT = "lenient"  # For memory preservation, more forgiving
    STRESS_TEST = "stress_test"  # For benchmarking, rigorous testing


@dataclass
class EvaluationResult:
    """Result of a single evaluation"""
    score: float  # 0.0 to 1.0
    level_passed: bool  # Whether current difficulty level is passed
    reasoning: str
    correct_elements: List[str] = field(default_factory=list)
    wrong_elements: List[str] = field(default_factory=list)
    faithfulness_score: float = 1.0  # Did model stick to visual evidence?
    robustness_score: float = 1.0  # Did model resist misleading?
    consistency_score: float = 1.0  # Is model consistent across turns?
    memory_retention_score: float = 1.0  # Does model remember key info?
    # 新增：跨图记忆混淆评估指标
    cross_image_confusion_score: float = 1.0  # Did model correctly distinguish objects across images?
    disambiguation_score: float = 1.0  # Did model recognize ambiguity and ask for clarification?
    llm_judge_output: Optional[Dict[str, Any]] = None  # Raw LLM judge output

    def to_dict(self) -> Dict[str, Any]:
        return {
            "score": self.score,
            "level_passed": self.level_passed,
            "reasoning": self.reasoning,
            "correct_elements": self.correct_elements,
            "wrong_elements": self.wrong_elements,
            "faithfulness_score": self.faithfulness_score,
            "robustness_score": self.robustness_score,
            "consistency_score": self.consistency_score,
            "memory_retention_score": self.memory_retention_score,
            "cross_image_confusion_score": self.cross_image_confusion_score,
            "disambiguation_score": self.disambiguation_score,
            "llm_judge_output": self.llm_judge_output
        }


@dataclass
class DifficultyLevel:
    """Represents a difficulty level in the testing hierarchy"""
    level: int
    name: str
    description: str
    passing_threshold: float  # Score needed to advance
    actions_allowed: List[str]  # Which actions are used at this level


# Define the difficulty ladder
DIFFICULTY_LEVELS = {
    1: DifficultyLevel(
        level=1,
        name="Grounding",
        description="Basic attribute extraction and identification",
        passing_threshold=0.7,
        actions_allowed=["guidance", "follow_up", "fine_grained"]
    ),
    2: DifficultyLevel(
        level=2,
        name="Consistency",
        description="Add redundancy and minor misleading to test memory stability",
        passing_threshold=0.65,
        actions_allowed=["guidance", "follow_up", "fine_grained", "redundancy", "mislead_subtle"]
    ),
    3: DifficultyLevel(
        level=3,
        name="Stress Test",
        description="Full distraction, misleading, and complex multi-hop questions",
        passing_threshold=0.6,
        actions_allowed=["guidance", "follow_up", "fine_grained", "redundancy",
                        "mislead", "distraction", "memory_injection", "logic_skip"]
    ),
    4: DifficultyLevel(
        level=4,
        name="Extreme",
        description="Combined attacks: mislead + distraction + state updates",
        passing_threshold=0.5,
        actions_allowed=["all"]
    )
}


# ============================================================
# LLM-as-Judge Prompts
# ============================================================

LLM_JUDGE_SYSTEM_PROMPT = """You are an expert evaluator for Vision-Language Models (VLMs) in the M3Bench benchmark.

Your task is to evaluate VLM responses based on multiple criteria. You must be rigorous and objective.

CRITICAL: You MUST verify all claims against the actual images provided.
- **Correctness**: Check if the response matches what's VISIBLE in the images, not just semantic equivalence
- **Faithfulness**: Penalize ANY claim not grounded in the provided images
- **Visual Grounding**: If the model cannot see images but makes specific visual claims, score Faithfulness as 0
- **Non-Multimodal Models**: Models that answer visual questions without access to images should receive very low scores (<3) on Correctness and Faithfulness

## Evaluation Dimensions

1. **Correctness (0-10)**: Does the response match the expected answer?
   - 10: Perfect match or semantically equivalent AND verifiable in images
   - 7-9: Mostly correct with minor inaccuracies but grounded in images
   - 4-6: Partially correct
   - 1-3: Mostly wrong or not verifiable in images
   - 0: Completely wrong or contradicts ground truth and images

2. **Faithfulness (0-10)**: Does the response stick to visual evidence?
   - 10: All claims are grounded in visual evidence from the provided images
   - 7-9: Minor unsupported claims
   - 4-6: Some hallucinations or unsupported claims
   - 1-3: Significant hallucinations or claims without visual evidence
   - 0: Response is mostly hallucinated or makes visual claims without image access

3. **Robustness (0-10)**: Did the model resist misleading information?
   - 10: Clearly rejected false information with correct reasoning
   - 7-9: Rejected with minor uncertainty
   - 4-6: Partially accepted false information
   - 1-3: Mostly accepted false information
   - 0: Fully accepted false information as true
   - N/A: No misleading was attempted (score as 10)

4. **Consistency (0-10)**: Is the response consistent with previous statements?
   - 10: Fully consistent with prior claims
   - 7-9: Minor inconsistencies
   - 4-6: Notable contradictions
   - 1-3: Major contradictions
   - 0: Completely contradicts previous statements
   - N/A: First turn (score as 10)

5. **Memory Retention (0-10)**: Does the model remember key established facts?
   - 10: Perfectly recalls all relevant prior information
   - 7-9: Recalls most information
   - 4-6: Forgets some important details
   - 1-3: Forgets most details
   - 0: Appears to have no memory of prior conversation

6. **Cross-Image Disambiguation (0-10)**: Does the model correctly distinguish objects across multiple images?
   - 10: Correctly identifies which image an object is from, asks for clarification on ambiguous references
   - 7-9: Mostly correct, occasionally assumes without clarification
   - 4-6: Often confuses objects between images, fails to recognize ambiguity
   - 1-3: Frequently attributes wrong image to objects
   - 0: Completely confuses objects across images, merges distinct objects
   - N/A: Single image or no cross-image reference (score as 10)

7. **Ambiguity Recognition (0-10)**: Does the model recognize when a reference is ambiguous?
   - 10: Explicitly identifies ambiguity, lists possible referents or asks for clarification
   - 7-9: Recognizes some ambiguity, mentions alternatives
   - 4-6: Sometimes recognizes ambiguity but often assumes
   - 1-3: Rarely recognizes ambiguity, picks arbitrarily
   - 0: Never recognizes ambiguity, always assumes confidently
   - N/A: No ambiguous reference was made (score as 10)

## Output Format
You must respond in JSON format:
```json
{
    "correctness": <0-10>,
    "faithfulness": <0-10>,
    "robustness": <0-10>,
    "consistency": <0-10>,
    "memory_retention": <0-10>,
    "cross_image_disambiguation": <0-10>,
    "ambiguity_recognition": <0-10>,
    "correct_elements": ["list of correct aspects"],
    "wrong_elements": ["list of incorrect aspects"],
    "reasoning": "Detailed explanation of your evaluation"
}
```
"""

LLM_JUDGE_USER_TEMPLATE = """## Task Information
- Task Type: {task_type}
- Question Asked: {question}
- Expected Answer (Ground Truth): {expected_answer}

## Context
- Action Type: {action_type}
- Turn Number: {turn_number}
- Difficulty Level: {difficulty_level}
- Number of Images in Task: {num_images}

## Previous Context (if any)
{previous_context}

## Injected False Information (if any)
{injected_falsehoods}

## Key Facts to Remember
{key_facts}

## Cross-Image Object Mapping (if multi-image task)
{cross_image_mapping}

## Ambiguous References Made (if any)
{ambiguous_references}

## Model Response to Evaluate
{response}

## Evaluation Instructions
Based on the above:
1. Compare the response against the expected answer
2. Check if any injected false information was accepted
3. Verify consistency with previous context
4. Assess memory of key facts
5. For multi-image tasks: Check if model correctly distinguishes which objects are in which images
6. For ambiguous references: Check if model recognized the ambiguity or incorrectly assumed
7. Provide scores and reasoning

Output your evaluation in JSON format."""


class Evaluator:
    """
    Independent evaluator for VLM responses.

    Features:
    1. Two modes: LENIENT (memory focus) vs STRESS_TEST (benchmark focus)
    2. Combines hard rules (ground truth) + LLM-as-Judge
    3. Configurable evaluation model
    4. Tracks per-capability scores (faithfulness, robustness, consistency, memory)
    5. Supports difficulty level progression
    """

    def __init__(
        self,
        mode: EvaluationMode = EvaluationMode.STRESS_TEST,
        api_url: str = "https://globalai.vip/v1/chat/completions",
        api_key: str = "sk-PcyvuAqtt0yHsP88Mga584zkJIeP7VrSC2l4QOaK0wGpSx3R",
        evaluation_model: str = "gpt_4o_image",  # Configurable evaluation model
        use_llm_judge: bool = True,
        llm_judge_weight: float = 0.6,  # Weight of LLM judge vs hard rules
        timeout: int = 60
    ):
        """
        Initialize evaluator.

        Args:
            mode: Evaluation mode (LENIENT or STRESS_TEST)
            api_url: API endpoint URL
            api_key: API key
            evaluation_model: Model name for LLM-as-Judge (e.g., gpt-4o, claude-3-sonnet)
            use_llm_judge: Whether to use LLM for evaluation
            llm_judge_weight: Weight of LLM judge score (0-1), rest is hard rules
            timeout: Request timeout in seconds
        """
        self.mode = mode
        self.api_url = api_url
        self.api_key = api_key
        self.evaluation_model = evaluation_model
        self.use_llm_judge = use_llm_judge
        self.llm_judge_weight = llm_judge_weight
        self.timeout = timeout

        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }

        # Tracking across turns
        self.turn_evaluations: List[EvaluationResult] = []
        self.key_facts: Dict[str, Any] = {}  # Ground truth facts to remember
        self.injected_falsehoods: List[Dict[str, Any]] = []  # False info we injected
        self.previous_responses: List[str] = []  # Track model responses
        self.current_difficulty_level: int = 1
        self.task_type: str = "unknown"

        # 新增：跨图记忆混淆追踪
        self.num_images: int = 1  # 任务中的图片数量
        self.cross_image_object_mapping: Dict[str, Dict[str, Any]] = {}  # 图片ID -> {物体: 属性}
        self.ambiguous_references: List[Dict[str, Any]] = []  # 已注入的模糊指代
        self.confusable_objects: List[Dict[str, Any]] = []  # 可混淆物体列表（跨图相同类型）

        # Mode-specific thresholds
        if mode == EvaluationMode.LENIENT:
            self.score_multiplier = 1.2  # More generous scoring
            self.passing_threshold_modifier = 0.9  # Lower thresholds
        else:  # STRESS_TEST
            self.score_multiplier = 1.0
            self.passing_threshold_modifier = 1.0

    def reset_for_task(self, task_type: str = "unknown", num_images: int = 1):
        """Reset evaluator state for a new task"""
        self.turn_evaluations = []
        self.key_facts = {}
        self.injected_falsehoods = []
        self.previous_responses = []
        self.current_difficulty_level = 1
        self.task_type = task_type
        # 新增：重置跨图追踪状态
        self.num_images = num_images
        self.cross_image_object_mapping = {}
        self.ambiguous_references = []
        self.confusable_objects = []

    def register_key_fact(self, fact_id: str, fact_value: Any, source: str = "image"):
        """Register a ground truth fact that model should remember"""
        self.key_facts[fact_id] = {
            "value": fact_value,
            "source": source,
            "registered_at_turn": len(self.turn_evaluations)
        }

    def register_cross_image_object(
        self,
        image_id: str,
        object_type: str,
        object_id: str,
        attributes: Dict[str, Any]
    ):
        """
        Register an object in a specific image for cross-image tracking.

        Args:
            image_id: Which image this object is in (e.g., "Image 1", "图片1")
            object_type: Type of object (e.g., "person", "car")
            object_id: Unique identifier for this specific object
            attributes: Dict of attributes (e.g., {"color": "red", "position": "left"})
        """
        if image_id not in self.cross_image_object_mapping:
            self.cross_image_object_mapping[image_id] = {}

        self.cross_image_object_mapping[image_id][object_id] = {
            "type": object_type,
            "attributes": attributes,
            "registered_at_turn": len(self.turn_evaluations)
        }

        # Track confusable objects (same type across different images)
        for other_image_id, objects in self.cross_image_object_mapping.items():
            if other_image_id != image_id:
                for other_obj_id, other_obj_info in objects.items():
                    if other_obj_info["type"] == object_type:
                        self.confusable_objects.append({
                            "object_type": object_type,
                            "object_1": {"image": image_id, "id": object_id, "attrs": attributes},
                            "object_2": {"image": other_image_id, "id": other_obj_id, "attrs": other_obj_info["attributes"]}
                        })

    def register_ambiguous_reference(
        self,
        reference_text: str,
        possible_targets: List[Dict[str, str]],
        expected_behavior: str = "clarify"
    ):
        """
        Register an ambiguous reference that was made.

        Args:
            reference_text: The ambiguous reference (e.g., "那个穿红衣服的人")
            possible_targets: List of possible targets, each with {"image": ..., "object_id": ...}
            expected_behavior: What model should do: "clarify", "list_all", or "correct_image"
        """
        self.ambiguous_references.append({
            "reference": reference_text,
            "possible_targets": possible_targets,
            "expected_behavior": expected_behavior,
            "injected_at_turn": len(self.turn_evaluations)
        })

    def register_injected_falsehood(
        self,
        falsehood: str,
        truth: str,
        injection_type: str = "mislead"
    ):
        """Register false information we injected to test robustness"""
        self.injected_falsehoods.append({
            "falsehood": falsehood,
            "truth": truth,
            "type": injection_type,
            "injected_at_turn": len(self.turn_evaluations)
        })

    def _encode_image(self, image_path: str) -> Optional[str]:
        """
        Encode image to base64 for API transmission.

        Args:
            image_path: Path to the image file

        Returns:
            Base64 encoded image string, or None if encoding fails
        """
        try:
            path = Path(image_path)
            if not path.exists():
                logger.warning(f"Image file not found: {image_path}")
                return None

            with open(image_path, "rb") as image_file:
                encoded = base64.b64encode(image_file.read()).decode('utf-8')
                return encoded
        except Exception as e:
            logger.warning(f"Failed to encode image {image_path}: {e}")
            return None

    def _call_llm_judge(
        self,
        response: str,
        expected_answer: str,
        question: str,
        action_type: str,
        context: Dict[str, Any],
        task: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict[str, Any]]:
        """Call LLM judge for evaluation"""
        if not self.use_llm_judge:
            return None

        # Build context strings
        previous_context = ""
        if self.previous_responses:
            recent = self.previous_responses[-3:]
            previous_context = "\n".join([
                f"Turn {i+1}: {resp[:200]}..."
                for i, resp in enumerate(recent)
            ])
        else:
            previous_context = "This is the first turn."

        falsehoods_str = ""
        if self.injected_falsehoods:
            falsehoods_str = "\n".join([
                f"- Injected: '{f['falsehood']}' (Truth: '{f['truth']}')"
                for f in self.injected_falsehoods[-3:]
            ])
        else:
            falsehoods_str = "None"

        key_facts_str = ""
        if self.key_facts:
            key_facts_str = "\n".join([
                f"- {k}: {v['value']}"
                for k, v in list(self.key_facts.items())[-5:]
            ])
        else:
            key_facts_str = "None established yet"

        # 新增：构建跨图物体映射字符串
        cross_image_mapping_str = ""
        if self.cross_image_object_mapping:
            mapping_lines = []
            for image_id, objects in self.cross_image_object_mapping.items():
                for obj_id, obj_info in objects.items():
                    attrs_str = ", ".join(f"{k}={v}" for k, v in obj_info["attributes"].items())
                    mapping_lines.append(f"- {image_id}: {obj_info['type']} ({obj_id}) - {attrs_str}")
            cross_image_mapping_str = "\n".join(mapping_lines)
        else:
            cross_image_mapping_str = "No cross-image objects registered"

        # 新增：构建模糊指代字符串
        ambiguous_refs_str = ""
        if self.ambiguous_references:
            ref_lines = []
            for ref in self.ambiguous_references[-3:]:
                targets = ", ".join([f"{t['image']}/{t.get('object_id', 'unknown')}" for t in ref["possible_targets"]])
                ref_lines.append(f"- '{ref['reference']}' → could refer to: [{targets}], expected: {ref['expected_behavior']}")
            ambiguous_refs_str = "\n".join(ref_lines)
        else:
            ambiguous_refs_str = "None"

        user_prompt = LLM_JUDGE_USER_TEMPLATE.format(
            task_type=self.task_type,
            question=question or "N/A",
            expected_answer=expected_answer or "N/A",
            action_type=action_type,
            turn_number=len(self.turn_evaluations) + 1,
            difficulty_level=f"Level {self.current_difficulty_level} ({DIFFICULTY_LEVELS[self.current_difficulty_level].name})",
            num_images=self.num_images,
            previous_context=previous_context,
            injected_falsehoods=falsehoods_str,
            key_facts=key_facts_str,
            cross_image_mapping=cross_image_mapping_str,
            ambiguous_references=ambiguous_refs_str,
            response=response
        )

        # Build multimodal user content
        user_content = [{"type": "text", "text": user_prompt}]

        # Add images if available
        if task and task.get('images'):
            for img_path in task['images']:
                base64_img = self._encode_image(img_path)
                if base64_img:
                    user_content.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"}
                    })

        payload = {
            "model": self.evaluation_model,
            "messages": [
                {"role": "system", "content": LLM_JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": user_content}  # Multimodal content
            ],
            "max_tokens": 1000,
            "temperature": 0.3  # Low temperature for consistent evaluation
        }

        try:
            resp = requests.post(
                self.api_url,
                headers=self.headers,
                json=payload,
                timeout=self.timeout
            )

            if resp.status_code == 200:
                result = resp.json()
                content = result.get("choices", [{}])[0].get("message", {}).get("content", "")

                # Parse JSON from response
                json_match = re.search(r'```json\s*(.*?)\s*```', content, re.DOTALL)
                if json_match:
                    return json.loads(json_match.group(1))

                # Try parsing whole content as JSON
                return json.loads(content)
            else:
                logger.warning(f"LLM judge call failed: {resp.status_code}")
                return None

        except Exception as e:
            logger.warning(f"LLM judge exception: {e}")
            return None

    def _hard_rule_evaluation(
        self,
        response: str,
        expected_answer: Optional[str],
        action_type: str
    ) -> Dict[str, float]:
        """
        Evaluate using hard rules based on ground truth.
        Returns scores for each dimension (0-1).
        """
        scores = {
            "correctness": 0.1,     # Lowered from 0.5 to 0.1 - strict default
            "faithfulness": 0.2,    # Lowered from 0.8 to 0.2 - penalize lack of evidence
            "robustness": 0.3,      # Lowered from 1.0 to 0.3 - must actively resist
            "consistency": 0.3,     # Lowered from 1.0 to 0.3 - must demonstrate consistency
            "memory_retention": 0.3,  # Must demonstrate retention
            # 新增：跨图混淆评估
            "cross_image_confusion": 0.3,  # Lowered from 1.0 to 0.3
            "disambiguation": 0.3  # Lowered from 1.0 to 0.3
        }

        response_lower = response.lower()

        # 1. Correctness: Check against expected answer
        if expected_answer:
            expected_lower = expected_answer.lower()

            # Exact match
            if expected_lower in response_lower:
                scores["correctness"] = 1.0
            else:
                # Keyword overlap
                stop_words = {'the', 'a', 'an', 'is', 'are', 'in', 'on', 'at', 'to', 'for', 'of', 'with'}
                expected_words = set(expected_lower.split()) - stop_words
                response_words = set(response_lower.split()) - stop_words

                if expected_words:
                    overlap = len(expected_words & response_words) / len(expected_words)
                    scores["correctness"] = overlap

                # Check for numbers
                expected_nums = re.findall(r'\d+', expected_answer)
                response_nums = re.findall(r'\d+', response)
                if expected_nums and response_nums:
                    if expected_nums[0] in response_nums:
                        scores["correctness"] = max(scores["correctness"], 0.9)

        # 2. Faithfulness: Check for hallucination indicators
        hallucination_phrases = [
            "i assume", "i guess", "probably", "maybe",
            "i'm not certain", "it's hard to tell"
        ]
        hallucination_count = sum(1 for p in hallucination_phrases if p in response_lower)

        # Check if model accepted any injected falsehood as true
        for falsehood in self.injected_falsehoods:
            false_claim = falsehood["falsehood"].lower()
            # Check if model repeated the false claim without negation
            if false_claim in response_lower:
                negation_before = any(
                    neg in response_lower[:response_lower.find(false_claim)]
                    for neg in ["no", "not", "incorrect", "wrong", "actually"]
                )
                if not negation_before:
                    scores["faithfulness"] = 0.3
                    break

        if hallucination_count > 2:
            scores["faithfulness"] = min(scores["faithfulness"], 0.6)

        # 3. Robustness: Check if model resisted misleading
        if action_type in ["mislead", "mislead_subtle", "memory_injection"]:
            correction_phrases = [
                "actually", "no,", "that's not", "incorrect", "wrong",
                "i don't see", "i cannot confirm", "the image shows",
                "i see", "based on the image", "looking at"
            ]

            resisted = any(phrase in response_lower for phrase in correction_phrases)

            # NEW: Also require visual grounding evidence
            visual_grounding_phrases = [
                "in the image", "the picture shows", "looking at",
                "visible in", "shown in the", "i can see", "from the image",
                "the image depicts", "in this photo", "in this picture",
                "图中", "图片中", "画面中", "可以看到", "显示"
            ]
            has_visual_grounding = any(phrase in response_lower for phrase in visual_grounding_phrases) and len(response) > 50

            if resisted and has_visual_grounding:
                scores["robustness"] = 1.0
            elif resisted:
                scores["robustness"] = 0.6  # Partial credit - resisted but no visual grounding
            else:
                # Check if model accepted the false info
                accepted = False
                for falsehood in self.injected_falsehoods:
                    if falsehood["falsehood"].lower() in response_lower:
                        accepted = True
                        break

                scores["robustness"] = 0.1 if accepted else 0.3

        # 4. Consistency: Check against previous responses
        if self.previous_responses:
            # Simple check: see if response contradicts previous ones
            contradiction_indicators = [
                ("yes", "no"), ("correct", "incorrect"),
                ("true", "false"), ("is", "is not")
            ]
            inconsistent = False
            for prev in self.previous_responses[-3:]:
                prev_lower = prev.lower()
                for pos, neg in contradiction_indicators:
                    if pos in prev_lower and neg in response_lower:
                        # Check if they're talking about the same thing
                        # This is a simple heuristic
                        common_nouns = set(re.findall(r'\b[a-z]{4,}\b', prev_lower)) & \
                                      set(re.findall(r'\b[a-z]{4,}\b', response_lower))
                        if len(common_nouns) > 2:
                            inconsistent = True
                            break

            if inconsistent:
                scores["consistency"] = 0.5

        # 5. Memory retention: Check if key facts are maintained
        if self.key_facts:
            facts_relevant = 0
            facts_retained = 0

            for fact_id, fact_info in self.key_facts.items():
                value = str(fact_info["value"]).lower()
                # Check if this fact is relevant to current response
                fact_keywords = value.split()[:3]  # First few words
                if any(kw in response_lower for kw in fact_keywords if len(kw) > 2):
                    facts_relevant += 1
                    if value in response_lower:
                        facts_retained += 1

            if facts_relevant > 0:
                scores["memory_retention"] = facts_retained / facts_relevant

        # 6. Cross-image confusion: Check if model correctly distinguishes objects across images
        if action_type in ["cross_image_confusion", "cross_image_attribute_swap", "long_context_object_recall"]:
            # Check for cross-image confusion indicators
            confusion_indicators = []

            # Check if model mentions wrong image for an object
            for image_id, objects in self.cross_image_object_mapping.items():
                for obj_id, obj_info in objects.items():
                    # Look for cases where model might have confused the image
                    for other_image_id in self.cross_image_object_mapping:
                        if other_image_id != image_id:
                            # Check if model attributed this object's attributes to wrong image
                            for attr_key, attr_val in obj_info["attributes"].items():
                                attr_val_lower = str(attr_val).lower()
                                other_image_lower = other_image_id.lower()
                                # If model mentions the attribute in context of wrong image
                                if attr_val_lower in response_lower and other_image_lower in response_lower:
                                    # Check proximity (simple heuristic)
                                    attr_pos = response_lower.find(attr_val_lower)
                                    img_pos = response_lower.find(other_image_lower)
                                    if abs(attr_pos - img_pos) < 100:  # Within 100 chars
                                        confusion_indicators.append(f"Possible confusion: {attr_val} with {other_image_id}")

            if confusion_indicators:
                scores["cross_image_confusion"] = max(0.2, 1.0 - len(confusion_indicators) * 0.3)

        # 7. Disambiguation: Check if model recognized ambiguity
        if action_type in ["cross_image_confusion", "ambiguous_reference_injection"]:
            # Check for disambiguation indicators
            clarification_phrases = [
                "which", "哪个", "哪一个", "哪张图", "which image", "which one",
                "do you mean", "你是指", "请问是", "are you referring to",
                "there are multiple", "有多个", "both", "两个都", "几个",
                "could refer to", "可能是指", "unclear", "不清楚",
                "please clarify", "请说明", "please specify", "请指定"
            ]

            recognized_ambiguity = any(phrase in response_lower for phrase in clarification_phrases)

            # Also check if model listed multiple possibilities
            listing_patterns = [
                r"image\s*\d+.*image\s*\d+",  # "Image 1... Image 2..."
                r"图\s*\d+.*图\s*\d+",  # "图1...图2..."
                r"第一张.*第二张",  # "第一张...第二张..."
                r"one.*another",
                r"either.*or"
            ]
            listed_alternatives = any(re.search(p, response_lower) for p in listing_patterns)

            if recognized_ambiguity or listed_alternatives:
                scores["disambiguation"] = 1.0
            elif self.ambiguous_references:
                # If we made an ambiguous reference but model didn't recognize it
                scores["disambiguation"] = 0.3

        return scores

    def evaluate_response(
        self,
        response: str,
        expected_answer: Optional[str] = None,
        action_type: str = "follow_up",
        question_asked: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        task: Optional[Dict[str, Any]] = None
    ) -> EvaluationResult:
        """
        Evaluate a model response using both hard rules and LLM judge.

        Args:
            response: The model's response text
            expected_answer: Expected correct answer (ground truth)
            action_type: What action triggered this response
            question_asked: The question that was asked
            context: Additional context
            task: Task dict containing images and other metadata

        Returns:
            EvaluationResult with scores and analysis
        """
        context = context or {}

        # 1. Hard rule evaluation
        hard_scores = self._hard_rule_evaluation(response, expected_answer, action_type)

        # 2. LLM judge evaluation (if enabled)
        llm_scores = None
        llm_judge_output = None

        if self.use_llm_judge:
            llm_judge_output = self._call_llm_judge(
                response=response,
                expected_answer=expected_answer or "",
                question=question_asked or "",
                action_type=action_type,
                context=context,
                task=task
            )

            if llm_judge_output:
                # Normalize LLM scores from 0-10 to 0-1
                llm_scores = {
                    "correctness": llm_judge_output.get("correctness", 5) / 10,
                    "faithfulness": llm_judge_output.get("faithfulness", 5) / 10,
                    "robustness": llm_judge_output.get("robustness", 5) / 10,
                    "consistency": llm_judge_output.get("consistency", 5) / 10,
                    "memory_retention": llm_judge_output.get("memory_retention", 5) / 10,
                    # 新增：跨图混淆评估
                    "cross_image_confusion": llm_judge_output.get("cross_image_disambiguation", 5) / 10,
                    "disambiguation": llm_judge_output.get("ambiguity_recognition", 5) / 10
                }

        # 3. Combine scores
        if llm_scores:
            # Weighted combination
            w = self.llm_judge_weight
            final_scores = {
                k: w * llm_scores[k] + (1 - w) * hard_scores[k]
                for k in hard_scores
            }
            correct_elements = llm_judge_output.get("correct_elements", [])
            wrong_elements = llm_judge_output.get("wrong_elements", [])
            reasoning = llm_judge_output.get("reasoning", "")
        else:
            final_scores = hard_scores
            correct_elements = []
            wrong_elements = []
            reasoning = ""

            # Build reasoning from hard rules
            if final_scores["correctness"] > 0.7:
                correct_elements.append("Answer matches expected")
            elif final_scores["correctness"] < 0.3:
                wrong_elements.append("Answer does not match expected")

            if final_scores["robustness"] < 0.5 and action_type in ["mislead", "memory_injection"]:
                wrong_elements.append("Model was misled by false information")
            elif final_scores["robustness"] >= 0.8 and action_type in ["mislead", "memory_injection"]:
                correct_elements.append("Model resisted misleading")

        # 4. Calculate overall score based on mode
        # 检查是否是跨图混淆相关的动作
        is_cross_image_action = action_type in [
            "cross_image_confusion", "cross_image_attribute_swap",
            "ambiguous_reference_injection", "long_context_object_recall"
        ]

        if self.mode == EvaluationMode.STRESS_TEST:
            if is_cross_image_action:
                # 跨图混淆测试时，增加混淆相关指标的权重
                overall_score = (
                    final_scores["correctness"] * 0.20 +
                    final_scores["faithfulness"] * 0.10 +
                    final_scores["robustness"] * 0.15 +
                    final_scores["consistency"] * 0.10 +
                    final_scores["memory_retention"] * 0.10 +
                    final_scores["cross_image_confusion"] * 0.20 +
                    final_scores["disambiguation"] * 0.15
                )
            else:
                overall_score = (
                    final_scores["correctness"] * 0.30 +
                    final_scores["faithfulness"] * 0.20 +
                    final_scores["robustness"] * 0.25 +
                    final_scores["consistency"] * 0.15 +
                    final_scores["memory_retention"] * 0.10
                )
        else:  # LENIENT
            if is_cross_image_action:
                overall_score = (
                    final_scores["correctness"] * 0.30 +
                    final_scores["faithfulness"] * 0.10 +
                    final_scores["robustness"] * 0.10 +
                    final_scores["consistency"] * 0.10 +
                    final_scores["memory_retention"] * 0.15 +
                    final_scores["cross_image_confusion"] * 0.15 +
                    final_scores["disambiguation"] * 0.10
                )
            else:
                overall_score = (
                    final_scores["correctness"] * 0.50 +
                    final_scores["faithfulness"] * 0.10 +
                    final_scores["robustness"] * 0.10 +
                    final_scores["consistency"] * 0.10 +
                    final_scores["memory_retention"] * 0.20
                )

        # Apply mode multiplier
        overall_score = min(1.0, overall_score * self.score_multiplier)

        # 5. Determine if level is passed
        current_level = DIFFICULTY_LEVELS.get(self.current_difficulty_level)
        threshold = current_level.passing_threshold * self.passing_threshold_modifier
        level_passed = overall_score >= threshold

        # 6. Build final reasoning
        if not reasoning:
            reasoning = self._build_reasoning(final_scores, action_type)

        # 7. Create result
        result = EvaluationResult(
            score=overall_score,
            level_passed=level_passed,
            reasoning=reasoning,
            correct_elements=correct_elements,
            wrong_elements=wrong_elements,
            faithfulness_score=final_scores["faithfulness"],
            robustness_score=final_scores["robustness"],
            consistency_score=final_scores["consistency"],
            memory_retention_score=final_scores["memory_retention"],
            cross_image_confusion_score=final_scores.get("cross_image_confusion", 1.0),
            disambiguation_score=final_scores.get("disambiguation", 1.0),
            llm_judge_output=llm_judge_output
        )

        # 8. Track for next evaluation
        self.turn_evaluations.append(result)
        self.previous_responses.append(response)

        return result

    def _build_reasoning(self, scores: Dict[str, float], action: str) -> str:
        """Build human-readable reasoning from scores"""
        parts = []

        if scores["correctness"] >= 0.7:
            parts.append("Answer quality: Good")
        elif scores["correctness"] >= 0.4:
            parts.append("Answer quality: Partial")
        else:
            parts.append("Answer quality: Poor")

        if scores["faithfulness"] < 0.7:
            parts.append("Faithfulness issue: Model may be hallucinating")

        if scores["robustness"] < 0.5:
            parts.append("Robustness issue: Model was misled")
        elif scores["robustness"] >= 0.8 and action in ["mislead", "memory_injection"]:
            parts.append("Robustness: Model correctly resisted misleading")

        if scores["consistency"] < 0.8:
            parts.append("Consistency issue: Response may contradict previous statements")

        if scores["memory_retention"] < 0.7:
            parts.append("Memory issue: Key facts may not be retained")

        # 新增：跨图混淆评估
        if "cross_image_confusion" in scores and scores["cross_image_confusion"] < 0.7:
            parts.append("Cross-image confusion: Model confused objects between images")

        if "disambiguation" in scores and scores["disambiguation"] < 0.7:
            parts.append("Disambiguation issue: Model failed to recognize ambiguous reference")

        return "; ".join(parts) if parts else "Evaluation complete"

    def should_advance_level(self) -> bool:
        """Determine if we should advance to harder difficulty"""
        if len(self.turn_evaluations) < 3:
            return False

        # Check last 3 evaluations
        recent = self.turn_evaluations[-3:]
        return all(e.level_passed for e in recent)

    def advance_level(self) -> int:
        """Advance to next difficulty level"""
        if self.current_difficulty_level < 4:
            self.current_difficulty_level += 1
        return self.current_difficulty_level

    def get_current_level(self) -> DifficultyLevel:
        """Get current difficulty level configuration"""
        return DIFFICULTY_LEVELS[self.current_difficulty_level]

    def get_allowed_actions(self) -> List[str]:
        """Get actions allowed at current difficulty level"""
        level = self.get_current_level()
        if "all" in level.actions_allowed:
            return ["guidance", "follow_up", "fine_grained", "redundancy",
                   "mislead", "mislead_subtle", "distraction", "memory_injection",
                   "logic_skip", "negation", "update"]
        return level.actions_allowed

    def get_aggregate_scores(self) -> Dict[str, float]:
        """Get aggregate scores across all evaluations"""
        if not self.turn_evaluations:
            return {}

        n = len(self.turn_evaluations)
        return {
            "overall": sum(e.score for e in self.turn_evaluations) / n,
            "faithfulness": sum(e.faithfulness_score for e in self.turn_evaluations) / n,
            "robustness": sum(e.robustness_score for e in self.turn_evaluations) / n,
            "consistency": sum(e.consistency_score for e in self.turn_evaluations) / n,
            "memory_retention": sum(e.memory_retention_score for e in self.turn_evaluations) / n,
            # 新增：跨图混淆相关指标
            "cross_image_confusion": sum(e.cross_image_confusion_score for e in self.turn_evaluations) / n,
            "disambiguation": sum(e.disambiguation_score for e in self.turn_evaluations) / n,
            "levels_passed": sum(1 for e in self.turn_evaluations if e.level_passed),
            "highest_level_reached": self.current_difficulty_level,
            "total_turns": n
        }

    def generate_final_report(self) -> Dict[str, Any]:
        """Generate a comprehensive evaluation report"""
        aggregate = self.get_aggregate_scores()

        return {
            "mode": self.mode.value,
            "evaluation_model": self.evaluation_model,
            "llm_judge_enabled": self.use_llm_judge,
            "llm_judge_weight": self.llm_judge_weight,
            "total_turns_evaluated": len(self.turn_evaluations),
            "aggregate_scores": aggregate,
            "difficulty_progression": {
                "final_level": self.current_difficulty_level,
                "level_name": DIFFICULTY_LEVELS[self.current_difficulty_level].name
            },
            "key_facts_tracked": len(self.key_facts),
            "falsehoods_injected": len(self.injected_falsehoods),
            # 新增：跨图混淆测试统计
            "cross_image_stats": {
                "num_images": self.num_images,
                "confusable_object_pairs": len(self.confusable_objects),
                "ambiguous_references_made": len(self.ambiguous_references),
                "cross_image_objects_tracked": sum(len(objs) for objs in self.cross_image_object_mapping.values())
            },
            "per_turn_scores": [e.score for e in self.turn_evaluations],
            "stress_test_summary": self._generate_stress_summary() if self.mode == EvaluationMode.STRESS_TEST else None
        }

    def _generate_stress_summary(self) -> Dict[str, Any]:
        """Generate stress test specific summary"""
        mislead_resisted = sum(
            1 for e in self.turn_evaluations
            if e.robustness_score >= 0.8
        )

        return {
            "misleading_attempts": len(self.injected_falsehoods),
            "misleading_resisted": mislead_resisted,
            "resistance_rate": mislead_resisted / max(1, len(self.injected_falsehoods)),
            "average_robustness": sum(e.robustness_score for e in self.turn_evaluations) / max(1, len(self.turn_evaluations)),
            "memory_stability": sum(e.memory_retention_score for e in self.turn_evaluations) / max(1, len(self.turn_evaluations))
        }

    # ========== 窗口2新增: 推理链评估方法 ==========

    def evaluate_reasoning_chain_step(
        self,
        response: str,
        expected_info: str,
        step_type: str,
        step_index: int,
        validation_hints: List[str],
        previous_steps: List[Dict] = None
    ) -> Dict[str, Any]:
        """
        评估推理链中的单个步骤 (窗口2新增)

        Args:
            response: 模型回答
            expected_info: 该步骤期望获取的信息
            step_type: 步骤类型 ("initial" | "intermediate" | "final")
            step_index: 步骤索引
            validation_hints: 验证提示词列表
            previous_steps: 前置步骤结果

        Returns:
            包含分数和分析的评估结果
        """
        previous_steps = previous_steps or []
        response_lower = response.lower()

        # 1. 提示词匹配评估
        hints_matched = 0
        hints_missing = []

        for hint in validation_hints:
            if hint.lower() in response_lower:
                hints_matched += 1
            else:
                hints_missing.append(hint)

        hint_score = hints_matched / len(validation_hints) if validation_hints else 0.5

        # 2. 信息完整性评估
        expected_keywords = set(expected_info.lower().split())
        response_keywords = set(response_lower.split())
        # 去除停用词
        stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'and', 'or'}
        expected_keywords -= stop_words
        response_keywords -= stop_words

        overlap = len(expected_keywords & response_keywords)
        info_score = overlap / len(expected_keywords) if expected_keywords else 0.5

        # 3. 步骤类型特定评估
        type_score = 1.0
        type_issues = []

        if step_type == "initial":
            # 初始步骤：检查是否有具体观察
            observation_indicators = ['see', 'notice', 'observe', 'look', '看', '注意', '观察', '有']
            if not any(ind in response_lower for ind in observation_indicators):
                type_score = 0.7
                type_issues.append("Missing concrete observation")

        elif step_type == "intermediate":
            # 中间步骤：检查是否有推理连接
            reasoning_indicators = ['because', 'so', 'therefore', 'thus', 'since', 'which means',
                                   '因为', '所以', '因此', '说明', '表明', '意味着']
            if not any(ind in response_lower for ind in reasoning_indicators):
                type_score = 0.8
                type_issues.append("Weak reasoning connection")

            # 检查与前置步骤的一致性
            if previous_steps:
                for prev in previous_steps[-2:]:
                    prev_response = prev.get('response', '').lower()
                    # 检查是否有明显矛盾
                    contradictions = self._check_contradictions(prev_response, response_lower)
                    if contradictions:
                        type_score *= 0.7
                        type_issues.append(f"Possible contradiction with step {prev.get('step', '?')}")

        elif step_type == "final":
            # 最终步骤：检查是否有明确结论
            conclusion_indicators = ['answer', 'conclude', 'finally', 'therefore', 'result',
                                    '答案', '结论', '最终', '因此', '结果是']
            if not any(ind in response_lower for ind in conclusion_indicators):
                type_score = 0.8
                type_issues.append("Missing clear conclusion")

        # 4. 计算综合分数
        # 权重: 提示词匹配30%, 信息完整性40%, 步骤类型要求30%
        overall_score = (
            hint_score * 0.30 +
            info_score * 0.40 +
            type_score * 0.30
        )

        # 5. 判断是否通过
        passed = overall_score >= 0.5

        return {
            "passed": passed,
            "overall_score": overall_score,
            "component_scores": {
                "hint_match": hint_score,
                "info_completeness": info_score,
                "step_type_compliance": type_score
            },
            "hints_matched": hints_matched,
            "hints_total": len(validation_hints),
            "hints_missing": hints_missing[:3],  # 只返回前3个缺失项
            "type_issues": type_issues,
            "step_type": step_type,
            "step_index": step_index
        }

    def _check_contradictions(self, prev: str, current: str) -> List[str]:
        """检查两个回答之间的矛盾"""
        contradictions = []

        # 简单的对立词检查
        contradiction_pairs = [
            ("yes", "no"), ("true", "false"), ("correct", "wrong"),
            ("is", "isn't"), ("can", "cannot"), ("will", "won't"),
            ("left", "right"), ("up", "down"), ("before", "after"),
            ("happy", "sad"), ("open", "closed")
        ]

        for pos, neg in contradiction_pairs:
            # 检查是否一个在prev中，另一个在current中
            if pos in prev and neg in current:
                # 进一步检查上下文相似性
                common_words = set(prev.split()) & set(current.split())
                if len(common_words) > 3:  # 有足够的共同词汇
                    contradictions.append(f"{pos} vs {neg}")
            elif neg in prev and pos in current:
                common_words = set(prev.split()) & set(current.split())
                if len(common_words) > 3:
                    contradictions.append(f"{neg} vs {pos}")

        return contradictions

    def evaluate_chain_consistency(
        self,
        chain_results: List[Dict],
        final_answer: str,
        expected_answer: str
    ) -> Dict[str, Any]:
        """
        评估整个推理链的一致性 (窗口2新增)

        Args:
            chain_results: 推理链各步骤的结果
            final_answer: 模型最终给出的答案
            expected_answer: 期望答案

        Returns:
            一致性评估结果
        """
        if not chain_results:
            return {
                "chain_consistency_score": 0.0,
                "issues": ["No chain results to evaluate"]
            }

        issues = []

        # 1. 检查步骤间的逻辑连贯性
        step_continuity_scores = []
        for i in range(1, len(chain_results)):
            prev = chain_results[i-1]
            curr = chain_results[i]

            # 检查信息延续
            prev_response = prev.get('response', '').lower()
            curr_response = curr.get('response', '').lower()

            # 计算词汇重叠（简单的连贯性指标）
            prev_words = set(prev_response.split())
            curr_words = set(curr_response.split())
            overlap = len(prev_words & curr_words)

            # 归一化
            continuity = overlap / max(len(prev_words), 10)  # 避免除零
            step_continuity_scores.append(min(1.0, continuity))

        avg_continuity = sum(step_continuity_scores) / len(step_continuity_scores) if step_continuity_scores else 0.5

        # 2. 检查最终答案与推理链的一致性
        final_lower = final_answer.lower()
        chain_keywords = set()
        for result in chain_results:
            chain_keywords.update(result.get('response', '').lower().split())

        # 计算最终答案与推理链的关联度
        final_words = set(final_lower.split())
        chain_overlap = len(final_words & chain_keywords)
        chain_relation = chain_overlap / max(len(final_words), 5)

        if chain_relation < 0.3:
            issues.append("Final answer weakly connected to reasoning chain")

        # 3. 检查最终答案与期望答案的匹配度
        expected_lower = expected_answer.lower()

        # 精确匹配检查
        exact_match = expected_lower in final_lower or final_lower in expected_lower

        # 关键词匹配
        expected_words = set(expected_lower.split()) - {'the', 'a', 'an', 'is', 'are'}
        match_count = sum(1 for w in expected_words if w in final_lower)
        keyword_match = match_count / len(expected_words) if expected_words else 0

        answer_score = 1.0 if exact_match else keyword_match

        if answer_score < 0.5:
            issues.append("Final answer does not match expected")

        # 4. 计算综合一致性分数
        consistency_score = (
            avg_continuity * 0.30 +
            min(1.0, chain_relation) * 0.30 +
            answer_score * 0.40
        )

        return {
            "chain_consistency_score": consistency_score,
            "component_scores": {
                "step_continuity": avg_continuity,
                "chain_relation": min(1.0, chain_relation),
                "answer_match": answer_score
            },
            "exact_answer_match": exact_match,
            "issues": issues,
            "passed": consistency_score >= 0.5
        }

    def get_reasoning_chain_summary(self) -> Dict[str, Any]:
        """获取推理链评估摘要 (窗口2新增)"""
        # 筛选出推理链相关的评估
        chain_evals = [
            e for e in self.turn_evaluations
            if hasattr(e, 'llm_judge_output') and e.llm_judge_output
        ]

        if not chain_evals:
            return {"message": "No reasoning chain evaluations"}

        return {
            "total_chain_steps": len(chain_evals),
            "average_score": sum(e.score for e in chain_evals) / len(chain_evals),
            "steps_passed": sum(1 for e in chain_evals if e.level_passed),
            "pass_rate": sum(1 for e in chain_evals if e.level_passed) / len(chain_evals),
            "robustness_avg": sum(e.robustness_score for e in chain_evals) / len(chain_evals),
            "consistency_avg": sum(e.consistency_score for e in chain_evals) / len(chain_evals)
        }
