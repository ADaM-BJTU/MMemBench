"""
Cross-Image Memory Confusion Test Module
=========================================

This module implements the cross-image memory confusion test for VLMs.

The test is designed to evaluate:
1. Whether VLMs can correctly distinguish objects of the same type across different images
2. Whether VLMs recognize ambiguous references and ask for clarification
3. Whether VLMs maintain accurate cross-image object mapping in long conversations

Key Metrics:
1. Confusion Rate: How often the model confuses objects between images
2. Disambiguation Score: How well the model recognizes and handles ambiguous references
3. Long-Context Retention: Accuracy of cross-image memory after many filler turns

Evaluation Criteria:
1. "Confusable Object Count" - Number of same-type objects across images
2. "Conversation Length" - Number of turns before testing recall
"""

import random
import logging
import json
import os
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
from datetime import datetime

from .prompt_templates import PromptTemplates
from .action_space import ACTION_DEFINITIONS
from .evaluator import Evaluator, EvaluationMode, EvaluationResult
from .context_padder import ContextPadder, FillerConfig
from .memory_store import MemoryStore

logger = logging.getLogger(__name__)


# ============================================================
# Test Logger - 结构化日志记录
# ============================================================

@dataclass
class TurnLog:
    """单轮对话的日志记录"""
    turn_index: int
    phase: str
    action_type: str
    user_message: str
    model_response: Optional[str] = None
    image_refs: List[str] = field(default_factory=list)
    target_objects: List[Dict[str, Any]] = field(default_factory=list)
    expected_answer: Optional[str] = None
    evaluation: Optional[Dict[str, Any]] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "turn_index": self.turn_index,
            "phase": self.phase,
            "action_type": self.action_type,
            "user_message": self.user_message,
            "model_response": self.model_response,
            "image_refs": self.image_refs,
            "target_objects": self.target_objects,
            "expected_answer": self.expected_answer,
            "evaluation": self.evaluation,
            "timestamp": self.timestamp,
            "metadata": self.metadata
        }


@dataclass
class TestSessionLog:
    """完整测试会话的日志"""
    session_id: str
    start_time: str
    end_time: Optional[str] = None
    config: Dict[str, Any] = field(default_factory=dict)
    images_info: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)
    confusable_pairs: List[Dict[str, Any]] = field(default_factory=list)
    turns: List[TurnLog] = field(default_factory=list)
    final_metrics: Dict[str, Any] = field(default_factory=dict)
    status: str = "running"  # running, completed, failed

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "session_id": self.session_id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "config": self.config,
            "images_info": self.images_info,
            "confusable_pairs": self.confusable_pairs,
            "turns": [t.to_dict() for t in self.turns],
            "final_metrics": self.final_metrics,
            "status": self.status
        }


class TestLogger:
    """
    测试日志管理器

    功能：
    1. 记录每轮生成的内容
    2. 记录模型响应
    3. 记录评估结果
    4. 记录图片信息
    5. 导出为JSON文件
    """

    def __init__(self, log_dir: str = "./logs/cross_image_confusion"):
        """
        初始化日志管理器

        Args:
            log_dir: 日志保存目录
        """
        self.log_dir = log_dir
        self.current_session: Optional[TestSessionLog] = None

        # 确保日志目录存在
        os.makedirs(log_dir, exist_ok=True)

    def start_session(
        self,
        config: Dict[str, Any],
        images_info: Dict[str, List[Dict[str, Any]]]
    ) -> str:
        """
        开始新的测试会话

        Args:
            config: 测试配置
            images_info: 图片信息 {image_id: [objects]}

        Returns:
            session_id: 会话ID
        """
        session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        self.current_session = TestSessionLog(
            session_id=session_id,
            start_time=datetime.now().isoformat(),
            config=config,
            images_info=images_info
        )

        logger.info(f"Started test session: {session_id}")
        return session_id

    def set_confusable_pairs(self, pairs: List[Dict[str, Any]]):
        """记录可混淆物体对"""
        if self.current_session:
            self.current_session.confusable_pairs = pairs

    def log_turn(
        self,
        turn_index: int,
        phase: str,
        action_type: str,
        user_message: str,
        image_refs: Optional[List[str]] = None,
        target_objects: Optional[List[Dict[str, Any]]] = None,
        expected_answer: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> TurnLog:
        """
        记录一轮对话的生成内容

        Args:
            turn_index: 轮次索引
            phase: 测试阶段
            action_type: 动作类型
            user_message: 生成的用户消息
            image_refs: 涉及的图片ID列表
            target_objects: 目标物体信息
            expected_answer: 期望答案
            metadata: 其他元数据

        Returns:
            TurnLog: 创建的日志记录
        """
        if self.current_session is None:
            raise RuntimeError("No active session. Call start_session() first.")

        turn_log = TurnLog(
            turn_index=turn_index,
            phase=phase,
            action_type=action_type,
            user_message=user_message,
            image_refs=image_refs or [],
            target_objects=target_objects or [],
            expected_answer=expected_answer,
            metadata=metadata or {}
        )

        self.current_session.turns.append(turn_log)

        logger.debug(
            f"Turn {turn_index} [{phase}]: {action_type} - "
            f"Images: {image_refs}, Message: {user_message[:50]}..."
        )

        return turn_log

    def update_turn_response(
        self,
        turn_index: int,
        model_response: str,
        evaluation: Optional[Dict[str, Any]] = None
    ):
        """
        更新某轮的模型响应和评估结果

        Args:
            turn_index: 轮次索引
            model_response: 模型响应
            evaluation: 评估结果
        """
        if self.current_session is None:
            return

        for turn in self.current_session.turns:
            if turn.turn_index == turn_index:
                turn.model_response = model_response
                turn.evaluation = evaluation
                break

    def end_session(
        self,
        final_metrics: Dict[str, Any],
        status: str = "completed"
    ):
        """
        结束测试会话

        Args:
            final_metrics: 最终评估指标
            status: 会话状态 (completed/failed)
        """
        if self.current_session is None:
            return

        self.current_session.end_time = datetime.now().isoformat()
        self.current_session.final_metrics = final_metrics
        self.current_session.status = status

        logger.info(
            f"Ended session {self.current_session.session_id} with status: {status}"
        )

    def save_to_json(self, filepath: Optional[str] = None) -> str:
        """
        保存日志到JSON文件

        Args:
            filepath: 文件路径，如果不指定则自动生成

        Returns:
            保存的文件路径
        """
        if self.current_session is None:
            raise RuntimeError("No session to save")

        if filepath is None:
            filepath = os.path.join(
                self.log_dir,
                f"{self.current_session.session_id}.json"
            )

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.current_session.to_dict(), f, ensure_ascii=False, indent=2)

        logger.info(f"Saved session log to: {filepath}")
        return filepath

    def get_turn_summary(self) -> List[Dict[str, Any]]:
        """获取所有轮次的摘要信息"""
        if self.current_session is None:
            return []

        return [
            {
                "turn": t.turn_index,
                "phase": t.phase,
                "action": t.action_type,
                "images": t.image_refs,
                "score": t.evaluation.get("score") if t.evaluation else None,
                "has_response": t.model_response is not None
            }
            for t in self.current_session.turns
        ]

    def get_phase_statistics(self) -> Dict[str, Dict[str, Any]]:
        """获取各阶段的统计信息"""
        if self.current_session is None:
            return {}

        phase_stats = {}
        for turn in self.current_session.turns:
            phase = turn.phase
            if phase not in phase_stats:
                phase_stats[phase] = {
                    "turn_count": 0,
                    "total_score": 0.0,
                    "scored_turns": 0,
                    "action_types": []
                }

            phase_stats[phase]["turn_count"] += 1
            if turn.action_type not in phase_stats[phase]["action_types"]:
                phase_stats[phase]["action_types"].append(turn.action_type)

            if turn.evaluation and "score" in turn.evaluation:
                phase_stats[phase]["total_score"] += turn.evaluation["score"]
                phase_stats[phase]["scored_turns"] += 1

        # 计算平均分
        for phase, stats in phase_stats.items():
            if stats["scored_turns"] > 0:
                stats["avg_score"] = stats["total_score"] / stats["scored_turns"]
            else:
                stats["avg_score"] = None

        return phase_stats

    def export_readable_log(self, filepath: Optional[str] = None) -> str:
        """
        导出可读的文本日志

        Args:
            filepath: 文件路径

        Returns:
            保存的文件路径
        """
        if self.current_session is None:
            raise RuntimeError("No session to export")

        if filepath is None:
            filepath = os.path.join(
                self.log_dir,
                f"{self.current_session.session_id}_readable.txt"
            )

        lines = []
        lines.append("=" * 80)
        lines.append(f"Cross-Image Confusion Test Log")
        lines.append(f"Session ID: {self.current_session.session_id}")
        lines.append(f"Start Time: {self.current_session.start_time}")
        lines.append(f"End Time: {self.current_session.end_time or 'N/A'}")
        lines.append(f"Status: {self.current_session.status}")
        lines.append("=" * 80)

        # 配置信息
        lines.append("\n--- Configuration ---")
        for key, value in self.current_session.config.items():
            lines.append(f"  {key}: {value}")

        # 图片信息
        lines.append("\n--- Images ---")
        for img_id, objects in self.current_session.images_info.items():
            lines.append(f"\n  {img_id}:")
            for obj in objects:
                lines.append(f"    - {obj.get('type', 'unknown')} ({obj.get('id', 'N/A')})")
                if 'attributes' in obj:
                    attrs = ", ".join(f"{k}={v}" for k, v in obj['attributes'].items())
                    lines.append(f"      Attributes: {attrs}")

        # 可混淆物体对
        lines.append(f"\n--- Confusable Object Pairs ({len(self.current_session.confusable_pairs)}) ---")
        for pair in self.current_session.confusable_pairs:
            lines.append(f"  - {pair.get('object_type', 'unknown')}: "
                        f"{pair.get('image_1_id', '?')} <-> {pair.get('image_2_id', '?')}")

        # 对话轮次
        lines.append("\n--- Conversation Turns ---")
        for turn in self.current_session.turns:
            lines.append(f"\n[Turn {turn.turn_index}] Phase: {turn.phase} | Action: {turn.action_type}")
            lines.append(f"  Images: {', '.join(turn.image_refs) if turn.image_refs else 'N/A'}")
            lines.append(f"  User: {turn.user_message}")
            if turn.model_response:
                response_preview = turn.model_response[:200] + "..." if len(turn.model_response) > 200 else turn.model_response
                lines.append(f"  Model: {response_preview}")
            if turn.expected_answer:
                lines.append(f"  Expected: {turn.expected_answer}")
            if turn.evaluation:
                lines.append(f"  Evaluation:")
                lines.append(f"    Score: {turn.evaluation.get('score', 'N/A')}")
                if 'cross_image_confusion_score' in turn.evaluation:
                    lines.append(f"    Cross-Image Confusion: {turn.evaluation['cross_image_confusion_score']}")
                if 'disambiguation_score' in turn.evaluation:
                    lines.append(f"    Disambiguation: {turn.evaluation['disambiguation_score']}")
                if 'reasoning' in turn.evaluation:
                    lines.append(f"    Reasoning: {turn.evaluation['reasoning'][:100]}...")

        # 最终指标
        lines.append("\n--- Final Metrics ---")
        for key, value in self.current_session.final_metrics.items():
            if isinstance(value, dict):
                lines.append(f"  {key}:")
                for k, v in value.items():
                    lines.append(f"    {k}: {v}")
            else:
                lines.append(f"  {key}: {value}")

        lines.append("\n" + "=" * 80)

        with open(filepath, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        logger.info(f"Saved readable log to: {filepath}")
        return filepath


class ConfusionTestPhase(Enum):
    """Phases of the cross-image confusion test"""
    GROUNDING = "grounding"  # Establish baseline facts about each image
    FILLER_INJECTION = "filler_injection"  # Inject filler to lengthen context
    AMBIGUOUS_REFERENCE = "ambiguous_reference"  # Make ambiguous references
    ATTRIBUTE_SWAP = "attribute_swap"  # Test with swapped attributes
    LONG_CONTEXT_RECALL = "long_context_recall"  # Test memory after long context


@dataclass
class ConfusableObjectPair:
    """Represents a pair of confusable objects across images"""
    object_type: str  # e.g., "person", "car", "dog"
    image_1_id: str
    image_1_object_id: str
    image_1_attributes: Dict[str, Any]
    image_2_id: str
    image_2_object_id: str
    image_2_attributes: Dict[str, Any]
    # Common attributes that could cause confusion
    common_attributes: List[str] = field(default_factory=list)
    # Distinguishing attributes
    distinguishing_attributes: Dict[str, Tuple[Any, Any]] = field(default_factory=dict)


@dataclass
class CrossImageConfusionTestConfig:
    """Configuration for cross-image confusion test"""
    # Number of images in the test
    num_images: int = 3
    # Minimum number of confusable object pairs
    min_confusable_pairs: int = 2
    # Filler configuration
    min_filler_turns: int = 5
    max_filler_turns: int = 15
    # Phases to run
    phases_to_run: List[ConfusionTestPhase] = field(default_factory=lambda: list(ConfusionTestPhase))
    # Language preference
    language: str = "cn"  # "cn" or "en"
    # Object types to focus on (for dataset balancing)
    object_types: List[str] = field(default_factory=lambda: [
        "person", "vehicle", "animal", "furniture", "food", "building", "plant"
    ])
    # Difficulty level (affects ambiguity of references)
    difficulty: int = 3  # 1-4


class CrossImageConfusionTester:
    """
    Main class for running cross-image memory confusion tests.

    This tester:
    1. Sets up confusable object pairs across multiple images
    2. Establishes baseline facts through grounding turns
    3. Injects filler content to lengthen the context
    4. Makes ambiguous references to test disambiguation
    5. Tests attribute recall with swapped attributes
    6. Evaluates long-context object memory
    7. Logs all interactions for analysis
    """

    def __init__(
        self,
        config: Optional[CrossImageConfusionTestConfig] = None,
        evaluator: Optional[Evaluator] = None,
        context_padder: Optional[ContextPadder] = None,
        memory_store: Optional[MemoryStore] = None,
        log_dir: Optional[str] = None,
        enable_logging: bool = True
    ):
        """
        Initialize the cross-image confusion tester.

        Args:
            config: Test configuration
            evaluator: Evaluator instance (creates new if None)
            context_padder: Context padder instance (creates new if None)
            memory_store: Memory store instance (creates new if None)
            log_dir: Directory for saving logs (default: ./logs/cross_image_confusion)
            enable_logging: Whether to enable structured logging
        """
        self.config = config or CrossImageConfusionTestConfig()
        self.evaluator = evaluator or Evaluator(mode=EvaluationMode.STRESS_TEST)
        self.context_padder = context_padder or ContextPadder(use_weak_model=False)
        self.memory_store = memory_store or MemoryStore()
        self.templates = PromptTemplates()

        # Logging
        self.enable_logging = enable_logging
        self.test_logger: Optional[TestLogger] = None
        if enable_logging:
            self.test_logger = TestLogger(log_dir or "./logs/cross_image_confusion")

        # State tracking
        self.confusable_pairs: List[ConfusableObjectPair] = []
        self.established_facts: Dict[str, Dict[str, Any]] = {}  # image_id -> {object_id: facts}
        self.current_phase: ConfusionTestPhase = ConfusionTestPhase.GROUNDING
        self.turn_count: int = 0
        self.filler_turns_injected: int = 0
        self.ambiguous_references_made: List[Dict[str, Any]] = []
        self.test_results: Dict[str, Any] = {}
        self.objects_per_image: Dict[str, List[Dict[str, Any]]] = {}  # Store for logging

    def setup_confusable_objects(
        self,
        objects_per_image: Dict[str, List[Dict[str, Any]]]
    ) -> List[ConfusableObjectPair]:
        """
        Identify confusable object pairs from the provided image objects.

        Args:
            objects_per_image: Dict mapping image_id to list of objects
                Each object should have: {"id": str, "type": str, "attributes": Dict}

        Returns:
            List of ConfusableObjectPair instances
        """
        self.confusable_pairs = []
        image_ids = list(objects_per_image.keys())

        # Find objects of the same type across different images
        for i, img1_id in enumerate(image_ids):
            for img2_id in image_ids[i + 1:]:
                for obj1 in objects_per_image[img1_id]:
                    for obj2 in objects_per_image[img2_id]:
                        # Check if they're the same type
                        if obj1["type"] == obj2["type"]:
                            # Find common and distinguishing attributes
                            common_attrs = []
                            distinguishing_attrs = {}

                            all_attr_keys = set(obj1["attributes"].keys()) | set(obj2["attributes"].keys())
                            for attr_key in all_attr_keys:
                                val1 = obj1["attributes"].get(attr_key)
                                val2 = obj2["attributes"].get(attr_key)
                                if val1 == val2:
                                    if val1 is not None:
                                        common_attrs.append(attr_key)
                                else:
                                    if val1 is not None and val2 is not None:
                                        distinguishing_attrs[attr_key] = (val1, val2)

                            pair = ConfusableObjectPair(
                                object_type=obj1["type"],
                                image_1_id=img1_id,
                                image_1_object_id=obj1["id"],
                                image_1_attributes=obj1["attributes"],
                                image_2_id=img2_id,
                                image_2_object_id=obj2["id"],
                                image_2_attributes=obj2["attributes"],
                                common_attributes=common_attrs,
                                distinguishing_attributes=distinguishing_attrs
                            )
                            self.confusable_pairs.append(pair)

                            # Register with evaluator
                            self.evaluator.register_cross_image_object(
                                img1_id, obj1["type"], obj1["id"], obj1["attributes"]
                            )
                            self.evaluator.register_cross_image_object(
                                img2_id, obj2["type"], obj2["id"], obj2["attributes"]
                            )

        logger.info(f"Found {len(self.confusable_pairs)} confusable object pairs")
        return self.confusable_pairs

    def generate_grounding_turn(
        self,
        image_id: str,
        object_info: Dict[str, Any]
    ) -> Dict[str, str]:
        """
        Generate a grounding turn to establish facts about an object.

        Args:
            image_id: ID of the image
            object_info: Object information {"id": str, "type": str, "attributes": Dict}

        Returns:
            Dict with "user_message" for the grounding question
        """
        obj_type = object_info["type"]
        attrs = object_info["attributes"]

        # Get object info from templates
        obj_info = self.templates.get_confusable_object_info(obj_type, self.config.language)
        synonym = random.choice(obj_info["synonyms"]) if obj_info["synonyms"] else obj_type

        if self.config.language == "cn":
            questions = [
                f"看看{image_id}，里面的{synonym}是什么样的？请详细描述它的特征。",
                f"关于{image_id}中的{synonym}，能告诉我它的颜色、大小和位置吗？",
                f"请描述{image_id}里{synonym}的所有可见特征。",
            ]
        else:
            questions = [
                f"Look at {image_id}. What does the {synonym} look like? Please describe its features in detail.",
                f"About the {synonym} in {image_id}, can you tell me its color, size, and position?",
                f"Please describe all visible features of the {synonym} in {image_id}.",
            ]

        return {
            "user_message": random.choice(questions),
            "action_type": "guidance",
            "expected_elements": list(attrs.values()),
            "image_id": image_id,
            "object_id": object_info["id"]
        }

    def generate_ambiguous_reference_turn(
        self,
        pair: ConfusableObjectPair
    ) -> Dict[str, Any]:
        """
        Generate a turn with an ambiguous reference to test disambiguation.

        Args:
            pair: A confusable object pair

        Returns:
            Dict with "user_message" and metadata
        """
        obj_type = pair.object_type
        obj_info = self.templates.get_confusable_object_info(obj_type, self.config.language)
        synonym = random.choice(obj_info["synonyms"]) if obj_info["synonyms"] else obj_type

        # Generate ambiguous reference based on difficulty
        if self.config.difficulty <= 2:
            # Just use the object type without any qualifier
            if self.config.language == "cn":
                reference = f"那个{synonym}"
            else:
                reference = f"the {synonym}"
        else:
            # Use a common attribute (if any) that both have
            if pair.common_attributes:
                common_attr = random.choice(pair.common_attributes)
                common_val = pair.image_1_attributes.get(common_attr, "")
                if self.config.language == "cn":
                    reference = f"那个{common_val}的{synonym}"
                else:
                    reference = f"the {common_val} {synonym}"
            else:
                # Use a vague descriptor
                if self.config.language == "cn":
                    vague = random.choice(["那个", "之前提到的", "图里的"])
                    reference = f"{vague}{synonym}"
                else:
                    vague = random.choice(["that", "the mentioned", "the"])
                    reference = f"{vague} {synonym}"

        # Generate question using the ambiguous reference
        if self.config.language == "cn":
            questions = [
                f"{reference}，它的位置在哪里？",
                f"你能再说说{reference}的详细特征吗？",
                f"{reference}在做什么？",
                f"关于{reference}，它的颜色是什么？",
            ]
        else:
            questions = [
                f"Where is {reference} located?",
                f"Can you describe the detailed features of {reference}?",
                f"What is {reference} doing?",
                f"About {reference}, what color is it?",
            ]

        user_message = random.choice(questions)

        # Track the ambiguous reference
        ref_info = {
            "reference": reference,
            "full_question": user_message,
            "possible_targets": [
                {"image": pair.image_1_id, "object_id": pair.image_1_object_id},
                {"image": pair.image_2_id, "object_id": pair.image_2_object_id}
            ],
            "expected_behavior": "clarify"
        }
        self.ambiguous_references_made.append(ref_info)

        # Register with evaluator
        self.evaluator.register_ambiguous_reference(
            reference,
            ref_info["possible_targets"],
            "clarify"
        )

        return {
            "user_message": user_message,
            "action_type": "cross_image_confusion",
            "ambiguous_reference": reference,
            "possible_targets": ref_info["possible_targets"]
        }

    def generate_attribute_swap_turn(
        self,
        pair: ConfusableObjectPair
    ) -> Dict[str, Any]:
        """
        Generate a turn that swaps attributes between images to test memory.

        Args:
            pair: A confusable object pair with distinguishing attributes

        Returns:
            Dict with "user_message" and metadata
        """
        if not pair.distinguishing_attributes:
            # No distinguishing attributes, skip
            return None

        # Pick a distinguishing attribute to swap
        attr_key = random.choice(list(pair.distinguishing_attributes.keys()))
        val1, val2 = pair.distinguishing_attributes[attr_key]

        obj_info = self.templates.get_confusable_object_info(pair.object_type, self.config.language)
        synonym = random.choice(obj_info["synonyms"]) if obj_info["synonyms"] else pair.object_type

        # Swap: attribute from image 1 to image 2
        if self.config.language == "cn":
            questions = [
                f"你刚才说{pair.image_2_id}里的{synonym}是{val1}的，对吧？",
                f"{pair.image_2_id}那个{val1}的{synonym}，在做什么？",
                f"关于{pair.image_2_id}的{synonym}，它的{attr_key}是{val1}没错吧？",
            ]
        else:
            questions = [
                f"You mentioned the {synonym} in {pair.image_2_id} was {val1}, right?",
                f"The {val1} {synonym} in {pair.image_2_id}, what's it doing?",
                f"About the {synonym} in {pair.image_2_id}, its {attr_key} is {val1}, correct?",
            ]

        user_message = random.choice(questions)

        # Register as injected falsehood
        self.evaluator.register_injected_falsehood(
            f"{pair.image_2_id} {synonym} is {val1}",
            f"{pair.image_2_id} {synonym} is actually {val2}",
            "cross_image_attribute_swap"
        )

        return {
            "user_message": user_message,
            "action_type": "cross_image_attribute_swap",
            "swapped_attribute": attr_key,
            "correct_value": val2,
            "swapped_value": val1,
            "target_image": pair.image_2_id,
            "expected_answer": f"{pair.image_2_id}中的{synonym}的{attr_key}是{val2}"
        }

    def generate_long_context_recall_turn(
        self,
        pair: ConfusableObjectPair
    ) -> Dict[str, Any]:
        """
        Generate a turn to test memory recall after long context.

        Args:
            pair: A confusable object pair

        Returns:
            Dict with "user_message" and metadata
        """
        obj_info = self.templates.get_confusable_object_info(pair.object_type, self.config.language)
        synonym = random.choice(obj_info["synonyms"]) if obj_info["synonyms"] else pair.object_type

        # Pick a specific attribute to ask about
        if pair.distinguishing_attributes:
            attr_key = random.choice(list(pair.distinguishing_attributes.keys()))
            expected_val = pair.distinguishing_attributes[attr_key][0]  # Value from image 1
            target_image = pair.image_1_id
        elif pair.image_1_attributes:
            attr_key = random.choice(list(pair.image_1_attributes.keys()))
            expected_val = pair.image_1_attributes[attr_key]
            target_image = pair.image_1_id
        else:
            attr_key = "特征"
            expected_val = "之前描述的"
            target_image = pair.image_1_id

        if self.config.language == "cn":
            questions = [
                f"让我们回顾一下最开始讨论的内容。{target_image}里那个{synonym}，它的{attr_key}是什么来着？",
                f"经过这么多讨论，我想确认一下：{target_image}中{synonym}的{attr_key}是什么？",
                f"在最初的几轮对话中，我们提到{target_image}有{synonym}。它的{attr_key}还是{expected_val}吗？",
            ]
        else:
            questions = [
                f"Let's go back to what we discussed at the beginning. The {synonym} in {target_image}, what was its {attr_key}?",
                f"After all this discussion, I want to confirm: what's the {attr_key} of the {synonym} in {target_image}?",
                f"In the first few turns, we noted {target_image} had a {synonym}. Is its {attr_key} still {expected_val}?",
            ]

        return {
            "user_message": random.choice(questions),
            "action_type": "long_context_object_recall",
            "target_image": target_image,
            "target_object": synonym,
            "queried_attribute": attr_key,
            "expected_answer": str(expected_val)
        }

    def inject_filler_turns(
        self,
        num_turns: int,
        image_refs: List[str],
        objects: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Generate filler turns to lengthen the context.

        Args:
            num_turns: Number of filler turns to generate
            image_refs: List of image reference strings
            objects: List of object names/types

        Returns:
            List of filler turn dicts
        """
        filler_turns = []

        for i in range(num_turns):
            context = {
                "turn_index": self.turn_count + i,
                "summary": f"Discussing images: {', '.join(image_refs)}"
            }

            filler = self.context_padder.generate_multi_image_filler_turn(
                turn_context=context,
                image_refs=image_refs,
                objects=objects
            )
            filler_turns.append(filler)

        self.filler_turns_injected += num_turns
        return filler_turns

    def run_test_sequence(
        self,
        objects_per_image: Dict[str, List[Dict[str, Any]]]
    ) -> Dict[str, Any]:
        """
        Run the complete cross-image confusion test sequence.

        Args:
            objects_per_image: Dict mapping image_id to list of objects

        Returns:
            Test results dict
        """
        # Reset state
        self.turn_count = 0
        self.filler_turns_injected = 0
        self.ambiguous_references_made = []
        self.test_results = {}
        self.objects_per_image = objects_per_image  # Store for logging

        # Setup
        image_ids = list(objects_per_image.keys())
        self.evaluator.reset_for_task(
            task_type="cross_image_confusion_test",
            num_images=len(image_ids)
        )

        # Start logging session
        if self.enable_logging and self.test_logger:
            config_dict = {
                "num_images": self.config.num_images,
                "min_confusable_pairs": self.config.min_confusable_pairs,
                "min_filler_turns": self.config.min_filler_turns,
                "max_filler_turns": self.config.max_filler_turns,
                "difficulty": self.config.difficulty,
                "language": self.config.language,
                "phases": [p.value for p in self.config.phases_to_run]
            }
            self.test_logger.start_session(config_dict, objects_per_image)

        # Find confusable pairs
        pairs = self.setup_confusable_objects(objects_per_image)
        if not pairs:
            logger.warning("No confusable object pairs found")
            if self.enable_logging and self.test_logger:
                self.test_logger.end_session({"error": "No confusable pairs"}, "failed")
            return {"error": "No confusable object pairs found", "pairs": 0}

        # Log confusable pairs
        if self.enable_logging and self.test_logger:
            pairs_dicts = [
                {
                    "object_type": p.object_type,
                    "image_1_id": p.image_1_id,
                    "image_1_object_id": p.image_1_object_id,
                    "image_2_id": p.image_2_id,
                    "image_2_object_id": p.image_2_object_id,
                    "common_attributes": p.common_attributes,
                    "distinguishing_attributes": {
                        k: list(v) for k, v in p.distinguishing_attributes.items()
                    }
                }
                for p in pairs
            ]
            self.test_logger.set_confusable_pairs(pairs_dicts)

        test_sequence = []
        all_objects = []
        for img_id, objs in objects_per_image.items():
            for obj in objs:
                all_objects.append(obj["type"])

        # Phase 1: Grounding
        if ConfusionTestPhase.GROUNDING in self.config.phases_to_run:
            self.current_phase = ConfusionTestPhase.GROUNDING
            for img_id, objs in objects_per_image.items():
                for obj in objs:
                    turn = self.generate_grounding_turn(img_id, obj)
                    test_sequence.append(turn)

                    # Log turn
                    if self.enable_logging and self.test_logger:
                        self.test_logger.log_turn(
                            turn_index=self.turn_count,
                            phase=self.current_phase.value,
                            action_type=turn.get("action_type", "guidance"),
                            user_message=turn.get("user_message", ""),
                            image_refs=[img_id],
                            target_objects=[obj],
                            expected_answer=None,
                            metadata={"expected_elements": turn.get("expected_elements", [])}
                        )

                    self.turn_count += 1

        # Phase 2: Filler Injection
        if ConfusionTestPhase.FILLER_INJECTION in self.config.phases_to_run:
            self.current_phase = ConfusionTestPhase.FILLER_INJECTION
            num_fillers = random.randint(
                self.config.min_filler_turns,
                self.config.max_filler_turns
            )
            fillers = self.inject_filler_turns(num_fillers, image_ids, list(set(all_objects)))
            for f in fillers:
                f["action_type"] = "distraction"
                test_sequence.append(f)

                # Log filler turn
                if self.enable_logging and self.test_logger:
                    self.test_logger.log_turn(
                        turn_index=self.turn_count,
                        phase=self.current_phase.value,
                        action_type="distraction",
                        user_message=f.get("user_message", ""),
                        image_refs=image_ids,
                        target_objects=[],
                        expected_answer=None,
                        metadata={"filler_type": "multi_image"}
                    )

                self.turn_count += 1

        # Phase 3: Ambiguous Reference
        if ConfusionTestPhase.AMBIGUOUS_REFERENCE in self.config.phases_to_run:
            self.current_phase = ConfusionTestPhase.AMBIGUOUS_REFERENCE
            for pair in pairs[:min(3, len(pairs))]:  # Test up to 3 pairs
                turn = self.generate_ambiguous_reference_turn(pair)
                test_sequence.append(turn)

                # Log turn
                if self.enable_logging and self.test_logger:
                    self.test_logger.log_turn(
                        turn_index=self.turn_count,
                        phase=self.current_phase.value,
                        action_type=turn.get("action_type", "cross_image_confusion"),
                        user_message=turn.get("user_message", ""),
                        image_refs=[pair.image_1_id, pair.image_2_id],
                        target_objects=turn.get("possible_targets", []),
                        expected_answer="Should ask for clarification",
                        metadata={
                            "ambiguous_reference": turn.get("ambiguous_reference", ""),
                            "object_type": pair.object_type
                        }
                    )

                self.turn_count += 1

        # Phase 4: Attribute Swap
        if ConfusionTestPhase.ATTRIBUTE_SWAP in self.config.phases_to_run:
            self.current_phase = ConfusionTestPhase.ATTRIBUTE_SWAP
            for pair in pairs[:min(2, len(pairs))]:  # Test up to 2 pairs
                turn = self.generate_attribute_swap_turn(pair)
                if turn:
                    test_sequence.append(turn)

                    # Log turn
                    if self.enable_logging and self.test_logger:
                        self.test_logger.log_turn(
                            turn_index=self.turn_count,
                            phase=self.current_phase.value,
                            action_type=turn.get("action_type", "cross_image_attribute_swap"),
                            user_message=turn.get("user_message", ""),
                            image_refs=[turn.get("target_image", "")],
                            target_objects=[{"image": turn.get("target_image"), "type": pair.object_type}],
                            expected_answer=turn.get("expected_answer", ""),
                            metadata={
                                "swapped_attribute": turn.get("swapped_attribute"),
                                "correct_value": turn.get("correct_value"),
                                "swapped_value": turn.get("swapped_value")
                            }
                        )

                    self.turn_count += 1

        # Phase 5: Long Context Recall (after more fillers)
        if ConfusionTestPhase.LONG_CONTEXT_RECALL in self.config.phases_to_run:
            self.current_phase = ConfusionTestPhase.LONG_CONTEXT_RECALL
            # Add more fillers before recall test
            more_fillers = self.inject_filler_turns(
                random.randint(3, 6),
                image_ids,
                list(set(all_objects))
            )
            for f in more_fillers:
                f["action_type"] = "distraction"
                test_sequence.append(f)

                # Log filler turn
                if self.enable_logging and self.test_logger:
                    self.test_logger.log_turn(
                        turn_index=self.turn_count,
                        phase=self.current_phase.value,
                        action_type="distraction",
                        user_message=f.get("user_message", ""),
                        image_refs=image_ids,
                        target_objects=[],
                        expected_answer=None,
                        metadata={"filler_type": "pre_recall"}
                    )

                self.turn_count += 1

            # Now test recall
            for pair in pairs[:min(2, len(pairs))]:
                turn = self.generate_long_context_recall_turn(pair)
                test_sequence.append(turn)

                # Log turn
                if self.enable_logging and self.test_logger:
                    self.test_logger.log_turn(
                        turn_index=self.turn_count,
                        phase=self.current_phase.value,
                        action_type=turn.get("action_type", "long_context_object_recall"),
                        user_message=turn.get("user_message", ""),
                        image_refs=[turn.get("target_image", "")],
                        target_objects=[{
                            "image": turn.get("target_image"),
                            "object": turn.get("target_object"),
                            "attribute": turn.get("queried_attribute")
                        }],
                        expected_answer=turn.get("expected_answer", ""),
                        metadata={}
                    )

                self.turn_count += 1

        # Compile results
        self.test_results = {
            "total_turns": self.turn_count,
            "filler_turns": self.filler_turns_injected,
            "confusable_pairs": len(pairs),
            "ambiguous_references": len(self.ambiguous_references_made),
            "test_sequence": test_sequence,
            "config": {
                "num_images": self.config.num_images,
                "min_confusable_pairs": self.config.min_confusable_pairs,
                "difficulty": self.config.difficulty,
                "language": self.config.language
            }
        }

        return self.test_results

    def evaluate_confusion_metrics(
        self,
        model_responses: List[str]
    ) -> Dict[str, Any]:
        """
        Evaluate the model's performance on cross-image confusion.

        Args:
            model_responses: List of model responses to the test sequence

        Returns:
            Evaluation metrics dict
        """
        if not self.test_results.get("test_sequence"):
            return {"error": "No test sequence to evaluate"}

        test_sequence = self.test_results["test_sequence"]

        if len(model_responses) != len(test_sequence):
            logger.warning(
                f"Response count ({len(model_responses)}) != "
                f"test sequence count ({len(test_sequence)})"
            )

        # Evaluate each response
        for i, (turn, response) in enumerate(zip(test_sequence, model_responses)):
            action_type = turn.get("action_type", "follow_up")
            expected = turn.get("expected_answer", "")

            result = self.evaluator.evaluate_response(
                response=response,
                expected_answer=expected,
                action_type=action_type,
                question_asked=turn.get("user_message", ""),
                context=turn
            )

            # Log the response and evaluation
            if self.enable_logging and self.test_logger:
                self.test_logger.update_turn_response(
                    turn_index=i,
                    model_response=response,
                    evaluation=result.to_dict()
                )

            logger.info(
                f"Turn {i + 1} ({action_type}): "
                f"score={result.score:.2f}, "
                f"confusion_score={result.cross_image_confusion_score:.2f}, "
                f"disambiguation_score={result.disambiguation_score:.2f}"
            )

        # Generate final report
        report = self.evaluator.generate_final_report()

        # Add confusion-specific metrics
        report["confusion_test_metrics"] = {
            "confusable_pairs_tested": len(self.confusable_pairs),
            "ambiguous_references_made": len(self.ambiguous_references_made),
            "filler_turns_injected": self.filler_turns_injected,
            "avg_cross_image_confusion_score": report["aggregate_scores"].get("cross_image_confusion", 0),
            "avg_disambiguation_score": report["aggregate_scores"].get("disambiguation", 0),
            "conversation_length": self.turn_count
        }

        # End logging session and save
        if self.enable_logging and self.test_logger:
            self.test_logger.end_session(report, "completed")

        return report

    def save_logs(
        self,
        json_path: Optional[str] = None,
        readable_path: Optional[str] = None
    ) -> Dict[str, str]:
        """
        Save test logs to files.

        Args:
            json_path: Path for JSON log (auto-generated if None)
            readable_path: Path for readable text log (auto-generated if None)

        Returns:
            Dict with paths to saved files
        """
        if not self.enable_logging or not self.test_logger:
            return {"error": "Logging not enabled"}

        saved_files = {}

        try:
            saved_files["json"] = self.test_logger.save_to_json(json_path)
        except Exception as e:
            logger.error(f"Failed to save JSON log: {e}")
            saved_files["json_error"] = str(e)

        try:
            saved_files["readable"] = self.test_logger.export_readable_log(readable_path)
        except Exception as e:
            logger.error(f"Failed to save readable log: {e}")
            saved_files["readable_error"] = str(e)

        return saved_files

    def get_log_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the current test log.

        Returns:
            Dict with log summary information
        """
        if not self.enable_logging or not self.test_logger:
            return {"error": "Logging not enabled"}

        return {
            "turn_summary": self.test_logger.get_turn_summary(),
            "phase_statistics": self.test_logger.get_phase_statistics(),
            "session_id": self.test_logger.current_session.session_id if self.test_logger.current_session else None
        }


# Convenience function for quick testing
def create_confusion_test(
    objects_per_image: Dict[str, List[Dict[str, Any]]],
    config: Optional[CrossImageConfusionTestConfig] = None,
    log_dir: Optional[str] = None,
    enable_logging: bool = True
) -> CrossImageConfusionTester:
    """
    Create and setup a cross-image confusion test.

    Args:
        objects_per_image: Dict mapping image_id to list of objects
        config: Optional test configuration
        log_dir: Directory for saving logs
        enable_logging: Whether to enable logging

    Returns:
        Configured CrossImageConfusionTester instance
    """
    tester = CrossImageConfusionTester(
        config=config,
        log_dir=log_dir,
        enable_logging=enable_logging
    )
    tester.setup_confusable_objects(objects_per_image)
    return tester


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("Testing Cross-Image Confusion Module with Logging...")

    # Example: Two images with similar objects
    objects_per_image = {
        "Image 1": [
            {"id": "person_1", "type": "person", "attributes": {"color": "red", "position": "left", "action": "standing"}},
            {"id": "car_1", "type": "vehicle", "attributes": {"color": "blue", "size": "large"}}
        ],
        "Image 2": [
            {"id": "person_2", "type": "person", "attributes": {"color": "blue", "position": "right", "action": "walking"}},
            {"id": "car_2", "type": "vehicle", "attributes": {"color": "red", "size": "small"}}
        ],
        "Image 3": [
            {"id": "person_3", "type": "person", "attributes": {"color": "red", "position": "center", "action": "sitting"}}
        ]
    }

    # Create test with default config and logging enabled
    config = CrossImageConfusionTestConfig(
        num_images=3,
        min_filler_turns=3,
        max_filler_turns=5,
        difficulty=3,
        language="cn"
    )

    # Enable logging with custom directory
    tester = CrossImageConfusionTester(
        config=config,
        log_dir="./logs/cross_image_confusion",
        enable_logging=True
    )
    results = tester.run_test_sequence(objects_per_image)

    print(f"\n--- Test Results ---")
    print(f"Total turns: {results['total_turns']}")
    print(f"Filler turns: {results['filler_turns']}")
    print(f"Confusable pairs: {results['confusable_pairs']}")
    print(f"Ambiguous references: {results['ambiguous_references']}")

    print(f"\n--- Test Sequence Sample ---")
    for i, turn in enumerate(results['test_sequence'][:5]):
        print(f"Turn {i + 1} [{turn.get('action_type', 'unknown')}]:")
        print(f"  User: {turn.get('user_message', 'N/A')[:80]}...")
        if 'expected_answer' in turn:
            print(f"  Expected: {turn['expected_answer'][:50]}...")
        print()

    # Simulate model responses (in real use, these come from VLM)
    print("\n--- Simulating Model Responses ---")
    fake_responses = [
        f"模拟响应 {i+1}: 我看到了图中的物体..."
        for i in range(results['total_turns'])
    ]

    # Evaluate with logging
    report = tester.evaluate_confusion_metrics(fake_responses)

    print(f"\n--- Evaluation Report ---")
    print(f"Mode: {report.get('mode', 'N/A')}")
    print(f"Total turns evaluated: {report.get('total_turns_evaluated', 0)}")
    if 'aggregate_scores' in report:
        print(f"Aggregate scores: {report['aggregate_scores']}")
    if 'confusion_test_metrics' in report:
        print(f"Confusion test metrics: {report['confusion_test_metrics']}")

    # Save logs
    print("\n--- Saving Logs ---")
    saved_files = tester.save_logs()
    print(f"Saved files: {saved_files}")

    # Get log summary
    print("\n--- Log Summary ---")
    summary = tester.get_log_summary()
    print(f"Session ID: {summary.get('session_id', 'N/A')}")
    print(f"Phase statistics: {summary.get('phase_statistics', {})}")
    print(f"Turn count: {len(summary.get('turn_summary', []))}")

