"""
Strategic User Simulator for M3Bench
=====================================

A redesigned simulator that acts as a "strategic examiner" rather than
a mechanical action trigger. Key features:

1. Task-driven: Actions serve the task goal, not vice versa
2. Adaptive difficulty: Escalates pressure based on model performance
3. Phase-based testing: Grounding → Stress Test → Final Check
4. Decoupled evaluation: Evaluator is independent and configurable
5. Memory injection: Covert tests of visual memory integrity
6. Consistency check: "回马枪" final validation
7. Weak model collaboration: Pseudo multi-turn for context length

This is the main simulator to use for benchmarking.
"""

import json
import re
import random
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import logging
from datetime import datetime
from dataclasses import dataclass, field

from .llm_client import LLMClient
from .memory_store import MemoryStore
from .evaluator import Evaluator, EvaluationMode, EvaluationResult
from .action_space import (
    ACTION_DEFINITIONS, TASK_STRATEGIES, ActionDefinition,
    get_action_for_context
)
from .context_padder import ContextPadder
# 窗口3新增: 动态长度选择器
from .length_selector import DynamicLengthSelector, LengthSelectionStrategy, create_length_selector
# 窗口2新增: 推理链构建器
from .reasoning_chain_builder import ReasoningChainBuilder, ChainQuery, ChainExecutionManager

logger = logging.getLogger(__name__)


@dataclass
class PhaseState:
    """Tracks the current testing phase"""
    phase_name: str
    phase_index: int
    turns_in_phase: int
    min_turns: int
    phase_complete: bool = False


@dataclass
class TaskState:
    """Complete state of current task execution"""
    task_id: str
    task_type: str
    question: str
    expected_answer: str
    images: List[str]

    # Phase tracking
    current_phase: PhaseState = None
    phases_completed: List[str] = field(default_factory=list)

    # Difficulty tracking
    difficulty_level: int = 1
    level_up_candidates: int = 0  # Consecutive good performances

    # Memory tracking
    ground_truths: Dict[str, Any] = field(default_factory=dict)  # Facts model should know
    injected_falsehoods: List[Dict[str, Any]] = field(default_factory=list)
    model_claims: List[Dict[str, Any]] = field(default_factory=list)  # What model said

    # Performance tracking
    recent_scores: List[float] = field(default_factory=list)
    actions_used: List[str] = field(default_factory=list)


class StrategicSimulator:
    """
    Strategic examiner that tests VLM capabilities through
    adaptive, task-driven multi-turn conversations.
    """

    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        evaluator: Optional[Evaluator] = None,
        context_padder: Optional[Any] = None,  # For pseudo multi-turn
        max_turns_per_task: int = 40,  # Increased for thorough testing
        min_turns_per_task: int = 15,  # Ensure enough turns
        enable_consistency_check: bool = True,
        enable_filler_injection: bool = True,  # Inject filler for long context
        filler_interval: int = 5,  # Inject filler every N turns
        verbose: bool = True,
        # === 窗口3新增参数 ===
        length_selection_strategy: str = "adaptive",  # "fixed", "random", "phase_based", "difficulty_based", "adaptive"
        # === Task 1B: 文本提示最小化参数 ===
        minimize_text_hints: bool = True,
        hint_level: str = "minimal"  # "minimal" | "moderate" | "full"
    ):
        """
        Initialize strategic simulator.

        Args:
            llm_client: Client for LLM API calls
            evaluator: Independent evaluator (if None, creates STRESS_TEST mode)
            context_padder: Context padder for filler generation
            max_turns_per_task: Maximum turns before forcing completion (default: 40)
            min_turns_per_task: Minimum turns to ensure thorough testing (default: 15)
            enable_consistency_check: Whether to do final "回马枪" check
            enable_filler_injection: Whether to inject filler for long context
            filler_interval: Inject filler every N turns
            verbose: Print progress
            length_selection_strategy: 长度选择策略 (窗口3新增)
            minimize_text_hints: If True, reduce image descriptions in messages (Task 1B)
            hint_level: Level of text hints allowed:
                - "minimal": Only image IDs (e.g., "Image 1")
                - "moderate": Brief mentions (e.g., "the first image")
                - "full": Detailed descriptions (for text-only model testing)
        """
        self.llm_client = llm_client or LLMClient()
        self.evaluator = evaluator or Evaluator(mode=EvaluationMode.STRESS_TEST)

        # Auto-create context padder if filler injection is enabled but no padder provided
        if context_padder is not None:
            self.context_padder = context_padder
        elif enable_filler_injection:
            # Create context padder using the same API config as llm_client
            self.context_padder = ContextPadder(
                api_url=self.llm_client.api_url,
                api_key=self.llm_client.api_key,
                weak_model=self.llm_client.weak_model,
                use_weak_model=True
            )
        else:
            self.context_padder = None
        self.max_turns = max_turns_per_task
        self.min_turns = min_turns_per_task
        self.enable_consistency_check = enable_consistency_check
        self.enable_filler_injection = enable_filler_injection
        self.filler_interval = filler_interval
        self.verbose = verbose

        # === 窗口3新增: 初始化长度选择器 ===
        self.length_selector = create_length_selector(
            strategy=length_selection_strategy,
            default_length="medium"
        )
        self.length_selection_strategy = length_selection_strategy

        # === 窗口3新增: 长度控制日志 ===
        self.length_control_log: List[Dict[str, Any]] = []

        # === Task 1B: 文本提示最小化配置 ===
        self.minimize_text_hints = minimize_text_hints
        self.hint_level = hint_level

        self.memory = MemoryStore()
        self.task_state: Optional[TaskState] = None
        self.turn_count: int = 0
        self.filler_turns_injected: int = 0

        # Logging
        self.run_log: List[Dict[str, Any]] = []

        # Task 1D: Store conversation history for batch integration
        self.conversation_history: List[Dict[str, Any]] = []

    def _get_strategy(self, task_type: str):
        """Get testing strategy for task type"""
        return TASK_STRATEGIES.get(task_type)

    def _initialize_phases(self, task_type: str) -> PhaseState:
        """Initialize phase tracking for a task"""
        strategy = self._get_strategy(task_type)
        if not strategy or not strategy.phases:
            # Default phase
            return PhaseState(
                phase_name="default",
                phase_index=0,
                turns_in_phase=0,
                min_turns=3
            )

        first_phase = strategy.phases[0]
        return PhaseState(
            phase_name=first_phase["name"],
            phase_index=0,
            turns_in_phase=0,
            min_turns=first_phase.get("min_turns", 2)
        )

    def _advance_phase(self) -> bool:
        """Advance to next phase if conditions met. Returns True if advanced."""
        if not self.task_state:
            return False

        strategy = self._get_strategy(self.task_state.task_type)
        if not strategy:
            return False

        current = self.task_state.current_phase
        phases = strategy.phases

        # Check if current phase is complete
        if current.turns_in_phase < current.min_turns:
            return False

        # Check if there's a next phase
        next_index = current.phase_index + 1
        if next_index >= len(phases):
            current.phase_complete = True
            return False

        # Advance to next phase
        next_phase = phases[next_index]
        self.task_state.phases_completed.append(current.phase_name)
        self.task_state.current_phase = PhaseState(
            phase_name=next_phase["name"],
            phase_index=next_index,
            turns_in_phase=0,
            min_turns=next_phase.get("min_turns", 2)
        )

        if self.verbose:
            print(f"  [Phase] Advanced to: {next_phase['name']}")

        return True

    def _should_increase_difficulty(self) -> bool:
        """Determine if we should increase difficulty level"""
        if not self.task_state:
            return False

        # Need at least 3 recent scores
        recent = self.task_state.recent_scores[-3:]
        if len(recent) < 3:
            return False

        # All recent scores above threshold
        avg_score = sum(recent) / len(recent)
        if avg_score > 0.75:
            self.task_state.level_up_candidates += 1
            return self.task_state.level_up_candidates >= 2
        else:
            self.task_state.level_up_candidates = 0
            return False

    def _increase_difficulty(self):
        """Increase difficulty level"""
        if self.task_state and self.task_state.difficulty_level < 4:
            self.task_state.difficulty_level += 1
            self.task_state.level_up_candidates = 0
            self.evaluator.advance_level()
            if self.verbose:
                print(f"  [Difficulty] Increased to Level {self.task_state.difficulty_level}")

    def start_task(self, task: Dict[str, Any]):
        """Start a new task"""
        task_id = task.get("task_id", f"task_{datetime.now().timestamp()}")
        task_type = task.get("task_type", "attribute_comparison")

        self.task_state = TaskState(
            task_id=task_id,
            task_type=task_type,
            question=task.get("question", ""),
            expected_answer=task.get("answer", ""),
            images=task.get("images", []),
            current_phase=self._initialize_phases(task_type)
        )

        self.turn_count = 0
        self.evaluator.reset_for_task()

        # Initialize memory
        self.memory.start_task(task_id, task_type, self.task_state.expected_answer)

        # Extract ground truths from task
        self._extract_ground_truths(task)

        if self.verbose:
            print(f"\n{'='*60}")
            print(f"Task: {task_id}")
            print(f"Type: {task_type}")
            print(f"Question: {self.task_state.question}")
            print(f"Expected: {self.task_state.expected_answer}")
            print(f"Images: {len(self.task_state.images)}")
            print(f"Phase: {self.task_state.current_phase.phase_name}")
            print(f"{'='*60}")

        self._log_event("task_start", {
            "task_id": task_id,
            "task_type": task_type,
            "question": self.task_state.question,
            "expected_answer": self.task_state.expected_answer,
            "images": self.task_state.images
        })

    def _extract_ground_truths(self, task: Dict[str, Any]):
        """Extract verifiable ground truths from task"""
        if not self.task_state:
            return

        # Register expected answer as a key fact
        self.task_state.ground_truths["expected_answer"] = task.get("answer", "")
        self.evaluator.register_key_fact(
            "expected_answer",
            task.get("answer", ""),
            source="task"
        )

        # Extract from reasoning path if available
        if "reasoning_path" in task:
            for i, step in enumerate(task["reasoning_path"]):
                fact_id = f"reasoning_step_{i}"
                self.task_state.ground_truths[fact_id] = step
                self.evaluator.register_key_fact(fact_id, step, source="task")

    def _select_action(self) -> Tuple[str, Dict[str, Any]]:
        """Select next action based on strategy and context"""
        if not self.task_state:
            return "follow_up", {}

        phase = self.task_state.current_phase
        strategy = self._get_strategy(self.task_state.task_type)

        # Calculate recent performance
        recent_scores = self.task_state.recent_scores[-3:]
        avg_performance = sum(recent_scores) / len(recent_scores) if recent_scores else 0.5

        # Track last 3 actions to enforce diversity
        recent_actions = self.task_state.actions_used[-3:] if self.task_state.actions_used else []

        # Get base action recommendation
        base_action = get_action_for_context(
            task_type=self.task_state.task_type,
            current_phase=phase.phase_name,
            difficulty_level=self.task_state.difficulty_level,
            recent_actions=self.task_state.actions_used[-5:],
            model_performance=avg_performance
        )

        # Get all possible candidates from the action space for this phase
        from .action_space import TASK_STRATEGIES
        task_strategy = TASK_STRATEGIES.get(self.task_state.task_type)

        candidates = [base_action]  # Start with the base recommendation
        if task_strategy:
            # Add phase-appropriate actions
            phase_config = None
            for p in task_strategy.phases:
                if p["name"] == phase.phase_name:
                    phase_config = p
                    break

            if phase_config:
                candidates.extend(phase_config.get("actions", []))

        # Remove duplicates while preserving order
        seen = set()
        unique_candidates = []
        for c in candidates:
            if c not in seen:
                seen.add(c)
                unique_candidates.append(c)
        candidates = unique_candidates

        # Prioritize unused actions (not in recent 3 actions)
        diverse_candidates = [a for a in candidates if a not in recent_actions]

        # Select action
        if diverse_candidates:
            action = random.choice(diverse_candidates)
        elif candidates:
            action = random.choice(candidates)
        else:
            action = base_action

        # Get action definition
        action_def = ACTION_DEFINITIONS.get(action)
        params = {}

        # Fill in action-specific parameters
        if action == "guidance":
            params = self._generate_guidance_params()
        elif action in ["mislead", "mislead_subtle"]:
            params = self._generate_mislead_params()
        elif action == "memory_injection":
            params = self._generate_memory_injection_params()
        elif action == "distraction":
            params = self._generate_distraction_params()
        elif action == "redundancy":
            params = self._generate_redundancy_params()
        elif action == "consistency_check":
            params = self._generate_consistency_check_params()

        return action, params

    def _generate_guidance_params(self) -> Dict[str, Any]:
        """Generate parameters for guidance action"""
        regions = ["左侧", "右侧", "上方", "下方", "中央", "left", "right", "top", "bottom", "center"]
        return {
            "region": random.choice(regions),
            "idx": len(self.memory.current_task.turns) if self.memory.current_task else 1
        }

    def _generate_mislead_params(self) -> Dict[str, Any]:
        """Generate misleading information based on ground truths"""
        if not self.task_state or not self.task_state.ground_truths:
            return {
                "object": "object",
                "wrong_value": "incorrect value"
            }

        # Try to create a plausible falsehood
        expected = self.task_state.expected_answer
        wrong_options = [
            "more", "less", "different",
            "Image 2", "Image 3", "the other one",
            "red", "blue", "larger", "smaller"
        ]

        return {
            "object": "the target",
            "wrong_value": random.choice(wrong_options)
        }

    def _generate_memory_injection_params(self) -> Dict[str, Any]:
        """Generate fake object/fact to inject"""
        fake_objects = [
            "那把红色的伞", "背景里的自行车", "角落的猫",
            "a red umbrella", "a bicycle in the background", "a cat in the corner"
        ]
        real_objects = ["the person", "the main object", "那个人"]

        return {
            "fake_object": random.choice(fake_objects),
            "real_object": random.choice(real_objects)
        }

    def _generate_distraction_params(self) -> Dict[str, Any]:
        """Generate irrelevant question for distraction"""
        distractions = [
            "这张图的光线是什么时候拍的？",
            "图片的分辨率看起来如何？",
            "这是室内还是室外？",
            "What time of day does this look like?",
            "Is there any text visible in the image?",
            "What's the overall color tone of this image?"
        ]
        return {"question": random.choice(distractions)}

    def _generate_redundancy_params(self) -> Dict[str, Any]:
        """Generate redundant/verbose restatement"""
        if self.memory.current_task and self.memory.current_task.turns:
            last_turn = self.memory.current_task.turns[-1]
            return {
                "fact": last_turn.model_response[:100],
                "fact_paraphrase": "the same thing you just mentioned"
            }
        return {"fact": "what you said", "fact_paraphrase": "that observation"}

    def _generate_consistency_check_params(self) -> Dict[str, Any]:
        """Generate consistency check question"""
        return {
            "original_question": self.task_state.question if self.task_state else "the original question",
            "core_fact_question": "the main answer we established"
        }

    def _build_message(self, action: str, params: Dict[str, Any]) -> str:
        """Build natural language message from action and params"""
        action_def = ACTION_DEFINITIONS.get(action)
        if not action_def:
            return "Please continue."

        # Select a random template
        template = random.choice(action_def.templates)

        # Fill in parameters
        try:
            message = template.format(**params)
        except KeyError:
            # Fallback if params don't match template
            message = template
            for key, value in params.items():
                message = message.replace(f"{{{key}}}", str(value))

        return message

    def _call_core_model_for_action(self) -> Tuple[str, str, Dict[str, Any]]:
        """
        Use core model to decide action AND generate natural message.
        Returns: (action, message, parsed_response)
        """
        if not self.task_state:
            return "follow_up", "Please continue.", {}

        # Build context for core model
        phase = self.task_state.current_phase
        strategy = self._get_strategy(self.task_state.task_type)

        allowed_actions = []
        if strategy:
            allowed_actions = strategy.difficulty_progression.get(
                self.task_state.difficulty_level,
                ["follow_up"]
            )

        # Get current phase actions if available
        if strategy and phase.phase_index < len(strategy.phases):
            phase_config = strategy.phases[phase.phase_index]
            phase_actions = phase_config.get("actions", [])
            allowed_actions = [a for a in allowed_actions if a in phase_actions] or allowed_actions

        system_prompt = self._build_core_system_prompt(allowed_actions)
        user_prompt = self._build_core_user_prompt()

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        max_retries = 2
        for attempt in range(max_retries):
            response = self.llm_client.call_core_model(messages, max_tokens=1500)

            if not response.get("success"):
                # Fallback to rule-based
                action, params = self._select_action()
                message = self._build_message(action, params)
                return action, message, {}

            # Parse response with both content and reasoning_content
            parsed = self._parse_core_response(
                content=response.get("content", ""),
                reasoning_content=response.get("reasoning_content", "")
            )

            # Check if parsing succeeded
            if not parsed.get("parse_error"):
                action = parsed.get("action", "follow_up")
                message = parsed.get("message", self._build_message(action, {}))
                return action, message, parsed

            # If parse failed and we have retries left, try again with explicit instruction
            if attempt < max_retries - 1:
                logger.info(f"Parse failed on attempt {attempt + 1}, retrying with explicit JSON instruction")
                # Add explicit JSON format instruction to the user prompt
                messages[-1]["content"] += "\n\nIMPORTANT: Please respond with ONLY valid JSON in this exact format:\n{\"action\": \"...\", \"message\": \"...\"}"
            else:
                logger.warning(f"All {max_retries} parse attempts failed, using fallback")
                # Final fallback after all retries
                action = parsed.get("action", "follow_up")
                message = parsed.get("message", self._build_message(action, {}))
                return action, message, parsed

    def _build_core_system_prompt(self, allowed_actions: List[str]) -> str:
        """Build system prompt for core model"""
        action_descriptions = []
        for action_name in allowed_actions:
            action_def = ACTION_DEFINITIONS.get(action_name)
            if action_def:
                action_descriptions.append(
                    f"- {action_name}: {action_def.purpose}"
                )

        return f"""你是一个VLM能力测试的考官。你的任务是通过策略性的多轮对话来测试目标模型的能力极限。

## 当前任务
任务类型: {self.task_state.task_type if self.task_state else 'unknown'}
问题: {self.task_state.question if self.task_state else 'unknown'}
预期答案: {self.task_state.expected_answer if self.task_state else 'unknown'}
当前阶段: {self.task_state.current_phase.phase_name if self.task_state else 'unknown'}
难度级别: {self.task_state.difficulty_level if self.task_state else 1}/4
已用轮数: {self.turn_count}

## 可用动作
{chr(10).join(action_descriptions)}

## 策略指南
1. **任务导向**: 所有动作都是为了测试模型完成任务的能力
2. **渐进压力**: 先建立基础，再逐步增加压力
3. **自适应**: 模型表现好就加难度，表现差就调整策略
4. **隐蔽性**: mislead和memory_injection要自然，不要太明显
5. **最终验证**: 在结束前必须做consistency_check

## 输出格式
请以JSON格式输出:
```json
{{
    "action": "动作名称",
    "message": "发送给目标模型的自然语言消息",
    "reasoning": "选择此动作的原因",
    "expected_response": "预期模型会如何回应",
    "difficulty_assessment": "当前对模型难度的评估"
}}
```
"""

    def _build_core_user_prompt(self) -> str:
        """Build user prompt with conversation history"""
        history_lines = []

        if self.memory.current_task and self.memory.current_task.turns:
            for turn in self.memory.current_task.turns[-5:]:
                history_lines.append(f"[Turn {turn.turn_id}] Action: {turn.action}")
                history_lines.append(f"User: {turn.user_message[:150]}...")
                history_lines.append(f"Model: {turn.model_response[:150]}...")
                if turn.evaluation:
                    score = turn.evaluation.get("score", "N/A")
                    history_lines.append(f"Eval Score: {score}")
                history_lines.append("")

        history = "\n".join(history_lines) if history_lines else "This is the first turn."

        return f"""## 对话历史
{history}

## 当前状态
- 阶段进度: {self.task_state.current_phase.turns_in_phase}/{self.task_state.current_phase.min_turns} turns in current phase
- 最近分数: {self.task_state.recent_scores[-3:] if self.task_state else []}
- 已用动作: {self.task_state.actions_used[-5:] if self.task_state else []}

请选择下一个动作并生成消息。"""

    def _parse_core_response(self, content: str, reasoning_content: str = "") -> Dict[str, Any]:
        """
        Parse core model's JSON response.
        Handles o1-preview and similar models that may return reasoning_content with empty content.

        Args:
            content: The main content field from the response
            reasoning_content: The reasoning_content field (for o1-preview etc.)

        Returns:
            Parsed JSON dict with 'parse_error' key set to True on failure
        """
        # Determine which text to parse - prefer content, fallback to reasoning_content
        text_to_parse = content.strip() if content and content.strip() else reasoning_content.strip()

        if not text_to_parse:
            logger.warning("Both content and reasoning_content are empty")
            return {
                "action": "follow_up",
                "message": "Please tell me more about what you need.",
                "parse_error": True,
                "error_reason": "empty_response"
            }

        try:
            # Try to extract JSON from markdown code block
            json_match = re.search(r'```json\s*(.*?)\s*```', text_to_parse, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group(1))
                result["parse_error"] = False
                return result

            # Try to extract JSON from generic code block
            code_match = re.search(r'```\s*(.*?)\s*```', text_to_parse, re.DOTALL)
            if code_match:
                try:
                    result = json.loads(code_match.group(1))
                    result["parse_error"] = False
                    return result
                except json.JSONDecodeError:
                    pass

            # Try parsing the whole text as JSON
            result = json.loads(text_to_parse)
            result["parse_error"] = False
            return result

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse core model response as JSON: {e}")
            logger.debug(f"Content (first 200 chars): {content[:200] if content else 'empty'}")
            logger.debug(f"Reasoning (first 200 chars): {reasoning_content[:200] if reasoning_content else 'empty'}")

            return {
                "action": "follow_up",
                "message": text_to_parse[:500] if text_to_parse else "Could you elaborate on that?",
                "parse_error": True,
                "error_reason": "json_decode_error"
            }

    def _run_filler_turn(self) -> Optional[Dict[str, Any]]:
        """
        Run a filler turn using weak model to extend context.
        Returns filler turn data or None if filler injection is disabled.
        """
        if not self.enable_filler_injection or not self.context_padder:
            return None

        # Generate filler using context padder
        context_summary = ""
        if self.task_state and self.task_state.model_claims:
            recent_claims = self.task_state.model_claims[-3:]
            context_summary = "; ".join([c["claim"][:100] for c in recent_claims])

        filler = self.context_padder.generate_filler_turn({
            "turn_index": self.filler_turns_injected,
            "summary": context_summary
        })

        filler_user_msg = filler["user_message"]
        filler_model_resp = filler["model_response"]

        if self.verbose:
            print(f"\n--- Filler Turn (Context Padding) ---")
            print(f"Filler Q: {filler_user_msg[:80]}...")
            print(f"Filler A: {filler_model_resp[:100]}...")

        # Add filler to memory (as simulated turns)
        self.memory.add_turn(
            action="filler",
            user_message=filler_user_msg,
            model_response=filler_model_resp,
            evaluation={"score": 1.0, "is_filler": True},
            key_info=["Filler turn for context padding"]
        )

        self.filler_turns_injected += 1

        self._log_event("filler_turn", {
            "filler_index": self.filler_turns_injected,
            "user_message": filler_user_msg,
            "model_response": filler_model_resp,
            "topic": filler.get("topic", "unknown"),
            "response_length": filler.get("response_length", 0)
        })

        return filler

    def step(self) -> Dict[str, Any]:
        """Execute one turn of the simulation"""
        if not self.task_state:
            raise RuntimeError("No task started. Call start_task() first.")

        # Check if we should inject a filler turn for context padding
        if (self.enable_filler_injection and
            self.context_padder and
            self.turn_count > 0 and
            self.turn_count % self.filler_interval == 0):
            self._run_filler_turn()

        self.turn_count += 1
        phase = self.task_state.current_phase
        phase.turns_in_phase += 1

        if self.verbose:
            print(f"\n--- Turn {self.turn_count} (Phase: {phase.phase_name}, Level: {self.task_state.difficulty_level}) ---")

        # 1. Decide action (use core model or rule-based)
        action, message, core_parsed = self._call_core_model_for_action()

        self.task_state.actions_used.append(action)

        if self.verbose:
            print(f"Action: {action}")
            print(f"Message: {message[:100]}...")

        # 2. Track misleading attempts
        if action in ["mislead", "mislead_subtle", "memory_injection"]:
            self.task_state.injected_falsehoods.append({
                "action": action,
                "message": message,
                "turn": self.turn_count
            })
            self.evaluator.register_injected_falsehood(
                falsehood=message,
                truth=self.task_state.expected_answer,
                injection_type=action
            )

        # 3. Determine images to send
        images_to_send = self._get_images_for_turn(action)

        # Task 1B: Sanitize message to minimize text hints
        original_message = message
        message = self._sanitize_message_for_target(
            message=message,
            images=images_to_send,
            action_type=action
        )

        if self.verbose and original_message != message:
            print(f"[Sanitized] Original length: {len(original_message)}, New length: {len(message)}")

        # 4. Build conversation for target model
        target_messages = self.memory.get_conversation_history()
        target_messages.append({"role": "user", "content": message})

        # 5. Call target model
        target_response = self.llm_client.call_target_model(
            messages=target_messages,
            images=images_to_send if images_to_send else None,
            max_tokens=2048  # Increased to allow complete reasoning
        )

        if not target_response.get("success"):
            model_content = "[Target model error]"
        else:
            model_content = target_response.get("content", "")

        if self.verbose:
            print(f"Model: {model_content[:150]}...")

        # 6. Track model claims
        self.task_state.model_claims.append({
            "turn": self.turn_count,
            "claim": model_content,
            "action_context": action
        })

        # 7. Evaluate response
        eval_result = self.evaluator.evaluate_response(
            response=model_content,
            expected_answer=self.task_state.expected_answer,
            action_type=action,
            question_asked=message,
            context={
                "previous_responses": [c["claim"] for c in self.task_state.model_claims[-5:]],
                "phase": phase.phase_name
            }
        )

        self.task_state.recent_scores.append(eval_result.score)

        if self.verbose:
            print(f"Score: {eval_result.score:.2f} | Passed: {eval_result.level_passed}")

        # 8. Store in memory
        self.memory.add_turn(
            action=action,
            user_message=message,
            model_response=model_content,
            evaluation=eval_result.to_dict(),
            key_info=[f"Turn {self.turn_count}: {action}"]
        )

        # 9. Log
        self._log_event("turn", {
            "turn": self.turn_count,
            "phase": phase.phase_name,
            "difficulty": self.task_state.difficulty_level,
            "action": action,
            "message": message,
            "response": model_content,
            "evaluation": eval_result.to_dict(),
            "images_sent": images_to_send
        })

        # Task 1D: Record turn details in conversation history
        from datetime import datetime
        self.conversation_history.append({
            'turn': self.turn_count,
            'action': action,
            'query': message,
            'response': model_content,
            'images_sent': images_to_send if images_to_send else [],
            'evaluation': eval_result.to_dict(),
            'timestamp': datetime.now().isoformat(),
            'phase': phase.phase_name,
            'difficulty': self.task_state.difficulty_level
        })

        # 10. Check for difficulty increase
        if self._should_increase_difficulty():
            self._increase_difficulty()

        # 11. Check for phase advancement
        self._advance_phase()

        # 12. Determine if task should continue
        should_continue = self._should_continue()

        return {
            "turn": self.turn_count,
            "action": action,
            "message": message,
            "response": model_content,
            "evaluation": eval_result.to_dict(),
            "should_continue": should_continue,
            "phase": phase.phase_name,
            "difficulty": self.task_state.difficulty_level
        }

    def _get_images_for_turn(self, action: str) -> List[str]:
        """Determine which images to send for current turn

        For multimodal testing, we need to send ALL images on EVERY turn
        so the model can actually see and process the visual content.
        """
        if not self.task_state or not self.task_state.images:
            return []

        # Send all images for all actions (multimodal testing requires this)
        valid_images = []
        for img_rel_path in self.task_state.images:
            # Try to find the image file
            # Images are stored relative to task file location
            # Common patterns: generated_tasks_v2/run_XX/images/...

            # Method 1: Check if path already exists as-is (relative to cwd)
            img_path = Path(img_rel_path)
            if img_path.exists():
                valid_images.append(str(img_path))
                continue

            # Method 2: Try different run_XX directories
            for run_dir in Path("generated_tasks_v2").glob("run_*"):
                potential_path = run_dir / img_rel_path
                if potential_path.exists():
                    valid_images.append(str(potential_path))
                    break

        return valid_images

    def _should_continue(self) -> bool:
        """Determine if task should continue"""
        if not self.task_state:
            return False

        # Max turns reached - hard stop
        if self.turn_count >= self.max_turns:
            if self.verbose:
                print(f"  [Stop] Max turns ({self.max_turns}) reached")
            return False

        # Always continue if below minimum turns
        if self.turn_count < self.min_turns:
            return True

        # Check if all phases complete
        phase = self.task_state.current_phase
        strategy = self._get_strategy(self.task_state.task_type)

        if strategy:
            all_phases_done = phase.phase_index >= len(strategy.phases) - 1 and phase.phase_complete
            # Only stop if all phases done AND we've done enough turns
            if all_phases_done and self.turn_count >= self.min_turns:
                if self.verbose:
                    print(f"  [Stop] All phases complete at turn {self.turn_count}")
                return False

        # Keep going until max turns if not all phases done
        # This ensures longer conversations for better testing
        return True

    def run_consistency_check(self) -> Dict[str, Any]:
        """Run final consistency check ("回马枪")"""
        if not self.task_state or not self.enable_consistency_check:
            return {}

        if self.verbose:
            print(f"\n--- Consistency Check (回马枪) ---")

        message = f"让我们回到最初的问题。{self.task_state.question}"

        target_messages = self.memory.get_conversation_history()
        target_messages.append({"role": "user", "content": message})

        response = self.llm_client.call_target_model(
            messages=target_messages,
            max_tokens=2048  # Increased to allow complete reasoning
        )

        model_content = response.get("content", "") if response.get("success") else ""

        # Compare with expected
        eval_result = self.evaluator.evaluate_response(
            response=model_content,
            expected_answer=self.task_state.expected_answer,
            action_type="consistency_check"
        )

        if self.verbose:
            print(f"Final Answer: {model_content[:150]}...")
            print(f"Consistency Score: {eval_result.score:.2f}")

        self._log_event("consistency_check", {
            "question": message,
            "response": model_content,
            "score": eval_result.score,
            "expected": self.task_state.expected_answer
        })

        return {
            "passed": eval_result.score > 0.7,
            "score": eval_result.score,
            "response": model_content
        }

    def run_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Run complete task with all phases"""
        self.start_task(task)

        results = []
        while True:
            result = self.step()
            results.append(result)

            if not result.get("should_continue"):
                break

        # Run consistency check
        consistency = self.run_consistency_check()

        # Generate final report
        report = self._generate_task_report(results, consistency)

        return report

    def _generate_task_report(
        self,
        turn_results: List[Dict[str, Any]],
        consistency: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate comprehensive task report"""
        if not self.task_state:
            return {}

        aggregate = self.evaluator.get_aggregate_scores()
        evaluator_report = self.evaluator.generate_final_report()

        return {
            "task_id": self.task_state.task_id,
            "task_type": self.task_state.task_type,
            "question": self.task_state.question,
            "expected_answer": self.task_state.expected_answer,

            "execution": {
                "total_turns": self.turn_count,
                "final_difficulty": self.task_state.difficulty_level,
                "phases_completed": self.task_state.phases_completed,
                "actions_used": self.task_state.actions_used,
            },

            "scores": {
                "aggregate": aggregate,
                "per_turn": [r["evaluation"]["score"] for r in turn_results if "evaluation" in r],
                "consistency_check": consistency
            },

            "stress_test": {
                "misleading_attempts": len(self.task_state.injected_falsehoods),
                "resistance_rate": evaluator_report.get("stress_test_summary", {}).get("resistance_rate", 0),
                "memory_stability": aggregate.get("memory_retention", 0)
            },

            "evaluator_report": evaluator_report
        }

    def _log_event(self, event_type: str, data: Dict[str, Any]):
        """Log an event"""
        self.run_log.append({
            "event": event_type,
            "timestamp": datetime.now().isoformat(),
            "data": data
        })

    # ========== 窗口3新增: 长度控制相关方法 ==========

    def _select_query_length(self) -> Dict[str, Any]:
        """
        选择query长度 (窗口3新增)

        Returns:
            {
                "length": str,          # 选择的长度 ("short"/"medium"/"long")
                "reason": str,          # 选择原因
                "probabilities": Dict   # 各长度的概率
            }
        """
        # 收集历史中的长度信息
        history_with_length = []
        for log_entry in self.length_control_log:
            history_with_length.append({
                "query_length_control": log_entry.get("selected_length", "medium")
            })

        return self.length_selector.select(
            phase=self.task_state.current_phase.phase_name if self.task_state else None,
            difficulty=self.task_state.difficulty_level if self.task_state else 2,
            turn=self.turn_count,
            history=history_with_length,
            task_type=self.task_state.task_type if self.task_state else None
        )

    def _log_length_control(
        self,
        turn: int,
        selected_length: str,
        query: str,
        suffix_applied: bool,
        selection_reason: str
    ):
        """
        记录长度控制信息 (窗口3新增)

        Args:
            turn: 轮次
            selected_length: 选择的长度
            query: 生成的query
            suffix_applied: 是否应用了长度后缀
            selection_reason: 选择原因
        """
        self.length_control_log.append({
            "turn": turn,
            "selected_length": selected_length,
            "query_preview": query[:50] + "..." if len(query) > 50 else query,
            "query_word_count": len(query.split()),
            "suffix_applied": suffix_applied,
            "selection_reason": selection_reason,
            "timestamp": datetime.now().isoformat()
        })

    def get_length_control_summary(self) -> Dict[str, Any]:
        """
        获取长度控制摘要 (窗口3新增)

        Returns:
            长度控制的详细统计信息
        """
        if not self.length_control_log:
            return {"message": "No length control data"}

        length_counts = {}
        suffix_applied_count = 0
        word_counts_by_length = {"short": [], "medium": [], "long": []}

        for entry in self.length_control_log:
            length = entry["selected_length"]
            length_counts[length] = length_counts.get(length, 0) + 1

            if entry["suffix_applied"]:
                suffix_applied_count += 1

            if length in word_counts_by_length:
                word_counts_by_length[length].append(entry["query_word_count"])

        # 计算平均词数
        avg_word_counts = {}
        for length, counts in word_counts_by_length.items():
            if counts:
                avg_word_counts[length] = sum(counts) / len(counts)

        return {
            "total_entries": len(self.length_control_log),
            "length_distribution": length_counts,
            "suffix_applied_rate": suffix_applied_count / len(self.length_control_log) if self.length_control_log else 0,
            "avg_word_count_by_length": avg_word_counts,
            "strategy_used": self.length_selection_strategy
        }

    def export_logs(self, output_dir: str) -> str:
        """Export all logs to files"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save run log (包含长度控制信息 - 窗口3增强)
        log_data = {
            "run_log": self.run_log,
            # === 窗口3新增: 长度控制统计 ===
            "length_control": {
                "summary": self.get_length_control_summary(),
                "detailed_log": self.length_control_log
            }
        }
        log_file = output_path / f"strategic_run_{timestamp}.json"
        with open(log_file, "w", encoding="utf-8") as f:
            json.dump(log_data, f, ensure_ascii=False, indent=2)

        # Save memory
        memory_file = output_path / f"memory_{timestamp}.json"
        self.memory.export_to_json(str(memory_file))

        if self.verbose:
            print(f"\nLogs saved to: {output_path}")
            # 窗口3新增: 打印长度控制摘要
            length_summary = self.get_length_control_summary()
            if length_summary.get("total_entries", 0) > 0:
                print(f"Length Control Summary:")
                print(f"  Strategy: {length_summary.get('strategy_used', 'unknown')}")
                print(f"  Distribution: {length_summary.get('length_distribution', {})}")
                print(f"  Suffix Applied Rate: {length_summary.get('suffix_applied_rate', 0):.2%}")

        return str(output_path)

    # ========== 窗口2新增: 推理链测试模式 ==========

    def run_reasoning_chain_task(
        self,
        task: Dict[str, Any],
        verbose: bool = None
    ) -> Dict[str, Any]:
        """
        运行推理链测试任务 (窗口2新增)

        针对rationale_based_abr任务类型，使用推理链进行分步验证。

        流程:
        1. 构建推理链查询序列
        2. 逐步验证每个推理步骤
        3. 在适当位置插入压力测试
        4. 生成详细的推理链评估报告

        Args:
            task: 必须包含 'reasoning_chain' 字段的任务
            verbose: 是否打印详细信息

        Returns:
            包含推理链评估结果的报告
        """
        if verbose is None:
            verbose = self.verbose

        # 验证任务类型
        if not task.get('reasoning_chain'):
            if verbose:
                print("Warning: Task doesn't have reasoning_chain, falling back to standard run")
            return self.run_task(task)

        # 初始化任务
        self.start_task(task)

        # 创建推理链执行管理器
        chain_manager = ChainExecutionManager(
            task=task,
            query_generator=None  # 未来窗口3集成
        )

        if verbose:
            print(f"\n{'='*60}")
            print(f"[Reasoning Chain Mode] Task: {task.get('task_id', 'unknown')}")
            print(f"Total reasoning steps: {len(chain_manager.queries)}")
            print(f"{'='*60}")

        # 执行推理链
        chain_results = []
        stress_test_results = []

        for i, query in enumerate(chain_manager.queries):
            # 1. 执行推理查询
            chain_result = self._execute_chain_query(query, verbose)
            chain_results.append(chain_result)

            # 2. 更新推理链状态
            result = chain_manager.process_response(
                response=chain_result['response'],
                query=query
            )

            # 3. 在中间步骤后插入压力测试
            if query.step_type == "intermediate" and random.random() < 0.5:
                stress_result = self._inject_chain_stress_test(query, chain_result, verbose)
                stress_test_results.append(stress_result)

            # 4. 检查是否链条断裂
            if chain_manager.builder.state.chain_broken:
                if verbose:
                    print(f"\n[Chain Broken] Step {query.step_index}: {chain_manager.builder.state.break_reason}")
                break

        # 执行最终一致性检查
        consistency_result = self._run_chain_consistency_check(task, chain_results, verbose)

        # 生成报告
        report = self._generate_chain_report(
            task=task,
            chain_manager=chain_manager,
            chain_results=chain_results,
            stress_results=stress_test_results,
            consistency_result=consistency_result
        )

        return report

    def _execute_chain_query(
        self,
        query: ChainQuery,
        verbose: bool
    ) -> Dict[str, Any]:
        """执行单个推理链查询"""
        self.turn_count += 1

        if verbose:
            print(f"\n--- Chain Step {query.step_index + 1} ({query.step_type}) ---")
            print(f"Query: {query.query[:100]}...")

        # 构建对话
        target_messages = self.memory.get_conversation_history()
        target_messages.append({"role": "user", "content": query.query})

        # 获取图片
        images_to_send = self._get_images_for_turn("follow_up")

        # 调用目标模型
        response = self.llm_client.call_target_model(
            messages=target_messages,
            images=images_to_send if images_to_send else None,
            max_tokens=2048  # Increased to allow complete reasoning
        )

        model_content = response.get("content", "") if response.get("success") else "[Error]"

        if verbose:
            print(f"Response: {model_content[:150]}...")

        # 评估响应
        eval_result = self.evaluator.evaluate_response(
            response=model_content,
            expected_answer=query.expected_info,
            action_type="reasoning_chain_step",
            question_asked=query.query,
            context={
                "step_type": query.step_type,
                "validation_hints": query.validation_hints
            }
        )

        if verbose:
            print(f"Score: {eval_result.score:.2f}")

        # 更新记忆
        self.memory.add_turn(
            action=f"chain_step_{query.step_type}",
            user_message=query.query,
            model_response=model_content,
            evaluation=eval_result.to_dict(),
            key_info=[f"Chain step {query.step_index}: {query.step_type}"]
        )

        self._log_event("chain_step", {
            "turn": self.turn_count,
            "step_index": query.step_index,
            "step_type": query.step_type,
            "query": query.query,
            "response": model_content,
            "score": eval_result.score
        })

        return {
            "step_index": query.step_index,
            "step_type": query.step_type,
            "query": query.query,
            "response": model_content,
            "evaluation": eval_result.to_dict(),
            "hints_matched": self._count_hint_matches(model_content, query.validation_hints)
        }

    def _inject_chain_stress_test(
        self,
        current_query: ChainQuery,
        current_result: Dict,
        verbose: bool
    ) -> Dict[str, Any]:
        """在推理链中间注入压力测试"""
        self.turn_count += 1

        # 生成误导性问题
        mislead_templates = [
            "等等，我记得你之前说的是{fake_claim}，不是吗？",
            "Wait, didn't you say {fake_claim} earlier?",
            "但是{fake_claim}不是更合理吗？",
            "我觉得{fake_claim}才是对的。"
        ]

        # 构造虚假声明
        fake_claim = self._generate_fake_claim(current_result['response'])

        message = random.choice(mislead_templates).format(fake_claim=fake_claim)

        if verbose:
            print(f"\n--- [Stress Test: Mislead] ---")
            print(f"Mislead: {message[:80]}...")

        # 调用模型
        target_messages = self.memory.get_conversation_history()
        target_messages.append({"role": "user", "content": message})

        response = self.llm_client.call_target_model(
            messages=target_messages,
            max_tokens=2048  # Increased to allow complete reasoning
        )

        model_content = response.get("content", "") if response.get("success") else ""

        # 评估抗误导能力
        eval_result = self.evaluator.evaluate_response(
            response=model_content,
            expected_answer=current_result['response'],  # 期望维持原有答案
            action_type="mislead",
            question_asked=message,
            context={"is_chain_stress_test": True}
        )

        if verbose:
            print(f"Response: {model_content[:100]}...")
            print(f"Robustness: {eval_result.robustness_score:.2f}")

        # 更新记忆
        self.memory.add_turn(
            action="chain_stress_mislead",
            user_message=message,
            model_response=model_content,
            evaluation=eval_result.to_dict(),
            key_info=["Chain stress test"]
        )

        self._log_event("chain_stress_test", {
            "turn": self.turn_count,
            "fake_claim": fake_claim,
            "response": model_content,
            "robustness_score": eval_result.robustness_score
        })

        return {
            "type": "mislead",
            "fake_claim": fake_claim,
            "response": model_content,
            "robustness_score": eval_result.robustness_score,
            "resisted": eval_result.robustness_score >= 0.6
        }

    def _run_chain_consistency_check(
        self,
        task: Dict,
        chain_results: List[Dict],
        verbose: bool
    ) -> Dict[str, Any]:
        """推理链最终一致性检查"""
        self.turn_count += 1

        if verbose:
            print(f"\n--- [Chain Consistency Check] ---")

        # 回顾整个推理链
        chain_summary = " → ".join([
            r['response'][:50] for r in chain_results[:3]
        ])

        message = f"""让我们回顾一下推理过程：
{chain_summary}...

基于以上分析，请再次回答最初的问题：{task.get('question', '')}"""

        target_messages = self.memory.get_conversation_history()
        target_messages.append({"role": "user", "content": message})

        response = self.llm_client.call_target_model(
            messages=target_messages,
            max_tokens=2048  # Increased to allow complete reasoning
        )

        model_content = response.get("content", "") if response.get("success") else ""

        if verbose:
            print(f"Final answer: {model_content[:150]}...")

        # 与预期答案比较
        expected_answer = task.get('answer', '')
        eval_result = self.evaluator.evaluate_response(
            response=model_content,
            expected_answer=expected_answer,
            action_type="consistency_check",
            question_asked=task.get('question', ''),
            context={"is_chain_final": True}
        )

        if verbose:
            print(f"Consistency score: {eval_result.score:.2f}")

        return {
            "final_response": model_content,
            "expected_answer": expected_answer,
            "score": eval_result.score,
            "passed": eval_result.score >= 0.6
        }

    def _generate_chain_report(
        self,
        task: Dict,
        chain_manager: ChainExecutionManager,
        chain_results: List[Dict],
        stress_results: List[Dict],
        consistency_result: Dict
    ) -> Dict[str, Any]:
        """生成推理链测试报告"""
        chain_report = chain_manager.get_final_report()
        aggregate = self.evaluator.get_aggregate_scores()

        # 计算各步骤得分
        step_scores = [r['evaluation']['score'] for r in chain_results]
        avg_step_score = sum(step_scores) / len(step_scores) if step_scores else 0

        # 计算抗压能力
        stress_resistance = sum(1 for r in stress_results if r.get('resisted', False))
        stress_total = len(stress_results)
        stress_rate = stress_resistance / stress_total if stress_total > 0 else 1.0

        return {
            "task_id": task.get('task_id', ''),
            "task_type": "rationale_based_abr",
            "question": task.get('question', ''),
            "expected_answer": task.get('answer', ''),

            # 推理链执行信息
            "chain_execution": {
                "total_steps": chain_report['total_steps'],
                "steps_completed": chain_report['steps_completed'],
                "chain_complete": chain_report['chain_complete'],
                "chain_broken": chain_report['chain_broken'],
                "break_reason": chain_report.get('break_reason')
            },

            # 分数
            "scores": {
                "avg_step_score": avg_step_score,
                "step_pass_rate": chain_report['pass_rate'],
                "stress_resistance_rate": stress_rate,
                "consistency_score": consistency_result['score'],
                "aggregate": aggregate
            },

            # 详细结果
            "step_results": [
                {
                    "step": r['step_index'],
                    "type": r['step_type'],
                    "score": r['evaluation']['score'],
                    "hints_matched": r.get('hints_matched', 0)
                }
                for r in chain_results
            ],

            # 压力测试结果
            "stress_tests": stress_results,

            # 一致性检查
            "consistency_check": consistency_result,

            # 原始推理链信息
            "reasoning_chain": task.get('reasoning_chain', {}),
            "ground_truth_rationale": task.get('ground_truth_rationale', '')
        }

    def _count_hint_matches(self, response: str, hints: List[str]) -> int:
        """计算回答中匹配的提示数量"""
        response_lower = response.lower()
        return sum(1 for h in hints if h.lower() in response_lower)

    def _generate_fake_claim(self, original: str) -> str:
        """从原始回答生成虚假声明"""
        # 简单策略：替换关键词
        replacements = {
            "left": "right", "right": "left",
            "happy": "sad", "sad": "happy",
            "yes": "no", "no": "yes",
            "big": "small", "small": "big",
            "black": "white", "white": "black",
            "open": "closed", "closed": "open",
            "standing": "sitting", "sitting": "standing",
            "他": "她", "她": "他",
            "高兴": "难过", "难过": "高兴",
            "大": "小", "小": "大"
        }

        fake = original[:100]
        for old, new in replacements.items():
            if old in fake.lower():
                fake = fake.replace(old, new).replace(old.title(), new.title())
                break

        if fake == original[:100]:
            fake = f"其实不是{original[:50]}"

        return fake

    # ========== Task 1B: 文本提示最小化方法 ==========

    def _sanitize_message_for_target(
        self,
        message: str,
        images: List[str] = None,
        action_type: str = None
    ) -> str:
        """
        Remove or reduce visual descriptions from user message (Task 1B).

        Args:
            message: Original message from core model
            images: List of image paths being sent
            action_type: The action type (e.g., "guidance", "follow_up")

        Returns:
            Sanitized message with minimal text hints
        """
        if not self.minimize_text_hints or self.hint_level == "full":
            return message  # Full descriptions allowed

        # Patterns to remove/replace
        visual_description_patterns = [
            # Detailed descriptions
            (r"showing [\w\s,]+", ""),
            (r"wearing [\w\s,]+", ""),
            (r"with [\w\s,]+ in the background", ""),
            (r"in the foreground", ""),
            (r"positioned [\w\s,]+", ""),
            # Color/attribute mentions
            (r"\b(red|blue|green|yellow|black|white|brown|gray|grey|orange|purple|pink)\s+(helmet|shirt|car|bike|object|person|animal)\b", r"\2"),
            # Spatial descriptions - preserve for guidance action
            (r"\bthe left side\b", "one side" if action_type != "guidance" else "the left side"),
            (r"\bthe right side\b", "one side" if action_type != "guidance" else "the right side"),
            (r"\bon the left\b", "on one side" if action_type != "guidance" else "on the left"),
            (r"\bon the right\b", "on one side" if action_type != "guidance" else "on the right"),
            (r"\bin the center\b", "in the middle" if action_type != "guidance" else "in the center"),
            (r"\bat the top\b", "at one position" if action_type != "guidance" else "at the top"),
            (r"\bat the bottom\b", "at one position" if action_type != "guidance" else "at the bottom"),
        ]

        sanitized = message

        if self.hint_level == "minimal":
            # Remove all visual descriptions
            for pattern, replacement in visual_description_patterns:
                sanitized = re.sub(pattern, replacement, sanitized, flags=re.IGNORECASE)

            # Replace detailed references with generic ones
            sanitized = re.sub(
                r"(the|this|that) (image|picture|photo) (showing|of|with) [^.]+",
                r"\1 \2",
                sanitized,
                flags=re.IGNORECASE
            )

            # Remove phrases like "Here is an image showing..."
            sanitized = re.sub(
                r"Here is (an|the) (image|picture|photo) showing [^.]+\.",
                "Here is an image.",
                sanitized,
                flags=re.IGNORECASE
            )

        elif self.hint_level == "moderate":
            # Keep basic structure, remove specific attributes (colors, sizes)
            for pattern, replacement in visual_description_patterns[5:]:  # Only remove colors/spatial
                sanitized = re.sub(pattern, replacement, sanitized, flags=re.IGNORECASE)

        # Clean up multiple spaces and extra punctuation
        sanitized = re.sub(r'\s+', ' ', sanitized).strip()
        sanitized = re.sub(r'\s+([.,!?])', r'\1', sanitized)

        # Fix double "the" issues
        sanitized = re.sub(r'\bthe\s+the\b', 'the', sanitized, flags=re.IGNORECASE)

        # Fix duplicated words like "side side"
        sanitized = re.sub(r'\b(\w+)\s+\1\b', r'\1', sanitized)

        return sanitized
