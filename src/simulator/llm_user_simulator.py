"""
LLM-Driven User Simulator for M3Bench
======================================

Core simulator that uses an LLM to drive multi-turn conversations
with a target VLM model.

重构版本:
- 使用 ContextBuilder 统一打包上下文
- 使用 SimulatorState 管理状态机
- 使用 EntityExtractor 提取实体（可配置 simple/llm）
- 核心模型输出 JSON（query + reasoning + evaluation）
- 规则控制长度后缀
"""

import json
import re
from typing import Dict, List, Any, Optional, Literal
from pathlib import Path
import logging
from datetime import datetime

from .llm_client import LLMClient
from .memory_store import MemoryStore
from .task_config import TASK_CONFIGS, CORE_MODEL_SYSTEM_PROMPT
from .context_builder import ContextBuilder
from .simulator_state import SimulatorState, Phase, ResponseEvaluation
from .entity_extractor import EntityExtractor

logger = logging.getLogger(__name__)


# 长度控制后缀（规则决定）
LENGTH_SUFFIXES = {
    "short": "请简洁回答。",
    "medium": "",  # 不加后缀
    "long": "请详细说明。",
    "precise": "请给出精确的答案。"
}


class LLMUserSimulator:
    """
    LLM-driven user simulator for testing VLMs.

    Features:
    1. Uses a core LLM to decide actions and generate messages
    2. Maintains memory of conversation and key information
    3. Evaluates target model responses
    4. Supports multiple task types
    5. Generates detailed logs for analysis
    """

    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        max_turns_per_task: int = 10,
        verbose: bool = True,
        entity_extraction_mode: Literal["simple", "llm"] = "simple"
    ):
        """
        Initialize the simulator.

        Args:
            llm_client: LLM client for API calls
            max_turns_per_task: Maximum turns before forcing next task
            verbose: Print detailed progress
            entity_extraction_mode: Entity 提取模式 ("simple" 或 "llm")
        """
        self.llm_client = llm_client or LLMClient()
        self.max_turns_per_task = max_turns_per_task
        self.verbose = verbose

        # 核心组件
        self.memory = MemoryStore()
        self.context_builder = ContextBuilder()
        self.entity_extractor = EntityExtractor(
            extraction_mode=entity_extraction_mode,
            llm_client=self.llm_client if entity_extraction_mode == "llm" else None
        )

        # 状态
        self.current_task: Optional[Dict[str, Any]] = None
        self.state: Optional[SimulatorState] = None
        self.images_shown: List[str] = []
        self.image_descriptions: List[str] = []  # 图片描述缓存

        # Logging
        self.run_log: List[Dict[str, Any]] = []

    def _format_task_info(self, task: Dict[str, Any]) -> str:
        """Format task information for the core model prompt"""
        task_type = task.get("task_type", "unknown")
        task_config = TASK_CONFIGS.get(task_type, {})

        info_lines = [
            f"Task ID: {task.get('task_id', 'unknown')}",
            f"Task Type: {task_type} ({task_config.get('name_zh', '')})",
            f"",
            f"Task Description:",
            task_config.get("description", "No description available."),
            f"",
            f"Images: {len(task.get('images', []))} image(s)",
            f"Question: {task.get('question', 'N/A')}",
            f"Expected Answer: {task.get('answer', 'N/A')}",
            f"",
            f"Multi-turn Strategy:",
        ]

        strategy = task_config.get("multi_turn_strategy", {})
        for phase, desc in strategy.items():
            if not phase.startswith("noise"):
                info_lines.append(f"  - {phase}: {desc}")

        return "\n".join(info_lines)

    def _parse_core_model_response(self, response_text: str, reasoning_text: str = "") -> Dict[str, Any]:
        """
        Parse JSON response from core model.
        Handles o1-preview and similar models that may return reasoning_content with empty content.

        Args:
            response_text: The main content field from the response
            reasoning_text: The reasoning_content field (for o1-preview etc.)

        Returns:
            Parsed JSON dict with 'parse_error' key set to True on failure
        """
        # Determine which text to parse - prefer response_text, fallback to reasoning_text
        text_to_parse = response_text.strip() if response_text and response_text.strip() else reasoning_text.strip()

        if not text_to_parse:
            logger.warning("Both response_text and reasoning_text are empty")
            return {
                "action": "follow_up",
                "message_to_model": "Please tell me more.",
                "reasoning": "Empty response from model",
                "evaluation_of_last_response": "",
                "key_info_to_remember": [],
                "task_progress": "incomplete",
                "confidence": 0.5,
                "parse_error": True,
                "error_reason": "empty_response"
            }

        try:
            # Look for JSON block
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

            # Try parsing the whole response as JSON
            result = json.loads(text_to_parse)
            result["parse_error"] = False
            return result

        except json.JSONDecodeError as e:
            # Fallback: create a basic response
            logger.warning(f"Failed to parse core model response as JSON: {e}")
            logger.debug(f"Response text (first 200 chars): {response_text[:200] if response_text else 'empty'}")
            logger.debug(f"Reasoning text (first 200 chars): {reasoning_text[:200] if reasoning_text else 'empty'}")

            return {
                "action": "follow_up",
                "message_to_model": text_to_parse[:500] if text_to_parse else "Please continue.",
                "reasoning": "Failed to parse response",
                "evaluation_of_last_response": "",
                "key_info_to_remember": [],
                "task_progress": "incomplete",
                "confidence": 0.5,
                "parse_error": True,
                "error_reason": "json_decode_error"
            }

    def _build_core_model_prompt(self) -> List[Dict[str, str]]:
        """Build the prompt for the core model"""
        task_info = self._format_task_info(self.current_task) if self.current_task else "No task loaded."
        task_goal = TASK_CONFIGS.get(
            self.current_task.get("task_type", ""),
            {}
        ).get("goal", "Complete the task") if self.current_task else "No goal."

        memory_content = self.memory.get_formatted_memory()

        system_prompt = CORE_MODEL_SYSTEM_PROMPT.format(
            task_info=task_info,
            task_goal=task_goal,
            memory_content=memory_content
        )

        messages = [{"role": "system", "content": system_prompt}]

        # Add conversation history
        if self.memory.current_task and self.memory.current_task.turns:
            history_summary = []
            for turn in self.memory.current_task.turns[-5:]:  # Last 5 turns
                history_summary.append(f"Turn {turn.turn_id}:")
                history_summary.append(f"  You said: {turn.user_message[:100]}...")
                history_summary.append(f"  Model replied: {turn.model_response[:100]}...")
                if turn.evaluation:
                    history_summary.append(f"  Your evaluation: {turn.evaluation.get('score', 'N/A')}/5")

            messages.append({
                "role": "user",
                "content": f"Here's the recent conversation history:\n\n" + "\n".join(history_summary) +
                          "\n\nWhat should be your next action? Remember to respond in JSON format."
            })
        else:
            messages.append({
                "role": "user",
                "content": "This is the start of the task. What should be your first action? " +
                          "Remember to respond in JSON format."
            })

        return messages

    def start_task(self, task: Dict[str, Any]):
        """Start a new task"""
        self.current_task = task
        self.images_shown = []
        self.image_descriptions = []

        # 初始化状态机
        task_config = TASK_CONFIGS.get(task.get("task_type", ""), {})
        self.state = SimulatorState(
            task_id=task.get("task_id", "unknown"),
            task_type=task.get("task_type", "unknown"),
            phase=Phase.GROUNDING,
            current_turn=0,
            max_turns=self.max_turns_per_task,
            rationale=task_config.get("goal", task.get("question", "")),
            key_entities=self._extract_key_entities(task)
        )

        self.memory.start_task(
            task_id=task.get("task_id", "unknown"),
            task_type=task.get("task_type", "unknown"),
            expected_answer=task.get("answer", "")
        )

        if self.verbose:
            print(f"\n{'='*60}")
            print(f"Starting Task: {task.get('task_id')}")
            print(f"Type: {task.get('task_type')}")
            print(f"Question: {task.get('question')}")
            print(f"Expected: {task.get('answer')}")
            print(f"{'='*60}")

        # Log task start
        self.run_log.append({
            "event": "task_start",
            "task_id": task.get("task_id"),
            "task_type": task.get("task_type"),
            "question": task.get("question"),
            "expected_answer": task.get("answer"),
            "images": task.get("images", []),
            "timestamp": datetime.now().isoformat()
        })

    def _extract_key_entities(self, task: Dict[str, Any]) -> List[str]:
        """从任务中提取关键实体"""
        entities = []

        # 从问题和答案中提取
        question = task.get("question", "")
        answer = task.get("answer", "")
        text = f"{question} {answer}"

        # 简单提取：使用 entity extractor
        extracted = self.entity_extractor.extract(text)
        entities.extend(extracted.get("objects", [])[:5])

        return entities

    def step(self, vlm_response: Optional[str] = None) -> Dict[str, Any]:
        """
        Execute one turn of the simulation.

        Args:
            vlm_response: 待测 VLM 的上一轮回复（第一轮为 None）

        Returns:
            Dict with turn information including:
            - action: The action chosen
            - user_query: Message to send to target model (with length suffix)
            - raw_query: Original query without length suffix
            - reasoning: Core model's reasoning
            - evaluation: Evaluation of VLM's last response
            - should_continue: Whether to continue this task
        """
        if self.current_task is None:
            raise RuntimeError("No task started. Call start_task() first.")

        if self.state is None:
            raise RuntimeError("State not initialized. Call start_task() first.")

        turn_num = self.state.current_turn

        if self.verbose:
            print(f"\n--- Turn {turn_num} ---")
            print(f"Phase: {self.state.phase.value}")

        # Step 1: 提取实体（如果有 VLM 回复）
        entities = {}
        if vlm_response:
            entities = self.entity_extractor.extract(vlm_response)
            if self.verbose:
                print(f"Extracted entities: {list(entities.get('objects', []))[:5]}")

        # Step 2: 使用 ContextBuilder 构建上下文
        if vlm_response:
            context_prompt = self.context_builder.build(
                vlm_response=vlm_response,
                entities=entities,
                history_summary=self.memory.get_formatted_memory(),
                state=self.state,
                ground_truth=self.current_task,
                image_descriptions=self.image_descriptions
            )
        else:
            # 第一轮，没有 VLM 回复
            context_prompt = self.context_builder.build_initial_prompt(
                state=self.state,
                ground_truth=self.current_task,
                image_descriptions=self.image_descriptions
            )

        # Step 3: 调用核心模型
        core_response = self.llm_client.call_core_model(
            messages=[{"role": "user", "content": context_prompt}],
            max_tokens=1500
        )

        if not core_response.get("success"):
            logger.error(f"Core model call failed: {core_response.get('error')}")
            return {"error": "Core model failed", "should_continue": False}

        # Step 4: 解析核心模型输出 (支持 o1-preview 等模型的 reasoning_content)
        parsed = self._parse_core_model_response(
            response_text=core_response.get("content", ""),
            reasoning_text=core_response.get("reasoning_content", "")
        )

        action = parsed.get("action", "follow_up")
        raw_query = parsed.get("query", parsed.get("message_to_model", "请继续。"))
        task_progress = parsed.get("task_progress", "incomplete")
        reasoning = parsed.get("reasoning", "")

        # Step 5: 添加长度控制后缀（规则决定）
        length_suffix = self._get_length_suffix()
        user_query = f"{raw_query} {length_suffix}".strip() if length_suffix else raw_query

        if self.verbose:
            print(f"Action: {action}")
            print(f"Query: {user_query[:100]}...")
            print(f"Progress: {task_progress}")

        # Step 6: 记录评估（如果有）
        evaluation_data = parsed.get("evaluation", {})
        if evaluation_data and vlm_response:
            evaluation = ResponseEvaluation(
                reasoning_quality=evaluation_data.get("reasoning_quality", 0),
                reasoning_notes=evaluation_data.get("reasoning_notes", ""),
                memory_stability=evaluation_data.get("memory_stability", 0),
                memory_notes=evaluation_data.get("memory_notes", ""),
                confidence_calibration=evaluation_data.get("confidence_calibration", 0),
                confidence_notes=evaluation_data.get("confidence_notes", ""),
                overall_notes=evaluation_data.get("overall_notes", "")
            )
            self.state.record_evaluation(evaluation)

        # Step 7: 更新状态
        self.state.record_action(action)
        self.state.task_progress = task_progress

        # Log core model output (完整 JSON 存储)
        self.run_log.append({
            "event": "core_model_decision",
            "turn": turn_num,
            "phase": self.state.phase.value,
            "action": action,
            "raw_query": raw_query,
            "user_query": user_query,
            "reasoning": reasoning,
            "evaluation": evaluation_data,
            "task_progress": task_progress,
            "entities_extracted": entities,
            "key_observations": parsed.get("key_observations", []),
            "timestamp": datetime.now().isoformat()
        })

        # Step 8: 检查是否结束
        if action == "next_task" or task_progress in ["complete", "failed"]:
            self.state.mark_complete(task_progress, reasoning)
            self.memory.complete_task(
                status=task_progress,
                final_evaluation={"reasoning": reasoning}
            )
            return {
                "action": action,
                "user_query": user_query,
                "raw_query": raw_query,
                "reasoning": reasoning,
                "evaluation": evaluation_data,
                "should_continue": False,
                "task_status": task_progress
            }

        # Step 9: 推进轮次和阶段
        self.state.advance_turn()
        self.state.update_phase()

        # 检查是否达到最大轮数
        should_continue = not self.state.should_end

        return {
            "action": action,
            "user_query": user_query,
            "raw_query": raw_query,
            "reasoning": reasoning,
            "evaluation": evaluation_data,
            "should_continue": should_continue,
            "phase": self.state.phase.value,
            "turn": turn_num
        }

    def _get_length_suffix(self) -> str:
        """根据 phase 和 action 规则决定长度后缀"""
        if self.state is None:
            return ""

        phase = self.state.phase

        # Final 阶段需要精确答案
        if phase == Phase.FINAL:
            return LENGTH_SUFFIXES["precise"]

        # 前期不加后缀
        if self.state.current_turn <= 1:
            return LENGTH_SUFFIXES["medium"]

        # 压力测试阶段简洁回答
        if phase == Phase.STRESS_TEST:
            return LENGTH_SUFFIXES["short"]

        return LENGTH_SUFFIXES["medium"]

    def step_with_target(self, vlm_response: Optional[str] = None) -> Dict[str, Any]:
        """
        执行一轮模拟，包括调用待测 VLM

        这是完整的一轮：生成 query -> 调用 VLM -> 返回结果
        """
        # 生成 query
        result = self.step(vlm_response)

        if not result.get("should_continue", False) or "error" in result:
            return result

        # 确定要发送的图片
        images_to_send = self._get_images_to_send(result.get("action", ""))

        # 调用待测 VLM
        target_messages = self.memory.get_conversation_history(n_turns=5)
        target_messages.append({"role": "user", "content": result["user_query"]})

        target_response = self.llm_client.call_target_model(
            messages=target_messages,
            images=images_to_send if images_to_send else None,
            max_tokens=800
        )

        if not target_response.get("success"):
            logger.error(f"Target model call failed: {target_response.get('error')}")
            target_content = "[Target model error]"
        else:
            target_content = target_response.get("content", "")

        if self.verbose:
            print(f"Target response: {target_content[:200]}...")

        # 存储到 memory
        self.memory.add_turn(
            action=result.get("action", "follow_up"),
            user_message=result["user_query"],
            model_response=target_content,
            evaluation=result.get("evaluation", {}),
            key_info=result.get("key_observations", [])
        )

        # Log target model response
        self.run_log.append({
            "event": "target_model_response",
            "turn": result.get("turn", 0),
            "images_sent": images_to_send,
            "target_response": target_content,
            "timestamp": datetime.now().isoformat()
        })

        result["target_response"] = target_content
        result["images_sent"] = images_to_send

        return result

    def _get_images_to_send(self, action: str) -> List[str]:
        """确定要发送给待测 VLM 的图片"""
        images_to_send = []
        task_images = self.current_task.get("images", [])

        if action == "guidance" and task_images:
            # For guidance, show next image if not all shown
            if len(self.images_shown) < len(task_images):
                next_img_idx = len(self.images_shown)
                next_img = task_images[next_img_idx]

                # Construct full path
                img_path = Path("generated_tasks_v2/run_12") / next_img
                if img_path.exists():
                    images_to_send.append(str(img_path))
                    self.images_shown.append(next_img)

                    # 记录图片描述（如果有）
                    self.image_descriptions.append(f"Image {next_img_idx}: {next_img}")

        return images_to_send

    def run_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run a complete task from start to finish.

        Returns:
            Task result summary
        """
        self.start_task(task)

        turns_executed = 0
        vlm_response = None  # 第一轮没有 VLM 回复

        while True:
            result = self.step_with_target(vlm_response)
            turns_executed += 1

            if not result.get("should_continue", False) or "error" in result:
                break

            # 下一轮的 VLM 回复
            vlm_response = result.get("target_response")

        # Final summary
        summary = {
            "task_id": task.get("task_id"),
            "task_type": task.get("task_type"),
            "turns_executed": turns_executed,
            "final_status": result.get("task_status", "unknown"),
            "statistics": self.memory.get_statistics(),
            "cumulative_scores": self.state.cumulative_scores if self.state else {}
        }

        if self.verbose:
            print(f"\n{'='*60}")
            print(f"Task Completed: {task.get('task_id')}")
            print(f"Turns: {turns_executed}")
            print(f"Status: {result.get('task_status', 'unknown')}")
            if self.state:
                print(f"Cumulative Scores: {self.state.cumulative_scores}")
            print(f"{'='*60}")

        return summary

    def run_multiple_tasks(self, tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Run multiple tasks in sequence"""
        results = []
        for i, task in enumerate(tasks):
            if self.verbose:
                print(f"\n\n{'#'*60}")
                print(f"# Task {i+1}/{len(tasks)}")
                print(f"{'#'*60}")

            result = self.run_task(task)
            results.append(result)

        return results

    def export_log(self, output_dir: str) -> str:
        """
        Export run log to files.

        Returns:
            Path to the output directory
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save detailed run log
        log_file = output_path / f"run_log_{timestamp}.json"
        with open(log_file, "w", encoding="utf-8") as f:
            json.dump(self.run_log, f, ensure_ascii=False, indent=2)

        # Save memory state
        memory_file = output_path / f"memory_state_{timestamp}.json"
        self.memory.export_to_json(str(memory_file))

        # Save summary
        summary = {
            "run_timestamp": timestamp,
            "total_events": len(self.run_log),
            "statistics": self.memory.get_statistics(),
            "files": {
                "run_log": str(log_file),
                "memory_state": str(memory_file)
            }
        }

        summary_file = output_path / f"summary_{timestamp}.json"
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        if self.verbose:
            print(f"\nLogs exported to: {output_path}")
            print(f"  - Run log: {log_file.name}")
            print(f"  - Memory: {memory_file.name}")
            print(f"  - Summary: {summary_file.name}")

        return str(output_path)


# Test
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("Testing LLMUserSimulator...")

    # Create simulator
    simulator = LLMUserSimulator(verbose=True)

    # Test with a simple task
    test_task = {
        "task_id": "test_001",
        "task_type": "attribute_comparison",
        "question": "Which image shows the most people?",
        "answer": "Image 0 with 5 people",
        "images": ["images/test1.jpg", "images/test2.jpg"]
    }

    # Just test initialization
    simulator.start_task(test_task)
    print("\nSimulator initialized successfully!")
    print(f"Memory state: {simulator.memory.get_formatted_memory()}")
