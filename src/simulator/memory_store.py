"""
Memory Store for M3Bench User Simulator
========================================

Stores and manages key information extracted during conversations,
including evaluations, key observations, and conversation history.
"""

from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import dataclass, field
import json


@dataclass
class TurnRecord:
    """Record of a single conversation turn"""
    turn_id: int
    action: str
    user_message: str
    model_response: str
    evaluation: Dict[str, Any] = field(default_factory=dict)
    key_info: List[str] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "turn_id": self.turn_id,
            "action": self.action,
            "user_message": self.user_message,
            "model_response": self.model_response,
            "evaluation": self.evaluation,
            "key_info": self.key_info,
            "timestamp": self.timestamp
        }


@dataclass
class TaskRecord:
    """Record of a complete task execution"""
    task_id: str
    task_type: str
    expected_answer: str
    turns: List[TurnRecord] = field(default_factory=list)
    final_evaluation: Dict[str, Any] = field(default_factory=dict)
    status: str = "incomplete"  # incomplete, complete, failed
    start_time: str = field(default_factory=lambda: datetime.now().isoformat())
    end_time: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "task_type": self.task_type,
            "expected_answer": self.expected_answer,
            "turns": [t.to_dict() for t in self.turns],
            "final_evaluation": self.final_evaluation,
            "status": self.status,
            "start_time": self.start_time,
            "end_time": self.end_time
        }


class MemoryStore:
    """
    Memory store for conversation history and key information.

    Features:
    1. Store turn-by-turn conversation records
    2. Store extracted key information
    3. Store evaluations
    4. Provide formatted memory for prompts
    5. Support task-level aggregation
    """

    def __init__(self):
        """Initialize memory store"""
        self.current_task: Optional[TaskRecord] = None
        self.completed_tasks: List[TaskRecord] = []
        self.global_key_info: List[str] = []

        # Extracted entities/information
        self.extracted_entities: Dict[str, Any] = {
            "objects": [],
            "attributes": {},
            "counts": {},
            "relations": []
        }

    def start_task(self, task_id: str, task_type: str, expected_answer: str):
        """Start a new task"""
        if self.current_task is not None:
            # Save current task as completed/failed
            self.current_task.status = "abandoned"
            self.current_task.end_time = datetime.now().isoformat()
            self.completed_tasks.append(self.current_task)

        self.current_task = TaskRecord(
            task_id=task_id,
            task_type=task_type,
            expected_answer=expected_answer
        )

        # Clear task-specific memory
        self.extracted_entities = {
            "objects": [],
            "attributes": {},
            "counts": {},
            "relations": []
        }

    def add_turn(
        self,
        action: str,
        user_message: str,
        model_response: str,
        evaluation: Optional[Dict[str, Any]] = None,
        key_info: Optional[List[str]] = None
    ):
        """Add a conversation turn"""
        if self.current_task is None:
            raise RuntimeError("No active task. Call start_task() first.")

        turn = TurnRecord(
            turn_id=len(self.current_task.turns),
            action=action,
            user_message=user_message,
            model_response=model_response,
            evaluation=evaluation or {},
            key_info=key_info or []
        )

        self.current_task.turns.append(turn)

        # Add key info to global store
        if key_info:
            self.global_key_info.extend(key_info)

    def add_key_info(self, info: str):
        """Add a piece of key information"""
        if info not in self.global_key_info:
            self.global_key_info.append(info)

    def add_extracted_entity(self, entity_type: str, entity: Any):
        """Add an extracted entity"""
        if entity_type == "objects":
            if entity not in self.extracted_entities["objects"]:
                self.extracted_entities["objects"].append(entity)
        elif entity_type == "attributes":
            obj, attrs = entity
            if obj not in self.extracted_entities["attributes"]:
                self.extracted_entities["attributes"][obj] = {}
            self.extracted_entities["attributes"][obj].update(attrs)
        elif entity_type == "counts":
            obj, count = entity
            self.extracted_entities["counts"][obj] = count
        elif entity_type == "relations":
            self.extracted_entities["relations"].append(entity)

    def complete_task(
        self,
        status: str = "complete",
        final_evaluation: Optional[Dict[str, Any]] = None
    ):
        """Complete the current task"""
        if self.current_task is None:
            return

        self.current_task.status = status
        self.current_task.end_time = datetime.now().isoformat()
        if final_evaluation:
            self.current_task.final_evaluation = final_evaluation

        self.completed_tasks.append(self.current_task)
        self.current_task = None

    def get_conversation_history(self, n_turns: Optional[int] = None) -> List[Dict[str, str]]:
        """Get conversation history as list of messages"""
        if self.current_task is None:
            return []

        turns = self.current_task.turns
        if n_turns:
            turns = turns[-n_turns:]

        history = []
        for turn in turns:
            history.append({"role": "user", "content": turn.user_message})
            history.append({"role": "assistant", "content": turn.model_response})

        return history

    def get_formatted_memory(self) -> str:
        """Get formatted memory content for prompts"""
        lines = []

        # Current task info
        if self.current_task:
            lines.append(f"Current Task: {self.current_task.task_id} ({self.current_task.task_type})")
            lines.append(f"Turns so far: {len(self.current_task.turns)}")
            lines.append("")

        # Key information
        if self.global_key_info:
            lines.append("Key Information Remembered:")
            for i, info in enumerate(self.global_key_info[-10:], 1):  # Last 10
                lines.append(f"  {i}. {info}")
            lines.append("")

        # Extracted entities
        if self.extracted_entities["objects"]:
            lines.append(f"Objects identified: {', '.join(self.extracted_entities['objects'])}")

        if self.extracted_entities["counts"]:
            counts_str = ", ".join(f"{k}: {v}" for k, v in self.extracted_entities["counts"].items())
            lines.append(f"Counts: {counts_str}")

        if self.extracted_entities["relations"]:
            lines.append(f"Relations: {len(self.extracted_entities['relations'])} recorded")

        # Recent evaluations
        if self.current_task and self.current_task.turns:
            recent_evals = [t.evaluation for t in self.current_task.turns[-3:] if t.evaluation]
            if recent_evals:
                lines.append("")
                lines.append("Recent Evaluations:")
                for eval_dict in recent_evals:
                    if "score" in eval_dict:
                        lines.append(f"  Score: {eval_dict['score']}/5")
                    if "reasoning" in eval_dict:
                        lines.append(f"  Note: {eval_dict['reasoning'][:100]}...")

        return "\n".join(lines) if lines else "No memory recorded yet."

    def get_last_model_response(self) -> Optional[str]:
        """Get the last model response"""
        if self.current_task and self.current_task.turns:
            return self.current_task.turns[-1].model_response
        return None

    def get_last_evaluation(self) -> Optional[Dict[str, Any]]:
        """Get the last evaluation"""
        if self.current_task and self.current_task.turns:
            return self.current_task.turns[-1].evaluation
        return None

    def export_to_dict(self) -> Dict[str, Any]:
        """Export all memory to dictionary"""
        return {
            "current_task": self.current_task.to_dict() if self.current_task else None,
            "completed_tasks": [t.to_dict() for t in self.completed_tasks],
            "global_key_info": self.global_key_info,
            "extracted_entities": self.extracted_entities
        }

    def export_to_json(self, filepath: str):
        """Export all memory to JSON file"""
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.export_to_dict(), f, ensure_ascii=False, indent=2)

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the conversation"""
        stats = {
            "total_tasks": len(self.completed_tasks) + (1 if self.current_task else 0),
            "completed_tasks": sum(1 for t in self.completed_tasks if t.status == "complete"),
            "failed_tasks": sum(1 for t in self.completed_tasks if t.status == "failed"),
            "total_turns": sum(len(t.turns) for t in self.completed_tasks),
            "key_info_count": len(self.global_key_info)
        }

        if self.current_task:
            stats["total_turns"] += len(self.current_task.turns)
            stats["current_task_turns"] = len(self.current_task.turns)

        # Average evaluation scores
        all_evals = []
        for task in self.completed_tasks + ([self.current_task] if self.current_task else []):
            for turn in task.turns:
                if turn.evaluation and "score" in turn.evaluation:
                    all_evals.append(turn.evaluation["score"])

        if all_evals:
            stats["average_score"] = sum(all_evals) / len(all_evals)

        return stats


# Test
if __name__ == "__main__":
    print("Testing MemoryStore...")

    memory = MemoryStore()

    # Start a task
    memory.start_task("task_001", "attribute_comparison", "Image 1 with 5 people")

    # Add turns
    memory.add_turn(
        action="guidance",
        user_message="Let's look at the first image. What do you see?",
        model_response="I see several people standing in a park.",
        evaluation={"score": 4, "reasoning": "Good observation"},
        key_info=["multiple people in image 1", "location: park"]
    )

    memory.add_turn(
        action="follow_up",
        user_message="How many people exactly?",
        model_response="I count 5 people in the image.",
        evaluation={"score": 5, "reasoning": "Correct count"},
        key_info=["Image 1: 5 people"]
    )

    memory.add_extracted_entity("objects", "person")
    memory.add_extracted_entity("counts", ("person", 5))

    # Get formatted memory
    print("\n--- Formatted Memory ---")
    print(memory.get_formatted_memory())

    # Get statistics
    print("\n--- Statistics ---")
    print(memory.get_statistics())

    # Export
    print("\n--- Export ---")
    print(json.dumps(memory.export_to_dict(), indent=2, ensure_ascii=False))
