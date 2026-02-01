"""
Mantis Dataset Loader for M3Bench
==================================

Loads data from TIGER-Lab/Mantis-Instruct dataset (parquet format).
Supports multiple subsets:
- visual_story_telling: Multi-image storytelling with temporal sequences
- spot-the-diff: Image comparison tasks
- nlvr2: Visual reasoning with image pairs
- contrastive_caption: Contrastive image-text pairs
- nextqa / star: Video QA with temporal understanding

Key features:
1. Parquet file loading
2. Image path resolution (F: drive)
3. Conversation extraction for multi-turn tasks
4. Temporal state tracking groundtruth generation
"""

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import random

logger = logging.getLogger(__name__)

# Try to import pandas and pyarrow for parquet support
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    logger.warning("pandas not available - parquet loading will be limited")


class MantisLoader:
    """Load and parse Mantis-Instruct dataset."""

    # Dataset root on F: drive
    DEFAULT_ROOT = "F:/datasets/TIGER-Lab_Mantis-Instruct"

    # Subset configurations
    SUBSET_CONFIG = {
        "visual_story_telling": {
            "image_dir": "image",
            "description": "Multi-image storytelling with temporal sequences",
            "task_types": ["temporal_state", "story_comprehension", "image_sequence"]
        },
        "spot-the-diff": {
            "image_dir": "images",
            "description": "Image comparison - find differences",
            "task_types": ["difference_detection", "change_tracking"]
        },
        "nlvr2": {
            "image_dir": "images",
            "description": "Visual reasoning with image pairs",
            "task_types": ["visual_reasoning", "comparison"]
        },
        "contrastive_caption": {
            "image_dir": "images",
            "description": "Contrastive image-text matching",
            "task_types": ["caption_matching", "image_text_alignment"]
        },
        "nextqa": {
            "image_dir": "images",
            "description": "Video QA with temporal understanding",
            "task_types": ["temporal_reasoning", "video_qa"]
        },
        "star": {
            "image_dir": "images",
            "description": "Situated reasoning about actions and relations",
            "task_types": ["action_reasoning", "temporal_state"]
        }
    }

    def __init__(self, data_root: str = None):
        """
        Initialize Mantis loader.

        Args:
            data_root: Root directory of Mantis dataset (default: F: drive location)
        """
        self.data_root = Path(data_root or self.DEFAULT_ROOT)
        self._cache = {}

    def load_subset(
        self,
        subset: str,
        split: str = "train",
        max_samples: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Load a subset of Mantis dataset.

        Args:
            subset: Subset name (e.g., "visual_story_telling")
            split: Data split ("train", "val")
            max_samples: Maximum samples to load

        Returns:
            List of samples in unified format
        """
        if not HAS_PANDAS:
            raise ImportError("pandas is required to load parquet files")

        subset_dir = self.data_root / subset

        if not subset_dir.exists():
            logger.error(f"Subset directory not found: {subset_dir}")
            return []

        # Find parquet file
        parquet_files = list(subset_dir.glob(f"{split}*.parquet"))
        if not parquet_files:
            parquet_files = list(subset_dir.glob("*.parquet"))

        if not parquet_files:
            logger.error(f"No parquet files found in {subset_dir}")
            return []

        # Load parquet
        parquet_file = parquet_files[0]
        logger.info(f"Loading {parquet_file}")

        df = pd.read_parquet(parquet_file)

        if max_samples:
            df = df.head(max_samples)

        # Get image directory
        config = self.SUBSET_CONFIG.get(subset, {"image_dir": "image"})
        image_dir = subset_dir / config["image_dir"]

        # Convert to list of dicts
        samples = []
        for idx, row in df.iterrows():
            sample = self._parse_row(row, subset, image_dir)
            if sample:
                sample['sample_idx'] = idx
                samples.append(sample)

        logger.info(f"Loaded {len(samples)} samples from {subset}")
        return samples

    def _parse_row(
        self,
        row: Any,
        subset: str,
        image_dir: Path
    ) -> Optional[Dict[str, Any]]:
        """Parse a single row from parquet."""
        try:
            sample = {
                "dataset_id": f"mantis_{subset}",
                "sample_id": row.get("id", str(row.name)),
                "source": row.get("source", subset)
            }

            # Parse images
            images_data = row.get("images", [])
            image_paths = []
            for img in images_data:
                if isinstance(img, dict):
                    img_path = img.get("path", "")
                else:
                    img_path = str(img)

                if img_path:
                    # Resolve full path
                    full_path = image_dir / img_path
                    image_paths.append(str(full_path))

            sample["images"] = image_paths
            sample["num_images"] = len(image_paths)

            # Parse conversation
            conversation = row.get("conversation", [])
            if isinstance(conversation, str):
                conversation = json.loads(conversation)

            sample["conversation"] = conversation
            sample["num_turns"] = len(conversation) // 2  # user + assistant pairs

            # Extract QA pairs
            qa_pairs = []
            for i in range(0, len(conversation) - 1, 2):
                if i + 1 < len(conversation):
                    user_turn = conversation[i]
                    assistant_turn = conversation[i + 1]
                    if user_turn.get("role") == "user" and assistant_turn.get("role") == "assistant":
                        qa_pairs.append({
                            "question": user_turn.get("content", ""),
                            "answer": assistant_turn.get("content", "")
                        })

            sample["qa_pairs"] = qa_pairs

            return sample

        except Exception as e:
            logger.warning(f"Error parsing row: {e}")
            return None

    def get_available_subsets(self) -> List[str]:
        """Get list of available subsets."""
        available = []
        for subset in self.SUBSET_CONFIG.keys():
            subset_dir = self.data_root / subset
            if subset_dir.exists():
                available.append(subset)
        return available


class TemporalStateTaskGenerator:
    """
    Generate Temporal State Maintenance tasks from Mantis data.

    Task Definition:
    ================
    The model views a sequence of images (e.g., story panels) and must
    track state changes across the sequence. The simulator will:
    1. Show images progressively
    2. Inject misleading information about previous states
    3. Ask the model to recall/verify earlier states
    4. Test if the model maintains accurate temporal memory

    Groundtruth Format:
    {
        "task_id": str,
        "task_type": "temporal_state_tracking",
        "images": [path1, path2, ...],
        "image_sequence_info": [
            {"idx": 0, "description": "...", "key_entities": [...], "state": {...}},
            {"idx": 1, "description": "...", "key_entities": [...], "state": {...}},
        ],
        "state_transitions": [
            {"from_idx": 0, "to_idx": 1, "changes": [...], "unchanged": [...]},
        ],
        "verification_questions": [
            {"question": "...", "answer": "...", "refers_to_idx": 0, "type": "recall"},
            {"question": "...", "answer": "...", "refers_to_idx": [0,1], "type": "comparison"},
        ],
        "misleading_candidates": [
            {"claim": "false claim about image 0", "truth": "actual truth", "target_idx": 0},
        ]
    }
    """

    def __init__(self, mantis_loader: MantisLoader):
        """
        Initialize generator.

        Args:
            mantis_loader: MantisLoader instance
        """
        self.loader = mantis_loader

    def generate_from_visual_story_telling(
        self,
        num_tasks: int = 50,
        min_images: int = 3,
        max_images: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Generate temporal state tasks from visual_story_telling subset.

        Args:
            num_tasks: Number of tasks to generate
            min_images: Minimum images per task
            max_images: Maximum images per task

        Returns:
            List of task dictionaries
        """
        # Load source data
        source_data = self.loader.load_subset(
            "visual_story_telling",
            max_samples=num_tasks * 3  # Load extra for filtering
        )

        # Filter samples with enough images
        valid_samples = [
            s for s in source_data
            if min_images <= s["num_images"] <= max_images
        ]

        if len(valid_samples) < num_tasks:
            logger.warning(f"Only {len(valid_samples)} valid samples, requested {num_tasks}")

        tasks = []
        for i, sample in enumerate(valid_samples[:num_tasks]):
            task = self._generate_task_from_sample(sample, i)
            if task:
                tasks.append(task)

        logger.info(f"Generated {len(tasks)} temporal state tasks")
        return tasks

    def _generate_task_from_sample(
        self,
        sample: Dict[str, Any],
        task_idx: int
    ) -> Optional[Dict[str, Any]]:
        """Generate a single task from a sample."""
        try:
            images = sample["images"]
            qa_pairs = sample["qa_pairs"]

            # Extract key information from QA pairs
            image_info = []
            for idx in range(len(images)):
                info = {
                    "idx": idx,
                    "image_path": images[idx],
                    "description": "",
                    "key_entities": [],
                    "state": {}
                }

                # Try to extract info from QA if available
                if idx < len(qa_pairs):
                    qa = qa_pairs[idx]
                    info["description"] = qa["answer"][:200] if qa["answer"] else ""
                    # Extract entities (simplified - in production use NER)
                    info["key_entities"] = self._extract_entities(qa["answer"])

                image_info.append(info)

            # Generate state transitions
            transitions = []
            for i in range(len(images) - 1):
                transitions.append({
                    "from_idx": i,
                    "to_idx": i + 1,
                    "changes": [],  # Would be filled by deeper analysis
                    "unchanged": []
                })

            # Generate verification questions
            verification_questions = self._generate_verification_questions(
                images, image_info, qa_pairs
            )

            # Generate misleading candidates
            misleading = self._generate_misleading_candidates(image_info, qa_pairs)

            # Build the main question
            main_question = self._build_main_question(sample, qa_pairs)

            # 生成全局唯一的task_id
            rand_chars = ''.join(random.choices('abcdefghijklmnopqrstuvwxyz0123456789', k=4))
            unique_task_id = f"ts_mantis_{int(time.time())}_{rand_chars}_{task_idx:03d}"

            return {
                "task_id": unique_task_id,
                "task_type": "temporal_state_tracking",
                "question": main_question["question"],
                "answer": main_question["answer"],
                "images": images,
                "image_sequence_info": image_info,
                "state_transitions": transitions,
                "verification_questions": verification_questions,
                "misleading_candidates": misleading,
                "metadata": {
                    "source_id": sample["sample_id"],
                    "source_dataset": "mantis_visual_story_telling",
                    "num_images": len(images),
                    "original_qa_pairs": qa_pairs
                }
            }

        except Exception as e:
            logger.warning(f"Error generating task: {e}")
            return None

    def _extract_entities(self, text: str) -> List[str]:
        """Extract key entities from text (simplified)."""
        # In production, use NER or LLM
        # Here we do simple keyword extraction
        import re
        # Find capitalized words (likely proper nouns/entities)
        entities = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
        # Also find quoted terms
        quoted = re.findall(r'"([^"]+)"', text)
        entities.extend(quoted)
        return list(set(entities))[:5]  # Limit to 5

    def _generate_verification_questions(
        self,
        images: List[str],
        image_info: List[Dict],
        qa_pairs: List[Dict]
    ) -> List[Dict[str, Any]]:
        """Generate verification questions for temporal state checking."""
        questions = []

        # Recall questions (ask about earlier states)
        if len(images) >= 2:
            questions.append({
                "question": "What was shown in the first image of this sequence?",
                "answer": image_info[0]["description"] if image_info else "the beginning of the story",
                "refers_to_idx": 0,
                "type": "recall"
            })

        # Comparison questions
        if len(images) >= 3:
            questions.append({
                "question": "How does the scene change from the first to the last image?",
                "answer": f"The sequence progresses from: {image_info[0].get('description', '')[:50]}... to: {image_info[-1].get('description', '')[:50]}...",
                "refers_to_idx": [0, len(images) - 1],
                "type": "comparison"
            })

        # Sequence questions
        questions.append({
            "question": f"How many images are in this sequence?",
            "answer": str(len(images)),
            "refers_to_idx": list(range(len(images))),
            "type": "counting"
        })

        return questions

    def _generate_misleading_candidates(
        self,
        image_info: List[Dict],
        qa_pairs: List[Dict]
    ) -> List[Dict[str, Any]]:
        """Generate misleading statements for robustness testing."""
        misleading = []

        if len(image_info) >= 2:
            # Swap information between images
            misleading.append({
                "claim": f"The first image shows {image_info[-1].get('description', 'the ending')[:50]}",
                "truth": f"The first image actually shows {image_info[0].get('description', 'the beginning')[:50]}",
                "target_idx": 0,
                "type": "temporal_swap"
            })

            # Fabricate non-existent content
            misleading.append({
                "claim": "I noticed there's a car in the second image",
                "truth": "The claim about a car may not be accurate - verify with the actual image",
                "target_idx": 1,
                "type": "fabrication"
            })

        return misleading

    def _build_main_question(
        self,
        sample: Dict[str, Any],
        qa_pairs: List[Dict]
    ) -> Dict[str, str]:
        """Build the main task question."""
        # Use the first QA pair if available
        if qa_pairs:
            return {
                "question": qa_pairs[0]["question"],
                "answer": qa_pairs[0]["answer"]
            }

        # Fallback
        return {
            "question": "Describe the sequence of events shown in these images.",
            "answer": "This is a visual story sequence."
        }

    def export_tasks(
        self,
        tasks: List[Dict[str, Any]],
        output_file: str
    ):
        """Export tasks to JSONL file."""
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            for task in tasks:
                f.write(json.dumps(task, ensure_ascii=False) + '\n')

        logger.info(f"Exported {len(tasks)} tasks to {output_path}")


def generate_mantis_tasks(
    output_dir: str = "generated_tasks_mantis",
    num_tasks: int = 50,
    data_root: str = None
) -> str:
    """
    Convenience function to generate Mantis tasks.

    Args:
        output_dir: Output directory for generated tasks
        num_tasks: Number of tasks to generate
        data_root: Mantis data root (default: F: drive)

    Returns:
        Path to generated JSONL file
    """
    loader = MantisLoader(data_root)
    generator = TemporalStateTaskGenerator(loader)

    tasks = generator.generate_from_visual_story_telling(num_tasks=num_tasks)

    output_file = Path(output_dir) / "temporal_state_mantis.jsonl"
    generator.export_tasks(tasks, str(output_file))

    return str(output_file)


# Test
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("Testing Mantis Loader...")

    loader = MantisLoader()
    print(f"Available subsets: {loader.get_available_subsets()}")

    # Test loading
    samples = loader.load_subset("visual_story_telling", max_samples=5)
    print(f"\nLoaded {len(samples)} samples")

    if samples:
        sample = samples[0]
        print(f"\nSample structure:")
        print(f"  ID: {sample['sample_id']}")
        print(f"  Images: {sample['num_images']}")
        print(f"  Turns: {sample['num_turns']}")
        if sample['qa_pairs']:
            print(f"  First Q: {sample['qa_pairs'][0]['question'][:80]}...")
            print(f"  First A: {sample['qa_pairs'][0]['answer'][:80]}...")

    # Test task generation
    print("\nTesting Task Generation...")
    generator = TemporalStateTaskGenerator(loader)
    tasks = generator.generate_from_visual_story_telling(num_tasks=3)

    if tasks:
        task = tasks[0]
        print(f"\nGenerated task:")
        print(f"  Task ID: {task['task_id']}")
        print(f"  Question: {task['question'][:80]}...")
        print(f"  Images: {len(task['images'])}")
        print(f"  Verification Qs: {len(task['verification_questions'])}")
        print(f"  Misleading candidates: {len(task['misleading_candidates'])}")
