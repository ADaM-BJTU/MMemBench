"""
DataGenerator V2: Config-driven task generation for M3Bench
===========================================================
Unified generator that uses dataset_configs.yaml to drive task generation.

Supports:
1. Attribute Bridge Reasoning (ABR)
2. Attribute Comparison (AC) - NEW!
3. Visual Noise Filtering (VNF)
4. Relation Comparison (RC)
"""

import random
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path

from .config_loader import ConfigLoader
from .task_generators import (
    AttributeComparisonGenerator,
    EnhancedVNFGenerator
)
# 窗口2新增: 基于Rationale的ABR生成器
from .rationale_based_generator import RationaleBasedABRGenerator

logger = logging.getLogger(__name__)


class DataGeneratorV2:
    """Config-driven data generator for M3Bench tasks."""

    def __init__(self, data_loader, config_file: str = "dataset_configs.yaml"):
        """
        Initialize DataGenerator with config support.

        Args:
            data_loader: DataLoader instance
            config_file: Path to dataset configuration file
        """
        self.loader = data_loader
        self.config = ConfigLoader(config_file)

        # Task generator registry
        self.task_generators = {
            'attribute_comparison': self._generate_attribute_comparison,
            'visual_noise_filtering': self._generate_visual_noise_filtering,
            'attribute_bridge_reasoning': self._generate_attribute_bridge_reasoning,
            'relation_comparison': self._generate_relation_comparison,
            # 窗口2新增: 基于Rationale的ABR
            'rationale_based_abr': self._generate_rationale_based_abr
        }

    def generate_task(self,
                     task_type: str,
                     source_dataset: str,
                     num_samples: int = 100,
                     split: str = "train",
                     **kwargs) -> List[Dict[str, Any]]:
        """
        Generate tasks using dataset configuration.

        Args:
            task_type: Task type (e.g., 'attribute_comparison')
            source_dataset: Dataset ID (e.g., 'mscoco14')
            num_samples: Number of samples to generate
            split: Data split to use
            **kwargs: Override config parameters

        Returns:
            List of generated task samples
        """
        # Validate dataset and task
        dataset_config = self.config.get_dataset_config(source_dataset)
        if not dataset_config:
            raise ValueError(f"Unknown dataset: {source_dataset}")

        if not dataset_config.supports_task(task_type):
            logger.warning(f"{source_dataset} does not support {task_type}")
            return []

        if not dataset_config.is_task_enabled(task_type):
            logger.warning(f"{task_type} is disabled for {source_dataset}")
            return []

        # Get task configuration
        task_config = dataset_config.get_task_config(task_type)
        task_config.update(kwargs)  # Allow runtime overrides

        # Get templates
        templates = {
            'question_templates': self.config.get_task_template(task_type, 'question_templates'),
            'answer_template': self.config.get_task_template(task_type, 'answer_template')
        }

        # Load source data
        logger.info(f"Loading {source_dataset} data (split={split})...")
        source_data = self.loader.load_dataset(
            source_dataset,
            split=split,
            max_samples=num_samples * 10  # Load extra for filtering
        )

        if not source_data:
            logger.warning(f"No data loaded from {source_dataset}")
            return []

        # Generate tasks
        generator_func = self.task_generators.get(task_type)
        if not generator_func:
            raise ValueError(f"No generator for task type: {task_type}")

        logger.info(f"Generating {task_type} tasks from {source_dataset}...")
        tasks = generator_func(
            source_dataset,
            source_data,
            num_samples,
            task_config,
            templates
        )

        # Apply quality control
        tasks = self._apply_quality_control(tasks, task_type)

        logger.info(f"Generated {len(tasks)} {task_type} tasks from {source_dataset}")
        return tasks

    def generate_all_tasks_for_dataset(self,
                                      dataset_id: str,
                                      num_samples_per_task: int = 10,
                                      split: str = "train") -> Dict[str, List[Dict]]:
        """
        Generate all enabled tasks for a dataset.

        Args:
            dataset_id: Dataset ID
            num_samples_per_task: Samples to generate per task type
            split: Data split

        Returns:
            Dictionary mapping task_type -> List[task_samples]
        """
        dataset_config = self.config.get_dataset_config(dataset_id)
        if not dataset_config:
            logger.error(f"Unknown dataset: {dataset_id}")
            return {}

        all_tasks = {}

        for task_type in dataset_config.supported_tasks:
            if dataset_config.is_task_enabled(task_type):
                try:
                    tasks = self.generate_task(
                        task_type=task_type,
                        source_dataset=dataset_id,
                        num_samples=num_samples_per_task,
                        split=split
                    )
                    if tasks:
                        all_tasks[task_type] = tasks
                except Exception as e:
                    logger.error(f"Failed to generate {task_type} for {dataset_id}: {e}")
                    import traceback
                    traceback.print_exc()

        return all_tasks

    # ==================== Task Generators ====================

    def _generate_attribute_comparison(self,
                                      source_dataset: str,
                                      source_data: List[Dict],
                                      num_samples: int,
                                      task_config: Dict[str, Any],
                                      templates: Dict[str, Any]) -> List[Dict]:
        """Generate Attribute Comparison tasks."""

        if source_dataset == "mscoco14":
            return AttributeComparisonGenerator.generate_from_mscoco(
                source_data, num_samples, task_config, templates
            )
        elif source_dataset == "vcr":
            return AttributeComparisonGenerator.generate_from_vcr(
                source_data, num_samples, task_config, templates
            )
        elif source_dataset == "visual_genome":
            return AttributeComparisonGenerator.generate_from_visual_genome(
                source_data, num_samples, task_config, templates
            )
        else:
            logger.warning(f"Attribute comparison not implemented for {source_dataset}")
            return []

    def _generate_visual_noise_filtering(self,
                                        source_dataset: str,
                                        source_data: List[Dict],
                                        num_samples: int,
                                        task_config: Dict[str, Any],
                                        templates: Dict[str, Any]) -> List[Dict]:
        """Generate Visual Noise Filtering tasks."""

        if source_dataset == "mscoco14":
            return EnhancedVNFGenerator.generate_from_mscoco(
                source_data, num_samples, task_config, templates
            )
        elif source_dataset == "vcr":
            return EnhancedVNFGenerator.generate_from_vcr(
                source_data, num_samples, task_config, templates
            )
        else:
            logger.warning(f"VNF not implemented for {source_dataset}")
            return []

    def _generate_attribute_bridge_reasoning(self,
                                            source_dataset: str,
                                            source_data: List[Dict],
                                            num_samples: int,
                                            task_config: Dict[str, Any],
                                            templates: Dict[str, Any]) -> List[Dict]:
        """Generate Attribute Bridge Reasoning tasks."""

        # Import existing generator methods
        from .generator import DataGenerator

        # Create a temporary generator instance to reuse existing methods
        temp_gen = DataGenerator(self.loader)

        min_hops = task_config.get('min_hops', 2)
        max_hops = task_config.get('max_hops', 3)

        if source_dataset == "mscoco14":
            return temp_gen._abr_from_mscoco(source_data, num_samples, min_hops, max_hops)
        elif source_dataset == "vcr":
            return temp_gen._abr_from_vcr(source_data, num_samples, min_hops, max_hops)
        elif source_dataset == "visual_genome":
            return temp_gen._abr_from_visual_genome(source_data, num_samples, min_hops, max_hops)
        else:
            logger.warning(f"ABR not implemented for {source_dataset}")
            return []

    def _generate_relation_comparison(self,
                                     source_dataset: str,
                                     source_data: List[Dict],
                                     num_samples: int,
                                     task_config: Dict[str, Any],
                                     templates: Dict[str, Any]) -> List[Dict]:
        """Generate Relation Comparison tasks."""

        # Import existing generator methods
        from .generator import DataGenerator
        temp_gen = DataGenerator(self.loader)

        n_images = task_config.get('n_images', 3)

        if source_dataset == "mscoco14":
            return temp_gen._rc_from_mscoco(source_data, num_samples, n_images)
        elif source_dataset == "visual_genome":
            return temp_gen._rc_from_visual_genome(source_data, num_samples, n_images)
        else:
            logger.warning(f"RC not implemented for {source_dataset}")
            return []

    # ==================== 窗口2新增: Rationale-Based ABR ====================

    def _generate_rationale_based_abr(self,
                                      source_dataset: str,
                                      source_data: List[Dict],
                                      num_samples: int,
                                      task_config: Dict[str, Any],
                                      templates: Dict[str, Any]) -> List[Dict]:
        """
        生成基于Rationale的属性桥接推理任务 (窗口2新增)

        仅支持VCR数据集（具有rationale标注）

        Args:
            source_dataset: 源数据集ID
            source_data: 源数据
            num_samples: 目标样本数
            task_config: 任务配置
            templates: 问题模板

        Returns:
            生成的任务列表
        """
        if source_dataset != "vcr":
            logger.warning(f"Rationale-based ABR only supports VCR dataset, got {source_dataset}")
            return []

        # 验证数据是否包含rationale
        samples_with_rationale = [
            s for s in source_data
            if s.get('rationale_text') or s.get('reasoning_steps')
        ]

        if not samples_with_rationale:
            logger.warning("No VCR samples with rationale found. "
                          "Make sure to load with include_rationale=True")
            return []

        logger.info(f"Generating rationale-based ABR from {len(samples_with_rationale)} VCR samples with rationale")

        # 使用Rationale生成器
        tasks = RationaleBasedABRGenerator.generate_from_vcr(
            source_data=samples_with_rationale,
            num_samples=num_samples,
            task_config=task_config,
            templates=templates
        )

        logger.info(f"Generated {len(tasks)} rationale-based ABR tasks")
        return tasks

    def _apply_quality_control(self,
                              tasks: List[Dict],
                              task_type: str) -> List[Dict]:
        """Apply quality control filters to generated tasks."""

        qc_rules = self.config.get_quality_rules(task_type)
        if not qc_rules:
            return tasks

        filtered_tasks = []

        for task in tasks:
            # Check reasoning depth
            if 'min_reasoning_depth' in qc_rules:
                min_depth = qc_rules['min_reasoning_depth']
                if task.get('reasoning_depth', 0) < min_depth:
                    continue

            if 'max_reasoning_depth' in qc_rules:
                max_depth = qc_rules['max_reasoning_depth']
                if task.get('reasoning_depth', 999) > max_depth:
                    continue

            # Check required fields
            if 'question' not in task or 'answer' not in task or 'images' not in task:
                logger.warning(f"Task {task.get('task_id')} missing required fields")
                continue

            # Check image paths exist (if configured)
            global_qc = self.config.get_quality_rules()
            if global_qc.get('require_valid_image_path', False):
                all_exist = True
                for img_path in task['images']:
                    if not Path(img_path).exists():
                        logger.debug(f"Image not found: {img_path}")
                        all_exist = False
                        break

                if not all_exist:
                    continue

            filtered_tasks.append(task)

        if len(filtered_tasks) < len(tasks):
            logger.info(f"Quality control: kept {len(filtered_tasks)}/{len(tasks)} tasks")

        return filtered_tasks

    def get_supported_datasets(self, task_type: Optional[str] = None) -> List[str]:
        """Get datasets that support a specific task type."""
        if task_type:
            return self.config.get_datasets_supporting_task(task_type)
        return self.config.get_all_dataset_ids()

    def get_supported_tasks(self, dataset_id: Optional[str] = None) -> List[str]:
        """Get supported task types for a dataset."""
        if dataset_id:
            dataset_config = self.config.get_dataset_config(dataset_id)
            if dataset_config:
                return dataset_config.supported_tasks
            return []
        return self.config.get_supported_tasks()


# Backward compatibility: keep original DataGenerator interface
class DataGenerator(DataGeneratorV2):
    """Backward compatible DataGenerator using V2 implementation."""
    pass
