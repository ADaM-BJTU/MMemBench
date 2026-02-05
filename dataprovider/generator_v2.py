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
    EnhancedVNFGenerator,
    AttributeBridgeReasoningGenerator,
    RelationComparisonGenerator,
    QADatasetGenerators
)

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
            # Full names
            'attribute_comparison': self._generate_attribute_comparison,
            'visual_noise_filtering': self._generate_visual_noise_filtering,
            'attribute_bridge_reasoning': self._generate_attribute_bridge_reasoning,
            'relation_comparison': self._generate_relation_comparison,
            # Short names
            'AC': self._generate_attribute_comparison,
            'VNF': self._generate_visual_noise_filtering,
            'ABR': self._generate_attribute_bridge_reasoning,
            'RC': self._generate_relation_comparison,
            'LNF': self._generate_logical_noise_filtering,
            'logical_noise_filtering': self._generate_logical_noise_filtering
        }

    @staticmethod
    def _vcr_tokens_to_text(tokens, objects: List[str]) -> str:
        """
        将 VCR 的 token 列表转换为自然语言文本。
        
        VCR 数据集使用特殊格式，其中对象引用以整数列表形式嵌入，如：
        ["Does", [2], "feel", "comfortable", "?"]
        其中 [2] 表示 objects[2] 对应的对象。
        
        Args:
            tokens: VCR 格式的 token 列表
            objects: 对象名称列表（通常来自 sample['objects']）
        
        Returns:
            转换后的自然语言文本
        """
        if not tokens:
            return ""
        
        # 如果已经是字符串，直接返回
        if isinstance(tokens, str):
            return tokens
        
        text_parts = []
        for token in tokens:
            if isinstance(token, str):
                text_parts.append(token)
            elif isinstance(token, list):
                # 对象索引列表，如 [0, 1] 表示 "person1 and person2"
                obj_names = []
                for idx in token:
                    if isinstance(idx, int) and 0 <= idx < len(objects):
                        obj_names.append(objects[idx])
                    else:
                        obj_names.append("someone")
                if len(obj_names) == 1:
                    text_parts.append(obj_names[0])
                elif len(obj_names) == 2:
                    text_parts.append(f"{obj_names[0]} and {obj_names[1]}")
                else:
                    text_parts.append(", ".join(obj_names[:-1]) + f" and {obj_names[-1]}")
            elif isinstance(token, int):
                # 单个整数索引
                if 0 <= token < len(objects):
                    text_parts.append(objects[token])
                else:
                    text_parts.append("someone")
            else:
                text_parts.append(str(token))
        
        return " ".join(text_parts)

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
        elif source_dataset == "scienceqa":
            return QADatasetGenerators.generate_ac_from_scienceqa(
                source_data, num_samples, task_config, templates
            )
        elif source_dataset in ["docvqa", "realworldqa"]:
            # 其他QA数据集使用通用AC生成器
            return QADatasetGenerators.generate_ac_from_qa(
                source_data, source_dataset, num_samples, task_config, templates
            )
        else:
            # 为其他数据集提供通用实现
            logger.info(f"Generating attribute comparison tasks for {source_dataset}")
            samples = []
            
            comparison_metric = task_config.get('comparison_metric', 'position')
            
            
            # 安全地获取和转换数据
            def safe_get(data, key, default=None):
                val = data.get(key, default)
                if val is None:
                    return default
                import numpy as np
                if isinstance(val, np.ndarray):
                    if val.size == 1:
                        return val.item()
                    else:
                        return val.tolist()
                elif isinstance(val, (np.integer, np.floating)):
                    return val.item()
                elif isinstance(val, bytes):
                    return val.decode('utf-8', errors='ignore') if len(val) < 1000 else None
                return val
            
            for i in range(min(num_samples, len(source_data))):
                sample = source_data[i]
                
                # 获取图像路径
                images = safe_get(sample, 'image_path')
                if not images:
                    continue
                images = [images] if isinstance(images, str) else images
                
                question = safe_get(sample, 'question', 'Compare attributes in this data')
                answer_idx = safe_get(sample, 'answer', 0)
                choices = safe_get(sample, 'choices', [])
                
                # 构建答案
                try:
                    has_choices = isinstance(choices, (list, tuple)) and len(choices) > 0
                except TypeError:
                    has_choices = False
                
                if has_choices:
                    try:
                        if isinstance(answer_idx, int) and 0 <= answer_idx < len(choices):
                            answer = choices[answer_idx]
                        else:
                            answer = str(answer_idx)
                    except (TypeError, IndexError):
                        answer = str(answer_idx)
                else:
                    answer = str(answer_idx)
                
                # 确保answer是字符串
                if isinstance(answer, (list, tuple)):
                    answer = ', '.join(str(a) for a in answer)
                elif isinstance(answer, bytes):
                    answer = answer.decode('utf-8', errors='ignore')
                else:
                    answer = str(answer)
                
                # 获取元数据
                subject = safe_get(sample, 'subject', '')
                topic = safe_get(sample, 'topic', '')
                category = safe_get(sample, 'category', '')
                skill = safe_get(sample, 'skill', '')
                
                # 构建推理证据
                evidence = {
                    'question': question,
                    'answer': answer,
                    'choices': choices if has_choices else [],
                    'subject': subject,
                    'topic': topic,
                    'category': category,
                    'skill': skill,
                    'comparison_metric': comparison_metric
                }
                
                task = {
                    'task_id': f"ac_{source_dataset}_{len(samples)}",
                    'task_type': 'attribute_comparison',
                    'images': images,
                    'question': question,
                    'answer': answer,
                    'choices': choices,
                    'reasoning_evidence': evidence,
                    'metadata': {
                        'source_dataset': source_dataset,
                        'comparison_metric': comparison_metric,
                        'subject': subject,
                        'topic': topic,
                        'category': category,
                        'skill': skill,
                        'grade': safe_get(sample, 'grade', '')
                    }
                }
                samples.append(task)
            
            logger.info(f"Generated {len(samples)} attribute comparison tasks from {len(source_data)} attempts")
            return samples

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
        elif source_dataset in ["scienceqa", "docvqa", "realworldqa"]:
            # QA数据集使用通用VNF生成器
            return QADatasetGenerators.generate_vnf_from_qa(
                source_data, source_dataset, num_samples, task_config, templates
            )
        else:
            # 为其他数据集提供通用实现
            logger.info(f"Generating VNF tasks for {source_dataset}")
            samples = []
            
            # 安全地获取和转换数据
            def safe_get(data, key, default=None):
                val = data.get(key, default)
                if val is None:
                    return default
                import numpy as np
                if isinstance(val, np.ndarray):
                    if val.size == 1:
                        return val.item()
                    else:
                        return val.tolist()
                elif isinstance(val, (np.integer, np.floating)):
                    return val.item()
                elif isinstance(val, bytes):
                    return val.decode('utf-8', errors='ignore') if len(val) < 1000 else None
                return val
            
            for i in range(min(num_samples, len(source_data))):
                sample = source_data[i]
                
                # 安全地获取图像路径
                images = safe_get(sample, 'image_path')
                if not images:
                    continue
                images = [images] if isinstance(images, str) else images
                
                question = safe_get(sample, 'question', 'Identify valid information in this data')
                answer_idx = safe_get(sample, 'answer', 0)
                choices = safe_get(sample, 'choices', [])
                
                # 构建答案
                try:
                    has_choices = isinstance(choices, (list, tuple)) and len(choices) > 0
                except TypeError:
                    has_choices = False
                
                if has_choices:
                    try:
                        if isinstance(answer_idx, int) and 0 <= answer_idx < len(choices):
                            answer = choices[answer_idx]
                        else:
                            answer = str(answer_idx)
                    except (TypeError, IndexError):
                        answer = str(answer_idx)
                else:
                    answer = str(answer_idx)
                
                # 确保answer是字符串
                if isinstance(answer, (list, tuple)):
                    answer = ', '.join(str(a) for a in answer)
                elif isinstance(answer, bytes):
                    answer = answer.decode('utf-8', errors='ignore')
                else:
                    answer = str(answer)
                
                # 获取元数据
                subject = safe_get(sample, 'subject', '')
                topic = safe_get(sample, 'topic', '')
                category = safe_get(sample, 'category', '')
                
                # 构建推理证据
                evidence = {
                    'question': question,
                    'answer': answer,
                    'choices': choices if has_choices else [],
                    'subject': subject,
                    'topic': topic,
                    'category': category,
                    'noise_level': task_config.get('noise_level', 'moderate')
                }
                
                task = {
                    'task_id': f"vnf_{source_dataset}_{len(samples)}",
                    'task_type': 'visual_noise_filtering',
                    'images': images,
                    'question': question,
                    'answer': answer,
                    'choices': choices,
                    'reasoning_evidence': evidence,
                    'metadata': {
                        'source_dataset': source_dataset,
                        'noise_level': task_config.get('noise_level', 'moderate'),
                        'subject': subject,
                        'topic': topic,
                        'category': category
                    }
                }
                samples.append(task)
            
            logger.info(f"Generated {len(samples)} VNF tasks from {source_dataset}")
            return samples

    def _generate_attribute_bridge_reasoning(self,
                                            source_dataset: str,
                                            source_data: List[Dict],
                                            num_samples: int,
                                            task_config: Dict[str, Any],
                                            templates: Dict[str, Any]) -> List[Dict]:
        """Generate Attribute Bridge Reasoning tasks."""
        
        # MSCOCO14 专用生成器
        if source_dataset == "mscoco14":
            return AttributeBridgeReasoningGenerator.generate_from_mscoco(
                source_data, num_samples, task_config, templates
            )
        
        min_hops = task_config.get('min_hops', 2)
        max_hops = task_config.get('max_hops', 3)
        samples = []
        
        # 安全地获取和转换数据
        def safe_get(data, key, default=None):
            val = data.get(key, default)
            if val is None:
                return default
            import numpy as np
            if isinstance(val, np.ndarray):
                # 多元素numpy数组转为列表，单元素取.item()
                if val.size == 1:
                    return val.item()
                else:
                    return val.tolist()
            elif isinstance(val, (np.integer, np.floating)):
                return val.item()
            elif isinstance(val, bytes):
                return val.decode('utf-8', errors='ignore') if len(val) < 1000 else None
            return val
        
        # Generate basic ABR tasks
        for i in range(min(num_samples, len(source_data))):
            sample = source_data[i]
            
            # 获取图像路径
            images = safe_get(sample, 'image_path')
            if not images:
                continue
            images = [images] if isinstance(images, str) else images
            
            # VCR 数据集特殊处理：将 token 列表转换为自然语言
            if source_dataset == 'vcr':
                objects = sample.get('objects', [])
                
                # 转换问题
                raw_question = sample.get('question', [])
                question = self._vcr_tokens_to_text(raw_question, objects)
                
                # 转换答案选项
                raw_answer_choices = sample.get('answer_choices', [])
                choices = []
                for choice_tokens in raw_answer_choices:
                    choice_text = self._vcr_tokens_to_text(choice_tokens, objects)
                    choices.append(choice_text)
                
                # 获取正确答案索引并转换为文本
                answer_label = sample.get('answer_label', 0)
                if isinstance(answer_label, int) and 0 <= answer_label < len(choices):
                    answer = choices[answer_label]
                else:
                    answer = str(answer_label)
                
                has_choices = len(choices) > 0
            else:
                # 其他数据集的原有逻辑
                question = safe_get(sample, 'question', 'What is the relationship in this data?')
                answer_idx = safe_get(sample, 'answer', 0)
                choices = safe_get(sample, 'choices', [])
                
                # 构建答案
                try:
                    has_choices = isinstance(choices, (list, tuple)) and len(choices) > 0
                except TypeError:
                    has_choices = False
                
                if has_choices:
                    try:
                        if isinstance(answer_idx, int) and 0 <= answer_idx < len(choices):
                            answer = choices[answer_idx]
                        else:
                            answer = str(answer_idx)
                    except (TypeError, IndexError):
                        answer = str(answer_idx)
                else:
                    answer = str(answer_idx)
                
                # 确保answer是字符串
                if isinstance(answer, (list, tuple)):
                    answer = ', '.join(str(a) for a in answer)
                elif isinstance(answer, bytes):
                    answer = answer.decode('utf-8', errors='ignore')
                else:
                    answer = str(answer)
            
            # 获取元数据
            subject = safe_get(sample, 'subject', '')
            topic = safe_get(sample, 'topic', '')
            category = safe_get(sample, 'category', '')
            skill = safe_get(sample, 'skill', '')
            
            # 构建推理证据
            evidence = {
                'question': question,
                'answer': answer,
                'choices': choices if has_choices else [],
                'subject': subject,
                'topic': topic,
                'category': category,
                'skill': skill
            }
            
            task = {
                'task_id': f"abr_{source_dataset}_{len(samples)}",
                'task_type': 'attribute_bridge_reasoning',
                'images': images,
                'question': question,
                'answer': answer,
                'choices': choices if has_choices else [],
                'reasoning_evidence': evidence,
                'metadata': {
                    'source_dataset': source_dataset,
                    'min_hops': min_hops,
                    'max_hops': max_hops,
                    'subject': subject,
                    'topic': topic,
                    'category': category,
                    'skill': skill,
                    'grade': safe_get(sample, 'grade', '')
                }
            }
            samples.append(task)
        
        logger.info(f"Generated {len(samples)} ABR tasks from {source_dataset}")
        return samples

    def _generate_relation_comparison(self,
                                     source_dataset: str,
                                     source_data: List[Dict],
                                     num_samples: int,
                                     task_config: Dict[str, Any],
                                     templates: Dict[str, Any]) -> List[Dict]:
        """Generate Relation Comparison tasks."""
        
        # MSCOCO14 专用生成器
        if source_dataset == "mscoco14":
            return RelationComparisonGenerator.generate_from_mscoco(
                source_data, num_samples, task_config, templates
            )
        
        # VCR 专用生成器
        if source_dataset == "vcr":
            return RelationComparisonGenerator.generate_from_vcr(
                source_data, num_samples, task_config, templates
            )
        
        # QA数据集专用生成器
        if source_dataset == "scienceqa":
            return QADatasetGenerators.generate_rc_from_scienceqa(
                source_data, num_samples, task_config, templates
            )
        if source_dataset == "docvqa":
            return QADatasetGenerators.generate_rc_from_docvqa(
                source_data, num_samples, task_config, templates
            )
        if source_dataset == "realworldqa":
            return QADatasetGenerators.generate_rc_from_realworldqa(
                source_data, num_samples, task_config, templates
            )
        
        # 初始化变量
        samples = []
        n_images = task_config.get('n_images', 3)
        
        # 安全地获取和转换数据
        def safe_get(data, key, default=None):
            val = data.get(key, default)
            if val is None:
                return default
            import numpy as np
            if isinstance(val, np.ndarray):
                if val.size == 1:
                    return val.item()
                else:
                    return val.tolist()
            elif isinstance(val, (np.integer, np.floating)):
                return val.item()
            elif isinstance(val, bytes):
                return val.decode('utf-8', errors='ignore') if len(val) < 1000 else None
            return val
        
        # Generate basic RC tasks
        for i in range(min(num_samples, len(source_data))):
            sample = source_data[i]
            
            # 获取图像路径
            images = safe_get(sample, 'image_path')
            if not images:
                continue
            images = [images] if isinstance(images, str) else images
            
            # VCR 数据集特殊处理：将 token 列表转换为自然语言
            if source_dataset == 'vcr':
                objects = sample.get('objects', [])
                
                # 转换问题
                raw_question = sample.get('question', [])
                question = self._vcr_tokens_to_text(raw_question, objects)
                
                # 转换答案选项
                raw_answer_choices = sample.get('answer_choices', [])
                choices = []
                for choice_tokens in raw_answer_choices:
                    choice_text = self._vcr_tokens_to_text(choice_tokens, objects)
                    choices.append(choice_text)
                
                # 获取正确答案索引并转换为文本
                answer_label = sample.get('answer_label', 0)
                if isinstance(answer_label, int) and 0 <= answer_label < len(choices):
                    answer = choices[answer_label]
                else:
                    answer = str(answer_label)
                
                has_choices = len(choices) > 0
            else:
                # 其他数据集的原有逻辑
                question = safe_get(sample, 'question', 'Compare the elements in this data.')
                answer_idx = safe_get(sample, 'answer', 0)
                choices = safe_get(sample, 'choices', [])
                
                # 构建答案
                try:
                    has_choices = isinstance(choices, (list, tuple)) and len(choices) > 0
                except TypeError:
                    has_choices = False
                
                if has_choices:
                    try:
                        if isinstance(answer_idx, int) and 0 <= answer_idx < len(choices):
                            answer = choices[answer_idx]
                        else:
                            answer = str(answer_idx)
                    except (TypeError, IndexError):
                        answer = str(answer_idx)
                else:
                    answer = str(answer_idx)
                
                # 确保answer是字符串
                if isinstance(answer, (list, tuple)):
                    answer = ', '.join(str(a) for a in answer)
                elif isinstance(answer, bytes):
                    answer = answer.decode('utf-8', errors='ignore')
                else:
                    answer = str(answer)
            
            # 获取元数据
            subject = safe_get(sample, 'subject', '')
            topic = safe_get(sample, 'topic', '')
            category = safe_get(sample, 'category', '')
            skill = safe_get(sample, 'skill', '')
            
            # 构建推理证据
            evidence = {
                'question': question,
                'answer': answer,
                'choices': choices if has_choices else [],
                'subject': subject,
                'topic': topic,
                'category': category,
                'skill': skill,
                'comparison_type': 'count'
            }
            
            task = {
                'task_id': f"rc_{source_dataset}_{len(samples)}",
                'task_type': 'relation_comparison',
                'images': images,
                'question': question,
                'answer': answer,
                'choices': choices if has_choices else [],
                'reasoning_evidence': evidence,
                'metadata': {
                    'source_dataset': source_dataset,
                    'n_images': n_images,
                    'subject': subject,
                    'topic': topic,
                    'category': category,
                    'skill': skill,
                    'grade': safe_get(sample, 'grade', '')
                }
            }
            samples.append(task)
        
        logger.info(f"Generated {len(samples)} RC tasks from {source_dataset}")
        return samples

    def _generate_logical_noise_filtering(self,
                                          source_dataset: str,
                                          source_data: List[Dict],
                                          num_samples: int,
                                          task_config: Dict[str, Any],
                                          templates: Dict[str, Any]) -> List[Dict]:
        """Generate Logical Noise Filtering tasks."""
        
        if source_dataset == "scienceqa":
            return QADatasetGenerators.generate_lnf_from_scienceqa(
                source_data, num_samples, task_config, templates
            )
        elif source_dataset in ["docvqa", "realworldqa"]:
            # 暂时为其他QA数据集保留接口，如果将来需要支持
            if hasattr(QADatasetGenerators, 'generate_lnf_from_qa'):
                return QADatasetGenerators.generate_lnf_from_qa(
                    source_data, source_dataset, num_samples, task_config, templates
                )
            else:
                logger.warning(f"LNF generator for {source_dataset} not fully implemented yet.")
                return []
        
        logger.warning(f"LNF not supported for {source_dataset}")
        return []

    def _apply_quality_control(self,
                              tasks: List[Dict],
                              task_type: str) -> List[Dict]:
        """Apply quality control filters to generated tasks."""

        qc_rules = self.config.get_quality_rules(task_type)
        if not qc_rules:
            return tasks

        filtered_tasks = []
        rejected_reasons = {}

        for task in tasks:
            # Track rejection reason
            rejected = False
            reason = ""

            # Check reasoning depth
            if 'min_reasoning_depth' in qc_rules:
                min_depth = qc_rules['min_reasoning_depth']
                if task.get('reasoning_depth', 0) < min_depth:
                    reason = f"reasoning_depth too low ({task.get('reasoning_depth', 0)} < {min_depth})"
                    rejected = True

            if not rejected and 'max_reasoning_depth' in qc_rules:
                max_depth = qc_rules['max_reasoning_depth']
                if task.get('reasoning_depth', 999) > max_depth:
                    reason = f"reasoning_depth too high ({task.get('reasoning_depth', 999)} > {max_depth})"
                    rejected = True

            # Check required fields
            if not rejected:
                if 'question' not in task:
                    reason = "missing question"
                    rejected = True
                elif 'answer' not in task:
                    reason = "missing answer"
                    rejected = True
                elif 'images' not in task:
                    reason = "missing images"
                    rejected = True

            # Check image paths exist (if configured)
            if not rejected:
                global_qc = self.config.get_quality_rules()
                if global_qc.get('require_valid_image_path', False):
                    all_exist = True
                    missing_images = []
                    for img_path in task['images']:
                        if not Path(img_path).exists():
                            missing_images.append(img_path)
                            all_exist = False

                    if not all_exist:
                        reason = f"missing images: {missing_images}"
                        rejected = True

            # Check task type specific rules
            if not rejected and task_type == 'VNF':
                if 'min_distractor_images' in qc_rules:
                    min_distractors = qc_rules['min_distractor_images']
                    if len(task.get('images', [])) - 1 < min_distractors:
                        reason = f"not enough distractor images"
                        rejected = True

            if not rejected:
                filtered_tasks.append(task)
            else:
                rejected_reasons[task.get('task_id', 'unknown')] = reason

        if len(filtered_tasks) < len(tasks):
            logger.info(f"Quality control: kept {len(filtered_tasks)}/{len(tasks)} tasks")
            # Log first few rejection reasons for debugging
            if rejected_reasons:
                logger.info(f"Sample rejection reasons: {dict(list(rejected_reasons.items())[:3])}")

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

    def save_generated_tasks(self, tasks: List[Dict], output_path: str) -> None:
        """Save generated tasks to file in JSONL format."""
        import json
        from pathlib import Path
        
        # Ensure output directory exists
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Write tasks in JSONL format
        with open(output_path, 'w', encoding='utf-8') as f:
            for task in tasks:
                json.dump(task, f, ensure_ascii=False)
                f.write('\n')
        
        logger.info(f"Saved {len(tasks)} tasks to {output_path}")


# Backward compatibility: keep original DataGenerator interface
class DataGenerator(DataGeneratorV2):
    """Backward compatible DataGenerator using V2 implementation."""
    pass
