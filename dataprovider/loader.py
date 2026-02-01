"""
DataLoader: Load datasets by ID and extract needed data
========================================================

Supports multiple dataset formats:
- MSCOCO14: Object detection with bounding boxes, categories, captions
- MMMU: Multi-discipline multimodal understanding questions
- MM-NIAH: Needle in haystack with retrieval/counting/reasoning
- MMDocIR: Long document multi-modal retrieval
- VCR (VisualCommonReasoning): Question-answer-rationale with objects
- Sherlock: Clue-inference pairs with bounding boxes
- ScienceQA: Science questions with images and explanations
- RealworldQA: Real-world visual questions
- DocVQA/InfoGraphVQA: Document visual question answering
- SlideVQA: Slide deck visual questions
- MMStar: Vision-indispensable multi-modal benchmark
- GQA: Visual reasoning questions with scene graphs
- Visual Genome: Scene graphs with objects, attributes, relationships
"""

import json
import jsonlines
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
import logging

logger = logging.getLogger(__name__)


class DataLoader:
    """Load and parse various multimodal datasets."""

    SUPPORTED_DATASETS = [
        "mscoco14",
        "mmmu",
        "mm-niah",
        "mmdocir",
        "vcr",
        "sherlock",
        "scienceqa",
        "realworldqa",
        "docvqa",
        "infographvqa",
        "slidevqa",
        "mmstar",
        "gqa",
        "visual_genome"
    ]

    def __init__(self, data_root: str = "data"):
        """
        Initialize DataLoader.

        Args:
            data_root: Root directory containing all datasets
        """
        self.data_root = Path(data_root)
        self._dataset_cache = {}

    def load_dataset(self,
                    dataset_id: str,
                    split: str = "train",
                    max_samples: Optional[int] = None,
                    **kwargs) -> List[Dict[str, Any]]:
        """
        Load a dataset by ID.

        Args:
            dataset_id: Dataset identifier (e.g., "mscoco14", "mmmu")
            split: Data split ("train", "val", "test")
            max_samples: Maximum number of samples to load
            **kwargs: Additional dataset-specific parameters

        Returns:
            List of data samples in unified format
        """
        dataset_id = dataset_id.lower()

        if dataset_id not in self.SUPPORTED_DATASETS:
            raise ValueError(f"Unsupported dataset: {dataset_id}. "
                           f"Supported: {self.SUPPORTED_DATASETS}")

        # Dispatch to specific loader
        loader_method = getattr(self, f"_load_{dataset_id.replace('-', '_')}")
        data = loader_method(split=split, **kwargs)

        if max_samples:
            data = data[:max_samples]

        logger.info(f"Loaded {len(data)} samples from {dataset_id} ({split})")
        return data

    # ==================== MSCOCO14 ====================

    def _load_mscoco14(self, split: str = "train", **kwargs) -> List[Dict]:
        """
        Load MSCOCO14 dataset.

        Returns format:
        {
            'dataset_id': 'mscoco14',
            'sample_id': str,
            'image_path': str,
            'image_id': int,
            'objects': [{'id', 'category', 'bbox', 'area', 'segmentation'}],
            'captions': [str],
            'width': int,
            'height': int
        }
        """
        # Try multiple possible locations for MSCOCO14
        possible_dirs = [
            self.data_root / "MSCOCO" / "MSCOCO14",  # Standard structure
            self.data_root / "MSCOCO14",              # Direct
            Path("E:/Dataset/MSCOCO/MSCOCO14")       # Absolute fallback
        ]

        data_dir = None
        for dir_path in possible_dirs:
            if dir_path.exists():
                data_dir = dir_path
                break

        if data_dir is None:
            data_dir = possible_dirs[0]  # Use first as default

        # Load annotations - try both 'annotations' and 'example' directories
        # First try 'annotations' (standard COCO structure)
        ann_file = data_dir / "annotations" / f"instances_{split}2014.json"
        cap_file = data_dir / "annotations" / f"captions_{split}2014.json"

        # Fallback to 'example' directory if annotations not found
        if not ann_file.exists():
            ann_file = data_dir / "example" / f"instances_{split}2014.json"
            cap_file = data_dir / "example" / f"captions_{split}2014.json"

        with open(ann_file, 'r') as f:
            ann_data = json.load(f)

        with open(cap_file, 'r') as f:
            cap_data = json.load(f)

        # Build image id to annotations mapping
        image_to_anns = {}
        for ann in ann_data['annotations']:
            img_id = ann['image_id']
            if img_id not in image_to_anns:
                image_to_anns[img_id] = []
            image_to_anns[img_id].append(ann)

        # Build image id to captions mapping
        image_to_caps = {}
        for cap in cap_data['annotations']:
            img_id = cap['image_id']
            if img_id not in image_to_caps:
                image_to_caps[img_id] = []
            image_to_caps[img_id].append(cap['caption'])

        # Build category id to name mapping
        cat_id_to_name = {cat['id']: cat['name'] for cat in ann_data['categories']}

        # Construct samples
        samples = []
        for img_info in ann_data['images']:
            img_id = img_info['id']

            # Get objects
            objects = []
            if img_id in image_to_anns:
                for ann in image_to_anns[img_id]:
                    objects.append({
                        'id': ann['id'],
                        'category': cat_id_to_name[ann['category_id']],
                        'category_id': ann['category_id'],
                        'bbox': ann['bbox'],  # [x, y, width, height]
                        'area': ann['area'],
                        'segmentation': ann.get('segmentation', None),
                        'iscrowd': ann.get('iscrowd', 0)
                    })

            # Build image path - try multiple locations
            file_name = img_info['file_name']
            possible_paths = [
                data_dir / file_name,  # Direct: MSCOCO14/COCO_train2014_*.jpg
                data_dir / f"{split}2014" / file_name,  # Standard: MSCOCO14/train2014/COCO_train2014_*.jpg
                data_dir / "images" / file_name  # Alternative: MSCOCO14/images/COCO_train2014_*.jpg
            ]

            image_path = None
            for path in possible_paths:
                if path.exists():
                    image_path = str(path)
                    break

            # Use first option even if not exists (for consistency)
            if image_path is None:
                image_path = str(possible_paths[1])  # Use split2014/ subdirectory as default

            samples.append({
                'dataset_id': 'mscoco14',
                'sample_id': f"mscoco14_{img_id}",
                'image_path': image_path,
                'image_id': img_id,
                'file_name': img_info['file_name'],
                'objects': objects,
                'captions': image_to_caps.get(img_id, []),
                'width': img_info['width'],
                'height': img_info['height']
            })

        return samples

    # ==================== Visual Genome ====================

    def _load_visual_genome(self, split: str = "train", **kwargs) -> List[Dict]:
        """
        Load Visual Genome dataset with scene graphs.

        Returns format:
        {
            'dataset_id': 'visual_genome',
            'sample_id': str,
            'image_path': str,
            'scene_graph': {
                'objects': {id: {'name', 'bbox', 'attributes'}},
                'relationships': [{'subject', 'predicate', 'object'}]
            }
        }
        """
        data_dir = self.data_root / "VisualGenome"

        # Check if data directory exists
        if not data_dir.exists():
            logger.warning(
                f"Visual Genome 数据集目录不存在: {data_dir}\n"
                f"请下载数据集到: {data_dir}\n"
                f"下载地址: https://homes.cs.washington.edu/~ranjay/visualgenome/api.html"
            )
            return []

        # Load objects and relationships (VG uses combined format, not split-based)
        objects_file = data_dir / "objects.json"
        relationships_file = data_dir / "relationships.json"

        if not objects_file.exists() or not relationships_file.exists():
            logger.warning(
                f"Visual Genome 数据文件不存在:\n"
                f"  需要: {objects_file}\n"
                f"  需要: {relationships_file}\n"
                f"请确保已下载并解压到: {data_dir}"
            )
            return []

        logger.info(f"Loading Visual Genome objects from {objects_file}...")
        with open(objects_file, 'r') as f:
            objects_data = json.load(f)

        logger.info(f"Loading Visual Genome relationships from {relationships_file}...")
        with open(relationships_file, 'r') as f:
            relationships_data = json.load(f)

        # Build image_id -> relationships mapping
        rel_by_image = {}
        for rel_item in relationships_data:
            img_id = rel_item['image_id']
            if img_id not in rel_by_image:
                rel_by_image[img_id] = []
            rel_by_image[img_id].extend(rel_item.get('relationships', []))

        # Process objects and combine with relationships
        samples = []
        max_samples = kwargs.get('max_samples', 100)  # Limit for faster loading

        for idx, obj_item in enumerate(objects_data):
            if idx >= max_samples:
                break

            img_id = obj_item['image_id']

            # Parse objects
            objects = {}
            for obj in obj_item.get('objects', []):
                obj_id = str(obj['object_id'])
                objects[obj_id] = {
                    'name': obj.get('names', [''])[0] if obj.get('names') else '',
                    'bbox': [obj.get('x', 0), obj.get('y', 0),
                            obj.get('w', 0), obj.get('h', 0)],
                    'attributes': obj.get('attributes', [])
                }

            # Parse relationships for this image
            relationships = []
            for rel in rel_by_image.get(img_id, []):
                relationships.append({
                    'subject': str(rel.get('subject', {}).get('object_id', '')),
                    'predicate': rel.get('predicate', ''),
                    'object': str(rel.get('object', {}).get('object_id', ''))
                })

            # Determine image path - VG has images in VG_100K and VG_100K_2
            img_path_1 = data_dir / "2016" / "VG_100K" / f"{img_id}.jpg"
            img_path_2 = data_dir / "2016" / "VG_100K_2" / f"{img_id}.jpg"

            if img_path_1.exists():
                img_path = str(img_path_1)
            elif img_path_2.exists():
                img_path = str(img_path_2)
            else:
                img_path = str(img_path_1)  # Use first as default

            samples.append({
                'dataset_id': 'visual_genome',
                'sample_id': f"vg_{img_id}",
                'image_id': img_id,
                'image_path': img_path,
                'scene_graph': {
                    'objects': objects,
                    'relationships': relationships
                }
            })

        return samples

    # ==================== VCR ====================

    def _load_vcr(self, split: str = "train", **kwargs) -> List[Dict]:
        """
        Load Visual Commonsense Reasoning dataset (增强版 - 包含rationale).

        Args:
            split: "train" | "val" | "test"
            **kwargs:
                include_rationale: bool = True  # 是否包含rationale
                include_all_choices: bool = False  # 是否包含所有选项
                decode_tokens: bool = True  # 是否解码token为文本

        Returns format:
        {
            'dataset_id': 'vcr',
            'sample_id': str,
            'image_path': str,

            # 问题和答案 (原始token和解码文本)
            'question': List[Union[str, List[int]]],  # 原始token
            'question_text': str,  # 解码后的问题文本
            'answer_choices': List[List[Union[str, List[int]]]],
            'answer_label': int,
            'answer_text': str,  # 正确答案文本

            # ⭐ Rationale (新增)
            'rationale_choices': List[List[Union[str, List[int]]]],
            'rationale_label': int,
            'rationale_text': str,  # 正确rationale文本
            'reasoning_steps': List[str],  # 拆分的推理步骤

            # 物体信息
            'objects': List[str],
            'metadata': {'boxes', 'segms', 'width', 'height'}
        }
        """
        include_rationale = kwargs.get('include_rationale', True)
        include_all_choices = kwargs.get('include_all_choices', False)
        decode_tokens = kwargs.get('decode_tokens', True)

        # Try both directory names (handle typo in original dataset)
        data_dir = self._find_vcr_directory()
        if data_dir is None:
            logger.warning("VCR directory not found")
            return []

        # Load annotations
        ann_file = data_dir / f"{split}.jsonl"

        if not ann_file.exists():
            logger.warning(f"VCR file not found: {ann_file}")
            return []

        samples = []
        valid_count = 0
        error_count = 0
        max_samples = kwargs.get('max_samples', None)

        with open(ann_file, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                # Early exit if we have enough samples
                if max_samples and valid_count >= max_samples:
                    break

                try:
                    item = json.loads(line)
                    objects = item.get('objects', [])

                    # Skip metadata loading for faster processing
                    metadata = {}

                    # 解码问题
                    question_tokens = item.get('question', [])
                    question_text = self._decode_vcr_tokens(question_tokens, objects) if decode_tokens else str(question_tokens)

                    # 解码正确答案
                    answer_choices = item.get('answer_choices', [])
                    answer_label = item.get('answer_label', 0)
                    answer_tokens = answer_choices[answer_label] if answer_label < len(answer_choices) else []
                    answer_text = self._decode_vcr_tokens(answer_tokens, objects) if decode_tokens else str(answer_tokens)

                    sample = {
                        'dataset_id': 'vcr',
                        'sample_id': f"vcr_{split}_{valid_count}",
                        'annot_id': item.get('annot_id', f"{split}-{valid_count}"),
                        'image_path': self._resolve_vcr_image_path(data_dir, item.get('img_fn', '')),

                        # 问题和答案
                        'question': question_tokens,
                        'question_text': question_text,
                        'answer_choices': answer_choices,
                        'answer_label': answer_label,
                        'answer_text': answer_text,

                        # 物体信息
                        'objects': objects,
                        'metadata_fn': item.get('metadata_fn', ''),
                        'metadata': metadata,

                        # 电影信息
                        'movie': item.get('movie', ''),
                        'img_id': item.get('img_id', '')
                    }

                    # ⭐ 添加Rationale
                    if include_rationale:
                        rationale_choices = item.get('rationale_choices', [])
                        rationale_label = item.get('rationale_label', -1)

                        sample['rationale_choices'] = rationale_choices
                        sample['rationale_label'] = rationale_label

                        if rationale_label >= 0 and rationale_label < len(rationale_choices):
                            rationale_tokens = rationale_choices[rationale_label]
                            rationale_text = self._decode_vcr_tokens(rationale_tokens, objects) if decode_tokens else str(rationale_tokens)

                            # 拆分推理步骤
                            reasoning_steps = self._extract_reasoning_steps(rationale_text)

                            sample.update({
                                'rationale_text': rationale_text,
                                'rationale_tokens': rationale_tokens,
                                'reasoning_steps': reasoning_steps
                            })
                        else:
                            sample.update({
                                'rationale_text': '',
                                'rationale_tokens': [],
                                'reasoning_steps': []
                            })

                    # 添加所有选项的解码文本
                    if include_all_choices and decode_tokens:
                        sample['all_answer_choices_text'] = [
                            self._decode_vcr_tokens(c, objects) for c in answer_choices
                        ]
                        sample['all_rationale_choices_text'] = [
                            self._decode_vcr_tokens(c, objects) for c in rationale_choices
                        ] if include_rationale else []

                    samples.append(sample)
                    valid_count += 1

                except (json.JSONDecodeError, KeyError) as e:
                    error_count += 1
                    if error_count <= 5:  # Only log first 5 errors
                        logger.warning(f"Skipping invalid line {idx+1} in VCR {split}: {str(e)[:100]}")
                    continue

        if error_count > 0:
            logger.info(f"Loaded {valid_count} valid VCR samples, skipped {error_count} invalid lines")

        logger.info(f"Loaded {len(samples)} VCR samples with rationale support")
        return samples

    def _decode_vcr_tokens(self, tokens: List, objects: List[str]) -> str:
        """
        解码VCR token列表为可读文本

        规则:
        - 字符串token: 直接拼接
        - [n]: 替换为[objects[n]_n]
        - [n, m, ...]: 替换为[objects[n]_n] and [objects[m]_m]

        Example:
            tokens = [[0], "'", "s", "expression", "is", "twisted"]
            objects = ["person", "person", "car"]
            result = "[person_0]'s expression is twisted"
        """
        if not tokens:
            return ""

        result = []
        for token in tokens:
            if isinstance(token, list):
                # 物体引用 [0] 或 [0, 1]
                refs = []
                for obj_idx in token:
                    if isinstance(obj_idx, int) and obj_idx < len(objects):
                        refs.append(f"[{objects[obj_idx]}_{obj_idx}]")
                    else:
                        refs.append(f"[object_{obj_idx}]")
                result.append(" and ".join(refs))
            else:
                result.append(str(token))

        # 智能拼接（处理标点符号）
        text = ""
        for i, part in enumerate(result):
            # 不在标点符号前加空格
            if i > 0 and part not in ".,!?':;)\"" and not text.endswith("(\"'"):
                text += " "
            text += part

        return text.strip()

    def _extract_reasoning_steps(self, rationale_text: str) -> List[str]:
        """
        从rationale文本中提取推理步骤

        策略:
        1. 按句号分割
        2. 识别因果连接词
        3. 保持步骤间的逻辑关系

        Returns:
            List[str]: 推理步骤列表
        """
        import re

        if not rationale_text:
            return []

        # 分割句子
        sentences = re.split(r'[.!?]+', rationale_text)
        sentences = [s.strip() for s in sentences if s.strip()]

        # 如果只有一句，尝试用连接词分割
        if len(sentences) == 1:
            # 尝试按因果词分割
            causal_patterns = [
                r'\s+because\s+',
                r'\s+so\s+',
                r'\s+therefore\s+',
                r'\s+since\s+',
                r'\s+thus\s+',
                r',\s+and\s+',
                r',\s+which\s+',
            ]
            for pattern in causal_patterns:
                parts = re.split(pattern, sentences[0], flags=re.IGNORECASE)
                if len(parts) > 1:
                    sentences = [p.strip() for p in parts if p.strip()]
                    break

        return sentences

    def _find_vcr_directory(self) -> Optional[Path]:
        """查找VCR数据目录"""
        possible_paths = [
            self.data_root / "VisualCommenReasoning",
            self.data_root / "VisualCommenReasnoning",  # Typo version
            self.data_root / "VisualCommonsenseReasoning",
            self.data_root / "VCR",
            self.data_root / "vcr",
            Path("E:/Dataset/VisualCommenReasoning"),
            Path("E:/Dataset/VisualCommenReasnoning"),
            Path("data/VisualCommenReasoning")
        ]

        for path in possible_paths:
            if path.exists():
                return path

        return None

    def _resolve_vcr_image_path(self, data_dir: Path, img_fn: str) -> str:
        """解析VCR图片路径"""
        if not img_fn:
            return ""

        # 尝试多种路径
        candidates = [
            data_dir / 'vcr1images' / img_fn,
            data_dir / img_fn,
            data_dir / "images" / img_fn,
        ]

        for candidate in candidates:
            if candidate.exists():
                return str(candidate)

        # 返回第一个候选路径（即使不存在）
        return str(candidates[0])

    # ==================== Sherlock ====================

    def _load_sherlock(self, split: str = "train", **kwargs) -> List[Dict]:
        """
        Load Sherlock dataset (clue-inference pairs).

        Returns format:
        {
            'dataset_id': 'sherlock',
            'sample_id': str,
            'image_path': str (URL),
            'clue': str,
            'inference': str,
            'confidence': float,
            'bboxes': List[{'left', 'top', 'width', 'height'}],
            'width': int,
            'height': int
        }
        """
        data_dir = self.data_root / "Sherlock-VCR"

        # Check if data directory exists
        if not data_dir.exists():
            logger.warning(
                f"Sherlock 数据集目录不存在: {data_dir}\n"
                f"请下载数据集到: {data_dir}\n"
                f"下载地址: https://github.com/yukezhu/sherlock"
            )
            return []

        # VCR image directory (Sherlock uses VCR images)
        vcr_dir = self.data_root / "VisualCommenReasnoning"
        if not vcr_dir.exists():
            vcr_dir = self.data_root / "VisualCommonReasoning"

        # Load annotations
        if split == "train":
            ann_file = data_dir / "sherlock_train_v1_1.json"
        else:
            ann_file = data_dir / "sherlock_val_with_split_idxs_v1_1.json"

        if not ann_file.exists():
            logger.warning(
                f"Sherlock 标注文件不存在: {ann_file}\n"
                f"需要文件: sherlock_{split}_v1_1.json\n"
                f"请确保已下载并放置在: {data_dir}"
            )
            return []

        with open(ann_file, 'r') as f:
            data = json.load(f)

        samples = []
        max_samples = kwargs.get('max_samples', 100)

        for idx, item in enumerate(data):
            if idx >= max_samples:
                break

            inputs = item['inputs']

            # Extract local image path from URL
            # URL format: http://s3-us-west-2.amazonaws.com/ai2-rowanz/vcr1images/movieclips_*/xxx.jpg
            image_url = inputs['image']['url']
            # Extract the path after vcr1images/
            if 'vcr1images/' in image_url:
                img_rel_path = image_url.split('vcr1images/')[-1]
                # Use VCR local directory
                image_path = str(vcr_dir / 'vcr1images' / img_rel_path)
            else:
                image_path = image_url  # Keep URL if can't parse

            samples.append({
                'dataset_id': 'sherlock',
                'sample_id': item['instance_id'],
                'image_path': image_path,
                'image_url': image_url,  # Keep original URL
                'width': inputs['image']['width'],
                'height': inputs['image']['height'],
                'bboxes': inputs['bboxes'],
                'clue': inputs['clue'],
                'confidence': inputs['confidence'],
                'obs_idx': inputs['obs_idx'],
                'inference': item['targets']['inference']
            })

        return samples

    # ==================== ScienceQA ====================

    def _load_scienceqa(self, split: str = "train", **kwargs) -> List[Dict]:
        """
        Load ScienceQA dataset.

        Returns format:
        {
            'dataset_id': 'scienceqa',
            'sample_id': str,
            'image': Optional[image bytes],
            'question': str,
            'choices': List[str],
            'answer': int,
            'hint': str,
            'task': str,
            'grade': str,
            'subject': str,
            'topic': str,
            'category': str,
            'lecture': str,
            'solution': str
        }
        """
        # Note: ScienceQA is in parquet format, we'll provide a placeholder
        # for actual implementation
        logger.warning("ScienceQA loader requires parquet support - returning empty")
        return []

    # ==================== RealworldQA ====================

    def _load_realworldqa(self, split: str = "train", **kwargs) -> List[Dict]:
        """
        Load RealworldQA dataset.

        Returns format:
        {
            'dataset_id': 'realworldqa',
            'sample_id': str,
            'image': image bytes,
            'question': str,
            'answer': str
        }
        """
        # Note: RealworldQA is in parquet format
        logger.warning("RealworldQA loader requires parquet support - returning empty")
        return []

    # ==================== MM-NIAH ====================

    def _load_mm_niah(self, split: str = "val", task: str = "retrieval-text", **kwargs) -> List[Dict]:
        """
        Load MM-NIAH (Needle in Multimodal Haystack) dataset.

        Args:
            task: One of ["retrieval-text", "retrieval-image",
                         "counting-text", "counting-image",
                         "reasoning-text", "reasoning-image"]

        Returns format:
        {
            'dataset_id': 'mm-niah',
            'sample_id': str,
            'task_type': str,
            'images_list': List[str],
            'context': str (with <image> placeholders),
            'question': str,
            'answer': Union[str, List],
            'meta': {
                'placed_depth': Union[float, List[float]],
                'context_length': int,
                'num_images': int,
                'needles': List[str],
                'choices': Optional[List[str]],
                'choices_image_path': Optional[List[str]]
            }
        }
        """
        data_dir = self.data_root / "MM-NIAH" / "example"

        # Load task file
        task_file = data_dir / f"{task}.jsonl"

        if not task_file.exists():
            logger.warning(f"MM-NIAH task file not found: {task_file}")
            return []

        samples = []
        with jsonlines.open(task_file) as reader:
            for item in reader:
                samples.append({
                    'dataset_id': 'mm-niah',
                    'sample_id': f"mm-niah_{task}_{item['id']}",
                    'task_type': task,
                    'images_list': item['images_list'],
                    'context': item['context'],
                    'question': item['question'],
                    'answer': item['answer'],
                    'meta': item['meta']
                })

        return samples

    # ==================== MMMU ====================

    def _load_mmmu(self, split: str = "validation", subject: str = "all", **kwargs) -> List[Dict]:
        """
        Load MMMU (Massive Multi-discipline Multimodal Understanding) dataset.

        Args:
            subject: Subject name or "all" for all subjects

        Returns format:
        {
            'dataset_id': 'mmmu',
            'sample_id': str,
            'subject': str,
            'question': str,
            'options': str,
            'answer': str,
            'question_type': str,
            'subfield': str,
            'topic_difficulty': str,
            'images': List[str],  # paths to image_1 through image_7
            'img_type': str,
            'explanation': str
        }
        """
        # Note: MMMU is in HuggingFace datasets format
        # This is a placeholder for the actual implementation
        logger.warning("MMMU loader requires HuggingFace datasets - returning empty")
        return []

    # ==================== MMDocIR ====================

    def _load_mmdocir(self, split: str = "test", **kwargs) -> List[Dict]:
        """
        Load MMDocIR (Multi-Modal Document IR) dataset.

        Returns format:
        {
            'dataset_id': 'mmdocir',
            'sample_id': str,
            'doc_name': str,
            'domain': str,
            'questions': List[{
                'Q': str,
                'A': str,
                'type': str,
                'page_id': List[int],
                'layout_mapping': List[{'page', 'page_size', 'bbox'}]
            }],
            'pages': List[{
                'page_id': int,
                'image_path': str,
                'ocr_text': str,
                'vlm_text': str
            }]
        }
        """
        data_dir = self.data_root / "MMDocIR" / "example"

        # Load annotations
        ann_file = data_dir / "MMDocIR_annotations.jsonl"

        if not ann_file.exists():
            logger.warning(f"MMDocIR annotations not found: {ann_file}")
            return []

        samples = []
        with jsonlines.open(ann_file) as reader:
            for item in reader:
                samples.append({
                    'dataset_id': 'mmdocir',
                    'sample_id': f"mmdocir_{item['doc_name']}",
                    'doc_name': item['doc_name'],
                    'domain': item['domain'],
                    'questions': item['questions'],
                    'page_indices': item['page_indices'],
                    'layout_indices': item['layout_indinces']
                })

        return samples

    # ==================== DocVQA ====================

    def _load_docvqa(self, split: str = "validation", **kwargs) -> List[Dict]:
        """
        Load DocVQA dataset from parquet files.

        Returns format:
        {
            'dataset_id': 'docvqa',
            'sample_id': str,
            'image': bytes (embedded in parquet),
            'question': str,
            'answers': List[str],
            'question_id': int,
            'data_split': str
        }
        """
        data_dir = self.data_root / "lmms-lab_DocVQA" / "DocVQA"

        # Check if data directory exists
        if not data_dir.exists():
            logger.warning(
                f"DocVQA 数据集目录不存在: {data_dir}\n"
                f"请下载数据集到: {data_dir}\n"
                f"下载地址: https://huggingface.co/datasets/lmms-lab/DocVQA"
            )
            return []

        # Try to import pyarrow for parquet support
        try:
            import pyarrow.parquet as pq
        except ImportError:
            logger.warning(
                "DocVQA 需要 pyarrow 库读取 parquet 文件\n"
                "请安装: pip install pyarrow"
            )
            return []

        # Find parquet files for the split
        parquet_files = sorted(data_dir.glob(f"{split}-*.parquet"))

        if not parquet_files:
            logger.warning(
                f"DocVQA parquet 文件不存在: {data_dir}/{split}-*.parquet\n"
                f"可用的 split: train, test, validation"
            )
            return []

        logger.info(f"Found {len(parquet_files)} parquet files for DocVQA {split}")

        samples = []
        max_samples = kwargs.get('max_samples', 100)

        # Read from first parquet file only (for speed)
        table = pq.read_table(parquet_files[0])
        df = table.to_pandas()

        for idx, row in df.iterrows():
            if idx >= max_samples:
                break

            samples.append({
                'dataset_id': 'docvqa',
                'sample_id': f"docvqa_{split}_{idx}",
                'image': row.get('image', None),  # PIL Image or bytes
                'question': row.get('question', ''),
                'answers': row.get('answers', []),
                'question_id': row.get('questionId', idx),
                'data_split': row.get('data_split', split),
                'question_types': row.get('question_types', [])
            })

        logger.info(f"Loaded {len(samples)} samples from DocVQA {split}")
        return samples

    # ==================== InfoGraphVQA ====================

    def _load_infographvqa(self, split: str = "validation", **kwargs) -> List[Dict]:
        """Load InfoGraphVQA dataset."""
        logger.warning("InfoGraphVQA loader requires HuggingFace datasets - returning empty")
        return []

    # ==================== SlideVQA ====================

    def _load_slidevqa(self, split: str = "train", **kwargs) -> List[Dict]:
        """Load SlideVQA dataset."""
        logger.warning("SlideVQA loader requires HuggingFace datasets - returning empty")
        return []

    # ==================== MMStar ====================

    def _load_mmstar(self, split: str = "val", **kwargs) -> List[Dict]:
        """Load MMStar dataset."""
        logger.warning("MMStar loader requires parquet support - returning empty")
        return []

    # ==================== GQA ====================

    def _load_gqa(self, split: str = "train", **kwargs) -> List[Dict]:
        """
        Load GQA dataset with scene graphs.

        Returns format:
        {
            'dataset_id': 'gqa',
            'sample_id': str,
            'image_path': str,
            'scene_graph': {
                'objects': {id: {'name', 'bbox', 'attributes'}},
                'relationships': [{'subject', 'predicate', 'object'}]
            }
        }
        """
        data_dir = self.data_root / "gqa"

        # Check if data directory exists
        if not data_dir.exists():
            logger.warning(
                f"GQA 数据集目录不存在: {data_dir}\n"
                f"请下载数据集到: {data_dir}\n"
                f"下载地址: https://cs.stanford.edu/people/dorarad/gqa/download.html"
            )
            return []

        # Load scene graphs
        sg_file = data_dir / f"{split}_sceneGraphs.json"

        if not sg_file.exists():
            logger.warning(
                f"GQA scene graph 文件不存在: {sg_file}\n"
                f"需要文件: {split}_sceneGraphs.json\n"
                f"请确保已下载并放置在: {data_dir}"
            )
            return []

        with open(sg_file, 'r') as f:
            scene_graphs = json.load(f)

        samples = []
        for image_id, sg in scene_graphs.items():
            # Parse objects
            objects = {}
            for obj_id, obj in sg.get('objects', {}).items():
                objects[obj_id] = {
                    'name': obj.get('name', ''),
                    'bbox': [obj.get('x', 0), obj.get('y', 0),
                            obj.get('w', 0), obj.get('h', 0)],
                    'attributes': obj.get('attributes', [])
                }

            # Parse relationships
            relationships = []
            for obj_id, obj in sg.get('objects', {}).items():
                for rel in obj.get('relations', []):
                    relationships.append({
                        'subject': obj_id,
                        'predicate': rel.get('name', ''),
                        'object': rel.get('object', '')
                    })

            samples.append({
                'dataset_id': 'gqa',
                'sample_id': f"gqa_{image_id}",
                'image_id': image_id,
                'image_path': str(data_dir / "images" / f"{image_id}.jpg"),
                'scene_graph': {
                    'objects': objects,
                    'relationships': relationships
                }
            })

        return samples

    # ==================== Helper Methods ====================

    def get_dataset_info(self, dataset_id: str) -> Dict[str, Any]:
        """
        Get metadata about a dataset.

        Returns:
            Dictionary with dataset information:
            - name: Dataset name
            - description: Brief description
            - splits: Available splits
            - modalities: Data modalities
            - task_types: Supported task types
        """
        info = {
            'mscoco14': {
                'name': 'MSCOCO 2014',
                'description': 'Object detection with bounding boxes and captions',
                'splits': ['train', 'val'],
                'modalities': ['image', 'text', 'bbox'],
                'task_types': ['object_detection', 'captioning', 'segmentation']
            },
            'visual_genome': {
                'name': 'Visual Genome',
                'description': 'Scene graphs with objects, attributes, and relationships',
                'splits': ['train', 'val', 'test'],
                'modalities': ['image', 'text', 'bbox', 'scene_graph'],
                'task_types': ['visual_reasoning', 'relationship_detection']
            },
            'vcr': {
                'name': 'Visual Commonsense Reasoning',
                'description': 'Question-answer-rationale with visual grounding',
                'splits': ['train', 'val', 'test'],
                'modalities': ['image', 'text', 'bbox'],
                'task_types': ['qa', 'reasoning', 'visual_grounding']
            },
            'sherlock': {
                'name': 'Sherlock',
                'description': 'Clue-inference pairs with bounding boxes',
                'splits': ['train', 'val'],
                'modalities': ['image', 'text', 'bbox'],
                'task_types': ['inference', 'visual_reasoning']
            },
            'mm-niah': {
                'name': 'MM-NIAH',
                'description': 'Needle in multimodal haystack (retrieval/counting/reasoning)',
                'splits': ['val', 'test'],
                'modalities': ['image', 'text'],
                'task_types': ['retrieval', 'counting', 'reasoning']
            },
            'mmmu': {
                'name': 'MMMU',
                'description': 'Massive multi-discipline multimodal understanding',
                'splits': ['dev', 'validation', 'test'],
                'modalities': ['image', 'text'],
                'task_types': ['qa', 'multiple_choice', 'reasoning']
            },
            'mmdocir': {
                'name': 'MMDocIR',
                'description': 'Multi-modal document retrieval',
                'splits': ['test'],
                'modalities': ['image', 'text', 'bbox', 'layout'],
                'task_types': ['document_qa', 'retrieval']
            }
        }

        return info.get(dataset_id.lower(), {})

    def filter_by_attributes(self,
                           samples: List[Dict],
                           filters: Dict[str, Any]) -> List[Dict]:
        """
        Filter samples by attributes.

        Args:
            samples: List of samples
            filters: Dictionary of attribute filters
                e.g., {'num_objects': (2, 5), 'has_relationships': True}

        Returns:
            Filtered samples
        """
        filtered = []

        for sample in samples:
            match = True

            for key, value in filters.items():
                # Range filter
                if isinstance(value, tuple) and len(value) == 2:
                    min_val, max_val = value
                    sample_val = self._get_nested_value(sample, key)
                    if not (min_val <= sample_val <= max_val):
                        match = False
                        break
                # Exact match
                else:
                    sample_val = self._get_nested_value(sample, key)
                    if sample_val != value:
                        match = False
                        break

            if match:
                filtered.append(sample)

        return filtered

    @staticmethod
    def _get_nested_value(d: Dict, key: str) -> Any:
        """Get nested dictionary value by dot-separated key."""
        keys = key.split('.')
        value = d
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
            else:
                return None
        return value