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
import random
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

    def __init__(self, data_root: str = "d:\\install_file\\M3Bench\\dataset"):
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

        if max_samples and max_samples < len(data):
            data = random.sample(data, max_samples)
        elif max_samples:
            data = data[:max_samples]  # 数据量不足时使用全部

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
            self.data_root / "MSCOCO",  # New structure: d:\install_file\M3Bench\dataset\MSCOCO
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

        # For test split, try image_info_test2014.json if instances_test2014.json not found
        if split == 'test' and not ann_file.exists():
            ann_file = data_dir / "annotations" / "image_info_test2014.json"
            cap_file = data_dir / "annotations" / "captions_val2014.json"  # Use val captions as fallback

            # Fallback to example directory for test info
            if not ann_file.exists():
                ann_file = data_dir / "example" / "image_info_test2014.json"

        if not ann_file.exists():
            logger.error(f"MSCOCO标注文件不存在: {ann_file}")
            return []

        # Load annotations
        try:
            with open(ann_file, 'r', encoding='utf-8') as f:
                ann_data = json.load(f)

            # Try to load captions if file exists
            cap_data = {}
            if cap_file.exists():
                with open(cap_file, 'r', encoding='utf-8') as f:
                    cap_data = json.load(f)
        except Exception as e:
            logger.error(f"加载MSCOCO标注文件失败: {e}")
            return []

        # Build image id to annotations mapping
        image_to_anns = {}
        if 'annotations' in ann_data:
            for ann in ann_data['annotations']:
                img_id = ann['image_id']
                if img_id not in image_to_anns:
                    image_to_anns[img_id] = []
                image_to_anns[img_id].append(ann)

        # Build image id to captions mapping
        image_to_caps = {}
        if 'annotations' in cap_data:
            for cap in cap_data['annotations']:
                img_id = cap['image_id']
                if img_id not in image_to_caps:
                    image_to_caps[img_id] = []
                image_to_caps[img_id].append(cap['caption'])

        # Build category id to name mapping
        cat_id_to_name = {}
        if 'categories' in ann_data:
            cat_id_to_name = {cat['id']: cat['name'] for cat in ann_data['categories']}

        # Construct samples
        samples = []
        if 'images' in ann_data:
            for img_info in ann_data['images']:
                img_id = img_info['id']

                # Get objects
                objects = []
                if img_id in image_to_anns:
                    for ann in image_to_anns[img_id]:
                        if 'category_id' in ann and ann['category_id'] in cat_id_to_name:
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
                
                # 首先检查实际存在的目录
                image_path = None
                
                # 优先检查val2014目录（因为它存在且包含图片）
                val_dir = data_dir / "val2014"
                if val_dir.exists():
                    val_img_path = val_dir / file_name
                    if val_img_path.exists():
                        image_path = str(val_img_path)
                        logger.debug(f"Found image in val2014: {image_path}")
                
                # 如果val2014中没找到，检查其他可能的目录
                if image_path is None:
                    possible_dirs = ['test2014', 'images', 'example']
                    for dir_name in possible_dirs:
                        dir_path = data_dir / dir_name
                        if dir_path.exists():
                            img_path = dir_path / file_name
                            if img_path.exists():
                                image_path = str(img_path)
                                logger.debug(f"Found image in {dir_name}: {image_path}")
                                break
                
                # 如果还是没找到，尝试直接在data_dir下查找
                if image_path is None:
                    img_path = data_dir / file_name
                    if img_path.exists():
                        image_path = str(img_path)
                        logger.debug(f"Found image in root: {image_path}")
                
                # 如果仍然没找到，使用val2014作为默认目录（因为它实际存在）
                if image_path is None:
                    image_path = str(data_dir / "val2014" / file_name)
                    logger.debug(f"Using default path in val2014: {image_path}")

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

        logger.info(f"加载MSCOCO14数据集完成，共 {len(samples)} 张图片")
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
        data_dir = self.data_root / "visual_genome"

        # Load scene graphs
        sg_file = data_dir / f"scene_graphs_{split}.json"

        if not sg_file.exists():
            logger.warning(f"Scene graph file not found: {sg_file}")
            return []

        with open(sg_file, 'r') as f:
            scene_graphs = json.load(f)

        samples = []
        for sg in scene_graphs:
            # Parse objects
            objects = {}
            for obj in sg.get('objects', []):
                obj_id = str(obj['object_id'])
                objects[obj_id] = {
                    'name': obj['names'][0] if obj.get('names') else '',
                    'bbox': [obj['x'], obj['y'], obj['w'], obj['h']],
                    'attributes': obj.get('attributes', [])
                }

            # Parse relationships
            relationships = []
            for rel in sg.get('relationships', []):
                relationships.append({
                    'subject': str(rel['subject_id']),
                    'predicate': rel['predicate'],
                    'object': str(rel['object_id'])
                })

            samples.append({
                'dataset_id': 'visual_genome',
                'sample_id': f"vg_{sg['image_id']}",
                'image_id': sg['image_id'],
                'image_path': str(data_dir / "images" / f"{sg['image_id']}.jpg"),
                'scene_graph': {
                    'objects': objects,
                    'relationships': relationships
                }
            })

        return samples

    # ==================== VCR ====================

    def _load_vcr(self, split: str = "train", **kwargs) -> List[Dict]:
        """
        Load Visual Commonsense Reasoning dataset.

        Returns format:
        {
            'dataset_id': 'vcr',
            'sample_id': str,
            'image_path': str,
            'question': List[Union[str, List[int]]],
            'answer_choices': List[List[Union[str, List[int]]]],
            'answer_label': int,
            'rationale_choices': List[List[Union[str, List[int]]]],
            'rationale_label': int,
            'objects': List[str],
            'metadata': {'boxes', 'segms', 'width', 'height'}
        }
        """
        # Try both directory names (handle typo in original dataset)
        data_dir = self.data_root / "VisualCommenReasoning"
        if not data_dir.exists():
            data_dir = self.data_root / "VisualCommenReasnoning"  # Typo version

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

                    # Skip metadata loading for faster processing
                    # metadata can be loaded later if needed
                    metadata = {}

                    samples.append({
                        'dataset_id': 'vcr',
                        'sample_id': f"vcr_{split}_{valid_count}",
                        'annot_id': item.get('annot_id', f"{split}-{valid_count}"),
                        'image_path': str(data_dir / 'vcr1images' / item['img_fn']),
                        'question': item['question'],
                        'answer_choices': item['answer_choices'],
                        'answer_label': item['answer_label'],
                        'rationale_choices': item.get('rationale_choices', []),
                        'rationale_label': item.get('rationale_label', -1),
                        'objects': item['objects'],
                        'metadata_fn': item.get('metadata_fn', ''),  # Store path for later loading
                        'metadata': metadata
                    })
                    valid_count += 1

                except (json.JSONDecodeError, KeyError) as e:
                    error_count += 1
                    if error_count <= 5:  # Only log first 5 errors
                        logger.warning(f"Skipping invalid line {idx+1} in VCR {split}: {str(e)[:100]}")
                    continue

        if error_count > 0:
            logger.info(f"Loaded {valid_count} valid VCR samples, skipped {error_count} invalid lines")

        return samples

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
        data_dir = self.data_root / "Sherlock"

        # Load annotations
        if split == "train":
            ann_file = data_dir / "sherlock_train_v1_1.json"
        else:
            ann_file = data_dir / "sherlock_val_with_split_idxs_v1_1.json"

        if not ann_file.exists():
            logger.warning(f"Sherlock file not found: {ann_file}")
            return []

        with open(ann_file, 'r') as f:
            data = json.load(f)

        samples = []
        for item in data:
            inputs = item['inputs']
            samples.append({
                'dataset_id': 'sherlock',
                'sample_id': item['instance_id'],
                'image_path': inputs['image']['url'],
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
        """
        import glob
        import pandas as pd
        import pyarrow
        from pathlib import Path
        
        data_dir = self.data_root / "ScienceQA"
        images_output_dir = data_dir / "images"
        images_output_dir.mkdir(exist_ok=True)

        parquet_files = list(glob.glob(str(data_dir / f"{split}-*.parquet")))
        if not parquet_files:
            logger.warning(f"ScienceQA parquet file not found for split {split}")
            return []

        parquet_file = parquet_files[0]
        logger.info(f"Loading ScienceQA from: {parquet_file}")

        try:
            df = pd.read_parquet(parquet_file)
            samples = []
            for idx, row in df.iterrows():
                image_bytes = row.get('image')
                image_path = None
                
                if image_bytes is not None and len(image_bytes) > 0:
                    if isinstance(image_bytes, dict) and 'bytes' in image_bytes:
                        image_bytes = image_bytes['bytes']
                    
                    if image_bytes is not None and isinstance(image_bytes, (bytes, bytearray)):
                        image_filename = f"scienceqa_{split}_{idx:06d}.png"
                        image_path = str(images_output_dir / image_filename)
                        try:
                            with open(image_path, 'wb') as f:
                                f.write(image_bytes)
                        except Exception as img_err:
                            logger.warning(f"Failed to save image for sample {idx}: {img_err}")
                            image_path = None
                
                sample = {
                    'dataset_id': 'scienceqa',
                    'sample_id': f"scienceqa_{split}_{idx:06d}",
                    'image': image_bytes,
                    'image_path': image_path,
                    'question': row.get('question', ''),
                    'answer': row.get('answer', 0),
                    'subject': row.get('subject', ''),
                    'topic': row.get('topic', ''),
                    'grade': row.get('grade', ''),
                    'task': row.get('task', ''),
                    'category': row.get('category', ''),
                    'hint': row.get('hint', ''),
                    'lecture': row.get('lecture', ''),
                    'solution': row.get('solution', ''),
                    'choices': row.get('choices', row.get('option', []))
                }
                samples.append(sample)
            
            logger.info(f"Loaded {len(samples)} samples from ScienceQA ({split}), saved {sum(1 for s in samples if s['image_path'])} images")
            return samples
            
        except Exception as e:
            logger.error(f"Error loading ScienceQA: {type(e).__name__}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return []

    # ==================== RealworldQA ====================

    def _load_realworldqa(self, split: str = "test", **kwargs) -> List[Dict]:
        """
        Load RealworldQA dataset.
        使用已保存的图片文件（位于 data/images 目录）

        Returns format:
        {
            'dataset_id': 'realworldqa',
            'sample_id': str,
            'image_path': str,
            'question': str,
            'answer': str
        }
        """
        import glob
        import pandas as pd
        import pyarrow
        from pathlib import Path
        import os
        
        data_dir = self.data_root / "xai-org_RealworldQA" / "data"
        images_dir = data_dir / "images"
        
        # 图片命名格式: realworldqa_000000.png, realworldqa_000001.png, ...
        image_pattern = str(images_dir / "realworldqa_*.png")
        saved_images = {}
        
        # 收集已保存的图片文件
        for img_path in glob.glob(image_pattern):
            img_name = os.path.basename(img_path)
            # 提取序号: realworldqa_000000.png -> 0
            try:
                idx = int(img_name.replace('realworldqa_', '').replace('.png', ''))
                saved_images[idx] = img_path
            except ValueError:
                pass
        
        logger.info(f"Found {len(saved_images)} saved images in {images_dir}")

        # 使用glob模式匹配parquet文件
        parquet_files = list(glob.glob(str(data_dir / f"{split}-*.parquet")))

        if not parquet_files:
            logger.warning(f"RealworldQA parquet file not found for split {split}")
            return []

        logger.info(f"Found {len(parquet_files)} RealworldQA parquet files")

        all_samples = []
        sample_idx = 0
        
        for parquet_file in sorted(parquet_files):
            logger.info(f"Loading RealworldQA from: {parquet_file}")
            
            # 尝试使用pyarrow引擎
            try:
                df = pd.read_parquet(parquet_file, engine='pyarrow')
                logger.info(f"Successfully loaded {parquet_file} with pyarrow engine")
            except Exception as e:
                logger.warning(f"Failed to load {parquet_file} with pyarrow: {e}")
                # 尝试使用fastparquet引擎
                try:
                    df = pd.read_parquet(parquet_file, engine='fastparquet')
                    logger.info(f"Successfully loaded {parquet_file} with fastparquet engine")
                except Exception as e2:
                    logger.error(f"Failed to load {parquet_file} with both engines: {e2}")
                    continue
            
            for idx, row in df.iterrows():
                # 使用已保存的图片路径
                image_path = saved_images.get(sample_idx)
                
                sample = {
                    'dataset_id': 'realworldqa',
                    'sample_id': f"realworldqa_{sample_idx}",
                    'image_path': image_path,
                    'question': row.get('question', ''),
                    'answer': row.get('answer', '')
                }
                all_samples.append(sample)
                sample_idx += 1
        
        if all_samples:
            matched_images = sum(1 for s in all_samples if s['image_path'])
            logger.info(f"Loaded {len(all_samples)} samples from RealworldQA ({split}), matched {matched_images} images")
        else:
            logger.warning(f"No samples loaded from RealworldQA ({split})")
        
        return all_samples

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
        Load DocVQA dataset.

        Returns format:
        {
            'dataset_id': 'docvqa',
            'sample_id': str,
            'question': str,
            'answer': str,
            'answers': List[str],
            'image': bytes,
            'image_path': str
        }
        """
        import glob
        import pandas as pd
        import pyarrow
        from pathlib import Path
        
        data_dir = self.data_root / "lmms-lab_DocVQA" / "DocVQA"

        # 创建图片输出目录
        images_output_dir = data_dir / "images"
        images_output_dir.mkdir(exist_ok=True)

        # 使用glob模式匹配parquet文件
        parquet_files = list(glob.glob(str(data_dir / f"{split}-*.parquet")))

        if not parquet_files:
            logger.warning(f"DocVQA parquet file not found for split {split}")
            return []

        logger.info(f"Found {len(parquet_files)} DocVQA parquet files")

        try:
            all_samples = []
            for parquet_file in parquet_files:
                logger.info(f"Loading DocVQA from: {parquet_file}")
                df = pd.read_parquet(parquet_file)
                
                for idx, row in df.iterrows():
                    # 处理嵌入的图片
                    image_bytes = row.get('image')
                    image_path = None
                    
                    if image_bytes is not None and len(image_bytes) > 0:
                        # pyarrow可能使用字典格式存储图片 {'bytes': ..., 'path': ...}
                        if isinstance(image_bytes, dict):
                            # 从字典中提取bytes
                            if 'bytes' in image_bytes:
                                image_bytes = image_bytes['bytes']
                            else:
                                image_bytes = None
                        
                        if image_bytes is not None and isinstance(image_bytes, (bytes, bytearray)):
                            image_filename = f"docvqa_{split}_{len(all_samples):06d}.png"
                            image_path = str(images_output_dir / image_filename)
                            
                            try:
                                with open(image_path, 'wb') as f:
                                    f.write(image_bytes)
                            except Exception as img_err:
                                logger.warning(f"Failed to save image for sample {len(all_samples)}: {img_err}")
                                image_path = None
                    
                    # 获取答案列表（正确使用answers字段）
                    answers_list = []
                    if 'answers' in row:
                        answers_val = row['answers']
                        if isinstance(answers_val, list):
                            answers_list = answers_val
                        else:
                            answers_list = [str(answers_val)]
                    
                    # 选择第一个答案作为主要答案
                    answer = answers_list[0] if answers_list else ''
                    
                    sample = {
                        'dataset_id': 'docvqa',
                        'sample_id': f"docvqa_{len(all_samples)}",
                        'questionId': row.get('questionId', ''),
                        'question': row.get('question', ''),
                        'answer': answer,
                        'answers': answers_list,
                        'question_types': row.get('question_types', []),
                        'image': image_bytes,
                        'image_path': image_path,
                        'docId': str(row.get('docId', '')),
                        'data_split': row.get('data_split', split)
                    }
                    all_samples.append(sample)
            
            logger.info(f"Loaded {len(all_samples)} samples from DocVQA ({split}), saved {sum(1 for s in all_samples if s['image_path']) } images")
            return all_samples
            
        except Exception as e:
            logger.error(f"Error loading DocVQA: {type(e).__name__}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return []

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
        """Load GQA dataset with scene graphs."""
        logger.warning("GQA loader not yet implemented - returning empty")
        return []

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