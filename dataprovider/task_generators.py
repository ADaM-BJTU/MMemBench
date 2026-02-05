"""
Task-specific generators for M3Bench
=====================================
Contains specialized generation logic for each task type.
Separated from main generator for better modularity.
"""

import random
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class AttributeComparisonGenerator:
    """Generate Attribute Comparison tasks."""

    @staticmethod
    def generate_from_mscoco(source_data: List[Dict],
                            num_samples: int,
                            task_config: Dict[str, Any],
                            templates: Dict[str, Any]) -> List[Dict]:
        """
        Generate Attribute Comparison tasks from MSCOCO.

        Strategy:
        1. Find images with same object category
        2. Compare their attributes (position, size, count)
        3. Generate questions about differences
        """
        samples = []
        n_images = task_config.get('n_images', 3)
        comparison_types = task_config.get('comparison_types', ['count_comparison', 'same_object_different_attributes', 'find_by_attribute'])
        target_categories = task_config.get('target_categories', 'all')

        # Group samples by category
        category_to_samples = {}
        for sample in source_data:
            for obj in sample['objects']:
                cat = obj['category']
                if target_categories == 'all' or cat in target_categories:
                    if cat not in category_to_samples:
                        category_to_samples[cat] = []
                    if sample not in category_to_samples[cat]:
                        category_to_samples[cat].append(sample)

        # Filter categories with enough samples
        valid_categories = {
            cat: samples_list
            for cat, samples_list in category_to_samples.items()
            if len(samples_list) >= n_images
        }

        if not valid_categories:
            logger.warning("No valid categories for attribute comparison")
            return []

        attempts = 0
        max_attempts = num_samples * 20

        while len(samples) < num_samples and attempts < max_attempts:
            attempts += 1

            # Select random category and comparison type
            category = random.choice(list(valid_categories.keys()))
            comparison_type = random.choice(comparison_types)

            # Select images
            candidate_samples = valid_categories[category]
            selected_samples = random.sample(candidate_samples, min(n_images, len(candidate_samples)))

            # Generate task based on comparison type
            task = None

            if comparison_type == "count_comparison":
                task = AttributeComparisonGenerator._generate_count_comparison(
                    selected_samples, category, templates
                )

            elif comparison_type == "same_object_different_attributes":
                task = AttributeComparisonGenerator._generate_spatial_comparison(
                    selected_samples, category, templates
                )

            elif comparison_type == "find_by_attribute":
                task = AttributeComparisonGenerator._generate_find_by_attribute(
                    selected_samples, category, templates
                )

            if task:
                task['task_id'] = f"ac_mscoco_{len(samples)}"
                task['task_type'] = 'attribute_comparison'
                task['metadata'] = {
                    'source_dataset': 'mscoco14',
                    'comparison_type': comparison_type,
                    'target_category': category
                }
                samples.append(task)

        logger.info(f"Generated {len(samples)} attribute comparison tasks from {attempts} attempts")
        return samples

    @staticmethod
    def _generate_count_comparison(samples: List[Dict],
                                   category: str,
                                   templates: Dict[str, Any]) -> Optional[Dict]:
        """Generate count comparison task."""
        # Count objects in each image
        counts = []
        for sample in samples:
            count = sum(1 for obj in sample['objects'] if obj['category'] == category)
            counts.append(count)

        # Need at least some variation
        if len(set(counts)) < 2:
            return None

        max_count = max(counts)
        max_indices = [i for i, c in enumerate(counts) if c == max_count]
        answer_idx = max_indices[0]

        # Get question template
        question_templates = templates.get('question_templates', [])
        if not question_templates:
            question = f"Which image has the most {category}s?"
        else:
            template = random.choice(question_templates)
            question = template.format(category=category, target_category=category)

        answer = f"Image {answer_idx} has {max_count} {category}(s)"

        # Reasoning evidence
        evidence = []
        for i, sample in enumerate(samples):
            objs_in_image = [obj for obj in sample['objects'] if obj['category'] == category]
            evidence.append({
                'image_idx': i,
                'count': counts[i],
                'objects': [
                    {
                        'category': obj['category'],
                        'bbox': obj['bbox'],
                        'area': obj['area']
                    }
                    for obj in objs_in_image
                ]
            })

        return {
            'images': [s['image_path'] for s in samples],
            'question': question,
            'answer': answer,
            'answer_image_idx': answer_idx,
            'comparison_metric': 'count',
            'comparison_values': counts,
            'reasoning_evidence': evidence,
            'reasoning_depth': 2  # Count + compare
        }

    @staticmethod
    def _generate_spatial_comparison(samples: List[Dict],
                                     category: str,
                                     templates: Dict[str, Any]) -> Optional[Dict]:
        """Generate spatial attribute comparison task."""
        # Find objects in each image and compare their positions
        image_objects = []
        for sample in samples:
            objs = [obj for obj in sample['objects'] if obj['category'] == category]
            if objs:
                # Use the first/largest object
                obj = max(objs, key=lambda x: x['area'])
                x, y, w, h = obj['bbox']
                image_objects.append({
                    'sample': sample,
                    'object': obj,
                    'center_x': x + w/2,
                    'center_y': y + h/2,
                    'relative_x': (x + w/2) / sample['width'],  # Normalized position
                    'relative_y': (y + h/2) / sample['height']
                })
            else:
                return None  # Skip if not all images have the object

        if len(image_objects) < 2:
            return None

        # Find interesting attribute to compare
        # Example: leftmost, rightmost, topmost, bottommost
        leftmost_idx = min(range(len(image_objects)), key=lambda i: image_objects[i]['relative_x'])
        rightmost_idx = max(range(len(image_objects)), key=lambda i: image_objects[i]['relative_x'])
        topmost_idx = min(range(len(image_objects)), key=lambda i: image_objects[i]['relative_y'])
        bottommost_idx = max(range(len(image_objects)), key=lambda i: image_objects[i]['relative_y'])

        # Choose a comparison that has clear difference
        comparisons = [
            ('leftmost', leftmost_idx, image_objects[leftmost_idx]['relative_x']),
            ('rightmost', rightmost_idx, image_objects[rightmost_idx]['relative_x']),
            ('topmost', topmost_idx, image_objects[topmost_idx]['relative_y']),
            ('bottommost', bottommost_idx, image_objects[bottommost_idx]['relative_y'])
        ]

        # Select one with clear winner
        selected_comparison = random.choice(comparisons)
        position_type, answer_idx, position_value = selected_comparison

        question = f"In which image is the {category} positioned {position_type}?"
        answer = f"Image {answer_idx} has the {category} {position_type}"

        evidence = []
        for i, img_obj in enumerate(image_objects):
            evidence.append({
                'image_idx': i,
                'object': {
                    'category': category,
                    'bbox': img_obj['object']['bbox'],
                    'center': (img_obj['center_x'], img_obj['center_y']),
                    'relative_position': (img_obj['relative_x'], img_obj['relative_y'])
                }
            })

        return {
            'images': [io['sample']['image_path'] for io in image_objects],
            'question': question,
            'answer': answer,
            'answer_image_idx': answer_idx,
            'comparison_metric': f'position_{position_type}',
            'comparison_values': [io['relative_x'] if 'most' in position_type and 'left' in position_type or 'right' in position_type
                                 else io['relative_y'] for io in image_objects],
            'reasoning_evidence': evidence,
            'reasoning_depth': 2
        }

    @staticmethod
    def _generate_find_by_attribute(samples: List[Dict],
                                    category: str,
                                    templates: Dict[str, Any]) -> Optional[Dict]:
        """Generate 'find by attribute' task."""
        # Find images where the target object has distinguishing attributes
        # For COCO, we can use size as an attribute

        image_objects = []
        for sample in samples:
            objs = [obj for obj in sample['objects'] if obj['category'] == category]
            if objs:
                total_area = sum(obj['area'] for obj in objs)
                largest_obj = max(objs, key=lambda x: x['area'])
                image_objects.append({
                    'sample': sample,
                    'object': largest_obj,
                    'total_area': total_area,
                    'count': len(objs)
                })
            else:
                return None

        if len(image_objects) < 2:
            return None

        # Find the image with the largest object
        largest_idx = max(range(len(image_objects)), key=lambda i: image_objects[i]['object']['area'])
        smallest_idx = min(range(len(image_objects)), key=lambda i: image_objects[i]['object']['area'])

        # Choose attribute to query
        attributes = ['largest', 'smallest']
        chosen_attr = random.choice(attributes)

        if chosen_attr == 'largest':
            answer_idx = largest_idx
            question = f"Which image contains the largest {category}?"
        else:
            answer_idx = smallest_idx
            question = f"Which image contains the smallest {category}?"

        answer = f"Image {answer_idx}"

        evidence = []
        for i, img_obj in enumerate(image_objects):
            evidence.append({
                'image_idx': i,
                'object': {
                    'category': category,
                    'bbox': img_obj['object']['bbox'],
                    'area': img_obj['object']['area']
                },
                'is_answer': i == answer_idx
            })

        return {
            'images': [io['sample']['image_path'] for io in image_objects],
            'question': question,
            'answer': answer,
            'answer_image_idx': answer_idx,
            'comparison_metric': f'size_{chosen_attr}',
            'comparison_values': [io['object']['area'] for io in image_objects],
            'reasoning_evidence': evidence,
            'reasoning_depth': 2
        }

    @staticmethod
    def generate_from_vcr(source_data: List[Dict],
                         num_samples: int,
                         task_config: Dict[str, Any],
                         templates: Dict[str, Any]) -> List[Dict]:
        """
        Generate Attribute Comparison tasks from VCR.

        VCR has rich contextual information, we can compare:
        1. Number of people across scenes
        2. Relationship complexity
        3. Context matching
        """
        samples = []
        n_images = task_config.get('n_images', 3)
        use_metadata = task_config.get('use_metadata', True)

        attempts = 0
        max_attempts = num_samples * 20

        while len(samples) < num_samples and attempts < max_attempts:
            attempts += 1

            # Select random samples
            if len(source_data) < n_images:
                break

            selected_samples = random.sample(source_data, n_images)

            # Compare number of people (most common object in VCR)
            people_counts = []
            for sample in selected_samples:
                count = sum(1 for obj in sample['objects'] if obj == 'person')
                people_counts.append(count)

            # Need variation
            if len(set(people_counts)) < 2:
                continue

            max_count = max(people_counts)
            answer_idx = people_counts.index(max_count)

            question = "Which image shows the most people?"
            answer = f"Image {answer_idx} with {max_count} people"

            evidence = []
            for i, sample in enumerate(selected_samples):
                evidence.append({
                    'image_idx': i,
                    'people_count': people_counts[i],
                    'total_objects': len(sample['objects']),
                    'objects': sample['objects']
                })

            task = {
                'task_id': f"ac_vcr_{len(samples)}",
                'task_type': 'attribute_comparison',
                'images': [s['image_path'] for s in selected_samples],
                'question': question,
                'answer': answer,
                'answer_image_idx': answer_idx,
                'comparison_metric': 'people_count',
                'comparison_values': people_counts,
                'reasoning_evidence': evidence,
                'reasoning_depth': 2,
                'metadata': {
                    'source_dataset': 'vcr',
                    'comparison_type': 'count_comparison'
                }
            }

            samples.append(task)

        logger.info(f"Generated {len(samples)} VCR attribute comparison tasks")
        return samples


class EnhancedVNFGenerator:
    """Enhanced Visual Noise Filtering generator."""

    @staticmethod
    def generate_from_mscoco(source_data: List[Dict],
                            num_samples: int,
                            task_config: Dict[str, Any],
                            templates: Dict[str, Any]) -> List[Dict]:
        """
        Enhanced VNF generation from MSCOCO with better distractor selection.
        """
        samples = []
        n_distractors = task_config.get('n_distractors', 3)
        distractor_strategy = task_config.get('distractor_strategy', 'non_overlapping_categories')

        valid_data = [s for s in source_data
                     if len(s['objects']) > 0 and len(s['captions']) > 0]

        if len(valid_data) < num_samples * (n_distractors + 1):
            logger.warning(f"Not enough valid data for VNF")
            num_samples = len(valid_data) // (n_distractors + 1)

        random.shuffle(valid_data)

        for idx in range(num_samples):
            target_sample = valid_data[idx]

            # Select unique object as target
            target_obj = random.choice(target_sample['objects'])
            target_category = target_obj['category']

            # Enhanced distractor selection
            if distractor_strategy == 'non_overlapping_categories':
                distractors = EnhancedVNFGenerator._select_non_overlapping_distractors(
                    target_sample, valid_data, n_distractors
                )
            elif distractor_strategy == 'confusing':
                distractors = EnhancedVNFGenerator._select_confusing_distractors(
                    target_sample, valid_data, n_distractors
                )
            else:
                distractors = valid_data[idx+1:idx+1+n_distractors]

            if len(distractors) < n_distractors:
                continue

            # Shuffle images
            all_images = distractors + [target_sample]
            random.shuffle(all_images)
            target_idx = all_images.index(target_sample)

            # Generate question (multiple templates)
            question_templates = templates.get('question_templates', [])
            # Handle different template formats
            if isinstance(question_templates, dict):
                question_templates = question_templates.get('object_based', [])
            if question_templates:
                template = random.choice(question_templates)
                # Try different variable names for compatibility
                try:
                    question = template.format(target=target_category)
                except KeyError:
                    try:
                        question = template.format(target_category=target_category)
                    except KeyError:
                        question = f"Which image contains a {target_category}?"
            else:
                question = f"Which image contains a {target_category}?"

            answer = f"Image {target_idx}"

            evidence = {
                'target': {
                    'image_idx': target_idx,
                    'target_object': {
                        'category': target_category,
                        'bbox': target_obj['bbox'],
                        'area': target_obj['area']
                    },
                    'all_objects': [obj['category'] for obj in target_sample['objects']]
                },
                'distractors': [
                    {
                        'image_idx': i if i < target_idx else i + 1,
                        'objects': [obj['category'] for obj in img['objects']]
                    }
                    for i, img in enumerate(distractors)
                ]
            }

            samples.append({
                'task_id': f"vnf_mscoco_{idx}",
                'task_type': 'visual_noise_filtering',
                'images': [img['image_path'] for img in all_images],
                'question': question,
                'target_image_idx': target_idx,
                'answer': answer,
                'reasoning_depth': 1,
                'reasoning_evidence': evidence,
                'metadata': {
                    'source_dataset': 'mscoco14',
                    'n_total_images': len(all_images),
                    'target_category': target_category,
                    'distractor_strategy': distractor_strategy
                }
            })

        return samples

    @staticmethod
    def _select_non_overlapping_distractors(target_sample: Dict,
                                           candidates: List[Dict],
                                           n: int) -> List[Dict]:
        """Select distractors with no overlapping object categories."""
        target_categories = {obj['category'] for obj in target_sample['objects']}
        distractors = []

        for candidate in candidates:
            if candidate['sample_id'] == target_sample['sample_id']:
                continue

            candidate_categories = {obj['category'] for obj in candidate['objects']}

            if not target_categories.intersection(candidate_categories):
                distractors.append(candidate)

            if len(distractors) >= n:
                break

        return distractors

    @staticmethod
    def _select_confusing_distractors(target_sample: Dict,
                                     candidates: List[Dict],
                                     n: int) -> List[Dict]:
        """Select distractors that are visually similar but don't contain target."""
        target_obj = random.choice(target_sample['objects'])
        target_category = target_obj['category']

        # Find similar categories (e.g., cat vs dog, car vs truck)
        similar_categories = {
            'cat': ['dog', 'sheep', 'cow'],
            'dog': ['cat', 'horse', 'bear'],
            'car': ['truck', 'bus', 'train'],
            'person': ['person'],  # Use different contexts
            'chair': ['couch', 'bed', 'bench']
        }

        similar_cats = similar_categories.get(target_category, [])
        distractors = []

        for candidate in candidates:
            if candidate['sample_id'] == target_sample['sample_id']:
                continue

            # Check if has similar object but not target
            candidate_categories = {obj['category'] for obj in candidate['objects']}

            if target_category not in candidate_categories:
                if any(cat in candidate_categories for cat in similar_cats):
                    distractors.append(candidate)

            if len(distractors) >= n:
                break

        # Fallback to non-overlapping if not enough confusing ones
        if len(distractors) < n:
            return EnhancedVNFGenerator._select_non_overlapping_distractors(
                target_sample, candidates, n
            )

        return distractors[:n]

    @staticmethod
    def _vcr_tokens_to_text(tokens: List[str], objects: List[str]) -> str:
        """Convert VCR tokens to text (fallback implementation)."""
        # Simple implementation to convert tokens to readable text
        text = []
        for token in tokens:
            # Handle case where token is not a string
            if not isinstance(token, str):
                # If token is a list, join it
                if isinstance(token, list):
                    text.append(' '.join(str(t) for t in token))
                else:
                    text.append(str(token))
                continue
                
            if token.startswith('@'):
                # Handle object references like @1
                try:
                    obj_idx = int(token[1:])
                    if obj_idx < len(objects):
                        text.append(objects[obj_idx])
                    else:
                        text.append('object')
                except (ValueError, IndexError):
                    text.append('object')
            else:
                text.append(token)
        return ' '.join(text)

    @staticmethod
    def generate_from_vcr(source_data: List[Dict],
                         num_samples: int,
                         task_config: Dict[str, Any],
                         templates: Dict[str, Any]) -> List[Dict]:
        """Generate VNF tasks from VCR using natural Q&A."""
        samples = []
        n_distractors = task_config.get('n_distractors', 3)
        use_original_qa = task_config.get('use_original_qa', True)

        # Filter by quality
        quality_filters = task_config.get('quality_filters', {})
        valid_data = source_data

        if 'answer_likelihood' in quality_filters:
            valid_likelihoods = quality_filters['answer_likelihood']
            valid_data = [s for s in valid_data
                         if s.get('answer_likelihood') in valid_likelihoods]

        random.shuffle(valid_data)

        for idx in range(min(num_samples, len(valid_data) - n_distractors)):
            target_sample = valid_data[idx]

            # Use VCR's natural question or fallback
            try:
                # Try original approach first
                from dataprovider.generator import DataGenerator
                question_text = DataGenerator._vcr_tokens_to_text(
                    target_sample['question'],
                    target_sample['objects']
                )
            except ImportError:
                # Fallback implementation
                question_text = EnhancedVNFGenerator._vcr_tokens_to_text(
                    target_sample.get('question', ['Which', 'image', 'shows', 'the', 'correct', 'scene?']),
                    target_sample['objects']
                )
            except (KeyError, TypeError):
                # Fallback if question or objects not available
                question_text = "Which image shows the correct scene?"

            # Select distractors (different scenes)
            distractors = valid_data[idx+1:idx+1+n_distractors]

            all_images = distractors + [target_sample]
            random.shuffle(all_images)
            target_idx = all_images.index(target_sample)

            # Get answer
            try:
                # Try original approach first
                from dataprovider.generator import DataGenerator
                answer_tokens = target_sample['answer_choices'][target_sample['answer_label']]
                answer_text = DataGenerator._vcr_tokens_to_text(
                    answer_tokens,
                    target_sample['objects']
                )
            except (ImportError, KeyError, TypeError):
                # Fallback to simple answer
                answer_text = "The correct scene"

            evidence = {
                'target': {
                    'image_idx': target_idx,
                    'question_orig': target_sample.get('question_orig'),
                    'answer_orig': target_sample.get('answer_orig'),
                    'rationale_orig': target_sample.get('rationale_orig'),
                    'objects': target_sample['objects']
                }
            }

            samples.append({
                'task_id': f"vnf_vcr_{idx}",
                'task_type': 'visual_noise_filtering',
                'images': [img['image_path'] for img in all_images],
                'question': question_text,
                'target_image_idx': target_idx,
                'answer': f"Image {target_idx}: {answer_text}",
                'reasoning_depth': 1,
                'reasoning_evidence': evidence,
                'metadata': {
                    'source_dataset': 'vcr',
                    'n_total_images': len(all_images),
                    'distractor_strategy': 'temporal_separation',
                    'uses_original_qa': use_original_qa
                }
            })

        return samples


class AttributeBridgeReasoningGenerator:
    """Generate Attribute Bridge Reasoning tasks from MSCOCO14.
    
    ABR任务需要多跳推理：
    1. 定位某个物体
    2. 基于其属性找到相关物体
    3. 回答关于关联物体的问题
    """
    
    @staticmethod
    def generate_from_mscoco(source_data: List[Dict],
                            num_samples: int,
                            task_config: Dict[str, Any],
                            templates: Dict[str, Any]) -> List[Dict]:
        """
        Generate ABR tasks from MSCOCO using spatial relationships.
        
        Strategy:
        1. Find images with multiple objects
        2. Identify spatial relationships between objects
        3. Generate multi-hop reasoning questions
        """
        samples = []
        min_hops = task_config.get('min_hops', 2)
        max_hops = task_config.get('max_hops', 3)
        
        # Filter images with at least 2 objects
        valid_data = [s for s in source_data 
                     if len(s.get('objects', [])) >= 2]
        
        if not valid_data:
            logger.warning("No valid images with multiple objects for ABR")
            return []
        
        random.shuffle(valid_data)
        
        attempts = 0
        max_attempts = num_samples * 20
        
        while len(samples) < num_samples and attempts < max_attempts:
            attempts += 1
            
            sample = random.choice(valid_data)
            objects = sample.get('objects', [])
            
            if len(objects) < 2:
                continue
            
            # Select two objects to create a reasoning chain
            obj1, obj2 = random.sample(objects, 2)
            
            # Calculate spatial relationship
            x1, y1, w1, h1 = obj1['bbox']
            x2, y2, w2, h2 = obj2['bbox']
            
            center1_x = x1 + w1/2
            center1_y = y1 + h1/2
            center2_x = x2 + w2/2
            center2_y = y2 + h2/2
            
            # Determine spatial relation
            dx = center2_x - center1_x
            dy = center2_y - center1_y
            
            if abs(dx) > abs(dy):
                relation = "to the right of" if dx > 0 else "to the left of"
            else:
                relation = "below" if dy > 0 else "above"
            
            # Generate reasoning chain question
            question_types = [
                'spatial_chain',
                'attribute_chain',
                'category_chain'
            ]
            q_type = random.choice(question_types)
            
            if q_type == 'spatial_chain':
                question = f"There is a {obj1['category']} in the image. What object is located {relation} it?"
                answer = obj2['category']
                reasoning_chain = [
                    f"Step 1: Locate the {obj1['category']} at position ({center1_x:.0f}, {center1_y:.0f})",
                    f"Step 2: Look {relation} this object",
                    f"Step 3: Identify the object as a {obj2['category']}"
                ]
            elif q_type == 'attribute_chain':
                # Compare sizes
                area1 = obj1.get('area', w1 * h1)
                area2 = obj2.get('area', w2 * h2)
                size_rel = "larger" if area2 > area1 else "smaller"
                
                question = f"Find the {obj1['category']} in the image. What object nearby is {size_rel} than it?"
                answer = obj2['category']
                reasoning_chain = [
                    f"Step 1: Find the {obj1['category']} with area {area1:.0f}",
                    f"Step 2: Compare with nearby objects",
                    f"Step 3: The {obj2['category']} has area {area2:.0f}, which is {size_rel}"
                ]
            else:  # category_chain
                question = f"Starting from the {obj1['category']}, what type of object can you find {relation} it?"
                answer = obj2['category']
                reasoning_chain = [
                    f"Step 1: Locate the {obj1['category']}",
                    f"Step 2: Move {relation} in the image",
                    f"Step 3: Identify the {obj2['category']}"
                ]
            
            # Build evidence
            evidence = {
                'start_object': {
                    'category': obj1['category'],
                    'bbox': obj1['bbox'],
                    'center': (center1_x, center1_y)
                },
                'end_object': {
                    'category': obj2['category'],
                    'bbox': obj2['bbox'],
                    'center': (center2_x, center2_y)
                },
                'spatial_relation': relation,
                'reasoning_chain': reasoning_chain,
                'num_hops': min_hops
            }
            
            task = {
                'task_id': f"abr_mscoco_{len(samples)}",
                'task_type': 'attribute_bridge_reasoning',
                'images': [sample['image_path']],
                'question': question,
                'answer': answer,
                'choices': [],  # Could generate distractors here
                'reasoning_chain': reasoning_chain,
                'reasoning_evidence': evidence,
                'metadata': {
                    'source_dataset': 'mscoco14',
                    'min_hops': min_hops,
                    'max_hops': max_hops,
                    'question_type': q_type,
                    'start_category': obj1['category'],
                    'end_category': obj2['category']
                }
            }
            
            samples.append(task)
        
        logger.info(f"Generated {len(samples)} ABR tasks from MSCOCO14 ({attempts} attempts)")
        return samples


class RelationComparisonGenerator:
    """Generate Relation Comparison tasks from MSCOCO14.
    
    RC任务对比多张图片中物体的关系：
    1. 物体数量关系
    2. 物体空间布局关系
    3. 物体共现关系
    """
    
    @staticmethod
    def generate_from_mscoco(source_data: List[Dict],
                            num_samples: int,
                            task_config: Dict[str, Any],
                            templates: Dict[str, Any]) -> List[Dict]:
        """
        Generate RC tasks from MSCOCO using relationship comparison.
        
        Strategy:
        1. Select multiple images with same object categories
        2. Compare the relationships between objects
        3. Generate comparative questions
        """
        samples = []
        n_images = task_config.get('n_images', 3)
        comparison_types = task_config.get('comparison_types', 
            ['co_occurrence', 'spatial_density', 'relationship_complexity'])
        
        # Group by categories for meaningful comparison
        category_to_samples = {}
        for sample in source_data:
            categories = set(obj['category'] for obj in sample.get('objects', []))
            for cat in categories:
                if cat not in category_to_samples:
                    category_to_samples[cat] = []
                category_to_samples[cat].append(sample)
        
        # Filter categories with enough samples
        valid_categories = {
            cat: samps for cat, samps in category_to_samples.items()
            if len(samps) >= n_images
        }
        
        if not valid_categories:
            logger.warning("No valid categories for relation comparison")
            return []
        
        attempts = 0
        max_attempts = num_samples * 20
        
        while len(samples) < num_samples and attempts < max_attempts:
            attempts += 1
            
            # Select category and comparison type
            category = random.choice(list(valid_categories.keys()))
            comparison_type = random.choice(comparison_types)
            
            # Select images
            candidate_samples = valid_categories[category]
            if len(candidate_samples) < n_images:
                continue
                
            selected_samples = random.sample(candidate_samples, n_images)
            
            task = None
            
            if comparison_type == 'co_occurrence':
                task = RelationComparisonGenerator._generate_co_occurrence(
                    selected_samples, category, templates
                )
            elif comparison_type == 'spatial_density':
                task = RelationComparisonGenerator._generate_spatial_density(
                    selected_samples, category, templates
                )
            elif comparison_type == 'relationship_complexity':
                task = RelationComparisonGenerator._generate_relationship_complexity(
                    selected_samples, category, templates
                )
            
            if task:
                task['task_id'] = f"rc_mscoco_{len(samples)}"
                task['task_type'] = 'relation_comparison'
                task['metadata'] = {
                    'source_dataset': 'mscoco14',
                    'n_images': n_images,
                    'comparison_type': comparison_type,
                    'target_category': category
                }
                samples.append(task)
        
        logger.info(f"Generated {len(samples)} RC tasks from MSCOCO14 ({attempts} attempts)")
        return samples
    
    @staticmethod
    def generate_from_vcr(source_data: List[Dict],
                         num_samples: int,
                         task_config: Dict[str, Any],
                         templates: Dict[str, Any]) -> List[Dict]:
        """
        Generate RC tasks from VCR using relationship comparison.
        
        VCR专用策略:
        1. 对比多个场景中的人物互动复杂度
        2. 对比场景的拥挤程度（人物数量）
        3. 对比不同场景中的对象类型多样性
        """
        samples = []
        n_images = task_config.get('n_images', 3)
        comparison_types = task_config.get('comparison_types', 
            ['interaction_complexity', 'scene_crowdedness', 'object_diversity'])
        
        # 过滤有效数据（需要有objects字段）
        valid_data = [s for s in source_data 
                     if s.get('objects') and len(s.get('objects', [])) > 0]
        
        if len(valid_data) < n_images:
            logger.warning(f"Not enough VCR samples for RC task: {len(valid_data)} < {n_images}")
            return []
        
        attempts = 0
        max_attempts = num_samples * 20
        
        while len(samples) < num_samples and attempts < max_attempts:
            attempts += 1
            
            # 随机选择对比类型和样本
            comparison_type = random.choice(comparison_types)
            selected_samples = random.sample(valid_data, n_images)
            
            task = None
            
            if comparison_type == 'interaction_complexity':
                task = RelationComparisonGenerator._generate_vcr_interaction_complexity(
                    selected_samples, templates
                )
            elif comparison_type == 'scene_crowdedness':
                task = RelationComparisonGenerator._generate_vcr_scene_crowdedness(
                    selected_samples, templates
                )
            elif comparison_type == 'object_diversity':
                task = RelationComparisonGenerator._generate_vcr_object_diversity(
                    selected_samples, templates
                )
            
            if task:
                task['task_id'] = f"rc_vcr_{len(samples)}"
                task['task_type'] = 'relation_comparison'
                task['metadata'] = {
                    'source_dataset': 'vcr',
                    'n_images': n_images,
                    'comparison_type': comparison_type
                }
                samples.append(task)
        
        logger.info(f"Generated {len(samples)} RC tasks from VCR ({attempts} attempts)")
        return samples
    
    @staticmethod
    def _generate_vcr_interaction_complexity(samples: List[Dict],
                                              templates: Dict[str, Any]) -> Optional[Dict]:
        """
        Generate task comparing interaction complexity between scenes.
        
        Complexity = number of person * (number of person - 1) / 2 (possible pairwise interactions)
        """
        complexity_info = []
        
        for sample in samples:
            objects = sample.get('objects', [])
            person_count = sum(1 for obj in objects if obj == 'person')
            
            # 计算潜在的人物互动数量（两两组合）
            if person_count >= 2:
                interaction_potential = person_count * (person_count - 1) // 2
            else:
                interaction_potential = 0
            
            complexity_info.append({
                'sample': sample,
                'person_count': person_count,
                'interaction_potential': interaction_potential,
                'all_objects': objects
            })
        
        # 检查是否有差异
        potentials = [info['interaction_potential'] for info in complexity_info]
        if len(set(potentials)) < 2:
            return None  # 没有足够差异
        
        # 找出最复杂的场景
        max_idx = max(range(len(complexity_info)), 
                     key=lambda i: complexity_info[i]['interaction_potential'])
        
        question = "Which image shows a scene with the most potential for character interactions?"
        answer = f"Image {max_idx} with {complexity_info[max_idx]['person_count']} characters ({complexity_info[max_idx]['interaction_potential']} potential interactions)"
        
        evidence = []
        for i, info in enumerate(complexity_info):
            evidence.append({
                'image_idx': i,
                'person_count': info['person_count'],
                'interaction_potential': info['interaction_potential'],
                'is_answer': i == max_idx
            })
        
        return {
            'images': [info['sample']['image_path'] for info in complexity_info],
            'question': question,
            'answer': answer,
            'answer_image_idx': max_idx,
            'comparison_metric': 'interaction_complexity',
            'comparison_values': potentials,
            'reasoning_evidence': evidence,
            'reasoning_depth': 2
        }
    
    @staticmethod
    def _generate_vcr_scene_crowdedness(samples: List[Dict],
                                         templates: Dict[str, Any]) -> Optional[Dict]:
        """
        Generate task comparing scene crowdedness (total number of objects).
        """
        crowdedness_info = []
        
        for sample in samples:
            objects = sample.get('objects', [])
            total_objects = len(objects)
            
            crowdedness_info.append({
                'sample': sample,
                'total_objects': total_objects,
                'object_types': list(set(objects))
            })
        
        # 检查是否有差异
        totals = [info['total_objects'] for info in crowdedness_info]
        if len(set(totals)) < 2:
            return None
        
        max_idx = max(range(len(crowdedness_info)), 
                     key=lambda i: crowdedness_info[i]['total_objects'])
        
        question = "Which image shows the most crowded scene with the most objects?"
        answer = f"Image {max_idx} with {crowdedness_info[max_idx]['total_objects']} objects"
        
        evidence = []
        for i, info in enumerate(crowdedness_info):
            evidence.append({
                'image_idx': i,
                'total_objects': info['total_objects'],
                'object_types': info['object_types'],
                'is_answer': i == max_idx
            })
        
        return {
            'images': [info['sample']['image_path'] for info in crowdedness_info],
            'question': question,
            'answer': answer,
            'answer_image_idx': max_idx,
            'comparison_metric': 'scene_crowdedness',
            'comparison_values': totals,
            'reasoning_evidence': evidence,
            'reasoning_depth': 2
        }
    
    @staticmethod
    def _generate_vcr_object_diversity(samples: List[Dict],
                                        templates: Dict[str, Any]) -> Optional[Dict]:
        """
        Generate task comparing object type diversity across scenes.
        """
        diversity_info = []
        
        for sample in samples:
            objects = sample.get('objects', [])
            unique_types = set(objects)
            
            diversity_info.append({
                'sample': sample,
                'unique_count': len(unique_types),
                'unique_types': list(unique_types),
                'total_objects': len(objects)
            })
        
        # 检查是否有差异
        diversities = [info['unique_count'] for info in diversity_info]
        if len(set(diversities)) < 2:
            return None
        
        max_idx = max(range(len(diversity_info)), 
                     key=lambda i: diversity_info[i]['unique_count'])
        
        question = "Which image shows the greatest variety of different object types?"
        answer = f"Image {max_idx} with {diversity_info[max_idx]['unique_count']} different types: {', '.join(diversity_info[max_idx]['unique_types'][:5])}"
        
        evidence = []
        for i, info in enumerate(diversity_info):
            evidence.append({
                'image_idx': i,
                'unique_count': info['unique_count'],
                'unique_types': info['unique_types'],
                'total_objects': info['total_objects'],
                'is_answer': i == max_idx
            })
        
        return {
            'images': [info['sample']['image_path'] for info in diversity_info],
            'question': question,
            'answer': answer,
            'answer_image_idx': max_idx,
            'comparison_metric': 'object_diversity',
            'comparison_values': diversities,
            'reasoning_evidence': evidence,
            'reasoning_depth': 2
        }
    
    @staticmethod
    def _generate_co_occurrence(samples: List[Dict],
                               category: str,
                               templates: Dict[str, Any]) -> Optional[Dict]:
        """Generate task about object co-occurrence relationships."""
        # Find which other objects co-occur with the target category
        co_occurrences = []
        
        for sample in samples:
            other_cats = set(obj['category'] for obj in sample.get('objects', [])
                           if obj['category'] != category)
            co_occurrences.append({
                'sample': sample,
                'other_categories': other_cats,
                'total_other': len(other_cats)
            })
        
        # Find image with most diverse co-occurrences
        max_idx = max(range(len(co_occurrences)), 
                     key=lambda i: co_occurrences[i]['total_other'])
        
        question = f"Which image shows a {category} with the most variety of other objects?"
        answer = f"Image {max_idx} has {co_occurrences[max_idx]['total_other']} different object types"
        
        evidence = []
        for i, co in enumerate(co_occurrences):
            evidence.append({
                'image_idx': i,
                'target_category': category,
                'co_occurring': list(co['other_categories']),
                'count': co['total_other']
            })
        
        return {
            'images': [co['sample']['image_path'] for co in co_occurrences],
            'question': question,
            'answer': answer,
            'answer_image_idx': max_idx,
            'comparison_metric': 'co_occurrence_diversity',
            'comparison_values': [co['total_other'] for co in co_occurrences],
            'reasoning_evidence': evidence,
            'reasoning_depth': 2
        }
    
    @staticmethod
    def _generate_spatial_density(samples: List[Dict],
                                  category: str,
                                  templates: Dict[str, Any]) -> Optional[Dict]:
        """Generate task about spatial density of objects."""
        density_info = []
        
        for sample in samples:
            objs = [obj for obj in sample.get('objects', []) 
                   if obj['category'] == category]
            
            if not objs:
                return None
            
            # Calculate density (objects per image area)
            total_area = sum(obj.get('area', 0) for obj in objs)
            img_area = sample.get('width', 1) * sample.get('height', 1)
            density = total_area / img_area if img_area > 0 else 0
            
            density_info.append({
                'sample': sample,
                'count': len(objs),
                'total_area': total_area,
                'density': density
            })
        
        # Find image with highest density
        max_idx = max(range(len(density_info)), 
                     key=lambda i: density_info[i]['density'])
        
        question = f"In which image does the {category} occupy the largest portion of the scene?"
        answer = f"Image {max_idx}"
        
        evidence = []
        for i, info in enumerate(density_info):
            evidence.append({
                'image_idx': i,
                'category': category,
                'object_count': info['count'],
                'total_area': info['total_area'],
                'density': info['density']
            })
        
        return {
            'images': [info['sample']['image_path'] for info in density_info],
            'question': question,
            'answer': answer,
            'answer_image_idx': max_idx,
            'comparison_metric': 'spatial_density',
            'comparison_values': [info['density'] for info in density_info],
            'reasoning_evidence': evidence,
            'reasoning_depth': 2
        }
    
    @staticmethod
    def _generate_relationship_complexity(samples: List[Dict],
                                          category: str,
                                          templates: Dict[str, Any]) -> Optional[Dict]:
        """Generate task about relationship complexity between objects."""
        complexity_info = []
        
        for sample in samples:
            objects = sample.get('objects', [])
            target_objs = [obj for obj in objects if obj['category'] == category]
            other_objs = [obj for obj in objects if obj['category'] != category]
            
            if not target_objs:
                return None
            
            # Complexity = number of potential relationships
            # (target objects * other objects)
            complexity = len(target_objs) * len(other_objs)
            
            complexity_info.append({
                'sample': sample,
                'target_count': len(target_objs),
                'other_count': len(other_objs),
                'complexity': complexity,
                'all_categories': list(set(obj['category'] for obj in objects))
            })
        
        # Find image with most complex scene
        max_idx = max(range(len(complexity_info)), 
                     key=lambda i: complexity_info[i]['complexity'])
        
        question = f"Which image shows the {category} in the most complex scene with other objects?"
        answer = f"Image {max_idx} with {complexity_info[max_idx]['complexity']} potential relationships"
        
        evidence = []
        for i, info in enumerate(complexity_info):
            evidence.append({
                'image_idx': i,
                'target_category': category,
                'target_count': info['target_count'],
                'other_object_count': info['other_count'],
                'complexity_score': info['complexity'],
                'all_categories': info['all_categories']
            })
        
        return {
            'images': [info['sample']['image_path'] for info in complexity_info],
            'question': question,
            'answer': answer,
            'answer_image_idx': max_idx,
            'comparison_metric': 'relationship_complexity',
            'comparison_values': [info['complexity'] for info in complexity_info],
            'reasoning_evidence': evidence,
            'reasoning_depth': 2
        }


class QADatasetGenerators:
    """
    QA数据集（ScienceQA、DocVQA、RealworldQA）的专用任务生成器。
    
    通过多样本聚合策略，将单图QA转换为多图推理任务。
    """
    
    # ==================== RC (Relation Comparison) ====================
    
    @staticmethod
    def generate_rc_from_scienceqa(source_data: List[Dict],
                                   num_samples: int,
                                   task_config: Dict[str, Any],
                                   templates: Dict[str, Any]) -> List[Dict]:
        """
        ScienceQA RC生成器：课程/解释匹配任务。
        
        策略：
        1. 选择包含lecture或explanation的样本作为目标
        2. 选择干扰样本（同一subject但内容不同）
        3. 生成匹配问题
        """
        samples = []
        n_distractors = task_config.get('n_distractors', 3)
        
        # 过滤出有有效image且有lecture/explanation的样本
        valid_data = []
        for s in source_data:
            if not QADatasetGenerators._has_valid_image(s):
                continue
            # 检查是否有lecture或explanation
            lecture = QADatasetGenerators._safe_get(s, 'lecture', '')
            explanation = QADatasetGenerators._safe_get(s, 'solution', '') # solution通常包含explanation
            if lecture or explanation:
                valid_data.append(s)
        
        if len(valid_data) < n_distractors + 1:
            logger.warning("Not enough ScienceQA samples with lecture/explanation for RC task")
            return []
            
        attempts = 0
        max_attempts = num_samples * 20
        
        while len(samples) < num_samples and attempts < max_attempts:
            attempts += 1
            
            target_sample = random.choice(valid_data)
            subject = QADatasetGenerators._safe_get(target_sample, 'subject', '')
            
            # 选择干扰项：优先选择同一subject的
            same_subject_candidates = [s for s in source_data 
                                     if QADatasetGenerators._safe_get(s, 'subject', '') == subject
                                     and s is not target_sample
                                     and QADatasetGenerators._has_valid_image(s)]
            
            other_candidates = [s for s in source_data 
                               if s is not target_sample 
                               and s not in same_subject_candidates
                               and QADatasetGenerators._has_valid_image(s)]
                               
            distractors = []
            # 尝试获取同一subject的干扰项
            if len(same_subject_candidates) >= n_distractors:
                distractors = random.sample(same_subject_candidates, n_distractors)
            else:
                distractors = same_subject_candidates + \
                    random.sample(other_candidates, min(len(other_candidates), n_distractors - len(same_subject_candidates)))
            
            if len(distractors) < n_distractors:
                continue
                
            # 组合并打乱
            all_samples = distractors + [target_sample]
            random.shuffle(all_samples)
            target_idx = -1
            for i, s in enumerate(all_samples):
                if s is target_sample:
                    target_idx = i
                    break
            
            images = [s['image_path'] for s in all_samples]
            
            # 构建问题
            lecture = QADatasetGenerators._safe_get(target_sample, 'lecture', '')
            explanation = QADatasetGenerators._safe_get(target_sample, 'solution', '')
            
            # 清理文本（去除 "Explanation:" 前缀等）
            if explanation.lower().startswith("explanation:"):
                explanation = explanation[12:].strip()
                
            content = lecture if lecture else explanation
            # 截断过长的文本
            if len(content) > 300:
                content = content[:297] + "..."
                
            question = f"Which image matches the following scientific explanation/lecture?\n\n\"{content}\""
            answer = f"Image {target_idx}"
            
            evidence = {
                'target': {
                    'image_idx': target_idx,
                    'subject': subject,
                    'topic': QADatasetGenerators._safe_get(target_sample, 'topic', ''),
                    'lecture': lecture,
                    'explanation': explanation
                },
                'distractors': [{'image_idx': i, 'subject': QADatasetGenerators._safe_get(s, 'subject', '')} 
                              for i, s in enumerate(all_samples) if i != target_idx]
            }
            
            samples.append({
                'task_id': f"rc_scienceqa_{len(samples)}",
                'task_type': 'relation_comparison',
                'images': images,
                'question': question,
                'answer': answer,
                'answer_image_idx': target_idx,
                'comparison_metric': 'text_image_matching',
                'comparison_values': [],
                'reasoning_evidence': evidence,
                'reasoning_depth': 2,
                'metadata': {
                    'source_dataset': 'scienceqa',
                    'n_images': len(images),
                    'comparison_type': 'lecture_matching',
                    'subject': subject
                }
            })
        
        logger.info(f"Generated {len(samples)} RC tasks from ScienceQA ({attempts} attempts)")
        return samples
    
    @staticmethod
    def generate_rc_from_docvqa(source_data: List[Dict],
                                num_samples: int,
                                task_config: Dict[str, Any],
                                templates: Dict[str, Any]) -> List[Dict]:
        """
        DocVQA RC生成器：对比文档信息密度。
        """
        samples = []
        n_images = task_config.get('n_images', 3)
        
        # DocVQA没有明确分类，随机选择
        valid_data = [s for s in source_data if s.get('image_path')]
        
        if len(valid_data) < n_images:
            logger.warning("Not enough DocVQA samples for RC task")
            return []
        
        attempts = 0
        max_attempts = num_samples * 10
        
        while len(samples) < num_samples and attempts < max_attempts:
            attempts += 1
            
            selected = random.sample(valid_data, n_images)
            
            images = [s['image_path'] for s in selected]
            
            # 对比问题长度/答案复杂度
            question_lengths = [len(str(s.get('question', ''))) for s in selected]
            max_idx = question_lengths.index(max(question_lengths))
            
            question = "Compare these documents. Which one contains the most complex information requiring deeper analysis?"
            answer = f"Image {max_idx} with question length {question_lengths[max_idx]}"
            
            evidence = []
            for i, s in enumerate(selected):
                evidence.append({
                    'image_idx': i,
                    'original_question': s.get('question', ''),
                    'original_answer': str(s.get('answers', [''])[0]) if s.get('answers') else '',
                    'question_length': question_lengths[i],
                    'is_answer': i == max_idx
                })
            
            samples.append({
                'task_id': f"rc_docvqa_{len(samples)}",
                'task_type': 'relation_comparison',
                'images': images,
                'question': question,
                'answer': answer,
                'answer_image_idx': max_idx,
                'comparison_metric': 'information_complexity',
                'comparison_values': question_lengths,
                'reasoning_evidence': evidence,
                'reasoning_depth': 2,
                'metadata': {
                    'source_dataset': 'docvqa',
                    'n_images': n_images,
                    'comparison_type': 'complexity_based'
                }
            })
        
        logger.info(f"Generated {len(samples)} RC tasks from DocVQA ({attempts} attempts)")
        return samples
    
    @staticmethod
    def generate_rc_from_realworldqa(source_data: List[Dict],
                                     num_samples: int,
                                     task_config: Dict[str, Any],
                                     templates: Dict[str, Any]) -> List[Dict]:
        """
        RealworldQA RC生成器：对比现实场景复杂度。
        """
        samples = []
        n_images = task_config.get('n_images', 3)
        
        valid_data = [s for s in source_data if s.get('image_path')]
        
        if len(valid_data) < n_images:
            logger.warning("Not enough RealworldQA samples for RC task")
            return []
        
        attempts = 0
        max_attempts = num_samples * 10
        
        while len(samples) < num_samples and attempts < max_attempts:
            attempts += 1
            
            selected = random.sample(valid_data, n_images)
            
            images = [s['image_path'] for s in selected]
            
            # 对比问题长度（更长的问题可能意味着更复杂的场景）
            question_lengths = [len(str(s.get('question', ''))) for s in selected]
            max_idx = question_lengths.index(max(question_lengths)) if max(question_lengths) > 0 else 0
            
            question = "Compare these real-world scenarios. Which image presents the most complex situation requiring the longest description?"
            answer = f"Image {max_idx} with question length {question_lengths[max_idx]}"
            
            evidence = []
            for i, s in enumerate(selected):
                evidence.append({
                    'image_idx': i,
                    'original_question': s.get('question', ''),
                    'question_length': question_lengths[i],
                    'is_answer': i == max_idx
                })
            
            samples.append({
                'task_id': f"rc_realworldqa_{len(samples)}",
                'task_type': 'relation_comparison',
                'images': images,
                'question': question,
                'answer': answer,
                'answer_image_idx': max_idx,
                'comparison_metric': 'question_complexity',
                'comparison_values': question_lengths,
                'reasoning_evidence': evidence,
                'reasoning_depth': 2,
                'metadata': {
                    'source_dataset': 'realworldqa',
                    'n_images': n_images,
                    'comparison_type': 'complexity_based'
                }
            })
        
        logger.info(f"Generated {len(samples)} RC tasks from RealworldQA ({attempts} attempts)")
        return samples
    
    @staticmethod
    def _safe_get(sample, key, default=''):
        """安全获取字段值，处理numpy类型"""
        import numpy as np
        val = sample.get(key, default)
        if isinstance(val, np.ndarray):
            if val.size == 0:
                return default
            return str(val.flat[0]) if val.size == 1 else default
        return val if val is not None else default

    @staticmethod
    def _has_valid_image(sample):
        """检查是否有有效图片"""
        import numpy as np
        img = sample.get('image_path')
        if img is None:
            return False
        if isinstance(img, np.ndarray):
            return img.size > 0
        return bool(img)

    @staticmethod
    def _get_answer_string(sample):
        """安全地从sample中获取答案字符串，处理choices和answer可能为numpy类型的情况"""
        import numpy as np
        choices = sample.get('choices', [])
        answer_idx = sample.get('answer', 0)
        
        # Check if choices is valid/non-empty
        has_choices = False
        if isinstance(choices, np.ndarray):
            has_choices = choices.size > 0
        elif choices: # list or other sequence
            has_choices = True
            
        if has_choices:
            # Handle answer_idx
            try:
                idx = int(answer_idx)
                if 0 <= idx < len(choices):
                    val = choices[idx]
                    return str(val)
            except (ValueError, TypeError, IndexError):
                pass
                
        # Fallback to answer_idx string
        if isinstance(answer_idx, (np.ndarray, np.generic)):
            if isinstance(answer_idx, np.ndarray) and answer_idx.size == 1:
                return str(answer_idx.flat[0])
            return str(answer_idx)
        return str(answer_idx)

    @staticmethod
    def _parse_choice_from_question(question: str, choice_letter: str) -> str:
        """
        从问题文本中解析指定选项字母对应的内容。
        
        支持的格式：
        - A. 选项内容
        - A) 选项内容
        - (A) 选项内容
        
        Args:
            question: 包含选项的问题文本
            choice_letter: 选项字母（A/B/C/D）
            
        Returns:
            解析出的选项内容，如果解析失败则返回空字符串
        """
        import re
        
        # 尝试多种选项格式
        patterns = [
            rf'{choice_letter}\.\s*(.+?)(?=\n[A-D]\.|\n[A-D]\)|\nPlease|$)',  # A. 内容
            rf'{choice_letter}\)\s*(.+?)(?=\n[A-D]\.|\n[A-D]\)|\nPlease|$)',  # A) 内容
            rf'\({choice_letter}\)\s*(.+?)(?=\n[A-D]\.|\n[A-D]\)|\nPlease|$)',  # (A) 内容
        ]
        
        for pattern in patterns:
            match = re.search(pattern, question, re.IGNORECASE | re.DOTALL)
            if match:
                content = match.group(1).strip()
                # 清理可能的尾部换行和多余空白
                content = content.replace('\n', ' ').strip()
                return content
        
        return ""

    @staticmethod
    def _clean_question_text(question: str) -> str:
        """
        Clean question text by removing options and specific instructions.
        """
        import re
        
        # Remove specific instruction
        instruction = "Please answer directly with only the letter of the correct option and nothing else."
        question = question.replace(instruction, "")
        
        # Split into lines
        lines = question.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line_stripped = line.strip()
            if not line_stripped:
                continue
                
            # Skip options (A. xxx, B. xxx, (A) xxx, A) xxx)
            # Match A. or A) or (A) at start of line
            if re.match(r'^[\(]?[A-E][\.\)]\s+', line_stripped):
                continue
                
            cleaned_lines.append(line_stripped)
            
        return " ".join(cleaned_lines).strip()



    # ==================== VNF (Visual Noise Filtering) ====================
    
    @staticmethod
    def generate_vnf_from_qa(source_data: List[Dict],
                             source_dataset: str,
                             num_samples: int,
                             task_config: Dict[str, Any],
                             templates: Dict[str, Any]) -> List[Dict]:
        """
        通用QA数据集VNF生成器：添加干扰图片。
        
        策略：
        1. 选择目标样本
        2. 添加n个干扰图片（来自其他样本）
        3. 生成"哪张图片包含正确答案"问题
        """
        samples = []
        n_distractors = task_config.get('n_distractors', 3)
        
        valid_data = [s for s in source_data if QADatasetGenerators._has_valid_image(s)]
        
        if len(valid_data) < n_distractors + 1:
            logger.warning(f"Not enough {source_dataset} samples for VNF task")
            return []
        
        for idx in range(min(num_samples, len(valid_data) - n_distractors)):
            target_sample = valid_data[idx]
            
            # 选择干扰样本，使用identity check避免numpy比较错误
            distractor_candidates = [s for s in valid_data if s is not target_sample]
            distractors = random.sample(distractor_candidates, min(n_distractors, len(distractor_candidates)))
            
            # 组合图片并打乱
            all_samples = distractors + [target_sample]
            random.shuffle(all_samples)
            # 使用identity check查找索引
            target_idx = -1
            for i, s in enumerate(all_samples):
                if s is target_sample:
                    target_idx = i
                    break
            
            images = [s['image_path'] for s in all_samples]
            
            # 获取原始问答
            original_question = QADatasetGenerators._safe_get(target_sample, 'question', '')
            original_answer = QADatasetGenerators._get_answer_string(target_sample)
            
            # 如果答案是选项字母（A/B/C/D），尝试从问题中提取对应的选项内容
            display_answer = original_answer
            if original_answer and len(original_answer) == 1 and original_answer.upper() in 'ABCD':
                # 尝试从问题中解析选项
                parsed_choice = QADatasetGenerators._parse_choice_from_question(
                    original_question, original_answer.upper()
                )
                if parsed_choice:
                    display_answer = parsed_choice
            
            # Clean question for display (remove options and instructions)
            cleaned_question = QADatasetGenerators._clean_question_text(original_question)
            
            # 生成VNF问题
            question = f"Based on the question: '{cleaned_question}', which image contains the answer '{display_answer}'?"
            answer = f"Image {target_idx}"
            
            evidence = {
                'target': {
                    'image_idx': target_idx,
                    'original_question': original_question,
                    'original_answer': original_answer
                }
            }
            
            samples.append({
                'task_id': f"vnf_{source_dataset}_{len(samples)}",
                'task_type': 'visual_noise_filtering',
                'images': images,
                'question': question,
                'answer': answer,
                'target_image_idx': target_idx,
                'reasoning_depth': 1,
                'reasoning_evidence': evidence,
                'metadata': {
                    'source_dataset': source_dataset,
                    'n_total_images': len(images),
                    'distractor_strategy': 'random_from_dataset',
                    'uses_original_qa': True
                }
            })
        
        logger.info(f"Generated {len(samples)} VNF tasks from {source_dataset}")
        return samples
    
    # ==================== AC (Attribute Comparison) ====================
    
    @staticmethod
    def generate_ac_from_qa(source_data: List[Dict],
                            source_dataset: str,
                            num_samples: int,
                            task_config: Dict[str, Any],
                            templates: Dict[str, Any]) -> List[Dict]:
        """
        通用QA数据集AC生成器：对比不同样本的属性。
        """
        samples = []
        n_images = task_config.get('n_images', 3)
        
        valid_data = [s for s in source_data if QADatasetGenerators._has_valid_image(s)]
        
        if len(valid_data) < n_images:
            logger.warning(f"Not enough {source_dataset} samples for AC task")
            return []
        
        for idx in range(min(num_samples, len(valid_data) // n_images)):
            selected = random.sample(valid_data, n_images)
            
            images = [s['image_path'] for s in selected]
            
            # 对比各样本的特征
            # 使用答案长度作为"复杂度"指标
            complexities = []
            for s in selected:
                answer = QADatasetGenerators._get_answer_string(s)
                complexities.append(len(answer))
            
            max_idx = complexities.index(max(complexities)) if complexities else 0
            
            question = f"Compare the content across these {n_images} images. Which one has the most detailed answer?"
            answer = f"Image {max_idx} with answer length {complexities[max_idx]}"
            
            evidence = []
            for i, s in enumerate(selected):
                evidence.append({
                    'image_idx': i,
                    'original_question': s.get('question', ''),
                    'answer_complexity': complexities[i],
                    'is_answer': i == max_idx
                })
            
            samples.append({
                'task_id': f"ac_{source_dataset}_{idx}",
                'task_type': 'attribute_comparison',
                'images': images,
                'question': question,
                'answer': answer,
                'answer_image_idx': max_idx,
                'comparison_metric': 'answer_detail',
                'comparison_values': complexities,
                'reasoning_evidence': evidence,
                'reasoning_depth': 2,
                'metadata': {
                    'source_dataset': source_dataset,
                    'n_images': n_images,
                    'comparison_type': 'answer_complexity'
                }
            })
        
        logger.info(f"Generated {len(samples)} AC tasks from {source_dataset}")
        return samples

    # ==================== LNF (Logical Noise Filtering) ====================

    @staticmethod
    def generate_lnf_from_scienceqa(source_data: List[Dict],
                                    num_samples: int,
                                    task_config: Dict[str, Any],
                                    templates: Dict[str, Any]) -> List[Dict]:
        """
        ScienceQA LNF生成器：基于图片和Lecture/Explanation的匹配逻辑。
        
        策略：
        1. 给定图片
        2. 选项：正确的Lecture vs 干扰Lecture（来自其他topic）
        3. 验证模型是否能通过视觉现象匹配科学原理
        """
        samples = []
        n_options = task_config.get('n_options', 4)
        
        # 预处理：收集有效的lecture/explanation，按topic索引
        valid_items = []
        topic_map = {}
        
        for s in source_data:
            if not QADatasetGenerators._has_valid_image(s):
                continue
            
            # 获取内容 (优先Lecture，其次Explanation)
            lecture = QADatasetGenerators._safe_get(s, 'lecture', '')
            explanation = QADatasetGenerators._safe_get(s, 'solution', '')
            
            content = lecture if lecture else explanation
            
            # 清理
            if explanation.lower().startswith("explanation:"):
                explanation = explanation[12:].strip()
                if not lecture: content = explanation
            
            if not content or len(content) < 10: # 忽略太短的内容
                continue
                
            item = {
                'sample': s,
                'content': content,
                'topic': QADatasetGenerators._safe_get(s, 'topic', 'general')
            }
            valid_items.append(item)
            
            topic = item['topic']
            if topic not in topic_map:
                topic_map[topic] = []
            topic_map[topic].append(item)
            
        if len(valid_items) < n_options:
            logger.warning("Not enough valid ScienceQA samples for LNF task")
            return []
            
        attempts = 0
        max_attempts = num_samples * 10
        
        while len(samples) < num_samples and attempts < max_attempts:
            attempts += 1
            
            target_item = random.choice(valid_items)
            target_topic = target_item['topic']
            
            # 选择干扰项：优先选择不同topic的lecture
            distractor_topics = [t for t in topic_map.keys() if t != target_topic]
            distractors = []
            
            if len(distractor_topics) >= n_options - 1:
                # 每个干扰项来自不同topic，增加区分度
                selected_topics = random.sample(distractor_topics, n_options - 1)
                for t in selected_topics:
                    distractors.append(random.choice(topic_map[t]))
            else:
                # 随机选择
                other_items = [i for i in valid_items if i['topic'] != target_topic]
                if len(other_items) < n_options - 1:
                    continue
                distractors = random.sample(other_items, n_options - 1)
                
            # 构建选项
            options_data = [target_item] + distractors
            random.shuffle(options_data)
            
            target_idx = -1
            options_text = []
            
            for i, opt in enumerate(options_data):
                if opt is target_item:
                    target_idx = i
                # 处理文本长度
                text = opt['content']
                if len(text) > 300:
                    text = text[:297] + "..."
                # 添加序号 (1-based)
                options_text.append(f"{i+1}. {text}")
            
            question = random.choice(templates.get('question_templates', [
                "Which of the following scientific principles explains the phenomenon or concept shown in this image?"
            ]))
            
            # 格式化选项字符串 (如果模型需要显式的选项列表)
            # 在这里我们把选项放入 choices 字段，让 generator_v2 统一处理或在 question 中 formatted
            # M3Bench通常支持 choices 字段
            
            answer = f"Option {target_idx + 1}"
            
            evidence = {
                'target_topic': target_topic,
                'target_content_full': target_item['content'],
                'distractor_topics': [d['topic'] for d in distractors]
            }
            
            samples.append({
                'task_id': f"lnf_scienceqa_{len(samples)}",
                'task_type': 'logical_noise_filtering',
                'images': [target_item['sample']['image_path']],
                'question': question,
                'answer': answer, # Option X
                'choices': options_text,
                'answer_image_idx': 0, # 虽然是单图任务，但为了兼容性
                'comparison_metric': 'principle_matching',
                'comparison_values': [],
                'reasoning_evidence': evidence,
                'reasoning_depth': 2,
                'metadata': {
                    'source_dataset': 'scienceqa',
                    'n_options': n_options,
                    'topic': target_topic
                }
            })
            
        logger.info(f"Generated {len(samples)} LNF tasks from ScienceQA")
        return samples

    @staticmethod
    def generate_ac_from_scienceqa(source_data: List[Dict],
                                   num_samples: int,
                                   task_config: Dict[str, Any],
                                   templates: Dict[str, Any]) -> List[Dict]:
        """
        ScienceQA AC生成器：基于Subject和Topic的分类识别。
        
        策略：
        1. 按照Subject (Biology, Physics) 分组
        2. 选择一个Topic作为目标
        3. 问题：哪个图片属于该Topic？
        4. 干扰项：同Subject但不同Topic的图片
        """
        samples = []
        n_images = task_config.get('n_images', 3)
        
        # 按照subject->topic分组
        hierarchy = {}
        for sample in source_data:
            if not QADatasetGenerators._has_valid_image(sample):
                continue
            subject = QADatasetGenerators._safe_get(sample, 'subject', 'general')
            topic = QADatasetGenerators._safe_get(sample, 'topic', 'unknown')
            
            if subject not in hierarchy:
                hierarchy[subject] = {}
            if topic not in hierarchy[subject]:
                hierarchy[subject][topic] = []
            hierarchy[subject][topic].append(sample)
            
        attempts = 0
        max_attempts = num_samples * 30
        
        valid_subjects = [s for s in hierarchy.keys() if len(hierarchy[s]) >= 2] # 至少有两个topic的subject
        
        while len(samples) < num_samples and attempts < max_attempts:
            attempts += 1
            if not valid_subjects:
                break
                
            subject = random.choice(valid_subjects)
            topics = list(hierarchy[subject].keys())
            
            # 选择目标topic和其他topic
            if len(topics) < 2:
                continue
                
            target_topic = random.choice(topics)
            other_topics = [t for t in topics if t != target_topic]
            
            # 检查是否有足够的样本
            target_candidates = hierarchy[subject][target_topic]
            if not target_candidates:
                continue
                
            target_sample = random.choice(target_candidates)
            
            # 选择干扰样本（来自同一subject的其他topic）
            distractors = []
            distrator_candidates = []
            for t in other_topics:
                distrator_candidates.extend(hierarchy[subject][t])
            
            if len(distrator_candidates) < n_images - 1:
                continue
                
            distractors = random.sample(distrator_candidates, n_images - 1)
            
            # 组合并打乱
            all_samples = distractors + [target_sample]
            random.shuffle(all_samples)
            target_idx = -1
            for i, s in enumerate(all_samples):
                if s is target_sample:
                    target_idx = i
                    break
            
            images = [s['image_path'] for s in all_samples]
            
            question = f"All these images belong to the subject of {subject}. Which image specifically relates to the topic of '{target_topic}'?"
            answer = f"Image {target_idx}"
            
            evidence = {
                'subject': subject,
                'options': [{'image_idx': i, 'topic': QADatasetGenerators._safe_get(s, 'topic', '')} for i, s in enumerate(all_samples)],
                'target_topic': target_topic
            }
            
            samples.append({
                'task_id': f"ac_scienceqa_{len(samples)}",
                'task_type': 'attribute_comparison',
                'images': images,
                'question': question,
                'answer': answer,
                'answer_image_idx': target_idx,
                'comparison_metric': 'topic_classification',
                'comparison_values': [],
                'reasoning_evidence': evidence,
                'reasoning_depth': 2,
                'metadata': {
                    'source_dataset': 'scienceqa',
                    'n_images': n_images,
                    'comparison_type': 'topic_classification',
                    'subject': subject,
                    'target_topic': target_topic
                }
            })
            
        logger.info(f"Generated {len(samples)} AC tasks from ScienceQA ({attempts} attempts)")
        return samples