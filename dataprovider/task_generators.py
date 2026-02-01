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

from .task_id_generator import TaskIDGenerator

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
        comparison_types = task_config.get('comparison_types', [])
        target_categories = task_config.get('target_categories', 'all')

        # 使用全局唯一ID生成器
        id_gen = TaskIDGenerator(task_type='attribute_comparison', dataset='mscoco14')

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
                task['task_id'] = id_gen.next()
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
        question_templates = templates.get('question_templates', {}).get('count_comparison', [])
        if not question_templates:
            question = f"Which image has the most {category}s?"
        else:
            template = random.choice(question_templates)
            question = template.format(category=category)

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

        # 使用全局唯一ID生成器
        id_gen = TaskIDGenerator(task_type='attribute_comparison', dataset='vcr')

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
                'task_id': id_gen.next(),
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

    @staticmethod
    def generate_from_visual_genome(source_data: List[Dict],
                                    num_samples: int,
                                    task_config: Dict[str, Any],
                                    templates: Dict[str, Any]) -> List[Dict]:
        """
        Generate Attribute Comparison tasks from Visual Genome.

        Visual Genome has rich attribute annotations, we can compare:
        1. Object attributes (color, size, material)
        2. Object counts
        3. Relationship complexity
        """
        samples = []
        n_images = task_config.get('n_images', 3)
        comparison_types = task_config.get('comparison_types', ['attribute', 'count'])

        # 使用全局唯一ID生成器
        id_gen = TaskIDGenerator(task_type='attribute_comparison', dataset='visual_genome')

        attempts = 0
        max_attempts = num_samples * 20

        while len(samples) < num_samples and attempts < max_attempts:
            attempts += 1

            # Select random samples
            if len(source_data) < n_images:
                break

            selected_samples = random.sample(source_data, n_images)

            # Extract objects from Visual Genome format
            all_objects = []
            for sample in selected_samples:
                objects = sample.get('objects', [])
                # Visual Genome objects have 'name' or 'names' field
                obj_names = []
                for obj in objects:
                    name = obj.get('name') or (obj.get('names', ['unknown'])[0] if obj.get('names') else 'unknown')
                    obj_names.append(name)
                all_objects.append(obj_names)

            # Choose comparison type
            comparison_type = random.choice(comparison_types) if comparison_types else 'count'

            if comparison_type == 'count':
                # Compare object counts
                counts = [len(objs) for objs in all_objects]

                if len(set(counts)) < 2:
                    continue

                max_count = max(counts)
                answer_idx = counts.index(max_count)

                question = "Which image contains the most objects?"
                answer = f"Image {answer_idx} with {max_count} objects"

                evidence = []
                for i, (sample, objs) in enumerate(zip(selected_samples, all_objects)):
                    evidence.append({
                        'image_idx': i,
                        'object_count': len(objs),
                        'objects': objs[:10]  # Limit for readability
                    })

            elif comparison_type == 'attribute':
                # Compare based on common attributes
                # Look for color attributes in objects
                color_counts = []
                for sample in selected_samples:
                    objects = sample.get('objects', [])
                    colors = []
                    for obj in objects:
                        attrs = obj.get('attributes', [])
                        for attr in attrs:
                            if isinstance(attr, str) and attr.lower() in ['red', 'blue', 'green', 'yellow', 'white', 'black', 'brown']:
                                colors.append(attr.lower())
                    color_counts.append(len(set(colors)))

                if len(set(color_counts)) < 2:
                    continue

                max_colors = max(color_counts)
                answer_idx = color_counts.index(max_colors)

                question = "Which image shows the most variety of colors?"
                answer = f"Image {answer_idx} with {max_colors} distinct colors"

                evidence = []
                for i, (sample, count) in enumerate(zip(selected_samples, color_counts)):
                    evidence.append({
                        'image_idx': i,
                        'color_count': count
                    })
            else:
                continue

            samples.append({
                'task_id': id_gen.next(),
                'task_type': 'attribute_comparison',
                'images': [s.get('image_path', s.get('url', '')) for s in selected_samples],
                'question': question,
                'answer': answer,
                'comparison_type': comparison_type,
                'reasoning_evidence': evidence,
                'reasoning_depth': 2,
                'metadata': {
                    'source_dataset': 'visual_genome',
                    'source_ids': [s.get('image_id', '') for s in selected_samples]
                }
            })

        logger.info(f"Generated {len(samples)} Visual Genome attribute comparison tasks")
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

        # 使用全局唯一ID生成器
        id_gen = TaskIDGenerator(task_type='visual_noise_filtering', dataset='mscoco14')

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
            question_templates = templates.get('question_templates', {}).get('object_based', [])
            if question_templates:
                template = random.choice(question_templates)
                question = template.format(target=target_category)
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
                'task_id': id_gen.next(),
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
    def generate_from_vcr(source_data: List[Dict],
                         num_samples: int,
                         task_config: Dict[str, Any],
                         templates: Dict[str, Any]) -> List[Dict]:
        """Generate VNF tasks from VCR using natural Q&A."""
        samples = []
        n_distractors = task_config.get('n_distractors', 3)
        use_original_qa = task_config.get('use_original_qa', True)

        # 使用全局唯一ID生成器
        id_gen = TaskIDGenerator(task_type='visual_noise_filtering', dataset='vcr')

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

            # Use VCR's natural question
            from dataprovider.generator import DataGenerator
            question_text = DataGenerator._vcr_tokens_to_text(
                target_sample['question'],
                target_sample['objects']
            )

            # Select distractors (different scenes)
            distractors = valid_data[idx+1:idx+1+n_distractors]

            all_images = distractors + [target_sample]
            random.shuffle(all_images)
            target_idx = all_images.index(target_sample)

            # Get answer
            answer_tokens = target_sample['answer_choices'][target_sample['answer_label']]
            answer_text = DataGenerator._vcr_tokens_to_text(
                answer_tokens,
                target_sample['objects']
            )

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
                'task_id': id_gen.next(),
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
