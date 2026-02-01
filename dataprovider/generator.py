"""
DataGenerator - 数据生成器
==========================

包含任务生成所需的基础方法，主要用于兼容旧代码。
实际任务生成请使用 generator_v2.py 中的 DataGeneratorV2。

主要方法:
- _vcr_tokens_to_text: VCR token格式转换为文本
- _abr_from_mscoco: 从MSCOCO生成ABR任务
- _abr_from_vcr: 从VCR生成ABR任务
- _abr_from_visual_genome: 从Visual Genome生成ABR任务
- _rc_from_mscoco: 从MSCOCO生成RC任务
- _rc_from_visual_genome: 从Visual Genome生成RC任务
"""

import random
from typing import Dict, List, Any, Optional, Tuple


class DataGenerator:
    """
    数据生成器基类

    提供任务生成的基础方法，包括VCR token解析、ABR/RC任务生成等。
    """

    def __init__(self, data_loader=None):
        """
        初始化数据生成器

        Args:
            data_loader: 数据加载器实例（可选）
        """
        self.data_loader = data_loader

    @staticmethod
    def _vcr_tokens_to_text(tokens: List[Any], objects: List[str]) -> str:
        """
        将VCR的混合token(字符串+对象索引)转换为文本

        VCR数据集使用特殊的token格式:
        - 字符串token: 直接的文本 (如 "What", "is", "?")
        - 对象索引: 列表形式的索引 (如 [0], [0,1])

        Args:
            tokens: token列表，包含字符串和/或对象索引列表
                   例如: ["What", "are", [0, 1], "doing", "near", [2], "?"]
            objects: 对象名称列表
                   例如: ["person", "dog", "tree"]

        Returns:
            转换后的文本字符串
            例如: "What are person and dog doing near tree ?"

        Examples:
            >>> tokens = ["What", "is", "happening", "?"]
            >>> objects = ["person", "car"]
            >>> DataGenerator._vcr_tokens_to_text(tokens, objects)
            'What is happening ?'

            >>> tokens = ["What", "are", [0, 1], "doing", "?"]
            >>> objects = ["person", "dog"]
            >>> DataGenerator._vcr_tokens_to_text(tokens, objects)
            'What are person and dog doing ?'
        """
        result = []

        for token in tokens:
            if isinstance(token, str):
                # 直接文本token
                result.append(token)
            elif isinstance(token, list):
                # 对象索引引用
                if len(token) == 0:
                    continue
                elif len(token) == 1:
                    # 单个对象引用
                    idx = token[0]
                    if 0 <= idx < len(objects):
                        result.append(objects[idx])
                    else:
                        result.append(f"[object_{idx}]")
                else:
                    # 多个对象引用，用 "and" 连接
                    obj_names = []
                    for idx in token:
                        if 0 <= idx < len(objects):
                            obj_names.append(objects[idx])
                        else:
                            obj_names.append(f"[object_{idx}]")
                    result.append(" and ".join(obj_names))
            elif isinstance(token, int):
                # 直接的对象索引（不是列表形式）
                if 0 <= token < len(objects):
                    result.append(objects[token])
                else:
                    result.append(f"[object_{token}]")

        return " ".join(result)

    @staticmethod
    def _abr_from_mscoco(
        source_data: List[Dict[str, Any]],
        num_samples: int,
        min_hops: int = 2,
        max_hops: int = 3,
        task_config: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        从MSCOCO生成Attribute Bridge Reasoning (ABR)任务

        ABR任务要求模型通过多步推理，从一个属性桥接到另一个属性。

        Args:
            source_data: MSCOCO数据列表
            num_samples: 生成的样本数
            min_hops: 最小推理跳数
            max_hops: 最大推理跳数
            task_config: 任务配置

        Returns:
            ABR任务列表
        """
        from .task_id_generator import TaskIDGenerator

        samples = []
        id_gen = TaskIDGenerator('attribute_bridge_reasoning', 'mscoco14')

        # 筛选包含足够对象的数据
        valid_data = []
        for item in source_data:
            annotations = item.get('annotations', [])
            if len(annotations) >= min_hops:
                valid_data.append(item)

        if not valid_data:
            return samples

        random.shuffle(valid_data)

        for idx in range(min(num_samples, len(valid_data))):
            item = valid_data[idx]
            annotations = item.get('annotations', [])

            # 选择用于推理链的对象
            num_hops = random.randint(min_hops, min(max_hops, len(annotations)))
            selected = random.sample(annotations, num_hops)

            # 构建推理链
            reasoning_chain = []
            for i, ann in enumerate(selected):
                category = ann.get('category_name', ann.get('category_id', f'object_{i}'))
                bbox = ann.get('bbox', [0, 0, 0, 0])
                reasoning_chain.append({
                    'step': i + 1,
                    'object': category,
                    'bbox': bbox,
                    'description': f"Identify {category} in the image"
                })

            # 生成问题
            start_obj = selected[0].get('category_name', 'first object')
            end_obj = selected[-1].get('category_name', 'last object')
            question = f"Starting from the {start_obj}, what object can you reach through {num_hops-1} spatial relationships?"

            samples.append({
                'task_id': id_gen.next(),
                'task_type': 'attribute_bridge_reasoning',
                'images': [item.get('image_path', item.get('file_name', ''))],
                'question': question,
                'answer': end_obj,
                'reasoning_chain': reasoning_chain,
                'reasoning_depth': num_hops,
                'metadata': {
                    'source_dataset': 'mscoco14',
                    'source_id': item.get('id', item.get('image_id', ''))
                }
            })

        return samples

    @staticmethod
    def _abr_from_vcr(
        source_data: List[Dict[str, Any]],
        num_samples: int,
        min_hops: int = 2,
        max_hops: int = 3,
        task_config: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        从VCR生成Attribute Bridge Reasoning (ABR)任务

        利用VCR的rationale来构建推理链。

        Args:
            source_data: VCR数据列表
            num_samples: 生成的样本数
            min_hops: 最小推理跳数
            max_hops: 最大推理跳数
            task_config: 任务配置

        Returns:
            ABR任务列表
        """
        from .task_id_generator import TaskIDGenerator

        samples = []
        id_gen = TaskIDGenerator('attribute_bridge_reasoning', 'vcr')

        # 筛选包含rationale的数据
        valid_data = [item for item in source_data
                      if item.get('rationale_choices') and item.get('rationale_label') is not None]

        if not valid_data:
            return samples

        random.shuffle(valid_data)

        for idx in range(min(num_samples, len(valid_data))):
            item = valid_data[idx]
            objects = item.get('objects', [])

            # 获取问题和答案
            question_tokens = item.get('question', [])
            question_text = DataGenerator._vcr_tokens_to_text(question_tokens, objects)

            answer_idx = item.get('answer_label', 0)
            answer_choices = item.get('answer_choices', [])
            if answer_idx < len(answer_choices):
                answer_tokens = answer_choices[answer_idx]
                answer_text = DataGenerator._vcr_tokens_to_text(answer_tokens, objects)
            else:
                answer_text = "Unknown"

            # 获取rationale
            rationale_idx = item.get('rationale_label', 0)
            rationale_choices = item.get('rationale_choices', [])
            if rationale_idx < len(rationale_choices):
                rationale_tokens = rationale_choices[rationale_idx]
                rationale_text = DataGenerator._vcr_tokens_to_text(rationale_tokens, objects)
            else:
                rationale_text = ""

            # 构建推理链
            reasoning_chain = [
                {'step': 1, 'description': f"Question: {question_text}"},
                {'step': 2, 'description': f"Rationale: {rationale_text}"},
                {'step': 3, 'description': f"Answer: {answer_text}"}
            ]

            samples.append({
                'task_id': id_gen.next(),
                'task_type': 'attribute_bridge_reasoning',
                'images': [item.get('image_path', item.get('img_fn', ''))],
                'question': question_text,
                'answer': answer_text,
                'rationale': rationale_text,
                'reasoning_chain': reasoning_chain,
                'reasoning_depth': 3,
                'metadata': {
                    'source_dataset': 'vcr',
                    'source_id': item.get('annot_id', ''),
                    'objects': objects
                }
            })

        return samples

    @staticmethod
    def _abr_from_visual_genome(
        source_data: List[Dict[str, Any]],
        num_samples: int,
        min_hops: int = 2,
        max_hops: int = 3,
        task_config: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        从Visual Genome生成Attribute Bridge Reasoning (ABR)任务

        利用Visual Genome的关系图来构建推理链。

        Args:
            source_data: Visual Genome数据列表
            num_samples: 生成的样本数
            min_hops: 最小推理跳数
            max_hops: 最大推理跳数
            task_config: 任务配置

        Returns:
            ABR任务列表
        """
        from .task_id_generator import TaskIDGenerator

        samples = []
        id_gen = TaskIDGenerator('attribute_bridge_reasoning', 'visual_genome')

        # 筛选包含关系的数据
        valid_data = []
        for item in source_data:
            relationships = item.get('relationships', [])
            if len(relationships) >= min_hops - 1:
                valid_data.append(item)

        if not valid_data:
            return samples

        random.shuffle(valid_data)

        for idx in range(min(num_samples, len(valid_data))):
            item = valid_data[idx]
            relationships = item.get('relationships', [])
            objects = item.get('objects', [])

            # 选择推理跳数
            num_rels = min(max_hops - 1, len(relationships))
            selected_rels = random.sample(relationships, num_rels) if num_rels > 0 else []

            # 构建推理链
            reasoning_chain = []
            current_objects = []

            for i, rel in enumerate(selected_rels):
                subj = rel.get('subject', {})
                obj = rel.get('object', {})
                predicate = rel.get('predicate', 'related to')

                subj_name = subj.get('name', subj.get('names', ['object'])[0] if isinstance(subj.get('names'), list) else 'object')
                obj_name = obj.get('name', obj.get('names', ['object'])[0] if isinstance(obj.get('names'), list) else 'object')

                reasoning_chain.append({
                    'step': i + 1,
                    'subject': subj_name,
                    'predicate': predicate,
                    'object': obj_name,
                    'description': f"{subj_name} {predicate} {obj_name}"
                })

                current_objects.extend([subj_name, obj_name])

            if reasoning_chain:
                start_obj = reasoning_chain[0]['subject']
                end_obj = reasoning_chain[-1]['object']
                question = f"What is the relationship path from {start_obj} to {end_obj}?"
                answer = " -> ".join([step['description'] for step in reasoning_chain])
            else:
                question = "Describe the objects in this image."
                answer = "No clear relationship chain found."

            samples.append({
                'task_id': id_gen.next(),
                'task_type': 'attribute_bridge_reasoning',
                'images': [item.get('image_path', item.get('url', ''))],
                'question': question,
                'answer': answer,
                'reasoning_chain': reasoning_chain,
                'reasoning_depth': len(reasoning_chain) + 1,
                'metadata': {
                    'source_dataset': 'visual_genome',
                    'source_id': item.get('image_id', '')
                }
            })

        return samples

    @staticmethod
    def _rc_from_mscoco(
        source_data: List[Dict[str, Any]],
        num_samples: int,
        task_config: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        从MSCOCO生成Relation Comparison (RC)任务

        RC任务要求模型比较不同图片中对象之间的空间关系。

        Args:
            source_data: MSCOCO数据列表
            num_samples: 生成的样本数
            task_config: 任务配置

        Returns:
            RC任务列表
        """
        from .task_id_generator import TaskIDGenerator

        samples = []
        id_gen = TaskIDGenerator('relation_comparison', 'mscoco14')

        # 筛选包含多个对象的数据
        valid_data = [item for item in source_data
                      if len(item.get('annotations', [])) >= 2]

        if len(valid_data) < 2:
            return samples

        random.shuffle(valid_data)
        n_images = task_config.get('n_images', 2) if task_config else 2

        for idx in range(0, min(num_samples, len(valid_data) - n_images + 1), n_images):
            selected_items = valid_data[idx:idx + n_images]

            # 收集所有图片中的关系
            all_relations = []
            images = []

            for img_idx, item in enumerate(selected_items):
                annotations = item.get('annotations', [])
                images.append(item.get('image_path', item.get('file_name', '')))

                if len(annotations) >= 2:
                    # 选择两个对象并确定空间关系
                    obj1, obj2 = random.sample(annotations, 2)
                    cat1 = obj1.get('category_name', 'object1')
                    cat2 = obj2.get('category_name', 'object2')

                    # 基于bbox确定空间关系
                    bbox1 = obj1.get('bbox', [0, 0, 0, 0])
                    bbox2 = obj2.get('bbox', [0, 0, 0, 0])

                    relation = DataGenerator._determine_spatial_relation(bbox1, bbox2)

                    all_relations.append({
                        'image_idx': img_idx,
                        'object1': cat1,
                        'object2': cat2,
                        'relation': relation
                    })

            if len(all_relations) >= 2:
                # 生成比较问题
                rel1 = all_relations[0]
                rel2 = all_relations[1] if len(all_relations) > 1 else all_relations[0]

                question = f"Compare the spatial relationship between objects in Image 1 and Image 2. In Image 1, what is the relationship between {rel1['object1']} and {rel1['object2']}? In Image 2, what is the relationship between {rel2['object1']} and {rel2['object2']}?"

                answer = f"Image 1: {rel1['object1']} is {rel1['relation']} {rel1['object2']}. Image 2: {rel2['object1']} is {rel2['relation']} {rel2['object2']}."

                samples.append({
                    'task_id': id_gen.next(),
                    'task_type': 'relation_comparison',
                    'images': images,
                    'question': question,
                    'answer': answer,
                    'relations': all_relations,
                    'metadata': {
                        'source_dataset': 'mscoco14',
                        'source_ids': [item.get('id', item.get('image_id', '')) for item in selected_items]
                    }
                })

        return samples

    @staticmethod
    def _rc_from_visual_genome(
        source_data: List[Dict[str, Any]],
        num_samples: int,
        task_config: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        从Visual Genome生成Relation Comparison (RC)任务

        利用Visual Genome丰富的关系标注来生成比较任务。

        Args:
            source_data: Visual Genome数据列表
            num_samples: 生成的样本数
            task_config: 任务配置

        Returns:
            RC任务列表
        """
        from .task_id_generator import TaskIDGenerator

        samples = []
        id_gen = TaskIDGenerator('relation_comparison', 'visual_genome')

        # 筛选包含关系的数据
        valid_data = [item for item in source_data
                      if len(item.get('relationships', [])) >= 1]

        if len(valid_data) < 2:
            return samples

        random.shuffle(valid_data)
        n_images = task_config.get('n_images', 2) if task_config else 2

        for idx in range(0, min(num_samples, len(valid_data) - n_images + 1), n_images):
            selected_items = valid_data[idx:idx + n_images]

            # 收集关系
            all_relations = []
            images = []

            for img_idx, item in enumerate(selected_items):
                relationships = item.get('relationships', [])
                images.append(item.get('image_path', item.get('url', '')))

                if relationships:
                    rel = random.choice(relationships)
                    subj = rel.get('subject', {})
                    obj = rel.get('object', {})
                    predicate = rel.get('predicate', 'related to')

                    subj_name = subj.get('name', 'object1')
                    obj_name = obj.get('name', 'object2')

                    all_relations.append({
                        'image_idx': img_idx,
                        'subject': subj_name,
                        'predicate': predicate,
                        'object': obj_name
                    })

            if len(all_relations) >= 2:
                rel1 = all_relations[0]
                rel2 = all_relations[1]

                question = f"Compare the relationships in the two images. Image 1: How is {rel1['subject']} related to {rel1['object']}? Image 2: How is {rel2['subject']} related to {rel2['object']}?"

                answer = f"Image 1: {rel1['subject']} {rel1['predicate']} {rel1['object']}. Image 2: {rel2['subject']} {rel2['predicate']} {rel2['object']}."

                samples.append({
                    'task_id': id_gen.next(),
                    'task_type': 'relation_comparison',
                    'images': images,
                    'question': question,
                    'answer': answer,
                    'relations': all_relations,
                    'metadata': {
                        'source_dataset': 'visual_genome',
                        'source_ids': [item.get('image_id', '') for item in selected_items]
                    }
                })

        return samples

    @staticmethod
    def _determine_spatial_relation(bbox1: List[float], bbox2: List[float]) -> str:
        """
        根据两个bounding box确定空间关系

        Args:
            bbox1: 第一个对象的bbox [x, y, width, height]
            bbox2: 第二个对象的bbox [x, y, width, height]

        Returns:
            空间关系描述字符串
        """
        if len(bbox1) < 4 or len(bbox2) < 4:
            return "near"

        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2

        # 计算中心点
        cx1, cy1 = x1 + w1/2, y1 + h1/2
        cx2, cy2 = x2 + w2/2, y2 + h2/2

        # 确定相对位置
        dx = cx2 - cx1
        dy = cy2 - cy1

        # 判断主要方向
        if abs(dx) > abs(dy):
            if dx > 0:
                return "to the left of"
            else:
                return "to the right of"
        else:
            if dy > 0:
                return "above"
            else:
                return "below"


# 导出兼容旧代码的别名
vcr_tokens_to_text = DataGenerator._vcr_tokens_to_text
