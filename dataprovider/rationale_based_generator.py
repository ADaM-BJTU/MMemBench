"""
Rationale-Based Task Generator
==============================

使用VCR等数据集的rationale标注生成多跳推理任务。

核心思想:
- Rationale包含推理链: 观察A → 推断B → 结论C
- 将推理链拆解为多个验证点
- 每一步都需要基于前一步的答案
"""

import random
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field

from .task_id_generator import TaskIDGenerator

logger = logging.getLogger(__name__)


@dataclass
class ReasoningStep:
    """推理链中的单个步骤"""
    step_index: int
    step_type: str  # "observation" | "inference" | "conclusion"
    content: str
    depends_on: List[int]  # 依赖的前置步骤索引
    verification_query: str  # 用于验证的问题
    expected_response_hints: List[str]  # 期望回答中应包含的关键词


@dataclass
class ReasoningChain:
    """完整的推理链"""
    chain_id: str
    source_sample_id: str
    question: str
    final_answer: str
    steps: List[ReasoningStep]
    total_hops: int

    def to_dict(self) -> Dict:
        return {
            'chain_id': self.chain_id,
            'source_sample_id': self.source_sample_id,
            'question': self.question,
            'final_answer': self.final_answer,
            'total_hops': self.total_hops,
            'steps': [
                {
                    'step_index': s.step_index,
                    'step_type': s.step_type,
                    'content': s.content,
                    'depends_on': s.depends_on,
                    'verification_query': s.verification_query,
                    'expected_hints': s.expected_response_hints
                }
                for s in self.steps
            ]
        }


class RationaleBasedABRGenerator:
    """
    基于Rationale的属性桥接推理任务生成器

    生成的任务包含:
    - 明确的推理链
    - 每步的验证问题
    - 预期的回答提示
    """

    # 推理步骤类型的验证问题模板
    VERIFICATION_TEMPLATES = {
        "observation": [
            "你能看到{observation_target}吗？",
            "图中的{observation_target}是什么样的？",
            "描述一下{observation_target}。",
            "Can you see {observation_target}?",
            "What does {observation_target} look like?",
            "Please describe {observation_target}."
        ],
        "inference": [
            "根据{evidence}，你能推断出什么？",
            "这说明了什么？",
            "基于这个观察，{inference_question}？",
            "Based on {evidence}, what can you infer?",
            "What does this indicate?",
            "Given this observation, what can we conclude?"
        ],
        "conclusion": [
            "所以，{conclusion_question}？",
            "综合以上，你的结论是什么？",
            "因此，{final_question}",
            "So, {conclusion_question}?",
            "What's your final conclusion?",
            "Based on all of this, what is the answer?"
        ]
    }

    @classmethod
    def generate_from_vcr(
        cls,
        source_data: List[Dict],
        num_samples: int,
        task_config: Dict[str, Any],
        templates: Dict[str, Any]
    ) -> List[Dict]:
        """
        从VCR数据生成基于rationale的ABR任务

        Args:
            source_data: VCR样本列表（必须包含rationale）
            num_samples: 目标样本数
            task_config: 任务配置
                - min_reasoning_steps: int = 2
                - max_reasoning_steps: int = 5
                - require_visual_grounding: bool = True
            templates: 问题模板

        Returns:
            生成的任务列表
        """
        min_steps = task_config.get('min_reasoning_steps', 2)
        max_steps = task_config.get('max_reasoning_steps', 5)
        require_grounding = task_config.get('require_visual_grounding', True)

        # 使用全局唯一ID生成器
        id_gen = TaskIDGenerator(task_type='rationale_based_abr', dataset='vcr')

        tasks = []
        attempts = 0
        max_attempts = num_samples * 5

        # 筛选有效样本
        valid_samples = [
            s for s in source_data
            if s.get('reasoning_steps') and len(s.get('reasoning_steps', [])) >= min_steps
        ]

        if not valid_samples:
            logger.warning(f"No valid VCR samples with >= {min_steps} reasoning steps")
            # 降低要求，使用所有有rationale的样本
            valid_samples = [
                s for s in source_data
                if s.get('rationale_text')
            ]

        logger.info(f"Found {len(valid_samples)} valid VCR samples for rationale-based ABR")

        for sample in valid_samples:
            if len(tasks) >= num_samples:
                break

            attempts += 1
            if attempts > max_attempts:
                break

            # 构建推理链
            reasoning_chain = cls._build_reasoning_chain(sample)

            if reasoning_chain is None:
                continue

            if len(reasoning_chain.steps) < min_steps:
                continue

            # 构建任务
            task = cls._build_task_from_chain(
                sample=sample,
                reasoning_chain=reasoning_chain,
                id_gen=id_gen
            )

            if task:
                tasks.append(task)

        logger.info(f"Generated {len(tasks)} rationale-based ABR tasks from {attempts} attempts")
        return tasks

    @classmethod
    def _build_reasoning_chain(cls, sample: Dict) -> Optional[ReasoningChain]:
        """
        从VCR样本构建推理链

        策略:
        1. 将rationale_text作为整体推理
        2. 使用reasoning_steps作为分步验证
        3. 为每步生成验证问题
        """
        try:
            sample_id = sample.get('sample_id', 'unknown')
            question = sample.get('question_text', '')
            answer = sample.get('answer_text', '')
            rationale = sample.get('rationale_text', '')
            reasoning_steps_text = sample.get('reasoning_steps', [])

            if not reasoning_steps_text:
                # 如果没有分步，使用整个rationale作为一步
                reasoning_steps_text = [rationale] if rationale else []

            if not reasoning_steps_text:
                return None

            # 构建推理步骤
            steps = []
            for i, step_text in enumerate(reasoning_steps_text):
                # 确定步骤类型
                if i == 0:
                    step_type = "observation"
                elif i == len(reasoning_steps_text) - 1:
                    step_type = "conclusion"
                else:
                    step_type = "inference"

                # 提取关键词作为期望回答提示
                hints = cls._extract_key_hints(step_text)

                # 生成验证问题
                verification_query = cls._generate_verification_query(
                    step_type=step_type,
                    step_content=step_text,
                    question=question,
                    previous_steps=[s.content for s in steps]
                )

                step = ReasoningStep(
                    step_index=i,
                    step_type=step_type,
                    content=step_text,
                    depends_on=list(range(i)) if i > 0 else [],
                    verification_query=verification_query,
                    expected_response_hints=hints
                )
                steps.append(step)

            return ReasoningChain(
                chain_id=f"chain_{sample_id}",
                source_sample_id=sample_id,
                question=question,
                final_answer=answer,
                steps=steps,
                total_hops=len(steps)
            )

        except Exception as e:
            logger.debug(f"Failed to build reasoning chain: {e}")
            return None

    @classmethod
    def _generate_verification_query(
        cls,
        step_type: str,
        step_content: str,
        question: str,
        previous_steps: List[str]
    ) -> str:
        """为推理步骤生成验证问题"""

        templates = cls.VERIFICATION_TEMPLATES.get(step_type, cls.VERIFICATION_TEMPLATES["inference"])
        template = random.choice(templates)

        # 提取step_content中的关键部分
        key_part = cls._extract_observation_target(step_content)

        # 填充模板
        query = template.format(
            observation_target=key_part,
            evidence=key_part if previous_steps else "你看到的",
            inference_question=f"关于{key_part}",
            conclusion_question=question.rstrip("?？"),
            final_question=question
        )

        return query

    @staticmethod
    def _extract_observation_target(step_content: str) -> str:
        """从步骤内容中提取观察目标"""
        import re

        # 优先提取[object_n]格式的引用
        obj_refs = re.findall(r'\[([^\]]+)\]', step_content)
        if obj_refs:
            return obj_refs[0]

        # 否则提取前几个词
        words = step_content.split()[:5]
        return " ".join(words)

    @staticmethod
    def _extract_key_hints(step_content: str) -> List[str]:
        """提取关键词作为回答提示"""
        import re

        hints = []

        # 提取物体引用
        obj_refs = re.findall(r'\[([^\]]+)\]', step_content)
        hints.extend(obj_refs)

        # 提取形容词和名词（简化）
        key_words = [
            'expression', 'face', 'looking', 'standing', 'sitting',
            'holding', 'wearing', 'walking', 'running', 'talking',
            'smiling', 'crying', 'angry', 'happy', 'sad', 'surprised',
            '表情', '看', '站', '坐', '拿', '穿', '走', '说话',
            '笑', '哭', '生气', '开心', '难过', '惊讶'
        ]

        for word in key_words:
            if word.lower() in step_content.lower():
                hints.append(word)

        return hints[:5]  # 最多5个提示

    @classmethod
    def _build_task_from_chain(
        cls,
        sample: Dict,
        reasoning_chain: ReasoningChain,
        id_gen: TaskIDGenerator
    ) -> Optional[Dict]:
        """从推理链构建任务"""

        try:
            task = {
                'task_id': id_gen.next(),
                'task_type': 'rationale_based_abr',

                # 图片
                'images': [sample.get('image_path', '')],

                # 问题和答案
                'question': reasoning_chain.question,
                'answer': reasoning_chain.final_answer,

                # ⭐ 推理链核心信息
                'reasoning_chain': reasoning_chain.to_dict(),
                'reasoning_depth': reasoning_chain.total_hops,

                # 原始rationale作为ground truth
                'ground_truth_rationale': sample.get('rationale_text', ''),

                # 分步验证信息
                'verification_queries': [
                    {
                        'step': step.step_index,
                        'query': step.verification_query,
                        'expected_hints': step.expected_response_hints,
                        'step_type': step.step_type
                    }
                    for step in reasoning_chain.steps
                ],

                # 元数据
                'metadata': {
                    'source_dataset': 'vcr',
                    'source_sample_id': sample.get('sample_id', ''),
                    'has_explicit_reasoning': True,
                    'movie': sample.get('movie', ''),
                    'objects': sample.get('objects', []),
                    'annot_id': sample.get('annot_id', '')
                }
            }

            return task

        except Exception as e:
            logger.debug(f"Failed to build task from chain: {e}")
            return None


class RationaleAnalyzer:
    """
    Rationale分析工具

    用于分析VCR等数据集中rationale的特征，
    帮助理解数据分布和优化任务生成策略。
    """

    @staticmethod
    def analyze_vcr_rationales(samples: List[Dict], max_samples: int = 100) -> Dict[str, Any]:
        """
        分析VCR数据集中的rationale特征

        输出:
        - Rationale平均长度
        - 推理类型分布
        - 物体引用频率
        - 复杂推理链占比
        """
        import re
        from collections import Counter

        stats = {
            'total_samples': 0,
            'samples_with_rationale': 0,
            'rationale_lengths': [],
            'reasoning_step_counts': [],
            'object_references': [],
            'has_causal': 0,
            'has_comparison': 0,
            'has_temporal': 0,
            'has_emotional': 0,
        }

        causal_keywords = ['because', 'so', 'therefore', 'thus', 'since', 'as', 'hence']
        comparison_keywords = ['like', 'similar', 'different', 'more', 'less', 'same', 'unlike']
        temporal_keywords = ['before', 'after', 'then', 'when', 'while', 'during', 'now']
        emotional_keywords = ['happy', 'sad', 'angry', 'upset', 'excited', 'scared', 'surprised', 'disgusted']

        for sample in samples[:max_samples]:
            stats['total_samples'] += 1

            rationale_text = sample.get('rationale_text', '')
            if not rationale_text:
                continue

            stats['samples_with_rationale'] += 1

            # 统计长度
            stats['rationale_lengths'].append(len(rationale_text.split()))

            # 统计推理步骤数
            reasoning_steps = sample.get('reasoning_steps', [])
            stats['reasoning_step_counts'].append(len(reasoning_steps))

            # 统计物体引用
            obj_refs = len(re.findall(r'\[([^\]]+)\]', rationale_text))
            stats['object_references'].append(obj_refs)

            # 检测推理类型
            text_lower = rationale_text.lower()

            if any(kw in text_lower for kw in causal_keywords):
                stats['has_causal'] += 1
            if any(kw in text_lower for kw in comparison_keywords):
                stats['has_comparison'] += 1
            if any(kw in text_lower for kw in temporal_keywords):
                stats['has_temporal'] += 1
            if any(kw in text_lower for kw in emotional_keywords):
                stats['has_emotional'] += 1

        # 计算汇总统计
        n = stats['samples_with_rationale'] or 1
        summary = {
            'total_samples': stats['total_samples'],
            'samples_with_rationale': stats['samples_with_rationale'],
            'rationale_coverage': stats['samples_with_rationale'] / stats['total_samples'] if stats['total_samples'] > 0 else 0,
            'avg_rationale_length': sum(stats['rationale_lengths']) / n if stats['rationale_lengths'] else 0,
            'avg_reasoning_steps': sum(stats['reasoning_step_counts']) / n if stats['reasoning_step_counts'] else 0,
            'avg_object_references': sum(stats['object_references']) / n if stats['object_references'] else 0,
            'causal_reasoning_rate': stats['has_causal'] / n,
            'comparison_rate': stats['has_comparison'] / n,
            'temporal_rate': stats['has_temporal'] / n,
            'emotional_content_rate': stats['has_emotional'] / n,
            'step_distribution': Counter(stats['reasoning_step_counts']).most_common(5)
        }

        return summary

    @staticmethod
    def print_analysis_report(summary: Dict[str, Any]):
        """打印分析报告"""
        print("\n" + "="*60)
        print("VCR Rationale Analysis Report")
        print("="*60)
        print(f"Total samples analyzed: {summary['total_samples']}")
        print(f"Samples with rationale: {summary['samples_with_rationale']} ({summary['rationale_coverage']*100:.1f}%)")
        print(f"\nAverage rationale length: {summary['avg_rationale_length']:.1f} words")
        print(f"Average reasoning steps: {summary['avg_reasoning_steps']:.1f}")
        print(f"Average object references: {summary['avg_object_references']:.1f}")
        print(f"\nReasoning types:")
        print(f"  - Causal reasoning: {summary['causal_reasoning_rate']*100:.1f}%")
        print(f"  - Comparison: {summary['comparison_rate']*100:.1f}%")
        print(f"  - Temporal: {summary['temporal_rate']*100:.1f}%")
        print(f"  - Emotional content: {summary['emotional_content_rate']*100:.1f}%")
        print(f"\nStep distribution (top 5):")
        for steps, count in summary['step_distribution']:
            print(f"  - {steps} steps: {count} samples")
        print("="*60)
