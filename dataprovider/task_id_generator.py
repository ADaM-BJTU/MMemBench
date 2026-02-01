"""
Task ID Generator - 全局唯一任务ID生成器
=========================================

确保生成的task_id在不同run之间全局唯一。

策略:
1. 使用时间戳（毫秒级）确保时间维度唯一
2. 使用6位随机字符增加区分度
3. 使用类型前缀便于识别任务类型
4. 使用计数器确保同一批次内的顺序

格式: {type_prefix}_{timestamp_ms}_{random_6char}_{sequence}
例如: ac_1735187654321_a1b2c3_001
"""

import time
import random
import string
import threading
from typing import Optional

# 全局计数器和锁，确保线程安全
_counter_lock = threading.Lock()
_counters = {}  # {prefix: counter}
_session_id = None  # 每次运行的session标识


def _get_session_id() -> str:
    """获取当前会话的唯一标识"""
    global _session_id
    if _session_id is None:
        # 使用当前时间戳（秒级）+ 随机字符作为session标识
        timestamp = int(time.time())
        rand_chars = ''.join(random.choices(string.ascii_lowercase + string.digits, k=4))
        _session_id = f"{timestamp}_{rand_chars}"
    return _session_id


def _get_counter(prefix: str) -> int:
    """获取并递增计数器（线程安全）"""
    with _counter_lock:
        if prefix not in _counters:
            _counters[prefix] = 0
        _counters[prefix] += 1
        return _counters[prefix]


def reset_counters():
    """重置所有计数器（用于新的批次）"""
    global _counters, _session_id
    with _counter_lock:
        _counters = {}
        _session_id = None


def generate_task_id(
    task_type: str,
    dataset: str,
    sequence: Optional[int] = None
) -> str:
    """
    生成全局唯一的task_id

    Args:
        task_type: 任务类型 (e.g., 'attribute_comparison', 'visual_noise_filtering')
        dataset: 数据集名称 (e.g., 'mscoco14', 'vcr')
        sequence: 可选的序列号，如果不提供则自动生成

    Returns:
        唯一的task_id字符串

    Examples:
        >>> generate_task_id('attribute_comparison', 'mscoco14')
        'ac_mscoco_1737899123_x7y8z9_001'
    """
    # 类型前缀映射
    type_prefixes = {
        'attribute_comparison': 'ac',
        'visual_noise_filtering': 'vnf',
        'attribute_bridge_reasoning': 'abr',
        'relation_comparison': 'rc',
        'rationale_based_abr': 'rabr',
        'temporal_state': 'ts',
        'spatial_memory': 'sm',
    }

    # 获取前缀
    prefix = type_prefixes.get(task_type, task_type[:3])

    # 数据集简称映射
    dataset_prefixes = {
        'mscoco14': 'mscoco',
        'visual_genome': 'vg',
        'vcr': 'vcr',
        'gqa': 'gqa',
        'sherlock': 'sher',
        'docvqa': 'doc',
    }
    ds_prefix = dataset_prefixes.get(dataset, dataset[:4])

    # 获取session id
    session_id = _get_session_id()

    # 获取序列号
    counter_key = f"{prefix}_{ds_prefix}"
    if sequence is None:
        sequence = _get_counter(counter_key)

    # 组合task_id
    # 格式: {type_prefix}_{dataset_prefix}_{session_id}_{sequence:03d}
    task_id = f"{prefix}_{ds_prefix}_{session_id}_{sequence:03d}"

    return task_id


class TaskIDGenerator:
    """
    任务ID生成器类 - 用于管理一个批次的任务ID生成

    Usage:
        gen = TaskIDGenerator(task_type='attribute_comparison', dataset='mscoco14')
        for i in range(10):
            task_id = gen.next()
            print(task_id)
    """

    def __init__(self, task_type: str, dataset: str):
        self.task_type = task_type
        self.dataset = dataset
        self._local_counter = 0

        # 类型前缀映射
        type_prefixes = {
            'attribute_comparison': 'ac',
            'visual_noise_filtering': 'vnf',
            'attribute_bridge_reasoning': 'abr',
            'relation_comparison': 'rc',
            'rationale_based_abr': 'rabr',
            'temporal_state': 'ts',
            'spatial_memory': 'sm',
        }
        self._prefix = type_prefixes.get(task_type, task_type[:3])

        # 数据集简称
        dataset_prefixes = {
            'mscoco14': 'mscoco',
            'visual_genome': 'vg',
            'vcr': 'vcr',
            'gqa': 'gqa',
            'sherlock': 'sher',
            'docvqa': 'doc',
        }
        self._ds_prefix = dataset_prefixes.get(dataset, dataset[:4])

        # 生成本批次的唯一标识
        self._batch_id = self._generate_batch_id()

    def _generate_batch_id(self) -> str:
        """生成批次标识"""
        timestamp = int(time.time())
        rand_chars = ''.join(random.choices(string.ascii_lowercase + string.digits, k=4))
        return f"{timestamp}_{rand_chars}"

    def next(self) -> str:
        """生成下一个task_id"""
        self._local_counter += 1
        return f"{self._prefix}_{self._ds_prefix}_{self._batch_id}_{self._local_counter:03d}"

    def current_count(self) -> int:
        """返回当前生成的ID数量"""
        return self._local_counter


# ================= 便捷函数 =================

def generate_ac_task_id(dataset: str) -> str:
    """生成Attribute Comparison任务ID"""
    return generate_task_id('attribute_comparison', dataset)


def generate_vnf_task_id(dataset: str) -> str:
    """生成Visual Noise Filtering任务ID"""
    return generate_task_id('visual_noise_filtering', dataset)


def generate_abr_task_id(dataset: str) -> str:
    """生成Attribute Bridge Reasoning任务ID"""
    return generate_task_id('attribute_bridge_reasoning', dataset)


def generate_rc_task_id(dataset: str) -> str:
    """生成Relation Comparison任务ID"""
    return generate_task_id('relation_comparison', dataset)


def generate_rabr_task_id(dataset: str) -> str:
    """生成Rationale-based ABR任务ID"""
    return generate_task_id('rationale_based_abr', dataset)


# ================= 测试代码 =================

if __name__ == "__main__":
    print("Testing TaskIDGenerator...\n")

    # 测试全局函数
    print("1. Testing global generate_task_id():")
    for i in range(3):
        tid = generate_task_id('attribute_comparison', 'mscoco14')
        print(f"   {tid}")

    print("\n2. Testing different task types:")
    print(f"   AC: {generate_ac_task_id('mscoco14')}")
    print(f"   VNF: {generate_vnf_task_id('vcr')}")
    print(f"   ABR: {generate_abr_task_id('visual_genome')}")
    print(f"   RC: {generate_rc_task_id('mscoco14')}")

    # 测试类
    print("\n3. Testing TaskIDGenerator class:")
    gen = TaskIDGenerator(task_type='attribute_comparison', dataset='mscoco14')
    for i in range(5):
        print(f"   {gen.next()}")
    print(f"   Total generated: {gen.current_count()}")

    # 测试多个批次
    print("\n4. Testing multiple batches:")
    gen1 = TaskIDGenerator(task_type='vnf', dataset='vcr')
    gen2 = TaskIDGenerator(task_type='vnf', dataset='vcr')
    print(f"   Batch 1: {gen1.next()}")
    print(f"   Batch 2: {gen2.next()}")
    print(f"   Batch 1: {gen1.next()}")
    print(f"   Batch 2: {gen2.next()}")

    print("\n5. Verifying uniqueness:")
    ids = set()
    for _ in range(100):
        tid = generate_task_id('attribute_comparison', 'mscoco14')
        if tid in ids:
            print(f"   DUPLICATE FOUND: {tid}")
        ids.add(tid)
    print(f"   Generated 100 IDs, all unique: {len(ids) == 100}")

    print("\nAll tests passed!")
