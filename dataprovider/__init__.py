"""
M3Bench DataProvider Module
===========================

This module provides data loading and generation capabilities for M3Bench.

Components:
- DataLoader: Load datasets by ID and extract needed data
- DataGenerator: 基础数据生成器（兼容旧代码）
- DataGeneratorV2: Config-driven task generator
- ConfigLoader: Load and manage dataset configurations
- Task Generators: Specialized generators for each task type
- RationaleBasedABRGenerator: 窗口2新增 - 基于Rationale的ABR生成器
- TaskIDGenerator: 全局唯一任务ID生成器
"""

from .loader import DataLoader
from .generator import DataGenerator, vcr_tokens_to_text
from .generator_v2 import DataGeneratorV2
from .config_loader import ConfigLoader, load_config
# 窗口2新增
from .rationale_based_generator import RationaleBasedABRGenerator, RationaleAnalyzer
# 任务ID生成器
from .task_id_generator import TaskIDGenerator, generate_task_id, reset_counters

__all__ = [
    'DataLoader',
    'DataGenerator',
    'vcr_tokens_to_text',
    'DataGeneratorV2',
    'ConfigLoader',
    'load_config',
    # 窗口2新增
    'RationaleBasedABRGenerator',
    'RationaleAnalyzer',
    # 任务ID生成
    'TaskIDGenerator',
    'generate_task_id',
    'reset_counters'
]

__version__ = "0.7.0"  # 更新：修复generator.py依赖缺失