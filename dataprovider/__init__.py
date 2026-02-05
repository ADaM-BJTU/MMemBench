"""
M3Bench DataProvider Module
===========================

This module provides data loading and generation capabilities for M3Bench.

Components:
- DataLoader: Load datasets by ID and extract needed data
- DataGeneratorV2: Config-driven task generator
- ConfigLoader: Load and manage dataset configurations
- Task Generators: Specialized generators for each task type
"""

from .loader import DataLoader
from .generator_v2 import DataGeneratorV2
from .config_loader import ConfigLoader, load_config

__all__ = [
    'DataLoader',
    'DataGeneratorV2',
    'ConfigLoader',
    'load_config'
]

__version__ = "0.4.0"