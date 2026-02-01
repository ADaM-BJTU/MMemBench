"""
M3Bench Simulator Module
========================

Two-tier architecture:
1. Strategic Simulator (NEW): Task-driven adaptive testing with difficulty escalation
2. LLM User Simulator (Refactored): Context-driven LLM simulation

Core Components:
- StrategicSimulator: Main simulator for benchmarking (recommended)
- LLMUserSimulator: Refactored with ContextBuilder + SimulatorState
- Evaluator: Independent evaluation module (STRESS_TEST / LENIENT modes)
- ActionSpace: Structured action definitions with clear purposes
- ContextPadder: Pseudo multi-turn for long context testing
- MemoryStore: Conversation memory management
- LLMClient: API client for LLM calls

New Components (Task 3):
- ContextBuilder: 统一构建核心模型上下文
- SimulatorState: 状态机管理

New Components (Window 3 - Query Generation Enhancement):
- LLMQueryGenerator: LLM驱动的Query生成器
- HybridQueryGenerator: 混合模式Query生成器
- DynamicLengthSelector: 动态长度选择器
- LengthSelectionStrategy: 长度选择策略枚举

New Components (Window 2 - VCR Rationale & Reasoning Chain):
- ReasoningChainBuilder: 推理链构建器
- ChainQuery: 推理链查询数据类
- ChainExecutionManager: 推理链执行管理器
"""

# New architecture (recommended)
from .strategic_simulator import StrategicSimulator
from .evaluator import Evaluator, EvaluationMode, EvaluationResult, DIFFICULTY_LEVELS
from .action_space import ACTION_DEFINITIONS, TASK_STRATEGIES, ActionCategory
from .context_padder import ContextPadder, create_long_context_conversation

# Batch processing (Window 1)
from .batch_task_simulator import BatchTaskSimulator, BatchConfig, BatchResult

# Core utilities
from .llm_client import LLMClient
from .memory_store import MemoryStore

# Refactored LLM Simulator (Task 3)
from .llm_user_simulator import LLMUserSimulator
from .context_builder import ContextBuilder
from .simulator_state import SimulatorState, Phase, ResponseEvaluation

# Entity extraction and action selection
from .entity_extractor import EntityExtractor
from .action_selector import ActionSelector

# Legacy components
from .user_simulator import UserSimulator
from .query_generator import QueryGenerator
from .prompt_templates import PromptTemplates

# Window 3: Query Generation Enhancement
from .llm_query_generator import LLMQueryGenerator, HybridQueryGenerator
from .length_selector import (
    DynamicLengthSelector,
    LengthSelectionStrategy,
    get_length_selector_presets,
    create_length_selector
)

# Window 2: VCR Rationale & Reasoning Chain
from .reasoning_chain_builder import (
    ReasoningChainBuilder,
    ChainQuery,
    ChainExecutionManager,
    ChainExecutionState
)

__all__ = [
    # New architecture
    'StrategicSimulator',
    'Evaluator',
    'EvaluationMode',
    'EvaluationResult',
    'DIFFICULTY_LEVELS',
    'ACTION_DEFINITIONS',
    'TASK_STRATEGIES',
    'ActionCategory',
    'ContextPadder',
    'create_long_context_conversation',

    # Batch processing (Window 1)
    'BatchTaskSimulator',
    'BatchConfig',
    'BatchResult',

    # Core utilities
    'LLMClient',
    'MemoryStore',

    # Refactored LLM Simulator (Task 3)
    'LLMUserSimulator',
    'ContextBuilder',
    'SimulatorState',
    'Phase',
    'ResponseEvaluation',

    # Entity & Action
    'EntityExtractor',
    'ActionSelector',

    # Legacy
    'UserSimulator',
    'QueryGenerator',
    'PromptTemplates',

    # Window 3: Query Generation Enhancement
    'LLMQueryGenerator',
    'HybridQueryGenerator',
    'DynamicLengthSelector',
    'LengthSelectionStrategy',
    'get_length_selector_presets',
    'create_length_selector',

    # Window 2: VCR Rationale & Reasoning Chain
    'ReasoningChainBuilder',
    'ChainQuery',
    'ChainExecutionManager',
    'ChainExecutionState'
]
