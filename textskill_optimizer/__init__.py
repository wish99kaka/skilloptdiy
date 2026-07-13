"""Text-space optimization for reusable agent skill documents."""

from .interfaces import (
    EDITOR_CAPABILITY_ATOMIC_EDITS,
    EDITOR_CAPABILITY_FULL_REPLACEMENT,
    SkillEditor,
    SkillRunner,
    SkillScorer,
)
from .command_editor import CommandEditorConfig, CommandSkillEditor
from .models import (
    AtomicEdit,
    EditProposal,
    EvaluationReport,
    OptimizationHistoryItem,
    OptimizationResult,
    OptimizerStateUpdate,
    RejectedProposal,
    Score,
    Task,
    TaskOutput,
    TaskResult,
)
from .optimizer import OptimizerConfig, SkillOptimizer
from .executive_optimizer import (
    ExecutiveOptimizationResult,
    ExecutiveOptimizerConfig,
    ExecutiveSkillOptimizer,
)

__all__ = [
    "AtomicEdit",
    "EditProposal",
    "EDITOR_CAPABILITY_ATOMIC_EDITS",
    "EDITOR_CAPABILITY_FULL_REPLACEMENT",
    "ExecutiveOptimizationResult",
    "ExecutiveOptimizerConfig",
    "ExecutiveSkillOptimizer",
    "CommandEditorConfig",
    "CommandSkillEditor",
    "EvaluationReport",
    "OptimizationHistoryItem",
    "OptimizationResult",
    "OptimizerConfig",
    "OptimizerStateUpdate",
    "RejectedProposal",
    "Score",
    "SkillEditor",
    "SkillOptimizer",
    "SkillRunner",
    "SkillScorer",
    "Task",
    "TaskOutput",
    "TaskResult",
]
