"""Paper-faithful SkillOpt contracts, independent from extension protocols."""

from .config import (
    PaperProfile,
    PaperProfileAssessment,
    PaperProfileViolation,
    ProfileViolation,
    assess_paper_profile,
    load_paper_profile,
)
from .claims import ClaimClass, EvidenceLevel
from .backend import (
    OptimizerBackend,
    OptimizerRequest,
    OptimizerResponse,
    OptimizerStage,
)
from .contract import ContractViolation, PaperRunAssessment, assess_paper_run
from .data import (
    DataFirewallViolation,
    PaperDataAccessPolicy,
    RunPhase,
    SelectionDecision,
    SelectionScore,
    SplitRole,
    strict_selection_decision,
)
from .lineage import LineageAssessment, RunLineage, assess_lineage
from .registry import ConsumedSplit, ConsumedSplitRegistry
from .provenance import canonical_json_sha256
from .types import (
    AlgorithmEvent,
    AlgorithmEventType,
    PaperEdit,
    PaperEditOperation,
    PaperState,
)

__all__ = [
    "AlgorithmEvent",
    "AlgorithmEventType",
    "ClaimClass",
    "ContractViolation",
    "ConsumedSplit",
    "ConsumedSplitRegistry",
    "DataFirewallViolation",
    "EvidenceLevel",
    "LineageAssessment",
    "OptimizerBackend",
    "OptimizerRequest",
    "OptimizerResponse",
    "OptimizerStage",
    "PaperProfile",
    "PaperProfileAssessment",
    "PaperProfileViolation",
    "PaperRunAssessment",
    "PaperEdit",
    "PaperEditOperation",
    "PaperState",
    "PaperDataAccessPolicy",
    "ProfileViolation",
    "RunLineage",
    "RunPhase",
    "SelectionDecision",
    "SelectionScore",
    "SplitRole",
    "assess_lineage",
    "assess_paper_run",
    "assess_paper_profile",
    "canonical_json_sha256",
    "load_paper_profile",
    "strict_selection_decision",
]
