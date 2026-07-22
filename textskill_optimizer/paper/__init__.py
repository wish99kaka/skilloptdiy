"""Lazy public surface for paper-faithful contracts.

Keeping this module import-free is part of the final-test firewall: Python
executes a package initializer before any child module, so eager convenience
exports would otherwise load optimization code during a cold final-only import.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any


_EXPORT_MODULES = {
    "AlgorithmEvent": "types",
    "AlgorithmEventType": "types",
    "CheckpointAuthenticator": "checkpoint",
    "ClaimClass": "claims",
    "CodeIdentity": "zero_cost",
    "ContractViolation": "contract",
    "ConsumedSplit": "registry",
    "ConsumedSplitRegistry": "registry",
    "ControllerArtifact": "controller_process",
    "ControllerRegistration": "controller_process",
    "ControllerRegistry": "controller_process",
    "ControllerRole": "controller_process",
    "DataFirewallViolation": "errors",
    "EvidenceLevel": "claims",
    "EpochBatchCursor": "epoch_plan",
    "EpochCursor": "epoch_plan",
    "EpochBufferRecord": "types",
    "EpochCompletionResult": "epoch_loop",
    "EpochStepResult": "epoch_loop",
    "FastStepResult": "fast_loop",
    "LineageAssessment": "lineage",
    "LongitudinalEvidence": "longitudinal",
    "LongitudinalState": "longitudinal",
    "LongitudinalTaskComparison": "longitudinal",
    "OptimizerBackend": "backend",
    "OptimizerPayload": "data",
    "OptimizerRequest": "backend",
    "OptimizerResponse": "backend",
    "OptimizerRetryPolicy": "fast_loop",
    "OptimizerStage": "backend",
    "PaperArtifactKind": "artifacts",
    "PaperArtifactLineage": "artifacts",
    "PaperArtifactRecord": "artifacts",
    "PaperDataAccessPolicy": "data",
    "ObservedFailurePattern": "types",
    "PaperEdit": "types",
    "PaperEditOperation": "types",
    "PaperEditSource": "types",
    "PaperEpochPlan": "epoch_plan",
    "PaperEpochLoop": "epoch_loop",
    "PaperEpochCheckpoint": "checkpoint",
    "PaperFastLoop": "fast_loop",
    "PaperMechanismSpec": "epoch_plan",
    "PaperOptimizationController": "optimization",
    "PaperProfile": "config",
    "PaperProfileAssessment": "config",
    "PaperProfileViolation": "config",
    "PaperProvenanceAssessment": "provenance_lint",
    "PaperProvenanceViolation": "provenance_lint",
    "PaperRunAssessment": "contract",
    "PaperState": "types",
    "PaperSuggestion": "types",
    "PaperSuggestionPriority": "types",
    "PaperSuggestionType": "types",
    "ProfileViolation": "config",
    "ProvenanceLintViolation": "provenance_lint",
    "RunLineage": "lineage",
    "RunPhase": "data",
    "SelectionController": "data",
    "SelectionDecision": "data",
    "SelectionScore": "data",
    "SplitRole": "data",
    "SkillContractViolation": "errors",
    "StepTrainEvidence": "data",
    "TrainController": "data",
    "TrainEvidenceBatch": "data",
    "ZeroCostGateDecision": "zero_cost",
    "ZeroCostGateEvidence": "zero_cost",
    "ZeroCostGateViolation": "zero_cost",
    "assess_lineage": "lineage",
    "assess_paper_profile": "config",
    "assess_paper_provenance": "provenance_lint",
    "assess_paper_run": "contract",
    "assess_zero_cost_gate": "zero_cost",
    "canonical_json_sha256": "provenance",
    "load_paper_profile": "config",
    "strict_selection_decision": "data",
}

__all__ = sorted(_EXPORT_MODULES)


def __getattr__(name: str) -> Any:
    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(name)
    value = getattr(import_module(f"{__name__}.{module_name}"), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
