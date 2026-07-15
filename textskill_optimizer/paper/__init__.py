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
    "ClaimClass": "claims",
    "ContractViolation": "contract",
    "ConsumedSplit": "registry",
    "ConsumedSplitRegistry": "registry",
    "ControllerArtifact": "controller_process",
    "ControllerRegistration": "controller_process",
    "ControllerRegistry": "controller_process",
    "ControllerRole": "controller_process",
    "DataFirewallViolation": "errors",
    "EvidenceLevel": "claims",
    "LineageAssessment": "lineage",
    "OptimizerBackend": "backend",
    "OptimizerPayload": "data",
    "OptimizerRequest": "backend",
    "OptimizerResponse": "backend",
    "OptimizerStage": "backend",
    "PaperDataAccessPolicy": "data",
    "PaperEdit": "types",
    "PaperEditOperation": "types",
    "PaperOptimizationController": "optimization",
    "PaperProfile": "config",
    "PaperProfileAssessment": "config",
    "PaperProfileViolation": "config",
    "PaperRunAssessment": "contract",
    "PaperState": "types",
    "ProfileViolation": "config",
    "RunLineage": "lineage",
    "RunPhase": "data",
    "SelectionController": "data",
    "SelectionDecision": "data",
    "SelectionScore": "data",
    "SplitRole": "data",
    "TrainController": "data",
    "TrainEvidenceBatch": "data",
    "assess_lineage": "lineage",
    "assess_paper_profile": "config",
    "assess_paper_run": "contract",
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
