"""Injected optimizer-model seam for the paper engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Protocol, runtime_checkable


class OptimizerStage(str, Enum):
    REFLECT_FAILURE = "reflect_failure"
    REFLECT_SUCCESS = "reflect_success"
    REFINE = "refine"
    MERGE_FAILURE = "merge_failure"
    MERGE_SUCCESS = "merge_success"
    MERGE_FINAL_FAILURE_PRIORITIZED = "merge_final_failure_prioritized"
    RANK_TOP_L = "rank_top_l"
    PROPOSE_PATCH = "propose_patch"
    PROPOSE_SLOW_UPDATE = "propose_slow_update"
    UPDATE_META_SKILL = "update_meta_skill"


@dataclass(frozen=True)
class OptimizerRequest:
    call_id: str
    stage: OptimizerStage
    prompt: str
    response_schema: Mapping[str, Any]
    system_prompt: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.call_id.strip():
            raise ValueError("optimizer request requires a call_id")
        if not self.prompt.strip():
            raise ValueError("optimizer request requires a prompt")


@dataclass(frozen=True)
class OptimizerResponse:
    call_id: str
    payload: Mapping[str, Any]
    model_id: str
    usage: Mapping[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.call_id.strip():
            raise ValueError("optimizer response requires a call_id")
        if not self.model_id.strip():
            raise ValueError("optimizer response requires a model_id")


@runtime_checkable
class OptimizerBackend(Protocol):
    """One injected model-call interface shared by every paper optimizer stage."""

    def complete(self, request: OptimizerRequest) -> OptimizerResponse:
        """Return one schema-constrained optimizer response."""

        ...
