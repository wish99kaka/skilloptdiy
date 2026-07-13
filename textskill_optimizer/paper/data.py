"""Data-firewall value types used at paper controller seams."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping


class DataFirewallViolation(ValueError):
    """Raised when non-authorized data crosses a paper controller seam."""


class SplitRole(str, Enum):
    TRAIN = "train"
    SELECTION = "selection"
    TEST = "test"


class RunPhase(str, Enum):
    OPTIMIZATION = "optimization"
    FINAL_EVALUATION = "final_evaluation"


@dataclass(frozen=True)
class PaperDataAccessPolicy:
    """Fail-closed access matrix for optimization and final evaluation."""

    def require(self, *, split: SplitRole, phase: RunPhase) -> None:
        allowed = {
            RunPhase.OPTIMIZATION: {SplitRole.TRAIN, SplitRole.SELECTION},
            RunPhase.FINAL_EVALUATION: {SplitRole.TEST},
        }
        if split not in allowed[phase]:
            raise DataFirewallViolation(
                f"{split.value} split is forbidden during {phase.value}"
            )


@dataclass(frozen=True)
class SelectionScore:
    """The complete selection-to-controller payload: one finite scalar."""

    value: float

    def __post_init__(self) -> None:
        if isinstance(self.value, bool) or not isinstance(self.value, (int, float)):
            raise DataFirewallViolation("selection score must be numeric")
        if not math.isfinite(self.value):
            raise DataFirewallViolation("selection score must be finite")

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "SelectionScore":
        unexpected = set(payload) - {"score"}
        if unexpected:
            names = ", ".join(sorted(unexpected))
            raise DataFirewallViolation(
                f"selection payload contains forbidden fields: {names}"
            )
        if "score" not in payload:
            raise DataFirewallViolation("selection payload is missing scalar field: score")
        return cls(value=payload["score"])

    def to_payload(self) -> dict[str, float]:
        return {"score": float(self.value)}


@dataclass(frozen=True)
class SelectionDecision:
    current: SelectionScore
    candidate: SelectionScore
    accepted: bool

    @property
    def delta(self) -> float:
        return float(self.candidate.value - self.current.value)


def strict_selection_decision(
    *, current: SelectionScore, candidate: SelectionScore
) -> SelectionDecision:
    """Apply the paper comparator; equality is always a rejection."""

    return SelectionDecision(
        current=current,
        candidate=candidate,
        accepted=candidate.value > current.value,
    )
