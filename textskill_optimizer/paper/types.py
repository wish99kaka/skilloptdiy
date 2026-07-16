"""Paper-only event and state types; no extension optimizer types are reused."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from enum import Enum
from importlib.resources import files
from typing import Any, Mapping

from .data import SelectionScore
from .schema_validation import validate_schema


class AlgorithmEventType(str, Enum):
    RUN_STARTED = "run_started"
    EPOCH_STARTED = "epoch_started"
    STEP_STARTED = "step_started"
    ROLLOUT_COLLECTED = "rollout_collected"
    FAILURE_REFLECTED = "failure_reflected"
    SUCCESS_REFLECTED = "success_reflected"
    ANALYST_REFINED = "analyst_refined"
    MERGE_FAILURE = "merge_failure"
    MERGE_SUCCESS = "merge_success"
    MERGE_FINAL_FAILURE_PRIORITIZED = "merge_final_failure_prioritized"
    RANK_TOP_L = "rank_top_l"
    PATCH_APPLIED = "patch_applied"
    SELECTION_SCORED = "selection_scored"
    CANDIDATE_ACCEPTED = "candidate_accepted"
    CANDIDATE_REJECTED = "candidate_rejected"
    SLOW_UPDATE_SKIPPED = "slow_update_skipped"
    SLOW_UPDATE_PROPOSED = "slow_update_proposed"
    META_UPDATE_SKIPPED = "meta_update_skipped"
    META_UPDATE_COMPLETED = "meta_update_completed"
    EPOCH_COMPLETED = "epoch_completed"
    RUN_COMPLETED = "run_completed"


class PaperEditOperation(str, Enum):
    APPEND = "append"
    INSERT_AFTER = "insert_after"
    REPLACE = "replace"
    DELETE = "delete"


class PaperEditSource(str, Enum):
    FAILURE = "failure"
    SUCCESS = "success"


@dataclass(frozen=True)
class ObservedFailurePattern:
    failure_type: str
    count: int
    description: str

    def __post_init__(self) -> None:
        if any(
            type(value) is not str or not value.strip()
            for value in (self.failure_type, self.description)
        ):
            raise ValueError("observed failure pattern requires non-empty text")
        if type(self.count) is not int or self.count < 1:
            raise ValueError("observed failure pattern count must be positive")

    def to_dict(self) -> dict[str, Any]:
        return {
            "failure_type": self.failure_type,
            "count": self.count,
            "description": self.description,
        }


@dataclass(frozen=True)
class PaperEdit:
    edit_id: str
    operation: PaperEditOperation
    target: str = ""
    content: str = ""
    rationale: str = ""
    support_count: int = 1
    source_type: PaperEditSource | None = None

    def __post_init__(self) -> None:
        if type(self.edit_id) is not str or not self.edit_id.strip():
            raise ValueError("paper edit requires an edit_id")
        if type(self.operation) is not PaperEditOperation:
            raise ValueError("paper edit operation must be exact PaperEditOperation")
        if any(
            type(value) is not str
            for value in (self.target, self.content, self.rationale)
        ):
            raise ValueError("paper edit text fields must be exact strings")
        if self.operation in {
            PaperEditOperation.INSERT_AFTER,
            PaperEditOperation.REPLACE,
            PaperEditOperation.DELETE,
        } and not self.target:
            raise ValueError(f"{self.operation.value} requires a target")
        if self.operation in {
            PaperEditOperation.APPEND,
            PaperEditOperation.INSERT_AFTER,
            PaperEditOperation.REPLACE,
        } and not self.content.strip():
            raise ValueError(f"{self.operation.value} requires content")
        if self.operation is PaperEditOperation.DELETE and self.content:
            raise ValueError("delete cannot carry replacement content")
        if type(self.support_count) is not int or self.support_count < 1:
            raise ValueError("paper edit support_count must be >= 1")
        if (
            self.source_type is not None
            and type(self.source_type) is not PaperEditSource
        ):
            raise ValueError("paper edit source_type must be failure or success")


@dataclass(frozen=True)
class EpochBufferRecord:
    """Train-derived step evidence plus rejected scalar-gate feedback."""

    epoch: int
    step: int
    failure_patterns: tuple[ObservedFailurePattern, ...]
    rejected_edits: tuple[PaperEdit, ...]
    score_delta: float | None

    def __post_init__(self) -> None:
        if type(self.epoch) is not int or self.epoch < 1:
            raise ValueError("epoch buffer record requires positive epoch")
        if type(self.step) is not int or self.step < 1:
            raise ValueError("epoch buffer record requires positive step")
        if type(self.failure_patterns) is not tuple or any(
            type(item) is not ObservedFailurePattern
            for item in self.failure_patterns
        ):
            raise ValueError("epoch buffer requires exact failure patterns")
        if type(self.rejected_edits) is not tuple or any(
            type(item) is not PaperEdit for item in self.rejected_edits
        ):
            raise ValueError("epoch buffer requires exact rejected edits")
        if self.rejected_edits:
            if (
                type(self.score_delta) is not float
                or not math.isfinite(self.score_delta)
                or self.score_delta > 0
            ):
                raise ValueError("rejected edits require a finite non-positive delta")
        elif self.score_delta is not None:
            if type(self.score_delta) is not float or not math.isfinite(
                self.score_delta
            ):
                raise ValueError("epoch buffer delta must be finite")

    def to_optimizer_payload(self) -> dict[str, Any]:
        return {
            "epoch": self.epoch,
            "step": self.step,
            "failure_patterns": [
                pattern.to_dict() for pattern in self.failure_patterns
            ],
            "rejected_edits": [
                {
                    "edit_id": edit.edit_id,
                    "op": edit.operation.value,
                    "target": edit.target,
                    "content": edit.content,
                    "source_type": (
                        edit.source_type.value
                        if edit.source_type is not None
                        else None
                    ),
                }
                for edit in self.rejected_edits
            ],
            "score_delta": self.score_delta,
        }

    def to_checkpoint_dict(self) -> dict[str, Any]:
        return {
            "epoch": self.epoch,
            "step": self.step,
            "failure_patterns": [
                pattern.to_dict() for pattern in self.failure_patterns
            ],
            "rejected_edits": [
                {
                    "edit_id": edit.edit_id,
                    "operation": edit.operation.value,
                    "target": edit.target,
                    "content": edit.content,
                    "rationale": edit.rationale,
                    "support_count": edit.support_count,
                    "source_type": (
                        edit.source_type.value
                        if edit.source_type is not None
                        else None
                    ),
                }
                for edit in self.rejected_edits
            ],
            "score_delta": self.score_delta,
        }

    @classmethod
    def from_checkpoint_mapping(
        cls,
        payload: Mapping[str, Any],
    ) -> "EpochBufferRecord":
        expected = {
            "epoch",
            "step",
            "failure_patterns",
            "rejected_edits",
            "score_delta",
        }
        if type(payload) is not dict or set(payload) != expected:
            raise ValueError("invalid epoch buffer checkpoint record")
        if type(payload["failure_patterns"]) is not list or type(
            payload["rejected_edits"]
        ) is not list:
            raise ValueError("invalid epoch buffer checkpoint lists")
        patterns: list[ObservedFailurePattern] = []
        for item in payload["failure_patterns"]:
            if type(item) is not dict or set(item) != {
                "failure_type",
                "count",
                "description",
            }:
                raise ValueError("invalid checkpoint failure pattern")
            patterns.append(ObservedFailurePattern(**item))
        edits: list[PaperEdit] = []
        for item in payload["rejected_edits"]:
            if type(item) is not dict or set(item) != {
                "edit_id",
                "operation",
                "target",
                "content",
                "rationale",
                "support_count",
                "source_type",
            }:
                raise ValueError("invalid checkpoint rejected edit")
            edits.append(
                PaperEdit(
                    edit_id=item["edit_id"],
                    operation=PaperEditOperation(item["operation"]),
                    target=item["target"],
                    content=item["content"],
                    rationale=item["rationale"],
                    support_count=item["support_count"],
                    source_type=(
                        PaperEditSource(item["source_type"])
                        if item["source_type"] is not None
                        else None
                    ),
                )
            )
        score_delta = payload["score_delta"]
        if score_delta is not None and type(score_delta) in {int, float}:
            score_delta = float(score_delta)
        return cls(
            epoch=payload["epoch"],
            step=payload["step"],
            failure_patterns=tuple(patterns),
            rejected_edits=tuple(edits),
            score_delta=score_delta,
        )


@dataclass(frozen=True)
class PaperState:
    """Persisted paper state with explicit current, best, and optimizer-only text."""

    epoch: int
    step: int
    current_skill: str
    current_score: SelectionScore
    best_skill: str
    best_score: SelectionScore
    meta_skill: str = ""

    def __post_init__(self) -> None:
        if self.epoch < 0 or self.step < 0:
            raise ValueError("paper state epoch and step must be non-negative")
        if not self.current_skill.strip() or not self.best_skill.strip():
            raise ValueError("paper state requires current and best skills")
        if self.best_score.value < self.current_score.value:
            raise ValueError("best score cannot be below current score")


@dataclass(frozen=True)
class AlgorithmEvent:
    sequence: int
    event_type: AlgorithmEventType
    epoch: int | None
    step: int | None
    payload: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = "algorithm-event-v1"

    def __post_init__(self) -> None:
        if self.sequence < 0:
            raise ValueError("event sequence must be non-negative")
        if self.epoch is not None and self.epoch < 1:
            raise ValueError("event epoch must be >= 1")
        if self.step is not None and self.step < 1:
            raise ValueError("event step must be >= 1")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "sequence": self.sequence,
            "event_type": self.event_type.value,
            "epoch": self.epoch,
            "step": self.step,
            "payload": dict(self.payload),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "AlgorithmEvent":
        schema_path = files("textskill_optimizer.paper").joinpath(
            "schemas", "algorithm-event-v1.schema.json"
        )
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        violations = validate_schema(payload, schema)
        if violations:
            details = "; ".join(f"{item.path}: {item.message}" for item in violations)
            raise ValueError(f"invalid Algorithm 1 event: {details}")
        return cls(
            schema_version=payload["schema_version"],
            sequence=payload["sequence"],
            event_type=AlgorithmEventType(payload["event_type"]),
            epoch=payload["epoch"],
            step=payload["step"],
            payload=payload["payload"],
        )
