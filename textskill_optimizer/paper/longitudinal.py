"""Authenticated adjacent-epoch comparison over one deterministic train sample."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .data import TrainController, TrainEvidenceBatch
from .provenance import canonical_json_sha256


@dataclass(frozen=True)
class LongitudinalEvidence:
    previous: TrainEvidenceBatch
    current: TrainEvidenceBatch

    def __post_init__(self) -> None:
        if type(self.previous) is not TrainEvidenceBatch or type(
            self.current
        ) is not TrainEvidenceBatch:
            raise ValueError("longitudinal evidence requires exact train evidence")
        self.previous.__post_init__()
        self.current.__post_init__()


@dataclass(frozen=True)
class LongitudinalTaskComparison:
    task_id: str
    task_input: Any
    previous_output: Any
    previous_score: float
    previous_success: bool
    previous_trace: tuple[str, ...]
    current_output: Any
    current_score: float
    current_success: bool
    current_trace: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "task_input": self.task_input,
            "previous": {
                "output": self.previous_output,
                "score": self.previous_score,
                "success": self.previous_success,
                "trace": list(self.previous_trace),
            },
            "current": {
                "output": self.current_output,
                "score": self.current_score,
                "success": self.current_success,
                "trace": list(self.current_trace),
            },
        }


@dataclass(frozen=True)
class LongitudinalState:
    improvements: tuple[LongitudinalTaskComparison, ...]
    regressions: tuple[LongitudinalTaskComparison, ...]
    persistent_failures: tuple[LongitudinalTaskComparison, ...]
    stable_successes: tuple[LongitudinalTaskComparison, ...]

    def to_prompt_payload(self) -> dict[str, list[dict[str, Any]]]:
        return {
            "improvements": [item.to_dict() for item in self.improvements],
            "regressions": [item.to_dict() for item in self.regressions],
            "persistent_failures": [
                item.to_dict() for item in self.persistent_failures
            ],
            "stable_successes": [
                item.to_dict() for item in self.stable_successes
            ],
        }


def build_longitudinal_state(
    *,
    train: TrainController,
    evidence: LongitudinalEvidence,
    previous_skill: str,
    current_skill: str,
    sample_size: int,
    split_seed: int,
    epoch: int,
    batch_id: str,
    batch_seed: int,
) -> LongitudinalState:
    """Verify both signed batches, select identical tasks, and classify outcomes."""

    if type(train) is not TrainController:
        raise ValueError("longitudinal comparison requires exact TrainController")
    if type(evidence) is not LongitudinalEvidence:
        raise ValueError("longitudinal comparison requires exact evidence")
    evidence.__post_init__()
    if type(sample_size) is not int or sample_size < 1:
        raise ValueError("longitudinal sample size must be positive")
    if type(split_seed) is not int or type(epoch) is not int or epoch < 2:
        raise ValueError("longitudinal comparison requires deterministic epoch seed")
    if type(batch_id) is not str or not batch_id.strip():
        raise ValueError("longitudinal comparison requires scheduled batch_id")
    previous = train.verify(
        evidence.previous,
        current_skill=previous_skill,
        batch_id=batch_id,
        batch_seed=batch_seed,
        batch_size=sample_size,
    )
    current = train.verify(
        evidence.current,
        current_skill=current_skill,
        batch_id=batch_id,
        batch_seed=batch_seed,
        batch_size=sample_size,
    )
    previous_by_id = _index_tasks(previous)
    current_by_id = _index_tasks(current)
    if set(previous_by_id) != set(current_by_id):
        raise ValueError("longitudinal evidence must contain the same task IDs")
    if len(previous_by_id) < sample_size:
        raise ValueError("longitudinal evidence has fewer tasks than frozen sample size")
    selected_ids = sorted(
        previous_by_id,
        key=lambda task_id: canonical_json_sha256(
            {"split_seed": split_seed, "epoch": epoch, "task_id": task_id}
        ),
    )[:sample_size]
    buckets: dict[str, list[LongitudinalTaskComparison]] = {
        "improvements": [],
        "regressions": [],
        "persistent_failures": [],
        "stable_successes": [],
    }
    for task_id in selected_ids:
        before = previous_by_id[task_id]
        after = current_by_id[task_id]
        if before["task_input"] != after["task_input"]:
            raise ValueError("longitudinal task input changed between skill versions")
        comparison = LongitudinalTaskComparison(
            task_id=task_id,
            task_input=before["task_input"],
            previous_output=before["output"],
            previous_score=float(before["score"]),
            previous_success=before["success"],
            previous_trace=tuple(before["trace"]),
            current_output=after["output"],
            current_score=float(after["score"]),
            current_success=after["success"],
            current_trace=tuple(after["trace"]),
        )
        if not before["success"] and after["success"]:
            bucket = "improvements"
        elif before["success"] and not after["success"]:
            bucket = "regressions"
        elif not before["success"] and not after["success"]:
            bucket = "persistent_failures"
        else:
            bucket = "stable_successes"
        buckets[bucket].append(comparison)
    return LongitudinalState(
        improvements=tuple(buckets["improvements"]),
        regressions=tuple(buckets["regressions"]),
        persistent_failures=tuple(buckets["persistent_failures"]),
        stable_successes=tuple(buckets["stable_successes"]),
    )


def _index_tasks(
    trajectories: tuple[dict[str, Any], ...],
) -> dict[str, dict[str, Any]]:
    indexed = {item["task_id"]: item for item in trajectories}
    if len(indexed) != len(trajectories):
        raise ValueError("longitudinal evidence task IDs must be unique")
    return indexed
