"""Core data structures for text-skill optimization."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Task:
    """A single optimization or validation example."""

    id: str
    input: str
    expected: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "Task":
        task_id = payload.get("id")
        task_input = payload.get("input")
        if task_id is None:
            raise ValueError("Task is missing required field: id")
        if task_input is None:
            raise ValueError(f"Task {task_id!r} is missing required field: input")
        return cls(
            id=str(task_id),
            input=str(task_input),
            expected=payload.get("expected"),
            metadata=dict(payload.get("metadata") or {}),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "input": self.input,
            "expected": self.expected,
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class TaskOutput:
    """Output produced by a runner for one task."""

    value: Any
    trace: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "value": self.value,
            "trace": list(self.trace),
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class Score:
    """A normalized task score."""

    value: float
    success: bool = False
    message: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.value < 0.0 or self.value > 1.0:
            raise ValueError("Score.value must be in [0.0, 1.0]")

    def to_dict(self) -> dict[str, Any]:
        return {
            "value": self.value,
            "success": self.success,
            "message": self.message,
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class TaskResult:
    """A scored task execution."""

    task: Task
    output: TaskOutput
    score: Score

    def to_dict(self) -> dict[str, Any]:
        return {
            "task": self.task.to_dict(),
            "output": self.output.to_dict(),
            "score": self.score.to_dict(),
        }


@dataclass(frozen=True)
class EvaluationReport:
    """Aggregate scoring over a task set."""

    name: str
    results: list[TaskResult]

    @property
    def average_score(self) -> float:
        if not self.results:
            return 0.0
        return sum(item.score.value for item in self.results) / len(self.results)

    @property
    def pass_rate(self) -> float:
        if not self.results:
            return 0.0
        return sum(1 for item in self.results if item.score.success) / len(self.results)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "average_score": self.average_score,
            "pass_rate": self.pass_rate,
            "results": [item.to_dict() for item in self.results],
        }


@dataclass(frozen=True)
class AtomicEdit:
    """One localized add, delete, or replace operation on a skill document."""

    operation: str
    target: str
    content: str = ""
    rationale: str = ""
    priority: float = 0.0

    def __post_init__(self) -> None:
        if self.operation not in {"add", "delete", "replace"}:
            raise ValueError(f"Unsupported atomic edit operation: {self.operation!r}")
        if self.operation in {"delete", "replace"} and not self.target:
            raise ValueError(f"{self.operation} edit requires a non-empty target")
        if self.operation in {"add", "replace"} and not self.content.strip():
            raise ValueError(f"{self.operation} edit requires non-empty content")

    def to_dict(self) -> dict[str, Any]:
        return {
            "operation": self.operation,
            "target": self.target,
            "content": self.content,
            "rationale": self.rationale,
            "priority": self.priority,
        }


@dataclass(frozen=True)
class EditProposal:
    """A candidate skill update, optionally represented as atomic edits."""

    name: str
    skill_text: str = ""
    rationale: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    edits: tuple[AtomicEdit, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not self.skill_text.strip() and not self.edits:
            raise ValueError("EditProposal requires skill_text or at least one atomic edit")

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "skill_text": self.skill_text,
            "rationale": self.rationale,
            "metadata": self.metadata,
            "edits": [edit.to_dict() for edit in self.edits],
        }


@dataclass(frozen=True)
class OptimizerStateUpdate:
    """Optimizer-only meta guidance plus a candidate slow skill update."""

    meta_skill: str = ""
    slow_update: str = ""
    rationale: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "meta_skill": self.meta_skill,
            "slow_update": self.slow_update,
            "rationale": self.rationale,
        }


@dataclass(frozen=True)
class RejectedProposal:
    """A rejected skill edit retained as optimizer feedback."""

    epoch: int
    candidate: str
    reason: str
    rationale: str
    validation_score: float | None = None
    failed_task_ids: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "epoch": self.epoch,
            "candidate": self.candidate,
            "reason": self.reason,
            "rationale": self.rationale,
            "validation_score": self.validation_score,
            "failed_task_ids": list(self.failed_task_ids),
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class OptimizationHistoryItem:
    """One candidate evaluation during optimization."""

    epoch: int
    candidate: str
    accepted: bool
    validation_score: float | None
    rationale: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "epoch": self.epoch,
            "candidate": self.candidate,
            "accepted": self.accepted,
            "validation_score": self.validation_score,
            "rationale": self.rationale,
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class OptimizationResult:
    """Final optimized skill plus audit history."""

    best_skill_text: str
    best_validation_score: float
    history: list[OptimizationHistoryItem]
    final_validation_report: EvaluationReport
    rejected_buffer: list[RejectedProposal] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "best_skill_text": self.best_skill_text,
            "best_validation_score": self.best_validation_score,
            "history": [item.to_dict() for item in self.history],
            "final_validation_report": self.final_validation_report.to_dict(),
            "rejected_buffer": [item.to_dict() for item in self.rejected_buffer],
        }
