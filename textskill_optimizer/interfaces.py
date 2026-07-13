"""Plugin interfaces for runners, scorers, and skill editors."""

from __future__ import annotations

from typing import Any, Protocol

from .models import EditProposal, Score, Task, TaskOutput, TaskResult


class SkillRunner(Protocol):
    """Executes one task using the current skill document."""

    def run(self, skill_text: str, task: Task) -> TaskOutput:
        """Return the task output and trace."""


class SkillScorer(Protocol):
    """Scores one runner output against the task target."""

    def score(self, task: Task, output: TaskOutput) -> Score:
        """Return a normalized score in [0, 1]."""


class SkillEditor(Protocol):
    """Turns scored trajectories into bounded skill-document edits."""

    def propose(
        self,
        skill_text: str,
        train_results: list[TaskResult],
        *,
        epoch: int,
        rejected_buffer: list[dict[str, Any]] | None = None,
        meta_skill: str = "",
        optimizer_controls: dict[str, Any] | None = None,
    ) -> list[EditProposal]:
        """Return candidate replacement skill documents."""
