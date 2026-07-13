"""Plugin interfaces for runners, scorers, and skill editors."""

from __future__ import annotations

from typing import Any, Protocol

from .models import EditProposal, Score, Task, TaskOutput, TaskResult


EDITOR_CAPABILITY_ATOMIC_EDITS = "atomic_edits"
EDITOR_CAPABILITY_FULL_REPLACEMENT = "full_skill_replacement"


def require_editor_capability(editor: object, capability: str, *, protocol: str) -> None:
    """Reject an editor that cannot satisfy a protocol before work begins."""

    capabilities = frozenset(getattr(editor, "capabilities", ()))
    if capability in capabilities:
        return
    declared = ", ".join(sorted(capabilities)) or "none"
    raise ValueError(
        f"{protocol} protocol requires editor capability {capability!r}; "
        f"{type(editor).__name__} declares: {declared}"
    )


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

    capabilities: frozenset[str]

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
        """Return candidate skill-document edits."""
