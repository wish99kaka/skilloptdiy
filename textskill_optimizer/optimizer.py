"""Validation-gated text-skill optimization loop."""

from __future__ import annotations

import json
import inspect
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .interfaces import SkillEditor, SkillRunner, SkillScorer
from .io import load_text, write_json, write_text
from .models import (
    EvaluationReport,
    OptimizationHistoryItem,
    OptimizationResult,
    RejectedProposal,
    Task,
    TaskOutput,
    TaskResult,
)


@dataclass(frozen=True)
class OptimizerConfig:
    """Controls the optimization loop."""

    epochs: int = 3
    max_candidates_per_epoch: int = 4
    min_delta: float = 0.0
    max_skill_chars: int | None = 6000
    max_skill_delta_chars: int | None = 1800
    max_added_bullet_lines: int | None = 8
    rejected_buffer_limit: int = 20
    meta_skill_path: str | Path | None = None


class SkillOptimizer:
    """Optimizes a skill document while keeping the target runner frozen."""

    def __init__(
        self,
        runner: SkillRunner,
        scorer: SkillScorer,
        editor: SkillEditor,
        config: OptimizerConfig | None = None,
    ) -> None:
        self.runner = runner
        self.scorer = scorer
        self.editor = editor
        self.config = config or OptimizerConfig()

    def evaluate(
        self,
        skill_text: str,
        tasks: list[Task],
        *,
        name: str = "evaluation",
    ) -> EvaluationReport:
        results: list[TaskResult] = []
        for task in tasks:
            output = self.runner.run(skill_text, task)
            if not isinstance(output, TaskOutput):
                output = TaskOutput(value=output)
            score = self.scorer.score(task, output)
            results.append(TaskResult(task=task, output=output, score=score))
        return EvaluationReport(name=name, results=results)

    def optimize(
        self,
        initial_skill_text: str,
        train_tasks: list[Task],
        validation_tasks: list[Task],
        *,
        run_dir: str | Path | None = None,
    ) -> OptimizationResult:
        if not train_tasks:
            raise ValueError("train_tasks must not be empty")
        if not validation_tasks:
            raise ValueError("validation_tasks must not be empty")

        run_path = Path(run_dir) if run_dir is not None else None
        current_skill = initial_skill_text
        meta_skill = load_meta_skill(self.config.meta_skill_path)
        rejected_buffer: list[RejectedProposal] = []
        initial_report = self.evaluate(
            current_skill,
            validation_tasks,
            name="validation:initial",
        )
        best_score = initial_report.average_score
        history: list[OptimizationHistoryItem] = [
            OptimizationHistoryItem(
                epoch=0,
                candidate="initial",
                accepted=True,
                validation_score=best_score,
                rationale="Initial skill baseline.",
            )
        ]

        if run_path is not None:
            write_text(run_path / "best_skill.md", current_skill)
            write_json(run_path / "validation_initial.json", initial_report.to_dict())
            if meta_skill:
                write_text(run_path / "meta_skill.md", meta_skill)

        for epoch in range(1, self.config.epochs + 1):
            train_report = self.evaluate(
                current_skill,
                train_tasks,
                name=f"train:epoch:{epoch}",
            )
            proposals = propose_with_optional_context(
                self.editor,
                current_skill,
                train_report.results,
                epoch=epoch,
                rejected_buffer=rejected_payload(rejected_buffer, self.config.rejected_buffer_limit),
                meta_skill=meta_skill,
                optimizer_controls=learning_rate_controls(self.config),
            )
            proposals = proposals[: self.config.max_candidates_per_epoch]

            if run_path is not None:
                write_json(run_path / f"train_epoch_{epoch}.json", train_report.to_dict())

            candidate_records: list[dict[str, Any]] = []
            best_epoch_skill = current_skill
            best_epoch_score = best_score
            best_epoch_index: int | None = None

            for index, proposal in enumerate(proposals):
                learning_rate = assess_learning_rate(current_skill, proposal.skill_text, self.config)
                if run_path is not None:
                    safe_name = _safe_filename(proposal.name)
                    write_text(
                        run_path / f"candidate_epoch_{epoch}_{safe_name}.md",
                        proposal.skill_text,
                    )

                if not learning_rate["ok"]:
                    candidate_records.append(
                        {
                            "name": proposal.name,
                            "skill_text": proposal.skill_text,
                            "validation_score": None,
                            "rationale": proposal.rationale,
                            "failed_task_ids": [],
                            "rejection_reason": "learning_rate_exceeded",
                            "metadata": {
                                "learning_rate": learning_rate,
                                "proposal_metadata": proposal.metadata,
                            },
                        }
                    )
                    continue

                validation_report = self.evaluate(
                    proposal.skill_text,
                    validation_tasks,
                    name=f"validation:epoch:{epoch}:{proposal.name}",
                )
                validation_score = validation_report.average_score
                failed_task_ids = [
                    result.task.id
                    for result in validation_report.results
                    if not result.score.success
                ]
                candidate_records.append(
                    {
                        "name": proposal.name,
                        "skill_text": proposal.skill_text,
                        "validation_score": validation_score,
                        "rationale": proposal.rationale,
                        "failed_task_ids": failed_task_ids,
                        "rejection_reason": rejection_reason(validation_score, best_score),
                        "metadata": {
                            "learning_rate": learning_rate,
                            "proposal_metadata": proposal.metadata,
                        },
                    }
                )

                if run_path is not None:
                    safe_name = _safe_filename(proposal.name)
                    write_json(
                        run_path / f"validation_epoch_{epoch}_{safe_name}.json",
                        validation_report.to_dict(),
                    )

                record_index = len(candidate_records) - 1
                if validation_score > best_score + self.config.min_delta and validation_score > best_epoch_score:
                    best_epoch_skill = proposal.skill_text
                    best_epoch_score = validation_score
                    best_epoch_index = record_index

            for index, record in enumerate(candidate_records):
                accepted = index == best_epoch_index
                if not accepted:
                    rejected = RejectedProposal(
                        epoch=epoch,
                        candidate=str(record["name"]),
                        reason=str(record["rejection_reason"]),
                        rationale=str(record["rationale"]),
                        validation_score=record["validation_score"],
                        failed_task_ids=list(record["failed_task_ids"]),
                        metadata=dict(record["metadata"]),
                    )
                    rejected_buffer.append(rejected)
                history.append(
                    OptimizationHistoryItem(
                        epoch=epoch,
                        candidate=str(record["name"]),
                        accepted=accepted,
                        validation_score=record["validation_score"],
                        rationale=str(record["rationale"]),
                        metadata={
                            "rejection_reason": None if accepted else record["rejection_reason"],
                            **dict(record["metadata"]),
                        },
                    )
                )

            if best_epoch_index is not None:
                current_skill = best_epoch_skill
                best_score = best_epoch_score
                if run_path is not None:
                    write_text(run_path / "best_skill.md", current_skill)
            if run_path is not None:
                write_rejected_buffer_jsonl(run_path / "rejected_buffer.jsonl", rejected_buffer)

        final_report = self.evaluate(
            current_skill,
            validation_tasks,
            name="validation:final",
        )
        result = OptimizationResult(
            best_skill_text=current_skill,
            best_validation_score=best_score,
            history=history,
            final_validation_report=final_report,
            rejected_buffer=rejected_buffer,
        )
        if run_path is not None:
            write_json(run_path / "history.json", result.to_dict())
            write_rejected_buffer_jsonl(run_path / "rejected_buffer.jsonl", rejected_buffer)
        return result


def _safe_filename(value: str) -> str:
    cleaned = [ch if ch.isalnum() or ch in ("-", "_") else "-" for ch in value]
    return "".join(cleaned).strip("-") or "candidate"


def load_meta_skill(path: str | Path | None) -> str:
    if path is None:
        return ""
    meta_path = Path(path)
    if not meta_path.exists():
        raise ValueError(f"meta_skill_path does not exist: {meta_path}")
    return load_text(meta_path)


def learning_rate_controls(config: OptimizerConfig) -> dict[str, int | None]:
    return {
        "max_skill_chars": config.max_skill_chars,
        "max_skill_delta_chars": config.max_skill_delta_chars,
        "max_added_bullet_lines": config.max_added_bullet_lines,
    }


def assess_learning_rate(
    current_skill: str,
    candidate_skill: str,
    config: OptimizerConfig,
) -> dict[str, Any]:
    added_bullets = added_bullet_lines(current_skill, candidate_skill)
    metrics = {
        "skill_chars": len(candidate_skill),
        "skill_delta_chars": abs(len(candidate_skill) - len(current_skill)),
        "added_bullet_lines": len(added_bullets),
    }
    violations = []
    if config.max_skill_chars is not None and metrics["skill_chars"] > config.max_skill_chars:
        violations.append(
            f"skill_chars {metrics['skill_chars']} exceeds {config.max_skill_chars}"
        )
    if (
        config.max_skill_delta_chars is not None
        and metrics["skill_delta_chars"] > config.max_skill_delta_chars
    ):
        violations.append(
            "skill_delta_chars "
            f"{metrics['skill_delta_chars']} exceeds {config.max_skill_delta_chars}"
        )
    if (
        config.max_added_bullet_lines is not None
        and metrics["added_bullet_lines"] > config.max_added_bullet_lines
    ):
        violations.append(
            "added_bullet_lines "
            f"{metrics['added_bullet_lines']} exceeds {config.max_added_bullet_lines}"
        )
    return {
        "ok": not violations,
        "metrics": metrics,
        "violations": violations,
        "added_bullet_lines": added_bullets,
        "controls": learning_rate_controls(config),
    }


def added_bullet_lines(current_skill: str, candidate_skill: str) -> list[str]:
    current = {normalize_bullet_line(line) for line in current_skill.splitlines()}
    current.discard("")
    added = []
    for line in candidate_skill.splitlines():
        normalized = normalize_bullet_line(line)
        if normalized and normalized not in current:
            added.append(line.strip())
    return added


def normalize_bullet_line(line: str) -> str:
    stripped = line.strip()
    if not stripped.startswith(("-", "*")):
        return ""
    return " ".join(stripped.split())


def rejection_reason(validation_score: float, best_score: float) -> str:
    if validation_score <= best_score:
        return "validation_not_improved"
    return "lower_than_best_candidate"


def rejected_payload(
    rejected_buffer: list[RejectedProposal],
    limit: int,
) -> list[dict[str, Any]]:
    if limit <= 0:
        return []
    return [item.to_dict() for item in rejected_buffer[-limit:]]


def write_rejected_buffer_jsonl(
    path: str | Path,
    rejected_buffer: list[RejectedProposal],
) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(item.to_dict(), sort_keys=True) for item in rejected_buffer]
    target.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def propose_with_optional_context(
    editor: SkillEditor,
    skill_text: str,
    train_results: list[TaskResult],
    *,
    epoch: int,
    rejected_buffer: list[dict[str, Any]],
    meta_skill: str,
    optimizer_controls: dict[str, Any],
):
    kwargs: dict[str, Any] = {"epoch": epoch}
    try:
        signature = inspect.signature(editor.propose)
    except (TypeError, ValueError):
        kwargs.update(
            {
                "rejected_buffer": rejected_buffer,
                "meta_skill": meta_skill,
                "optimizer_controls": optimizer_controls,
            }
        )
        return editor.propose(skill_text, train_results, **kwargs)

    parameters = signature.parameters
    accepts_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    )
    optional_context = {
        "rejected_buffer": rejected_buffer,
        "meta_skill": meta_skill,
        "optimizer_controls": optimizer_controls,
    }
    for name, value in optional_context.items():
        if accepts_kwargs or name in parameters:
            kwargs[name] = value
    return editor.propose(skill_text, train_results, **kwargs)
