"""Paper-style text-space optimizer with bounded atomic edits and slow state."""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from .contract_evidence import contract_delta_evidence
from .edits import apply_atomic_edits, canonical_edit_key, merge_and_rank_atomic_edits, set_slow_update
from .interfaces import EDITOR_CAPABILITY_ATOMIC_EDITS, require_editor_capability
from .io import load_text, write_json, write_text
from .models import (
    AtomicEdit,
    EditProposal,
    EvaluationReport,
    OptimizationHistoryItem,
    OptimizerStateUpdate,
    RejectedProposal,
    Task,
    TaskOutput,
    TaskResult,
)
from .optimizer import propose_with_optional_context, rejected_payload, write_rejected_buffer_jsonl


@dataclass(frozen=True)
class ExecutiveOptimizerConfig:
    epochs: int = 4
    rollout_batch_size: int = 40
    reflection_minibatch_size: int = 8
    learning_rate: int = 4
    learning_rate_floor: int = 2
    learning_rate_schedule: str = "cosine"
    min_delta: float = 0.0
    max_skill_chars: int = 6000
    rejected_buffer_limit: int = 20
    slow_update_sample_size: int = 20
    enable_slow_update: bool = True
    seed: int = 42
    meta_skill_path: str | Path | None = None
    task_retry_limit: int = 0
    task_retry_backoff_seconds: float = 0.0
    fail_on_persistent_task_anomaly: bool = True
    validation_confirmation_rounds: int = 0
    validation_required_wins: int = 1
    validation_mean_delta: float = 0.0
    early_stop_rejection_limit: int = 0
    early_stop_validation_score: float | None = None

    def __post_init__(self) -> None:
        if self.epochs <= 0:
            raise ValueError("epochs must be positive")
        if self.rollout_batch_size <= 0:
            raise ValueError("rollout_batch_size must be positive")
        if self.reflection_minibatch_size <= 0:
            raise ValueError("reflection_minibatch_size must be positive")
        if self.learning_rate <= 0 or self.learning_rate_floor <= 0:
            raise ValueError("learning rates must be positive")
        if self.learning_rate_floor > self.learning_rate:
            raise ValueError("learning_rate_floor cannot exceed learning_rate")
        if self.learning_rate_schedule not in {"constant", "linear", "cosine"}:
            raise ValueError("learning_rate_schedule must be constant, linear, or cosine")
        if self.task_retry_limit < 0:
            raise ValueError("task_retry_limit must be non-negative")
        if self.task_retry_backoff_seconds < 0:
            raise ValueError("task_retry_backoff_seconds must be non-negative")
        if self.validation_confirmation_rounds < 0:
            raise ValueError("validation_confirmation_rounds must be non-negative")
        if not 1 <= self.validation_required_wins <= self.validation_confirmation_rounds + 1:
            raise ValueError(
                "validation_required_wins must be between 1 and "
                "validation_confirmation_rounds + 1"
            )
        if self.validation_mean_delta < 0:
            raise ValueError("validation_mean_delta must be non-negative")
        if self.early_stop_rejection_limit < 0:
            raise ValueError("early_stop_rejection_limit must be non-negative")
        if self.early_stop_validation_score is not None and self.early_stop_validation_score < 0:
            raise ValueError("early_stop_validation_score must be non-negative")


class PersistentTaskAnomaly(RuntimeError):
    """Raised when an evaluation task remains unhealthy after targeted retries."""


@dataclass(frozen=True)
class ValidationGateDecision:
    accepted: bool
    current_mean: float
    candidate_mean: float
    wins: int
    total_rounds: int
    round_scores: tuple[dict[str, Any], ...]
    candidate_report: EvaluationReport
    contract_evidence: dict[str, Any] = field(default_factory=dict)
    contract_policy_guard: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "accepted": self.accepted,
            "current_mean": self.current_mean,
            "candidate_mean": self.candidate_mean,
            "mean_delta": self.candidate_mean - self.current_mean,
            "wins": self.wins,
            "total_rounds": self.total_rounds,
            "round_scores": list(self.round_scores),
            "contract_evidence": self.contract_evidence,
            "contract_policy_guard": self.contract_policy_guard,
        }


@dataclass(frozen=True)
class ExecutiveOptimizationResult:
    best_skill_text: str
    best_validation_score: float
    history: list[OptimizationHistoryItem]
    final_validation_report: EvaluationReport
    rejected_buffer: list[RejectedProposal] = field(default_factory=list)
    meta_skill_text: str = ""
    accepted_steps: int = 0
    total_steps: int = 0
    stop_reason: str = "completed"
    checkpoint: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "best_skill_text": self.best_skill_text,
            "best_validation_score": self.best_validation_score,
            "history": [item.to_dict() for item in self.history],
            "final_validation_report": self.final_validation_report.to_dict(),
            "rejected_buffer": [item.to_dict() for item in self.rejected_buffer],
            "meta_skill_text": self.meta_skill_text,
            "accepted_steps": self.accepted_steps,
            "total_steps": self.total_steps,
            "stop_reason": self.stop_reason,
            "checkpoint": self.checkpoint,
        }


class ExecutiveSkillOptimizer:
    """Optimize one skill using trajectory batches and validation-gated atomic updates."""

    def __init__(
        self,
        runner: Any,
        scorer: Any,
        editor: Any,
        config: ExecutiveOptimizerConfig | None = None,
        *,
        retry_detector: Callable[[TaskResult], list[str]] | None = None,
    ) -> None:
        self.runner = runner
        self.scorer = scorer
        self.editor = editor
        self.config = config or ExecutiveOptimizerConfig()
        self.retry_detector = retry_detector

    def evaluate(
        self,
        skill_text: str,
        tasks: list[Task],
        *,
        name: str = "evaluation",
        timing_path: Path | None = None,
    ) -> EvaluationReport:
        evaluation_started = time.monotonic()
        write_timing_event(
            timing_path,
            "evaluation_started",
            evaluation_name=name,
            task_count=len(tasks),
        )
        results: list[TaskResult] = []
        for task in tasks:
            attempts: list[dict[str, Any]] = []
            result: TaskResult | None = None
            reasons: list[str] = []
            for attempt in range(self.config.task_retry_limit + 1):
                attempt_started = time.monotonic()
                write_timing_event(
                    timing_path,
                    "task_started",
                    evaluation_name=name,
                    task_id=task.id,
                    attempt=attempt,
                )
                scope = getattr(self.runner, "usage_scope", None)
                context = (
                    scope(evaluation_name=name, attempt=attempt)
                    if callable(scope)
                    else nullcontext()
                )
                try:
                    with context:
                        output = self.runner.run(skill_text, task)
                except Exception as exc:
                    write_timing_event(
                        timing_path,
                        "task_exception",
                        evaluation_name=name,
                        task_id=task.id,
                        attempt=attempt,
                        duration_seconds=time.monotonic() - attempt_started,
                        error_type=type(exc).__name__,
                        error=str(exc),
                    )
                    raise
                if not isinstance(output, TaskOutput):
                    output = TaskOutput(value=output)
                result = TaskResult(task=task, output=output, score=self.scorer.score(task, output))
                reasons = self.retry_detector(result) if self.retry_detector is not None else []
                output_metadata = output.metadata if isinstance(output.metadata, dict) else {}
                agent_metadata = output_metadata.get("agent") if isinstance(output_metadata.get("agent"), dict) else {}
                pre_test_metadata = (
                    output_metadata.get("pre_test") if isinstance(output_metadata.get("pre_test"), dict) else {}
                )
                post_test_metadata = (
                    output_metadata.get("post_test") if isinstance(output_metadata.get("post_test"), dict) else {}
                )
                write_timing_event(
                    timing_path,
                    "task_finished",
                    evaluation_name=name,
                    task_id=task.id,
                    attempt=attempt,
                    duration_seconds=time.monotonic() - attempt_started,
                    success=result.score.success,
                    score=result.score.value,
                    retryable=bool(reasons),
                    retry_reasons=list(reasons),
                    agent_duration_seconds=agent_metadata.get("duration_seconds"),
                    agent_returncode=agent_metadata.get("returncode"),
                    agent_timed_out=agent_metadata.get("timed_out"),
                    pre_test_duration_seconds=pre_test_metadata.get("duration_seconds"),
                    pre_test_returncode=pre_test_metadata.get("returncode"),
                    post_test_duration_seconds=post_test_metadata.get("duration_seconds"),
                    post_test_returncode=post_test_metadata.get("returncode"),
                )
                attempts.append(
                    {
                        "attempt": attempt,
                        "success": result.score.success,
                        "score": result.score.value,
                        "retryable": bool(reasons),
                        "reasons": list(reasons),
                    }
                )
                if not reasons:
                    break
                if attempt < self.config.task_retry_limit and self.config.task_retry_backoff_seconds:
                    time.sleep(self.config.task_retry_backoff_seconds)
            if result is None:
                raise RuntimeError(f"No evaluation result produced for task {task.id!r}")
            if self.retry_detector is not None:
                result.output.metadata["retry_policy"] = {
                    "max_retries": self.config.task_retry_limit,
                    "attempt_count": len(attempts),
                    "attempts": attempts,
                    "selected_attempt": len(attempts) - 1,
                    "persistent_anomaly": bool(reasons),
                }
            if reasons and self.config.fail_on_persistent_task_anomaly:
                joined = ",".join(reasons)
                write_timing_event(
                    timing_path,
                    "evaluation_finished",
                    evaluation_name=name,
                    task_count=len(tasks),
                    result_count=len(results),
                    duration_seconds=time.monotonic() - evaluation_started,
                    status="persistent_task_anomaly",
                )
                raise PersistentTaskAnomaly(
                    f"Task {task.id!r} remained unhealthy after {len(attempts)} attempts: {joined}"
                )
            results.append(result)
        write_timing_event(
            timing_path,
            "evaluation_finished",
            evaluation_name=name,
            task_count=len(tasks),
            result_count=len(results),
            duration_seconds=time.monotonic() - evaluation_started,
            status="complete",
        )
        return EvaluationReport(name=name, results=results)

    def validate_candidate(
        self,
        current_skill: str,
        candidate_skill: str,
        validation_tasks: list[Task],
        *,
        candidate_name: str,
        current_score: float | None = None,
        current_report: EvaluationReport | None = None,
        run_path: Path | None = None,
        candidate_contract_policy: dict[str, Any] | None = None,
    ) -> ValidationGateDecision:
        """Run an adaptive paired gate after a candidate first clears the selection baseline."""
        if current_score is None:
            current_score = current_report.average_score if current_report is not None else 0.0
        timing_path = run_path / "timing_events.jsonl" if run_path is not None else None
        validation_started = time.monotonic()
        write_timing_event(
            timing_path,
            "validation_started",
            candidate_name=candidate_name,
            task_count=len(validation_tasks),
            current_score=current_score,
        )
        current_reports: list[EvaluationReport] = [current_report] if current_report is not None else []
        candidate_reports = [
            self.evaluate(
                candidate_skill,
                validation_tasks,
                name=f"selection:{candidate_name}",
                timing_path=timing_path,
            )
        ]
        if run_path is not None:
            write_json(
                run_path / f"selection_{candidate_name}.json",
                candidate_reports[0].to_dict(),
            )
            write_timing_event(
                timing_path,
                "validation_report_written",
                candidate_name=candidate_name,
                report=f"selection_{candidate_name}.json",
                kind="candidate_initial",
            )

        initial_candidate_score = candidate_reports[0].average_score
        current_scores = [current_score]
        candidate_scores = [initial_candidate_score]
        round_scores: list[dict[str, Any]] = [
            {
                "round": 1,
                "kind": "initial",
                "execution_order": ["candidate"],
                "current_score": current_score,
                "candidate_score": initial_candidate_score,
                "candidate_won": initial_candidate_score > current_score + self.config.min_delta,
            }
        ]
        initial_improved = bool(round_scores[0]["candidate_won"])

        if initial_improved:
            for confirmation_index in range(1, self.config.validation_confirmation_rounds + 1):
                current_name = f"selection:{candidate_name}:confirm:{confirmation_index}:current"
                candidate_report_name = (
                    f"selection:{candidate_name}:confirm:{confirmation_index}:candidate"
                )
                if confirmation_index % 2:
                    execution_order = ["current", "candidate"]
                    current_report = self.evaluate(
                        current_skill,
                        validation_tasks,
                        name=current_name,
                        timing_path=timing_path,
                    )
                    candidate_report = self.evaluate(
                        candidate_skill,
                        validation_tasks,
                        name=candidate_report_name,
                        timing_path=timing_path,
                    )
                else:
                    execution_order = ["candidate", "current"]
                    candidate_report = self.evaluate(
                        candidate_skill,
                        validation_tasks,
                        name=candidate_report_name,
                        timing_path=timing_path,
                    )
                    current_report = self.evaluate(
                        current_skill,
                        validation_tasks,
                        name=current_name,
                        timing_path=timing_path,
                    )

                current_scores.append(current_report.average_score)
                candidate_scores.append(candidate_report.average_score)
                current_reports.append(current_report)
                candidate_reports.append(candidate_report)
                round_scores.append(
                    {
                        "round": confirmation_index + 1,
                        "kind": "confirmation",
                        "execution_order": execution_order,
                        "current_score": current_report.average_score,
                        "candidate_score": candidate_report.average_score,
                        "candidate_won": (
                            candidate_report.average_score
                            > current_report.average_score + self.config.min_delta
                        ),
                    }
                )
                if run_path is not None:
                    write_json(
                        run_path
                        / f"selection_{candidate_name}_confirm_{confirmation_index}_current.json",
                        current_report.to_dict(),
                    )
                    write_timing_event(
                        timing_path,
                        "validation_report_written",
                        candidate_name=candidate_name,
                        report=f"selection_{candidate_name}_confirm_{confirmation_index}_current.json",
                        kind="current_confirmation",
                        confirmation_index=confirmation_index,
                    )
                    write_json(
                        run_path
                        / f"selection_{candidate_name}_confirm_{confirmation_index}_candidate.json",
                        candidate_report.to_dict(),
                    )
                    write_timing_event(
                        timing_path,
                        "validation_report_written",
                        candidate_name=candidate_name,
                        report=f"selection_{candidate_name}_confirm_{confirmation_index}_candidate.json",
                        kind="candidate_confirmation",
                        confirmation_index=confirmation_index,
                    )

        candidate_report = EvaluationReport(
            name=f"selection:{candidate_name}:gate-evidence",
            results=[result for report in candidate_reports for result in report.results],
        )
        current_evidence_report = (
            EvaluationReport(
                name=f"selection:{candidate_name}:current-gate-evidence",
                results=[result for report in current_reports for result in report.results],
            )
            if current_reports
            else None
        )
        contract_evidence = contract_delta_evidence(current_evidence_report, candidate_report)
        current_mean = sum(current_scores) / len(current_scores)
        candidate_mean = sum(candidate_scores) / len(candidate_scores)
        wins = sum(bool(item["candidate_won"]) for item in round_scores)
        if self.config.validation_confirmation_rounds == 0:
            accepted = initial_improved
        else:
            accepted = (
                initial_improved
                and wins >= self.config.validation_required_wins
                and candidate_mean - current_mean >= self.config.validation_mean_delta
            )
        contract_policy_guard = evaluate_contract_policy_guard(
            contract_evidence,
            candidate_contract_policy,
        )
        accepted = accepted and bool(contract_policy_guard["passes"])
        decision = ValidationGateDecision(
            accepted=accepted,
            current_mean=current_mean,
            candidate_mean=candidate_mean,
            wins=wins,
            total_rounds=len(round_scores),
            round_scores=tuple(round_scores),
            candidate_report=candidate_report,
            contract_evidence=contract_evidence,
            contract_policy_guard=contract_policy_guard,
        )
        if run_path is not None:
            write_json(
                run_path / f"selection_{candidate_name}_gate.json",
                {
                    **decision.to_dict(),
                    "policy": {
                        "initial_min_delta": self.config.min_delta,
                        "confirmation_rounds": self.config.validation_confirmation_rounds,
                        "required_wins": self.config.validation_required_wins,
                        "mean_delta": self.config.validation_mean_delta,
                    },
                    "candidate_report": candidate_report.to_dict(),
                },
            )
            write_timing_event(
                timing_path,
                "validation_gate_written",
                candidate_name=candidate_name,
                report=f"selection_{candidate_name}_gate.json",
                accepted=accepted,
                current_mean=current_mean,
                candidate_mean=candidate_mean,
            )
        write_timing_event(
            timing_path,
            "validation_finished",
            candidate_name=candidate_name,
            duration_seconds=time.monotonic() - validation_started,
            accepted=accepted,
            current_mean=current_mean,
            candidate_mean=candidate_mean,
            total_rounds=len(round_scores),
        )
        return decision

    def optimize(
        self,
        initial_skill_text: str,
        train_tasks: list[Task],
        validation_tasks: list[Task],
        *,
        run_dir: str | Path | None = None,
    ) -> ExecutiveOptimizationResult:
        require_editor_capability(
            self.editor,
            EDITOR_CAPABILITY_ATOMIC_EDITS,
            protocol="executive",
        )
        if not train_tasks:
            raise ValueError("train_tasks must not be empty")
        if not validation_tasks:
            raise ValueError("validation_tasks must not be empty")

        run_path = Path(run_dir) if run_dir is not None else None
        if run_path is not None:
            run_path.mkdir(parents=True, exist_ok=True)
            config_payload = asdict(self.config)
            if config_payload.get("meta_skill_path") is not None:
                config_payload["meta_skill_path"] = str(config_payload["meta_skill_path"])
            write_json(run_path / "executive_config.json", config_payload)
        timing_path = run_path / "timing_events.jsonl" if run_path is not None else None

        current_skill = initial_skill_text
        meta_skill = load_text(self.config_meta_path) if self.config_meta_path is not None else ""
        initial_report = self.evaluate(
            current_skill,
            validation_tasks,
            name="selection:initial",
            timing_path=timing_path,
        )
        current_score = initial_report.average_score
        best_report = initial_report
        candidate_hashes = {skill_hash(current_skill)}
        rejected_all: list[RejectedProposal] = []
        history = [
            OptimizationHistoryItem(
                epoch=0,
                candidate="initial",
                accepted=True,
                validation_score=current_score,
                rationale="Initial skill baseline.",
            )
        ]
        accepted_steps = 0
        validation_rejection_streak = 0
        global_step = 0
        stop_reason = "completed"
        steps_per_epoch = math.ceil(len(train_tasks) / self.config.rollout_batch_size)
        total_scheduled_steps = self.config.epochs * steps_per_epoch

        if run_path is not None:
            write_text(run_path / "best_skill.md", current_skill)
            write_json(run_path / "selection_initial.json", initial_report.to_dict())
            if meta_skill:
                write_text(run_path / "meta_skill_initial.md", meta_skill)
            self.write_result_checkpoint(
                run_path,
                current_skill=current_skill,
                current_score=current_score,
                best_report=best_report,
                history=history,
                rejected_all=rejected_all,
                meta_skill=meta_skill,
                accepted_steps=accepted_steps,
                total_steps=global_step,
                stop_reason="running",
                last_candidate="initial",
                validation_rejection_streak=validation_rejection_streak,
            )

        if self.should_early_stop_validation_score(current_score):
            stop_reason = "early_stop_validation_score_target"

        for epoch in range(1, self.config.epochs + 1):
            if stop_reason != "completed":
                break
            previous_epoch_skill = current_skill
            epoch_rejected: list[RejectedProposal] = []
            batches = rollout_batches(
                train_tasks,
                self.config.rollout_batch_size,
                seed=self.config.seed,
                epoch=epoch,
            )
            for batch_index, batch in enumerate(batches, 1):
                global_step += 1
                budget = scheduled_learning_rate(
                    self.config.learning_rate,
                    self.config.learning_rate_floor,
                    self.config.learning_rate_schedule,
                    global_step,
                    total_scheduled_steps,
                )
                train_report = self.evaluate(
                    current_skill,
                    batch,
                    name=f"train:epoch:{epoch}:batch:{batch_index}",
                    timing_path=timing_path,
                )
                if run_path is not None:
                    write_json(
                        run_path / f"train_epoch_{epoch}_batch_{batch_index}.json",
                        train_report.to_dict(),
                    )

                local_proposals: list[EditProposal] = []
                minibatches = reflection_minibatches(
                    train_report.results,
                    self.config.reflection_minibatch_size,
                )
                for minibatch_index, (kind, results) in enumerate(minibatches, 1):
                    controls = self.optimizer_controls(
                        phase="reflection",
                        epoch=epoch,
                        step=global_step,
                        batch_index=batch_index,
                        minibatch_index=minibatch_index,
                        reflection_kind=kind,
                        atomic_edit_budget=budget,
                    )
                    proposals = propose_with_optional_context(
                        self.editor,
                        current_skill,
                        results,
                        epoch=epoch,
                        rejected_buffer=self.rejected_buffer_payload(rejected_all),
                        meta_skill=meta_skill,
                        optimizer_controls=controls,
                    )
                    for proposal in proposals:
                        if not proposal.edits:
                            raise ValueError(
                                "executive editor returned a non-atomic proposal "
                                f"{proposal.name!r}; every proposal must contain atomic edits"
                            )
                        proposal_policy_issues = evidence_guided_proposal_filter_issues(proposal)
                        if proposal_policy_issues:
                            self.reject_without_validation(
                                epoch,
                                proposal.name,
                                "proposal_policy_rejected",
                                proposal.rationale,
                                {
                                    "proposal": proposal.to_dict(),
                                    "step": global_step,
                                    "proposal_policy_issues": proposal_policy_issues,
                                },
                                epoch_rejected,
                                rejected_all,
                                history,
                            )
                            continue
                        priority = float(proposal.metadata.get("priority", 0.0))
                        penalty = local_proposal_penalty(proposal, rejected_all)
                        adjusted_priority = priority + (1.0 if kind == "failure" else 0.0) - penalty
                        local_proposals.append(
                            EditProposal(
                                name=proposal.name,
                                skill_text=proposal.skill_text,
                                rationale=proposal.rationale,
                                metadata={
                                    **proposal.metadata,
                                    "priority": adjusted_priority,
                                    "reflection_kind": kind,
                                    "minibatch_index": minibatch_index,
                                    "local_policy_penalty": penalty,
                                },
                                edits=penalize_edit_priorities(proposal.edits, penalty),
                            )
                        )

                merged = merge_and_rank_atomic_edits(local_proposals, budget=budget)
                if not merged.selected:
                    continue
                candidate_contract_policy = build_candidate_contract_policy(
                    merged.selected,
                    local_proposals,
                )
                candidate_name = f"atomic-epoch-{epoch}-batch-{batch_index}"
                rationale = "Merged and ranked trajectory-derived atomic edits."
                merge_metadata = {
                    "phase": "fast_update",
                    "step": global_step,
                    "atomic_edit_budget": budget,
                    "selected_edits": [edit.to_dict() for edit in merged.selected],
                    "ranked_edits": [
                        {**item.edit.to_dict(), "support": item.support}
                        for item in merged.ranked
                    ],
                    "duplicate_count": merged.duplicate_count,
                    "conflict_count": merged.conflict_count,
                    "candidate_contract_policy": candidate_contract_policy,
                }
                try:
                    candidate_skill = apply_atomic_edits(current_skill, merged.selected)
                except ValueError as exc:
                    self.reject_without_validation(
                        epoch,
                        candidate_name,
                        "atomic_edit_application_failed",
                        rationale,
                        {**merge_metadata, "error": str(exc)},
                        epoch_rejected,
                        rejected_all,
                        history,
                    )
                    continue
                if len(candidate_skill) > self.config.max_skill_chars:
                    self.reject_without_validation(
                        epoch,
                        candidate_name,
                        "max_skill_chars_exceeded",
                        rationale,
                        {**merge_metadata, "skill_chars": len(candidate_skill)},
                        epoch_rejected,
                        rejected_all,
                        history,
                    )
                    continue
                digest = skill_hash(candidate_skill)
                if digest in candidate_hashes:
                    self.reject_without_validation(
                        epoch,
                        candidate_name,
                        "duplicate_candidate_hash",
                        rationale,
                        merge_metadata,
                        epoch_rejected,
                        rejected_all,
                        history,
                    )
                    continue
                candidate_hashes.add(digest)
                if run_path is not None:
                    write_text(run_path / f"candidate_{candidate_name}.md", candidate_skill)
                    write_timing_event(
                        timing_path,
                        "candidate_written",
                        candidate_name=candidate_name,
                        epoch=epoch,
                        batch_index=batch_index,
                        step=global_step,
                        phase="fast_update",
                        skill_chars=len(candidate_skill),
                        candidate_file=f"candidate_{candidate_name}.md",
                    )

                decision = self.validate_candidate(
                    current_skill,
                    candidate_skill,
                    validation_tasks,
                    candidate_name=candidate_name,
                    current_score=current_score,
                    current_report=best_report,
                    run_path=run_path,
                    candidate_contract_policy=candidate_contract_policy,
                )
                report = decision.candidate_report
                accepted = decision.accepted
                rejection_reason = None if accepted else validation_rejection_reason(decision)
                history.append(
                    OptimizationHistoryItem(
                        epoch=epoch,
                        candidate=candidate_name,
                        accepted=accepted,
                        validation_score=decision.candidate_mean,
                        rationale=rationale,
                        metadata={
                            **merge_metadata,
                            "validation_gate": decision.to_dict(),
                            "rejection_reason": rejection_reason,
                        },
                    )
                )
                if accepted:
                    current_skill = candidate_skill
                    current_score = decision.candidate_mean
                    best_report = report
                    accepted_steps += 1
                    validation_rejection_streak = 0
                    if run_path is not None:
                        write_text(run_path / "best_skill.md", current_skill)
                    if self.should_early_stop_validation_score(current_score):
                        stop_reason = "early_stop_validation_score_target"
                else:
                    validation_rejection_streak += 1
                    rejected = RejectedProposal(
                        epoch=epoch,
                        candidate=candidate_name,
                        reason=rejection_reason or "validation_gate_rejected",
                        rationale=rationale,
                        validation_score=decision.candidate_mean,
                        failed_task_ids=[item.task.id for item in report.results if not item.score.success],
                        metadata={**merge_metadata, "validation_gate": decision.to_dict()},
                    )
                    epoch_rejected.append(rejected)
                    rejected_all.append(rejected)
                if run_path is not None:
                    self.write_result_checkpoint(
                        run_path,
                        current_skill=current_skill,
                        current_score=current_score,
                        best_report=best_report,
                        history=history,
                        rejected_all=rejected_all,
                        meta_skill=meta_skill,
                        accepted_steps=accepted_steps,
                        total_steps=global_step,
                        stop_reason="running",
                        last_candidate=candidate_name,
                        validation_rejection_streak=validation_rejection_streak,
                    )
                if stop_reason == "completed" and self.should_early_stop(validation_rejection_streak):
                    stop_reason = "early_stop_validation_rejection_limit"
                if stop_reason != "completed":
                    break

            if stop_reason != "completed":
                if run_path is not None:
                    write_rejected_buffer_jsonl(run_path / "rejected_buffer.jsonl", rejected_all)
                break

            if self.config.enable_slow_update and hasattr(self.editor, "update_state"):
                sampled = sample_tasks(
                    train_tasks,
                    self.config.slow_update_sample_size,
                    seed=self.config.seed,
                    epoch=epoch,
                )
                comparison = self.compare_epoch_skills(
                    previous_epoch_skill,
                    current_skill,
                    sampled,
                    epoch=epoch,
                )
                update = self.editor.update_state(
                    epoch=epoch,
                    current_skill=current_skill,
                    meta_skill=meta_skill,
                    comparison=comparison,
                    rejected_buffer=self.rejected_buffer_payload(rejected_all),
                    optimizer_controls=self.optimizer_controls(
                        phase="slow_meta_update",
                        epoch=epoch,
                        step=global_step,
                    ),
                )
                if not isinstance(update, OptimizerStateUpdate):
                    raise TypeError("editor.update_state must return OptimizerStateUpdate")
                if update.meta_skill.strip():
                    meta_skill = update.meta_skill.strip()
                    if run_path is not None:
                        write_text(run_path / f"meta_skill_epoch_{epoch}.md", meta_skill)
                if update.slow_update.strip():
                    candidate_name = f"slow-update-epoch-{epoch}"
                    candidate_skill = set_slow_update(current_skill, update.slow_update)
                    digest = skill_hash(candidate_skill)
                    if digest not in candidate_hashes and len(candidate_skill) <= self.config.max_skill_chars:
                        candidate_hashes.add(digest)
                        if run_path is not None:
                            write_timing_event(
                                timing_path,
                                "candidate_created",
                                candidate_name=candidate_name,
                                epoch=epoch,
                                phase="slow_update",
                                skill_chars=len(candidate_skill),
                            )
                        decision = self.validate_candidate(
                            current_skill,
                            candidate_skill,
                            validation_tasks,
                            candidate_name=candidate_name,
                            current_score=current_score,
                            current_report=best_report,
                            run_path=run_path,
                        )
                        report = decision.candidate_report
                        accepted = decision.accepted
                        history.append(
                            OptimizationHistoryItem(
                                epoch=epoch,
                                candidate=candidate_name,
                                accepted=accepted,
                                validation_score=decision.candidate_mean,
                                rationale=update.rationale or "Epoch-wise longitudinal update.",
                                metadata={
                                    "phase": "slow_update",
                                    "validation_gate": decision.to_dict(),
                                    "rejection_reason": (
                                        None if accepted else "validation_gate_rejected"
                                    ),
                                    "comparison_counts": comparison["counts"],
                                },
                            )
                        )
                        if run_path is not None:
                            write_text(run_path / f"candidate_{candidate_name}.md", candidate_skill)
                        if accepted:
                            current_skill = candidate_skill
                            current_score = decision.candidate_mean
                            best_report = report
                            accepted_steps += 1
                            validation_rejection_streak = 0
                            if run_path is not None:
                                write_text(run_path / "best_skill.md", current_skill)
                            if self.should_early_stop_validation_score(current_score):
                                stop_reason = "early_stop_validation_score_target"
                        else:
                            validation_rejection_streak += 1
                            rejected = RejectedProposal(
                                epoch=epoch,
                                candidate=candidate_name,
                                reason="validation_gate_rejected",
                                rationale=update.rationale or "Epoch-wise longitudinal update.",
                                validation_score=decision.candidate_mean,
                                failed_task_ids=[item.task.id for item in report.results if not item.score.success],
                                metadata={
                                    "phase": "slow_update",
                                    "validation_gate": decision.to_dict(),
                                    "comparison_counts": comparison["counts"],
                                },
                            )
                            epoch_rejected.append(rejected)
                            rejected_all.append(rejected)
                        if run_path is not None:
                            self.write_result_checkpoint(
                                run_path,
                                current_skill=current_skill,
                                current_score=current_score,
                                best_report=best_report,
                                history=history,
                                rejected_all=rejected_all,
                                meta_skill=meta_skill,
                                accepted_steps=accepted_steps,
                                total_steps=global_step,
                                stop_reason="running",
                                last_candidate=candidate_name,
                                validation_rejection_streak=validation_rejection_streak,
                            )
                        if self.should_early_stop(validation_rejection_streak):
                            stop_reason = "early_stop_validation_rejection_limit"

            if run_path is not None:
                write_rejected_buffer_jsonl(run_path / "rejected_buffer.jsonl", rejected_all)
            if stop_reason != "completed":
                break

        final_report = EvaluationReport(
            name="selection:final:accepted-gate-evidence",
            results=best_report.results,
        )
        result = ExecutiveOptimizationResult(
            best_skill_text=current_skill,
            best_validation_score=current_score,
            history=history,
            final_validation_report=final_report,
            rejected_buffer=rejected_all,
            meta_skill_text=meta_skill,
            accepted_steps=accepted_steps,
            total_steps=global_step,
            stop_reason=stop_reason,
            checkpoint={
                "validation_rejection_streak": validation_rejection_streak,
                "early_stop_rejection_limit": self.config.early_stop_rejection_limit,
            },
        )
        if run_path is not None:
            write_json(run_path / "result.json", result.to_dict())
            write_rejected_buffer_jsonl(run_path / "rejected_buffer.jsonl", rejected_all)
            write_json(run_path / "result_checkpoint.json", result.to_dict())
        return result

    def should_early_stop(self, validation_rejection_streak: int) -> bool:
        return (
            self.config.early_stop_rejection_limit > 0
            and validation_rejection_streak >= self.config.early_stop_rejection_limit
        )

    def should_early_stop_validation_score(self, validation_score: float) -> bool:
        return (
            self.config.early_stop_validation_score is not None
            and validation_score >= self.config.early_stop_validation_score
        )

    def write_result_checkpoint(
        self,
        run_path: Path,
        *,
        current_skill: str,
        current_score: float,
        best_report: EvaluationReport,
        history: list[OptimizationHistoryItem],
        rejected_all: list[RejectedProposal],
        meta_skill: str,
        accepted_steps: int,
        total_steps: int,
        stop_reason: str,
        last_candidate: str,
        validation_rejection_streak: int,
    ) -> None:
        checkpoint = ExecutiveOptimizationResult(
            best_skill_text=current_skill,
            best_validation_score=current_score,
            history=history,
            final_validation_report=EvaluationReport(
                name="selection:checkpoint:accepted-gate-evidence",
                results=best_report.results,
            ),
            rejected_buffer=rejected_all,
            meta_skill_text=meta_skill,
            accepted_steps=accepted_steps,
            total_steps=total_steps,
            stop_reason=stop_reason,
            checkpoint={
                "last_candidate": last_candidate,
                "validation_rejection_streak": validation_rejection_streak,
                "early_stop_rejection_limit": self.config.early_stop_rejection_limit,
            },
        )
        write_json(run_path / "result_checkpoint.json", checkpoint.to_dict())

    @property
    def config_meta_path(self) -> Path | None:
        value = getattr(self.config, "meta_skill_path", None)
        return Path(value) if value is not None else None

    def optimizer_controls(self, *, phase: str, epoch: int, step: int, **extra: Any) -> dict[str, Any]:
        return {
            "phase": phase,
            "epoch": epoch,
            "step": step,
            "learning_rate": self.config.learning_rate,
            "learning_rate_floor": self.config.learning_rate_floor,
            "learning_rate_schedule": self.config.learning_rate_schedule,
            "max_skill_chars": self.config.max_skill_chars,
            **extra,
        }

    def rejected_buffer_payload(self, rejected_all: list[RejectedProposal]) -> list[dict[str, Any]]:
        return rejected_payload(rejected_all, self.config.rejected_buffer_limit)

    def reject_without_validation(
        self,
        epoch: int,
        candidate: str,
        reason: str,
        rationale: str,
        metadata: dict[str, Any],
        epoch_rejected: list[RejectedProposal],
        rejected_all: list[RejectedProposal],
        history: list[OptimizationHistoryItem],
    ) -> None:
        rejected = RejectedProposal(
            epoch=epoch,
            candidate=candidate,
            reason=reason,
            rationale=rationale,
            validation_score=None,
            metadata=metadata,
        )
        epoch_rejected.append(rejected)
        rejected_all.append(rejected)
        history.append(
            OptimizationHistoryItem(
                epoch=epoch,
                candidate=candidate,
                accepted=False,
                validation_score=None,
                rationale=rationale,
                metadata={**metadata, "rejection_reason": reason},
            )
        )

    def compare_epoch_skills(
        self,
        previous_skill: str,
        current_skill: str,
        tasks: list[Task],
        *,
        epoch: int,
    ) -> dict[str, Any]:
        previous = self.evaluate(previous_skill, tasks, name=f"slow:epoch:{epoch}:previous")
        current = previous if previous_skill == current_skill else self.evaluate(
            current_skill,
            tasks,
            name=f"slow:epoch:{epoch}:current",
        )
        items = []
        counts = {"improvement": 0, "regression": 0, "persistent_failure": 0, "stable_success": 0}
        for old, new in zip(previous.results, current.results):
            if new.score.value > old.score.value:
                category = "improvement"
            elif new.score.value < old.score.value:
                category = "regression"
            elif new.score.success and old.score.success:
                category = "stable_success"
            else:
                category = "persistent_failure"
            counts[category] += 1
            items.append(
                {
                    "task_id": old.task.id,
                    "category": category,
                    "previous": old.to_dict(),
                    "current": new.to_dict(),
                }
            )
        return {"epoch": epoch, "counts": counts, "items": items}


def rollout_batches(tasks: list[Task], batch_size: int, *, seed: int, epoch: int) -> list[list[Task]]:
    ordered = list(tasks)
    random.Random(f"{seed}:epoch:{epoch}").shuffle(ordered)
    return [ordered[index : index + batch_size] for index in range(0, len(ordered), batch_size)]


def write_timing_event(path: Path | None, event: str, **payload: Any) -> None:
    if path is None:
        return
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "schema_version": 1,
        "event": event,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "monotonic_seconds": time.monotonic(),
        **payload,
    }
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def reflection_minibatches(results: list[TaskResult], minibatch_size: int) -> list[tuple[str, list[TaskResult]]]:
    failures = [result for result in results if not result.score.success]
    successes = [result for result in results if result.score.success]
    batches: list[tuple[str, list[TaskResult]]] = []
    for kind, group in (("failure", failures), ("success", successes)):
        for index in range(0, len(group), minibatch_size):
            batches.append((kind, group[index : index + minibatch_size]))
    return batches


def scheduled_learning_rate(initial: int, floor: int, schedule: str, step: int, total_steps: int) -> int:
    if total_steps <= 1 or schedule == "constant":
        return initial
    progress = min(1.0, max(0.0, (step - 1) / (total_steps - 1)))
    if schedule == "linear":
        value = initial + (floor - initial) * progress
    elif schedule == "cosine":
        value = floor + (initial - floor) * (1 + math.cos(math.pi * progress)) / 2
    else:
        raise ValueError(f"Unsupported learning-rate schedule: {schedule}")
    return max(floor, min(initial, int(round(value))))


def sample_tasks(tasks: list[Task], limit: int, *, seed: int, epoch: int) -> list[Task]:
    if limit <= 0 or limit >= len(tasks):
        return list(tasks)
    ordered = list(tasks)
    random.Random(f"{seed}:slow:{epoch}").shuffle(ordered)
    return ordered[:limit]


def skill_hash(skill_text: str) -> str:
    return hashlib.sha256(skill_text.encode("utf-8")).hexdigest()


def evidence_guided_proposal_filter_issues(proposal: EditProposal) -> list[str]:
    """Reject evidence-guided proposals that cannot plausibly move their declared target."""
    metadata = proposal.metadata if isinstance(proposal.metadata, dict) else {}
    if str(metadata.get("evidence_source") or "") != "contract_rejection_evidence":
        return []
    issues = []
    targeted = normalized_metadata_list(metadata.get("targeted_contracts"))
    protected = normalized_metadata_list(metadata.get("protected_contracts"))
    if not targeted:
        issues.append("missing_targeted_contracts")
    if len(targeted) > 1:
        issues.append("multiple_targeted_contracts")
    if not str(metadata.get("expected_behavior_change") or "").strip():
        issues.append("missing_expected_behavior_change")
    edit_text = proposal_edit_semantic_text(proposal)
    if is_guard_only_proposal(proposal):
        issues.append("target_mechanism_missing")
    elif targeted and missing_contract_mechanism_anchors(targeted, edit_text):
        issues.append("target_mechanism_missing")
    for contract in missing_contract_mechanism_anchors(protected, edit_text):
        issues.append(f"protected_mechanism_missing:{contract}")
    return issues


def is_guard_only_proposal(proposal: EditProposal) -> bool:
    edit_text = proposal_edit_semantic_text(proposal)
    if not edit_text:
        return True
    guard_phrases = (
        "run all existing",
        "re run all",
        "previously passing",
        "public tests",
        "before modifying",
        "before implementing",
        "before applying",
        "confirm every",
        "confirm all existing",
        "preserve existing",
        "remains fully intact",
        "do not alter working",
        "already passing",
    )
    mechanism_phrases = (
        "casefold",
        "str casefold",
        "calculate",
        "assign",
        "distribute",
        "fractional",
        "remainder",
        "floor",
        "remaining unallocated",
        "sort",
        "break ties",
        "ascending original",
        "return",
        "raise",
        "exception",
        "error",
        "invalid input",
        "validate inputs",
        "zero weight",
        "suffix",
        "deduplicate",
    )
    has_guard_language = any(phrase in edit_text for phrase in guard_phrases)
    has_mechanism_language = any(phrase in edit_text for phrase in mechanism_phrases)
    return has_guard_language and not has_mechanism_language


def proposal_edit_semantic_text(proposal: EditProposal) -> str:
    return normalize_semantic_text(" ".join(edit.content for edit in proposal.edits))


def missing_contract_mechanism_anchors(contracts: list[str], edit_text: str) -> list[str]:
    if not edit_text:
        return [contract for contract in contracts if contract]
    return [
        contract
        for contract in contracts
        if contract and not contract_mechanism_anchor_present(contract, edit_text)
    ]


def contract_mechanism_anchor_present(contract: str, edit_text: str) -> bool:
    if contract == "input_validation":
        return input_validation_mechanism_present(edit_text)
    if contract == "stable_order":
        return any(term in edit_text for term in ("stable", "order", "original", "index", "tie"))
    if contract == "largest_remainder":
        return largest_remainder_mechanism_present(edit_text)
    if contract == "unicode_casefold":
        return "casefold" in edit_text or ("unicode" in edit_text and "case" in edit_text)
    tokens = contract_anchor_tokens(contract)
    if not tokens:
        return True
    return any(token in edit_text for token in tokens)


def input_validation_mechanism_present(edit_text: str) -> bool:
    has_raise = any(term in edit_text for term in ("raise", "error", "valueerror", "exception"))
    has_invalid_class = any(
        term in edit_text
        for term in (
            "negative",
            "non negative",
            "nonnegative",
            "less than zero",
            "below zero",
        )
    )
    return has_raise and has_invalid_class


def largest_remainder_mechanism_present(edit_text: str) -> bool:
    has_remainder_core = (
        "remainder" in edit_text
        and any(term in edit_text for term in ("largest", "fractional"))
        and any(term in edit_text for term in ("floor", "quota", "integer"))
        and any(term in edit_text for term in ("distribute", "assign", "allocate"))
    )
    return has_remainder_core and zero_sum_all_zero_mechanism_present(edit_text)


def zero_sum_all_zero_mechanism_present(edit_text: str) -> bool:
    forbidden_zero_sum_semantics = (
        "zero sum weight inputs raise",
        "zero sum weights raise",
        "sum of weights is zero raise",
        "total weight is zero raise",
        "all zero weights raise",
        "zero weights raise",
        "zero sum weight inputs distribute",
        "zero sum weights distribute",
        "all zero weights distribute",
        "distribute equally",
        "average distribute",
    )
    if any(phrase in edit_text for phrase in forbidden_zero_sum_semantics):
        return False
    names_zero_sum_case = any(
        phrase in edit_text
        for phrase in (
            "zero sum",
            "sum of weights is zero",
            "total weight is zero",
            "all zero weights",
            "all weights are zero",
            "zero weights",
        )
    )
    returns_all_zero_output = any(
        phrase in edit_text
        for phrase in (
            "return all zero",
            "return all zeros",
            "return an all zero",
            "return a zero filled",
            "return zero filled",
            "all zero output",
            "all zero array",
            "all zero list",
            "zero filled output",
            "zeros matching",
            "output zeros",
        )
    )
    return names_zero_sum_case and returns_all_zero_output


def contract_anchor_tokens(contract: str) -> set[str]:
    raw_tokens = normalize_semantic_text(contract.replace("_", " ")).split()
    stopwords = {"contract", "contracts", "case", "cases", "edge", "edges", "rule", "rules"}
    tokens: set[str] = set()
    for token in raw_tokens:
        if len(token) < 4 or token in stopwords:
            continue
        tokens.add(token)
        if token.startswith("valid"):
            tokens.add("valid")
        if token.startswith("invalid"):
            tokens.add("invalid")
        if token.endswith("ing") and len(token) > 5:
            tokens.add(token[:-3])
        if token.endswith("ed") and len(token) > 4:
            tokens.add(token[:-2])
        if token.endswith("s") and len(token) > 4:
            tokens.add(token[:-1])
    return tokens


def build_candidate_contract_policy(
    selected_edits: tuple[AtomicEdit, ...],
    proposals: list[EditProposal],
) -> dict[str, Any]:
    selected_keys = {canonical_edit_key(edit) for edit in selected_edits}
    source_proposals = [
        proposal
        for proposal in proposals
        if selected_keys.intersection(canonical_edit_key(edit) for edit in proposal.edits)
    ]
    targeted = sorted(
        {
            contract
            for proposal in source_proposals
            for contract in normalized_metadata_list(proposal.metadata.get("targeted_contracts"))
        }
    )
    protected = sorted(
        {
            contract
            for proposal in source_proposals
            for contract in normalized_metadata_list(proposal.metadata.get("protected_contracts"))
        }
    )
    evidence_sources = sorted(
        {
            str(proposal.metadata.get("evidence_source") or "")
            for proposal in source_proposals
            if str(proposal.metadata.get("evidence_source") or "")
        }
    )
    return {
        "required": "contract_rejection_evidence" in evidence_sources,
        "source_proposals": [proposal.name for proposal in source_proposals],
        "targeted_contracts": targeted,
        "protected_contracts": protected,
        "evidence_sources": evidence_sources,
        "expected_behavior_change_present": any(
            bool(str(proposal.metadata.get("expected_behavior_change") or "").strip())
            for proposal in source_proposals
        ),
        "cooldown_override_present": any(
            bool(str(proposal.metadata.get("cooldown_override") or "").strip())
            for proposal in source_proposals
        ),
    }


def evaluate_contract_policy_guard(
    contract_evidence: dict[str, Any],
    candidate_contract_policy: dict[str, Any] | None,
) -> dict[str, Any]:
    policy = candidate_contract_policy if isinstance(candidate_contract_policy, dict) else {}
    required = bool(policy.get("required"))
    targeted = normalized_metadata_list(policy.get("targeted_contracts"))
    protected = normalized_metadata_list(policy.get("protected_contracts"))
    deltas = {
        str(contract): payload
        for contract, payload in dict(contract_evidence.get("contract_deltas") or {}).items()
        if isinstance(payload, dict)
    }
    targeted_outcomes = contract_guard_outcomes(targeted, deltas)
    protected_outcomes = contract_guard_outcomes(protected, deltas)
    issues = []
    targeted_evaluated = [
        outcome for outcome in targeted_outcomes.values() if outcome["delta"] is not None
    ]
    targeted_improved = [
        contract
        for contract, outcome in targeted_outcomes.items()
        if outcome["delta"] is not None and outcome["delta"] > 0
    ]
    protected_regressed = [
        contract
        for contract, outcome in protected_outcomes.items()
        if outcome["delta"] is not None and outcome["delta"] < 0
    ]
    if required and targeted and not targeted_evaluated:
        issues.append("targeted_contract_not_evaluated")
    elif required and targeted_evaluated and not targeted_improved:
        issues.append("targeted_contract_not_improved")
    if required and protected_regressed:
        issues.append("protected_contract_regressed")
    return {
        "required": required,
        "passes": not issues,
        "issues": issues,
        "targeted_contracts": targeted,
        "protected_contracts": protected,
        "targeted_improved_contracts": targeted_improved,
        "protected_regressed_contracts": protected_regressed,
        "targeted_contract_outcomes": targeted_outcomes,
        "protected_contract_outcomes": protected_outcomes,
        "policy": policy,
    }


def contract_guard_outcomes(
    contracts: list[str],
    deltas: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    outcomes = {}
    for contract in sorted(set(contracts)):
        if contract not in deltas:
            outcomes[contract] = {
                "current_accuracy": None,
                "candidate_accuracy": None,
                "delta": None,
                "status": "not_evaluated",
            }
            continue
        payload = deltas[contract]
        current_accuracy = float(payload.get("current_accuracy") or 0.0)
        candidate_accuracy = float(payload.get("candidate_accuracy") or 0.0)
        delta = float(payload.get("delta") or 0.0)
        if delta > 0:
            status = "improved"
        elif delta < 0:
            status = "regressed"
        elif candidate_accuracy < 1.0:
            status = "unchanged_failed"
        else:
            status = "unchanged_passed"
        outcomes[contract] = {
            "current_accuracy": current_accuracy,
            "candidate_accuracy": candidate_accuracy,
            "delta": delta,
            "status": status,
        }
    return outcomes


def validation_rejection_reason(decision: ValidationGateDecision) -> str:
    guard = decision.contract_policy_guard if isinstance(decision.contract_policy_guard, dict) else {}
    if guard.get("required") and not guard.get("passes"):
        return "contract_policy_rejected"
    return "validation_gate_rejected"


def local_proposal_penalty(proposal: EditProposal, rejected_all: list[RejectedProposal]) -> float:
    """Lower rank for weak or repeated proposal patterns without hard-rejecting them."""
    penalty = 0.0
    targeted = normalized_metadata_list(proposal.metadata.get("targeted_contracts"))
    if len(targeted) > 1:
        penalty += 0.5
    recently_failed = recent_failed_target_contracts(rejected_all)
    if set(targeted).intersection(recently_failed) and not declares_new_mechanism(proposal.metadata):
        penalty += 1.25
    if any(is_generic_contract_audit(edit.content) for edit in proposal.edits):
        penalty += 0.75
    rejected_signatures = rejected_edit_signatures(rejected_all)
    for edit in proposal.edits:
        signature = semantic_edit_signature(edit.content)
        if signature and signature in rejected_signatures:
            penalty += 1.0
            break
    return penalty


def recent_failed_target_contracts(
    rejected_all: list[RejectedProposal],
    *,
    recent_limit: int = 5,
) -> set[str]:
    contracts: set[str] = set()
    for item in rejected_all[-recent_limit:]:
        metadata = item.metadata if isinstance(item.metadata, dict) else {}
        gate = metadata.get("validation_gate")
        gate = gate if isinstance(gate, dict) else {}
        evidence = gate.get("contract_evidence")
        evidence = evidence if isinstance(evidence, dict) else {}
        for key in ("top_negative_contracts", "top_no_improvement_contracts"):
            for payload in evidence.get(key) or []:
                if not isinstance(payload, dict):
                    continue
                contract = str(payload.get("contract") or "").strip()
                if contract:
                    contracts.add(contract)
    return contracts


def declares_new_mechanism(metadata: dict[str, Any]) -> bool:
    for key in ("cooldown_override", "new_mechanism", "new_mechanism_explanation"):
        if str(metadata.get(key) or "").strip():
            return True
    return False


def penalize_edit_priorities(edits: tuple[AtomicEdit, ...], penalty: float) -> tuple[AtomicEdit, ...]:
    if penalty <= 0:
        return edits
    return tuple(
        AtomicEdit(
            operation=edit.operation,
            target=edit.target,
            content=edit.content,
            rationale=edit.rationale,
            priority=edit.priority - penalty,
        )
        for edit in edits
    )


def rejected_edit_signatures(rejected_all: list[RejectedProposal]) -> set[str]:
    signatures: set[str] = set()
    for item in rejected_all:
        metadata = item.metadata if isinstance(item.metadata, dict) else {}
        for edit in metadata.get("selected_edits", []) or []:
            if not isinstance(edit, dict):
                continue
            signature = semantic_edit_signature(str(edit.get("content") or ""))
            if signature:
                signatures.add(signature)
    return signatures


def semantic_edit_signature(content: str) -> str:
    normalized = normalize_semantic_text(content)
    if not normalized:
        return ""
    if is_generic_contract_audit(normalized):
        return "generic_contract_audit"
    tokens = normalized.split()
    return " ".join(tokens[:18])


def normalize_semantic_text(content: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9_ ]+", " ", content.lower())).strip()


def is_generic_contract_audit(content: str) -> bool:
    normalized = normalize_semantic_text(content)
    if not normalized:
        return False
    generic_phrases = (
        "all documented",
        "every required",
        "full contract",
        "all required",
        "each required",
        "verify all",
        "confirm all",
    )
    audit_words = ("contract", "contracts", "requirements", "invariants", "behavior")
    return any(phrase in normalized for phrase in generic_phrases) and any(
        word in normalized for word in audit_words
    )


def normalized_metadata_list(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value] if value else []
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if str(item)]
