"""One replayable Algorithm 1 fast step for paper-faithful optimization."""

from __future__ import annotations

import hashlib
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

from .backend import OptimizerRequest, OptimizerResponse, OptimizerStage
from .artifacts import OptimizerExchange
from .config import PaperProfile
from .data import (
    SelectionScore,
    StepTrainEvidence,
    TrainEvidenceBatch,
    strict_selection_decision,
)
from .epoch_plan import EpochBatchCursor, PaperMechanismSpec
from .errors import SkillContractViolation
from .optimization import PaperOptimizationController
from .patches import (
    PatchApplyResult,
    RewriteApplyResult,
    apply_paper_patch,
    apply_paper_rewrite,
)
from .prompts import load_optimizer_prompt
from .provenance import canonical_json_sha256
from .responses import (
    OptimizerContractViolation,
    ParsedPatchResponse,
    ParsedSuggestionResponse,
    learning_rate_response_schema,
    optimizer_response_schema,
    parse_learning_rate_response,
    parse_patch_response,
    parse_rank_response,
    parse_rewrite_response,
    parse_suggestion_rank_response,
    parse_suggestion_response,
    rewrite_response_schema,
)
from .types import (
    AlgorithmEvent,
    AlgorithmEventType,
    EpochBufferRecord,
    ObservedFailurePattern,
    PaperEdit,
    PaperEditSource,
    PaperSuggestion,
    PaperState,
)


@dataclass(frozen=True)
class OptimizerRetryPolicy:
    """Recorded provider policy for semantic merge/rank retries."""

    policy_id: str = "semantic-retry-once-v1"
    max_semantic_attempts: int = 2

    def __post_init__(self) -> None:
        if type(self.policy_id) is not str or not self.policy_id.strip():
            raise ValueError("optimizer retry policy requires policy_id")
        if (
            type(self.max_semantic_attempts) is not int
            or not 1 <= self.max_semantic_attempts <= 5
        ):
            raise ValueError("max_semantic_attempts must be between 1 and 5")


class _SemanticStageExhausted(RuntimeError):
    def __init__(
        self,
        *,
        stage: OptimizerStage,
        attempts: int,
        last_error: OptimizerContractViolation,
    ) -> None:
        super().__init__(str(last_error))
        self.stage = stage
        self.attempts = attempts


ParsedUpdateResponse = ParsedPatchResponse | ParsedSuggestionResponse


@dataclass(frozen=True)
class FastStepResult:
    """Complete state transition and reconstruction inputs for one fast step."""

    input_skill: str
    state: PaperState
    candidate_score: SelectionScore
    ranked_edits: tuple[PaperEdit, ...]
    apply_result: PatchApplyResult | RewriteApplyResult
    events: tuple[AlgorithmEvent, ...]
    failure_patterns: tuple[ObservedFailurePattern, ...] = ()
    skipped_stage: OptimizerStage | None = None
    ranked_suggestions: tuple[PaperSuggestion, ...] = ()
    rewrite_result: RewriteApplyResult | None = None
    optimizer_exchanges: tuple[OptimizerExchange, ...] = ()
    selection_skipped_reason: str | None = None

    def replay(self) -> PatchApplyResult | RewriteApplyResult:
        if self.rewrite_result is None:
            replayed: PatchApplyResult | RewriteApplyResult = apply_paper_patch(
                self.input_skill,
                self.ranked_edits,
            )
        else:
            replayed = apply_paper_rewrite(
                self.input_skill,
                self.ranked_suggestions,
                new_skill=self.rewrite_result.output_skill,
                reasoning=self.rewrite_result.reasoning,
                change_summary=self.rewrite_result.change_summary,
            )
        if replayed != self.apply_result:
            raise RuntimeError("fast-step artifact does not replay exactly")
        return replayed


@dataclass(frozen=True)
class ExternalGateResult:
    state: PaperState
    candidate_score: SelectionScore
    accepted: bool
    delta: float
    cache_hit: bool
    events: tuple[AlgorithmEvent, ...]
    selection_skipped_reason: str | None = None


@dataclass(frozen=True)
class _ExternalGatePreview:
    input_state: PaperState
    candidate_skill_sha256: str
    candidate_score: SelectionScore
    accepted: bool
    delta: float
    cache_hit: bool
    score_cache: tuple[tuple[str, SelectionScore], ...]
    selection_skipped_reason: str | None = None
    selection_skipped_message: str | None = None


class PaperFastLoop:
    """Execute the paper fast path behind one narrow, train-evidence-only API."""

    def __init__(
        self,
        controller: PaperOptimizationController,
        *,
        profile: PaperProfile,
        retry_policy: OptimizerRetryPolicy = OptimizerRetryPolicy(),
        _mechanisms: PaperMechanismSpec | None = None,
    ) -> None:
        if type(controller) is not PaperOptimizationController:
            raise ValueError(
                "paper fast loop requires exact PaperOptimizationController"
            )
        controller.__post_init__()
        if type(profile) is not PaperProfile:
            raise ValueError("paper fast loop requires exact PaperProfile")
        validated_profile = PaperProfile.from_mapping(profile.to_dict())
        mechanisms = (
            PaperMechanismSpec.from_profile(validated_profile)
            if _mechanisms is None
            else _mechanisms
        )
        if type(mechanisms) is not PaperMechanismSpec:
            raise ValueError("paper fast loop requires exact mechanism spec")
        mechanisms.require_profile(validated_profile)
        if type(retry_policy) is not OptimizerRetryPolicy:
            raise ValueError("paper fast loop requires exact OptimizerRetryPolicy")
        retry_policy.__post_init__()
        self._controller = controller
        self._profile = validated_profile
        self._mechanisms = mechanisms
        self._profile_sha256 = canonical_json_sha256(validated_profile.to_dict())
        self._retry_policy = retry_policy
        self._score_cache: dict[str, SelectionScore] = {}
        self._next_event_sequence = 0
        self._state: PaperState | None = None
        self._epoch_buffer: tuple[EpochBufferRecord, ...] = ()
        self._scheduled_batches: tuple[EpochBatchCursor, ...] = ()
        self._exchange_lock = threading.Lock()
        self._active_exchanges: list[OptimizerExchange] = []

    @property
    def score_cache(self) -> Mapping[str, SelectionScore]:
        return MappingProxyType(dict(self._score_cache))

    @property
    def next_event_sequence(self) -> int:
        return self._next_event_sequence

    @property
    def state(self) -> PaperState:
        if self._state is None:
            raise ValueError("paper fast loop must be initialized")
        return self._state

    def initialize(self, initial_skill: str) -> PaperState:
        """Evaluate Algorithm 1's initial skill through the selection owner."""

        self._controller.__post_init__()
        if type(initial_skill) is not str or not initial_skill.strip():
            raise ValueError("paper fast loop requires initial_skill")
        if self._score_cache:
            raise ValueError("paper fast loop is already initialized")
        self._controller.validate_skill(initial_skill)
        score = self._controller.selection.score(initial_skill)
        self._score_cache[_sha256(initial_skill)] = score
        self._state = PaperState(
            epoch=1,
            step=0,
            current_skill=initial_skill,
            current_score=score,
            best_skill=initial_skill,
            best_score=score,
        )
        return self._state

    def _record_lifecycle_event(
        self,
        event_type: AlgorithmEventType,
        payload: Mapping[str, Any],
    ) -> AlgorithmEvent:
        """Reserve one event in the shared sequence for the epoch owner."""

        if self._state is None:
            raise ValueError("paper fast loop must be initialized")
        if event_type not in {
            AlgorithmEventType.RUN_STARTED,
            AlgorithmEventType.EPOCH_STARTED,
            AlgorithmEventType.SLOW_UPDATE_SKIPPED,
            AlgorithmEventType.SLOW_UPDATE_PROPOSED,
            AlgorithmEventType.META_UPDATE_SKIPPED,
            AlgorithmEventType.META_UPDATE_COMPLETED,
            AlgorithmEventType.EPOCH_COMPLETED,
            AlgorithmEventType.RUN_COMPLETED,
        }:
            raise ValueError("event is not owned by the paper epoch loop")
        event = AlgorithmEvent(
            sequence=self._next_event_sequence,
            event_type=event_type,
            epoch=self._state.epoch,
            step=None,
            payload=dict(payload),
        )
        self._next_event_sequence += 1
        return event

    def _set_epoch_buffer(
        self,
        records: tuple[EpochBufferRecord, ...],
    ) -> None:
        """Set optimizer context derived by the owning epoch loop."""

        if self._state is None:
            raise ValueError("paper fast loop must be initialized")
        if type(records) is not tuple or any(
            type(record) is not EpochBufferRecord for record in records
        ):
            raise ValueError("paper fast loop requires exact epoch buffer records")
        if any(record.epoch != self._state.epoch for record in records):
            raise ValueError("epoch buffer cannot cross epoch boundaries")
        self._epoch_buffer = records

    def _prepare_epoch_step(
        self,
        records: tuple[EpochBufferRecord, ...],
        *,
        batches: tuple[EpochBatchCursor, ...],
    ) -> None:
        self._set_epoch_buffer(records)
        if type(batches) is not tuple or not batches or any(
            type(batch) is not EpochBatchCursor for batch in batches
        ):
            raise ValueError("paper epoch step requires scheduled batches")
        if [batch.accumulation_index for batch in batches] != list(
            range(1, len(batches) + 1)
        ):
            raise ValueError("paper epoch step batch order is not contiguous")
        self._scheduled_batches = batches

    def _begin_next_epoch(self) -> PaperState:
        """Advance lifecycle position without exposing state injection publicly."""

        state = self.state
        self._state = PaperState(
            epoch=state.epoch + 1,
            step=0,
            current_skill=state.current_skill,
            current_score=state.current_score,
            best_skill=state.best_skill,
            best_score=state.best_score,
            meta_skill=state.meta_skill,
        )
        self._epoch_buffer = ()
        return self._state

    def _checkpoint_payload(self) -> dict[str, Any]:
        state = self.state
        return {
            "state": {
                "epoch": state.epoch,
                "step": state.step,
                "current_skill": state.current_skill,
                "current_score": state.current_score.value,
                "best_skill": state.best_skill,
                "best_score": state.best_score.value,
                "meta_skill": state.meta_skill,
            },
            "score_cache": {
                key: score.value for key, score in sorted(self._score_cache.items())
            },
            "next_event_sequence": self._next_event_sequence,
        }

    def _restore_authenticated_checkpoint(self, payload: Mapping[str, Any]) -> None:
        if self._state is not None or self._score_cache or self._next_event_sequence:
            raise ValueError("paper fast loop restore requires a fresh instance")
        if type(payload) is not dict or set(payload) != {
            "state",
            "score_cache",
            "next_event_sequence",
        }:
            raise ValueError("invalid fast-loop checkpoint fields")
        state_payload = payload["state"]
        if type(state_payload) is not dict or set(state_payload) != {
            "epoch",
            "step",
            "current_skill",
            "current_score",
            "best_skill",
            "best_score",
            "meta_skill",
        }:
            raise ValueError("invalid paper state checkpoint")
        cache_payload = payload["score_cache"]
        if type(cache_payload) is not dict:
            raise ValueError("invalid score cache checkpoint")
        score_cache: dict[str, SelectionScore] = {}
        for skill_hash, raw_score in cache_payload.items():
            if type(skill_hash) is not str or len(skill_hash) != 64:
                raise ValueError("invalid score cache skill hash")
            if type(raw_score) not in {int, float}:
                raise ValueError("invalid score cache scalar")
            score_cache[skill_hash] = SelectionScore(float(raw_score))
        if type(payload["next_event_sequence"]) is not int or payload[
            "next_event_sequence"
        ] < 0:
            raise ValueError("invalid checkpoint event sequence")
        state = PaperState(
            epoch=state_payload["epoch"],
            step=state_payload["step"],
            current_skill=state_payload["current_skill"],
            current_score=SelectionScore(float(state_payload["current_score"])),
            best_skill=state_payload["best_skill"],
            best_score=SelectionScore(float(state_payload["best_score"])),
            meta_skill=state_payload["meta_skill"],
        )
        if score_cache.get(_sha256(state.current_skill)) != state.current_score:
            raise ValueError("checkpoint cache disagrees with current state")
        if score_cache.get(_sha256(state.best_skill)) != state.best_score:
            raise ValueError("checkpoint cache disagrees with best state")
        self._state = state
        self._score_cache = score_cache
        self._next_event_sequence = payload["next_event_sequence"]

    def _preview_external_candidate(
        self,
        candidate_skill: str,
    ) -> _ExternalGatePreview:
        state = self.state
        if type(candidate_skill) is not str or not candidate_skill.strip():
            raise ValueError("external candidate requires skill text")
        working_cache = dict(self._score_cache)
        candidate_hash = _sha256(candidate_skill)
        candidate_score = working_cache.get(candidate_hash)
        cache_hit = candidate_score is not None
        skipped_reason = None
        skipped_message = None
        if candidate_score is None:
            try:
                self._controller.validate_skill(candidate_skill)
            except SkillContractViolation as error:
                candidate_score = state.current_score
                skipped_reason = error.code
                skipped_message = str(error)
            else:
                candidate_score = self._controller.selection.score(candidate_skill)
                working_cache[candidate_hash] = candidate_score
        decision = (
            strict_selection_decision(
                current=state.current_score,
                candidate=candidate_score,
            )
            if skipped_reason is None
            else None
        )
        return _ExternalGatePreview(
            input_state=state,
            candidate_skill_sha256=candidate_hash,
            candidate_score=candidate_score,
            accepted=decision.accepted if decision is not None else False,
            delta=decision.delta if decision is not None else 0.0,
            cache_hit=cache_hit,
            score_cache=tuple(sorted(working_cache.items())),
            selection_skipped_reason=skipped_reason,
            selection_skipped_message=skipped_message,
        )

    def _commit_external_candidate(
        self,
        candidate_skill: str,
        *,
        source: str,
        proposal_event_type: AlgorithmEventType,
        proposal_payload: Mapping[str, Any],
        preview: _ExternalGatePreview,
    ) -> ExternalGateResult:
        if type(preview) is not _ExternalGatePreview:
            raise ValueError("external candidate commit requires exact preview")
        state = self.state
        if (
            state != preview.input_state
            or _sha256(candidate_skill) != preview.candidate_skill_sha256
        ):
            raise ValueError("external candidate changed after selection preview")
        if type(source) is not str or not source.strip():
            raise ValueError("external candidate requires source")
        if proposal_event_type is not AlgorithmEventType.SLOW_UPDATE_PROPOSED:
            raise ValueError("unsupported external candidate proposal event")
        candidate_hash = preview.candidate_skill_sha256
        candidate_score = preview.candidate_score
        events: list[AlgorithmEvent] = [
            AlgorithmEvent(
                sequence=self._next_event_sequence,
                event_type=proposal_event_type,
                epoch=state.epoch,
                step=None,
                payload=dict(proposal_payload),
            )
        ]
        if preview.selection_skipped_reason is None:
            self._append_event(
                events,
                AlgorithmEventType.SELECTION_SCORED,
                state,
                state.step,
                {
                    "candidate_skill_sha256": candidate_hash,
                    "candidate_score": candidate_score.value,
                    "current_score": state.current_score.value,
                    "cache_hit": preview.cache_hit,
                    "source": source,
                },
            )
        if preview.accepted:
            best_skill = state.best_skill
            best_score = state.best_score
            if candidate_score.value > best_score.value:
                best_skill = candidate_skill
                best_score = candidate_score
            next_state = PaperState(
                epoch=state.epoch,
                step=state.step,
                current_skill=candidate_skill,
                current_score=candidate_score,
                best_skill=best_skill,
                best_score=best_score,
                meta_skill=state.meta_skill,
            )
            event_type = AlgorithmEventType.CANDIDATE_ACCEPTED
        else:
            next_state = state
            event_type = AlgorithmEventType.CANDIDATE_REJECTED
        self._append_event(
            events,
            event_type,
            state,
            state.step,
            {
                "candidate_skill_sha256": candidate_hash,
                "delta": (
                    None
                    if preview.selection_skipped_reason is not None
                    else preview.delta
                ),
                "source": source,
                **(
                    {
                        "reason": "skill_contract_violation",
                        "violation_code": preview.selection_skipped_reason,
                        "violation_message": preview.selection_skipped_message,
                    }
                    if preview.selection_skipped_reason is not None
                    else {}
                ),
            },
        )
        self._score_cache = dict(preview.score_cache)
        self._state = next_state
        self._next_event_sequence += len(events)
        return ExternalGateResult(
            state=next_state,
            candidate_score=candidate_score,
            accepted=preview.accepted,
            delta=preview.delta,
            cache_hit=preview.cache_hit,
            events=tuple(events),
            selection_skipped_reason=preview.selection_skipped_reason,
        )

    def _update_meta_skill(self, meta_skill: str) -> PaperState:
        state = self.state
        if type(meta_skill) is not str or not meta_skill.strip():
            raise ValueError("meta skill must be non-empty")
        self._state = PaperState(
            epoch=state.epoch,
            step=state.step,
            current_skill=state.current_skill,
            current_score=state.current_score,
            best_skill=state.best_skill,
            best_score=state.best_score,
            meta_skill=meta_skill,
        )
        return self._state

    def run_step(
        self,
        *,
        train_evidence: TrainEvidenceBatch,
        edit_budget: int,
    ) -> FastStepResult:
        """Execute the default one-batch fast path."""

        if type(train_evidence) is not TrainEvidenceBatch:
            raise ValueError("run_step requires exact TrainEvidenceBatch")
        return self._run_step(
            train_evidence=StepTrainEvidence((train_evidence,)),
            analysis_budget=edit_budget,
            edit_budget=edit_budget,
        )

    def _run_accumulated_step(
        self,
        *,
        train_evidence: StepTrainEvidence,
        analysis_budget: int,
        edit_budget: int | None,
    ) -> FastStepResult:
        """Execute one update from separately reflected accumulation batches."""

        return self._run_step(
            train_evidence=train_evidence,
            analysis_budget=analysis_budget,
            edit_budget=edit_budget,
        )

    def _run_step(
        self,
        *,
        train_evidence: StepTrainEvidence,
        analysis_budget: int,
        edit_budget: int | None,
    ) -> FastStepResult:
        """Execute collect-to-gate once and commit cache/sequence atomically."""

        self._controller.__post_init__()
        if self._state is None:
            raise ValueError("paper fast loop must be initialized")
        state = self._state
        with self._exchange_lock:
            self._active_exchanges = []
        if type(train_evidence) is not StepTrainEvidence:
            raise ValueError("accumulated step requires exact StepTrainEvidence")
        train_evidence.__post_init__()
        if (
            type(analysis_budget) is not int
            or not self._profile.learning_rate_floor
            <= analysis_budget
            <= self._profile.learning_rate
        ):
            raise ValueError(
                "analysis_budget must be within the frozen learning-rate range"
            )
        if self._mechanisms.learning_rate_schedule == "autonomous":
            if edit_budget is not None:
                raise ValueError("autonomous steps cannot receive a fixed edit budget")
        elif (
            type(edit_budget) is not int
            or not 0 <= edit_budget <= analysis_budget
        ):
            raise ValueError("edit_budget must be within the analysis budget")
        if self._scheduled_batches:
            if len(train_evidence.batches) != len(self._scheduled_batches):
                raise ValueError("step evidence does not match accumulation plan")
            scheduled = self._scheduled_batches
        else:
            if len(train_evidence.batches) != 1:
                raise ValueError("unscheduled fast loop accepts one train batch")
            scheduled = (None,)

        step = state.step + 1
        call_prefix = f"e{state.epoch}-s{step}"
        working_cache = dict(self._score_cache)
        current_hash = _sha256(state.current_skill)
        cached_current = working_cache.get(current_hash)
        if cached_current is None:
            raise ValueError(
                "paper state current score was not initialized by selection owner"
            )
        if cached_current != state.current_score:
            raise ValueError("score cache disagrees with current paper state")

        events: list[AlgorithmEvent] = []
        self._append_event(
            events,
            AlgorithmEventType.STEP_STARTED,
            state,
            step,
            {
                "edit_budget": edit_budget,
                "analysis_budget": analysis_budget,
                "accumulation": len(train_evidence.batches),
                "current_skill_sha256": current_hash,
                **(
                    {"train_batch_id": scheduled[0].batch_id}
                    if scheduled[0] is not None
                    else {}
                ),
                **(
                    {
                        "train_batch_seed": scheduled[0].batch_seed,
                        "train_batch_size": scheduled[0].batch_size,
                        "train_batch_ids": [
                            batch.batch_id for batch in scheduled
                        ],
                    }
                    if scheduled[0] is not None
                    else {}
                ),
            },
        )
        observed_failure_patterns: list[ObservedFailurePattern] = []
        failure_proposals: list[ParsedUpdateResponse] = []
        success_proposals: list[ParsedUpdateResponse] = []
        rollout_scores: list[float] = []
        for accumulation_index, (batch_evidence, batch_cursor) in enumerate(
            zip(train_evidence.batches, scheduled),
            1,
        ):
            trajectories = self._controller.train.verify(
                batch_evidence,
                current_skill=state.current_skill,
                batch_id=(
                    batch_cursor.batch_id if batch_cursor is not None else None
                ),
                batch_seed=(
                    batch_cursor.batch_seed if batch_cursor is not None else None
                ),
                batch_size=(
                    batch_cursor.batch_size if batch_cursor is not None else None
                ),
            )
            failures = tuple(
                item for item in trajectories if not item["success"]
            )
            successes = tuple(item for item in trajectories if item["success"])
            rollout_scores.extend(float(item["score"]) for item in trajectories)
            self._append_event(
                events,
                AlgorithmEventType.ROLLOUT_COLLECTED,
                state,
                step,
                {
                    "trajectory_count": len(trajectories),
                    "failure_count": len(failures),
                    "success_count": len(successes),
                    "train_split_id": batch_evidence.split_id,
                    "accumulation_index": accumulation_index,
                    "accumulation": len(train_evidence.batches),
                    **(
                        {
                            "train_batch_id": batch_cursor.batch_id,
                            "train_batch_seed": batch_cursor.batch_seed,
                            "train_batch_size": batch_cursor.batch_size,
                        }
                        if batch_cursor is not None
                        else {}
                    ),
                },
            )
            batch_call_prefix = (
                call_prefix
                if len(train_evidence.batches) == 1
                else f"{call_prefix}-a{accumulation_index}"
            )
            failure_proposals.extend(
                self._reflect_group(
                    state=state,
                    step=step,
                    source=PaperEditSource.FAILURE,
                    trajectories=failures,
                    train_evidence=train_evidence,
                    batch_cursor=batch_cursor,
                    edit_budget=analysis_budget,
                    call_prefix=batch_call_prefix,
                    events=events,
                    observed_failure_patterns=observed_failure_patterns,
                )
            )
            success_proposals.extend(
                self._reflect_group(
                    state=state,
                    step=step,
                    source=PaperEditSource.SUCCESS,
                    trajectories=successes,
                    train_evidence=train_evidence,
                    batch_cursor=batch_cursor,
                    edit_budget=analysis_budget,
                    call_prefix=batch_call_prefix,
                    events=events,
                    observed_failure_patterns=observed_failure_patterns,
                )
            )

        try:
            failure_merge = self._hierarchical_merge(
                source=PaperEditSource.FAILURE,
                proposals=tuple(failure_proposals),
                call_prefix=call_prefix,
                state=state,
                step=step,
                train_evidence=train_evidence,
                edit_budget=analysis_budget,
                events=events,
            )
            success_merge = self._hierarchical_merge(
                source=PaperEditSource.SUCCESS,
                proposals=tuple(success_proposals),
                call_prefix=call_prefix,
                state=state,
                step=step,
                train_evidence=train_evidence,
                edit_budget=analysis_budget,
                events=events,
            )
            final_merge = self._merge(
                stage=OptimizerStage.MERGE_FINAL_FAILURE_PRIORITIZED,
                call_id=f"{call_prefix}-merge-final",
                state=state,
                step=step,
                train_evidence=train_evidence,
                edit_budget=analysis_budget,
                prompt_payload={
                    "current_skill": state.current_skill,
                    "failure_patch": _update_payload(failure_merge),
                    "success_patch": _update_payload(success_merge),
                    "meta_skill": state.meta_skill,
                },
                edit_id_prefix=f"{call_prefix}-merge-final-edit",
                event_type=AlgorithmEventType.MERGE_FINAL_FAILURE_PRIORITIZED,
                hierarchy_level=1,
                batch_index=1,
                events=events,
            )
            merged_items = _update_items(final_merge)
            if edit_budget is None:
                edit_budget = self._decide_learning_rate(
                    call_id=f"{call_prefix}-autonomous-lr",
                    state=state,
                    step=step,
                    train_evidence=train_evidence,
                    candidates=merged_items,
                    rollout_scores=tuple(rollout_scores),
                    events=events,
                )
            ranked_updates = self._rank(
                call_id=f"{call_prefix}-rank-top-l",
                state=state,
                step=step,
                train_evidence=train_evidence,
                edit_budget=edit_budget,
                candidates=merged_items,
                events=events,
            )
        except _SemanticStageExhausted as error:
            return self._skip_failed_semantic_stage(
                state=state,
                step=step,
                error=error,
                events=events,
                working_cache=working_cache,
                failure_patterns=tuple(observed_failure_patterns),
            )

        ranked_edits: tuple[PaperEdit, ...] = ()
        ranked_suggestions: tuple[PaperSuggestion, ...] = ()
        rewrite_result: RewriteApplyResult | None = None
        if self._mechanisms.update_mode == "patch":
            if any(type(item) is not PaperEdit for item in ranked_updates):
                raise RuntimeError("patch ranking returned non-edit items")
            ranked_edits = tuple(ranked_updates)
            apply_result: PatchApplyResult | RewriteApplyResult = (
                apply_paper_patch(state.current_skill, ranked_edits)
            )
            self._append_event(
                events,
                AlgorithmEventType.PATCH_APPLIED,
                state,
                step,
                {
                    "input_skill_sha256": apply_result.input_sha256,
                    "candidate_skill_sha256": apply_result.output_sha256,
                    "reports": [
                        {
                            "index": report.index,
                            "edit_id": report.edit_id,
                            "operation": report.operation.value,
                            "status": report.status,
                            "before_sha256": report.before_sha256,
                            "after_sha256": report.after_sha256,
                        }
                        for report in apply_result.reports
                    ],
                },
            )
        else:
            if any(type(item) is not PaperSuggestion for item in ranked_updates):
                raise RuntimeError("rewrite ranking returned non-suggestion items")
            ranked_suggestions = tuple(ranked_updates)
            if ranked_suggestions:
                rewrite_result = self._rewrite_skill(
                    call_id=f"{call_prefix}-rewrite-skill",
                    state=state,
                    step=step,
                    train_evidence=train_evidence,
                    suggestions=ranked_suggestions,
                )
                apply_result = rewrite_result
                self._append_event(
                    events,
                    AlgorithmEventType.REWRITE_APPLIED,
                    state,
                    step,
                    {
                        "input_skill_sha256": apply_result.input_sha256,
                        "candidate_skill_sha256": apply_result.output_sha256,
                        "suggestion_ids": [
                            item.suggestion_id for item in ranked_suggestions
                        ],
                        "change_summary": list(apply_result.change_summary),
                    },
                )
            else:
                apply_result = apply_paper_patch(state.current_skill, ())
                self._append_event(
                    events,
                    AlgorithmEventType.REWRITE_SKIPPED,
                    state,
                    step,
                    {
                        "reason": "no_ranked_suggestions",
                        "candidate_skill_sha256": apply_result.output_sha256,
                    },
                )

        candidate_hash = apply_result.output_sha256
        candidate_score = working_cache.get(candidate_hash)
        cache_hit = candidate_score is not None
        if candidate_score is None:
            try:
                self._controller.validate_skill(apply_result.output_skill)
            except SkillContractViolation as error:
                return self._reject_contract_candidate(
                    state=state,
                    step=step,
                    error=error,
                    events=events,
                    working_cache=working_cache,
                    apply_result=apply_result,
                    ranked_edits=ranked_edits,
                    ranked_suggestions=ranked_suggestions,
                    rewrite_result=rewrite_result,
                    failure_patterns=tuple(observed_failure_patterns),
                )
            candidate_score = self._controller.selection.score(apply_result.output_skill)
            working_cache[candidate_hash] = candidate_score
        decision = strict_selection_decision(
            current=state.current_score,
            candidate=candidate_score,
        )
        self._append_event(
            events,
            AlgorithmEventType.SELECTION_SCORED,
            state,
            step,
            {
                "candidate_skill_sha256": candidate_hash,
                "candidate_score": candidate_score.value,
                "current_score": state.current_score.value,
                "cache_hit": cache_hit,
            },
        )

        if decision.accepted:
            best_skill = state.best_skill
            best_score = state.best_score
            if candidate_score.value > state.best_score.value:
                best_skill = apply_result.output_skill
                best_score = candidate_score
            next_state = PaperState(
                epoch=state.epoch,
                step=step,
                current_skill=apply_result.output_skill,
                current_score=candidate_score,
                best_skill=best_skill,
                best_score=best_score,
                meta_skill=state.meta_skill,
            )
            decision_event = AlgorithmEventType.CANDIDATE_ACCEPTED
        else:
            next_state = PaperState(
                epoch=state.epoch,
                step=step,
                current_skill=state.current_skill,
                current_score=state.current_score,
                best_skill=state.best_skill,
                best_score=state.best_score,
                meta_skill=state.meta_skill,
            )
            decision_event = AlgorithmEventType.CANDIDATE_REJECTED
        self._append_event(
            events,
            decision_event,
            state,
            step,
            {
                "candidate_skill_sha256": candidate_hash,
                "delta": decision.delta,
            },
        )

        result = FastStepResult(
            input_skill=state.current_skill,
            state=next_state,
            candidate_score=candidate_score,
            ranked_edits=ranked_edits,
            apply_result=apply_result,
            events=tuple(events),
            failure_patterns=tuple(observed_failure_patterns),
            ranked_suggestions=ranked_suggestions,
            rewrite_result=rewrite_result,
            optimizer_exchanges=self._take_optimizer_exchanges(),
        )
        result.replay()
        self._score_cache = working_cache
        self._next_event_sequence += len(events)
        self._state = next_state
        return result

    def _reflect_group(
        self,
        *,
        state: PaperState,
        step: int,
        source: PaperEditSource,
        trajectories: tuple[dict[str, Any], ...],
        train_evidence: StepTrainEvidence,
        batch_cursor: EpochBatchCursor | None,
        edit_budget: int,
        call_prefix: str,
        events: list[AlgorithmEvent],
        observed_failure_patterns: list[ObservedFailurePattern],
    ) -> tuple[ParsedUpdateResponse, ...]:
        stage = (
            OptimizerStage.REFLECT_FAILURE
            if source is PaperEditSource.FAILURE
            else OptimizerStage.REFLECT_SUCCESS
        )
        event_type = (
            AlgorithmEventType.FAILURE_REFLECTED
            if source is PaperEditSource.FAILURE
            else AlgorithmEventType.SUCCESS_REFLECTED
        )
        minibatches = tuple(
            trajectories[start : start + self._profile.reflection_minibatch_size]
            for start in range(
                0,
                len(trajectories),
                self._profile.reflection_minibatch_size,
            )
        )

        def analyze(
            batch_index: int,
            minibatch: tuple[dict[str, Any], ...],
        ) -> tuple[
            ParsedUpdateResponse,
            tuple[ObservedFailurePattern, ...],
            tuple[tuple[AlgorithmEventType, Mapping[str, Any]], ...],
        ]:
            event_payloads: list[
                tuple[AlgorithmEventType, Mapping[str, Any]]
            ] = []
            call_id = f"{call_prefix}-reflect-{source.value}-b{batch_index}"
            response = self._complete(
                call_id=call_id,
                stage=stage,
                prompt_payload={
                    "current_skill": state.current_skill,
                    "trajectories": list(minibatch),
                    "edit_budget": edit_budget,
                    "meta_skill": state.meta_skill,
                },
                response_schema=optimizer_response_schema(
                    stage,
                    edit_budget=edit_budget,
                    update_mode=self._mechanisms.update_mode,
                ),
                train_evidence=train_evidence,
                batch_cursor=batch_cursor,
            )
            if self._mechanisms.update_mode == "patch":
                parsed: ParsedUpdateResponse = parse_patch_response(
                    stage=stage,
                    payload=response.payload,
                    edit_budget=edit_budget,
                    edit_id_prefix=f"{call_id}-edit",
                    expected_batch_size=len(minibatch),
                )
            else:
                parsed = parse_suggestion_response(
                    stage=stage,
                    payload=response.payload,
                    edit_budget=edit_budget,
                    suggestion_id_prefix=f"{call_id}-suggestion",
                    expected_batch_size=len(minibatch),
                )
            failure_patterns = parsed.failure_patterns
            event_payloads.append(
                (
                    event_type,
                    {
                    "call_id": call_id,
                    "batch_index": batch_index,
                    "batch_size": len(minibatch),
                    "edit_count": len(_update_items(parsed)),
                    },
                )
            )
            for round_number in range(1, self._profile.max_analyst_rounds + 1):
                refine_call_id = (
                    f"{call_prefix}-refine-{source.value}-b{batch_index}"
                    f"-r{round_number}"
                )
                response = self._complete(
                    call_id=refine_call_id,
                    stage=OptimizerStage.REFINE,
                    prompt_payload={
                        "current_skill": state.current_skill,
                        "trajectories": list(minibatch),
                        "source_type": source.value,
                        "prior_patch": _update_payload(parsed),
                        "round": round_number,
                        "max_rounds": self._profile.max_analyst_rounds,
                        "edit_budget": edit_budget,
                        "meta_skill": state.meta_skill,
                    },
                    response_schema=optimizer_response_schema(
                        OptimizerStage.REFINE,
                        edit_budget=edit_budget,
                        update_mode=self._mechanisms.update_mode,
                    ),
                    train_evidence=train_evidence,
                    batch_cursor=batch_cursor,
                )
                if self._mechanisms.update_mode == "patch":
                    parsed = parse_patch_response(
                        stage=OptimizerStage.REFINE,
                        payload=response.payload,
                        edit_budget=edit_budget,
                        edit_id_prefix=f"{refine_call_id}-edit",
                        source_type=source,
                    )
                else:
                    parsed = parse_suggestion_response(
                        stage=OptimizerStage.REFINE,
                        payload=response.payload,
                        edit_budget=edit_budget,
                        suggestion_id_prefix=(
                            f"{refine_call_id}-suggestion"
                        ),
                        source_type=source,
                    )
                event_payloads.append(
                    (
                        AlgorithmEventType.ANALYST_REFINED,
                        {
                        "call_id": refine_call_id,
                        "source_type": source.value,
                        "batch_index": batch_index,
                        "round": round_number,
                        "edit_count": len(_update_items(parsed)),
                        "converged": parsed.converged,
                        },
                    )
                )
                if parsed.converged:
                    break
            return parsed, failure_patterns, tuple(event_payloads)

        jobs = tuple(enumerate(minibatches, 1))
        if len(jobs) > 1 and self._mechanisms.analyst_workers > 1:
            with ThreadPoolExecutor(
                max_workers=min(self._mechanisms.analyst_workers, len(jobs)),
                thread_name_prefix="paper-analyst",
            ) as executor:
                futures = [
                    executor.submit(analyze, batch_index, minibatch)
                    for batch_index, minibatch in jobs
                ]
                results = [future.result() for future in futures]
        else:
            results = [
                analyze(batch_index, minibatch)
                for batch_index, minibatch in jobs
            ]

        proposals: list[ParsedUpdateResponse] = []
        for parsed, failure_patterns, event_payloads in results:
            observed_failure_patterns.extend(failure_patterns)
            for reflected_event, payload in event_payloads:
                self._append_event(
                    events,
                    reflected_event,
                    state,
                    step,
                    payload,
                )
            proposals.append(parsed)
        return tuple(proposals)

    def _hierarchical_merge(
        self,
        *,
        source: PaperEditSource,
        proposals: tuple[ParsedUpdateResponse, ...],
        call_prefix: str,
        state: PaperState,
        step: int,
        train_evidence: StepTrainEvidence,
        edit_budget: int,
        events: list[AlgorithmEvent],
    ) -> ParsedUpdateResponse:
        stage = (
            OptimizerStage.MERGE_FAILURE
            if source is PaperEditSource.FAILURE
            else OptimizerStage.MERGE_SUCCESS
        )
        event_type = (
            AlgorithmEventType.MERGE_FAILURE
            if source is PaperEditSource.FAILURE
            else AlgorithmEventType.MERGE_SUCCESS
        )
        current = proposals
        level = 1
        while True:
            batches = tuple(
                current[start : start + self._profile.merge_batch_size]
                for start in range(0, len(current), self._profile.merge_batch_size)
            )
            if not batches:
                batches = ((),)
            merged: list[ParsedUpdateResponse] = []
            for batch_index, batch in enumerate(batches, 1):
                call_id = (
                    f"{call_prefix}-merge-{source.value}"
                    f"-l{level}-b{batch_index}"
                )
                merged.append(
                    self._merge(
                        stage=stage,
                        call_id=call_id,
                        state=state,
                        step=step,
                        train_evidence=train_evidence,
                        edit_budget=edit_budget,
                        prompt_payload={
                            "current_skill": state.current_skill,
                            "source_type": source.value,
                            "patches": [_update_payload(item) for item in batch],
                            "meta_skill": state.meta_skill,
                        },
                        edit_id_prefix=f"{call_id}-edit",
                        event_type=event_type,
                        hierarchy_level=level,
                        batch_index=batch_index,
                        events=events,
                    )
                )
            if len(merged) == 1:
                return merged[0]
            current = tuple(merged)
            level += 1

    def _merge(
        self,
        *,
        stage: OptimizerStage,
        call_id: str,
        state: PaperState,
        step: int,
        train_evidence: StepTrainEvidence,
        edit_budget: int,
        prompt_payload: Mapping[str, Any],
        edit_id_prefix: str,
        event_type: AlgorithmEventType,
        hierarchy_level: int,
        batch_index: int,
        events: list[AlgorithmEvent],
    ) -> ParsedUpdateResponse:
        last_error: OptimizerContractViolation | None = None
        for attempt in range(1, self._retry_policy.max_semantic_attempts + 1):
            attempt_call_id = _attempt_call_id(call_id, attempt)
            try:
                response = self._complete(
                    call_id=attempt_call_id,
                    stage=stage,
                    prompt_payload=prompt_payload,
                    response_schema=optimizer_response_schema(
                        stage,
                        edit_budget=edit_budget,
                        update_mode=self._mechanisms.update_mode,
                    ),
                    train_evidence=train_evidence,
                    semantic_attempt=attempt,
                )
                response_id_prefix = (
                    edit_id_prefix
                    if attempt == 1
                    else f"{edit_id_prefix}-retry-{attempt - 1}"
                )
                if self._mechanisms.update_mode == "patch":
                    parsed: ParsedUpdateResponse = parse_patch_response(
                        stage=stage,
                        payload=response.payload,
                        edit_budget=edit_budget,
                        edit_id_prefix=response_id_prefix,
                    )
                else:
                    parsed = parse_suggestion_response(
                        stage=stage,
                        payload=response.payload,
                        edit_budget=edit_budget,
                        suggestion_id_prefix=response_id_prefix,
                    )
            except OptimizerContractViolation as error:
                last_error = error
                continue
            self._append_event(
                events,
                event_type,
                state,
                step,
                {
                    "call_id": attempt_call_id,
                    "edit_count": len(_update_items(parsed)),
                    "hierarchy_level": hierarchy_level,
                    "batch_index": batch_index,
                    "semantic_attempts": attempt,
                    "retry_policy_id": self._retry_policy.policy_id,
                },
            )
            return parsed
        assert last_error is not None
        raise _SemanticStageExhausted(
            stage=stage,
            attempts=self._retry_policy.max_semantic_attempts,
            last_error=last_error,
        )

    def _decide_learning_rate(
        self,
        *,
        call_id: str,
        state: PaperState,
        step: int,
        train_evidence: StepTrainEvidence,
        candidates: tuple[PaperEdit, ...] | tuple[PaperSuggestion, ...],
        rollout_scores: tuple[float, ...],
        events: list[AlgorithmEvent],
    ) -> int:
        candidate_count = len(candidates)
        if candidate_count == 0:
            self._append_event(
                events,
                AlgorithmEventType.LEARNING_RATE_DECIDED,
                state,
                step,
                {
                    "call_id": None,
                    "candidate_count": 0,
                    "raw_learning_rate": 0,
                    "learning_rate": 0,
                    "reason": "no_candidates",
                },
            )
            return 0
        rollout_n = len(rollout_scores)
        response = self._complete(
            call_id=call_id,
            stage=OptimizerStage.DECIDE_LEARNING_RATE,
            prompt_payload={
                "current_skill": state.current_skill,
                "update_mode": self._mechanisms.update_mode,
                "update_items": [_update_item_payload(item) for item in candidates],
                "candidate_count": candidate_count,
                "rollout": {
                    "count": rollout_n,
                    "hard": (
                        sum(score >= 1.0 for score in rollout_scores)
                        / rollout_n
                        if rollout_n
                        else 0.0
                    ),
                    "soft": (
                        sum(rollout_scores) / rollout_n if rollout_n else 0.0
                    ),
                },
                "meta_skill": state.meta_skill,
            },
            response_schema=learning_rate_response_schema(),
            train_evidence=train_evidence,
        )
        parsed = parse_learning_rate_response(
            payload=response.payload,
            candidate_count=candidate_count,
        )
        self._append_event(
            events,
            AlgorithmEventType.LEARNING_RATE_DECIDED,
            state,
            step,
            {
                "call_id": call_id,
                "candidate_count": candidate_count,
                "raw_learning_rate": parsed.raw_learning_rate,
                "learning_rate": parsed.learning_rate,
                "confidence": parsed.confidence,
                "risk_notes": list(parsed.risk_notes),
            },
        )
        return parsed.learning_rate

    def _rank(
        self,
        *,
        call_id: str,
        state: PaperState,
        step: int,
        train_evidence: StepTrainEvidence,
        edit_budget: int,
        candidates: tuple[PaperEdit, ...] | tuple[PaperSuggestion, ...],
        events: list[AlgorithmEvent],
    ) -> tuple[PaperEdit, ...] | tuple[PaperSuggestion, ...]:
        schema = optimizer_response_schema(
            OptimizerStage.RANK_TOP_L,
            edit_budget=edit_budget,
            candidate_count=len(candidates),
            update_mode=self._mechanisms.update_mode,
        )
        last_error: OptimizerContractViolation | None = None
        for attempt in range(1, self._retry_policy.max_semantic_attempts + 1):
            attempt_call_id = _attempt_call_id(call_id, attempt)
            try:
                response = self._complete(
                    call_id=attempt_call_id,
                    stage=OptimizerStage.RANK_TOP_L,
                    prompt_payload={
                        "current_skill": state.current_skill,
                        (
                            "edits"
                            if self._mechanisms.update_mode == "patch"
                            else "revise_suggestions"
                        ): [
                            _update_item_payload(item) for item in candidates
                        ],
                        "edit_budget": edit_budget,
                        "meta_skill": state.meta_skill,
                    },
                    response_schema=schema,
                    train_evidence=train_evidence,
                    semantic_attempt=attempt,
                )
                if self._mechanisms.update_mode == "patch":
                    ranked = parse_rank_response(
                        payload=response.payload,
                        candidates=tuple(
                            item
                            for item in candidates
                            if type(item) is PaperEdit
                        ),
                        edit_budget=edit_budget,
                    )
                else:
                    ranked = parse_suggestion_rank_response(
                        payload=response.payload,
                        candidates=tuple(
                            item
                            for item in candidates
                            if type(item) is PaperSuggestion
                        ),
                        edit_budget=edit_budget,
                    )
            except OptimizerContractViolation as error:
                last_error = error
                continue
            self._append_event(
                events,
                AlgorithmEventType.RANK_TOP_L,
                state,
                step,
                {
                    "call_id": attempt_call_id,
                    "candidate_count": len(candidates),
                    "selected_count": len(ranked),
                    "selected_edit_ids": [
                        (
                            item.edit_id
                            if type(item) is PaperEdit
                            else item.suggestion_id
                        )
                        for item in ranked
                    ],
                    "semantic_attempts": attempt,
                    "retry_policy_id": self._retry_policy.policy_id,
                },
            )
            return ranked
        assert last_error is not None
        raise _SemanticStageExhausted(
            stage=OptimizerStage.RANK_TOP_L,
            attempts=self._retry_policy.max_semantic_attempts,
            last_error=last_error,
        )

    def _rewrite_skill(
        self,
        *,
        call_id: str,
        state: PaperState,
        step: int,
        train_evidence: StepTrainEvidence,
        suggestions: tuple[PaperSuggestion, ...],
    ) -> RewriteApplyResult:
        response = self._complete(
            call_id=call_id,
            stage=OptimizerStage.REWRITE_SKILL,
            prompt_payload={
                "current_skill": state.current_skill,
                "selected_revise_suggestions": [
                    _suggestion_payload(item) for item in suggestions
                ],
                "meta_skill": state.meta_skill,
            },
            response_schema=rewrite_response_schema(),
            train_evidence=train_evidence,
        )
        parsed = parse_rewrite_response(response.payload)
        try:
            return apply_paper_rewrite(
                state.current_skill,
                suggestions,
                new_skill=parsed.new_skill,
                reasoning=parsed.reasoning,
                change_summary=parsed.change_summary,
            )
        except ValueError as error:
            raise OptimizerContractViolation(
                f"invalid full-skill rewrite: {error}"
            ) from error

    def _skip_failed_semantic_stage(
        self,
        *,
        state: PaperState,
        step: int,
        error: _SemanticStageExhausted,
        events: list[AlgorithmEvent],
        working_cache: dict[str, SelectionScore],
        failure_patterns: tuple[ObservedFailurePattern, ...],
    ) -> FastStepResult:
        apply_result = apply_paper_patch(state.current_skill, ())
        next_state = PaperState(
            epoch=state.epoch,
            step=step,
            current_skill=state.current_skill,
            current_score=state.current_score,
            best_skill=state.best_skill,
            best_score=state.best_score,
            meta_skill=state.meta_skill,
        )
        self._append_event(
            events,
            AlgorithmEventType.CANDIDATE_REJECTED,
            state,
            step,
            {
                "candidate_skill_sha256": apply_result.output_sha256,
                "delta": 0.0,
                "reason": "semantic_stage_exhausted",
                "failed_stage": error.stage.value,
                "attempts": error.attempts,
                "retry_policy_id": self._retry_policy.policy_id,
            },
        )
        result = FastStepResult(
            input_skill=state.current_skill,
            state=next_state,
            candidate_score=state.current_score,
            ranked_edits=(),
            apply_result=apply_result,
            events=tuple(events),
            failure_patterns=failure_patterns,
            skipped_stage=error.stage,
            optimizer_exchanges=self._take_optimizer_exchanges(),
        )
        result.replay()
        self._score_cache = working_cache
        self._next_event_sequence += len(events)
        self._state = next_state
        return result

    def _reject_contract_candidate(
        self,
        *,
        state: PaperState,
        step: int,
        error: SkillContractViolation,
        events: list[AlgorithmEvent],
        working_cache: dict[str, SelectionScore],
        apply_result: PatchApplyResult | RewriteApplyResult,
        ranked_edits: tuple[PaperEdit, ...],
        ranked_suggestions: tuple[PaperSuggestion, ...],
        rewrite_result: RewriteApplyResult | None,
        failure_patterns: tuple[ObservedFailurePattern, ...],
    ) -> FastStepResult:
        next_state = PaperState(
            epoch=state.epoch,
            step=step,
            current_skill=state.current_skill,
            current_score=state.current_score,
            best_skill=state.best_skill,
            best_score=state.best_score,
            meta_skill=state.meta_skill,
        )
        self._append_event(
            events,
            AlgorithmEventType.CANDIDATE_REJECTED,
            state,
            step,
            {
                "candidate_skill_sha256": apply_result.output_sha256,
                "delta": None,
                "reason": "skill_contract_violation",
                "violation_code": error.code,
                "violation_message": str(error),
            },
        )
        result = FastStepResult(
            input_skill=state.current_skill,
            state=next_state,
            candidate_score=state.current_score,
            ranked_edits=ranked_edits,
            apply_result=apply_result,
            events=tuple(events),
            failure_patterns=failure_patterns,
            ranked_suggestions=ranked_suggestions,
            rewrite_result=rewrite_result,
            optimizer_exchanges=self._take_optimizer_exchanges(),
            selection_skipped_reason=error.code,
        )
        result.replay()
        self._score_cache = working_cache
        self._next_event_sequence += len(events)
        self._state = next_state
        return result

    def _complete(
        self,
        *,
        call_id: str,
        stage: OptimizerStage,
        prompt_payload: Mapping[str, Any],
        response_schema: Mapping[str, Any],
        train_evidence: StepTrainEvidence,
        batch_cursor: EpochBatchCursor | None = None,
        semantic_attempt: int = 1,
    ) -> OptimizerResponse:
        try:
            prompt = json.dumps(
                {
                    **prompt_payload,
                    **(
                        {
                            "immutable_skill_contract": (
                                self._controller.skill_contract_description
                            )
                        }
                        if self._controller.skill_contract_description is not None
                        else {}
                    ),
                    "epoch_buffer": [
                        record.to_optimizer_payload()
                        for record in self._epoch_buffer
                    ],
                },
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        except (TypeError, ValueError) as error:
            raise OptimizerContractViolation(
                f"fast-loop optimizer prompt is not JSON-safe: {error}"
            ) from error
        authority = train_evidence.batches[0]
        scheduled = self._scheduled_batches
        metadata_batch = batch_cursor or (scheduled[0] if scheduled else None)
        request = OptimizerRequest(
            call_id=call_id,
            stage=stage,
            prompt=prompt,
            response_schema=response_schema,
            system_prompt=load_optimizer_prompt(
                stage,
                update_mode=self._mechanisms.update_mode,
            ),
            metadata={
                "protocol_id": self._profile.protocol_id,
                "paper_profile_sha256": self._profile_sha256,
                "data_sources": ["train"],
                "controller_registry_sha256": authority.registry_sha256,
                "train_controller_id": authority.controller_id,
                "train_split_id": authority.split_id,
                "train_split_manifest_sha256": (
                    authority.split_manifest_sha256
                ),
                "retry_policy_id": self._retry_policy.policy_id,
                "semantic_attempt": semantic_attempt,
                "semantic_max_attempts": (
                    self._retry_policy.max_semantic_attempts
                ),
                "accumulation": len(train_evidence.batches),
                "accumulation_index": (
                    batch_cursor.accumulation_index
                    if batch_cursor is not None
                    else None
                ),
                "train_batch_ids": [
                    batch.batch_id for batch in scheduled
                ],
                "train_batch_id": (
                    metadata_batch.batch_id
                    if metadata_batch is not None
                    else None
                ),
                "train_batch_seed": (
                    metadata_batch.batch_seed
                    if metadata_batch is not None
                    else None
                ),
                "train_batch_size": (
                    metadata_batch.batch_size
                    if metadata_batch is not None
                    else None
                ),
            },
        )
        response = self._controller.optimizer_backend.complete(request)
        if type(response) is not OptimizerResponse:
            raise OptimizerContractViolation(
                "optimizer backend must return exact OptimizerResponse"
            )
        if response.call_id != call_id:
            raise OptimizerContractViolation(
                "optimizer response call_id does not match its request"
            )
        with self._exchange_lock:
            self._active_exchanges.append(
                OptimizerExchange(request=request, response=response)
            )
        return response

    def _take_optimizer_exchanges(self) -> tuple[OptimizerExchange, ...]:
        with self._exchange_lock:
            exchanges = tuple(
                sorted(
                    self._active_exchanges,
                    key=lambda item: (
                        _stage_order(item.request.stage),
                        item.request.call_id,
                    ),
                )
            )
            self._active_exchanges = []
        return exchanges

    def _append_event(
        self,
        events: list[AlgorithmEvent],
        event_type: AlgorithmEventType,
        state: PaperState,
        step: int,
        payload: Mapping[str, Any],
    ) -> None:
        events.append(
            AlgorithmEvent(
                sequence=self._next_event_sequence + len(events),
                event_type=event_type,
                epoch=state.epoch,
                step=step,
                payload=dict(payload),
            )
        )


def _patch_payload(value: ParsedPatchResponse) -> dict[str, Any]:
    return {
        "reasoning": value.reasoning,
        "edits": [_edit_payload(edit) for edit in value.edits],
    }


def _update_items(
    value: ParsedUpdateResponse,
) -> tuple[PaperEdit, ...] | tuple[PaperSuggestion, ...]:
    if type(value) is ParsedPatchResponse:
        return value.edits
    if type(value) is ParsedSuggestionResponse:
        return value.suggestions
    raise ValueError("unknown parsed update response")


def _update_payload(value: ParsedUpdateResponse) -> dict[str, Any]:
    if type(value) is ParsedPatchResponse:
        return _patch_payload(value)
    if type(value) is ParsedSuggestionResponse:
        return {
            "reasoning": value.reasoning,
            "revise_suggestions": [
                _suggestion_payload(item) for item in value.suggestions
            ],
        }
    raise ValueError("unknown parsed update response")


def _update_item_payload(item: PaperEdit | PaperSuggestion) -> dict[str, Any]:
    if type(item) is PaperEdit:
        return _edit_payload(item)
    if type(item) is PaperSuggestion:
        return _suggestion_payload(item)
    raise ValueError("unknown paper update item")


def _edit_payload(edit: PaperEdit) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "edit_id": edit.edit_id,
        "op": edit.operation.value,
        "support_count": edit.support_count,
    }
    if edit.target:
        payload["target"] = edit.target
    if edit.content:
        payload["content"] = edit.content
    if edit.source_type is not None:
        payload["source_type"] = edit.source_type.value
    return payload


def _suggestion_payload(suggestion: PaperSuggestion) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "suggestion_id": suggestion.suggestion_id,
        "type": suggestion.suggestion_type.value,
        "title": suggestion.title,
        "motivation": suggestion.motivation,
        "instruction": suggestion.instruction,
        "priority_hint": suggestion.priority_hint.value,
        "support_count": suggestion.support_count,
    }
    if suggestion.source_type is not None:
        payload["source_type"] = suggestion.source_type.value
    return payload


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _attempt_call_id(call_id: str, attempt: int) -> str:
    if attempt == 1:
        return call_id
    return f"{call_id}-semantic-retry-{attempt - 1}"


def _stage_order(stage: OptimizerStage) -> int:
    order = {
        OptimizerStage.REFLECT_FAILURE: 0,
        OptimizerStage.REFLECT_SUCCESS: 0,
        OptimizerStage.REFINE: 1,
        OptimizerStage.MERGE_FAILURE: 2,
        OptimizerStage.MERGE_SUCCESS: 2,
        OptimizerStage.MERGE_FINAL_FAILURE_PRIORITIZED: 3,
        OptimizerStage.DECIDE_LEARNING_RATE: 4,
        OptimizerStage.RANK_TOP_L: 5,
        OptimizerStage.REWRITE_SKILL: 6,
    }
    return order.get(stage, 99)
