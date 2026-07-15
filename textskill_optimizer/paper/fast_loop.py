"""One replayable Algorithm 1 fast step for paper-faithful optimization."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

from .backend import OptimizerRequest, OptimizerResponse, OptimizerStage
from .config import PaperProfile
from .data import SelectionScore, TrainEvidenceBatch, strict_selection_decision
from .optimization import PaperOptimizationController
from .patches import PatchApplyResult, apply_paper_patch
from .prompts import load_optimizer_prompt
from .provenance import canonical_json_sha256
from .responses import (
    OptimizerContractViolation,
    ParsedPatchResponse,
    optimizer_response_schema,
    parse_patch_response,
    parse_rank_response,
)
from .types import (
    AlgorithmEvent,
    AlgorithmEventType,
    PaperEdit,
    PaperEditSource,
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


@dataclass(frozen=True)
class FastStepResult:
    """Complete state transition and reconstruction inputs for one fast step."""

    input_skill: str
    state: PaperState
    candidate_score: SelectionScore
    ranked_edits: tuple[PaperEdit, ...]
    apply_result: PatchApplyResult
    events: tuple[AlgorithmEvent, ...]
    skipped_stage: OptimizerStage | None = None

    def replay(self) -> PatchApplyResult:
        replayed = apply_paper_patch(self.input_skill, self.ranked_edits)
        if replayed != self.apply_result:
            raise RuntimeError("fast-step artifact does not replay exactly")
        return replayed


class PaperFastLoop:
    """Execute the paper fast path behind one narrow, train-evidence-only API."""

    def __init__(
        self,
        controller: PaperOptimizationController,
        *,
        profile: PaperProfile,
        retry_policy: OptimizerRetryPolicy = OptimizerRetryPolicy(),
    ) -> None:
        if type(controller) is not PaperOptimizationController:
            raise ValueError(
                "paper fast loop requires exact PaperOptimizationController"
            )
        controller.__post_init__()
        if type(profile) is not PaperProfile:
            raise ValueError("paper fast loop requires exact PaperProfile")
        validated_profile = PaperProfile.from_mapping(profile.to_dict())
        if type(retry_policy) is not OptimizerRetryPolicy:
            raise ValueError("paper fast loop requires exact OptimizerRetryPolicy")
        retry_policy.__post_init__()
        self._controller = controller
        self._profile = validated_profile
        self._profile_sha256 = canonical_json_sha256(validated_profile.to_dict())
        self._retry_policy = retry_policy
        self._score_cache: dict[str, SelectionScore] = {}
        self._next_event_sequence = 0
        self._state: PaperState | None = None

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

    def run_step(
        self,
        *,
        train_evidence: TrainEvidenceBatch,
        edit_budget: int,
    ) -> FastStepResult:
        """Execute collect-to-gate once and commit cache/sequence on success."""

        self._controller.__post_init__()
        if self._state is None:
            raise ValueError("paper fast loop must be initialized")
        state = self._state
        if type(train_evidence) is not TrainEvidenceBatch:
            raise ValueError("run_step requires exact TrainEvidenceBatch")
        if (
            type(edit_budget) is not int
            or not self._profile.learning_rate_floor
            <= edit_budget
            <= self._profile.learning_rate
        ):
            raise ValueError(
                "edit_budget must be within the frozen paper learning-rate range"
            )

        step = state.step + 1
        call_prefix = f"e{state.epoch}-s{step}"
        trajectories = self._controller.train.verify(
            train_evidence,
            current_skill=state.current_skill,
        )
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
            {"edit_budget": edit_budget, "current_skill_sha256": current_hash},
        )
        failures = tuple(item for item in trajectories if not item["success"])
        successes = tuple(item for item in trajectories if item["success"])
        self._append_event(
            events,
            AlgorithmEventType.ROLLOUT_COLLECTED,
            state,
            step,
            {
                "trajectory_count": len(trajectories),
                "failure_count": len(failures),
                "success_count": len(successes),
                "train_split_id": train_evidence.split_id,
            },
        )

        failure_proposals = self._reflect_group(
            state=state,
            step=step,
            source=PaperEditSource.FAILURE,
            trajectories=failures,
            train_evidence=train_evidence,
            edit_budget=edit_budget,
            call_prefix=call_prefix,
            events=events,
        )
        success_proposals = self._reflect_group(
            state=state,
            step=step,
            source=PaperEditSource.SUCCESS,
            trajectories=successes,
            train_evidence=train_evidence,
            edit_budget=edit_budget,
            call_prefix=call_prefix,
            events=events,
        )

        try:
            failure_merge = self._hierarchical_merge(
                source=PaperEditSource.FAILURE,
                proposals=failure_proposals,
                call_prefix=call_prefix,
                state=state,
                step=step,
                train_evidence=train_evidence,
                edit_budget=edit_budget,
                events=events,
            )
            success_merge = self._hierarchical_merge(
                source=PaperEditSource.SUCCESS,
                proposals=success_proposals,
                call_prefix=call_prefix,
                state=state,
                step=step,
                train_evidence=train_evidence,
                edit_budget=edit_budget,
                events=events,
            )
            final_merge = self._merge(
                stage=OptimizerStage.MERGE_FINAL_FAILURE_PRIORITIZED,
                call_id=f"{call_prefix}-merge-final",
                state=state,
                step=step,
                train_evidence=train_evidence,
                edit_budget=edit_budget,
                prompt_payload={
                    "current_skill": state.current_skill,
                    "failure_patch": _patch_payload(failure_merge),
                    "success_patch": _patch_payload(success_merge),
                    "meta_skill": state.meta_skill,
                },
                edit_id_prefix=f"{call_prefix}-merge-final-edit",
                event_type=AlgorithmEventType.MERGE_FINAL_FAILURE_PRIORITIZED,
                hierarchy_level=1,
                batch_index=1,
                events=events,
            )
            ranked_edits = self._rank(
                call_id=f"{call_prefix}-rank-top-l",
                state=state,
                step=step,
                train_evidence=train_evidence,
                edit_budget=edit_budget,
                candidates=final_merge.edits,
                events=events,
            )
        except _SemanticStageExhausted as error:
            return self._skip_failed_semantic_stage(
                state=state,
                step=step,
                error=error,
                events=events,
                working_cache=working_cache,
            )

        apply_result = apply_paper_patch(state.current_skill, ranked_edits)
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

        candidate_hash = apply_result.output_sha256
        candidate_score = working_cache.get(candidate_hash)
        cache_hit = candidate_score is not None
        if candidate_score is None:
            candidate_score = self._controller.selection.score(
                apply_result.output_skill
            )
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
        train_evidence: TrainEvidenceBatch,
        edit_budget: int,
        call_prefix: str,
        events: list[AlgorithmEvent],
    ) -> tuple[ParsedPatchResponse, ...]:
        proposals: list[ParsedPatchResponse] = []
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
        for batch_index, start in enumerate(
            range(0, len(trajectories), self._profile.reflection_minibatch_size),
            1,
        ):
            minibatch = trajectories[
                start : start + self._profile.reflection_minibatch_size
            ]
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
                ),
                train_evidence=train_evidence,
            )
            parsed = parse_patch_response(
                stage=stage,
                payload=response.payload,
                edit_budget=edit_budget,
                edit_id_prefix=f"{call_id}-edit",
                expected_batch_size=len(minibatch),
            )
            self._append_event(
                events,
                event_type,
                state,
                step,
                {
                    "call_id": call_id,
                    "batch_index": batch_index,
                    "batch_size": len(minibatch),
                    "edit_count": len(parsed.edits),
                },
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
                        "prior_patch": _patch_payload(parsed),
                        "round": round_number,
                        "max_rounds": self._profile.max_analyst_rounds,
                        "edit_budget": edit_budget,
                        "meta_skill": state.meta_skill,
                    },
                    response_schema=optimizer_response_schema(
                        OptimizerStage.REFINE,
                        edit_budget=edit_budget,
                    ),
                    train_evidence=train_evidence,
                )
                parsed = parse_patch_response(
                    stage=OptimizerStage.REFINE,
                    payload=response.payload,
                    edit_budget=edit_budget,
                    edit_id_prefix=f"{refine_call_id}-edit",
                    source_type=source,
                )
                self._append_event(
                    events,
                    AlgorithmEventType.ANALYST_REFINED,
                    state,
                    step,
                    {
                        "call_id": refine_call_id,
                        "source_type": source.value,
                        "batch_index": batch_index,
                        "round": round_number,
                        "edit_count": len(parsed.edits),
                        "converged": parsed.converged,
                    },
                )
                if parsed.converged:
                    break
            proposals.append(parsed)
        return tuple(proposals)

    def _hierarchical_merge(
        self,
        *,
        source: PaperEditSource,
        proposals: tuple[ParsedPatchResponse, ...],
        call_prefix: str,
        state: PaperState,
        step: int,
        train_evidence: TrainEvidenceBatch,
        edit_budget: int,
        events: list[AlgorithmEvent],
    ) -> ParsedPatchResponse:
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
            merged: list[ParsedPatchResponse] = []
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
                            "patches": [_patch_payload(item) for item in batch],
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
        train_evidence: TrainEvidenceBatch,
        edit_budget: int,
        prompt_payload: Mapping[str, Any],
        edit_id_prefix: str,
        event_type: AlgorithmEventType,
        hierarchy_level: int,
        batch_index: int,
        events: list[AlgorithmEvent],
    ) -> ParsedPatchResponse:
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
                    ),
                    train_evidence=train_evidence,
                    semantic_attempt=attempt,
                )
                parsed = parse_patch_response(
                    stage=stage,
                    payload=response.payload,
                    edit_budget=edit_budget,
                    edit_id_prefix=(
                        edit_id_prefix
                        if attempt == 1
                        else f"{edit_id_prefix}-retry-{attempt - 1}"
                    ),
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
                    "edit_count": len(parsed.edits),
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

    def _rank(
        self,
        *,
        call_id: str,
        state: PaperState,
        step: int,
        train_evidence: TrainEvidenceBatch,
        edit_budget: int,
        candidates: tuple[PaperEdit, ...],
        events: list[AlgorithmEvent],
    ) -> tuple[PaperEdit, ...]:
        schema = optimizer_response_schema(
            OptimizerStage.RANK_TOP_L,
            edit_budget=edit_budget,
            candidate_count=len(candidates),
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
                        "edits": [_edit_payload(edit) for edit in candidates],
                        "edit_budget": edit_budget,
                        "meta_skill": state.meta_skill,
                    },
                    response_schema=schema,
                    train_evidence=train_evidence,
                    semantic_attempt=attempt,
                )
                ranked = parse_rank_response(
                    payload=response.payload,
                    candidates=candidates,
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
                    "selected_edit_ids": [edit.edit_id for edit in ranked],
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

    def _skip_failed_semantic_stage(
        self,
        *,
        state: PaperState,
        step: int,
        error: _SemanticStageExhausted,
        events: list[AlgorithmEvent],
        working_cache: dict[str, SelectionScore],
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
            skipped_stage=error.stage,
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
        train_evidence: TrainEvidenceBatch,
        semantic_attempt: int = 1,
    ) -> OptimizerResponse:
        try:
            prompt = json.dumps(
                prompt_payload,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        except (TypeError, ValueError) as error:
            raise OptimizerContractViolation(
                f"fast-loop optimizer prompt is not JSON-safe: {error}"
            ) from error
        request = OptimizerRequest(
            call_id=call_id,
            stage=stage,
            prompt=prompt,
            response_schema=response_schema,
            system_prompt=load_optimizer_prompt(stage),
            metadata={
                "protocol_id": self._profile.protocol_id,
                "paper_profile_sha256": self._profile_sha256,
                "data_sources": ["train"],
                "controller_registry_sha256": train_evidence.registry_sha256,
                "train_controller_id": train_evidence.controller_id,
                "train_split_id": train_evidence.split_id,
                "train_split_manifest_sha256": (
                    train_evidence.split_manifest_sha256
                ),
                "retry_policy_id": self._retry_policy.policy_id,
                "semantic_attempt": semantic_attempt,
                "semantic_max_attempts": (
                    self._retry_policy.max_semantic_attempts
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
        return response

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


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _attempt_call_id(call_id: str, attempt: int) -> str:
    if attempt == 1:
        return call_id
    return f"{call_id}-semantic-retry-{attempt - 1}"
