"""Stateful epoch owner for paper-faithful optimization."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping

from .artifacts import (
    OptimizerExchange,
    PaperArtifactKind,
    PaperArtifactLedger,
    PaperArtifactLineage,
    optimizer_request_payload,
    optimizer_response_payload,
)
from .backend import OptimizerRequest, OptimizerResponse, OptimizerStage
from .checkpoint import CheckpointAuthenticator, PaperEpochCheckpoint
from .config import PaperProfile
from .controller_process import ControllerRole
from .data import SelectionScore, StepTrainEvidence
from .epoch_plan import EpochCursor, PaperEpochPlan
from .errors import DataFirewallViolation
from .fast_loop import ExternalGateResult, FastStepResult, PaperFastLoop
from .longitudinal import (
    LongitudinalEvidence,
    LongitudinalState,
    build_longitudinal_state,
)
from .optimization import PaperOptimizationController
from .patches import read_slow_update_field, write_slow_update_field
from .prompts import load_optimizer_prompt
from .provenance import canonical_json_sha256
from .responses import (
    OptimizerContractViolation,
    epoch_response_schema,
    parse_epoch_response,
)
from .types import (
    AlgorithmEvent,
    AlgorithmEventType,
    EpochBufferRecord,
    PaperEdit,
    PaperState,
    PaperSuggestion,
)


@dataclass(frozen=True)
class EpochStepResult:
    """One scheduled cursor and its complete fast-loop transition."""

    cursor: EpochCursor
    fast_step: FastStepResult

    @property
    def state(self) -> PaperState:
        return self.fast_step.state


@dataclass(frozen=True)
class EpochCompletionResult:
    """Committed epoch boundary and its lifecycle events."""

    completed_epoch: int
    state: PaperState
    events: tuple[AlgorithmEvent, ...]
    run_completed: bool


class PaperEpochLoop:
    """Own scheduler and lifecycle state behind a train-evidence-only API."""

    def __init__(
        self,
        controller: PaperOptimizationController,
        *,
        profile: PaperProfile,
        plan: PaperEpochPlan,
    ) -> None:
        if type(controller) is not PaperOptimizationController:
            raise ValueError("paper epoch loop requires exact controller")
        controller.__post_init__()
        if type(profile) is not PaperProfile:
            raise ValueError("paper epoch loop requires exact PaperProfile")
        validated_profile = PaperProfile.from_mapping(profile.to_dict())
        if type(plan) is not PaperEpochPlan:
            raise ValueError("paper epoch loop requires exact PaperEpochPlan")
        plan.__post_init__()
        plan.require_profile(validated_profile)
        registration = controller.train.registry.require(
            controller.train.controller_id,
            role=ControllerRole.TRAIN,
        )
        if (
            plan.train_split_id != registration.split_id
            or plan.train_split_manifest_sha256
            != registration.artifact("split_manifest").sha256
        ):
            raise ValueError("paper epoch plan does not match train registry")
        self._controller = controller
        self._profile = validated_profile
        self._plan = plan
        self._plan_sha256 = canonical_json_sha256(plan.to_dict())
        self._fast_loop = PaperFastLoop(
            controller,
            profile=validated_profile,
            _mechanisms=plan.mechanisms,
        )
        self._events: list[AlgorithmEvent] = []
        self._epoch_buffer: list[EpochBufferRecord] = []
        self._epoch_snapshots: list[str] = []
        self._epoch_snapshot_artifact_ids: list[str] = []
        self._run_completed = False
        self._artifacts = PaperArtifactLedger()
        self._artifact_heads: dict[str, str | None] = {}
        self._pending_epoch_exchanges: list[OptimizerExchange] = []

    @property
    def events(self) -> tuple[AlgorithmEvent, ...]:
        return tuple(self._events)

    @property
    def state(self) -> PaperState:
        return self._fast_loop.state

    @property
    def epoch_buffer(self) -> tuple[EpochBufferRecord, ...]:
        return tuple(self._epoch_buffer)

    @property
    def epoch_snapshots(self) -> tuple[str, ...]:
        return tuple(self._epoch_snapshots)

    @property
    def score_cache(self) -> Mapping[str, SelectionScore]:
        return self._fast_loop.score_cache

    @property
    def artifact_lineage(self) -> PaperArtifactLineage:
        return self._artifacts.lineage

    @property
    def longitudinal_skills(self) -> tuple[str, str]:
        state = self._fast_loop.state
        if (
            state.epoch < self._profile.slow_update.start_epoch
            or state.step != self._plan.steps_per_epoch
            or len(self._epoch_snapshots) != state.epoch - 1
        ):
            raise ValueError("longitudinal skills are available only at an epoch boundary")
        return self._epoch_snapshots[-1], state.current_skill

    def collect_train_evidence(self) -> StepTrainEvidence:
        if self._run_completed:
            raise ValueError("paper epoch run is complete")
        state = self._fast_loop.state
        next_step = state.step + 1
        if next_step > self._plan.steps_per_epoch:
            raise ValueError(f"epoch {state.epoch} is complete")
        cursor = self._plan.cursor(epoch=state.epoch, step=next_step)
        return StepTrainEvidence(
            tuple(
                self._controller.train.collect(
                    state.current_skill,
                    batch_id=batch.batch_id,
                    batch_seed=batch.batch_seed,
                    batch_size=batch.batch_size,
                )
                for batch in cursor.batches
            )
        )

    def collect_longitudinal_evidence(self) -> LongitudinalEvidence:
        previous_skill, current_skill = self.longitudinal_skills
        batch_id = self._longitudinal_batch_id(self.state.epoch)
        batch_seed = self._longitudinal_batch_seed(self.state.epoch)
        batch_size = self._profile.slow_update.sample_size
        return LongitudinalEvidence(
            previous=self._controller.train.collect(
                previous_skill,
                batch_id=batch_id,
                batch_seed=batch_seed,
                batch_size=batch_size,
            ),
            current=self._controller.train.collect(
                current_skill,
                batch_id=batch_id,
                batch_seed=batch_seed,
                batch_size=batch_size,
            ),
        )

    def initialize(self, initial_skill: str) -> PaperState:
        state = self._fast_loop.initialize(initial_skill)
        self._initialize_artifacts(state)
        self._events.append(
            self._fast_loop._record_lifecycle_event(
                AlgorithmEventType.RUN_STARTED,
                {
                    "paper_epoch_plan_sha256": self._plan_sha256,
                    "controller_registry_sha256": (
                        self._controller.train.registry.sha256
                    ),
                },
            )
        )
        self._events.append(
            self._fast_loop._record_lifecycle_event(
                AlgorithmEventType.EPOCH_STARTED,
                {
                    "steps_per_epoch": self._plan.steps_per_epoch,
                    "epoch_buffer_size": 0,
                },
            )
        )
        self._record_event_artifacts(tuple(self._events))
        return state

    def run_step(self, *, train_evidence: StepTrainEvidence) -> EpochStepResult:
        if self._run_completed:
            raise ValueError("paper epoch run is complete")
        if type(train_evidence) is not StepTrainEvidence:
            raise DataFirewallViolation(
                "scheduled batch evidence requires exact StepTrainEvidence"
            )
        state = self._fast_loop.state
        next_step = state.step + 1
        if next_step > self._plan.steps_per_epoch:
            raise ValueError(f"epoch {state.epoch} is complete")
        cursor = self._plan.cursor(epoch=state.epoch, step=next_step)
        self._fast_loop._prepare_epoch_step(
            tuple(self._epoch_buffer),
            batches=cursor.batches,
        )
        fast_step = self._fast_loop._run_accumulated_step(
            train_evidence=train_evidence,
            analysis_budget=cursor.analysis_budget,
            edit_budget=cursor.edit_budget,
        )
        self._events.extend(fast_step.events)
        self._record_fast_step_artifacts(train_evidence, fast_step)
        decision_event = next(
            event
            for event in reversed(fast_step.events)
            if event.event_type
            in {
                AlgorithmEventType.CANDIDATE_ACCEPTED,
                AlgorithmEventType.CANDIDATE_REJECTED,
            }
        )
        rejected = (
            decision_event.event_type is AlgorithmEventType.CANDIDATE_REJECTED
        )
        contract_rejected = fast_step.selection_skipped_reason is not None
        self._epoch_buffer.append(
            EpochBufferRecord(
                epoch=state.epoch,
                step=next_step,
                failure_patterns=fast_step.failure_patterns,
                rejected_edits=(
                    fast_step.ranked_edits
                    if rejected and not contract_rejected
                    else ()
                ),
                rejected_suggestions=(
                    fast_step.ranked_suggestions
                    if rejected and not contract_rejected
                    else ()
                ),
                score_delta=(
                    float(decision_event.payload["delta"])
                    if rejected and not contract_rejected
                    else None
                ),
            )
        )
        return EpochStepResult(cursor=cursor, fast_step=fast_step)

    def finish_epoch(
        self,
        *,
        longitudinal_evidence: LongitudinalEvidence | None = None,
    ) -> EpochCompletionResult:
        if self._run_completed:
            raise ValueError("paper epoch run is complete")
        state = self._fast_loop.state
        fast_epoch_end_skill = state.current_skill
        fast_epoch_end_skill_id = self._require_artifact_head("current_skill")
        if state.step != self._plan.steps_per_epoch:
            raise ValueError(
                f"epoch {state.epoch} has not completed all scheduled steps"
            )
        if state.epoch < self._profile.slow_update.start_epoch:
            if longitudinal_evidence is not None:
                raise ValueError("epoch 1 cannot consume longitudinal evidence")
            boundary_events = [
                self._fast_loop._record_lifecycle_event(
                    AlgorithmEventType.SLOW_UPDATE_SKIPPED,
                    {"reason": "before_start_epoch"},
                ),
                self._fast_loop._record_lifecycle_event(
                    AlgorithmEventType.META_UPDATE_SKIPPED,
                    {"reason": "before_start_epoch"},
                ),
            ]
            pre_recorded_event_count = 0
        else:
            if type(longitudinal_evidence) is not LongitudinalEvidence:
                raise ValueError(
                    f"epoch {state.epoch} requires authenticated longitudinal evidence"
                )
            boundary_events = list(
                self._run_slow_meta(longitudinal_evidence)
            )
            pre_recorded_event_count = len(boundary_events)
            state = self._fast_loop.state
        self._epoch_snapshots.append(fast_epoch_end_skill)
        self._epoch_snapshot_artifact_ids.append(fast_epoch_end_skill_id)
        boundary_events.append(
            self._fast_loop._record_lifecycle_event(
                AlgorithmEventType.EPOCH_COMPLETED,
                {
                    "epoch_buffer_size": len(self._epoch_buffer),
                    "current_skill_sha256": _sha256(state.current_skill),
                },
            )
        )
        completed_epoch = state.epoch
        run_completed = completed_epoch == self._plan.epochs
        if run_completed:
            boundary_events.append(
                self._fast_loop._record_lifecycle_event(
                    AlgorithmEventType.RUN_COMPLETED,
                    {"completed_epochs": completed_epoch},
                )
            )
            next_state = state
            self._run_completed = True
        else:
            next_state = self._fast_loop._begin_next_epoch()
            self._epoch_buffer = []
            boundary_events.append(
                self._fast_loop._record_lifecycle_event(
                    AlgorithmEventType.EPOCH_STARTED,
                    {
                        "steps_per_epoch": self._plan.steps_per_epoch,
                        "epoch_buffer_size": 0,
                    },
                )
            )
        self._events.extend(boundary_events)
        self._record_event_artifacts(
            tuple(boundary_events[pre_recorded_event_count:])
        )
        return EpochCompletionResult(
            completed_epoch=completed_epoch,
            state=next_state,
            events=tuple(boundary_events),
            run_completed=run_completed,
        )

    def _run_slow_meta(
        self,
        evidence: LongitudinalEvidence,
    ) -> tuple[AlgorithmEvent, ...]:
        self._pending_epoch_exchanges = []
        state = self._fast_loop.state
        previous_skill, current_epoch_skill = self.longitudinal_skills
        longitudinal_batch_id = self._longitudinal_batch_id(state.epoch)
        longitudinal = build_longitudinal_state(
            train=self._controller.train,
            evidence=evidence,
            previous_skill=previous_skill,
            current_skill=current_epoch_skill,
            sample_size=self._profile.slow_update.sample_size,
            split_seed=self._profile.split_seed,
            epoch=state.epoch,
            batch_id=longitudinal_batch_id,
            batch_seed=self._longitudinal_batch_seed(state.epoch),
        )
        common = {
            "previous_epoch_skill": previous_skill,
            "current_epoch_skill": current_epoch_skill,
            **(
                {
                    "immutable_skill_contract": (
                        self._controller.skill_contract_description
                    )
                }
                if self._controller.skill_contract_description is not None
                else {}
            ),
            "longitudinal": longitudinal.to_prompt_payload(),
            "epoch_buffer": [
                record.to_optimizer_payload() for record in self._epoch_buffer
            ],
        }
        slow_response = self._complete_epoch_stage(
            call_id=f"e{state.epoch}-slow-update",
            stage=OptimizerStage.PROPOSE_SLOW_UPDATE,
            prompt_payload={
                **common,
                "previous_slow_update": read_slow_update_field(
                    current_epoch_skill
                ),
            },
            longitudinal=longitudinal,
            evidence=evidence,
        )
        parsed_slow = parse_epoch_response(
            stage=OptimizerStage.PROPOSE_SLOW_UPDATE,
            payload=slow_response.payload,
        )
        slow_candidate = write_slow_update_field(
            current_epoch_skill,
            parsed_slow.content,
        )
        slow_preview = self._fast_loop._preview_external_candidate(
            slow_candidate
        )
        meta_response = self._complete_epoch_stage(
            call_id=f"e{state.epoch}-meta-skill",
            stage=OptimizerStage.UPDATE_META_SKILL,
            prompt_payload={
                **common,
                "previous_meta_skill": state.meta_skill,
            },
            longitudinal=longitudinal,
            evidence=evidence,
        )
        parsed_meta = parse_epoch_response(
            stage=OptimizerStage.UPDATE_META_SKILL,
            payload=meta_response.payload,
        )
        slow_gate = self._fast_loop._commit_external_candidate(
            slow_candidate,
            source="slow_update",
            proposal_event_type=AlgorithmEventType.SLOW_UPDATE_PROPOSED,
            proposal_payload={
                "candidate_skill_sha256": _sha256(slow_candidate),
                "longitudinal_sample_size": (
                    self._profile.slow_update.sample_size
                ),
                "train_batch_id": longitudinal_batch_id,
            },
            preview=slow_preview,
        )
        events = list(slow_gate.events)
        self._fast_loop._update_meta_skill(parsed_meta.content)
        events.append(
            self._fast_loop._record_lifecycle_event(
                AlgorithmEventType.META_UPDATE_COMPLETED,
                {
                    "meta_skill_sha256": _sha256(parsed_meta.content),
                    "target_visible": False,
                },
            )
        )
        self._record_slow_meta_artifacts(
            evidence=evidence,
            previous_skill=previous_skill,
            current_skill=current_epoch_skill,
            slow_candidate=slow_candidate,
            slow_gate=slow_gate,
            meta_skill=parsed_meta.content,
            events=tuple(events),
        )
        return tuple(events)

    def _complete_epoch_stage(
        self,
        *,
        call_id: str,
        stage: OptimizerStage,
        prompt_payload: Mapping[str, Any],
        longitudinal: LongitudinalState,
        evidence: LongitudinalEvidence,
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
                f"epoch optimizer prompt is not JSON-safe: {error}"
            ) from error
        sample_count = sum(
            len(items)
            for items in longitudinal.to_prompt_payload().values()
        )
        request = OptimizerRequest(
            call_id=call_id,
            stage=stage,
            prompt=prompt,
            response_schema=epoch_response_schema(stage),
            system_prompt=load_optimizer_prompt(stage),
            metadata={
                "protocol_id": self._profile.protocol_id,
                "paper_profile_sha256": canonical_json_sha256(
                    self._profile.to_dict()
                ),
                "data_sources": ["train"],
                "controller_registry_sha256": (
                    self._controller.train.registry.sha256
                ),
                "train_controller_id": self._controller.train.controller_id,
                "train_split_id": evidence.current.split_id,
                "train_split_manifest_sha256": (
                    evidence.current.split_manifest_sha256
                ),
                "longitudinal_sample_size": sample_count,
                "train_batch_id": self._longitudinal_batch_id(
                    self.state.epoch
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
        self._pending_epoch_exchanges.append(
            OptimizerExchange(request=request, response=response)
        )
        return response

    def _longitudinal_batch_id(self, epoch: int) -> str:
        return "slow-batch-" + canonical_json_sha256(
            {
                "plan_sha256": self._plan_sha256,
                "train_split_id": self._plan.train_split_id,
                "split_seed": self._profile.split_seed,
                "epoch": epoch,
                "sample_size": self._profile.slow_update.sample_size,
            }
        )[:20]

    def _longitudinal_batch_seed(self, epoch: int) -> int:
        return int(
            canonical_json_sha256(
                {
                    "plan_sha256": self._plan_sha256,
                    "split_seed": self._profile.split_seed,
                    "epoch": epoch,
                    "batch_id": self._longitudinal_batch_id(epoch),
                }
            )[:16],
            16,
        )

    def _initialize_artifacts(self, state: PaperState) -> None:
        registry = self._controller.train.registry
        profile_record = self._artifacts.add(
            PaperArtifactKind.PROFILE,
            self._profile.to_dict(),
        )
        registry_record = self._artifacts.add(
            PaperArtifactKind.CONTROLLER_REGISTRY,
            {
                "schema_version": "paper-controller-registry-v1",
                "registrations": [
                    item.to_manifest()
                    for item in sorted(
                        registry.registrations,
                        key=lambda value: value.controller_id,
                    )
                ],
            },
        )
        plan_record = self._artifacts.add(
            PaperArtifactKind.EPOCH_PLAN,
            self._plan.to_dict(),
            parent_ids=(profile_record.artifact_id,),
        )
        skill_record = self._artifacts.add(
            PaperArtifactKind.SKILL,
            {
                "role": "initial",
                "skill_text": state.current_skill,
                "skill_sha256": _sha256(state.current_skill),
            },
            parent_ids=(plan_record.artifact_id,),
        )
        score_record = self._artifacts.add(
            PaperArtifactKind.SELECTION_SCORE,
            {
                "role": "initial",
                "score": state.current_score.value,
                "skill_sha256": _sha256(state.current_skill),
            },
            parent_ids=(
                skill_record.artifact_id,
                registry_record.artifact_id,
            ),
        )
        self._artifact_heads = {
            "profile": profile_record.artifact_id,
            "plan": plan_record.artifact_id,
            "registry": registry_record.artifact_id,
            "current_skill": skill_record.artifact_id,
            "current_score": score_record.artifact_id,
            "meta_skill": None,
            "last_event": None,
        }

    def _record_slow_meta_artifacts(
        self,
        *,
        evidence: LongitudinalEvidence,
        previous_skill: str,
        current_skill: str,
        slow_candidate: str,
        slow_gate: ExternalGateResult,
        meta_skill: str,
        events: tuple[AlgorithmEvent, ...],
    ) -> None:
        plan_id = self._require_artifact_head("plan")
        registry_id = self._require_artifact_head("registry")
        current_skill_id = self._require_artifact_head("current_skill")
        if not self._epoch_snapshot_artifact_ids:
            raise RuntimeError("longitudinal lineage requires a prior epoch snapshot")
        previous_skill_id = self._epoch_snapshot_artifact_ids[-1]
        previous_record = next(
            record
            for record in self._artifacts.lineage.records
            if record.artifact_id == previous_skill_id
        )
        if (
            previous_record.kind is not PaperArtifactKind.SKILL
            or previous_record.payload["skill_text"] != previous_skill
        ):
            raise RuntimeError("longitudinal snapshot artifact is inconsistent")
        longitudinal_ids: list[str] = []
        for role, batch, skill_id in (
            ("previous", evidence.previous, previous_skill_id),
            ("current", evidence.current, current_skill_id),
        ):
            record = self._artifacts.add(
                PaperArtifactKind.LONGITUDINAL_EVIDENCE,
                {
                    "role": role,
                    "controller_id": batch.controller_id,
                    "registry_sha256": batch.registry_sha256,
                    "split_id": batch.split_id,
                    "split_manifest_sha256": batch.split_manifest_sha256,
                    "canonical_request": batch.canonical_request,
                    "canonical_payload": batch.canonical_payload,
                    "signature": batch.signature,
                },
                parent_ids=(skill_id, plan_id, registry_id),
            )
            longitudinal_ids.append(record.artifact_id)
        response_ids_by_stage: dict[OptimizerStage, list[str]] = {}
        for exchange in self._pending_epoch_exchanges:
            request_record = self._artifacts.add(
                PaperArtifactKind.OPTIMIZER_REQUEST,
                optimizer_request_payload(exchange.request),
                parent_ids=(current_skill_id, *longitudinal_ids),
            )
            response_record = self._artifacts.add(
                PaperArtifactKind.OPTIMIZER_RESPONSE,
                optimizer_response_payload(exchange.response),
                parent_ids=(request_record.artifact_id,),
            )
            response_ids_by_stage.setdefault(
                exchange.request.stage,
                [],
            ).append(response_record.artifact_id)
        slow_response_ids = tuple(
            response_ids_by_stage.get(OptimizerStage.PROPOSE_SLOW_UPDATE, ())
        )
        meta_response_ids = tuple(
            response_ids_by_stage.get(OptimizerStage.UPDATE_META_SKILL, ())
        )
        if len(slow_response_ids) != 1 or len(meta_response_ids) != 1:
            raise RuntimeError("slow/meta lineage requires one response per stage")
        update_record = self._artifacts.add(
            PaperArtifactKind.UPDATE_SET,
            {
                "update_mode": "slow_update",
                "input_skill_sha256": _sha256(current_skill),
                "slow_update_content": read_slow_update_field(slow_candidate),
            },
            parent_ids=slow_response_ids,
        )
        apply_record = self._artifacts.add(
            PaperArtifactKind.APPLY_REPORT,
            {
                "update_mode": "slow_update",
                "input_skill": current_skill,
                "input_sha256": _sha256(current_skill),
                "output_skill": slow_candidate,
                "output_sha256": _sha256(slow_candidate),
                "reports": [],
                "rewrite": None,
            },
            parent_ids=(current_skill_id, update_record.artifact_id),
        )
        candidate_record = self._artifacts.add(
            PaperArtifactKind.SKILL,
            {
                "role": "slow_candidate",
                "skill_text": slow_candidate,
                "skill_sha256": _sha256(slow_candidate),
            },
            parent_ids=(apply_record.artifact_id,),
        )
        score_record = self._artifacts.add(
            PaperArtifactKind.SELECTION_SCORE,
            {
                "role": (
                    "carried_current"
                    if slow_gate.selection_skipped_reason is not None
                    else "slow_candidate"
                ),
                "score": slow_gate.candidate_score.value,
                "skill_sha256": _sha256(
                    current_skill
                    if slow_gate.selection_skipped_reason is not None
                    else slow_candidate
                ),
            },
            parent_ids=(
                (
                    candidate_record.artifact_id,
                    self._require_artifact_head("current_score"),
                )
                if slow_gate.selection_skipped_reason is not None
                else (candidate_record.artifact_id, registry_id)
            ),
        )
        meta_record = self._artifacts.add(
            PaperArtifactKind.META_SKILL,
            {
                "meta_skill": meta_skill,
                "meta_skill_sha256": _sha256(meta_skill),
                "target_visible": False,
            },
            parent_ids=meta_response_ids,
        )
        input_score_id = self._require_artifact_head("current_score")
        for event in events:
            event_parents: tuple[str, ...]
            if event.event_type is AlgorithmEventType.SLOW_UPDATE_PROPOSED:
                event_parents = (
                    current_skill_id,
                    input_score_id,
                    update_record.artifact_id,
                    apply_record.artifact_id,
                    candidate_record.artifact_id,
                )
            elif event.event_type in {
                AlgorithmEventType.SELECTION_SCORED,
                AlgorithmEventType.CANDIDATE_ACCEPTED,
                AlgorithmEventType.CANDIDATE_REJECTED,
            }:
                event_parents = (
                    current_skill_id,
                    input_score_id,
                    candidate_record.artifact_id,
                    score_record.artifact_id,
                )
            elif event.event_type is AlgorithmEventType.META_UPDATE_COMPLETED:
                event_parents = (
                    current_skill_id,
                    meta_record.artifact_id,
                )
            else:
                raise RuntimeError("unexpected slow/meta lifecycle event")
            self._record_event_artifact(event, event_parents)
        if slow_gate.accepted:
            self._artifact_heads["current_skill"] = candidate_record.artifact_id
            self._artifact_heads["current_score"] = score_record.artifact_id
        self._artifact_heads["meta_skill"] = meta_record.artifact_id
        self._pending_epoch_exchanges = []

    def _record_fast_step_artifacts(
        self,
        evidence: StepTrainEvidence,
        fast_step: FastStepResult,
    ) -> None:
        input_skill_id = self._require_artifact_head("current_skill")
        input_score_id = self._require_artifact_head("current_score")
        plan_id = self._require_artifact_head("plan")
        registry_id = self._require_artifact_head("registry")
        evidence_ids: list[str] = []
        for accumulation_index, batch in enumerate(evidence.batches, 1):
            record = self._artifacts.add(
                PaperArtifactKind.TRAIN_EVIDENCE,
                {
                    "accumulation_index": accumulation_index,
                    "controller_id": batch.controller_id,
                    "registry_sha256": batch.registry_sha256,
                    "split_id": batch.split_id,
                    "split_manifest_sha256": batch.split_manifest_sha256,
                    "canonical_request": batch.canonical_request,
                    "canonical_payload": batch.canonical_payload,
                    "signature": batch.signature,
                },
                parent_ids=(input_skill_id, plan_id, registry_id),
            )
            evidence_ids.append(record.artifact_id)

        response_ids: list[str] = []
        response_ids_by_call: dict[str, str] = {}
        completed_exchanges: list[tuple[OptimizerRequest, str]] = []
        rewrite_exchanges = tuple(
            exchange
            for exchange in fast_step.optimizer_exchanges
            if exchange.request.stage is OptimizerStage.REWRITE_SKILL
        )
        pre_apply_exchanges = tuple(
            exchange
            for exchange in fast_step.optimizer_exchanges
            if exchange.request.stage is not OptimizerStage.REWRITE_SKILL
        )
        for exchange in pre_apply_exchanges:
            stage_parent_ids = _optimizer_stage_parent_ids(
                exchange.request,
                completed_exchanges=tuple(completed_exchanges),
            )
            request_record = self._artifacts.add(
                PaperArtifactKind.OPTIMIZER_REQUEST,
                optimizer_request_payload(exchange.request),
                parent_ids=tuple(
                    dict.fromkeys(
                        (input_skill_id, *evidence_ids, *stage_parent_ids)
                    )
                ),
            )
            response_record = self._artifacts.add(
                PaperArtifactKind.OPTIMIZER_RESPONSE,
                optimizer_response_payload(exchange.response),
                parent_ids=(request_record.artifact_id,),
            )
            response_ids.append(response_record.artifact_id)
            response_ids_by_call[exchange.request.call_id] = (
                response_record.artifact_id
            )
            completed_exchanges.append(
                (exchange.request, response_record.artifact_id)
            )

        rank_response_ids = tuple(
            response_ids_by_call[event.payload["call_id"]]
            for event in fast_step.events
            if event.event_type is AlgorithmEventType.RANK_TOP_L
        )

        update_record = self._artifacts.add(
            PaperArtifactKind.UPDATE_SET,
            {
                "update_mode": self._plan.mechanisms.update_mode,
                "ranked_edits": [
                    _edit_artifact_payload(item)
                    for item in fast_step.ranked_edits
                ],
                "ranked_suggestions": [
                    _suggestion_artifact_payload(item)
                    for item in fast_step.ranked_suggestions
                ],
            },
            parent_ids=(
                rank_response_ids
                or tuple(response_ids)
                or tuple(evidence_ids)
            ),
        )
        rewrite_response_ids: list[str] = []
        for exchange in rewrite_exchanges:
            request_record = self._artifacts.add(
                PaperArtifactKind.OPTIMIZER_REQUEST,
                optimizer_request_payload(exchange.request),
                parent_ids=(
                    input_skill_id,
                    *evidence_ids,
                    update_record.artifact_id,
                ),
            )
            response_record = self._artifacts.add(
                PaperArtifactKind.OPTIMIZER_RESPONSE,
                optimizer_response_payload(exchange.response),
                parent_ids=(request_record.artifact_id,),
            )
            response_ids.append(response_record.artifact_id)
            rewrite_response_ids.append(response_record.artifact_id)
            response_ids_by_call[exchange.request.call_id] = (
                response_record.artifact_id
            )
        apply_record = self._artifacts.add(
            PaperArtifactKind.APPLY_REPORT,
            _apply_artifact_payload(
                fast_step,
                update_mode=self._plan.mechanisms.update_mode,
            ),
            parent_ids=(
                input_skill_id,
                update_record.artifact_id,
                *rewrite_response_ids,
            ),
        )
        candidate_record = self._artifacts.add(
            PaperArtifactKind.SKILL,
            {
                "role": "candidate",
                "skill_text": fast_step.apply_result.output_skill,
                "skill_sha256": fast_step.apply_result.output_sha256,
            },
            parent_ids=(apply_record.artifact_id,),
        )
        carried_score = (
            fast_step.skipped_stage is not None
            or fast_step.selection_skipped_reason is not None
        )
        score_record = self._artifacts.add(
            PaperArtifactKind.SELECTION_SCORE,
            {
                "role": (
                    "carried_current" if carried_score else "candidate"
                ),
                "score": fast_step.candidate_score.value,
                "skill_sha256": (
                    _sha256(fast_step.input_skill)
                    if carried_score
                    else fast_step.apply_result.output_sha256
                ),
            },
            parent_ids=(
                (
                    candidate_record.artifact_id,
                    self._require_artifact_head("current_score"),
                )
                if carried_score
                else (candidate_record.artifact_id, registry_id)
            ),
        )
        accepted = any(
            event.event_type is AlgorithmEventType.CANDIDATE_ACCEPTED
            for event in fast_step.events
        )
        self._record_fast_step_event_artifacts(
            fast_step.events,
            input_skill_id=input_skill_id,
            input_score_id=input_score_id,
            evidence_ids=tuple(evidence_ids),
            response_ids=tuple(response_ids),
            response_ids_by_call=response_ids_by_call,
            update_id=update_record.artifact_id,
            apply_id=apply_record.artifact_id,
            candidate_id=candidate_record.artifact_id,
            candidate_score_id=score_record.artifact_id,
        )
        if accepted:
            self._artifact_heads["current_skill"] = candidate_record.artifact_id
            self._artifact_heads["current_score"] = score_record.artifact_id

    def _record_fast_step_event_artifacts(
        self,
        events: tuple[AlgorithmEvent, ...],
        *,
        input_skill_id: str,
        input_score_id: str,
        evidence_ids: tuple[str, ...],
        response_ids: tuple[str, ...],
        response_ids_by_call: Mapping[str, str],
        update_id: str,
        apply_id: str,
        candidate_id: str,
        candidate_score_id: str,
    ) -> None:
        optimizer_events = {
            AlgorithmEventType.FAILURE_REFLECTED,
            AlgorithmEventType.SUCCESS_REFLECTED,
            AlgorithmEventType.ANALYST_REFINED,
            AlgorithmEventType.MERGE_FAILURE,
            AlgorithmEventType.MERGE_SUCCESS,
            AlgorithmEventType.MERGE_FINAL_FAILURE_PRIORITIZED,
            AlgorithmEventType.LEARNING_RATE_DECIDED,
            AlgorithmEventType.RANK_TOP_L,
        }
        apply_events = {
            AlgorithmEventType.PATCH_APPLIED,
            AlgorithmEventType.REWRITE_APPLIED,
            AlgorithmEventType.REWRITE_SKIPPED,
        }
        gate_events = {
            AlgorithmEventType.SELECTION_SCORED,
            AlgorithmEventType.CANDIDATE_ACCEPTED,
            AlgorithmEventType.CANDIDATE_REJECTED,
        }
        for event in events:
            contextual_parents: list[str] = [input_skill_id, input_score_id]
            if event.event_type is AlgorithmEventType.ROLLOUT_COLLECTED:
                accumulation_index = event.payload["accumulation_index"]
                contextual_parents.append(evidence_ids[accumulation_index - 1])
            elif event.event_type in optimizer_events:
                call_id = event.payload.get("call_id")
                if call_id is not None:
                    response_id = response_ids_by_call.get(call_id)
                    if response_id is None:
                        raise RuntimeError(
                            "optimizer event is missing its response artifact"
                        )
                    contextual_parents.append(response_id)
                if event.event_type is AlgorithmEventType.RANK_TOP_L:
                    contextual_parents.append(update_id)
            elif event.event_type in apply_events:
                contextual_parents.extend(
                    (update_id, apply_id, candidate_id)
                )
            elif event.event_type in gate_events:
                contextual_parents.extend((candidate_id, candidate_score_id))
                if event.payload.get("reason") == "semantic_stage_exhausted":
                    contextual_parents.extend(response_ids)
            elif event.event_type is not AlgorithmEventType.STEP_STARTED:
                raise RuntimeError("unexpected fast-step algorithm event")
            self._record_event_artifact(event, tuple(contextual_parents))

    def _record_event_artifacts(
        self,
        events: tuple[AlgorithmEvent, ...],
        *,
        additional_parent_ids: tuple[str, ...] = (),
    ) -> None:
        for event in events:
            self._record_event_artifact(
                event,
                tuple(
                    item
                    for item in (
                        self._artifact_heads.get("current_skill"),
                        self._artifact_heads.get("current_score"),
                        *additional_parent_ids,
                    )
                    if item is not None
                ),
            )

    def _record_event_artifact(
        self,
        event: AlgorithmEvent,
        contextual_parent_ids: tuple[str, ...],
    ) -> None:
        parent_ids = tuple(
            dict.fromkeys(
                item
                for item in (
                    self._artifact_heads.get("plan"),
                    self._artifact_heads.get("last_event"),
                    *contextual_parent_ids,
                )
                if item is not None
            )
        )
        record = self._artifacts.add(
            PaperArtifactKind.ALGORITHM_EVENT,
            event.to_dict(),
            parent_ids=parent_ids,
        )
        self._artifact_heads["last_event"] = record.artifact_id

    def _require_artifact_head(self, name: str) -> str:
        value = self._artifact_heads.get(name)
        if type(value) is not str:
            raise RuntimeError(f"paper artifact lineage is missing {name} head")
        return value

    def checkpoint(
        self,
        authenticator: CheckpointAuthenticator,
    ) -> PaperEpochCheckpoint:
        if type(authenticator) is not CheckpointAuthenticator:
            raise ValueError("checkpoint requires exact authenticator")
        return authenticator.sign(
            {
                "schema_version": "paper-epoch-runtime-v2",
                "profile_sha256": canonical_json_sha256(
                    self._profile.to_dict()
                ),
                "plan_sha256": self._plan_sha256,
                "controller_registry_sha256": (
                    self._controller.train.registry.sha256
                ),
                "fast_loop": self._fast_loop._checkpoint_payload(),
                "epoch_buffer": [
                    record.to_checkpoint_dict() for record in self._epoch_buffer
                ],
                "epoch_snapshots": list(self._epoch_snapshots),
                "epoch_snapshot_artifact_ids": list(
                    self._epoch_snapshot_artifact_ids
                ),
                "events": [event.to_dict() for event in self._events],
                "run_completed": self._run_completed,
                "artifact_lineage": (
                    self._artifacts.lineage.to_checkpoint_list()
                ),
                "artifact_heads": dict(self._artifact_heads),
            }
        )

    @classmethod
    def resume(
        cls,
        controller: PaperOptimizationController,
        *,
        profile: PaperProfile,
        plan: PaperEpochPlan,
        checkpoint: PaperEpochCheckpoint,
        authenticator: CheckpointAuthenticator,
    ) -> "PaperEpochLoop":
        if type(authenticator) is not CheckpointAuthenticator:
            raise ValueError("resume requires exact checkpoint authenticator")
        payload = authenticator.verify(checkpoint)
        loop = cls(controller, profile=profile, plan=plan)
        loop._restore_checkpoint_payload(payload)
        return loop

    def _restore_checkpoint_payload(self, payload: Mapping[str, object]) -> None:
        expected = {
            "schema_version",
            "profile_sha256",
            "plan_sha256",
            "controller_registry_sha256",
            "fast_loop",
            "epoch_buffer",
            "epoch_snapshots",
            "epoch_snapshot_artifact_ids",
            "events",
            "run_completed",
            "artifact_lineage",
            "artifact_heads",
        }
        if type(payload) is not dict or set(payload) != expected:
            raise ValueError("invalid paper epoch runtime checkpoint")
        if payload["schema_version"] != "paper-epoch-runtime-v2":
            raise ValueError("unsupported paper epoch runtime checkpoint")
        identities = (
            payload["profile_sha256"]
            == canonical_json_sha256(self._profile.to_dict()),
            payload["plan_sha256"] == self._plan_sha256,
            payload["controller_registry_sha256"]
            == self._controller.train.registry.sha256,
        )
        if not all(identities):
            raise ValueError("checkpoint identity does not match runtime")
        if type(payload["run_completed"]) is not bool:
            raise ValueError("invalid checkpoint completion state")
        if type(payload["fast_loop"]) is not dict:
            raise ValueError("invalid fast-loop checkpoint payload")
        fast_state = payload["fast_loop"].get("state")
        if type(fast_state) is not dict:
            raise ValueError("invalid checkpoint scheduler state")
        epoch = fast_state.get("epoch")
        step = fast_state.get("step")
        if (
            type(epoch) is not int
            or type(step) is not int
            or not 1 <= epoch <= self._plan.epochs
            or not 0 <= step <= self._plan.steps_per_epoch
        ):
            raise ValueError("checkpoint scheduler state is outside frozen plan")
        if payload["run_completed"] and (
            epoch != self._plan.epochs
            or step != self._plan.steps_per_epoch
        ):
            raise ValueError("checkpoint completion state is inconsistent")
        self._run_completed = payload["run_completed"]
        self._artifacts = PaperArtifactLedger.from_checkpoint_list(
            payload["artifact_lineage"]
        )
        if type(payload["artifact_heads"]) is not dict or set(
            payload["artifact_heads"]
        ) != {
            "profile",
            "plan",
            "registry",
            "current_skill",
            "current_score",
            "meta_skill",
            "last_event",
        }:
            raise ValueError("invalid artifact lineage checkpoint heads")
        artifact_ids = {
            record.artifact_id for record in self._artifacts.lineage.records
        }
        if any(
            value is not None
            and (type(value) is not str or value not in artifact_ids)
            for value in payload["artifact_heads"].values()
        ):
            raise ValueError("artifact lineage checkpoint head is unknown")
        self._artifact_heads = dict(payload["artifact_heads"])
        self._fast_loop._restore_authenticated_checkpoint(payload["fast_loop"])
        if type(payload["epoch_buffer"]) is not list:
            raise ValueError("invalid epoch buffer checkpoint payload")
        self._epoch_buffer = [
            EpochBufferRecord.from_checkpoint_mapping(item)
            for item in payload["epoch_buffer"]
        ]
        state = self._fast_loop.state
        if len(self._epoch_buffer) != state.step or any(
            record.epoch != state.epoch
            or record.step != index
            for index, record in enumerate(self._epoch_buffer, 1)
        ):
            raise ValueError("checkpoint epoch buffer disagrees with scheduler state")
        if type(payload["epoch_snapshots"]) is not list or any(
            type(item) is not str or not item.strip()
            for item in payload["epoch_snapshots"]
        ):
            raise ValueError("invalid epoch snapshots checkpoint payload")
        self._epoch_snapshots = list(payload["epoch_snapshots"])
        expected_snapshots = state.epoch if self._run_completed else state.epoch - 1
        if len(self._epoch_snapshots) != expected_snapshots:
            raise ValueError("checkpoint snapshots disagree with current epoch")
        if type(payload["epoch_snapshot_artifact_ids"]) is not list or any(
            type(item) is not str or item not in artifact_ids
            for item in payload["epoch_snapshot_artifact_ids"]
        ):
            raise ValueError("invalid epoch snapshot artifact IDs")
        self._epoch_snapshot_artifact_ids = list(
            payload["epoch_snapshot_artifact_ids"]
        )
        if len(self._epoch_snapshot_artifact_ids) != expected_snapshots:
            raise ValueError("snapshot artifact IDs disagree with current epoch")
        for skill_text, artifact_id in zip(
            self._epoch_snapshots,
            self._epoch_snapshot_artifact_ids,
        ):
            snapshot_record = next(
                record
                for record in self._artifacts.lineage.records
                if record.artifact_id == artifact_id
            )
            if (
                snapshot_record.kind is not PaperArtifactKind.SKILL
                or snapshot_record.payload["skill_text"] != skill_text
            ):
                raise ValueError("epoch snapshot artifact is inconsistent")
        if type(payload["events"]) is not list:
            raise ValueError("invalid checkpoint event list")
        self._events = [AlgorithmEvent.from_dict(item) for item in payload["events"]]
        records_by_id = {
            record.artifact_id: record
            for record in self._artifacts.lineage.records
        }
        skill_head = records_by_id[self._require_artifact_head("current_skill")]
        score_head = records_by_id[self._require_artifact_head("current_score")]
        artifact_events = self._artifacts.lineage.records_of_kind(
            PaperArtifactKind.ALGORITHM_EVENT
        )
        if (
            skill_head.kind is not PaperArtifactKind.SKILL
            or skill_head.payload["skill_text"] != state.current_skill
            or score_head.kind is not PaperArtifactKind.SELECTION_SCORE
            or score_head.payload["score"] != state.current_score.value
            or len(artifact_events) != len(self._events)
            or [record.payload for record in artifact_events]
            != [event.to_dict() for event in self._events]
        ):
            raise ValueError("artifact lineage checkpoint disagrees with runtime state")
        meta_head = self._artifact_heads["meta_skill"]
        if (state.meta_skill and meta_head is None) or (
            not state.meta_skill and meta_head is not None
        ):
            raise ValueError("artifact lineage meta head disagrees with runtime")
        if [event.sequence for event in self._events] != list(
            range(len(self._events))
        ) or self._fast_loop.next_event_sequence != len(self._events):
            raise ValueError("checkpoint event sequence is not contiguous")
        if self._run_completed and (
            state.epoch != self._plan.epochs
            or state.step != self._plan.steps_per_epoch
            or not self._events
            or self._events[-1].event_type is not AlgorithmEventType.RUN_COMPLETED
        ):
            raise ValueError("checkpoint completion state is inconsistent")
        if (
            not self._run_completed
            and self._events
            and self._events[-1].event_type is AlgorithmEventType.RUN_COMPLETED
        ):
            raise ValueError("checkpoint completion event is inconsistent")
        self._fast_loop._set_epoch_buffer(tuple(self._epoch_buffer))


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _optimizer_stage_parent_ids(
    request: OptimizerRequest,
    *,
    completed_exchanges: tuple[tuple[OptimizerRequest, str], ...],
) -> tuple[str, ...]:
    """Return response artifacts actually consumed by a later optimizer stage."""

    by_call_id = {
        completed.call_id: response_id
        for completed, response_id in completed_exchanges
    }
    stage = request.stage
    if stage is OptimizerStage.REFINE:
        refine_prefix, separator, raw_round = request.call_id.rpartition("-r")
        if not separator or not raw_round.isdigit():
            raise RuntimeError("refine call ID cannot be linked to its input")
        round_number = int(raw_round)
        if round_number == 1:
            dependency_call_id = refine_prefix.replace(
                "-refine-",
                "-reflect-",
                1,
            )
        else:
            dependency_call_id = f"{refine_prefix}-r{round_number - 1}"
        response_id = by_call_id.get(dependency_call_id)
        if response_id is None:
            raise RuntimeError("refine lineage is missing its prior response")
        return (response_id,)

    if stage in {
        OptimizerStage.MERGE_FAILURE,
        OptimizerStage.MERGE_SUCCESS,
    }:
        source = (
            "failure"
            if stage is OptimizerStage.MERGE_FAILURE
            else "success"
        )
        source_reflect_stage = (
            OptimizerStage.REFLECT_FAILURE
            if source == "failure"
            else OptimizerStage.REFLECT_SUCCESS
        )
        return tuple(
            response_id
            for completed, response_id in completed_exchanges
            if completed.stage in {source_reflect_stage, stage}
            or (
                completed.stage is OptimizerStage.REFINE
                and f"-refine-{source}-" in completed.call_id
            )
        )

    upstream_stages: set[OptimizerStage]
    if stage is OptimizerStage.MERGE_FINAL_FAILURE_PRIORITIZED:
        upstream_stages = {
            OptimizerStage.MERGE_FAILURE,
            OptimizerStage.MERGE_SUCCESS,
            OptimizerStage.MERGE_FINAL_FAILURE_PRIORITIZED,
        }
    elif stage is OptimizerStage.DECIDE_LEARNING_RATE:
        upstream_stages = {
            OptimizerStage.MERGE_FINAL_FAILURE_PRIORITIZED,
        }
    elif stage is OptimizerStage.RANK_TOP_L:
        upstream_stages = {
            OptimizerStage.MERGE_FINAL_FAILURE_PRIORITIZED,
            OptimizerStage.DECIDE_LEARNING_RATE,
            OptimizerStage.RANK_TOP_L,
        }
    else:
        return ()
    return tuple(
        response_id
        for completed, response_id in completed_exchanges
        if completed.stage in upstream_stages
    )


def _edit_artifact_payload(edit: PaperEdit) -> dict[str, Any]:
    return {
        "edit_id": edit.edit_id,
        "operation": edit.operation.value,
        "target": edit.target,
        "content": edit.content,
        "rationale": edit.rationale,
        "support_count": edit.support_count,
        "source_type": (
            edit.source_type.value if edit.source_type is not None else None
        ),
    }


def _suggestion_artifact_payload(
    suggestion: PaperSuggestion,
) -> dict[str, Any]:
    return {
        "suggestion_id": suggestion.suggestion_id,
        "suggestion_type": suggestion.suggestion_type.value,
        "title": suggestion.title,
        "motivation": suggestion.motivation,
        "instruction": suggestion.instruction,
        "priority_hint": suggestion.priority_hint.value,
        "support_count": suggestion.support_count,
        "source_type": (
            suggestion.source_type.value
            if suggestion.source_type is not None
            else None
        ),
    }


def _apply_artifact_payload(
    fast_step: FastStepResult,
    *,
    update_mode: str,
) -> dict[str, Any]:
    return {
        "update_mode": update_mode,
        "input_skill": fast_step.input_skill,
        "input_sha256": fast_step.apply_result.input_sha256,
        "output_skill": fast_step.apply_result.output_skill,
        "output_sha256": fast_step.apply_result.output_sha256,
        "reports": [
            {
                "index": report.index,
                "edit_id": report.edit_id,
                "operation": report.operation.value,
                "status": report.status,
                "before_sha256": report.before_sha256,
                "after_sha256": report.after_sha256,
            }
            for report in fast_step.apply_result.reports
        ],
        "rewrite": (
            {
                "reasoning": fast_step.rewrite_result.reasoning,
                "change_summary": list(
                    fast_step.rewrite_result.change_summary
                ),
            }
            if fast_step.rewrite_result is not None
            else None
        ),
    }
