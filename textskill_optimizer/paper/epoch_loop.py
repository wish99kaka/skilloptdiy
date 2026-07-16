"""Stateful epoch owner for paper-faithful optimization."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping

from .backend import OptimizerRequest, OptimizerResponse, OptimizerStage
from .checkpoint import CheckpointAuthenticator, PaperEpochCheckpoint
from .config import PaperProfile
from .controller_process import ControllerRole
from .data import SelectionScore, TrainEvidenceBatch
from .epoch_plan import EpochCursor, PaperEpochPlan
from .fast_loop import FastStepResult, PaperFastLoop
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
    PaperState,
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
        self._fast_loop = PaperFastLoop(controller, profile=validated_profile)
        self._events: list[AlgorithmEvent] = []
        self._epoch_buffer: list[EpochBufferRecord] = []
        self._epoch_snapshots: list[str] = []
        self._run_completed = False

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
    def longitudinal_skills(self) -> tuple[str, str]:
        state = self._fast_loop.state
        if (
            state.epoch < self._profile.slow_update.start_epoch
            or state.step != self._plan.steps_per_epoch
            or len(self._epoch_snapshots) != state.epoch - 1
        ):
            raise ValueError("longitudinal skills are available only at an epoch boundary")
        return self._epoch_snapshots[-1], state.current_skill

    def collect_train_evidence(self) -> TrainEvidenceBatch:
        if self._run_completed:
            raise ValueError("paper epoch run is complete")
        state = self._fast_loop.state
        next_step = state.step + 1
        if next_step > self._plan.steps_per_epoch:
            raise ValueError(f"epoch {state.epoch} is complete")
        cursor = self._plan.cursor(epoch=state.epoch, step=next_step)
        return self._controller.train.collect(
            state.current_skill,
            batch_id=cursor.batch_id,
            batch_seed=cursor.batch_seed,
            batch_size=cursor.batch_size,
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
        return state

    def run_step(self, *, train_evidence: TrainEvidenceBatch) -> EpochStepResult:
        if self._run_completed:
            raise ValueError("paper epoch run is complete")
        state = self._fast_loop.state
        next_step = state.step + 1
        if next_step > self._plan.steps_per_epoch:
            raise ValueError(f"epoch {state.epoch} is complete")
        cursor = self._plan.cursor(epoch=state.epoch, step=next_step)
        self._fast_loop._prepare_epoch_step(
            tuple(self._epoch_buffer),
            batch_id=cursor.batch_id,
            batch_seed=cursor.batch_seed,
            batch_size=cursor.batch_size,
        )
        fast_step = self._fast_loop.run_step(
            train_evidence=train_evidence,
            edit_budget=cursor.edit_budget,
        )
        self._events.extend(fast_step.events)
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
        self._epoch_buffer.append(
            EpochBufferRecord(
                epoch=state.epoch,
                step=next_step,
                failure_patterns=fast_step.failure_patterns,
                rejected_edits=fast_step.ranked_edits if rejected else (),
                score_delta=(
                    float(decision_event.payload["delta"])
                    if rejected
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
        else:
            if type(longitudinal_evidence) is not LongitudinalEvidence:
                raise ValueError(
                    f"epoch {state.epoch} requires authenticated longitudinal evidence"
                )
            boundary_events = list(
                self._run_slow_meta(longitudinal_evidence)
            )
            state = self._fast_loop.state
        self._epoch_snapshots.append(fast_epoch_end_skill)
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

    def checkpoint(
        self,
        authenticator: CheckpointAuthenticator,
    ) -> PaperEpochCheckpoint:
        if type(authenticator) is not CheckpointAuthenticator:
            raise ValueError("checkpoint requires exact authenticator")
        return authenticator.sign(
            {
                "schema_version": "paper-epoch-runtime-v1",
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
                "events": [event.to_dict() for event in self._events],
                "run_completed": self._run_completed,
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
            "events",
            "run_completed",
        }
        if type(payload) is not dict or set(payload) != expected:
            raise ValueError("invalid paper epoch runtime checkpoint")
        if payload["schema_version"] != "paper-epoch-runtime-v1":
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
        if type(payload["events"]) is not list:
            raise ValueError("invalid checkpoint event list")
        self._events = [AlgorithmEvent.from_dict(item) for item in payload["events"]]
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
