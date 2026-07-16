"""Role-authenticated data-firewall types for the paper optimizer."""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping

from .controller_process import (
    ControllerRegistry,
    ControllerRole,
    canonical_json,
    canonical_json_sha256,
    invoke_optimization_controller,
    parse_signed_response,
    require_exact_keys,
    require_finite_scalar,
)
from .errors import DataFirewallViolation


class SplitRole(str, Enum):
    TRAIN = "train"
    SELECTION = "selection"
    TEST = "test"


class RunPhase(str, Enum):
    OPTIMIZATION = "optimization"
    FINAL_EVALUATION = "final_evaluation"


@dataclass(frozen=True)
class PaperDataAccessPolicy:
    """Fail-closed access matrix for optimization and final evaluation."""

    def require(self, *, split: SplitRole, phase: RunPhase) -> None:
        allowed = {
            RunPhase.OPTIMIZATION: {SplitRole.TRAIN, SplitRole.SELECTION},
            RunPhase.FINAL_EVALUATION: {SplitRole.TEST},
        }
        if split not in allowed[phase]:
            raise DataFirewallViolation(
                f"{split.value} split is forbidden during {phase.value}"
            )


@dataclass(frozen=True)
class SelectionScore:
    """The complete selection-to-optimizer result: one finite scalar."""

    value: float

    def __post_init__(self) -> None:
        require_finite_scalar(self.value, context="selection score")

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "SelectionScore":
        if type(payload) is not dict or set(payload) != {"score"}:
            forbidden = (
                "; forbidden fields: "
                + ", ".join(sorted(set(payload) - {"score"}))
                if isinstance(payload, Mapping) and set(payload) - {"score"}
                else ""
            )
            raise DataFirewallViolation(
                "selection response must contain exactly one scalar field: score"
                + forbidden
            )
        return cls(value=require_finite_scalar(payload["score"], context="selection score"))

    def to_payload(self) -> dict[str, float]:
        return {"score": float(self.value)}


@dataclass(frozen=True)
class SelectionController:
    """Invoke one registry-authenticated selection owner and retain one scalar."""

    registry: ControllerRegistry
    controller_id: str

    def __post_init__(self) -> None:
        if type(self.registry) is not ControllerRegistry:
            raise DataFirewallViolation("selection requires exact controller registry")
        self.registry.require(self.controller_id, role=ControllerRole.SELECTION)

    def score(self, skill_text: str) -> SelectionScore:
        self.__post_init__()
        if type(skill_text) is not str or not skill_text.strip():
            raise DataFirewallViolation("selection requires skill_text")
        registration = self.registry.require(
            self.controller_id, role=ControllerRole.SELECTION
        )
        response, _ = invoke_optimization_controller(
            registry=self.registry,
            controller_id=self.controller_id,
            role=ControllerRole.SELECTION,
            request={
                "operation": "score_selection",
                "skill_text": skill_text,
                "split_id": registration.split_id,
                "split_manifest_sha256": registration.artifact(
                    "split_manifest"
                ).sha256,
            },
        )
        return SelectionScore.from_payload(response)


@dataclass(frozen=True)
class TrainEvidenceBatch:
    """Signed response retained for independent optimizer-seam verification."""

    controller_id: str
    registry_sha256: str
    split_id: str
    split_manifest_sha256: str
    canonical_request: str
    canonical_payload: str
    signature: str

    def __post_init__(self) -> None:
        if type(self.controller_id) is not str or not self.controller_id:
            raise DataFirewallViolation("train evidence requires controller_id")
        for name in ("registry_sha256", "split_manifest_sha256"):
            value = getattr(self, name)
            if type(value) is not str or len(value) != 64:
                raise DataFirewallViolation(f"train evidence requires {name}")
        if type(self.split_id) is not str or not self.split_id:
            raise DataFirewallViolation("train evidence requires split_id")
        for name in ("canonical_request", "canonical_payload", "signature"):
            if type(getattr(self, name)) is not str:
                raise DataFirewallViolation(f"train evidence requires {name}")


@dataclass(frozen=True)
class TrainController:
    """Invoke and authenticate the registry's sole train data owner."""

    registry: ControllerRegistry
    controller_id: str

    def __post_init__(self) -> None:
        if type(self.registry) is not ControllerRegistry:
            raise DataFirewallViolation("train requires exact controller registry")
        self.registry.require(self.controller_id, role=ControllerRole.TRAIN)

    def collect(
        self,
        skill_text: str,
        *,
        batch_id: str | None = None,
        batch_seed: int | None = None,
        batch_size: int | None = None,
    ) -> TrainEvidenceBatch:
        self.__post_init__()
        if type(skill_text) is not str or not skill_text.strip():
            raise DataFirewallViolation("train collection requires skill_text")
        _validate_scheduled_batch(
            batch_id=batch_id,
            batch_seed=batch_seed,
            batch_size=batch_size,
        )
        registration = self.registry.require(
            self.controller_id, role=ControllerRole.TRAIN
        )
        request = {
            "operation": "collect_train",
            "skill_text": skill_text,
            "split_id": registration.split_id,
            "split_manifest_sha256": registration.artifact(
                "split_manifest"
            ).sha256,
        }
        if batch_id is not None:
            request["batch_id"] = batch_id
            request["batch_seed"] = batch_seed
            request["batch_size"] = batch_size
        response, signature = invoke_optimization_controller(
            registry=self.registry,
            controller_id=self.controller_id,
            role=ControllerRole.TRAIN,
            request=request,
        )
        _validated_train_payload(
            response,
            expected_split_id=registration.split_id,
            expected_split_manifest_sha256=registration.artifact(
                "split_manifest"
            ).sha256,
            expected_batch_id=batch_id,
            expected_batch_seed=batch_seed,
            expected_batch_size=batch_size,
        )
        # Preserve the exact signed payload. Validation above rejects every
        # side-channel field before evidence can be created.
        return TrainEvidenceBatch(
            controller_id=self.controller_id,
            registry_sha256=self.registry.sha256,
            split_id=registration.split_id,
            split_manifest_sha256=registration.artifact(
                "split_manifest"
            ).sha256,
            canonical_request=canonical_json(request),
            canonical_payload=canonical_json(response),
            signature=signature,
        )

    def verify(
        self,
        evidence: TrainEvidenceBatch,
        *,
        current_skill: str,
        batch_id: str | None = None,
        batch_seed: int | None = None,
        batch_size: int | None = None,
    ) -> tuple[dict[str, Any], ...]:
        self.__post_init__()
        if type(evidence) is not TrainEvidenceBatch:
            raise DataFirewallViolation("optimizer requires exact TrainEvidenceBatch")
        evidence.__post_init__()
        registration = self.registry.require(
            self.controller_id, role=ControllerRole.TRAIN
        )
        _validate_scheduled_batch(
            batch_id=batch_id,
            batch_seed=batch_seed,
            batch_size=batch_size,
        )
        if (
            evidence.controller_id != self.controller_id
            or evidence.registry_sha256 != self.registry.sha256
            or evidence.split_id != registration.split_id
            or evidence.split_manifest_sha256
            != registration.artifact("split_manifest").sha256
        ):
            raise DataFirewallViolation("train evidence authority does not match registry")
        expected_request = {
            "operation": "collect_train",
            "skill_text": current_skill,
            "split_id": registration.split_id,
            "split_manifest_sha256": registration.artifact(
                "split_manifest"
            ).sha256,
        }
        if batch_id is not None:
            expected_request["batch_id"] = batch_id
            expected_request["batch_seed"] = batch_seed
            expected_request["batch_size"] = batch_size
        if evidence.canonical_request != canonical_json(expected_request):
            raise DataFirewallViolation(
                "train evidence is not bound to current_skill and scheduled batch"
            )
        try:
            payload = json.loads(evidence.canonical_payload)
        except json.JSONDecodeError as error:
            raise DataFirewallViolation("train evidence payload is not JSON") from error
        envelope = {
            "controller_id": evidence.controller_id,
            "request_sha256": canonical_json_sha256(expected_request),
            "payload": payload,
            "signature": evidence.signature,
        }
        verified_payload, _ = parse_signed_response(
            registration=registration,
            request=expected_request,
            stdout=canonical_json(envelope),
        )
        return _validated_train_payload(
            verified_payload,
            expected_split_id=registration.split_id,
            expected_split_manifest_sha256=registration.artifact(
                "split_manifest"
            ).sha256,
            expected_batch_id=batch_id,
            expected_batch_seed=batch_seed,
            expected_batch_size=batch_size,
        )


@dataclass(frozen=True)
class OptimizerPayload:
    """Optimizer input whose train origin is verified at the backend seam."""

    current_skill: str
    train_evidence: TrainEvidenceBatch

    def __post_init__(self) -> None:
        if type(self.current_skill) is not str or not self.current_skill.strip():
            raise DataFirewallViolation("optimizer payload requires current_skill")
        if type(self.train_evidence) is not TrainEvidenceBatch:
            raise DataFirewallViolation("optimizer payload requires TrainEvidenceBatch")
        self.train_evidence.__post_init__()


@dataclass(frozen=True)
class SelectionDecision:
    current: SelectionScore
    candidate: SelectionScore
    accepted: bool

    @property
    def delta(self) -> float:
        return float(self.candidate.value - self.current.value)


def strict_selection_decision(
    *, current: SelectionScore, candidate: SelectionScore
) -> SelectionDecision:
    """Apply the paper comparator; equality is always a rejection."""

    return SelectionDecision(
        current=current,
        candidate=candidate,
        accepted=candidate.value > current.value,
    )


def _validated_train_payload(
    value: object,
    *,
    expected_split_id: str,
    expected_split_manifest_sha256: str,
    expected_batch_id: str | None = None,
    expected_batch_seed: int | None = None,
    expected_batch_size: int | None = None,
) -> tuple[dict[str, Any], ...]:
    expected_fields = {"split_id", "split_manifest_sha256", "trajectories"}
    if expected_batch_id is not None:
        expected_fields.update({"batch_id", "batch_seed", "batch_size"})
    require_exact_keys(
        value,
        expected_fields,
        context="train response",
    )
    if value["split_id"] != expected_split_id:
        raise DataFirewallViolation("train response split_id does not match registry")
    if value["split_manifest_sha256"] != expected_split_manifest_sha256:
        raise DataFirewallViolation(
            "train response split manifest does not match registry"
        )
    if expected_batch_id is not None:
        if (
            type(value["batch_id"]) is not str
            or type(value["batch_seed"]) is not int
            or type(value["batch_size"]) is not int
            or value["batch_id"] != expected_batch_id
            or value["batch_seed"] != expected_batch_seed
            or value["batch_size"] != expected_batch_size
        ):
            raise DataFirewallViolation("train response batch plan does not match request")
    raw = value["trajectories"]
    if type(raw) is not list or not raw:
        raise DataFirewallViolation("train response requires trajectories")
    if expected_batch_size is not None and len(raw) != expected_batch_size:
        raise DataFirewallViolation(
            "train response trajectory count does not match scheduled batch_size"
        )
    normalized = tuple(_normalize_train_trajectory(item) for item in raw)
    task_ids = [item["task_id"] for item in normalized]
    if len(task_ids) != len(set(task_ids)):
        raise DataFirewallViolation("train response task IDs must be unique")
    return normalized


def _validate_scheduled_batch(
    *,
    batch_id: str | None,
    batch_seed: int | None,
    batch_size: int | None,
) -> None:
    values = (batch_id, batch_seed, batch_size)
    if all(value is None for value in values):
        return
    if any(value is None for value in values):
        raise DataFirewallViolation(
            "scheduled train batch requires id, seed, and size together"
        )
    if type(batch_id) is not str or not batch_id.strip():
        raise DataFirewallViolation("train collection batch_id must be non-empty")
    if type(batch_seed) is not int or batch_seed < 0:
        raise DataFirewallViolation("train collection batch_seed must be non-negative")
    if type(batch_size) is not int or batch_size < 1:
        raise DataFirewallViolation("train collection batch_size must be positive")


def _normalize_train_trajectory(value: object) -> dict[str, Any]:
    expected = {"task_id", "task_input", "output", "score", "success", "trace"}
    require_exact_keys(value, expected, context="train trajectory")
    task_id = value["task_id"]
    if type(task_id) is not str or not task_id.strip():
        raise DataFirewallViolation("train trajectory requires task_id")
    score = require_finite_scalar(value["score"], context="train trajectory score")
    if type(value["success"]) is not bool:
        raise DataFirewallViolation("train trajectory success must be boolean")
    trace = value["trace"]
    if type(trace) is not list or any(type(item) is not str for item in trace):
        raise DataFirewallViolation("train trajectory trace must be a string list")
    task_io = json.loads(
        canonical_json(
            {"task_input": value["task_input"], "output": value["output"]}
        )
    )
    return {
        "task_id": task_id,
        "task_input": task_io["task_input"],
        "output": task_io["output"],
        "score": score,
        "success": value["success"],
        "trace": list(trace),
    }
