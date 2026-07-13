"""Machine-readable configuration contract for paper-faithful runs."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from importlib.resources import files
from pathlib import Path
from typing import Any, Mapping

from .schema_validation import validate_schema


class PaperProfileViolation(ValueError):
    """Raised when input cannot represent a paper-faithful profile."""


@dataclass(frozen=True)
class ProfileViolation:
    code: str
    path: str
    message: str


@dataclass(frozen=True)
class PaperProfileAssessment:
    profile: "PaperProfile | None"
    violations: tuple[ProfileViolation, ...]

    @property
    def compliant(self) -> bool:
        return not self.violations

    def require(self) -> "PaperProfile":
        if self.profile is None or self.violations:
            details = "; ".join(
                f"{item.path}: {item.message}" for item in self.violations
            )
            raise PaperProfileViolation(f"paper profile is not compliant: {details}")
        return self.profile


@dataclass(frozen=True)
class SelectionGateConfig:
    enabled: bool
    metric: str
    comparator: str


@dataclass(frozen=True)
class RejectedBufferConfig:
    enabled: bool
    scope: str


@dataclass(frozen=True)
class SlowUpdateConfig:
    enabled: bool
    start_epoch: int
    sample_size: int
    selection_gated: bool


@dataclass(frozen=True)
class MetaSkillConfig:
    enabled: bool
    start_epoch: int
    initial: str
    target_visible: bool


@dataclass(frozen=True)
class PaperProfile:
    """Validated paper profile exposed to the future paper engine."""

    profile: str
    protocol_id: str
    epochs: int
    split_seed: int
    default_split_ratio: str
    rollout_batch_size: int
    accumulation: int
    reflection_minibatch_size: int
    merge_batch_size: int
    analyst_workers: int
    max_analyst_rounds: int
    update_mode: str
    learning_rate: int
    learning_rate_floor: int
    learning_rate_schedule: str
    selection_gate: SelectionGateConfig
    rejected_buffer: RejectedBufferConfig
    slow_update: SlowUpdateConfig
    meta_skill: MetaSkillConfig
    early_stop: bool

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "PaperProfile":
        return assess_paper_profile(payload).require()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_EXPECTED_FIELDS: dict[str, frozenset[str]] = {
    "": frozenset(
        {
            "$schema",
            "profile",
            "protocol_id",
            "epochs",
            "split_seed",
            "default_split_ratio",
            "rollout_batch_size",
            "accumulation",
            "reflection_minibatch_size",
            "merge_batch_size",
            "analyst_workers",
            "max_analyst_rounds",
            "update_mode",
            "learning_rate",
            "learning_rate_floor",
            "learning_rate_schedule",
            "selection_gate",
            "rejected_buffer",
            "slow_update",
            "meta_skill",
            "early_stop",
        }
    ),
    "selection_gate": frozenset({"enabled", "metric", "comparator"}),
    "rejected_buffer": frozenset({"enabled", "scope"}),
    "slow_update": frozenset(
        {"enabled", "start_epoch", "sample_size", "selection_gated"}
    ),
    "meta_skill": frozenset({"enabled", "start_epoch", "initial", "target_visible"}),
}

_FORBIDDEN_EXTENSION_FIELDS = frozenset(
    {
        "benchmark_specific_prompt_rules",
        "confirmation_rounds",
        "contract_feedback",
        "contract_guard",
        "cooldown",
        "force_accept",
        "mixed_gate",
        "paired_confirmation",
        "selection_feedback_to_optimizer",
        "soft_gate",
        "targeting",
        "validation_confirmation_rounds",
        "validation_mean_delta",
        "validation_required_wins",
    }
)


def assess_paper_profile(payload: Mapping[str, Any]) -> PaperProfileAssessment:
    """Classify profile input without invoking an optimizer or mutating state."""

    violations: list[ProfileViolation] = []
    _inspect_fields(payload, "", violations)
    if violations:
        return PaperProfileAssessment(profile=None, violations=tuple(violations))
    schema_path = files("textskill_optimizer.paper").joinpath(
        "schemas", "paper-profile-v1.schema.json"
    )
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    schema_violations = validate_schema(payload, schema)
    if schema_violations:
        return PaperProfileAssessment(
            profile=None,
            violations=tuple(
                ProfileViolation(
                    code="invalid_profile_shape",
                    path=item.path,
                    message=item.message,
                )
                for item in schema_violations
            ),
        )
    try:
        profile = _profile_from_mapping(payload)
    except (KeyError, TypeError, ValueError) as error:
        return PaperProfileAssessment(
            profile=None,
            violations=(
                ProfileViolation(
                    code="invalid_profile_shape",
                    path="profile",
                    message=str(error),
                ),
            ),
        )
    violations.extend(_validate_invariants(profile))
    violations.extend(_validate_frozen_profile(profile))
    return PaperProfileAssessment(profile=profile, violations=tuple(violations))


def _inspect_fields(
    payload: Mapping[str, Any],
    parent: str,
    violations: list[ProfileViolation],
) -> None:
    expected = _EXPECTED_FIELDS.get(parent, frozenset())
    for key, value in payload.items():
        path = f"{parent}.{key}" if parent else key
        if key in _FORBIDDEN_EXTENSION_FIELDS:
            violations.append(
                ProfileViolation(
                    code="forbidden_extension_control",
                    path=path,
                    message="extension controls cannot enter a paper profile",
                )
            )
        elif key not in expected:
            violations.append(
                ProfileViolation(
                    code="unknown_profile_field",
                    path=path,
                    message="field is not part of paper-faithful-v1",
                )
            )
        if isinstance(value, Mapping):
            _inspect_fields(value, path, violations)


def _profile_from_mapping(payload: Mapping[str, Any]) -> PaperProfile:
    return PaperProfile(
        profile=str(payload["profile"]),
        protocol_id=str(payload["protocol_id"]),
        epochs=int(payload["epochs"]),
        split_seed=int(payload["split_seed"]),
        default_split_ratio=str(payload["default_split_ratio"]),
        rollout_batch_size=int(payload["rollout_batch_size"]),
        accumulation=int(payload["accumulation"]),
        reflection_minibatch_size=int(payload["reflection_minibatch_size"]),
        merge_batch_size=int(payload["merge_batch_size"]),
        analyst_workers=int(payload["analyst_workers"]),
        max_analyst_rounds=int(payload["max_analyst_rounds"]),
        update_mode=str(payload["update_mode"]),
        learning_rate=int(payload["learning_rate"]),
        learning_rate_floor=int(payload["learning_rate_floor"]),
        learning_rate_schedule=str(payload["learning_rate_schedule"]),
        selection_gate=SelectionGateConfig(**payload["selection_gate"]),
        rejected_buffer=RejectedBufferConfig(**payload["rejected_buffer"]),
        slow_update=SlowUpdateConfig(**payload["slow_update"]),
        meta_skill=MetaSkillConfig(**payload["meta_skill"]),
        early_stop=bool(payload["early_stop"]),
    )


def _validate_invariants(profile: PaperProfile) -> list[ProfileViolation]:
    violations: list[ProfileViolation] = []
    required_values = {
        "profile": (profile.profile, "paper-faithful-v1"),
        "protocol_id": (profile.protocol_id, "paper-faithful-v1"),
        "selection_gate.enabled": (profile.selection_gate.enabled, True),
        "selection_gate.metric": (profile.selection_gate.metric, "benchmark_native"),
        "selection_gate.comparator": (
            profile.selection_gate.comparator,
            "strict_greater",
        ),
        "rejected_buffer.scope": (profile.rejected_buffer.scope, "epoch"),
        "slow_update.selection_gated": (profile.slow_update.selection_gated, True),
        "meta_skill.target_visible": (profile.meta_skill.target_visible, False),
        "early_stop": (profile.early_stop, False),
    }
    for path, (actual, expected) in required_values.items():
        if actual != expected:
            violations.append(
                ProfileViolation(
                    code="paper_invariant_violation",
                    path=path,
                    message=f"must equal {expected!r}, got {actual!r}",
                )
            )
    return violations


def _validate_frozen_profile(profile: PaperProfile) -> list[ProfileViolation]:
    default_path = files("textskill_optimizer.paper").joinpath(
        "profiles", "paper-faithful-v1.json"
    )
    expected = json.loads(default_path.read_text(encoding="utf-8"))
    expected.pop("$schema", None)
    actual = profile.to_dict()
    violations: list[ProfileViolation] = []
    for path, expected_value in _leaf_values(expected):
        actual_value: Any = actual
        for segment in path.split("."):
            actual_value = actual_value[segment]
        if actual_value != expected_value:
            violations.append(
                ProfileViolation(
                    code="unregistered_profile_override",
                    path=path,
                    message=(
                        f"paper-faithful-v1 freezes {expected_value!r}; "
                        f"got {actual_value!r}"
                    ),
                )
            )
    return violations


def _leaf_values(payload: Mapping[str, Any], parent: str = "") -> list[tuple[str, Any]]:
    leaves: list[tuple[str, Any]] = []
    for key, value in payload.items():
        path = f"{parent}.{key}" if parent else key
        if isinstance(value, Mapping):
            leaves.extend(_leaf_values(value, path))
        else:
            leaves.append((path, value))
    return leaves


def load_paper_profile(path: str | Path | None = None) -> PaperProfile:
    """Load and validate the bundled profile or a caller-supplied JSON file."""

    profile_path = (
        Path(path)
        if path is not None
        else Path(str(files("textskill_optimizer.paper").joinpath(
            "profiles", "paper-faithful-v1.json"
        )))
    )
    payload = json.loads(profile_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise PaperProfileViolation("paper profile must be a JSON object")
    return PaperProfile.from_mapping(payload)
