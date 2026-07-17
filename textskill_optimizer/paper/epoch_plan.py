"""Deterministic mechanism, data, and learning-rate plan for paper runs."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Mapping

from .config import PaperProfile
from .provenance import canonical_json_sha256


_DEFAULT_SCOPE = "paper_faithful_default"
_MECHANISM_TEST_SCOPE = "mechanism_test"
_SCHEDULES = frozenset({"constant", "linear", "cosine", "autonomous"})
_UPDATE_MODES = frozenset({"patch", "rewrite_from_suggestions"})


@dataclass(frozen=True)
class PaperMechanismSpec:
    """Explicit runtime mechanism recipe; variants are mechanism-test only."""

    claim_scope: str
    accumulation: int
    analyst_workers: int
    learning_rate_schedule: str
    update_mode: str

    def __post_init__(self) -> None:
        if self.claim_scope not in {_DEFAULT_SCOPE, _MECHANISM_TEST_SCOPE}:
            raise ValueError("unsupported paper mechanism claim scope")
        if type(self.accumulation) is not int or self.accumulation < 1:
            raise ValueError("paper accumulation must be a positive integer")
        if type(self.analyst_workers) is not int or self.analyst_workers < 1:
            raise ValueError("paper analyst_workers must be a positive integer")
        if self.learning_rate_schedule not in _SCHEDULES:
            raise ValueError("unsupported paper learning-rate schedule")
        if self.update_mode not in _UPDATE_MODES:
            raise ValueError("unsupported paper update mode")
        if self.claim_scope == _DEFAULT_SCOPE and (
            self.accumulation != 1
            or self.analyst_workers != 16
            or self.learning_rate_schedule != "cosine"
            or self.update_mode != "patch"
        ):
            raise ValueError(
                "default mechanism scope cannot contain mechanism overrides"
            )

    @classmethod
    def from_profile(cls, profile: PaperProfile) -> "PaperMechanismSpec":
        validated = _validated_profile(profile)
        return cls(
            claim_scope=_DEFAULT_SCOPE,
            accumulation=validated.accumulation,
            analyst_workers=validated.analyst_workers,
            learning_rate_schedule=validated.learning_rate_schedule,
            update_mode=validated.update_mode,
        )

    @classmethod
    def for_mechanism_test(
        cls,
        profile: PaperProfile,
        *,
        accumulation: int | None = None,
        analyst_workers: int | None = None,
        learning_rate_schedule: str | None = None,
        update_mode: str | None = None,
    ) -> "PaperMechanismSpec":
        validated = _validated_profile(profile)
        spec = cls(
            claim_scope=_MECHANISM_TEST_SCOPE,
            accumulation=(
                validated.accumulation
                if accumulation is None
                else accumulation
            ),
            analyst_workers=(
                validated.analyst_workers
                if analyst_workers is None
                else analyst_workers
            ),
            learning_rate_schedule=(
                validated.learning_rate_schedule
                if learning_rate_schedule is None
                else learning_rate_schedule
            ),
            update_mode=(
                validated.update_mode if update_mode is None else update_mode
            ),
        )
        if (
            spec.accumulation == validated.accumulation
            and spec.analyst_workers == validated.analyst_workers
            and spec.learning_rate_schedule
            == validated.learning_rate_schedule
            and spec.update_mode == validated.update_mode
        ):
            raise ValueError("mechanism-test plan requires an explicit deviation")
        return spec

    @property
    def paper_claim_eligible(self) -> bool:
        return self.claim_scope == _DEFAULT_SCOPE

    def require_profile(self, profile: PaperProfile) -> None:
        validated = _validated_profile(profile)
        if self.claim_scope == _DEFAULT_SCOPE and self != self.from_profile(
            validated
        ):
            raise ValueError("default mechanisms do not match frozen profile")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "PaperMechanismSpec":
        expected = {
            "claim_scope",
            "accumulation",
            "analyst_workers",
            "learning_rate_schedule",
            "update_mode",
        }
        if type(payload) is not dict or set(payload) != expected:
            raise ValueError("paper mechanism spec has invalid fields")
        return cls(**payload)


@dataclass(frozen=True)
class EpochBatchCursor:
    accumulation_index: int
    batch_id: str
    batch_seed: int
    batch_size: int

    def __post_init__(self) -> None:
        if type(self.accumulation_index) is not int or self.accumulation_index < 1:
            raise ValueError("accumulation index must be positive")
        if type(self.batch_id) is not str or not self.batch_id.strip():
            raise ValueError("epoch batch cursor requires batch_id")
        if type(self.batch_seed) is not int or self.batch_seed < 0:
            raise ValueError("epoch batch cursor batch_seed must be non-negative")
        if type(self.batch_size) is not int or self.batch_size < 1:
            raise ValueError("epoch batch cursor batch_size must be positive")


@dataclass(frozen=True)
class EpochCursor:
    epoch: int
    step: int
    global_step: int
    batches: tuple[EpochBatchCursor, ...]
    analysis_budget: int
    edit_budget: int | None

    def __post_init__(self) -> None:
        if any(
            type(value) is not int or value < 1
            for value in (
                self.epoch,
                self.step,
                self.global_step,
                self.analysis_budget,
            )
        ):
            raise ValueError("epoch cursor numeric fields must be positive integers")
        if self.edit_budget is not None and (
            type(self.edit_budget) is not int or self.edit_budget < 0
        ):
            raise ValueError("epoch cursor edit_budget must be non-negative")
        if type(self.batches) is not tuple or not self.batches or any(
            type(batch) is not EpochBatchCursor for batch in self.batches
        ):
            raise ValueError("epoch cursor requires exact accumulation batches")
        if [batch.accumulation_index for batch in self.batches] != list(
            range(1, len(self.batches) + 1)
        ):
            raise ValueError("epoch cursor accumulation order is not contiguous")

    @property
    def batch_id(self) -> str:
        return self.batches[0].batch_id

    @property
    def batch_seed(self) -> int:
        return self.batches[0].batch_seed

    @property
    def batch_size(self) -> int:
        return self.batches[0].batch_size


@dataclass(frozen=True)
class PaperEpochPlan:
    """Frozen call plan bound to one profile and registered train split."""

    profile_sha256: str
    train_split_id: str
    train_split_manifest_sha256: str
    split_seed: int
    epochs: int
    steps_per_epoch: int
    rollout_batch_size: int
    learning_rate: int
    learning_rate_floor: int
    mechanisms: PaperMechanismSpec
    batch_ids: tuple[tuple[tuple[str, ...], ...], ...]
    schema_version: str = "paper-epoch-plan-v2"

    def __post_init__(self) -> None:
        if self.schema_version != "paper-epoch-plan-v2":
            raise ValueError("unsupported paper epoch plan schema")
        _require_sha256(self.profile_sha256, "profile_sha256")
        _require_sha256(
            self.train_split_manifest_sha256,
            "train_split_manifest_sha256",
        )
        if type(self.train_split_id) is not str or not self.train_split_id.strip():
            raise ValueError("paper epoch plan requires train_split_id")
        if type(self.split_seed) is not int:
            raise ValueError("paper epoch plan requires integer split_seed")
        if type(self.epochs) is not int or self.epochs < 1:
            raise ValueError("paper epoch plan requires positive epochs")
        if type(self.steps_per_epoch) is not int or self.steps_per_epoch < 1:
            raise ValueError("paper epoch plan requires positive steps_per_epoch")
        if type(self.rollout_batch_size) is not int or self.rollout_batch_size < 1:
            raise ValueError("paper epoch plan requires positive rollout_batch_size")
        if (
            type(self.learning_rate) is not int
            or type(self.learning_rate_floor) is not int
            or self.learning_rate_floor < 1
            or self.learning_rate < self.learning_rate_floor
        ):
            raise ValueError("paper epoch plan has invalid learning-rate bounds")
        if type(self.mechanisms) is not PaperMechanismSpec:
            raise ValueError("paper epoch plan requires exact mechanism spec")
        self.mechanisms.__post_init__()
        if not _valid_batch_grid(
            self.batch_ids,
            epochs=self.epochs,
            steps_per_epoch=self.steps_per_epoch,
            accumulation=self.mechanisms.accumulation,
        ):
            raise ValueError("paper epoch plan batch grid does not match its shape")
        expected = _build_batch_ids(
            profile_sha256=self.profile_sha256,
            mechanism_sha256=canonical_json_sha256(self.mechanisms.to_dict()),
            train_split_id=self.train_split_id,
            train_split_manifest_sha256=self.train_split_manifest_sha256,
            split_seed=self.split_seed,
            epochs=self.epochs,
            steps_per_epoch=self.steps_per_epoch,
            accumulation=self.mechanisms.accumulation,
        )
        if self.batch_ids != expected:
            raise ValueError("paper epoch plan batch IDs are not deterministic")

    @classmethod
    def build(
        cls,
        *,
        profile: PaperProfile,
        train_split_id: str,
        train_split_manifest_sha256: str,
        steps_per_epoch: int,
        mechanisms: PaperMechanismSpec | None = None,
        epochs_override: int | None = None,
    ) -> "PaperEpochPlan":
        validated = _validated_profile(profile)
        mechanism_spec = (
            PaperMechanismSpec.from_profile(validated)
            if mechanisms is None
            else mechanisms
        )
        if type(mechanism_spec) is not PaperMechanismSpec:
            raise ValueError("paper epoch plan requires exact mechanism spec")
        mechanism_spec.require_profile(validated)
        if epochs_override is None:
            execution_epochs = validated.epochs
        else:
            if mechanism_spec.paper_claim_eligible:
                raise ValueError(
                    "epoch overrides are allowed only for a mechanism-test plan"
                )
            if (
                type(epochs_override) is not int
                or not validated.slow_update.start_epoch
                <= epochs_override
                <= validated.epochs
            ):
                raise ValueError(
                    "mechanism-test epochs must expose slow/meta and not exceed the profile"
                )
            execution_epochs = epochs_override
        profile_sha256 = canonical_json_sha256(validated.to_dict())
        batch_ids = _build_batch_ids(
            profile_sha256=profile_sha256,
            mechanism_sha256=canonical_json_sha256(mechanism_spec.to_dict()),
            train_split_id=train_split_id,
            train_split_manifest_sha256=train_split_manifest_sha256,
            split_seed=validated.split_seed,
            epochs=execution_epochs,
            steps_per_epoch=steps_per_epoch,
            accumulation=mechanism_spec.accumulation,
        )
        return cls(
            profile_sha256=profile_sha256,
            train_split_id=train_split_id,
            train_split_manifest_sha256=train_split_manifest_sha256,
            split_seed=validated.split_seed,
            epochs=execution_epochs,
            steps_per_epoch=steps_per_epoch,
            rollout_batch_size=validated.rollout_batch_size,
            learning_rate=validated.learning_rate,
            learning_rate_floor=validated.learning_rate_floor,
            mechanisms=mechanism_spec,
            batch_ids=batch_ids,
        )

    @property
    def paper_claim_eligible(self) -> bool:
        return self.mechanisms.paper_claim_eligible

    @property
    def learning_rate_schedule(self) -> str:
        return self.mechanisms.learning_rate_schedule

    def cursor(self, *, epoch: int, step: int) -> EpochCursor:
        if (
            type(epoch) is not int
            or type(step) is not int
            or not 1 <= epoch <= self.epochs
            or not 1 <= step <= self.steps_per_epoch
        ):
            raise ValueError("cursor is outside epoch plan")
        global_step = (epoch - 1) * self.steps_per_epoch + step
        batch_ids = self.batch_ids[epoch - 1][step - 1]
        batches = tuple(
            EpochBatchCursor(
                accumulation_index=index,
                batch_id=batch_id,
                batch_seed=_batch_seed(
                    split_seed=self.split_seed,
                    epoch=epoch,
                    step=step,
                    accumulation_index=index,
                    batch_id=batch_id,
                ),
                batch_size=self.rollout_batch_size,
            )
            for index, batch_id in enumerate(batch_ids, 1)
        )
        return EpochCursor(
            epoch=epoch,
            step=step,
            global_step=global_step,
            batches=batches,
            analysis_budget=self.learning_rate,
            edit_budget=self._edit_budget(global_step),
        )

    def require_profile(self, profile: PaperProfile) -> None:
        """Bind copied fields to the frozen profile and explicit mechanisms."""

        validated = _validated_profile(profile)
        expected = {
            "profile_sha256": canonical_json_sha256(validated.to_dict()),
            "split_seed": validated.split_seed,
            "rollout_batch_size": validated.rollout_batch_size,
            "learning_rate": validated.learning_rate,
            "learning_rate_floor": validated.learning_rate_floor,
        }
        actual = {name: getattr(self, name) for name in expected}
        if actual != expected:
            raise ValueError("paper epoch plan fields do not match frozen profile")
        self.mechanisms.require_profile(validated)
        if self.mechanisms.paper_claim_eligible:
            if self.epochs != validated.epochs:
                raise ValueError("default paper epoch plan must use all profile epochs")
        elif not (
            validated.slow_update.start_epoch
            <= self.epochs
            <= validated.epochs
        ):
            raise ValueError(
                "mechanism-test epochs must expose slow/meta and not exceed the profile"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "profile_sha256": self.profile_sha256,
            "train_split_id": self.train_split_id,
            "train_split_manifest_sha256": self.train_split_manifest_sha256,
            "split_seed": self.split_seed,
            "epochs": self.epochs,
            "steps_per_epoch": self.steps_per_epoch,
            "rollout_batch_size": self.rollout_batch_size,
            "learning_rate": self.learning_rate,
            "learning_rate_floor": self.learning_rate_floor,
            "mechanisms": self.mechanisms.to_dict(),
            "batch_ids": [
                [list(step) for step in epoch] for epoch in self.batch_ids
            ],
        }

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "PaperEpochPlan":
        expected = {
            "schema_version",
            "profile_sha256",
            "train_split_id",
            "train_split_manifest_sha256",
            "split_seed",
            "epochs",
            "steps_per_epoch",
            "rollout_batch_size",
            "learning_rate",
            "learning_rate_floor",
            "mechanisms",
            "batch_ids",
        }
        if type(payload) is not dict or set(payload) != expected:
            raise ValueError("paper epoch plan must contain exactly its schema fields")
        raw_batch_ids = payload["batch_ids"]
        if type(raw_batch_ids) is not list or any(
            type(epoch) is not list
            or any(type(step) is not list for step in epoch)
            for epoch in raw_batch_ids
        ):
            raise ValueError("paper epoch plan batch_ids must be a three-level list")
        if type(payload["mechanisms"]) is not dict:
            raise ValueError("paper epoch plan mechanisms must be an object")
        return cls(
            schema_version=payload["schema_version"],
            profile_sha256=payload["profile_sha256"],
            train_split_id=payload["train_split_id"],
            train_split_manifest_sha256=payload[
                "train_split_manifest_sha256"
            ],
            split_seed=payload["split_seed"],
            epochs=payload["epochs"],
            steps_per_epoch=payload["steps_per_epoch"],
            rollout_batch_size=payload["rollout_batch_size"],
            learning_rate=payload["learning_rate"],
            learning_rate_floor=payload["learning_rate_floor"],
            mechanisms=PaperMechanismSpec.from_mapping(payload["mechanisms"]),
            batch_ids=tuple(
                tuple(tuple(step) for step in epoch) for epoch in raw_batch_ids
            ),
        )

    def _edit_budget(self, global_step: int) -> int | None:
        mode = self.mechanisms.learning_rate_schedule
        if mode == "autonomous":
            return None
        if mode == "constant":
            return self.learning_rate
        total_steps = self.epochs * self.steps_per_epoch
        if total_steps <= 1:
            return self.learning_rate
        t = min(global_step, total_steps) / total_steps
        if mode == "linear":
            value = self.learning_rate + (
                self.learning_rate_floor - self.learning_rate
            ) * t
        else:
            value = self.learning_rate_floor + 0.5 * (
                self.learning_rate - self.learning_rate_floor
            ) * (1 + math.cos(math.pi * t))
        return max(self.learning_rate_floor, round(value))


def _build_batch_ids(
    *,
    profile_sha256: str,
    mechanism_sha256: str,
    train_split_id: str,
    train_split_manifest_sha256: str,
    split_seed: int,
    epochs: int,
    steps_per_epoch: int,
    accumulation: int,
) -> tuple[tuple[tuple[str, ...], ...], ...]:
    return tuple(
        tuple(
            tuple(
                "train-batch-"
                + canonical_json_sha256(
                    {
                        "profile_sha256": profile_sha256,
                        "mechanism_sha256": mechanism_sha256,
                        "train_split_id": train_split_id,
                        "train_split_manifest_sha256": (
                            train_split_manifest_sha256
                        ),
                        "split_seed": split_seed,
                        "epoch": epoch,
                        "step": step,
                        "accumulation_index": accumulation_index,
                    }
                )[:20]
                for accumulation_index in range(1, accumulation + 1)
            )
            for step in range(1, steps_per_epoch + 1)
        )
        for epoch in range(1, epochs + 1)
    )


def _batch_seed(
    *,
    split_seed: int,
    epoch: int,
    step: int,
    accumulation_index: int,
    batch_id: str,
) -> int:
    return int(
        canonical_json_sha256(
            {
                "split_seed": split_seed,
                "epoch": epoch,
                "step": step,
                "accumulation_index": accumulation_index,
                "batch_id": batch_id,
            }
        )[:16],
        16,
    )


def _valid_batch_grid(
    value: object,
    *,
    epochs: int,
    steps_per_epoch: int,
    accumulation: int,
) -> bool:
    return bool(
        type(value) is tuple
        and len(value) == epochs
        and all(
            type(epoch) is tuple
            and len(epoch) == steps_per_epoch
            and all(
                type(step) is tuple
                and len(step) == accumulation
                and all(
                    type(batch_id) is str and bool(batch_id.strip())
                    for batch_id in step
                )
                for step in epoch
            )
            for epoch in value
        )
    )


def _validated_profile(profile: PaperProfile) -> PaperProfile:
    if type(profile) is not PaperProfile:
        raise ValueError("paper epoch plan requires exact PaperProfile")
    return PaperProfile.from_mapping(profile.to_dict())


def _require_sha256(value: object, name: str) -> None:
    if type(value) is not str or len(value) != 64:
        raise ValueError(f"paper epoch plan requires {name}")
    try:
        int(value, 16)
    except ValueError as error:
        raise ValueError(f"paper epoch plan requires hexadecimal {name}") from error
