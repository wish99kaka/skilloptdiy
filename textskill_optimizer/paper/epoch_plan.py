"""Deterministic data cursors and edit budgets for the paper epoch loop."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

from .config import PaperProfile
from .provenance import canonical_json_sha256


@dataclass(frozen=True)
class EpochCursor:
    epoch: int
    step: int
    global_step: int
    batch_id: str
    batch_seed: int
    batch_size: int
    edit_budget: int

    def __post_init__(self) -> None:
        if any(
            type(value) is not int or value < 1
            for value in (
                self.epoch,
                self.step,
                self.global_step,
                self.batch_size,
                self.edit_budget,
            )
        ):
            raise ValueError("epoch cursor numeric fields must be positive integers")
        if type(self.batch_seed) is not int or self.batch_seed < 0:
            raise ValueError("epoch cursor batch_seed must be non-negative")
        if type(self.batch_id) is not str or not self.batch_id.strip():
            raise ValueError("epoch cursor requires batch_id")


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
    learning_rate_schedule: str
    batch_ids: tuple[tuple[str, ...], ...]
    schema_version: str = "paper-epoch-plan-v1"

    def __post_init__(self) -> None:
        if self.schema_version != "paper-epoch-plan-v1":
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
        if self.learning_rate_schedule != "cosine":
            raise ValueError("M4 paper epoch plan requires frozen cosine schedule")
        if (
            type(self.batch_ids) is not tuple
            or len(self.batch_ids) != self.epochs
            or any(type(epoch) is not tuple for epoch in self.batch_ids)
            or any(len(epoch) != self.steps_per_epoch for epoch in self.batch_ids)
            or any(
                type(batch_id) is not str or not batch_id.strip()
                for epoch in self.batch_ids
                for batch_id in epoch
            )
        ):
            raise ValueError("paper epoch plan batch grid does not match its shape")
        expected = _build_batch_ids(
            profile_sha256=self.profile_sha256,
            train_split_id=self.train_split_id,
            train_split_manifest_sha256=self.train_split_manifest_sha256,
            split_seed=self.split_seed,
            epochs=self.epochs,
            steps_per_epoch=self.steps_per_epoch,
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
    ) -> "PaperEpochPlan":
        if type(profile) is not PaperProfile:
            raise ValueError("paper epoch plan requires exact PaperProfile")
        validated = PaperProfile.from_mapping(profile.to_dict())
        profile_sha256 = canonical_json_sha256(validated.to_dict())
        batch_ids = _build_batch_ids(
            profile_sha256=profile_sha256,
            train_split_id=train_split_id,
            train_split_manifest_sha256=train_split_manifest_sha256,
            split_seed=validated.split_seed,
            epochs=validated.epochs,
            steps_per_epoch=steps_per_epoch,
        )
        return cls(
            profile_sha256=profile_sha256,
            train_split_id=train_split_id,
            train_split_manifest_sha256=train_split_manifest_sha256,
            split_seed=validated.split_seed,
            epochs=validated.epochs,
            steps_per_epoch=steps_per_epoch,
            rollout_batch_size=validated.rollout_batch_size,
            learning_rate=validated.learning_rate,
            learning_rate_floor=validated.learning_rate_floor,
            learning_rate_schedule=validated.learning_rate_schedule,
            batch_ids=batch_ids,
        )

    def cursor(self, *, epoch: int, step: int) -> EpochCursor:
        if (
            type(epoch) is not int
            or type(step) is not int
            or not 1 <= epoch <= self.epochs
            or not 1 <= step <= self.steps_per_epoch
        ):
            raise ValueError("cursor is outside epoch plan")
        global_step = (epoch - 1) * self.steps_per_epoch + step
        return EpochCursor(
            epoch=epoch,
            step=step,
            global_step=global_step,
            batch_id=self.batch_ids[epoch - 1][step - 1],
            batch_seed=_batch_seed(
                split_seed=self.split_seed,
                epoch=epoch,
                step=step,
                batch_id=self.batch_ids[epoch - 1][step - 1],
            ),
            batch_size=self.rollout_batch_size,
            edit_budget=self._edit_budget(global_step),
        )

    def require_profile(self, profile: PaperProfile) -> None:
        """Bind every copied scheduler/data field to the hashed frozen profile."""

        if type(profile) is not PaperProfile:
            raise ValueError("paper epoch plan requires exact PaperProfile")
        validated = PaperProfile.from_mapping(profile.to_dict())
        expected = {
            "profile_sha256": canonical_json_sha256(validated.to_dict()),
            "split_seed": validated.split_seed,
            "epochs": validated.epochs,
            "rollout_batch_size": validated.rollout_batch_size,
            "learning_rate": validated.learning_rate,
            "learning_rate_floor": validated.learning_rate_floor,
            "learning_rate_schedule": validated.learning_rate_schedule,
        }
        actual = {name: getattr(self, name) for name in expected}
        if actual != expected:
            raise ValueError("paper epoch plan fields do not match frozen profile")

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
            "learning_rate_schedule": self.learning_rate_schedule,
            "batch_ids": [list(epoch) for epoch in self.batch_ids],
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
            "learning_rate_schedule",
            "batch_ids",
        }
        if type(payload) is not dict or set(payload) != expected:
            raise ValueError("paper epoch plan must contain exactly its schema fields")
        raw_batch_ids = payload["batch_ids"]
        if type(raw_batch_ids) is not list or any(
            type(epoch) is not list for epoch in raw_batch_ids
        ):
            raise ValueError("paper epoch plan batch_ids must be a nested list")
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
            learning_rate_schedule=payload["learning_rate_schedule"],
            batch_ids=tuple(tuple(epoch) for epoch in raw_batch_ids),
        )

    def _edit_budget(self, global_step: int) -> int:
        total_steps = self.epochs * self.steps_per_epoch
        if total_steps <= 1:
            return self.learning_rate
        t = min(global_step, total_steps) / total_steps
        value = self.learning_rate_floor + 0.5 * (
            self.learning_rate - self.learning_rate_floor
        ) * (1 + math.cos(math.pi * t))
        return max(self.learning_rate_floor, round(value))


def _build_batch_ids(
    *,
    profile_sha256: str,
    train_split_id: str,
    train_split_manifest_sha256: str,
    split_seed: int,
    epochs: int,
    steps_per_epoch: int,
) -> tuple[tuple[str, ...], ...]:
    return tuple(
        tuple(
            "train-batch-"
            + canonical_json_sha256(
                {
                    "profile_sha256": profile_sha256,
                    "train_split_id": train_split_id,
                    "train_split_manifest_sha256": (
                        train_split_manifest_sha256
                    ),
                    "split_seed": split_seed,
                    "epoch": epoch,
                    "step": step,
                }
            )[:20]
            for step in range(1, steps_per_epoch + 1)
        )
        for epoch in range(1, epochs + 1)
    )


def _batch_seed(*, split_seed: int, epoch: int, step: int, batch_id: str) -> int:
    return int(
        canonical_json_sha256(
            {
                "split_seed": split_seed,
                "epoch": epoch,
                "step": step,
                "batch_id": batch_id,
            }
        )[:16],
        16,
    )


def _require_sha256(value: object, name: str) -> None:
    if type(value) is not str or len(value) != 64:
        raise ValueError(f"paper epoch plan requires {name}")
    try:
        int(value, 16)
    except ValueError as error:
        raise ValueError(f"paper epoch plan requires hexadecimal {name}") from error
