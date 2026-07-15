"""Role-bound, content-addressed process registrations for paper data owners."""

from __future__ import annotations

import hashlib
import json
import math
import re
import subprocess
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from .errors import DataFirewallViolation
from .registry import ConsumedSplitRegistry


class ControllerRole(str, Enum):
    TRAIN = "train"
    SELECTION = "selection"
    TEST = "test"


@dataclass(frozen=True)
class ControllerArtifact:
    """A controller dependency whose bytes are fixed before any data access."""

    artifact_id: str
    path: str
    sha256: str

    def __post_init__(self) -> None:
        if type(self.artifact_id) is not str or not self.artifact_id.strip():
            raise DataFirewallViolation("controller artifact requires artifact_id")
        if type(self.path) is not str or not self.path:
            raise DataFirewallViolation("controller artifact requires path")
        _require_hash("controller artifact sha256", self.sha256, 64)

    def verify(self) -> None:
        try:
            actual = hashlib.sha256(Path(self.path).read_bytes()).hexdigest()
        except OSError as error:
            raise DataFirewallViolation("controller artifact cannot be read") from error
        if actual != self.sha256:
            raise DataFirewallViolation(
                f"controller artifact {self.artifact_id!r} hash does not match"
            )


@dataclass(frozen=True)
class ControllerRegistration:
    """One immutable split owner registered for exactly one data role."""

    controller_id: str
    role: ControllerRole
    split_id: str
    argv: tuple[str, ...]
    launch_artifact_ids: tuple[str, ...]
    response_public_key: str
    artifacts: tuple[ControllerArtifact, ...]
    timeout_seconds: float = 120.0
    max_output_chars: int = 1_000_000

    def __post_init__(self) -> None:
        if type(self.controller_id) is not str or not self.controller_id.strip():
            raise DataFirewallViolation("controller registration requires controller_id")
        if type(self.role) is not ControllerRole:
            raise DataFirewallViolation("controller registration requires exact role")
        if type(self.split_id) is not str or not self.split_id.strip():
            raise DataFirewallViolation("controller registration requires split_id")
        if type(self.argv) is not tuple or not self.argv:
            raise DataFirewallViolation("controller argv must be a non-empty tuple")
        if any(type(item) is not str or not item for item in self.argv):
            raise DataFirewallViolation("controller argv must contain non-empty strings")
        _require_hash("controller response public key", self.response_public_key, 64)
        if type(self.artifacts) is not tuple or not self.artifacts:
            raise DataFirewallViolation("controller registration requires artifacts")
        if any(type(item) is not ControllerArtifact for item in self.artifacts):
            raise DataFirewallViolation("controller artifacts require exact value types")
        artifact_ids = [item.artifact_id for item in self.artifacts]
        if len(artifact_ids) != len(set(artifact_ids)):
            raise DataFirewallViolation("controller artifact identifiers must be unique")
        if (
            type(self.launch_artifact_ids) is not tuple
            or not self.launch_artifact_ids
            or any(
                type(item) is not str or not item for item in self.launch_artifact_ids
            )
        ):
            raise DataFirewallViolation(
                "controller launch_artifact_ids must be a non-empty tuple"
            )
        if len(self.launch_artifact_ids) != len(set(self.launch_artifact_ids)):
            raise DataFirewallViolation("controller launch artifacts must be unique")
        if "runner" not in self.launch_artifact_ids:
            raise DataFirewallViolation(
                "controller launch artifacts must include the registered runner"
            )
        launch_paths = tuple(
            self.artifact(artifact_id).path
            for artifact_id in self.launch_artifact_ids
        )
        if self.argv[: len(launch_paths)] != launch_paths:
            raise DataFirewallViolation(
                "controller argv prefix must exactly match launch artifacts"
            )
        self.artifact("runner")
        self.artifact("split_manifest")
        if (
            isinstance(self.timeout_seconds, bool)
            or not isinstance(self.timeout_seconds, (int, float))
            or not math.isfinite(self.timeout_seconds)
            or self.timeout_seconds <= 0
        ):
            raise DataFirewallViolation("controller timeout must be positive and finite")
        if type(self.max_output_chars) is not int or self.max_output_chars <= 0:
            raise DataFirewallViolation("controller output limit must be positive")

    def artifact(self, artifact_id: str) -> ControllerArtifact:
        for artifact in self.artifacts:
            if artifact.artifact_id == artifact_id:
                return artifact
        raise DataFirewallViolation(
            f"controller registration is missing {artifact_id!r} artifact"
        )

    def verify_artifacts(self) -> None:
        self.__post_init__()
        for artifact in self.artifacts:
            artifact.verify()
        try:
            manifest = json.loads(
                Path(self.artifact("split_manifest").path).read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError) as error:
            raise DataFirewallViolation("split manifest must be a JSON object") from error
        if type(manifest) is not dict or manifest.get("split_id") != self.split_id:
            raise DataFirewallViolation(
                "split manifest identity does not match controller registration"
            )

    def to_manifest(self) -> dict[str, Any]:
        return {
            "controller_id": self.controller_id,
            "role": self.role.value,
            "split_id": self.split_id,
            "argv": list(self.argv),
            "launch_artifact_ids": list(self.launch_artifact_ids),
            "response_public_key": self.response_public_key,
            "artifacts": [
                {
                    "artifact_id": item.artifact_id,
                    "path": item.path,
                    "sha256": item.sha256,
                }
                for item in self.artifacts
            ],
            "timeout_seconds": float(self.timeout_seconds),
            "max_output_chars": self.max_output_chars,
        }


@dataclass(frozen=True)
class ControllerRegistry:
    """Frozen role registry; one runner/key can never own multiple roles."""

    registrations: tuple[ControllerRegistration, ...]

    def __post_init__(self) -> None:
        if type(self.registrations) is not tuple or not self.registrations:
            raise DataFirewallViolation("controller registry cannot be empty")
        if any(type(item) is not ControllerRegistration for item in self.registrations):
            raise DataFirewallViolation("controller registry requires exact registrations")
        controller_ids = [item.controller_id for item in self.registrations]
        if len(controller_ids) != len(set(controller_ids)):
            raise DataFirewallViolation("controller identifiers must be unique")
        split_ids = [item.split_id for item in self.registrations]
        if len(split_ids) != len(set(split_ids)):
            raise DataFirewallViolation("a split_id can have only one registered owner")
        manifest_hashes = [
            item.artifact("split_manifest").sha256 for item in self.registrations
        ]
        if len(manifest_hashes) != len(set(manifest_hashes)):
            raise DataFirewallViolation(
                "a split manifest can have only one registered owner"
            )
        consumed = ConsumedSplitRegistry.load()
        for item in self.registrations:
            if consumed.find(item.split_id) is not None:
                raise DataFirewallViolation(
                    f"consumed split cannot be registered: {item.split_id!r}"
                )
        owners: dict[tuple[str, str], ControllerRole] = {}
        for item in self.registrations:
            for identity in (
                ("runner", item.artifact("runner").sha256),
                ("response_key", item.response_public_key),
            ):
                previous = owners.get(identity)
                if previous is not None and previous is not item.role:
                    raise DataFirewallViolation(
                        "a controller runner or response key cannot cross data roles"
                    )
                owners[identity] = item.role

    def require(
        self,
        controller_id: str,
        *,
        role: ControllerRole,
    ) -> ControllerRegistration:
        self.__post_init__()
        for item in self.registrations:
            if item.controller_id == controller_id:
                if item.role is not role:
                    raise DataFirewallViolation(
                        f"controller {controller_id!r} is not registered for {role.value}"
                    )
                item.verify_artifacts()
                return item
        raise DataFirewallViolation(f"unknown controller registration: {controller_id!r}")

    @property
    def sha256(self) -> str:
        self.__post_init__()
        return canonical_json_sha256(
            {
                "schema_version": "paper-controller-registry-v1",
                "registrations": [
                    item.to_manifest()
                    for item in sorted(
                        self.registrations, key=lambda value: value.controller_id
                    )
                ],
            }
        )


def canonical_json(value: Any) -> str:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as error:
        raise DataFirewallViolation("controller value is not strict JSON") from error


def canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def parse_signed_response(
    *,
    registration: ControllerRegistration,
    request: Mapping[str, Any],
    stdout: str,
) -> tuple[dict[str, Any], str]:
    if len(stdout) > registration.max_output_chars:
        raise DataFirewallViolation("controller response exceeds output limit")
    try:
        envelope = json.loads(
            stdout,
            parse_constant=lambda value: _raise_non_finite(value),
        )
    except (json.JSONDecodeError, DataFirewallViolation) as error:
        raise DataFirewallViolation("controller response must be strict JSON") from error
    require_exact_keys(
        envelope,
        {"controller_id", "request_sha256", "payload", "signature"},
        context="controller response envelope",
    )
    request_sha256 = canonical_json_sha256(request)
    if envelope["controller_id"] != registration.controller_id:
        raise DataFirewallViolation("controller response identity does not match")
    if envelope["request_sha256"] != request_sha256:
        raise DataFirewallViolation("controller response request hash does not match")
    if type(envelope["payload"]) is not dict:
        raise DataFirewallViolation("controller response payload must be an object")
    if type(envelope["signature"]) is not str or re.fullmatch(
        r"[0-9a-f]{128}", envelope["signature"]
    ) is None:
        raise DataFirewallViolation("controller response signature is malformed")
    signed = {
        "controller_id": envelope["controller_id"],
        "request_sha256": envelope["request_sha256"],
        "payload": envelope["payload"],
    }
    try:
        Ed25519PublicKey.from_public_bytes(
            bytes.fromhex(registration.response_public_key)
        ).verify(bytes.fromhex(envelope["signature"]), canonical_json(signed).encode())
    except (InvalidSignature, ValueError) as error:
        raise DataFirewallViolation("controller response signature is invalid") from error
    return envelope["payload"], envelope["signature"]


def invoke_optimization_controller(
    *,
    registry: ControllerRegistry,
    controller_id: str,
    role: ControllerRole,
    request: Mapping[str, Any],
) -> tuple[dict[str, Any], str]:
    """Invoke only train/selection owners; final-test is structurally rejected."""

    if role not in (ControllerRole.TRAIN, ControllerRole.SELECTION):
        raise DataFirewallViolation(
            "optimization process invocation cannot target final-test controllers"
        )
    registration = registry.require(controller_id, role=role)
    request_text = canonical_json(request)
    try:
        completed = subprocess.run(
            registration.argv,
            text=True,
            input=request_text,
            capture_output=True,
            timeout=registration.timeout_seconds,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise DataFirewallViolation("controller process could not complete") from error
    if completed.returncode != 0:
        raise DataFirewallViolation(
            f"controller process failed with exit code {completed.returncode}"
        )
    return parse_signed_response(
        registration=registration,
        request=request,
        stdout=completed.stdout,
    )


def require_exact_keys(
    payload: Mapping[str, Any],
    expected: set[str],
    *,
    context: str,
) -> None:
    if type(payload) is not dict or set(payload) != expected:
        raise DataFirewallViolation(
            f"{context} must contain exactly: {', '.join(sorted(expected))}"
        )


def require_finite_scalar(value: object, *, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DataFirewallViolation(f"{context} must be a numeric scalar")
    if not math.isfinite(value):
        raise DataFirewallViolation(f"{context} must be finite")
    return float(value)


def _require_hash(name: str, value: str, length: int) -> None:
    if type(value) is not str or re.fullmatch(rf"[0-9a-f]{{{length}}}", value) is None:
        raise DataFirewallViolation(f"{name} must be {length} lowercase hex characters")


def _raise_non_finite(value: str) -> None:
    raise DataFirewallViolation(f"controller response contains non-finite value: {value}")
