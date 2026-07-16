"""Authenticated checkpoint envelope for paper epoch state."""

from __future__ import annotations

import hashlib
import hmac
import json
from dataclasses import dataclass, field
from typing import Any, Mapping

from .controller_process import canonical_json


_CHECKPOINT_SCHEMA = "paper-epoch-checkpoint-v1"


@dataclass(frozen=True)
class PaperEpochCheckpoint:
    key_id: str
    canonical_payload: str
    signature: str
    schema_version: str = _CHECKPOINT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != _CHECKPOINT_SCHEMA:
            raise ValueError("unsupported paper checkpoint schema")
        if type(self.key_id) is not str or not self.key_id.strip():
            raise ValueError("paper checkpoint requires key_id")
        if type(self.canonical_payload) is not str:
            raise ValueError("paper checkpoint requires canonical payload")
        try:
            payload = json.loads(self.canonical_payload)
        except json.JSONDecodeError as error:
            raise ValueError("paper checkpoint payload is not JSON") from error
        if type(payload) is not dict or canonical_json(payload) != self.canonical_payload:
            raise ValueError("paper checkpoint payload must be a canonical object")
        if type(self.signature) is not str or len(self.signature) != 64:
            raise ValueError("paper checkpoint requires HMAC-SHA256 signature")
        try:
            bytes.fromhex(self.signature)
        except ValueError as error:
            raise ValueError("paper checkpoint signature must be hexadecimal") from error

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "key_id": self.key_id,
            "payload": json.loads(self.canonical_payload),
            "signature": self.signature,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "PaperEpochCheckpoint":
        expected = {"schema_version", "key_id", "payload", "signature"}
        if type(value) is not dict or set(value) != expected:
            raise ValueError("paper checkpoint must contain exactly envelope fields")
        if type(value["payload"]) is not dict:
            raise ValueError("paper checkpoint payload must be an object")
        return cls(
            schema_version=value["schema_version"],
            key_id=value["key_id"],
            canonical_payload=canonical_json(value["payload"]),
            signature=value["signature"],
        )


@dataclass(frozen=True)
class CheckpointAuthenticator:
    """External HMAC capability; secret bytes are never stored in checkpoints."""

    key_id: str
    secret_key: bytes = field(repr=False)

    def __post_init__(self) -> None:
        if type(self.key_id) is not str or not self.key_id.strip():
            raise ValueError("checkpoint authenticator requires key_id")
        if type(self.secret_key) is not bytes or len(self.secret_key) < 32:
            raise ValueError("checkpoint authenticator requires at least 32 secret bytes")

    def sign(self, payload: Mapping[str, Any]) -> PaperEpochCheckpoint:
        if type(payload) is not dict:
            raise ValueError("checkpoint payload must be an exact object")
        canonical_payload = canonical_json(payload)
        return PaperEpochCheckpoint(
            key_id=self.key_id,
            canonical_payload=canonical_payload,
            signature=self._signature(canonical_payload),
        )

    def verify(self, checkpoint: PaperEpochCheckpoint) -> dict[str, Any]:
        if type(checkpoint) is not PaperEpochCheckpoint:
            raise ValueError("resume requires exact PaperEpochCheckpoint")
        checkpoint.__post_init__()
        if checkpoint.key_id != self.key_id or not hmac.compare_digest(
            checkpoint.signature,
            self._signature(checkpoint.canonical_payload),
        ):
            raise ValueError("paper checkpoint authentication failed")
        payload = json.loads(checkpoint.canonical_payload)
        assert type(payload) is dict
        return payload

    def _signature(self, canonical_payload: str) -> str:
        signed = (
            f"{_CHECKPOINT_SCHEMA}\n{self.key_id}\n{canonical_payload}"
        ).encode("utf-8")
        return hmac.new(self.secret_key, signed, hashlib.sha256).hexdigest()
