"""Content-addressed, replayable artifact lineage for paper runtimes."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping

from .backend import OptimizerRequest, OptimizerResponse


class PaperArtifactKind(str, Enum):
    PROFILE = "profile"
    EPOCH_PLAN = "epoch_plan"
    CONTROLLER_REGISTRY = "controller_registry"
    SKILL = "skill"
    SELECTION_SCORE = "selection_score"
    TRAIN_EVIDENCE = "train_evidence"
    LONGITUDINAL_EVIDENCE = "longitudinal_evidence"
    OPTIMIZER_REQUEST = "optimizer_request"
    OPTIMIZER_RESPONSE = "optimizer_response"
    UPDATE_SET = "update_set"
    APPLY_REPORT = "apply_report"
    META_SKILL = "meta_skill"
    ALGORITHM_EVENT = "algorithm_event"


@dataclass(frozen=True)
class OptimizerExchange:
    request: OptimizerRequest
    response: OptimizerResponse

    def __post_init__(self) -> None:
        if type(self.request) is not OptimizerRequest:
            raise ValueError("optimizer exchange requires exact request")
        if type(self.response) is not OptimizerResponse:
            raise ValueError("optimizer exchange requires exact response")
        if self.request.call_id != self.response.call_id:
            raise ValueError("optimizer exchange call IDs do not match")


@dataclass(frozen=True)
class PaperArtifactRecord:
    artifact_id: str
    kind: PaperArtifactKind
    content_sha256: str
    canonical_payload: str
    parent_ids: tuple[str, ...]
    schema_version: str = "paper-artifact-v1"

    def __post_init__(self) -> None:
        if self.schema_version != "paper-artifact-v1":
            raise ValueError("unsupported paper artifact schema")
        if type(self.kind) is not PaperArtifactKind:
            raise ValueError("paper artifact requires exact kind")
        if type(self.canonical_payload) is not str:
            raise ValueError("paper artifact requires canonical payload")
        try:
            payload = json.loads(self.canonical_payload)
        except json.JSONDecodeError as error:
            raise ValueError("paper artifact payload is not JSON") from error
        if _canonical_json(payload) != self.canonical_payload:
            raise ValueError("paper artifact payload is not canonical")
        expected_content_sha256 = _sha256(self.canonical_payload)
        if self.content_sha256 != expected_content_sha256:
            raise ValueError("paper artifact content hash does not match payload")
        if type(self.parent_ids) is not tuple or any(
            type(item) is not str or not item.strip() for item in self.parent_ids
        ):
            raise ValueError("paper artifact parent IDs must be exact strings")
        if len(self.parent_ids) != len(set(self.parent_ids)):
            raise ValueError("paper artifact parent IDs must be unique")
        if self.artifact_id != _artifact_id(
            self.kind,
            self.content_sha256,
            self.parent_ids,
        ):
            raise ValueError("paper artifact ID is not content-addressed")

    @property
    def payload(self) -> Mapping[str, Any]:
        return json.loads(self.canonical_payload)

    @classmethod
    def create(
        cls,
        *,
        kind: PaperArtifactKind,
        payload: Mapping[str, Any],
        parent_ids: tuple[str, ...] = (),
    ) -> "PaperArtifactRecord":
        if type(kind) is not PaperArtifactKind:
            raise ValueError("paper artifact requires exact kind")
        if type(payload) is not dict:
            raise ValueError("paper artifact payload must be an exact object")
        canonical_payload = _canonical_json(payload)
        content_sha256 = _sha256(canonical_payload)
        return cls(
            artifact_id=_artifact_id(kind, content_sha256, parent_ids),
            kind=kind,
            content_sha256=content_sha256,
            canonical_payload=canonical_payload,
            parent_ids=parent_ids,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "artifact_id": self.artifact_id,
            "kind": self.kind.value,
            "content_sha256": self.content_sha256,
            "canonical_payload": self.canonical_payload,
            "parent_ids": list(self.parent_ids),
        }

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "PaperArtifactRecord":
        expected = {
            "schema_version",
            "artifact_id",
            "kind",
            "content_sha256",
            "canonical_payload",
            "parent_ids",
        }
        if type(payload) is not dict or set(payload) != expected:
            raise ValueError("invalid paper artifact checkpoint record")
        if type(payload["parent_ids"]) is not list:
            raise ValueError("paper artifact parent_ids must be a list")
        return cls(
            schema_version=payload["schema_version"],
            artifact_id=payload["artifact_id"],
            kind=PaperArtifactKind(payload["kind"]),
            content_sha256=payload["content_sha256"],
            canonical_payload=payload["canonical_payload"],
            parent_ids=tuple(payload["parent_ids"]),
        )


@dataclass(frozen=True)
class PaperArtifactLineage:
    records: tuple[PaperArtifactRecord, ...]
    schema_version: str = "paper-artifact-lineage-v1"

    def __post_init__(self) -> None:
        if self.schema_version != "paper-artifact-lineage-v1":
            raise ValueError("unsupported paper artifact lineage schema")
        if type(self.records) is not tuple or any(
            type(record) is not PaperArtifactRecord for record in self.records
        ):
            raise ValueError("paper artifact lineage requires exact records")

    def verify(self) -> None:
        seen: set[str] = set()
        for record in self.records:
            record.__post_init__()
            if record.artifact_id in seen:
                raise ValueError("paper artifact lineage has duplicate IDs")
            missing = set(record.parent_ids) - seen
            if missing:
                raise ValueError("paper artifact lineage parent is missing or unordered")
            seen.add(record.artifact_id)

    def records_of_kind(
        self,
        kind: PaperArtifactKind,
    ) -> tuple[PaperArtifactRecord, ...]:
        if type(kind) is not PaperArtifactKind:
            raise ValueError("records_of_kind requires exact artifact kind")
        return tuple(record for record in self.records if record.kind is kind)

    @property
    def sha256(self) -> str:
        return _sha256(
            _canonical_json(
                {
                    "schema_version": self.schema_version,
                    "records": [record.to_dict() for record in self.records],
                }
            )
        )

    def to_checkpoint_list(self) -> list[dict[str, Any]]:
        self.verify()
        return [record.to_dict() for record in self.records]


class PaperArtifactLedger:
    """Append-only implementation hidden behind the immutable lineage view."""

    def __init__(self) -> None:
        self._records: list[PaperArtifactRecord] = []
        self._ids: set[str] = set()

    @property
    def lineage(self) -> PaperArtifactLineage:
        lineage = PaperArtifactLineage(tuple(self._records))
        lineage.verify()
        return lineage

    def add(
        self,
        kind: PaperArtifactKind,
        payload: Mapping[str, Any],
        *,
        parent_ids: tuple[str, ...] = (),
    ) -> PaperArtifactRecord:
        missing = set(parent_ids) - self._ids
        if missing:
            raise ValueError("paper artifact parents must already exist")
        record = PaperArtifactRecord.create(
            kind=kind,
            payload=payload,
            parent_ids=parent_ids,
        )
        if record.artifact_id in self._ids:
            existing = next(
                item
                for item in self._records
                if item.artifact_id == record.artifact_id
            )
            if existing != record:
                raise ValueError("paper artifact ID collision")
            return existing
        self._records.append(record)
        self._ids.add(record.artifact_id)
        return record

    @classmethod
    def from_checkpoint_list(
        cls,
        payload: object,
    ) -> "PaperArtifactLedger":
        if type(payload) is not list:
            raise ValueError("artifact lineage checkpoint must be a list")
        ledger = cls()
        for item in payload:
            record = PaperArtifactRecord.from_mapping(item)
            added = ledger.add(
                record.kind,
                record.payload,
                parent_ids=record.parent_ids,
            )
            if added != record:
                raise ValueError("artifact checkpoint is not canonical")
        return ledger


def optimizer_request_payload(request: OptimizerRequest) -> dict[str, Any]:
    return {
        "call_id": request.call_id,
        "stage": request.stage.value,
        "prompt": request.prompt,
        "response_schema": dict(request.response_schema),
        "system_prompt": request.system_prompt,
        "metadata": dict(request.metadata),
    }


def optimizer_response_payload(response: OptimizerResponse) -> dict[str, Any]:
    return {
        "call_id": response.call_id,
        "payload": dict(response.payload),
        "model_id": response.model_id,
        "usage": dict(response.usage),
    }


def _artifact_id(
    kind: PaperArtifactKind,
    content_sha256: str,
    parent_ids: tuple[str, ...],
) -> str:
    identity = _sha256(
        _canonical_json(
            {
                "kind": kind.value,
                "content_sha256": content_sha256,
                "parent_ids": list(parent_ids),
            }
        )
    )
    return f"{kind.value}:{identity}"


def _canonical_json(payload: Any) -> str:
    return json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()
