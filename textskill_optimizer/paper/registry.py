"""Immutable consumed-split registry used by claim eligibility checks."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path


@dataclass(frozen=True)
class ConsumedSplit:
    split_id: str
    protocol_id: str
    consumed_at: str
    attempt: int
    attempt_path: str
    attempt_sha256: str
    receipt_path: str
    receipt_sha256: str
    archive_sha256: str


@dataclass(frozen=True)
class ConsumedSplitRegistry:
    schema_version: str
    entries: tuple[ConsumedSplit, ...]

    @classmethod
    def load(cls, path: str | Path | None = None) -> "ConsumedSplitRegistry":
        registry_path = (
            Path(path)
            if path is not None
            else Path(
                str(
                    files("textskill_optimizer.paper").joinpath(
                        "consumed-splits-v1.json"
                    )
                )
            )
        )
        payload = json.loads(registry_path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != "consumed-splits-v1":
            raise ValueError("unsupported consumed-split registry schema")
        entries = tuple(ConsumedSplit(**item) for item in payload.get("entries", []))
        split_ids = [entry.split_id for entry in entries]
        if len(split_ids) != len(set(split_ids)):
            raise ValueError("consumed-split registry contains duplicate split_id values")
        for entry in entries:
            for name in ("attempt_sha256", "receipt_sha256", "archive_sha256"):
                if re.fullmatch(r"[0-9a-f]{64}", getattr(entry, name)) is None:
                    raise ValueError(f"invalid {name} for consumed split {entry.split_id!r}")
        return cls(schema_version=payload["schema_version"], entries=entries)

    def find(self, split_id: str) -> ConsumedSplit | None:
        return next((item for item in self.entries if item.split_id == split_id), None)
