"""Canonical identities for paper contract artifacts."""

from __future__ import annotations

import hashlib
import json
from typing import Any


def canonical_json_sha256(payload: Any) -> str:
    """Hash the normalized effective value, independent of JSON formatting."""

    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
