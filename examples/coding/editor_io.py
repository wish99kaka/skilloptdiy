"""Shared IO helpers for command editor scripts."""

from __future__ import annotations

import json
import sys
from typing import Any


def load_optimizer_payload_from_stdin() -> dict[str, Any]:
    try:
        payload = json.load(sys.stdin)
    except json.JSONDecodeError as exc:
        raise ValueError(
            "Expected optimizer JSON payload on stdin. "
            "Use this script via --editor-command or pipe a payload into it."
        ) from exc
    if not isinstance(payload, dict):
        raise ValueError("Optimizer payload must be a JSON object")
    return payload

