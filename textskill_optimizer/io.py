"""File IO helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .models import Task


def load_text(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8")


def write_text(path: str | Path, text: str) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")


def load_tasks_jsonl(path: str | Path) -> list[Task]:
    tasks: list[Task] = []
    for index, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSONL at line {index}: {exc}") from exc
        if not isinstance(payload, dict):
            raise ValueError(f"Task line {index} must be a JSON object")
        metadata = dict(payload.get("metadata") or {})
        metadata.setdefault("_task_file", str(Path(path).resolve()))
        metadata.setdefault("_task_dir", str(Path(path).resolve().parent))
        payload["metadata"] = metadata
        tasks.append(Task.from_dict(payload))
    return tasks


def write_json(path: str | Path, payload: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
