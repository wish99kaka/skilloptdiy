"""Append-only usage ledger helpers for optimizer experiments."""

from __future__ import annotations

import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


def estimate_tokens_from_chars(chars: int) -> int:
    """Return a coarse token estimate when a provider does not report usage."""

    if chars <= 0:
        return 0
    return int(math.ceil(chars / 4))


def append_usage_event(path: str | Path | None, event: dict[str, Any]) -> None:
    """Append one JSONL event if a ledger path is configured."""

    if path is None:
        return
    ledger_path = Path(path)
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "schema_version": 1,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        **_json_safe(event),
    }
    with ledger_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def usage_context_from_env() -> dict[str, Any]:
    raw = os.environ.get("TEXTSKILL_USAGE_CONTEXT_JSON", "")
    if not raw.strip():
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {"raw": raw}
    return parsed if isinstance(parsed, dict) else {"value": parsed}


def extract_chat_usage(response_payload: dict[str, Any]) -> dict[str, int | None]:
    usage = response_payload.get("usage") if isinstance(response_payload, dict) else None
    if not isinstance(usage, dict):
        return {"prompt_tokens": None, "completion_tokens": None, "total_tokens": None}
    prompt = _as_int(usage.get("prompt_tokens"))
    completion = _as_int(usage.get("completion_tokens"))
    total = _as_int(usage.get("total_tokens"))
    if total is None and (prompt is not None or completion is not None):
        total = (prompt or 0) + (completion or 0)
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": total,
    }


def summarize_usage_file(
    path: str | Path,
    *,
    include_kinds: Iterable[str] | None = None,
    exclude_kinds: Iterable[str] | None = None,
) -> dict[str, Any]:
    return summarize_usage_events(
        read_usage_events(path),
        include_kinds=include_kinds,
        exclude_kinds=exclude_kinds,
    )


def summarize_usage_files(
    paths: Iterable[str | Path],
    *,
    include_kinds: Iterable[str] | None = None,
    exclude_kinds: Iterable[str] | None = None,
) -> dict[str, Any]:
    events: list[dict[str, Any]] = []
    for path in paths:
        events.extend(read_usage_events(path))
    return summarize_usage_events(
        events,
        include_kinds=include_kinds,
        exclude_kinds=exclude_kinds,
    )


def read_usage_events(path: str | Path) -> list[dict[str, Any]]:
    ledger_path = Path(path)
    if not ledger_path.exists():
        return []
    events = []
    for line in ledger_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            events.append(payload)
    return events


def summarize_usage_events(
    events: Iterable[dict[str, Any]],
    *,
    include_kinds: Iterable[str] | None = None,
    exclude_kinds: Iterable[str] | None = None,
) -> dict[str, Any]:
    summary = _empty_summary()
    by_kind: dict[str, dict[str, Any]] = {}
    by_operation: dict[str, dict[str, Any]] = {}
    for event in filter_usage_events(events, include_kinds=include_kinds, exclude_kinds=exclude_kinds):
        _add_event(summary, event)
        kind = str(event.get("kind") or "unknown")
        operation = str(event.get("operation") or "unknown")
        _add_event(by_kind.setdefault(kind, _empty_summary(include_breakdowns=False)), event)
        _add_event(
            by_operation.setdefault(f"{kind}:{operation}", _empty_summary(include_breakdowns=False)),
            event,
        )
    summary["by_kind"] = by_kind
    summary["by_operation"] = by_operation
    return summary


def filter_usage_events(
    events: Iterable[dict[str, Any]],
    *,
    include_kinds: Iterable[str] | None = None,
    exclude_kinds: Iterable[str] | None = None,
) -> list[dict[str, Any]]:
    include = {str(kind) for kind in include_kinds or []}
    exclude = {str(kind) for kind in exclude_kinds or []}
    filtered = []
    for event in events:
        kind = str(event.get("kind") or "unknown")
        if include and kind not in include:
            continue
        if exclude and kind in exclude:
            continue
        filtered.append(event)
    return filtered


def combine_usage_summaries(summaries: Iterable[dict[str, Any]]) -> dict[str, Any]:
    combined = _empty_summary()
    by_kind: dict[str, dict[str, Any]] = {}
    by_operation: dict[str, dict[str, Any]] = {}
    for summary in summaries:
        _add_summary(combined, summary)
        for key, value in dict(summary.get("by_kind") or {}).items():
            _add_summary(by_kind.setdefault(str(key), _empty_summary(include_breakdowns=False)), value)
        for key, value in dict(summary.get("by_operation") or {}).items():
            _add_summary(
                by_operation.setdefault(str(key), _empty_summary(include_breakdowns=False)),
                value,
            )
    combined["by_kind"] = by_kind
    combined["by_operation"] = by_operation
    return combined


def _empty_summary(*, include_breakdowns: bool = True) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "calls": 0,
        "duration_seconds_total": 0.0,
        "actual_prompt_tokens": 0,
        "actual_completion_tokens": 0,
        "actual_total_tokens": 0,
        "actual_token_events": 0,
        "estimated_prompt_tokens": 0,
        "estimated_completion_tokens": 0,
        "estimated_total_tokens": 0,
        "estimated_token_events": 0,
        "input_chars": 0,
        "output_chars": 0,
    }
    if include_breakdowns:
        summary["by_kind"] = {}
        summary["by_operation"] = {}
    return summary


def _add_event(summary: dict[str, Any], event: dict[str, Any]) -> None:
    summary["calls"] += int(event.get("call_count") or 1)
    summary["duration_seconds_total"] += float(event.get("duration_seconds") or 0.0)
    summary["input_chars"] += int(event.get("input_chars") or 0)
    summary["output_chars"] += int(event.get("output_chars") or 0)

    actual_prompt = _as_int(event.get("actual_prompt_tokens"))
    actual_completion = _as_int(event.get("actual_completion_tokens"))
    actual_total = _as_int(event.get("actual_total_tokens"))
    if actual_total is None and (actual_prompt is not None or actual_completion is not None):
        actual_total = (actual_prompt or 0) + (actual_completion or 0)
    if actual_prompt is not None or actual_completion is not None or actual_total is not None:
        summary["actual_token_events"] += 1
        summary["actual_prompt_tokens"] += actual_prompt or 0
        summary["actual_completion_tokens"] += actual_completion or 0
        summary["actual_total_tokens"] += actual_total or 0

    estimated_prompt = _as_int(event.get("estimated_prompt_tokens"))
    estimated_completion = _as_int(event.get("estimated_completion_tokens"))
    estimated_total = _as_int(event.get("estimated_total_tokens"))
    if estimated_total is None and (estimated_prompt is not None or estimated_completion is not None):
        estimated_total = (estimated_prompt or 0) + (estimated_completion or 0)
    if estimated_prompt is not None or estimated_completion is not None or estimated_total is not None:
        summary["estimated_token_events"] += 1
        summary["estimated_prompt_tokens"] += estimated_prompt or 0
        summary["estimated_completion_tokens"] += estimated_completion or 0
        summary["estimated_total_tokens"] += estimated_total or 0


def _add_summary(target: dict[str, Any], source: dict[str, Any]) -> None:
    for key in (
        "calls",
        "actual_prompt_tokens",
        "actual_completion_tokens",
        "actual_total_tokens",
        "actual_token_events",
        "estimated_prompt_tokens",
        "estimated_completion_tokens",
        "estimated_total_tokens",
        "estimated_token_events",
        "input_chars",
        "output_chars",
    ):
        target[key] += int(source.get(key) or 0)
    target["duration_seconds_total"] += float(source.get("duration_seconds_total") or 0.0)


def _as_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value
