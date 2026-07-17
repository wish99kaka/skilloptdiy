"""Role-isolated SearchQA controller process used by the M7 entrypoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from .controller_process import canonical_json, canonical_json_sha256
from .epoch_plan import PaperEpochPlan
from .searchqa import SearchQAItem, load_searchqa_items, score_searchqa_response


SCRIPTED_IMPROVEMENT_TOKEN = "Return only the shortest answer supported by context."
MAX_CONTEXT_CHARS = 6000
TARGET_WORKERS = 24
_USAGE_LOCK = threading.Lock()


@dataclass(frozen=True)
class TargetResult:
    response: str
    prompt_tokens: int
    completion_tokens: int
    tokens_are_actual: bool
    duration_seconds: float


class TargetBudgetGuard:
    def __init__(self, args: argparse.Namespace) -> None:
        records = _read_usage(args.usage_ledger) + _read_usage(
            args.peer_usage_ledger
        )
        self._calls = len(records)
        self._tokens = sum(int(record.get("total_tokens", 0)) for record in records)
        self._reserved_tokens = 0
        self._stopped = False
        self._call_cap = args.target_call_cap
        self._token_cap = args.target_token_cap
        self._deadline = args.deadline_monotonic
        self._lock = threading.Lock()

    def reserve(self, *, estimated_prompt_tokens: int) -> int:
        with self._lock:
            self._require_time()
            if self._stopped:
                raise RuntimeError("budget_breach stop condition already triggered")
            if self._calls + 1 > self._call_cap:
                raise RuntimeError("budget_breach stop condition triggered: target_calls")
            if (
                self._tokens + self._reserved_tokens + estimated_prompt_tokens
                > self._token_cap
            ):
                self._stopped = True
                raise RuntimeError("budget_breach stop condition triggered: target_tokens")
            self._calls += 1
            self._reserved_tokens += estimated_prompt_tokens
            return estimated_prompt_tokens

    def settle(self, total_tokens: int, *, reservation: int) -> None:
        with self._lock:
            self._reserved_tokens -= reservation
            self._tokens += total_tokens
            if self._tokens + self._reserved_tokens > self._token_cap:
                self._stopped = True
                raise RuntimeError("budget_breach stop condition triggered: target_tokens")
            self._require_time()

    def remaining_seconds(self) -> float:
        with self._lock:
            self._require_time()
            return self._deadline - time.monotonic()

    def _require_time(self) -> None:
        if time.monotonic() >= self._deadline:
            raise RuntimeError("budget_breach stop condition triggered: wall_time_seconds")


def run_controller(role: str, argv: Sequence[str] | None = None) -> int:
    if role not in {"train", "selection"}:
        raise ValueError("SearchQA controller role must be train or selection")
    parser = _parser(role)
    args = parser.parse_args(argv)
    try:
        request = json.load(__import__("sys").stdin)
        items = load_searchqa_items(args.data)
        guard = TargetBudgetGuard(args) if args.backend == "coco" else None
        if role == "train":
            payload = _collect_train(args, request, items, guard=guard)
        else:
            payload = _score_selection(args, request, items, guard=guard)
        _emit_signed_response(args, request, payload)
    except Exception as error:
        print(str(error), file=__import__("sys").stderr)
        return 2
    return 0


def _parser(role: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--controller-id", required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--private-key", type=Path, required=True)
    parser.add_argument("--backend", choices=("scripted", "coco"), required=True)
    parser.add_argument("--target-model", required=True)
    parser.add_argument("--target-reasoning", required=True)
    parser.add_argument("--usage-ledger", type=Path, required=True)
    parser.add_argument("--peer-usage-ledger", type=Path, required=True)
    parser.add_argument("--target-call-cap", type=int, required=True)
    parser.add_argument("--target-token-cap", type=int, required=True)
    parser.add_argument("--deadline-monotonic", type=float, required=True)
    parser.add_argument("--coco-binary", type=Path)
    parser.add_argument("--rollout-prompt", type=Path, required=True)
    if role == "train":
        parser.add_argument("--plan", type=Path, required=True)
    return parser


def _collect_train(
    args: argparse.Namespace,
    request: dict[str, Any],
    items: tuple[SearchQAItem, ...],
    *,
    guard: TargetBudgetGuard | None,
) -> dict[str, Any]:
    expected = {
        "operation",
        "skill_text",
        "split_id",
        "split_manifest_sha256",
        "batch_id",
        "batch_seed",
        "batch_size",
    }
    if type(request) is not dict or set(request) != expected:
        raise ValueError("train controller request fields do not match the contract")
    if request["operation"] != "collect_train":
        raise ValueError("train controller received an invalid operation")
    batch = _planned_batch(
        items,
        plan_path=args.plan,
        batch_id=request["batch_id"],
        batch_seed=request["batch_seed"],
        batch_size=request["batch_size"],
    )
    with ThreadPoolExecutor(max_workers=min(TARGET_WORKERS, len(batch))) as executor:
        trajectories = list(
            executor.map(
                lambda item: _trajectory(
                    args, request["skill_text"], item, guard=guard
                ),
                batch,
            )
        )
    return {
        "split_id": request["split_id"],
        "split_manifest_sha256": request["split_manifest_sha256"],
        "batch_id": request["batch_id"],
        "batch_seed": request["batch_seed"],
        "batch_size": request["batch_size"],
        "trajectories": trajectories,
    }


def _score_selection(
    args: argparse.Namespace,
    request: dict[str, Any],
    items: tuple[SearchQAItem, ...],
    *,
    guard: TargetBudgetGuard | None,
) -> dict[str, float]:
    expected = {"operation", "skill_text", "split_id", "split_manifest_sha256"}
    if type(request) is not dict or set(request) != expected:
        raise ValueError("selection controller request fields do not match the contract")
    if request["operation"] != "score_selection":
        raise ValueError("selection controller received an invalid operation")
    with ThreadPoolExecutor(max_workers=min(TARGET_WORKERS, len(items))) as executor:
        scores = list(
            executor.map(
                lambda item: _evaluate_target(
                    args, request["skill_text"], item, guard=guard
                )[1],
                items,
            )
        )
    return {"score": float(sum(scores) / len(scores))}


def _trajectory(
    args: argparse.Namespace,
    skill_text: str,
    item: SearchQAItem,
    *,
    guard: TargetBudgetGuard | None,
) -> dict[str, Any]:
    result, exact_match = _evaluate_target(args, skill_text, item, guard=guard)
    return {
        "task_id": item.item_id,
        "task_input": {"question": item.question, "context": item.context},
        "output": result.response,
        "score": exact_match,
        "success": exact_match == 1.0,
        "trace": ["SearchQA response scored by the frozen exact-match contract"],
    }


def _evaluate_target(
    args: argparse.Namespace,
    skill_text: str,
    item: SearchQAItem,
    *,
    guard: TargetBudgetGuard | None,
) -> tuple[TargetResult, float]:
    prompt_template = args.rollout_prompt.read_text(encoding="utf-8")
    skill_section = f"## Skill\n{skill_text.strip()}\n\n" if skill_text.strip() else ""
    system = prompt_template.format(skill_section=skill_section)
    context = _truncate_context(item.context)
    prompt = f"{system}\n\n## Context\n{context}\n\n## Question\n{item.question}"
    token_reservation = 0
    if guard is not None:
        token_reservation = guard.reserve(
            estimated_prompt_tokens=_estimate_tokens(prompt)
        )
    if args.backend == "scripted":
        result = _scripted_target(prompt, skill_text, item)
    else:
        try:
            result = _coco_target(args, prompt, guard=guard)
        except Exception as error:
            _append_failed_usage(args, item, prompt=prompt, error=error)
            raise
    score = score_searchqa_response(result.response, item.answers).exact_match
    _append_usage(args, item, result, prompt=prompt)
    if guard is not None:
        guard.settle(
            result.prompt_tokens + result.completion_tokens,
            reservation=token_reservation,
        )
    return result, score


def _scripted_target(
    prompt: str,
    skill_text: str,
    item: SearchQAItem,
) -> TargetResult:
    started = time.monotonic()
    match = re.search(r"(\d+)$", item.item_id)
    baseline_success = (
        int(match.group(1)) % 2 == 0
        if match is not None
        else int(hashlib.sha256(item.item_id.encode()).hexdigest(), 16) % 2 == 0
    )
    if SCRIPTED_IMPROVEMENT_TOKEN in skill_text or baseline_success:
        answer = item.answers[0]
    else:
        answer = "__scripted_wrong_answer__"
    response = f"<answer>{answer}</answer>"
    return TargetResult(
        response=response,
        prompt_tokens=0,
        completion_tokens=0,
        tokens_are_actual=True,
        duration_seconds=time.monotonic() - started,
    )


def _coco_target(
    args: argparse.Namespace,
    prompt: str,
    *,
    guard: TargetBudgetGuard | None,
) -> TargetResult:
    if args.coco_binary is None or not args.coco_binary.is_file():
        raise RuntimeError("preregistered Coco binary is missing")
    timeout = 120.0
    if guard is not None:
        timeout = min(timeout, guard.remaining_seconds())
    argv = [
        str(args.coco_binary),
        "--print",
        "--output-format",
        "json",
        "--permission-mode",
        "bypass_permissions",
        "--query-timeout",
        "2m",
        prompt,
    ]
    started = time.monotonic()
    completed = subprocess.run(
        argv,
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )
    duration = time.monotonic() - started
    if completed.returncode != 0:
        raise RuntimeError(
            f"Coco target failed with exit code {completed.returncode}: "
            f"{completed.stderr[-500:]}"
        )
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError("Coco JSON output is not valid JSON") from error
    response = _find_response_text(payload)
    usage = _find_usage(payload)
    if usage is None:
        raise RuntimeError(
            "Coco JSON output did not expose token usage; paid M7 execution is blocked"
        )
    prompt_tokens = _usage_integer(usage, ("prompt_tokens", "input_tokens"))
    completion_tokens = _usage_integer(
        usage, ("completion_tokens", "output_tokens")
    )
    return TargetResult(
        response=response,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        tokens_are_actual=True,
        duration_seconds=duration,
    )


def _find_response_text(payload: Any) -> str:
    if type(payload) is str and payload.strip():
        return payload
    if type(payload) is dict:
        for key in ("result", "final_message", "output", "content", "text"):
            value = payload.get(key)
            if type(value) is str and value.strip():
                return value
        for value in payload.values():
            try:
                return _find_response_text(value)
            except RuntimeError:
                pass
    if type(payload) is list:
        for value in reversed(payload):
            try:
                return _find_response_text(value)
            except RuntimeError:
                pass
    raise RuntimeError("Coco JSON output did not contain a final response")


def _find_usage(payload: Any) -> dict[str, Any] | None:
    if type(payload) is dict:
        usage = payload.get("usage")
        if type(usage) is dict:
            return usage
        for value in payload.values():
            found = _find_usage(value)
            if found is not None:
                return found
    elif type(payload) is list:
        for value in reversed(payload):
            found = _find_usage(value)
            if found is not None:
                return found
    return None


def _usage_integer(usage: dict[str, Any], names: tuple[str, ...]) -> int:
    for name in names:
        value = usage.get(name)
        if type(value) is int and value >= 0:
            return value
    raise RuntimeError(f"Coco token usage is missing {names[0]}")


def _planned_batch(
    items: tuple[SearchQAItem, ...],
    *,
    plan_path: Path,
    batch_id: str,
    batch_seed: int,
    batch_size: int,
) -> tuple[SearchQAItem, ...]:
    plan = PaperEpochPlan.from_mapping(json.loads(plan_path.read_text(encoding="utf-8")))
    location: tuple[int, int, int] | None = None
    for epoch_index, epoch in enumerate(plan.batch_ids, 1):
        for step_index, step in enumerate(epoch, 1):
            for accumulation_index, planned_id in enumerate(step, 1):
                if planned_id == batch_id:
                    location = (epoch_index, step_index, accumulation_index)
    if location is None:
        if batch_size > len(items):
            raise ValueError("longitudinal batch exceeds the registered train split")
        values = list(items)
        random.Random(batch_seed).shuffle(values)
        return tuple(values[:batch_size])
    epoch, step, accumulation = location
    values = list(items)
    random.Random(plan.split_seed + epoch * 1000).shuffle(values)
    batch_index = (step - 1) * plan.mechanisms.accumulation + accumulation - 1
    start = batch_index * batch_size
    selected = values[start : start + batch_size]
    if len(selected) != batch_size:
        raise ValueError("scheduled SearchQA batch is not full")
    return tuple(selected)


def _truncate_context(context: str) -> str:
    if len(context) <= MAX_CONTEXT_CHARS:
        return context
    documents = context.split("[DOC]")
    result = ""
    for document in documents:
        candidate = result + "[DOC]" + document if result else document
        if len(candidate) > MAX_CONTEXT_CHARS:
            break
        result = candidate
    return result or context[:MAX_CONTEXT_CHARS] + "\n...[truncated]"


def _append_usage(
    args: argparse.Namespace,
    item: SearchQAItem,
    result: TargetResult,
    *,
    prompt: str,
) -> None:
    args.usage_ledger.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "kind": "target_model" if args.backend == "coco" else "scripted_target",
        "model_id": args.target_model,
        "reasoning": args.target_reasoning,
        "task_id": item.item_id,
        "external_call": args.backend == "coco",
        "prompt_tokens": result.prompt_tokens,
        "completion_tokens": result.completion_tokens,
        "total_tokens": result.prompt_tokens + result.completion_tokens,
        "estimated_prompt_tokens": _estimate_tokens(prompt),
        "estimated_completion_tokens": _estimate_tokens(result.response),
        "tokens_are_actual": result.tokens_are_actual,
        "duration_seconds": result.duration_seconds,
    }
    with _USAGE_LOCK:
        with args.usage_ledger.open("a", encoding="utf-8") as stream:
            stream.write(canonical_json(record) + "\n")


def _append_failed_usage(
    args: argparse.Namespace,
    item: SearchQAItem,
    *,
    prompt: str,
    error: Exception,
) -> None:
    args.usage_ledger.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "kind": "target_model",
        "model_id": args.target_model,
        "reasoning": args.target_reasoning,
        "task_id": item.item_id,
        "external_call": True,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "estimated_prompt_tokens": _estimate_tokens(prompt),
        "estimated_completion_tokens": 0,
        "tokens_are_actual": False,
        "duration_seconds": 0.0,
        "failed": True,
        "error_type": type(error).__name__,
    }
    with _USAGE_LOCK:
        with args.usage_ledger.open("a", encoding="utf-8") as stream:
            stream.write(canonical_json(record) + "\n")


def _read_usage(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _estimate_tokens(value: str) -> int:
    return max(1, (len(value) + 3) // 4)


def _emit_signed_response(
    args: argparse.Namespace,
    request: dict[str, Any],
    payload: dict[str, Any],
) -> None:
    private_hex = args.private_key.read_text(encoding="utf-8").strip()
    private_key = Ed25519PrivateKey.from_private_bytes(bytes.fromhex(private_hex))
    signed = {
        "controller_id": args.controller_id,
        "request_sha256": canonical_json_sha256(request),
        "payload": payload,
    }
    signature = private_key.sign(canonical_json(signed).encode("utf-8")).hex()
    print(canonical_json({**signed, "signature": signature}))
