"""Role-isolated SearchQA controller process used by the M7 entrypoint."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import queue
import random
import re
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from .controller_process import canonical_json, canonical_json_sha256
from .epoch_plan import PaperEpochPlan
from .searchqa import SearchQAItem, load_searchqa_items, score_searchqa_response


SCRIPTED_IMPROVEMENT_TOKEN = "Return only the shortest answer supported by context."
MAX_CONTEXT_CHARS = 6000
TARGET_WORKERS = 24
COCO_ACP_WORKERS = 5
ACP_PROTOCOL_VERSION = 1
ACP_STARTUP_TIMEOUT_SECONDS = 30.0
_USAGE_LOCK = threading.Lock()


@dataclass(frozen=True)
class TargetResult:
    response: str
    prompt_tokens: int
    completion_tokens: int
    tokens_are_actual: bool
    duration_seconds: float


class CocoTargetInvocationError(RuntimeError):
    """A Coco target failure annotated with whether the prompt was dispatched."""

    def __init__(
        self,
        message: str,
        *,
        external_call: bool,
        duration_seconds: float,
    ) -> None:
        super().__init__(message)
        self.external_call = external_call
        self.duration_seconds = duration_seconds


class _ACPConnection:
    """One serial JSON-RPC connection over ACP's line-delimited stdio transport."""

    _CLOSED = object()

    def __init__(self, process: subprocess.Popen[str]) -> None:
        if process.stdin is None or process.stdout is None or process.stderr is None:
            raise RuntimeError("Coco ACP process pipes are unavailable")
        self._process = process
        self._stdin = process.stdin
        self._stdout = process.stdout
        self._stderr = process.stderr
        self._events: queue.Queue[object] = queue.Queue()
        self._stderr_tail: collections.deque[str] = collections.deque(maxlen=80)
        self._messages: dict[str, list[str]] = {}
        self._request_id = 0
        self._write_lock = threading.Lock()
        threading.Thread(target=self._read_stdout, daemon=True).start()
        threading.Thread(target=self._read_stderr, daemon=True).start()

    def request(
        self,
        method: str,
        params: dict[str, Any],
        *,
        timeout: float,
        on_sent: Callable[[], None] | None = None,
    ) -> dict[str, Any]:
        if timeout <= 0:
            raise TimeoutError(f"Coco ACP {method} timed out")
        self._request_id += 1
        request_id = self._request_id
        self._write(
            {
                "jsonrpc": "2.0",
                "id": request_id,
                "method": method,
                "params": params,
            }
        )
        if on_sent is not None:
            on_sent()
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"Coco ACP {method} timed out")
            try:
                event = self._events.get(timeout=remaining)
            except queue.Empty as error:
                raise TimeoutError(f"Coco ACP {method} timed out") from error
            if event is self._CLOSED:
                detail = "".join(self._stderr_tail)[-500:].strip()
                suffix = f": {detail}" if detail else ""
                raise RuntimeError(f"Coco ACP connection closed during {method}{suffix}")
            if isinstance(event, Exception):
                raise event
            if type(event) is not dict:
                raise RuntimeError("Coco ACP emitted a non-object JSON-RPC message")
            incoming = event
            incoming_method = incoming.get("method")
            if type(incoming_method) is str:
                self._handle_incoming_method(incoming)
                continue
            if incoming.get("id") != request_id:
                raise RuntimeError("Coco ACP returned an unexpected JSON-RPC response id")
            error_payload = incoming.get("error")
            if error_payload is not None:
                raise RuntimeError(
                    f"Coco ACP {method} failed: "
                    f"{json.dumps(error_payload, ensure_ascii=False, sort_keys=True)}"
                )
            result = incoming.get("result")
            if type(result) is not dict:
                raise RuntimeError(f"Coco ACP {method} result must be an object")
            return result

    def take_message(self, session_id: str) -> str:
        return "".join(self._messages.pop(session_id, [])).strip()

    def close(self) -> None:
        try:
            self._stdin.close()
        except OSError:
            pass
        if self._process.poll() is not None:
            return
        self._process.terminate()
        try:
            self._process.wait(timeout=3.0)
        except subprocess.TimeoutExpired:
            self._process.kill()
            self._process.wait(timeout=3.0)

    def _read_stdout(self) -> None:
        try:
            for line in self._stdout:
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError as error:
                    self._events.put(
                        RuntimeError("Coco ACP emitted invalid JSON-RPC output")
                    )
                    self._stderr_tail.append(f"stdout parse error: {error}\n")
                    continue
                self._events.put(payload)
        finally:
            self._events.put(self._CLOSED)

    def _read_stderr(self) -> None:
        for line in self._stderr:
            self._stderr_tail.append(line)

    def _write(self, payload: dict[str, Any]) -> None:
        serialized = canonical_json(payload)
        try:
            with self._write_lock:
                self._stdin.write(serialized + "\n")
                self._stdin.flush()
        except (BrokenPipeError, OSError) as error:
            detail = "".join(self._stderr_tail)[-500:].strip()
            suffix = f": {detail}" if detail else ""
            raise RuntimeError(f"Coco ACP connection write failed{suffix}") from error

    def _handle_incoming_method(self, payload: dict[str, Any]) -> None:
        if payload["method"] == "session/update":
            params = payload.get("params")
            if type(params) is not dict:
                raise RuntimeError("Coco ACP session/update params must be an object")
            session_id = params.get("sessionId")
            update = params.get("update")
            if type(session_id) is not str or type(update) is not dict:
                raise RuntimeError("Coco ACP session/update is malformed")
            if update.get("sessionUpdate") == "agent_message_chunk":
                content = update.get("content")
                if type(content) is not dict or content.get("type") != "text":
                    raise RuntimeError("Coco ACP agent message chunk must be text")
                text = content.get("text")
                if type(text) is not str:
                    raise RuntimeError("Coco ACP agent message text is malformed")
                self._messages.setdefault(session_id, []).append(text)
        if "id" in payload:
            self._write(
                {
                    "jsonrpc": "2.0",
                    "id": payload["id"],
                    "error": {"code": -32601, "message": "Method not found"},
                }
            )


class _CocoACPWorker:
    def __init__(
        self,
        *,
        worker_id: int,
        cwd: Path,
        connection: _ACPConnection,
        primed_session_id: str,
    ) -> None:
        self.worker_id = worker_id
        self._cwd = cwd
        self._connection = connection
        self._primed_session_id = primed_session_id
        self._lock = threading.Lock()

    @classmethod
    def start(
        cls,
        *,
        worker_id: int,
        binary: Path,
        cwd: Path,
        timeout: float,
    ) -> _CocoACPWorker:
        if not binary.is_file():
            raise RuntimeError("preregistered Coco binary is missing")
        process = subprocess.Popen(
            [
                str(binary),
                "acp",
                "serve",
                "--permission-mode",
                "bypass_permissions",
            ],
            cwd=str(cwd),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        connection = _ACPConnection(process)
        deadline = time.monotonic() + timeout
        try:
            initialized = connection.request(
                "initialize",
                {
                    "protocolVersion": ACP_PROTOCOL_VERSION,
                    "clientCapabilities": {
                        "fs": {"readTextFile": False, "writeTextFile": False},
                        "terminal": False,
                    },
                },
                timeout=_remaining(deadline, "Coco ACP initialization"),
            )
            if initialized.get("protocolVersion") != ACP_PROTOCOL_VERSION:
                raise RuntimeError("Coco ACP protocol version is not supported")
            session_id = cls._new_session(
                connection,
                cwd,
                timeout=_remaining(deadline, "Coco ACP session initialization"),
            )
        except Exception:
            connection.close()
            raise
        return cls(
            worker_id=worker_id,
            cwd=cwd,
            connection=connection,
            primed_session_id=session_id,
        )

    def prompt(self, prompt: str, *, timeout: float) -> TargetResult:
        with self._lock:
            started = time.monotonic()
            deadline = started + timeout
            external_call = False

            def mark_external_call() -> None:
                nonlocal external_call
                external_call = True

            try:
                session_id = self._primed_session_id
                if session_id:
                    self._primed_session_id = ""
                else:
                    session_id = self._new_session(
                        self._connection,
                        self._cwd,
                        timeout=_remaining(deadline, "Coco ACP session creation"),
                    )
                result = self._connection.request(
                    "session/prompt",
                    {
                        "sessionId": session_id,
                        "prompt": [{"type": "text", "text": prompt}],
                    },
                    timeout=_remaining(deadline, "Coco ACP prompt"),
                    on_sent=mark_external_call,
                )
                response = self._connection.take_message(session_id)
                if not response:
                    raise RuntimeError("Coco ACP prompt did not emit an assistant message")
                usage = result.get("usage")
                prompt_tokens = _optional_usage_integer(usage, "inputTokens")
                completion_tokens = _optional_usage_integer(usage, "outputTokens")
                tokens_are_actual = (
                    prompt_tokens is not None and completion_tokens is not None
                )
                return TargetResult(
                    response=response,
                    prompt_tokens=prompt_tokens if tokens_are_actual else 0,
                    completion_tokens=completion_tokens if tokens_are_actual else 0,
                    tokens_are_actual=tokens_are_actual,
                    duration_seconds=time.monotonic() - started,
                )
            except Exception as error:
                if isinstance(error, CocoTargetInvocationError):
                    raise
                raise CocoTargetInvocationError(
                    str(error),
                    external_call=external_call,
                    duration_seconds=time.monotonic() - started,
                ) from error

    def close(self) -> None:
        self._connection.close()

    @staticmethod
    def _new_session(
        connection: _ACPConnection,
        cwd: Path,
        *,
        timeout: float,
    ) -> str:
        result = connection.request(
            "session/new",
            {"cwd": str(cwd), "mcpServers": []},
            timeout=timeout,
        )
        session_id = result.get("sessionId")
        if type(session_id) is not str or not session_id:
            raise RuntimeError("Coco ACP returned an invalid session id")
        return session_id


class CocoACPWorkerPool:
    """Five isolated Coco processes, serialized at startup and concurrent afterward."""

    def __init__(self, workers: list[Any]) -> None:
        self._workers = workers
        self._available: queue.Queue[Any] = queue.Queue()
        for worker in workers:
            self._available.put(worker)
        self._failure: Exception | None = None
        self._state_lock = threading.Lock()
        self._closed = False

    @classmethod
    def start(
        cls,
        *,
        binary: Path,
        cwd: Path,
        task_count: int,
        startup_timeout: Callable[[], float],
        worker_factory: Callable[..., Any] | None = None,
    ) -> CocoACPWorkerPool:
        if task_count <= 0:
            raise ValueError("Coco ACP pool requires at least one task")
        factory = worker_factory or _CocoACPWorker.start
        workers: list[Any] = []
        try:
            for worker_id in range(min(COCO_ACP_WORKERS, task_count)):
                available_startup_time = startup_timeout()
                if available_startup_time <= 0:
                    raise TimeoutError("Coco ACP worker startup timed out")
                workers.append(
                    factory(
                        worker_id=worker_id,
                        binary=binary,
                        cwd=cwd,
                        timeout=min(
                            ACP_STARTUP_TIMEOUT_SECONDS,
                            available_startup_time,
                        ),
                    )
                )
        except Exception:
            for worker in reversed(workers):
                worker.close()
            raise
        return cls(workers)

    def prompt(self, prompt: str, *, timeout: float) -> TargetResult:
        deadline = time.monotonic() + timeout
        with self._state_lock:
            self._require_healthy()
        try:
            worker = self._available.get(
                timeout=_remaining(deadline, "Coco ACP worker acquisition")
            )
        except queue.Empty as error:
            raise TimeoutError("Coco ACP worker acquisition timed out") from error
        try:
            with self._state_lock:
                self._require_healthy()
            return worker.prompt(
                prompt,
                timeout=_remaining(deadline, "Coco ACP target prompt"),
            )
        except Exception as error:
            with self._state_lock:
                if self._failure is None:
                    self._failure = error
            raise
        finally:
            self._available.put(worker)

    def close(self) -> None:
        with self._state_lock:
            if self._closed:
                return
            self._closed = True
        for worker in reversed(self._workers):
            worker.close()

    def _require_healthy(self) -> None:
        if self._closed:
            raise RuntimeError("Coco ACP worker pool is closed")
        if self._failure is not None:
            raise RuntimeError("Coco ACP worker pool stopped after a worker failure") from self._failure


def _remaining(deadline: float, operation: str) -> float:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError(f"{operation} timed out")
    return remaining


class TargetBudgetGuard:
    def __init__(self, args: argparse.Namespace) -> None:
        records = _read_usage(args.usage_ledger) + _read_usage(
            args.peer_usage_ledger
        )
        self._calls = len(records)
        self._call_cap = args.target_call_cap
        self._deadline = args.deadline_monotonic
        self._lock = threading.Lock()

    def reserve(self, *, estimated_prompt_tokens: int) -> int:
        with self._lock:
            self._require_time()
            if self._calls + 1 > self._call_cap:
                raise RuntimeError("budget_breach stop condition triggered: target_calls")
            self._calls += 1
            return 0

    def settle(self, total_tokens: int, *, reservation: int) -> None:
        with self._lock:
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
    else:
        parser.add_argument("--selection-audit", type=Path, required=True)
        parser.add_argument("--selection-audit-key", type=Path, required=True)
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
    coco_pool = _start_coco_pool(args, task_count=len(batch), guard=guard)
    try:
        with ThreadPoolExecutor(
            max_workers=_target_worker_count(args, len(batch))
        ) as executor:
            trajectories = list(
                executor.map(
                    lambda item: _trajectory(
                        args,
                        request["skill_text"],
                        item,
                        guard=guard,
                        coco_pool=coco_pool,
                    ),
                    batch,
                )
            )
    finally:
        if coco_pool is not None:
            coco_pool.close()
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
    coco_pool = _start_coco_pool(args, task_count=len(items), guard=guard)
    try:
        with ThreadPoolExecutor(
            max_workers=_target_worker_count(args, len(items))
        ) as executor:
            results = list(
                executor.map(
                    lambda item: _evaluate_target(
                        args,
                        request["skill_text"],
                        item,
                        guard=guard,
                        coco_pool=coco_pool,
                    ),
                    items,
                )
            )
    finally:
        if coco_pool is not None:
            coco_pool.close()
    scores = [score for _, score in results]
    aggregate = float(sum(scores) / len(scores))
    _append_selection_audit(
        args.selection_audit,
        key_path=args.selection_audit_key,
        request=request,
        items=items,
        results=tuple(results),
        score=aggregate,
    )
    return {"score": aggregate}


def _append_selection_audit(
    path: Path,
    *,
    key_path: Path,
    request: dict[str, Any],
    items: tuple[SearchQAItem, ...],
    results: tuple[tuple[TargetResult, float], ...],
    score: float,
) -> None:
    if len(items) != len(results):
        raise RuntimeError("selection audit result count does not match items")
    prior = list(unseal_searchqa_selection_audit(path, key_path=key_path))
    previous_hash = prior[-1]["record_sha256"] if prior else None
    record = {
        "schema_version": "searchqa-selection-audit-v1",
        "sequence": len(prior) + 1,
        "previous_record_sha256": previous_hash,
        "request_sha256": canonical_json_sha256(request),
        "skill_sha256": hashlib.sha256(
            request["skill_text"].encode("utf-8")
        ).hexdigest(),
        "score": score,
        "items": [
            {
                "task_id": item.item_id,
                "exact_match": exact_match,
                "predicted_answer": score_searchqa_response(
                    result.response,
                    item.answers,
                ).predicted_answer,
                "response_sha256": hashlib.sha256(
                    result.response.encode("utf-8")
                ).hexdigest(),
            }
            for item, (result, exact_match) in zip(items, results)
        ],
    }
    record["record_sha256"] = canonical_json_sha256(record)
    sequence = record["sequence"]
    aad = canonical_json(
        {
            "schema_version": "searchqa-selection-audit-sealed-v1",
            "sequence": sequence,
        }
    ).encode("utf-8")
    nonce = os.urandom(12)
    ciphertext = AESGCM(_selection_audit_key(key_path)).encrypt(
        nonce,
        canonical_json(record).encode("utf-8"),
        aad,
    )
    envelope = {
        "schema_version": "searchqa-selection-audit-sealed-v1",
        "sequence": sequence,
        "nonce": nonce.hex(),
        "ciphertext": ciphertext.hex(),
        "ciphertext_sha256": hashlib.sha256(ciphertext).hexdigest(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(canonical_json(envelope) + "\n")


def unseal_searchqa_selection_audit(
    path: Path,
    *,
    key_path: Path,
) -> tuple[dict[str, Any], ...]:
    if not path.exists():
        return ()
    key = _selection_audit_key(key_path)
    records = []
    for expected_sequence, line in enumerate(
        (line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()),
        1,
    ):
        envelope = json.loads(line)
        if type(envelope) is not dict or set(envelope) != {
            "schema_version",
            "sequence",
            "nonce",
            "ciphertext",
            "ciphertext_sha256",
        }:
            raise RuntimeError("sealed selection audit fields are invalid")
        try:
            nonce = bytes.fromhex(envelope["nonce"])
            ciphertext = bytes.fromhex(envelope["ciphertext"])
        except (TypeError, ValueError) as error:
            raise RuntimeError("sealed selection audit encoding is invalid") from error
        if (
            envelope["schema_version"]
            != "searchqa-selection-audit-sealed-v1"
            or envelope["sequence"] != expected_sequence
            or len(nonce) != 12
            or hashlib.sha256(ciphertext).hexdigest()
            != envelope["ciphertext_sha256"]
        ):
            raise RuntimeError("sealed selection audit envelope is invalid")
        aad = canonical_json(
            {
                "schema_version": envelope["schema_version"],
                "sequence": expected_sequence,
            }
        ).encode("utf-8")
        try:
            plaintext = AESGCM(key).decrypt(nonce, ciphertext, aad)
            record = json.loads(plaintext)
        except Exception as error:
            raise RuntimeError("sealed selection audit authentication failed") from error
        if type(record) is not dict:
            raise RuntimeError("sealed selection audit plaintext is invalid")
        records.append(record)
    return tuple(records)


def _selection_audit_key(path: Path) -> bytes:
    try:
        key = bytes.fromhex(path.read_text(encoding="utf-8").strip())
    except (OSError, ValueError) as error:
        raise RuntimeError("selection audit key is unreadable") from error
    if len(key) != 32:
        raise RuntimeError("selection audit key must contain 32 bytes")
    return key


def _trajectory(
    args: argparse.Namespace,
    skill_text: str,
    item: SearchQAItem,
    *,
    guard: TargetBudgetGuard | None,
    coco_pool: CocoACPWorkerPool | None,
) -> dict[str, Any]:
    result, exact_match = _evaluate_target(
        args,
        skill_text,
        item,
        guard=guard,
        coco_pool=coco_pool,
    )
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
    coco_pool: CocoACPWorkerPool | None = None,
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
            result = _coco_target(
                args,
                prompt,
                guard=guard,
                coco_pool=coco_pool,
            )
        except Exception as error:
            _append_failed_usage(
                args,
                item,
                prompt=prompt,
                error=error,
                external_call=(
                    error.external_call
                    if isinstance(error, CocoTargetInvocationError)
                    else False
                ),
            )
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
    coco_pool: CocoACPWorkerPool | None,
) -> TargetResult:
    if args.coco_binary is None or not args.coco_binary.is_file():
        raise RuntimeError("preregistered Coco binary is missing")
    if coco_pool is None:
        raise RuntimeError("Coco ACP worker pool is unavailable")
    timeout = 120.0
    if guard is not None:
        timeout = min(timeout, guard.remaining_seconds())
    return coco_pool.prompt(prompt, timeout=timeout)


def _start_coco_pool(
    args: argparse.Namespace,
    *,
    task_count: int,
    guard: TargetBudgetGuard | None,
) -> CocoACPWorkerPool | None:
    if args.backend != "coco":
        return None
    if args.coco_binary is None or not args.coco_binary.is_file():
        raise RuntimeError("preregistered Coco binary is missing")

    def startup_timeout() -> float:
        return guard.remaining_seconds() if guard is not None else ACP_STARTUP_TIMEOUT_SECONDS

    return CocoACPWorkerPool.start(
        binary=args.coco_binary,
        cwd=Path.cwd().resolve(),
        task_count=task_count,
        startup_timeout=startup_timeout,
    )


def _target_worker_count(args: argparse.Namespace, task_count: int) -> int:
    limit = COCO_ACP_WORKERS if args.backend == "coco" else TARGET_WORKERS
    return min(limit, task_count)


def _optional_usage_integer(usage: Any, name: str) -> int | None:
    if type(usage) is not dict:
        return None
    value = usage.get(name)
    return value if type(value) is int and value >= 0 else None


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
    external_call: bool,
) -> None:
    args.usage_ledger.parent.mkdir(parents=True, exist_ok=True)
    error_message = str(error)
    if len(error_message) > 1000:
        error_message = (
            error_message[:500] + "...[truncated]..." + error_message[-483:]
        )
    record = {
        "kind": "target_model",
        "model_id": args.target_model,
        "reasoning": args.target_reasoning,
        "task_id": item.item_id,
        "external_call": external_call,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "estimated_prompt_tokens": _estimate_tokens(prompt),
        "estimated_completion_tokens": 0,
        "tokens_are_actual": False,
        "duration_seconds": (
            error.duration_seconds
            if isinstance(error, CocoTargetInvocationError)
            else 0.0
        ),
        "failed": True,
        "error_type": type(error).__name__,
        "error_message": error_message,
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
