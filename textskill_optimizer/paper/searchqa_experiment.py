"""Independent M7 SearchQA preparation and execution entrypoint."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from collections import Counter
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import Any, Callable, Mapping

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from .artifacts import PaperArtifactKind, optimizer_request_payload
from .backend import OptimizerRequest, OptimizerResponse, OptimizerStage
from .checkpoint import CheckpointAuthenticator
from .config import load_paper_profile
from .controller_process import (
    ControllerArtifact,
    ControllerRegistration,
    ControllerRegistry,
    ControllerRole,
)
from .data import SelectionController, TrainController
from .epoch_loop import PaperEpochLoop
from .epoch_plan import PaperEpochPlan
from .epoch_plan import PaperMechanismSpec
from .errors import OptimizerProviderError, SkillContractViolation
from .optimization import PaperOptimizationController
from .preregistration import load_paper_preregistration
from .provenance import canonical_json_sha256
from .searchqa import (
    OFFICIAL_SEARCHQA_ID_MANIFEST_SHA256,
    SEARCHQA_DATASET_REPO,
    SEARCHQA_DATASET_REVISION,
    SearchQAContractViolation,
    load_searchqa_items,
    require_searchqa_skill_rollout_compatibility,
    verify_searchqa_materialization_receipt,
)
from .searchqa_controller_runtime import (
    ACP_STARTUP_TIMEOUT_SECONDS,
    COCO_ACP_WORKERS,
    SCRIPTED_IMPROVEMENT_TOKEN,
    unseal_searchqa_selection_audit,
)


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_TRAIN_RUNNER = _PROJECT_ROOT / "scripts" / "paper_searchqa_train_controller.py"
_SELECTION_RUNNER = (
    _PROJECT_ROOT / "scripts" / "paper_searchqa_selection_controller.py"
)
_CONTROLLER_RUNTIME = Path(__file__).resolve().with_name(
    "searchqa_controller_runtime.py"
)
_SEARCHQA_CONTRACT = Path(__file__).resolve().with_name("searchqa.py")
_EXPERIMENT_RUNTIME = Path(__file__).resolve()
_PROFILE_PATH = Path(
    str(
        files("textskill_optimizer.paper").joinpath(
            "profiles", "paper-faithful-v1.json"
        )
    )
).resolve()
_INITIAL_SKILL_PATH = Path(
    str(files("textskill_optimizer.paper").joinpath("searchqa_assets", "initial.md"))
).resolve()
_ROLLOUT_PROMPT_PATH = Path(
    str(
        files("textskill_optimizer.paper").joinpath(
            "searchqa_assets", "rollout_system.md"
        )
    )
).resolve()
_MECHANISM_SMOKE_WALL_TIME_SECONDS = 12_000.0


class PaidBudgetGuard:
    """Thread-safe optimizer call budget and shared wall-deadline guard."""

    def __init__(self, budgets: Mapping[str, Any], *, deadline: float) -> None:
        self._budgets = budgets
        self._deadline = deadline
        self._optimizer_calls = 0
        self._lock = threading.Lock()

    def reserve_optimizer_call(self, *, estimated_tokens: int) -> int:
        with self._lock:
            self._require_time()
            if self._optimizer_calls + 1 > self._budgets["optimizer_calls"]:
                raise RuntimeError("budget_breach stop condition triggered: optimizer_calls")
            self._optimizer_calls += 1
            return 0

    def settle_optimizer_tokens(self, tokens: int, *, reservation: int) -> None:
        with self._lock:
            self._require_time()

    def remaining_seconds(self) -> float:
        with self._lock:
            self._require_time()
            return self._deadline - time.monotonic()

    def check(self) -> None:
        with self._lock:
            self._require_time()

    def _require_time(self) -> None:
        if time.monotonic() >= self._deadline:
            raise RuntimeError("budget_breach stop condition triggered: wall_time_seconds")


@dataclass
class ScriptedSearchQAOptimizerBackend:
    requests: list[OptimizerRequest]
    responses: list[OptimizerResponse]

    def __init__(self) -> None:
        self.requests = []
        self.responses = []

    @property
    def failure_records(self) -> tuple[Mapping[str, Any], ...]:
        return ()

    def complete(self, request: OptimizerRequest) -> OptimizerResponse:
        self.requests.append(request)
        prompt = json.loads(request.prompt)
        edit = {"op": "append", "content": f"- {SCRIPTED_IMPROVEMENT_TOKEN}"}
        merge_edit = {
            **edit,
            "support_count": 1,
            "source_type": "failure",
        }
        if request.stage is OptimizerStage.REFLECT_FAILURE:
            payload: Mapping[str, Any] = {
                "batch_size": len(prompt["trajectories"]),
                "failure_summary": [
                    {
                        "failure_type": "answer_selection",
                        "count": len(prompt["trajectories"]),
                        "description": "response did not match a supported short answer",
                    }
                ],
                "patch": {"reasoning": "prefer the supported short answer", "edits": [edit]},
            }
        elif request.stage is OptimizerStage.REFLECT_SUCCESS:
            payload = {
                "batch_size": len(prompt["trajectories"]),
                "success_patterns": ["short grounded answers can match exactly"],
                "patch": {"reasoning": "preserve concise answers", "edits": []},
            }
        elif request.stage is OptimizerStage.REFINE:
            prior = prompt["prior_patch"]
            payload = {
                "reasoning": "retain the evidence-backed update",
                "edits": [
                    {
                        "op": item["op"],
                        "target": item.get("target", ""),
                        "content": item.get("content", ""),
                    }
                    for item in prior.get("edits", [])
                ],
                "converged": prompt["round"] == prompt["max_rounds"],
            }
        elif request.stage is OptimizerStage.MERGE_FAILURE:
            payload = {"reasoning": "merge recurring failure", "edits": [merge_edit]}
        elif request.stage is OptimizerStage.MERGE_SUCCESS:
            payload = {"reasoning": "no success edit", "edits": []}
        elif request.stage is OptimizerStage.MERGE_FINAL_FAILURE_PRIORITIZED:
            payload = {
                "reasoning": "prioritize the grounded answer failure",
                "edits": [merge_edit],
            }
        elif request.stage is OptimizerStage.RANK_TOP_L:
            payload = {
                "reasoning": "select the single supported update",
                "selected_indices": [0] if prompt["edits"] else [],
            }
        elif request.stage is OptimizerStage.PROPOSE_SLOW_UPDATE:
            payload = {
                "reasoning": "retain durable grounded-answer guidance",
                "slow_update_content": "Prefer a concise answer directly supported by context.",
            }
        elif request.stage is OptimizerStage.UPDATE_META_SKILL:
            payload = {
                "reasoning": "remember the stable optimization direction",
                "meta_skill_content": "Prefer general answer-selection rules over examples.",
            }
        else:
            raise AssertionError(f"unsupported scripted stage: {request.stage.value}")
        response = OptimizerResponse(
            call_id=request.call_id,
            payload=payload,
            model_id="scripted-optimizer-v1",
            usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        )
        self.responses.append(response)
        return response


class OpenAICompatiblePaperOptimizerBackend:
    """Strict JSON adapter with token usage recorded on an audit-only basis."""

    def __init__(
        self,
        *,
        model_id: str,
        reasoning_effort: str,
        budget_guard: PaidBudgetGuard | None = None,
        usage_ledger: Path | None = None,
    ) -> None:
        self.model_id = model_id
        self.reasoning_effort = reasoning_effort
        self.requests: list[OptimizerRequest] = []
        self.responses: list[OptimizerResponse] = []
        self._external_calls = 0
        self._actual_tokens = 0
        self._estimated_tokens = 0
        self._failures: list[dict[str, Any]] = []
        self._budget_guard = budget_guard
        self._usage_ledger = usage_ledger
        self._usage_lock = threading.Lock()
        self._failure_lock = threading.Lock()
        self._base_url = os.environ.get("EXTERNAL_LLM_BASE_URL", "").strip()
        self._api_key = os.environ.get("EXTERNAL_LLM_API_KEY", "").strip()
        configured_model = os.environ.get("EXTERNAL_LLM_MODEL", "").strip()
        if not self._base_url or not self._api_key:
            raise ValueError(
                "paid M7 optimizer requires EXTERNAL_LLM_BASE_URL and EXTERNAL_LLM_API_KEY"
            )
        if configured_model != model_id:
            raise ValueError(
                "EXTERNAL_LLM_MODEL does not match the preregistered optimizer model"
            )

    def complete(self, request: OptimizerRequest) -> OptimizerResponse:
        token_reservation = 0
        if self._budget_guard is not None:
            token_reservation = self._budget_guard.reserve_optimizer_call(
                estimated_tokens=_estimate_tokens(request.system_prompt + request.prompt)
            )
        self.requests.append(request)
        body: dict[str, Any] = {
            "model": self.model_id,
            "messages": [
                {"role": "system", "content": request.system_prompt},
                {"role": "user", "content": request.prompt},
            ],
            "response_format": {"type": "json_object"},
        }
        if self.reasoning_effort != "none":
            body["reasoning_effort"] = self.reasoning_effort
        url = self._base_url.rstrip("/")
        if not url.endswith("/chat/completions"):
            url += "/chat/completions"
        encoded = json.dumps(body, ensure_ascii=False, allow_nan=False).encode("utf-8")
        http_request = urllib.request.Request(
            url,
            data=encoded,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        timeout = 300.0
        if self._budget_guard is not None:
            timeout = min(timeout, self._budget_guard.remaining_seconds())
        estimated_input_tokens = _estimate_tokens(
            request.system_prompt + request.prompt
        )
        raw_body = b""
        try:
            with self._usage_lock:
                self._external_calls += 1
            with urllib.request.urlopen(http_request, timeout=timeout) as response:
                raw_body = response.read()
        except urllib.error.HTTPError as error:
            raw_body = error.read()
            diagnostic = _provider_response_diagnostic(raw_body=raw_body)
            diagnostic["http_status"] = int(error.code)
            failure = self._record_failure(
                request,
                code="optimizer_http_error",
                message=f"external optimizer HTTP {error.code}",
                diagnostic=diagnostic,
                usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                tokens_are_actual=False,
                estimated_total_tokens=estimated_input_tokens,
            )
            raise failure from error
        except (urllib.error.URLError, TimeoutError) as error:
            failure = self._record_failure(
                request,
                code="optimizer_transport_error",
                message=f"external optimizer transport failed: {type(error).__name__}",
                diagnostic=_provider_response_diagnostic(raw_body=raw_body),
                usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                tokens_are_actual=False,
                estimated_total_tokens=estimated_input_tokens,
            )
            raise failure from error

        try:
            provider = json.loads(raw_body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            diagnostic = _provider_response_diagnostic(raw_body=raw_body)
            if isinstance(error, json.JSONDecodeError):
                diagnostic["body_json_error"] = _json_error_diagnostic(error)
            failure = self._record_failure(
                request,
                code="optimizer_provider_body_invalid_json",
                message="external optimizer response body is not valid JSON",
                diagnostic=diagnostic,
                usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                tokens_are_actual=False,
                estimated_total_tokens=estimated_input_tokens,
            )
            raise failure from error

        diagnostic = _provider_response_diagnostic(
            raw_body=raw_body,
            provider=provider,
        )
        if type(provider) is not dict:
            failure = self._record_failure(
                request,
                code="optimizer_provider_envelope_not_object",
                message="external optimizer response envelope is not an object",
                diagnostic=diagnostic,
                usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                tokens_are_actual=False,
                estimated_total_tokens=estimated_input_tokens,
            )
            raise failure

        failed_usage, tokens_are_actual = _optional_provider_usage(
            provider.get("usage")
        )

        def fail(code: str, message: str) -> None:
            raise self._record_failure(
                request,
                code=code,
                message=message,
                diagnostic=diagnostic,
                usage=failed_usage,
                tokens_are_actual=tokens_are_actual,
                estimated_total_tokens=estimated_input_tokens,
            )

        if "choices" not in provider:
            fail(
                "optimizer_provider_choices_missing",
                "external optimizer response is missing choices",
            )
        choices = provider["choices"]
        if type(choices) is not list:
            fail(
                "optimizer_provider_choices_not_array",
                "external optimizer response choices is not an array",
            )
        if not choices:
            fail(
                "optimizer_provider_choices_empty",
                "external optimizer response choices is empty",
            )
        choice = choices[0]
        if type(choice) is not dict:
            fail(
                "optimizer_provider_choice_not_object",
                "external optimizer response first choice is not an object",
            )
        if "message" not in choice:
            fail(
                "optimizer_provider_message_missing",
                "external optimizer response choice is missing message",
            )
        message = choice["message"]
        if type(message) is not dict:
            fail(
                "optimizer_provider_message_not_object",
                "external optimizer response message is not an object",
            )
        if "content" not in message:
            fail(
                "optimizer_provider_content_missing",
                "external optimizer response message is missing content",
            )
        content = message["content"]
        if type(content) is not str:
            fail(
                "optimizer_provider_content_not_string",
                "external optimizer response message content is not a string",
            )
        if not content.strip():
            fail(
                "optimizer_provider_content_empty",
                "external optimizer response message content is empty",
            )
        try:
            payload = json.loads(content)
        except json.JSONDecodeError as error:
            diagnostic["content_json_error"] = _json_error_diagnostic(error)
            failure = self._record_failure(
                request,
                code="optimizer_provider_content_invalid_json",
                message="external optimizer response message content is not valid JSON",
                diagnostic=diagnostic,
                usage=failed_usage,
                tokens_are_actual=tokens_are_actual,
                estimated_total_tokens=estimated_input_tokens,
            )
            raise failure from error
        if type(payload) is not dict:
            fail(
                "optimizer_provider_content_not_object",
                "external optimizer response message content JSON is not an object",
            )
        normalized_usage = failed_usage
        estimated_total_tokens = estimated_input_tokens + _estimate_tokens(
            json.dumps(payload, ensure_ascii=False, sort_keys=True)
        )
        response = OptimizerResponse(
            call_id=request.call_id,
            payload=payload,
            model_id=self.model_id,
            usage=normalized_usage,
        )
        self.responses.append(response)
        self._append_usage(
            request.call_id,
            total_tokens=normalized_usage["total_tokens"],
            estimated_total_tokens=estimated_total_tokens,
            tokens_are_actual=tokens_are_actual,
            failed=False,
        )
        if self._budget_guard is not None:
            self._budget_guard.settle_optimizer_tokens(
                normalized_usage["total_tokens"], reservation=token_reservation
            )
        return response

    @property
    def external_calls(self) -> int:
        with self._usage_lock:
            return self._external_calls

    @property
    def actual_tokens(self) -> int:
        with self._usage_lock:
            return self._actual_tokens

    @property
    def estimated_tokens(self) -> int:
        with self._usage_lock:
            return self._estimated_tokens

    @property
    def failure_records(self) -> tuple[Mapping[str, Any], ...]:
        with self._failure_lock:
            return tuple(dict(record) for record in self._failures)

    def _record_failure(
        self,
        request: OptimizerRequest,
        *,
        code: str,
        message: str,
        diagnostic: Mapping[str, Any],
        usage: Mapping[str, int],
        tokens_are_actual: bool,
        estimated_total_tokens: int,
    ) -> OptimizerProviderError:
        error = OptimizerProviderError(code, message)
        record = {
            "schema_version": "paper-optimizer-failure-v1",
            "failure_code": code,
            "error_type": type(error).__name__,
            "error_message": message,
            "model_id": self.model_id,
            "request": optimizer_request_payload(request),
            "provider_response": dict(diagnostic),
            "usage": {
                "prompt_tokens": int(usage["prompt_tokens"]),
                "completion_tokens": int(usage["completion_tokens"]),
                "total_tokens": int(usage["total_tokens"]),
                "tokens_are_actual": tokens_are_actual,
            },
        }
        with self._failure_lock:
            self._failures.append(record)
        self._append_usage(
            request.call_id,
            total_tokens=int(usage["total_tokens"]),
            estimated_total_tokens=estimated_total_tokens,
            tokens_are_actual=tokens_are_actual,
            failed=True,
            failure_code=code,
        )
        return error

    def _append_usage(
        self,
        call_id: str,
        *,
        total_tokens: int,
        estimated_total_tokens: int,
        tokens_are_actual: bool,
        failed: bool,
        failure_code: str | None = None,
    ) -> None:
        with self._usage_lock:
            self._actual_tokens += total_tokens
            self._estimated_tokens += estimated_total_tokens
            if self._usage_ledger is None:
                return
            self._usage_ledger.parent.mkdir(parents=True, exist_ok=True)
            with self._usage_ledger.open("a", encoding="utf-8") as stream:
                stream.write(
                    json.dumps(
                        {
                            "call_id": call_id,
                            "external_call": True,
                            "failed": failed,
                            "model_id": self.model_id,
                            "total_tokens": total_tokens,
                            "estimated_total_tokens": estimated_total_tokens,
                            "tokens_are_actual": tokens_are_actual,
                            **(
                                {"failure_code": failure_code}
                                if failure_code is not None
                                else {}
                            ),
                        },
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                    + "\n"
                )


def _json_type_name(value: object) -> str:
    if value is None:
        return "null"
    if type(value) is bool:
        return "boolean"
    if type(value) is int or type(value) is float:
        return "number"
    if type(value) is str:
        return "string"
    if type(value) is list:
        return "array"
    if type(value) is dict:
        return "object"
    return type(value).__name__


def _json_error_diagnostic(error: json.JSONDecodeError) -> dict[str, Any]:
    return {
        "type": type(error).__name__,
        "line": error.lineno,
        "column": error.colno,
        "position": error.pos,
    }


def _add_text_field_diagnostic(
    diagnostic: dict[str, Any],
    *,
    prefix: str,
    value: object,
) -> None:
    diagnostic[f"{prefix}_type"] = _json_type_name(value)
    if type(value) is str:
        encoded = value.encode("utf-8")
        diagnostic[f"{prefix}_size_chars"] = len(value)
        diagnostic[f"{prefix}_size_bytes"] = len(encoded)
        diagnostic[f"{prefix}_sha256"] = hashlib.sha256(encoded).hexdigest()


def _provider_response_diagnostic(
    *,
    raw_body: bytes,
    provider: object | None = None,
) -> dict[str, Any]:
    """Return structural provider evidence without storing generated content."""

    diagnostic: dict[str, Any] = {
        "schema_version": "optimizer-provider-response-diagnostic-v1",
        "body_size_bytes": len(raw_body),
        "body_sha256": hashlib.sha256(raw_body).hexdigest(),
    }
    if provider is None:
        return diagnostic
    diagnostic["provider_json_type"] = _json_type_name(provider)
    if type(provider) is not dict:
        return diagnostic
    diagnostic["provider_keys"] = sorted(provider)
    choices = provider.get("choices")
    diagnostic["choices_type"] = _json_type_name(choices)
    if type(choices) is not list:
        return diagnostic
    diagnostic["choices_count"] = len(choices)
    if not choices:
        return diagnostic
    choice = choices[0]
    diagnostic["first_choice_type"] = _json_type_name(choice)
    if type(choice) is not dict:
        return diagnostic
    diagnostic["first_choice_keys"] = sorted(choice)
    finish_reason = choice.get("finish_reason")
    if finish_reason is None or type(finish_reason) is str:
        diagnostic["finish_reason"] = finish_reason
    else:
        diagnostic["finish_reason_type"] = _json_type_name(finish_reason)
    message = choice.get("message")
    diagnostic["message_type"] = _json_type_name(message)
    if type(message) is not dict:
        return diagnostic
    diagnostic["message_keys"] = sorted(message)
    if "content" in message:
        _add_text_field_diagnostic(
            diagnostic,
            prefix="content",
            value=message["content"],
        )
    if "reasoning_content" in message:
        _add_text_field_diagnostic(
            diagnostic,
            prefix="reasoning_content",
            value=message["reasoning_content"],
        )
    if "refusal" in message:
        _add_text_field_diagnostic(
            diagnostic,
            prefix="refusal",
            value=message["refusal"],
        )
    return diagnostic


def prepare_zero_call_searchqa_experiment(
    *,
    run_dir: str | Path,
    train_path: str | Path,
    selection_path: str | Path,
    materialization_receipt_path: str | Path,
    mechanism_smoke_scope: bool = False,
) -> Path:
    """Freeze a zero-call full-graph run over open SearchQA development data."""

    train_source = Path(train_path).resolve()
    selection_source = Path(selection_path).resolve()
    train_items = load_searchqa_items(train_source)
    selection_items = load_searchqa_items(selection_source)
    if len(train_items) != 40:
        raise ValueError("zero-call SearchQA smoke requires exactly 40 train items")
    if not 2 <= len(selection_items) <= 20:
        raise ValueError("zero-call SearchQA smoke requires 2 to 20 selection items")
    if {item.item_id for item in train_items} & {
        item.item_id for item in selection_items
    }:
        raise ValueError("SearchQA train and selection ids must be disjoint")
    materialization = verify_searchqa_materialization_receipt(
        materialization_receipt_path,
        train_path=train_source,
        selection_path=selection_source,
    )

    run_root = Path(run_dir).resolve()
    run_root.mkdir(parents=True, exist_ok=False)

    train_manifest_path = run_root / "train-split-manifest.json"
    selection_manifest_path = run_root / "selection-split-manifest.json"
    train_split_id = "searchqa-smoke-train-" + _sha256(train_source)[:16]
    selection_split_id = "searchqa-smoke-selection-" + _sha256(selection_source)[:16]
    _write_json(
        train_manifest_path,
        _split_manifest(
            split_id=train_split_id,
            role="train",
            items_path=train_source,
            count=len(train_items),
        ),
    )
    _write_json(
        selection_manifest_path,
        _split_manifest(
            split_id=selection_split_id,
            role="selection",
            items_path=selection_source,
            count=len(selection_items),
        ),
    )
    profile = load_paper_profile()
    mechanisms = (
        PaperMechanismSpec.for_mechanism_test(profile, analyst_workers=8)
        if mechanism_smoke_scope
        else None
    )
    plan = PaperEpochPlan.build(
        profile=profile,
        train_split_id=train_split_id,
        train_split_manifest_sha256=_sha256(train_manifest_path),
        steps_per_epoch=1,
        mechanisms=mechanisms,
        epochs_override=2 if mechanism_smoke_scope else None,
    )
    plan_path = run_root / "paper-epoch-plan.json"
    _write_json(plan_path, plan.to_dict())
    train_key_path = run_root / "train-controller.key"
    selection_key_path = run_root / "selection-controller.key"
    train_public = _write_private_key(train_key_path)
    selection_public = _write_private_key(selection_key_path)
    train_usage = run_root / "train-usage.jsonl"
    selection_usage = run_root / "selection-usage.jsonl"
    selection_audit = run_root / "selection-audit.sealed.jsonl"
    optimizer_usage = run_root / "optimizer-usage.jsonl"
    checkpoint_key_path = run_root / "checkpoint-auth.key"
    _write_secret_key(checkpoint_key_path)
    selection_audit_key_path = run_root / "selection-audit.key"
    _write_secret_key(selection_audit_key_path)
    authorities_path = run_root / "controller-authorities.json"
    _write_json(
        authorities_path,
        {
            "train_public_key": train_public,
            "selection_public_key": selection_public,
            "train_usage_path": str(train_usage),
            "selection_usage_path": str(selection_usage),
            "selection_audit_path": str(selection_audit),
            "optimizer_usage_path": str(optimizer_usage),
            "checkpoint_key_path": str(checkpoint_key_path),
            "selection_audit_key_path": str(selection_audit_key_path),
        },
    )

    artifact_paths = {
        "python_executable": Path(sys.executable).absolute(),
        "train_items": train_source,
        "selection_items": selection_source,
        "train_split_manifest": train_manifest_path,
        "selection_split_manifest": selection_manifest_path,
        "plan": plan_path,
        "train_runner": _TRAIN_RUNNER,
        "selection_runner": _SELECTION_RUNNER,
        "controller_runtime": _CONTROLLER_RUNTIME,
        "searchqa_contract": _SEARCHQA_CONTRACT,
        "experiment_runtime": _EXPERIMENT_RUNTIME,
        "profile": _PROFILE_PATH,
        "initial_skill": _INITIAL_SKILL_PATH,
        "rollout_prompt": _ROLLOUT_PROMPT_PATH,
        "train_private_key": train_key_path,
        "selection_private_key": selection_key_path,
        "checkpoint_key": checkpoint_key_path,
        "selection_audit_key": selection_audit_key_path,
        "controller_authorities": authorities_path,
        "materialization_receipt": materialization.receipt_path,
        "official_train_id_manifest": materialization.train_manifest_path,
        "official_selection_id_manifest": materialization.selection_manifest_path,
    }
    preregistration = {
        "schema_version": "paper-development-preregistration-v2",
        "protocol_id": "paper-faithful-v1",
        "stage": "zero_call_dry_run",
        "authorization": None,
        "benchmark": {
            "id": "searchqa",
            "source_repo": SEARCHQA_DATASET_REPO,
            "source_revision": SEARCHQA_DATASET_REVISION,
            "train_split_id": train_split_id,
            "selection_split_id": selection_split_id,
            "train_count": len(train_items),
            "selection_count": len(selection_items),
            "official_test_id_manifest_sha256": (
                OFFICIAL_SEARCHQA_ID_MANIFEST_SHA256["test"]
            ),
            "test_payload_status": "not_materialized",
        },
        "models": {
            "target_model": "scripted-searchqa-v1",
            "target_reasoning": "none",
            "optimizer_model": "scripted-optimizer-v1",
            "optimizer_reasoning": "none",
        },
        "execution": {
            "seed": profile.split_seed,
            "retry_policy": "semantic-retry-once-v1",
            "target_backend": "scripted",
            "optimizer_backend": "scripted",
            "profile_sha256": canonical_json_sha256(profile.to_dict()),
            "plan_artifact_id": "plan",
        },
        "budgets": {
            "target_calls": 600,
            "target_tokens": 1,
            "optimizer_calls": 300,
            "optimizer_tokens": 1,
            "wall_time_seconds": 3600.0,
            "safety_factor": 1.5,
            "token_policy": "audit_only",
        },
        "stop_conditions": [
            "budget_breach",
            "controller_failure",
            "data_firewall_violation",
            "selection_saturation",
        ],
        "test_access": {"allowed": False, "attempt": 0},
        "artifacts": [
            {
                "artifact_id": artifact_id,
                "path": str(path),
                "sha256": _sha256(path),
            }
            for artifact_id, path in sorted(artifact_paths.items())
        ],
    }
    preregistration_path = run_root / "preregistration.json"
    _write_json(preregistration_path, preregistration)
    load_paper_preregistration(preregistration_path)
    return preregistration_path


def prepare_searchqa_mechanism_smoke(
    *,
    run_dir: str | Path,
    train_path: str | Path,
    selection_path: str | Path,
    target_model: str,
    target_reasoning: str,
    optimizer_model: str,
    optimizer_reasoning: str,
    safety_factor: float,
    zero_cost_receipt_path: str | Path,
    materialization_receipt_path: str | Path,
    mechanism_dry_run_receipt_path: str | Path,
) -> Path:
    """Freeze the first paid stage without executing a model call."""

    coco_config_path = _default_coco_config_path()
    detected_target = detect_coco_model(coco_config_path)
    if detected_target == "configured-default" or target_model != detected_target:
        raise ValueError(
            "target_model must exactly match the readable Coco local default"
        )
    configured_optimizer = os.environ.get("EXTERNAL_LLM_MODEL", "").strip()
    if not configured_optimizer or optimizer_model != configured_optimizer:
        raise ValueError(
            "optimizer_model must exactly match EXTERNAL_LLM_MODEL"
        )
    code_commit, worktree_clean = _git_identity()
    if not worktree_clean or code_commit is None:
        raise ValueError(
            "paid M7 preparation requires a clean Git worktree and readable commit"
        )
    receipt_path = Path(zero_cost_receipt_path).resolve()
    _require_zero_cost_authorization(receipt_path, code_commit=code_commit)
    dry_run_receipt_path = Path(mechanism_dry_run_receipt_path).resolve()
    dry_run_preregistration, dry_run_receipt = _load_mechanism_dry_run_evidence(
        dry_run_receipt_path,
        train_path=Path(train_path).resolve(),
        selection_path=Path(selection_path).resolve(),
        materialization_receipt_path=Path(materialization_receipt_path).resolve(),
    )
    coco_binary_path = resolve_coco_binary()
    preregistration_path = prepare_zero_call_searchqa_experiment(
        run_dir=run_dir,
        train_path=train_path,
        selection_path=selection_path,
        materialization_receipt_path=materialization_receipt_path,
        mechanism_smoke_scope=True,
    )
    prepared_zero = load_paper_preregistration(preregistration_path)
    prepared_plan = PaperEpochPlan.from_mapping(
        json.loads(prepared_zero.artifact("plan").path.read_text(encoding="utf-8"))
    )
    if canonical_json_sha256(prepared_plan.to_dict()) != dry_run_receipt[
        "plan_sha256"
    ]:
        raise ValueError("mechanism dry-run plan does not match paid preparation")
    payload = json.loads(preregistration_path.read_text(encoding="utf-8"))
    payload["stage"] = "mechanism_smoke"
    payload["authorization"] = {
        "local_code_commit": code_commit,
        "zero_cost_receipt_artifact_id": "zero_cost_receipt",
        "mechanism_dry_run_receipt_artifact_id": "mechanism_dry_run_receipt",
        "mechanism_dry_run_preregistration_artifact_id": (
            "mechanism_dry_run_preregistration"
        ),
        "paid_development_authorized": True,
    }
    payload["models"] = {
        "target_model": target_model,
        "target_reasoning": target_reasoning,
        "optimizer_model": optimizer_model,
        "optimizer_reasoning": optimizer_reasoning,
    }
    payload["execution"].update(
        {
            "target_backend": "coco",
            "optimizer_backend": "openai_compatible",
        }
    )
    payload["budgets"] = _derive_mechanism_smoke_budgets(
        dry_run_receipt, safety_factor=safety_factor
    )
    for artifact_id, path in (
        ("zero_cost_receipt", receipt_path),
        ("mechanism_dry_run_receipt", dry_run_receipt_path),
        ("mechanism_dry_run_preregistration", dry_run_preregistration.source_path),
        ("coco_binary", coco_binary_path),
        ("coco_config", coco_config_path),
    ):
        payload["artifacts"].append(
            {
                "artifact_id": artifact_id,
                "path": str(path),
                "sha256": _sha256(path),
            }
        )
    _write_json(preregistration_path, payload)
    load_paper_preregistration(preregistration_path)
    return preregistration_path


def run_searchqa_experiment(preregistration_path: str | Path) -> Path:
    prereg = load_paper_preregistration(preregistration_path)
    payload = prereg.payload
    if prereg.stage != "zero_call_dry_run":
        code_commit, worktree_clean = _git_identity()
        authorization = payload["authorization"]
        if (
            not worktree_clean
            or code_commit is None
            or code_commit != authorization["local_code_commit"]
        ):
            raise ValueError(
                "paid M7 execution requires the preregistered clean Git commit"
            )
    run_root = prereg.source_path.parent
    authorities = json.loads(
        prereg.artifact("controller_authorities").path.read_text(encoding="utf-8")
    )
    _require_fresh_run(run_root, authorities)
    started = time.monotonic()
    deadline = started + float(payload["budgets"]["wall_time_seconds"])
    budget_guard = (
        None
        if prereg.stage == "zero_call_dry_run"
        else PaidBudgetGuard(payload["budgets"], deadline=deadline)
    )
    registry = _build_registry(prereg, authorities, deadline=deadline)
    if prereg.stage == "zero_call_dry_run":
        backend: ScriptedSearchQAOptimizerBackend | OpenAICompatiblePaperOptimizerBackend = (
            ScriptedSearchQAOptimizerBackend()
        )
    else:
        if (
            detect_coco_model(prereg.artifact("coco_config").path)
            != payload["models"]["target_model"]
        ):
            raise ValueError("Coco local default drifted after preregistration")
        backend = OpenAICompatiblePaperOptimizerBackend(
            model_id=payload["models"]["optimizer_model"],
            reasoning_effort=payload["models"]["optimizer_reasoning"],
            budget_guard=budget_guard,
            usage_ledger=Path(authorities["optimizer_usage_path"]),
        )
    controller = PaperOptimizationController(
        optimizer_backend=backend,
        train=TrainController(registry=registry, controller_id="searchqa-train-owner"),
        selection=SelectionController(
            registry=registry, controller_id="searchqa-selection-owner"
        ),
        skill_validator=_searchqa_skill_validator(
            prereg.artifact("rollout_prompt").path.read_text(encoding="utf-8")
        ),
        skill_contract_description=(
            "The rollout system requires the final answer inside "
            "<answer>...</answer> tags. Learned rules must preserve, and must "
            "never negate, that immutable output contract."
        ),
    )
    profile = load_paper_profile()
    plan = PaperEpochPlan.from_mapping(
        json.loads(prereg.artifact("plan").path.read_text(encoding="utf-8"))
    )
    loop = PaperEpochLoop(controller, profile=profile, plan=plan)
    receipt_path = run_root / "receipt.json"
    completed_steps = 0
    initial_state = None
    selection_unsaturated = None
    try:
        initial_state = loop.initialize(
            prereg.artifact("initial_skill").path.read_text(encoding="utf-8")
        )
        selection_unsaturated = 0.0 < initial_state.current_score.value < 1.0
        if not selection_unsaturated:
            wall_time = time.monotonic() - started
            usage = _usage_summary(
                Path(authorities["train_usage_path"]),
                Path(authorities["selection_usage_path"]),
                optimizer_backend=backend,
            )
            event_counts = Counter(event.event_type.value for event in loop.events)
            evidence_artifacts = _persist_run_evidence(
                run_root=run_root,
                loop=loop,
                authorities=authorities,
                optimizer_backend=backend,
            )
            _write_json(
                receipt_path,
                {
                    "schema_version": "paper-searchqa-development-stop-receipt-v2",
                    "status": "stopped",
                    "stage": prereg.stage,
                    "stop_reason": "selection_saturation",
                    "preregistration_sha256": _sha256(prereg.source_path),
                    "profile_sha256": canonical_json_sha256(profile.to_dict()),
                    "plan_sha256": canonical_json_sha256(plan.to_dict()),
                    "completed_epochs": 0,
                    "completed_steps": 0,
                    "initial_selection_score": initial_state.current_score.value,
                    "best_selection_score": initial_state.best_score.value,
                    "selection_unsaturated": False,
                    "full_call_graph_complete": False,
                    "event_counts": dict(sorted(event_counts.items())),
                    "usage": usage,
                    "wall_time_seconds": wall_time,
                    "test_access": dict(payload["test_access"]),
                    "test_payload_status": payload["benchmark"][
                        "test_payload_status"
                    ],
                    "claim_class": None,
                    "evidence_level": None,
                    "evidence_artifacts": evidence_artifacts,
                },
            )
            prereg.verify()
            raise RuntimeError("selection_saturation stop condition triggered")
        if budget_guard is not None:
            budget_guard.check()
        while True:
            while loop.state.step < plan.steps_per_epoch:
                loop.run_step(train_evidence=loop.collect_train_evidence())
                completed_steps += 1
                if budget_guard is not None:
                    budget_guard.check()
            longitudinal = (
                loop.collect_longitudinal_evidence()
                if loop.state.epoch >= profile.slow_update.start_epoch
                else None
            )
            completion = loop.finish_epoch(longitudinal_evidence=longitudinal)
            if budget_guard is not None:
                budget_guard.check()
            if completion.run_completed:
                break
        wall_time = time.monotonic() - started
        usage = _usage_summary(
            Path(authorities["train_usage_path"]),
            Path(authorities["selection_usage_path"]),
            optimizer_backend=backend,
        )
        event_counts = Counter(event.event_type.value for event in loop.events)
        required_events = {
            "run_started",
            "failure_reflected",
            "success_reflected",
            "analyst_refined",
            "merge_failure",
            "merge_success",
            "merge_final_failure_prioritized",
            "rank_top_l",
            "patch_applied",
            "selection_scored",
            "slow_update_skipped",
            "slow_update_proposed",
            "meta_update_skipped",
            "meta_update_completed",
            "run_completed",
        }
        full_graph = all(event_counts[name] > 0 for name in required_events) and (
            event_counts["candidate_accepted"]
            + event_counts["candidate_rejected"]
            > 0
        )
        _require_within_budgets(payload["budgets"], usage, wall_time)
        if not full_graph:
            raise RuntimeError("zero-call dry-run did not execute the full call graph")
    except Exception as error:
        if not receipt_path.exists():
            wall_time = time.monotonic() - started
            usage = _usage_summary(
                Path(authorities["train_usage_path"]),
                Path(authorities["selection_usage_path"]),
                optimizer_backend=backend,
            )
            event_counts = Counter(event.event_type.value for event in loop.events)
            error_message = str(error)
            if len(error_message) > 2000:
                error_message = (
                    error_message[:1000]
                    + "...[truncated]..."
                    + error_message[-983:]
                )
            state = loop.state if initial_state is not None else None
            error_code = getattr(error, "code", "unclassified_execution_error")
            if (
                type(error_code) is not str
                or re.fullmatch(r"[a-z][a-z0-9_]+", error_code) is None
            ):
                error_code = "unclassified_execution_error"
            error_call_id = None
            if isinstance(error, OptimizerProviderError):
                failure_records = backend.failure_records
                if (
                    not failure_records
                    or failure_records[-1]["failure_code"] != error_code
                ):
                    raise RuntimeError(
                        "optimizer provider failure lacks matching evidence"
                    ) from error
                error_call_id = failure_records[-1]["request"]["call_id"]
            evidence_artifacts = _persist_run_evidence(
                run_root=run_root,
                loop=loop,
                authorities=authorities,
                optimizer_backend=backend,
            )
            if (
                isinstance(error, OptimizerProviderError)
                and "optimizer_failures" not in evidence_artifacts
            ):
                raise RuntimeError(
                    "optimizer provider failure artifact was not sealed"
                ) from error
            _write_json(
                receipt_path,
                {
                    "schema_version": "paper-searchqa-development-stop-receipt-v2",
                    "status": "stopped",
                    "stage": prereg.stage,
                    "stop_reason": (
                        "budget_breach"
                        if error_message.startswith("budget_breach")
                        else "execution_error"
                    ),
                    "error_type": type(error).__name__,
                    "error_code": error_code,
                    "error_call_id": error_call_id,
                    "error_message": error_message,
                    "preregistration_sha256": _sha256(prereg.source_path),
                    "profile_sha256": canonical_json_sha256(profile.to_dict()),
                    "plan_sha256": canonical_json_sha256(plan.to_dict()),
                    "completed_epochs": state.epoch if state is not None else 0,
                    "completed_steps": completed_steps,
                    "initial_selection_score": (
                        initial_state.current_score.value
                        if initial_state is not None
                        else None
                    ),
                    "best_selection_score": (
                        state.best_score.value if state is not None else None
                    ),
                    "selection_unsaturated": selection_unsaturated,
                    "full_call_graph_complete": False,
                    "event_counts": dict(sorted(event_counts.items())),
                    "usage": usage,
                    "wall_time_seconds": wall_time,
                    "test_access": dict(payload["test_access"]),
                    "test_payload_status": payload["benchmark"][
                        "test_payload_status"
                    ],
                    "claim_class": None,
                    "evidence_level": None,
                    "evidence_artifacts": evidence_artifacts,
                },
            )
            prereg.verify()
        raise
    evidence_artifacts = _persist_run_evidence(
        run_root=run_root,
        loop=loop,
        authorities=authorities,
        optimizer_backend=backend,
    )
    receipt = {
        "schema_version": "paper-searchqa-development-receipt-v2",
        "status": "completed",
        "stage": prereg.stage,
        "preregistration_sha256": _sha256(prereg.source_path),
        "profile_sha256": canonical_json_sha256(profile.to_dict()),
        "plan_sha256": canonical_json_sha256(plan.to_dict()),
        "completed_epochs": loop.state.epoch,
        "completed_steps": completed_steps,
        "initial_selection_score": initial_state.current_score.value,
        "best_selection_score": loop.state.best_score.value,
        "selection_unsaturated": selection_unsaturated,
        "full_call_graph_complete": full_graph,
        "event_counts": dict(sorted(event_counts.items())),
        "usage": usage,
        "wall_time_seconds": wall_time,
        "test_access": dict(payload["test_access"]),
        "test_payload_status": payload["benchmark"]["test_payload_status"],
        "claim_class": "mechanism_test",
        "evidence_level": None,
        "evidence_artifacts": evidence_artifacts,
    }
    _write_json(receipt_path, receipt)
    prereg.verify()
    return receipt_path


def _build_registry(
    prereg, authorities: Mapping[str, Any], *, deadline: float
) -> ControllerRegistry:
    common_artifact_ids = [
        "controller_runtime",
        "searchqa_contract",
        "profile",
        "initial_skill",
        "rollout_prompt",
    ]
    if prereg.stage != "zero_call_dry_run":
        common_artifact_ids.extend(("coco_binary", "coco_config"))
    execution = prereg.payload["execution"]
    models = prereg.payload["models"]
    executable_path = prereg.artifact("python_executable").path
    if Path(sys.executable).absolute() != executable_path:
        raise ValueError("Python executable drifted after preregistration")
    budget_argv = (
        "--target-call-cap",
        str(prereg.payload["budgets"]["target_calls"]),
        "--target-token-cap",
        str(prereg.payload["budgets"]["target_tokens"]),
        "--deadline-monotonic",
        repr(deadline),
    )
    coco_argv = (
        ("--coco-binary", str(prereg.artifact("coco_binary").path))
        if prereg.stage != "zero_call_dry_run"
        else ()
    )
    train_argv = (
        str(executable_path),
        str(prereg.artifact("train_runner").path),
        "--controller-id",
        "searchqa-train-owner",
        "--data",
        str(prereg.artifact("train_items").path),
        "--private-key",
        str(prereg.artifact("train_private_key").path),
        "--backend",
        execution["target_backend"],
        "--target-model",
        models["target_model"],
        "--target-reasoning",
        models["target_reasoning"],
        "--usage-ledger",
        str(authorities["train_usage_path"]),
        "--peer-usage-ledger",
        str(authorities["selection_usage_path"]),
        "--rollout-prompt",
        str(prereg.artifact("rollout_prompt").path),
        "--plan",
        str(prereg.artifact("plan").path),
        *budget_argv,
        *coco_argv,
    )
    selection_argv = (
        str(executable_path),
        str(prereg.artifact("selection_runner").path),
        "--controller-id",
        "searchqa-selection-owner",
        "--data",
        str(prereg.artifact("selection_items").path),
        "--private-key",
        str(prereg.artifact("selection_private_key").path),
        "--backend",
        execution["target_backend"],
        "--target-model",
        models["target_model"],
        "--target-reasoning",
        models["target_reasoning"],
        "--usage-ledger",
        str(authorities["selection_usage_path"]),
        "--peer-usage-ledger",
        str(authorities["train_usage_path"]),
        "--rollout-prompt",
        str(prereg.artifact("rollout_prompt").path),
        "--selection-audit",
        str(authorities["selection_audit_path"]),
        "--selection-audit-key",
        str(authorities["selection_audit_key_path"]),
        *budget_argv,
        *coco_argv,
    )
    executable = ControllerArtifact(
        "executable", str(executable_path), prereg.artifact("python_executable").sha256
    )
    train_artifacts = (
        executable,
        _controller_artifact("runner", prereg.artifact("train_runner")),
        _controller_artifact(
            "split_manifest", prereg.artifact("train_split_manifest")
        ),
        _controller_artifact("data", prereg.artifact("train_items")),
        _controller_artifact("plan", prereg.artifact("plan")),
        _controller_artifact("private_key", prereg.artifact("train_private_key")),
        *(
            _controller_artifact(artifact_id, prereg.artifact(artifact_id))
            for artifact_id in common_artifact_ids
        ),
    )
    selection_artifacts = (
        executable,
        _controller_artifact("runner", prereg.artifact("selection_runner")),
        _controller_artifact(
            "split_manifest", prereg.artifact("selection_split_manifest")
        ),
        _controller_artifact("data", prereg.artifact("selection_items")),
        _controller_artifact(
            "private_key", prereg.artifact("selection_private_key")
        ),
        _controller_artifact(
            "selection_audit_key", prereg.artifact("selection_audit_key")
        ),
        *(
            _controller_artifact(artifact_id, prereg.artifact(artifact_id))
            for artifact_id in common_artifact_ids
        ),
    )
    benchmark = prereg.payload["benchmark"]
    train_timeout = _controller_timeout_seconds(
        stage=prereg.stage,
        task_count=benchmark["train_count"],
    )
    selection_timeout = _controller_timeout_seconds(
        stage=prereg.stage,
        task_count=benchmark["selection_count"],
    )
    return ControllerRegistry(
        registrations=(
            ControllerRegistration(
                controller_id="searchqa-train-owner",
                role=ControllerRole.TRAIN,
                split_id=benchmark["train_split_id"],
                argv=train_argv,
                launch_artifact_ids=("executable", "runner"),
                response_public_key=authorities["train_public_key"],
                artifacts=train_artifacts,
                timeout_seconds=train_timeout,
                max_output_chars=10_000_000,
            ),
            ControllerRegistration(
                controller_id="searchqa-selection-owner",
                role=ControllerRole.SELECTION,
                split_id=benchmark["selection_split_id"],
                argv=selection_argv,
                launch_artifact_ids=("executable", "runner"),
                response_public_key=authorities["selection_public_key"],
                artifacts=selection_artifacts,
                timeout_seconds=selection_timeout,
                max_output_chars=1_000_000,
            ),
        )
    )


def _controller_timeout_seconds(*, stage: str, task_count: int) -> float:
    if stage == "zero_call_dry_run":
        return 300.0
    prompt_waves = math.ceil(task_count / COCO_ACP_WORKERS)
    startup_seconds = COCO_ACP_WORKERS * ACP_STARTUP_TIMEOUT_SECONDS
    return startup_seconds + prompt_waves * 120.0 + 30.0


def _controller_artifact(artifact_id: str, preregistered) -> ControllerArtifact:
    return ControllerArtifact(
        artifact_id=artifact_id,
        path=str(preregistered.path),
        sha256=preregistered.sha256,
    )


def _split_manifest(
    *,
    split_id: str,
    role: str,
    items_path: Path,
    count: int,
) -> dict[str, Any]:
    return {
        "schema_version": "searchqa-development-split-v1",
        "split_id": split_id,
        "role": role,
        "item_count": count,
        "items_sha256": _sha256(items_path),
        "source_repo": SEARCHQA_DATASET_REPO,
        "source_revision": SEARCHQA_DATASET_REVISION,
        "test_payload_access": False,
    }


def _write_private_key(path: Path) -> str:
    private_key = Ed25519PrivateKey.generate()
    private_bytes = private_key.private_bytes(
        serialization.Encoding.Raw,
        serialization.PrivateFormat.Raw,
        serialization.NoEncryption(),
    )
    path.write_text(private_bytes.hex() + "\n", encoding="utf-8")
    os.chmod(path, 0o600)
    return private_key.public_key().public_bytes(
        serialization.Encoding.Raw,
        serialization.PublicFormat.Raw,
    ).hex()


def _write_secret_key(path: Path) -> None:
    path.write_text(os.urandom(32).hex() + "\n", encoding="utf-8")
    os.chmod(path, 0o600)


def _searchqa_skill_validator(
    rollout_prompt: str,
) -> Callable[[str], None]:
    def validate(skill_text: str) -> None:
        try:
            require_searchqa_skill_rollout_compatibility(
                skill_text,
                rollout_prompt,
            )
        except SearchQAContractViolation as error:
            raise SkillContractViolation(
                "searchqa_answer_wrapper_conflict",
                str(error),
            ) from error

    return validate


def _usage_summary(
    train_path: Path,
    selection_path: Path,
    *,
    optimizer_backend: ScriptedSearchQAOptimizerBackend
    | OpenAICompatiblePaperOptimizerBackend,
) -> dict[str, int]:
    records = []
    for path in (train_path, selection_path):
        if path.exists():
            records.extend(
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            )
    if isinstance(optimizer_backend, OpenAICompatiblePaperOptimizerBackend):
        external_optimizer_calls = optimizer_backend.external_calls
        optimizer_tokens = optimizer_backend.actual_tokens
        estimated_optimizer_tokens = optimizer_backend.estimated_tokens
    else:
        external_optimizer_calls = 0
        optimizer_tokens = sum(
            int(response.usage.get("total_tokens", 0))
            for response in optimizer_backend.responses
        )
        estimated_optimizer_tokens = sum(
            _estimate_tokens(request.system_prompt + request.prompt)
            + _estimate_tokens(
                json.dumps(response.payload, ensure_ascii=False, sort_keys=True)
            )
            for request, response in zip(
                optimizer_backend.requests, optimizer_backend.responses
            )
        )
    return {
        "logical_target_calls": len(records),
        "external_target_calls": sum(bool(item["external_call"]) for item in records),
        "target_tokens": sum(int(item["total_tokens"]) for item in records),
        "estimated_target_tokens": sum(
            int(item["estimated_prompt_tokens"])
            + int(item["estimated_completion_tokens"])
            for item in records
        ),
        "logical_optimizer_calls": len(optimizer_backend.requests),
        "external_optimizer_calls": external_optimizer_calls,
        "optimizer_tokens": optimizer_tokens,
        "estimated_optimizer_tokens": estimated_optimizer_tokens,
    }


def _require_within_budgets(
    budgets: Mapping[str, Any],
    usage: Mapping[str, int],
    wall_time: float,
) -> None:
    checks = {
        "target_calls": usage["logical_target_calls"],
        "optimizer_calls": usage["logical_optimizer_calls"],
        "wall_time_seconds": wall_time,
    }
    breached = [name for name, value in checks.items() if value > budgets[name]]
    if breached:
        raise RuntimeError("budget_breach stop condition triggered: " + ", ".join(breached))


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _persist_run_evidence(
    *,
    run_root: Path,
    loop: PaperEpochLoop,
    authorities: Mapping[str, Any],
    optimizer_backend: ScriptedSearchQAOptimizerBackend
    | OpenAICompatiblePaperOptimizerBackend,
) -> dict[str, dict[str, Any]]:
    """Seal complete runtime evidence before a terminal receipt is written."""

    evidence_paths: dict[str, Path] = {}
    lineage = loop.artifact_lineage
    lineage_path = run_root / "artifact-lineage.json"
    _write_json(
        lineage_path,
        {
            "schema_version": "paper-run-artifact-lineage-v1",
            "lineage_sha256": lineage.sha256,
            "records": lineage.to_checkpoint_list(),
        },
    )
    evidence_paths["artifact_lineage"] = lineage_path

    events_path = run_root / "events.json"
    _write_json(
        events_path,
        {
            "schema_version": "paper-run-events-v1",
            "events": [event.to_dict() for event in loop.events],
        },
    )
    evidence_paths["events"] = events_path

    skills_path = run_root / "candidate-skills.json"
    _write_json(
        skills_path,
        {
            "schema_version": "paper-run-candidate-skills-v1",
            "skills": [
                {
                    "artifact_id": record.artifact_id,
                    "parent_ids": list(record.parent_ids),
                    **dict(record.payload),
                }
                for record in lineage.records_of_kind(PaperArtifactKind.SKILL)
            ],
        },
    )
    evidence_paths["candidate_skills"] = skills_path

    optimizer_requests = {
        record.artifact_id: record
        for record in lineage.records_of_kind(PaperArtifactKind.OPTIMIZER_REQUEST)
    }
    exchanges = []
    for response in lineage.records_of_kind(PaperArtifactKind.OPTIMIZER_RESPONSE):
        request_ids = [
            parent_id
            for parent_id in response.parent_ids
            if parent_id in optimizer_requests
        ]
        if len(request_ids) != 1:
            raise RuntimeError("optimizer response lineage lacks one request parent")
        request = optimizer_requests[request_ids[0]]
        exchanges.append(
            {
                "request_artifact_id": request.artifact_id,
                "request": dict(request.payload),
                "response_artifact_id": response.artifact_id,
                "response": dict(response.payload),
            }
        )
    exchanges_path = run_root / "optimizer-exchanges.json"
    _write_json(
        exchanges_path,
        {
            "schema_version": "paper-run-optimizer-exchanges-v1",
            "exchanges": exchanges,
        },
    )
    evidence_paths["optimizer_exchanges"] = exchanges_path

    failure_records = optimizer_backend.failure_records
    if failure_records:
        _verify_optimizer_failure_records(failure_records)
        failures_path = run_root / "optimizer-failures.json"
        _write_json(
            failures_path,
            {
                "schema_version": "paper-run-optimizer-failures-v1",
                "failures": [dict(record) for record in failure_records],
            },
        )
        evidence_paths["optimizer_failures"] = failures_path

    try:
        state = loop.state
    except ValueError:
        state = None
    if state is not None:
        final_state_path = run_root / "final-state.json"
        _write_json(
            final_state_path,
            {
                "schema_version": "paper-run-final-state-v1",
                "epoch": state.epoch,
                "step": state.step,
                "current_skill": state.current_skill,
                "current_skill_sha256": hashlib.sha256(
                    state.current_skill.encode("utf-8")
                ).hexdigest(),
                "current_score": state.current_score.value,
                "best_skill": state.best_skill,
                "best_skill_sha256": hashlib.sha256(
                    state.best_skill.encode("utf-8")
                ).hexdigest(),
                "best_score": state.best_score.value,
                "meta_skill": state.meta_skill,
            },
        )
        evidence_paths["final_state"] = final_state_path
        final_skill_path = run_root / "final-skill.md"
        final_skill_path.write_text(state.current_skill, encoding="utf-8")
        evidence_paths["final_skill"] = final_skill_path
        best_skill_path = run_root / "best-skill.md"
        best_skill_path.write_text(state.best_skill, encoding="utf-8")
        evidence_paths["best_skill"] = best_skill_path

        checkpoint_key_path = Path(authorities["checkpoint_key_path"])
        checkpoint_key = bytes.fromhex(
            checkpoint_key_path.read_text(encoding="utf-8").strip()
        )
        authenticator = CheckpointAuthenticator(
            key_id="searchqa-runtime-" + _sha256(checkpoint_key_path)[:16],
            secret_key=checkpoint_key,
        )
        checkpoint_path = run_root / "checkpoint.json"
        _write_json(checkpoint_path, loop.checkpoint(authenticator).to_dict())
        evidence_paths["checkpoint"] = checkpoint_path

    sealed_audit = Path(authorities["selection_audit_path"])
    selection_audit = run_root / "selection-audit.jsonl"
    if sealed_audit.exists():
        records = unseal_searchqa_selection_audit(
            sealed_audit,
            key_path=Path(authorities["selection_audit_key_path"]),
        )
        selection_audit.write_text(
            "".join(
                json.dumps(
                    record,
                    ensure_ascii=False,
                    allow_nan=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
                for record in records
            ),
            encoding="utf-8",
        )
        sealed_audit.unlink()
    elif not selection_audit.exists():
        selection_audit.write_text("", encoding="utf-8")
    _verify_selection_audit(selection_audit)
    evidence_paths["selection_audit"] = selection_audit

    for name, raw_path in (
        ("train_usage", authorities["train_usage_path"]),
        ("selection_usage", authorities["selection_usage_path"]),
        ("optimizer_usage", authorities["optimizer_usage_path"]),
    ):
        path = Path(raw_path)
        if path.exists():
            evidence_paths[name] = path

    return {
        name: {
            "path": str(path.relative_to(run_root)),
            "sha256": _sha256(path),
            "size_bytes": path.stat().st_size,
        }
        for name, path in sorted(evidence_paths.items())
    }


def _verify_optimizer_failure_records(
    records: tuple[Mapping[str, Any], ...],
) -> None:
    expected_record_keys = {
        "schema_version",
        "failure_code",
        "error_type",
        "error_message",
        "model_id",
        "request",
        "provider_response",
        "usage",
    }
    expected_request_keys = {
        "call_id",
        "stage",
        "prompt",
        "response_schema",
        "system_prompt",
        "metadata",
    }
    expected_usage_keys = {
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "tokens_are_actual",
    }
    for record in records:
        if type(record) is not dict or set(record) != expected_record_keys:
            raise RuntimeError("optimizer failure evidence fields are invalid")
        request = record["request"]
        diagnostic = record["provider_response"]
        usage = record["usage"]
        if (
            record["schema_version"] != "paper-optimizer-failure-v1"
            or re.fullmatch(
                r"optimizer_[a-z0-9_]+", str(record["failure_code"])
            )
            is None
            or record["error_type"] != "OptimizerProviderError"
            or type(record["error_message"]) is not str
            or not record["error_message"]
            or type(record["model_id"]) is not str
            or not record["model_id"]
            or type(request) is not dict
            or set(request) != expected_request_keys
            or type(request["call_id"]) is not str
            or not request["call_id"]
            or type(request["stage"]) is not str
            or not request["stage"]
            or type(diagnostic) is not dict
            or diagnostic.get("schema_version")
            != "optimizer-provider-response-diagnostic-v1"
            or type(diagnostic.get("body_size_bytes")) is not int
            or diagnostic["body_size_bytes"] < 0
            or re.fullmatch(
                r"[0-9a-f]{64}", str(diagnostic.get("body_sha256"))
            )
            is None
            or any(
                key in diagnostic
                for key in ("content", "raw_body", "api_key", "authorization")
            )
            or type(usage) is not dict
            or set(usage) != expected_usage_keys
            or any(
                type(usage[name]) is not int or usage[name] < 0
                for name in ("prompt_tokens", "completion_tokens", "total_tokens")
            )
            or type(usage["tokens_are_actual"]) is not bool
        ):
            raise RuntimeError("optimizer failure evidence values are invalid")


def _verify_selection_audit(path: Path) -> None:
    records = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    previous_hash = None
    for sequence, record in enumerate(records, 1):
        if type(record) is not dict or set(record) != {
            "schema_version",
            "sequence",
            "previous_record_sha256",
            "request_sha256",
            "skill_sha256",
            "score",
            "items",
            "record_sha256",
        }:
            raise RuntimeError("selection audit fields do not match the contract")
        claimed_hash = record["record_sha256"]
        unsigned = dict(record)
        del unsigned["record_sha256"]
        if (
            record["schema_version"] != "searchqa-selection-audit-v1"
            or record["sequence"] != sequence
            or record["previous_record_sha256"] != previous_hash
            or re.fullmatch(r"[0-9a-f]{64}", str(record["request_sha256"]))
            is None
            or re.fullmatch(r"[0-9a-f]{64}", str(record["skill_sha256"]))
            is None
            or re.fullmatch(r"[0-9a-f]{64}", str(claimed_hash)) is None
            or canonical_json_sha256(unsigned) != claimed_hash
            or type(record["items"]) is not list
            or not record["items"]
        ):
            raise RuntimeError("selection audit hash chain is invalid")
        task_ids = []
        for item in record["items"]:
            if type(item) is not dict or set(item) != {
                "task_id",
                "exact_match",
                "predicted_answer",
                "response_sha256",
            }:
                raise RuntimeError("selection audit item fields are invalid")
            if (
                type(item["task_id"]) is not str
                or not item["task_id"]
                or type(item["exact_match"]) not in {int, float}
                or item["exact_match"] not in {0.0, 1.0}
                or type(item["predicted_answer"]) is not str
                or re.fullmatch(
                    r"[0-9a-f]{64}",
                    str(item["response_sha256"]),
                )
                is None
            ):
                raise RuntimeError("selection audit item value is invalid")
            task_ids.append(item["task_id"])
        expected_score = sum(
            float(item["exact_match"]) for item in record["items"]
        ) / len(record["items"])
        if (
            len(task_ids) != len(set(task_ids))
            or type(record["score"]) not in {int, float}
            or not math.isclose(float(record["score"]), expected_score)
        ):
            raise RuntimeError("selection audit aggregate is invalid")
        previous_hash = claimed_hash


def _verify_receipt_evidence_artifacts(
    payload: object,
    *,
    root: Path,
) -> None:
    required = {
        "artifact_lineage",
        "candidate_skills",
        "checkpoint",
        "events",
        "final_skill",
        "final_state",
        "optimizer_exchanges",
        "selection_audit",
    }
    if type(payload) is not dict or not required.issubset(payload):
        raise ValueError("receipt is missing required run evidence artifacts")
    resolved_root = root.resolve()
    for name, artifact in payload.items():
        if type(name) is not str or type(artifact) is not dict or set(artifact) != {
            "path",
            "sha256",
            "size_bytes",
        }:
            raise ValueError("receipt evidence artifact fields are invalid")
        raw_path = artifact["path"]
        if type(raw_path) is not str or not raw_path or Path(raw_path).is_absolute():
            raise ValueError("receipt evidence artifact path must be relative")
        path = (resolved_root / raw_path).resolve()
        if path.parent != resolved_root or not path.is_file():
            raise ValueError("receipt evidence artifact escaped or is missing")
        if (
            _sha256(path) != artifact["sha256"]
            or path.stat().st_size != artifact["size_bytes"]
        ):
            raise ValueError("receipt evidence artifact hash or size drifted")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _provider_usage(payload: object, name: str) -> int:
    if type(payload) is not dict or type(payload.get(name)) is not int:
        raise RuntimeError(f"external optimizer usage is missing {name}")
    value = payload[name]
    if value < 0:
        raise RuntimeError(f"external optimizer usage {name} cannot be negative")
    return value


def _optional_provider_usage(payload: object) -> tuple[dict[str, int], bool]:
    try:
        usage = {
            "prompt_tokens": _provider_usage(payload, "prompt_tokens"),
            "completion_tokens": _provider_usage(payload, "completion_tokens"),
            "total_tokens": _provider_usage(payload, "total_tokens"),
        }
    except RuntimeError:
        return {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }, False
    return usage, True


def _estimate_tokens(value: str) -> int:
    return max(1, (len(value) + 3) // 4)


def detect_coco_model(config_path: str | Path | None = None) -> str:
    path = (
        Path(config_path).expanduser()
        if config_path is not None
        else Path.home() / ".trae" / "traecli.yaml"
    )
    if not path.is_file():
        return "configured-default"
    model_indent: int | None = None
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        indent = len(line) - len(line.lstrip())
        if stripped == "model:":
            model_indent = indent
            continue
        if model_indent is None:
            continue
        if indent <= model_indent:
            break
        match = re.fullmatch(r"name:\s*['\"]?([^'\"]+?)['\"]?", stripped)
        if match:
            return match.group(1).strip()
    return "configured-default"


def resolve_coco_binary() -> Path:
    configured = os.environ.get("COCO_AGENT_BIN", "").strip()
    if configured:
        candidate = Path(configured).expanduser()
        if not candidate.is_absolute():
            resolved = shutil.which(configured)
            candidate = Path(resolved) if resolved else candidate
    else:
        preferred = Path("/Users/bytedance/.local/bin/coco")
        resolved = str(preferred) if preferred.is_file() else shutil.which("coco")
        candidate = Path(resolved) if resolved else preferred
    candidate = candidate.resolve()
    if not candidate.is_file() or not os.access(candidate, os.X_OK):
        raise ValueError("paid M7 preparation requires an executable Coco binary")
    return candidate


def _default_coco_config_path() -> Path:
    path = (Path.home() / ".trae" / "traecli.yaml").resolve()
    if not path.is_file():
        raise ValueError("paid M7 preparation requires the readable Coco default config")
    return path


def _require_fresh_run(run_root: Path, authorities: Mapping[str, Any]) -> None:
    paths = (
        run_root / "receipt.json",
        run_root / "artifact-lineage.json",
        run_root / "events.json",
        run_root / "candidate-skills.json",
        run_root / "optimizer-exchanges.json",
        run_root / "optimizer-failures.json",
        run_root / "final-state.json",
        run_root / "final-skill.md",
        run_root / "best-skill.md",
        run_root / "checkpoint.json",
        run_root / "selection-audit.jsonl",
        Path(authorities["train_usage_path"]),
        Path(authorities["selection_usage_path"]),
        Path(authorities["selection_audit_path"]),
        Path(authorities["optimizer_usage_path"]),
    )
    if any(path.exists() for path in paths):
        raise ValueError(
            "SearchQA preregistration is single-use; existing receipt or usage blocks rerun"
        )


def _derive_mechanism_smoke_budgets(
    receipt: Mapping[str, Any], *, safety_factor: float
) -> dict[str, Any]:
    if (
        isinstance(safety_factor, bool)
        or not isinstance(safety_factor, (int, float))
        or not math.isfinite(float(safety_factor))
        or not 1.25 <= float(safety_factor) <= 1.5
    ):
        raise ValueError("mechanism smoke safety factor must be between 1.25 and 1.5")
    usage = receipt["usage"]
    return {
        "target_calls": math.ceil(usage["logical_target_calls"] * safety_factor),
        "target_tokens": math.ceil(
            usage["estimated_target_tokens"] * safety_factor
        ),
        "optimizer_calls": math.ceil(
            usage["logical_optimizer_calls"] * safety_factor
        ),
        "optimizer_tokens": math.ceil(
            usage["estimated_optimizer_tokens"] * safety_factor
        ),
        "wall_time_seconds": _MECHANISM_SMOKE_WALL_TIME_SECONDS,
        "safety_factor": float(safety_factor),
        "token_policy": "audit_only",
    }


def _load_mechanism_dry_run_evidence(
    receipt_path: Path,
    *,
    train_path: Path,
    selection_path: Path,
    materialization_receipt_path: Path,
):
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("mechanism dry-run receipt must be readable JSON") from error
    expected_keys = {
        "schema_version",
        "status",
        "stage",
        "preregistration_sha256",
        "profile_sha256",
        "plan_sha256",
        "completed_epochs",
        "completed_steps",
        "initial_selection_score",
        "best_selection_score",
        "selection_unsaturated",
        "full_call_graph_complete",
        "event_counts",
        "usage",
        "wall_time_seconds",
        "test_access",
        "test_payload_status",
        "claim_class",
        "evidence_level",
        "evidence_artifacts",
    }
    if type(receipt) is not dict or set(receipt) != expected_keys:
        raise ValueError("mechanism dry-run receipt fields do not match the contract")
    usage_keys = {
        "logical_target_calls",
        "external_target_calls",
        "target_tokens",
        "estimated_target_tokens",
        "logical_optimizer_calls",
        "external_optimizer_calls",
        "optimizer_tokens",
        "estimated_optimizer_tokens",
    }
    usage = receipt["usage"]
    if (
        receipt["schema_version"] != "paper-searchqa-development-receipt-v2"
        or receipt["status"] != "completed"
        or receipt["stage"] != "zero_call_dry_run"
        or receipt["completed_epochs"] != 2
        or receipt["completed_steps"] != 2
        or receipt["selection_unsaturated"] is not True
        or receipt["full_call_graph_complete"] is not True
        or receipt["test_access"] != {"allowed": False, "attempt": 0}
        or receipt["test_payload_status"] != "not_materialized"
        or receipt["claim_class"] != "mechanism_test"
        or receipt["evidence_level"] is not None
        or type(usage) is not dict
        or set(usage) != usage_keys
        or usage["external_target_calls"] != 0
        or usage["external_optimizer_calls"] != 0
        or usage["target_tokens"] != 0
        or usage["optimizer_tokens"] != 0
        or any(
            type(usage[name]) is not int or usage[name] < 1
            for name in (
                "logical_target_calls",
                "estimated_target_tokens",
                "logical_optimizer_calls",
                "estimated_optimizer_tokens",
            )
        )
    ):
        raise ValueError("mechanism dry-run receipt is not eligible for paid caps")
    _verify_receipt_evidence_artifacts(
        receipt["evidence_artifacts"],
        root=receipt_path.parent,
    )
    dry_prereg_path = receipt_path.parent / "preregistration.json"
    dry_prereg = load_paper_preregistration(dry_prereg_path)
    checks = {
        "stage": dry_prereg.stage == "zero_call_dry_run",
        "preregistration": (
            _sha256(dry_prereg.source_path) == receipt["preregistration_sha256"]
        ),
        "train": dry_prereg.artifact("train_items").path == train_path,
        "selection": (
            dry_prereg.artifact("selection_items").path == selection_path
        ),
        "materialization": (
            dry_prereg.artifact("materialization_receipt").path
            == materialization_receipt_path
        ),
        "plan": canonical_json_sha256(
            PaperEpochPlan.from_mapping(
                json.loads(dry_prereg.artifact("plan").path.read_text(encoding="utf-8"))
            ).to_dict()
        )
        == receipt["plan_sha256"],
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(
            "mechanism dry-run artifacts do not match paid inputs: "
            + ", ".join(failed)
        )
    return dry_prereg, receipt


def _git_identity() -> tuple[str | None, bool]:
    commit = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=_PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    status = subprocess.run(
        ("git", "status", "--porcelain", "--untracked-files=normal"),
        cwd=_PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    if commit.returncode or status.returncode:
        return None, False
    code_commit = commit.stdout.strip()
    if re.fullmatch(r"[0-9a-f]{40}", code_commit) is None:
        return None, False
    return code_commit, not status.stdout.strip()


def _require_zero_cost_authorization(path: Path, *, code_commit: str) -> None:
    try:
        receipt = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("zero-cost receipt must be readable JSON") from error
    expected_keys = {
        "schema_version",
        "status",
        "external_calls",
        "network_guard_active",
        "paid_experiment_executed",
        "paid_development_authorized",
        "code_commit",
        "worktree_clean",
        "prompt_count",
        "prompt_snapshot_sha256",
        "source_lock_sha256",
        "golden_trace_sha256",
        "test_targets",
        "violations",
    }
    if (
        type(receipt) is not dict
        or set(receipt) != expected_keys
        or receipt["schema_version"] != "paper-zero-cost-gate-v1"
        or receipt["status"] != "passed"
        or receipt["external_calls"] != 0
        or receipt["network_guard_active"] is not True
        or receipt["paid_experiment_executed"] is not False
        or receipt["paid_development_authorized"] is not True
        or receipt["worktree_clean"] is not True
        or receipt["code_commit"] != code_commit
        or receipt["prompt_count"] != 18
        or receipt["test_targets"] != ["tests/conformance", "tests/provenance"]
        or receipt["violations"] != []
        or any(
            type(receipt[name]) is not str
            or re.fullmatch(r"[0-9a-f]{64}", receipt[name]) is None
            for name in (
                "prompt_snapshot_sha256",
                "source_lock_sha256",
                "golden_trace_sha256",
            )
        )
    ):
        raise ValueError(
            "zero-cost receipt does not authorize the current clean Git commit"
        )
    expected_hashes = {
        "prompt_snapshot_sha256": canonical_json_sha256(
            json.loads(
                (_PROJECT_ROOT / "docs/papers/prompt-snapshot-v1.json").read_text(
                    encoding="utf-8"
                )
            )
        ),
        "source_lock_sha256": canonical_json_sha256(
            json.loads(
                (_PROJECT_ROOT / "docs/papers/source-lock.json").read_text(
                    encoding="utf-8"
                )
            )
        ),
        "golden_trace_sha256": _sha256(
            _PROJECT_ROOT / "tests/conformance/golden/algorithm1-fast-loop-v1.json"
        ),
    }
    if any(receipt[name] != value for name, value in expected_hashes.items()):
        raise ValueError("zero-cost receipt lock hashes drifted from the current commit")
