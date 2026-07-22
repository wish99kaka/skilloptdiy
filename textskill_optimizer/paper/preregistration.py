"""Fail-closed preregistration contract for M7 development experiments."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .provenance import canonical_json_sha256
from .searchqa import SEARCHQA_DATASET_REPO, SEARCHQA_DATASET_REVISION


_STAGES = frozenset(
    {"zero_call_dry_run", "mechanism_smoke", "full_profile_pilot"}
)
_REQUIRED_STOP_CONDITIONS = frozenset(
    {
        "budget_breach",
        "controller_failure",
        "data_firewall_violation",
        "selection_saturation",
    }
)
_UNRESOLVED_MODEL_IDENTITIES = frozenset(
    {"default", "configured-default", "unknown", "tbd", "todo"}
)
_PROJECT_ROOT = Path(__file__).resolve().parents[2]


class PaperPreregistrationViolation(ValueError):
    """Raised when an experiment is not frozen tightly enough to execute."""


@dataclass(frozen=True)
class PreregisteredArtifact:
    artifact_id: str
    path: Path
    sha256: str

    def verify(self) -> None:
        if type(self.artifact_id) is not str or not self.artifact_id.strip():
            raise PaperPreregistrationViolation("artifact_id must be non-empty")
        _require_sha256(self.sha256, context=f"artifact {self.artifact_id} hash")
        if not self.path.is_file():
            raise PaperPreregistrationViolation(
                f"artifact is missing: {self.artifact_id} ({self.path})"
            )
        actual = hashlib.sha256(self.path.read_bytes()).hexdigest()
        if actual != self.sha256:
            raise PaperPreregistrationViolation(
                f"artifact hash drift: {self.artifact_id}"
            )


@dataclass(frozen=True)
class PaperDevelopmentPreregistration:
    source_path: Path
    payload: Mapping[str, Any]
    artifacts: tuple[PreregisteredArtifact, ...]

    @property
    def stage(self) -> str:
        return str(self.payload["stage"])

    @property
    def test_access_allowed(self) -> bool:
        return bool(self.payload["test_access"]["allowed"])

    def artifact(self, artifact_id: str) -> PreregisteredArtifact:
        for artifact in self.artifacts:
            if artifact.artifact_id == artifact_id:
                return artifact
        raise PaperPreregistrationViolation(
            f"preregistration has no artifact {artifact_id!r}"
        )

    def verify(self) -> "PaperDevelopmentPreregistration":
        _validate_payload(self.payload)
        ids = [artifact.artifact_id for artifact in self.artifacts]
        paths = [artifact.path for artifact in self.artifacts]
        if len(ids) != len(set(ids)) or len(paths) != len(set(paths)):
            raise PaperPreregistrationViolation(
                "preregistered artifacts require unique ids and paths"
            )
        for artifact in self.artifacts:
            artifact.verify()
        plan_artifact_id = self.payload["execution"]["plan_artifact_id"]
        self.artifact(plan_artifact_id)
        for artifact_id in (
            "materialization_receipt",
            "official_train_id_manifest",
            "official_selection_id_manifest",
        ):
            self.artifact(artifact_id)
        authorization = self.payload["authorization"]
        if authorization is not None:
            receipt_artifact = self.artifact(
                authorization["zero_cost_receipt_artifact_id"]
            )
            try:
                receipt = json.loads(receipt_artifact.path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as error:
                raise PaperPreregistrationViolation(
                    "zero-cost authorization receipt must be readable JSON"
                ) from error
            _validate_zero_cost_receipt(
                receipt, code_commit=authorization["local_code_commit"]
            )
            dry_receipt_artifact = self.artifact(
                authorization["mechanism_dry_run_receipt_artifact_id"]
            )
            dry_prereg_artifact = self.artifact(
                authorization["mechanism_dry_run_preregistration_artifact_id"]
            )
            try:
                dry_receipt = json.loads(
                    dry_receipt_artifact.path.read_text(encoding="utf-8")
                )
            except (OSError, json.JSONDecodeError) as error:
                raise PaperPreregistrationViolation(
                    "mechanism dry-run receipt must be readable JSON"
                ) from error
            _validate_dry_run_receipt(dry_receipt, budgets=self.payload["budgets"])
            _verify_dry_run_evidence_artifacts(
                dry_receipt["evidence_artifacts"],
                root=dry_receipt_artifact.path.parent,
            )
            dry_prereg = load_paper_preregistration(dry_prereg_artifact.path)
            if (
                dry_prereg.stage != "zero_call_dry_run"
                or dry_receipt["preregistration_sha256"]
                != hashlib.sha256(dry_prereg_artifact.path.read_bytes()).hexdigest()
                or dry_prereg.artifact("train_items").sha256
                != self.artifact("train_items").sha256
                or dry_prereg.artifact("selection_items").sha256
                != self.artifact("selection_items").sha256
                or dry_prereg.artifact("materialization_receipt").sha256
                != self.artifact("materialization_receipt").sha256
                or dry_prereg.artifact("plan").sha256
                != self.artifact("plan").sha256
                or canonical_json_sha256(
                    json.loads(self.artifact("plan").path.read_text(encoding="utf-8"))
                )
                != dry_receipt["plan_sha256"]
            ):
                raise PaperPreregistrationViolation(
                    "mechanism dry-run evidence does not match paid artifacts"
                )
            self.artifact("coco_binary")
            self.artifact("coco_config")
        return self


def load_paper_preregistration(
    path: str | Path,
) -> PaperDevelopmentPreregistration:
    source = Path(path).resolve()
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PaperPreregistrationViolation(
            f"preregistration must be readable strict JSON: {source}"
        ) from error
    _validate_payload(payload)
    artifacts = tuple(
        PreregisteredArtifact(
            artifact_id=item["artifact_id"],
            path=_resolve_artifact_path(source.parent, item["path"]),
            sha256=item["sha256"],
        )
        for item in payload["artifacts"]
    )
    return PaperDevelopmentPreregistration(
        source_path=source,
        payload=payload,
        artifacts=artifacts,
    ).verify()


def _validate_payload(payload: object) -> None:
    _require_exact_keys(
        payload,
        {
            "schema_version",
            "protocol_id",
            "stage",
            "authorization",
            "benchmark",
            "models",
            "execution",
            "budgets",
            "stop_conditions",
            "test_access",
            "artifacts",
        },
        context="preregistration",
    )
    if payload["schema_version"] != "paper-development-preregistration-v2":
        raise PaperPreregistrationViolation("unsupported preregistration schema")
    if payload["protocol_id"] != "paper-faithful-v1":
        raise PaperPreregistrationViolation("preregistration protocol drift")
    if payload["stage"] not in _STAGES:
        raise PaperPreregistrationViolation("unsupported development stage")
    _validate_authorization(payload["authorization"], stage=payload["stage"])
    _validate_benchmark(payload["benchmark"])
    _validate_models(payload["models"])
    _validate_execution(payload["execution"], stage=payload["stage"])
    _validate_budgets(payload["budgets"])
    stops = payload["stop_conditions"]
    if (
        type(stops) is not list
        or any(type(item) is not str or not item for item in stops)
        or len(stops) != len(set(stops))
        or not _REQUIRED_STOP_CONDITIONS.issubset(stops)
    ):
        raise PaperPreregistrationViolation(
            "preregistration requires all fail-closed stop conditions"
        )
    _require_exact_keys(
        payload["test_access"], {"allowed", "attempt"}, context="test_access"
    )
    if (
        payload["test_access"]["allowed"] is not False
        or payload["test_access"]["attempt"] != 0
    ):
        raise PaperPreregistrationViolation(
            "development preregistration cannot authorize test access"
        )
    artifacts = payload["artifacts"]
    if type(artifacts) is not list or not artifacts:
        raise PaperPreregistrationViolation(
            "preregistration requires hash-bound artifacts"
        )
    for item in artifacts:
        _require_exact_keys(
            item, {"artifact_id", "path", "sha256"}, context="artifact"
        )
        artifact_id = item["artifact_id"]
        if type(artifact_id) is str and re.search(
            r"(^|_)test($|_)", artifact_id.lower()
        ):
            raise PaperPreregistrationViolation(
                "development preregistration cannot bind a test payload artifact"
            )


def _validate_benchmark(payload: object) -> None:
    _require_exact_keys(
        payload,
        {
            "id",
            "source_repo",
            "source_revision",
            "train_split_id",
            "selection_split_id",
            "train_count",
            "selection_count",
            "official_test_id_manifest_sha256",
            "test_payload_status",
        },
        context="benchmark",
    )
    if (
        payload["id"] != "searchqa"
        or payload["source_repo"] != SEARCHQA_DATASET_REPO
        or payload["source_revision"] != SEARCHQA_DATASET_REVISION
    ):
        raise PaperPreregistrationViolation("SearchQA source identity drift")
    if payload["test_payload_status"] != "not_materialized":
        raise PaperPreregistrationViolation(
            "development preregistration requires the test payload to remain not materialized"
        )
    _require_sha256(
        payload["official_test_id_manifest_sha256"],
        context="official test id manifest",
    )
    for name in ("train_split_id", "selection_split_id"):
        if type(payload[name]) is not str or not payload[name].strip():
            raise PaperPreregistrationViolation(f"benchmark requires {name}")
    if payload["train_split_id"] == payload["selection_split_id"]:
        raise PaperPreregistrationViolation("train and selection split ids must differ")
    for name in ("train_count", "selection_count"):
        if type(payload[name]) is not int or payload[name] < 1:
            raise PaperPreregistrationViolation(f"benchmark requires positive {name}")


def _validate_authorization(payload: object, *, stage: str) -> None:
    if stage == "zero_call_dry_run":
        if payload is not None:
            raise PaperPreregistrationViolation(
                "zero-call dry-run must not claim paid authorization"
            )
        return
    _require_exact_keys(
        payload,
        {
            "local_code_commit",
            "zero_cost_receipt_artifact_id",
            "mechanism_dry_run_receipt_artifact_id",
            "mechanism_dry_run_preregistration_artifact_id",
            "paid_development_authorized",
        },
        context="authorization",
    )
    commit = payload["local_code_commit"]
    if type(commit) is not str or re.fullmatch(r"[0-9a-f]{40}", commit) is None:
        raise PaperPreregistrationViolation(
            "authorization requires a lowercase Git commit SHA"
        )
    if (
        payload["zero_cost_receipt_artifact_id"] != "zero_cost_receipt"
        or payload["mechanism_dry_run_receipt_artifact_id"]
        != "mechanism_dry_run_receipt"
        or payload["mechanism_dry_run_preregistration_artifact_id"]
        != "mechanism_dry_run_preregistration"
        or payload["paid_development_authorized"] is not True
    ):
        raise PaperPreregistrationViolation(
            "paid development requires the bound zero-cost authorization receipt"
        )


def _validate_zero_cost_receipt(payload: object, *, code_commit: str) -> None:
    expected = {
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
    _require_exact_keys(payload, expected, context="zero-cost receipt")
    if (
        payload["schema_version"] != "paper-zero-cost-gate-v1"
        or payload["status"] != "passed"
        or payload["external_calls"] != 0
        or payload["network_guard_active"] is not True
        or payload["paid_experiment_executed"] is not False
        or payload["paid_development_authorized"] is not True
        or payload["code_commit"] != code_commit
        or payload["worktree_clean"] is not True
        or payload["prompt_count"] != 18
        or payload["test_targets"] != ["tests/conformance", "tests/provenance"]
        or payload["violations"] != []
    ):
        raise PaperPreregistrationViolation(
            "zero-cost receipt does not authorize the preregistered clean commit"
        )
    for name in (
        "prompt_snapshot_sha256",
        "source_lock_sha256",
        "golden_trace_sha256",
    ):
        _require_sha256(payload[name], context=f"zero-cost receipt {name}")
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
        "golden_trace_sha256": hashlib.sha256(
            (
                _PROJECT_ROOT
                / "tests/conformance/golden/algorithm1-fast-loop-v1.json"
            ).read_bytes()
        ).hexdigest(),
    }
    if any(payload[name] != value for name, value in expected_hashes.items()):
        raise PaperPreregistrationViolation(
            "zero-cost receipt lock hashes do not match the current commit artifacts"
        )


def _validate_dry_run_receipt(
    payload: object, *, budgets: Mapping[str, Any]
) -> None:
    _require_exact_keys(
        payload,
        {
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
        },
        context="mechanism dry-run receipt",
    )
    _require_exact_keys(
        payload["usage"],
        {
            "logical_target_calls",
            "external_target_calls",
            "target_tokens",
            "estimated_target_tokens",
            "logical_optimizer_calls",
            "external_optimizer_calls",
            "optimizer_tokens",
            "estimated_optimizer_tokens",
        },
        context="mechanism dry-run usage",
    )
    usage = payload["usage"]
    if (
        payload["schema_version"] != "paper-searchqa-development-receipt-v2"
        or payload["status"] != "completed"
        or payload["stage"] != "zero_call_dry_run"
        or payload["completed_epochs"] != 2
        or payload["completed_steps"] != 2
        or payload["selection_unsaturated"] is not True
        or payload["full_call_graph_complete"] is not True
        or payload["test_access"] != {"allowed": False, "attempt": 0}
        or payload["test_payload_status"] != "not_materialized"
        or payload["claim_class"] != "mechanism_test"
        or payload["evidence_level"] is not None
        or usage["external_target_calls"] != 0
        or usage["external_optimizer_calls"] != 0
        or usage["target_tokens"] != 0
        or usage["optimizer_tokens"] != 0
    ):
        raise PaperPreregistrationViolation(
            "mechanism dry-run receipt is not eligible for paid execution"
        )
    for name in (
        "logical_target_calls",
        "estimated_target_tokens",
        "logical_optimizer_calls",
        "estimated_optimizer_tokens",
    ):
        if type(usage[name]) is not int or usage[name] < 1:
            raise PaperPreregistrationViolation(
                "mechanism dry-run usage must be positive"
            )
    factor = budgets["safety_factor"]
    expected_budgets = {
        "target_calls": math.ceil(usage["logical_target_calls"] * factor),
        "target_tokens": math.ceil(usage["estimated_target_tokens"] * factor),
        "optimizer_calls": math.ceil(usage["logical_optimizer_calls"] * factor),
        "optimizer_tokens": math.ceil(
            usage["estimated_optimizer_tokens"] * factor
        ),
        "wall_time_seconds": 12_000.0,
        "safety_factor": factor,
        "token_policy": "audit_only",
    }
    if dict(budgets) != expected_budgets:
        raise PaperPreregistrationViolation(
            "paid budgets are not mechanically derived from the dry-run receipt"
        )


def _verify_dry_run_evidence_artifacts(payload: object, *, root: Path) -> None:
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
        raise PaperPreregistrationViolation(
            "mechanism dry-run is missing required evidence artifacts"
        )
    resolved_root = root.resolve()
    for artifact in payload.values():
        _require_exact_keys(
            artifact,
            {"path", "sha256", "size_bytes"},
            context="mechanism dry-run evidence artifact",
        )
        raw_path = artifact["path"]
        if type(raw_path) is not str or not raw_path or Path(raw_path).is_absolute():
            raise PaperPreregistrationViolation(
                "mechanism dry-run evidence path must be relative"
            )
        path = (resolved_root / raw_path).resolve()
        if path.parent != resolved_root or not path.is_file():
            raise PaperPreregistrationViolation(
                "mechanism dry-run evidence artifact is missing or escaped"
            )
        _require_sha256(
            artifact["sha256"],
            context="mechanism dry-run evidence artifact",
        )
        if (
            hashlib.sha256(path.read_bytes()).hexdigest() != artifact["sha256"]
            or type(artifact["size_bytes"]) is not int
            or artifact["size_bytes"] < 0
            or path.stat().st_size != artifact["size_bytes"]
        ):
            raise PaperPreregistrationViolation(
                "mechanism dry-run evidence artifact drifted"
            )


def _validate_models(payload: object) -> None:
    _require_exact_keys(
        payload,
        {
            "target_model",
            "target_reasoning",
            "optimizer_model",
            "optimizer_reasoning",
        },
        context="models",
    )
    for name in (
        "target_model",
        "target_reasoning",
        "optimizer_model",
        "optimizer_reasoning",
    ):
        value = payload[name]
        if (
            type(value) is not str
            or not value.strip()
            or value.strip().lower() in _UNRESOLVED_MODEL_IDENTITIES
        ):
            raise PaperPreregistrationViolation(
                f"preregistration requires an exact model identity: {name}"
            )


def _validate_execution(payload: object, *, stage: str) -> None:
    _require_exact_keys(
        payload,
        {
            "seed",
            "retry_policy",
            "target_backend",
            "optimizer_backend",
            "profile_sha256",
            "plan_artifact_id",
        },
        context="execution",
    )
    if type(payload["seed"]) is not int:
        raise PaperPreregistrationViolation("execution seed must be an integer")
    if payload["retry_policy"] != "semantic-retry-once-v1":
        raise PaperPreregistrationViolation(
            "M7 retry policy is frozen to semantic-retry-once-v1"
        )
    _require_sha256(payload["profile_sha256"], context="paper profile")
    if type(payload["plan_artifact_id"]) is not str or not payload[
        "plan_artifact_id"
    ].strip():
        raise PaperPreregistrationViolation("execution requires plan_artifact_id")
    if stage == "zero_call_dry_run" and (
        payload["target_backend"] != "scripted"
        or payload["optimizer_backend"] != "scripted"
    ):
        raise PaperPreregistrationViolation(
            "zero-call dry-run requires scripted target and optimizer"
        )
    if stage != "zero_call_dry_run" and (
        payload["target_backend"] != "coco"
        or payload["optimizer_backend"] != "openai_compatible"
    ):
        raise PaperPreregistrationViolation(
            "paid development requires Coco and the external optimizer"
        )


def _validate_budgets(payload: object) -> None:
    _require_exact_keys(
        payload,
        {
            "target_calls",
            "target_tokens",
            "optimizer_calls",
            "optimizer_tokens",
            "wall_time_seconds",
            "safety_factor",
            "token_policy",
        },
        context="budgets",
    )
    for name in (
        "target_calls",
        "target_tokens",
        "optimizer_calls",
        "optimizer_tokens",
    ):
        if type(payload[name]) is not int or payload[name] < 1:
            raise PaperPreregistrationViolation(
                f"preregistration budget {name} must be positive"
            )
    wall = payload["wall_time_seconds"]
    if (
        isinstance(wall, bool)
        or not isinstance(wall, (int, float))
        or not math.isfinite(wall)
        or wall <= 0
    ):
        raise PaperPreregistrationViolation(
            "preregistration wall-time budget must be positive and finite"
        )
    factor = payload["safety_factor"]
    if (
        isinstance(factor, bool)
        or not isinstance(factor, (int, float))
        or not 1.25 <= float(factor) <= 1.5
    ):
        raise PaperPreregistrationViolation(
            "preregistration safety factor must be between 1.25 and 1.5"
        )
    if payload["token_policy"] != "audit_only":
        raise PaperPreregistrationViolation(
            "preregistration model-token policy must be audit_only"
        )


def _require_exact_keys(
    payload: object,
    expected: set[str],
    *,
    context: str,
) -> None:
    if type(payload) is not dict or set(payload) != expected:
        raise PaperPreregistrationViolation(
            f"{context} must contain exactly: {', '.join(sorted(expected))}"
        )


def _require_sha256(value: object, *, context: str) -> None:
    if type(value) is not str or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise PaperPreregistrationViolation(
            f"{context} must be a lowercase SHA256 hash"
        )


def _resolve_artifact_path(root: Path, raw_path: object) -> Path:
    if type(raw_path) is not str or not raw_path.strip():
        raise PaperPreregistrationViolation("artifact path must be non-empty")
    path = Path(raw_path)
    return (path if path.is_absolute() else root / path).absolute()
