#!/usr/bin/env python3
"""Shared stage policy for mechanical SkillOpt experiment tooling."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
BENCHMARK = ROOT / "examples/coding-hidden-v2"


STAGE_POLICIES: dict[str, dict[str, Any]] = {
    "mechanism_smoke": {
        "default_timeout_seconds": 7200,
        "allow_zero_confirmation": True,
        "default_validation_confirmation_rounds": 0,
        "default_validation_required_wins": 1,
        "default_early_stop_validation_score": None,
        "default_conditions": ["executive"],
        "development_gate": {
            "best_baseline_margin": 0.05,
            "min_seed_wins": 2,
            "contract_macro_margin": 0.0,
        },
    },
    "full_selection_development": {
        "default_timeout_seconds": 43200,
        "allow_zero_confirmation": False,
        "default_validation_confirmation_rounds": 1,
        "default_validation_required_wins": 2,
        "default_early_stop_validation_score": 1.0,
        "default_conditions": ["executive"],
        "requires_cached_baseline": True,
        "development_gate": {
            "best_baseline_margin": 0.05,
            "min_seed_wins": 2,
            "contract_macro_margin": 0.0,
        },
    },
    "same_run_baseline_matrix": {
        "default_timeout_seconds": 86400,
        "allow_zero_confirmation": False,
        "default_validation_confirmation_rounds": 1,
        "default_validation_required_wins": 2,
        "default_early_stop_validation_score": 1.0,
        "default_conditions": ["no_skill", "human_skill", "one_shot", "executive"],
        "requires_same_run_baseline": True,
        "development_gate": {
            "best_baseline_margin": 0.05,
            "min_seed_wins": 2,
            "contract_macro_margin": 0.0,
        },
    },
}


def stage_names() -> list[str]:
    return sorted(STAGE_POLICIES)


def stage_policy(stage: str) -> dict[str, Any]:
    if stage not in STAGE_POLICIES:
        raise ValueError(f"Unknown experiment stage: {stage}")
    return dict(STAGE_POLICIES[stage])


def validate_manifest_stage_policy(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    issues = []
    stage = str(manifest.get("experiment_stage") or "")
    policy = STAGE_POLICIES.get(stage)
    if policy is None:
        return [issue("unknown_stage", f"unknown experiment_stage: {stage or '<missing>'}")]
    command = [str(item) for item in manifest.get("command") or []]
    rounds = command_int_option(command, "--validation-confirmation-rounds")
    conditions = command_csv_option(command, "--conditions") or normalized_string_list(
        manifest.get("conditions")
    )
    baseline_summary = command_option_value(command, "--baseline-summary") or str(
        manifest.get("baseline_summary") or ""
    )
    if rounds is None:
        issues.append(issue("missing_validation_confirmation_rounds", "command must set --validation-confirmation-rounds"))
    elif rounds == 0 and not bool(policy.get("allow_zero_confirmation")):
        issues.append(
            issue(
                "confirmation_rounds_required",
                f"{stage} requires --validation-confirmation-rounds >= 1",
            )
        )
    if policy.get("requires_cached_baseline") and not baseline_summary:
        issues.append(issue("missing_cached_baseline", f"{stage} requires --baseline-summary"))
    if policy.get("requires_same_run_baseline"):
        condition_set = set(conditions)
        if "executive" not in condition_set:
            issues.append(issue("missing_executive_condition", f"{stage} requires executive condition"))
        if not condition_set.intersection({"no_skill", "human_skill", "one_shot"}):
            issues.append(issue("missing_baseline_condition", f"{stage} requires a baseline condition"))
    if set(conditions) == {"executive"} and stage != "mechanism_smoke" and not baseline_summary:
        issues.append(issue("executive_only_without_baseline", "executive-only development runs require --baseline-summary"))
    return issues


def default_stage_gate_criteria(stage: str, *, critical_contracts: list[str] | None = None) -> dict[str, Any]:
    criteria = dict(stage_policy(stage).get("development_gate") or {})
    if critical_contracts:
        criteria["critical_contracts"] = sorted(set(critical_contracts))
        criteria["critical_contract_regression_epsilon"] = 0.0
    return criteria


def critical_contracts_for_selection(
    *,
    selection_task_ids: set[str] | None = None,
    task_contracts: set[str] | None = None,
    benchmark: Path = BENCHMARK,
) -> list[str]:
    if task_contracts:
        return sorted(task_contracts)
    selected_ids = set(selection_task_ids or set())
    contracts = set()
    for task in load_jsonl(benchmark / "selection.jsonl"):
        if selected_ids and str(task.get("id")) not in selected_ids:
            continue
        metadata = task.get("metadata") if isinstance(task.get("metadata"), dict) else {}
        contracts.update(normalized_string_list(metadata.get("contract_tags")))
    return sorted(contracts)


def validate_task_ids(
    *,
    train_task_ids: set[str] | None,
    selection_task_ids: set[str] | None,
    benchmark: Path = BENCHMARK,
) -> list[dict[str, Any]]:
    issues = []
    for split, selected_ids in (
        ("train", set(train_task_ids or set())),
        ("selection", set(selection_task_ids or set())),
    ):
        if not selected_ids:
            continue
        available = {str(task.get("id")) for task in load_jsonl(benchmark / f"{split}.jsonl")}
        missing = sorted(selected_ids - available)
        if missing:
            issues.append(issue(f"missing_{split}_task_ids", f"unknown {split} task ids: {', '.join(missing)}"))
    return issues


def command_option_value(command: list[str], option: str) -> str:
    prefix = option + "="
    for index, item in enumerate(command):
        if item == option and index + 1 < len(command):
            return command[index + 1]
        if item.startswith(prefix):
            return item[len(prefix) :]
    return ""


def command_int_option(command: list[str], option: str) -> int | None:
    value = command_option_value(command, option)
    return int(value) if value else None


def command_csv_option(command: list[str], option: str) -> list[str]:
    return parse_csv(command_option_value(command, option))


def parse_csv(raw: str | None) -> list[str]:
    return [item.strip() for item in (raw or "").split(",") if item.strip()]


def normalized_string_list(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value] if value else []
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if str(item)]


def issue(code: str, message: str) -> dict[str, str]:
    return {"code": code, "message": message}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if isinstance(payload, dict):
            records.append(payload)
    return records
