#!/usr/bin/env python3
"""Build validated SkillOpt runner manifests from stage-level knobs."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from work.experiment_runner import validate_runner_manifest
from work.skillopt_stage_policy import (
    critical_contracts_for_selection,
    default_stage_gate_criteria,
    parse_csv,
    stage_names,
    stage_policy,
    validate_manifest_stage_policy,
    validate_task_ids,
)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = build_manifest(args)
    validate_manifest(manifest)
    write_json(args.out, manifest)
    print(f"runner_manifest={args.out}")
    return 0


def build_manifest(args: argparse.Namespace) -> dict[str, Any]:
    policy = stage_policy(args.stage)
    timeout_seconds = (
        args.timeout_seconds
        if args.timeout_seconds is not None
        else int(policy.get("default_timeout_seconds", 7200))
    )
    validation_confirmation_rounds = (
        args.validation_confirmation_rounds
        if args.validation_confirmation_rounds is not None
        else int(policy["default_validation_confirmation_rounds"])
    )
    validation_required_wins = (
        args.validation_required_wins
        if args.validation_required_wins is not None
        else int(policy["default_validation_required_wins"])
    )
    early_stop_validation_score = (
        args.early_stop_validation_score
        if args.early_stop_validation_score is not None
        else policy.get("default_early_stop_validation_score")
    )
    conditions = parse_csv(args.conditions) or list(policy["default_conditions"])
    train_task_ids = set(parse_csv(args.train_task_ids))
    selection_task_ids = set(parse_csv(args.selection_task_ids))
    task_contracts = set(parse_csv(args.task_contracts))
    critical_contracts = critical_contracts_for_selection(
        selection_task_ids=selection_task_ids,
        task_contracts=task_contracts,
    )
    development_gate = {
        **default_stage_gate_criteria(args.stage, critical_contracts=critical_contracts),
        **json_object(args.development_gate_json),
    }
    command = [
        args.python,
        "work/run_coding_hidden_v2_matrix.py",
        "--out",
        args.run_dir,
        "--seeds",
        args.seeds,
        "--seed-workers",
        str(args.seed_workers),
        "--epochs",
        str(args.epochs),
        "--rollout-batch-size",
        str(args.rollout_batch_size),
        "--reflection-minibatch-size",
        str(args.reflection_minibatch_size),
        "--learning-rate",
        str(args.learning_rate),
        "--learning-rate-floor",
        str(args.learning_rate_floor),
        "--learning-rate-schedule",
        args.learning_rate_schedule,
        "--slow-update-sample-size",
        str(args.slow_update_sample_size),
        "--task-timeout",
        str(args.task_timeout),
        "--task-retries",
        str(args.task_retries),
        "--retry-backoff-seconds",
        str(args.retry_backoff_seconds),
        "--validation-confirmation-rounds",
        str(validation_confirmation_rounds),
        "--validation-required-wins",
        str(validation_required_wins),
        "--validation-mean-delta",
        str(args.validation_mean_delta),
        "--development-gate-required-wins",
        str(development_gate["min_seed_wins"]),
        "--development-gate-mean-delta",
        str(development_gate["best_baseline_margin"]),
        "--early-stop-rejection-limit",
        str(args.early_stop_rejection_limit),
        "--editor-timeout",
        str(args.editor_timeout),
        "--conditions",
        ",".join(conditions),
        "--experiment-stage",
        args.stage,
    ]
    if early_stop_validation_score is not None:
        command.extend(["--early-stop-validation-score", str(early_stop_validation_score)])
    if args.disable_slow_update:
        command.append("--disable-slow-update")
    if train_task_ids:
        command.extend(["--train-task-ids", ",".join(sorted(train_task_ids))])
    if selection_task_ids:
        command.extend(["--selection-task-ids", ",".join(sorted(selection_task_ids))])
    if task_contracts:
        command.extend(["--task-contracts", ",".join(sorted(task_contracts))])
    if args.baseline_summary:
        command.extend(["--baseline-summary", args.baseline_summary])
    if args.task_limit:
        command.extend(["--task-limit", str(args.task_limit)])
    if args.resume:
        command.append("--resume")

    env = {
        "COCO_AGENT_TIMEOUT": str(args.task_timeout),
        "EXTERNAL_LLM_TIMEOUT": str(args.external_llm_timeout or args.editor_timeout),
    }
    base_url = args.external_llm_base_url or os.environ.get("EXTERNAL_LLM_BASE_URL", "")
    model = args.external_llm_model or os.environ.get("EXTERNAL_LLM_MODEL", "")
    if base_url:
        env["EXTERNAL_LLM_BASE_URL"] = base_url
    if model:
        env["EXTERNAL_LLM_MODEL"] = model

    return {
        "schema_version": 1,
        "experiment_type": "coding_hidden_v2_matrix",
        "experiment_stage": args.stage,
        "runner_role": "mechanical_execution_only",
        "timeout_seconds": timeout_seconds,
        "immutable_controls": {
            "do_not_change_coco_model": True,
            "target_model_policy": "read-local-default-without-override",
        },
        "out_dir": args.run_dir,
        "command": command,
        "acceptance": development_gate,
        "env": env,
        "env_passthrough": ["EXTERNAL_LLM_API_KEY"],
    }


def validate_manifest(manifest: dict[str, Any]) -> None:
    validate_runner_manifest(manifest)
    issues = validate_manifest_stage_policy(manifest)
    command = [str(item) for item in manifest["command"]]
    train_task_ids = set(parse_csv(command_value(command, "--train-task-ids")))
    selection_task_ids = set(parse_csv(command_value(command, "--selection-task-ids")))
    issues.extend(validate_task_ids(train_task_ids=train_task_ids, selection_task_ids=selection_task_ids))
    baseline_summary = command_value(command, "--baseline-summary")
    if baseline_summary and not resolve_workspace_path(baseline_summary).exists():
        issues.append({"code": "missing_baseline_summary", "message": f"baseline summary does not exist: {baseline_summary}"})
    if issues:
        messages = "; ".join(f"{item['code']}: {item['message']}" for item in issues)
        raise ValueError(messages)


def command_value(command: list[str], option: str) -> str:
    prefix = option + "="
    for index, item in enumerate(command):
        if item == option and index + 1 < len(command):
            return command[index + 1]
        if item.startswith(prefix):
            return item[len(prefix) :]
    return ""


def json_object(raw: str | None) -> dict[str, Any]:
    if not raw:
        return {}
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError("--development-gate-json must decode to an object")
    return payload


def resolve_workspace_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def write_json(path: str | Path, payload: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=stage_names(), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--python", default="python3")
    parser.add_argument("--timeout-seconds", type=int)
    parser.add_argument("--seeds", default="seed-a,seed-b,seed-c")
    parser.add_argument("--seed-workers", type=int, default=1)
    parser.add_argument("--conditions", default="")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--rollout-batch-size", type=int, default=1)
    parser.add_argument("--reflection-minibatch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=int, default=2)
    parser.add_argument("--learning-rate-floor", type=int, default=1)
    parser.add_argument("--learning-rate-schedule", choices=("constant", "linear", "cosine"), default="constant")
    parser.add_argument("--slow-update-sample-size", type=int, default=1)
    parser.add_argument("--disable-slow-update", dest="disable_slow_update", action="store_true")
    parser.add_argument("--enable-slow-update", dest="disable_slow_update", action="store_false")
    parser.set_defaults(disable_slow_update=True)
    parser.add_argument("--task-timeout", type=int, default=360)
    parser.add_argument("--task-retries", type=int, default=1)
    parser.add_argument("--retry-backoff-seconds", type=float, default=5.0)
    parser.add_argument("--validation-confirmation-rounds", type=int)
    parser.add_argument("--validation-required-wins", type=int)
    parser.add_argument("--validation-mean-delta", type=float, default=0.05)
    parser.add_argument("--early-stop-rejection-limit", type=int, default=0)
    parser.add_argument("--early-stop-validation-score", type=float)
    parser.add_argument("--editor-timeout", type=int, default=600)
    parser.add_argument("--external-llm-timeout", type=int)
    parser.add_argument("--external-llm-base-url")
    parser.add_argument("--external-llm-model")
    parser.add_argument("--train-task-ids", default="")
    parser.add_argument("--selection-task-ids", default="")
    parser.add_argument("--task-contracts", default="")
    parser.add_argument("--baseline-summary", default="")
    parser.add_argument("--task-limit", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--development-gate-json", help="JSON object overriding stage gate criteria")
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
