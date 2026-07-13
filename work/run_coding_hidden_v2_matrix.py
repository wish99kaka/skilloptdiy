#!/usr/bin/env python3
"""Run resumable development baselines and executive optimization on coding-hidden-v2."""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Sequence


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "work"))

from textskill_optimizer.command_editor import CommandEditorConfig, CommandSkillEditor
from textskill_optimizer.executive_optimizer import ExecutiveOptimizerConfig, ExecutiveSkillOptimizer
from textskill_optimizer.io import load_tasks_jsonl, load_text, write_json, write_text
from textskill_optimizer.plugins.coding import (
    CodingRunner,
    CodingScorer,
    coding_retryable_anomaly_reasons,
)
from textskill_optimizer.usage_ledger import (
    append_usage_event,
    combine_usage_summaries,
    estimate_tokens_from_chars,
    summarize_usage_file,
    summarize_usage_files,
)
from work.development_gate import build_development_gate
from work.run_coco_hidden_eval import build_coco_tasks


BENCHMARK = ROOT / "examples/coding-hidden-v2"
COCO_WRAPPER = ROOT / "examples/coding/coco_agent_wrapper.py"
EXTERNAL_EDITOR = ROOT / "examples/coding/openai_compatible_skill_editor.py"
EXPERIMENT_INTERNAL_USAGE_KINDS = ("optimizer_command", "optimizer_model_api")
TARGET_AGENT_USAGE_KINDS = ("target_agent_cli",)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    seeds = parse_seeds(args.seeds)
    args.conditions = parse_conditions(args.conditions)
    args.train_task_ids = parse_csv_set(args.train_task_ids)
    args.selection_task_ids = parse_csv_set(args.selection_task_ids)
    args.task_contracts = parse_csv_set(args.task_contracts)
    optimizer_model = require_external_editor_config()
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest(args, seeds, optimizer_model=optimizer_model)
    write_json(args.out / "experiment_manifest.json", manifest)

    env_updates = {
        "COCO_TASK_TIMEOUT": str(args.task_timeout),
    }
    if args.task_limit:
        env_updates["COCO_TASK_LIMIT"] = str(args.task_limit)

    with patched_env(env_updates):
        train_path = build_coco_tasks(BENCHMARK / "train.jsonl", COCO_WRAPPER)
        selection_path = build_coco_tasks(BENCHMARK / "selection.jsonl", COCO_WRAPPER)
        train_tasks = filter_tasks(
            load_tasks_jsonl(train_path),
            task_ids=args.train_task_ids,
            contract_tags=args.task_contracts,
            split_name="train",
        )
        selection_tasks = filter_tasks(
            load_tasks_jsonl(selection_path),
            task_ids=args.selection_task_ids,
            contract_tags=args.task_contracts,
            split_name="selection",
        )

        rows = run_seeds(args, seeds, train_tasks, selection_tasks)
    rows = merge_cached_baseline_rows(rows, args.baseline_summary, seeds, args.conditions)

    summary, aggregate_stdout = build_summary(manifest, rows)
    write_json(args.out / "summary.json", summary)
    print(aggregate_stdout)
    return 0


def build_summary(manifest: dict[str, Any], rows: list[dict[str, Any]]) -> tuple[dict[str, Any], str]:
    aggregate = aggregate_rows(rows)
    aggregate_stdout = json.dumps(aggregate, indent=2, sort_keys=True)
    development_gate = build_development_gate(
        rows,
        aggregate,
        dict(manifest.get("development_gate_criteria") or {}),
    )
    summary = {
        "manifest": manifest,
        "rows": rows,
        "aggregate": aggregate,
        "development_gate": development_gate,
        "locked_test_recommended": bool(development_gate["locked_test_recommended"]),
        "usage": build_usage_report(
            [
                row["usage_ledger_path"]
                for row in rows
                if row.get("usage_ledger_path") and not row.get("cached_baseline")
            ],
            aggregate_stdout_chars=len(aggregate_stdout),
        ),
    }
    return summary, aggregate_stdout


def run_seeds(
    args: argparse.Namespace,
    seeds: list[str],
    train_tasks: list,
    selection_tasks: list,
) -> list[dict[str, Any]]:
    workers = min(max(1, args.seed_workers), len(seeds))
    if workers == 1:
        return [
            row
            for seed in seeds
            for row in run_seed(args, seed, train_tasks, selection_tasks)
        ]
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="skillopt-seed") as executor:
        futures = {
            seed: executor.submit(run_seed, args, seed, train_tasks, selection_tasks)
            for seed in seeds
        }
        return [row for seed in seeds for row in futures[seed].result()]


def run_seed(args: argparse.Namespace, seed: str, train_tasks: list, selection_tasks: list) -> list[dict[str, Any]]:
    rows = []
    scorer = CodingScorer()
    conditions = (
        ("no_skill", BENCHMARK / "no_skill.md"),
        ("human_skill", BENCHMARK / "skill.md"),
    )
    for condition, skill_path in conditions:
        if condition not in args.conditions:
            continue
        run_dir = args.out / seed / condition
        usage_path = run_dir / "usage_ledger.jsonl"
        report_path = run_dir / "selection.json"
        if report_path.exists() and args.resume:
            report = json.loads(report_path.read_text(encoding="utf-8"))
            duration = load_duration(run_dir)
        else:
            run_dir.mkdir(parents=True, exist_ok=True)
            evaluator = build_baseline_evaluator(
                args,
                scorer,
                usage_path=usage_path,
                usage_context=usage_context(seed, condition),
            )
            started = time.monotonic()
            report_obj = evaluator.evaluate(load_text(skill_path), selection_tasks, name=f"{condition}:{seed}")
            duration = time.monotonic() - started
            report = report_obj.to_dict()
            write_json(report_path, report)
            write_json(run_dir / "timing.json", {"duration_seconds": duration})
        rows.append(build_row(seed, condition, report, duration, run_dir))

    if "one_shot" in args.conditions:
        one_shot_dir = args.out / seed / "one_shot"
        one_shot_usage_path = one_shot_dir / "usage_ledger.jsonl"
        one_shot_skill = one_shot_dir / "skill.md"
        one_shot_report = one_shot_dir / "selection.json"
        if not (args.resume and one_shot_skill.exists() and one_shot_report.exists()):
            one_shot_dir.mkdir(parents=True, exist_ok=True)
            skill_payload = generate_one_shot_skill(
                train_tasks,
                seed,
                args.editor_timeout,
                usage_ledger_path=one_shot_usage_path,
                usage_context=usage_context(seed, "one_shot"),
            )
            write_text(one_shot_skill, str(skill_payload["skill_text"]))
            write_json(one_shot_dir / "generation.json", skill_payload)
            evaluator = build_baseline_evaluator(
                args,
                scorer,
                usage_path=one_shot_usage_path,
                usage_context=usage_context(seed, "one_shot"),
            )
            started = time.monotonic()
            report_obj = evaluator.evaluate(load_text(one_shot_skill), selection_tasks, name=f"one_shot:{seed}")
            duration = time.monotonic() - started
            write_json(one_shot_report, report_obj.to_dict())
            write_json(one_shot_dir / "timing.json", {"duration_seconds": duration})
        report = json.loads(one_shot_report.read_text(encoding="utf-8"))
        rows.append(build_row(seed, "one_shot", report, load_duration(one_shot_dir), one_shot_dir))

    if "executive" not in args.conditions:
        return rows
    executive_dir = args.out / seed / "executive"
    executive_usage_path = executive_dir / "usage_ledger.jsonl"
    result_path = executive_dir / "result.json"
    if not (args.resume and result_path.exists()):
        runner = CodingRunner(
            usage_ledger_path=executive_usage_path,
            usage_context=usage_context(seed, "executive"),
        )
        editor = CommandSkillEditor(
            CommandEditorConfig(
                command=f"{sys.executable} {EXTERNAL_EDITOR}",
                timeout_seconds=args.editor_timeout,
                proposal_log_path=executive_dir / "proposals.jsonl",
                proposal_log_seed=seed,
                proposal_log_case="executive",
                usage_ledger_path=executive_usage_path,
                usage_context=usage_context(seed, "executive"),
            )
        )
        optimizer = ExecutiveSkillOptimizer(
            runner,
            scorer,
            editor,
            build_executive_config(args, seed),
            retry_detector=coding_retryable_anomaly_reasons,
        )
        started = time.monotonic()
        result = optimizer.optimize(
            load_text(BENCHMARK / "skill.md"),
            train_tasks,
            selection_tasks,
            run_dir=executive_dir,
        )
        duration = time.monotonic() - started
        write_json(executive_dir / "timing.json", {"duration_seconds": duration})
        write_json(result_path, result.to_dict())
    result = json.loads(result_path.read_text(encoding="utf-8"))
    rows.append(
        build_row(
            seed,
            "executive",
            result["final_validation_report"],
            load_duration(executive_dir),
            executive_dir,
            extra={
                "accepted_steps": result.get("accepted_steps"),
                "total_steps": result.get("total_steps"),
                "best_validation_score": result.get("best_validation_score"),
            },
        )
    )
    return rows


def build_baseline_evaluator(
    args: argparse.Namespace,
    scorer: CodingScorer,
    *,
    usage_path: Path,
    usage_context: dict[str, Any],
) -> ExecutiveSkillOptimizer:
    return ExecutiveSkillOptimizer(
        CodingRunner(usage_ledger_path=usage_path, usage_context=usage_context),
        scorer,
        editor=NoopEditor(),
        config=ExecutiveOptimizerConfig(
            epochs=1,
            enable_slow_update=False,
            task_retry_limit=args.task_retries,
            task_retry_backoff_seconds=args.retry_backoff_seconds,
            fail_on_persistent_task_anomaly=False,
        ),
        retry_detector=coding_retryable_anomaly_reasons,
    )


def build_executive_config(args: argparse.Namespace, seed: str) -> ExecutiveOptimizerConfig:
    return ExecutiveOptimizerConfig(
        epochs=args.epochs,
        rollout_batch_size=args.rollout_batch_size,
        reflection_minibatch_size=args.reflection_minibatch_size,
        learning_rate=args.learning_rate,
        learning_rate_floor=args.learning_rate_floor,
        learning_rate_schedule=args.learning_rate_schedule,
        rejected_buffer_limit=args.rejected_buffer_limit,
        slow_update_sample_size=args.slow_update_sample_size,
        enable_slow_update=not args.disable_slow_update,
        seed=stable_seed(seed),
        meta_skill_path=ROOT / "work/meta_skill.md",
        task_retry_limit=args.task_retries,
        task_retry_backoff_seconds=args.retry_backoff_seconds,
        fail_on_persistent_task_anomaly=False,
        validation_confirmation_rounds=args.validation_confirmation_rounds,
        validation_required_wins=args.validation_required_wins,
        validation_mean_delta=args.validation_mean_delta,
        early_stop_rejection_limit=args.early_stop_rejection_limit,
        early_stop_validation_score=getattr(args, "early_stop_validation_score", None),
    )


def usage_context(seed: str, condition: str) -> dict[str, Any]:
    return {
        "benchmark": "coding-hidden-v2",
        "seed": seed,
        "condition": condition,
    }


def generate_one_shot_skill(
    train_tasks: list,
    seed: str,
    timeout: int,
    *,
    usage_ledger_path: Path | None = None,
    usage_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    contexts = []
    for task in train_tasks:
        repo = Path(task.metadata["repo"])
        if not repo.is_absolute():
            repo = Path(task.metadata["_task_dir"]) / repo
        files = {}
        for path in sorted(repo.rglob("*.py")):
            if "__pycache__" not in path.parts:
                files[str(path.relative_to(repo))] = path.read_text(encoding="utf-8")
        contexts.append({"task_input": task.input, "public_files": files})
    payload = {"operation": "one_shot_skill", "seed_label": seed, "development_context": contexts}
    payload_text = json.dumps(payload)
    env = os.environ.copy()
    if usage_ledger_path is not None:
        env["TEXTSKILL_USAGE_LEDGER_PATH"] = str(usage_ledger_path)
    if usage_context:
        env["TEXTSKILL_USAGE_CONTEXT_JSON"] = json.dumps(usage_context)
    started = time.monotonic()
    try:
        completed = subprocess.run(
            [sys.executable, str(EXTERNAL_EDITOR)],
            input=payload_text,
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
            env=env,
        )
    except subprocess.TimeoutExpired as exc:
        duration = time.monotonic() - started
        stdout = safe_text(exc.stdout)
        stderr = safe_text(exc.stderr) or f"Command timed out after {timeout}s"
        record_one_shot_command_usage(
            usage_ledger_path,
            usage_context or {},
            payload_text=payload_text,
            stdout=stdout,
            stderr=stderr,
            returncode=124,
            duration_seconds=duration,
            timed_out=True,
        )
        raise
    duration = time.monotonic() - started
    record_one_shot_command_usage(
        usage_ledger_path,
        usage_context or {},
        payload_text=payload_text,
        stdout=completed.stdout,
        stderr=completed.stderr,
        returncode=completed.returncode,
        duration_seconds=duration,
        timed_out=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"One-shot skill generation failed: {completed.stderr}")
    parsed = json.loads(completed.stdout)
    if not isinstance(parsed.get("skill_text"), str) or not parsed["skill_text"].strip():
        raise ValueError("One-shot editor did not return non-empty skill_text")
    return parsed


def record_one_shot_command_usage(
    usage_ledger_path: Path | None,
    usage_context: dict[str, Any],
    *,
    payload_text: str,
    stdout: str,
    stderr: str,
    returncode: int,
    duration_seconds: float,
    timed_out: bool,
) -> None:
    output_chars = len(stdout) + len(stderr)
    append_usage_event(
        usage_ledger_path,
        {
            "kind": "optimizer_command",
            "operation": "one_shot_skill",
            "context": usage_context,
            "command": f"{sys.executable} {EXTERNAL_EDITOR}",
            "returncode": returncode,
            "timed_out": timed_out,
            "duration_seconds": duration_seconds,
            "input_chars": len(payload_text),
            "output_chars": output_chars,
            "estimated_prompt_tokens": estimate_tokens_from_chars(len(payload_text)),
            "estimated_completion_tokens": estimate_tokens_from_chars(output_chars),
        },
    )


def build_row(
    seed: str,
    condition: str,
    report: dict[str, Any],
    duration: float,
    run_dir: Path,
    *,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    usage_path = run_dir / "usage_ledger.jsonl"
    experiment_internal_usage_summary = summarize_usage_file(
        usage_path,
        include_kinds=EXPERIMENT_INTERNAL_USAGE_KINDS,
    )
    return {
        "seed": seed,
        "condition": condition,
        "task_accuracy": float(report["pass_rate"]),
        "average_score": float(report["average_score"]),
        "family_macro_accuracy": family_macro_accuracy(report),
        "contract_macro_accuracy": contract_macro_accuracy(report),
        "contract_breakdown": contract_breakdown(report),
        "duration_seconds": duration,
        "run_dir": str(run_dir),
        "usage_ledger_path": str(usage_path),
        "experiment_internal_usage_summary": experiment_internal_usage_summary,
        **(extra or {}),
    }


def family_macro_accuracy(report: dict[str, Any]) -> float:
    by_family: dict[str, list[bool]] = {}
    for result in report.get("results", []):
        metadata = result.get("task", {}).get("metadata", {})
        family = str(metadata.get("benchmark_family") or "unknown")
        by_family.setdefault(family, []).append(bool(result.get("score", {}).get("success")))
    if not by_family:
        return 0.0
    return sum(sum(values) / len(values) for values in by_family.values()) / len(by_family)


def contract_macro_accuracy(report: dict[str, Any]) -> float:
    breakdown = contract_breakdown(report)
    if not breakdown:
        return 0.0
    return sum(item["accuracy"] for item in breakdown.values()) / len(breakdown)


def contract_breakdown(report: dict[str, Any]) -> dict[str, dict[str, float | int]]:
    by_contract: dict[str, dict[str, int]] = {}
    for result in report.get("results", []):
        metadata = result.get("task", {}).get("metadata", {})
        tags = metadata.get("contract_tags") or ["unknown_contract"]
        success = bool(result.get("score", {}).get("success"))
        for tag in dict.fromkeys(str(item) for item in tags):
            bucket = by_contract.setdefault(tag, {"passed": 0, "total": 0})
            bucket["total"] += 1
            if success:
                bucket["passed"] += 1
    return {
        tag: {
            "passed": counts["passed"],
            "total": counts["total"],
            "accuracy": counts["passed"] / counts["total"] if counts["total"] else 0.0,
        }
        for tag, counts in sorted(by_contract.items())
    }


def aggregate_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    aggregate = {}
    for condition in sorted({row["condition"] for row in rows}):
        selected = [row for row in rows if row["condition"] == condition]
        aggregate[condition] = {
            "runs": len(selected),
            "task_accuracy_mean": mean(row["task_accuracy"] for row in selected),
            "task_accuracy_stddev": stddev(row["task_accuracy"] for row in selected),
            "family_macro_mean": mean(row["family_macro_accuracy"] for row in selected),
            "family_macro_stddev": stddev(row["family_macro_accuracy"] for row in selected),
            "contract_macro_mean": mean(
                row.get("contract_macro_accuracy", row.get("family_macro_accuracy", 0.0)) for row in selected
            ),
            "contract_macro_stddev": stddev(
                row.get("contract_macro_accuracy", row.get("family_macro_accuracy", 0.0)) for row in selected
            ),
            "contract_breakdown": aggregate_contract_breakdowns(selected),
            "duration_seconds_total": sum(row["duration_seconds"] for row in selected),
            "experiment_internal_usage_summary": combine_usage_summaries(
                row.get("experiment_internal_usage_summary", {}) for row in selected
            ),
        }
    return aggregate


def aggregate_contract_breakdowns(rows: list[dict[str, Any]]) -> dict[str, dict[str, float | int]]:
    totals: dict[str, dict[str, int]] = {}
    for row in rows:
        for tag, payload in dict(row.get("contract_breakdown") or {}).items():
            if not isinstance(payload, dict):
                continue
            bucket = totals.setdefault(str(tag), {"passed": 0, "total": 0})
            bucket["passed"] += int(payload.get("passed") or 0)
            bucket["total"] += int(payload.get("total") or 0)
    return {
        tag: {
            "passed": counts["passed"],
            "total": counts["total"],
            "accuracy": counts["passed"] / counts["total"] if counts["total"] else 0.0,
        }
        for tag, counts in sorted(totals.items())
    }


def build_usage_report(
    usage_ledger_paths: list[str | Path],
    *,
    aggregate_stdout_chars: int,
) -> dict[str, Any]:
    internal = summarize_usage_files(
        usage_ledger_paths,
        include_kinds=EXPERIMENT_INTERNAL_USAGE_KINDS,
    )
    target = summarize_usage_files(
        usage_ledger_paths,
        include_kinds=TARGET_AGENT_USAGE_KINDS,
    )
    return {
        "primary_scope": "executor_io_proxy",
        "actual_executor_tokens_available": False,
        "executor_io_summary": {
            "actual_tokens_available": False,
            "proxy": "experiment CLI stdout plus operator discipline; Codex context tokens are not visible to this script",
            "aggregate_stdout_chars": aggregate_stdout_chars,
            "estimated_aggregate_stdout_tokens": estimate_tokens_from_chars(aggregate_stdout_chars),
            "recommended_protocol": [
                "inspect summary.json first, not raw result JSON",
                "exclude runs/ from broad rg/find unless explicitly auditing artifacts",
                "use compact helper scripts for failure and usage summaries",
                "cap command output and aggregate before reading large logs",
            ],
        },
        "experiment_internal_usage": internal,
        "excluded_from_primary_usage": {
            "target_agent_kinds": list(TARGET_AGENT_USAGE_KINDS),
            "target_agent_event_count": target["calls"],
            "reason": "Coco/CCR/Kilo token usage is not the current accounting scope.",
        },
    }


def merge_cached_baseline_rows(
    rows: list[dict[str, Any]],
    baseline_summary: Path | None,
    seeds: list[str],
    conditions: set[str],
) -> list[dict[str, Any]]:
    if baseline_summary is None:
        return rows
    cached_rows = load_cached_baseline_rows(baseline_summary, seeds=seeds)
    present_conditions = {str(row.get("condition")) for row in rows}
    imported = [
        row
        for row in cached_rows
        if row.get("condition") not in conditions and row.get("condition") not in present_conditions
    ]
    return imported + rows


def load_cached_baseline_rows(path: Path, *, seeds: list[str]) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    allowed_seeds = set(seeds)
    rows = []
    for row in payload.get("rows") or []:
        if not isinstance(row, dict):
            continue
        if row.get("condition") == "executive":
            continue
        if str(row.get("seed")) not in allowed_seeds:
            continue
        copied = dict(row)
        copied["cached_baseline"] = True
        copied["cached_baseline_source"] = str(path)
        rows.append(copied)
    return rows


def mean(values) -> float:
    items = list(values)
    return statistics.mean(items) if items else 0.0


def stddev(values) -> float:
    items = list(values)
    return statistics.stdev(items) if len(items) > 1 else 0.0


def build_manifest(
    args: argparse.Namespace,
    seeds: list[str],
    *,
    optimizer_model: str,
) -> dict[str, Any]:
    lock = json.loads((BENCHMARK / "test.lock.json").read_text(encoding="utf-8"))
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "experiment_stage": infer_experiment_stage(args),
        "benchmark": "coding-hidden-v2",
        "locked_test_sha256": lock["archive_sha256"],
        "development_only": True,
        "target_harness": "coco",
        "target_model": detect_coco_model(),
        "target_model_policy": "read-local-default-without-override",
        "optimizer_harness": "openai-compatible-external-editor",
        "optimizer_model": optimizer_model,
        "seeds": seeds,
        "seed_workers": args.seed_workers,
        "conditions": sorted(args.conditions),
        "baseline_summary": str(args.baseline_summary) if args.baseline_summary else None,
        "optimizer_config": {
            "epochs": args.epochs,
            "rollout_batch_size": args.rollout_batch_size,
            "reflection_minibatch_size": args.reflection_minibatch_size,
            "learning_rate": args.learning_rate,
            "learning_rate_floor": args.learning_rate_floor,
            "learning_rate_schedule": args.learning_rate_schedule,
            "slow_update_sample_size": args.slow_update_sample_size,
            "enable_slow_update": not args.disable_slow_update,
            "task_retries": args.task_retries,
            "retry_backoff_seconds": args.retry_backoff_seconds,
            "validation_confirmation_rounds": args.validation_confirmation_rounds,
            "validation_required_wins": args.validation_required_wins,
            "validation_mean_delta": args.validation_mean_delta,
            "early_stop_rejection_limit": args.early_stop_rejection_limit,
            "early_stop_validation_score": getattr(args, "early_stop_validation_score", None),
        },
        "development_gate_criteria": {
            "best_baseline_margin": (
                args.development_gate_mean_delta
                if args.development_gate_mean_delta is not None
                else args.validation_mean_delta
            ),
            "min_seed_wins": (
                args.development_gate_required_wins
                if args.development_gate_required_wins is not None
                else args.validation_required_wins
            ),
        },
        "task_limit": args.task_limit or None,
        "task_filter": {
            "train_task_ids": sorted(args.train_task_ids),
            "selection_task_ids": sorted(args.selection_task_ids),
            "task_contracts": sorted(args.task_contracts),
        },
    }


def require_external_editor_config() -> str:
    if not os.environ.get("EXTERNAL_LLM_BASE_URL", "").strip():
        raise ValueError("EXTERNAL_LLM_BASE_URL is required for the external optimizer")
    model = os.environ.get("EXTERNAL_LLM_MODEL", "").strip()
    if not model:
        raise ValueError("EXTERNAL_LLM_MODEL is required for the external optimizer")
    return model


def detect_coco_model(config_path: Path | None = None) -> str:
    path = config_path or Path.home() / ".trae/traecli.yaml"
    if not path.exists():
        return "configured-default"
    lines = path.read_text(encoding="utf-8").splitlines()
    model_indent: int | None = None
    for line in lines:
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


def stable_seed(value: str) -> int:
    return int.from_bytes(value.encode("utf-8"), "little") % (2**31 - 1)


def load_duration(run_dir: Path) -> float:
    path = run_dir / "timing.json"
    if not path.exists():
        return 0.0
    return float(json.loads(path.read_text(encoding="utf-8"))["duration_seconds"])


def safe_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def parse_seeds(raw: str) -> list[str]:
    seeds = [item.strip() for item in raw.split(",") if item.strip()]
    if len(seeds) < 3:
        raise ValueError("At least three seed labels are required")
    return seeds


def parse_conditions(raw: str) -> set[str]:
    allowed = {"no_skill", "human_skill", "one_shot", "executive"}
    conditions = {item.strip() for item in raw.split(",") if item.strip()}
    if not conditions:
        raise ValueError("--conditions must include at least one condition")
    unknown = conditions - allowed
    if unknown:
        raise ValueError(f"Unknown conditions: {', '.join(sorted(unknown))}")
    return conditions


def infer_experiment_stage(args: argparse.Namespace) -> str:
    explicit = str(getattr(args, "experiment_stage", "") or "").strip()
    if explicit:
        return explicit
    if int(args.validation_confirmation_rounds) == 0:
        return "mechanism_smoke"
    if set(args.conditions) == {"executive"}:
        return "full_selection_development"
    return "same_run_baseline_matrix"


def parse_csv_set(raw: str | None) -> set[str]:
    return {item.strip() for item in (raw or "").split(",") if item.strip()}


def filter_tasks(
    tasks: list,
    *,
    task_ids: set[str],
    contract_tags: set[str],
    split_name: str,
) -> list:
    selected = list(tasks)
    if task_ids:
        available = {str(task.id) for task in selected}
        missing = task_ids - available
        if missing:
            raise ValueError(f"Unknown {split_name} task ids: {', '.join(sorted(missing))}")
        selected = [task for task in selected if str(task.id) in task_ids]
    if contract_tags:
        selected = [
            task
            for task in selected
            if contract_tags & {str(tag) for tag in task.metadata.get("contract_tags", [])}
        ]
    if not selected:
        raise ValueError(f"{split_name} task filter selected no tasks")
    return selected


@contextmanager
def patched_env(updates: dict[str, str]) -> Iterator[None]:
    previous = {key: os.environ.get(key) for key in updates}
    os.environ.update(updates)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


class NoopEditor:
    def propose(self, *args, **kwargs):
        return []


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "runs/coding-hidden-v2-matrix")
    parser.add_argument("--seeds", default="seed-a,seed-b,seed-c")
    parser.add_argument("--seed-workers", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--rollout-batch-size", type=int, default=10)
    parser.add_argument("--reflection-minibatch-size", type=int, default=5)
    parser.add_argument("--learning-rate", type=int, default=4)
    parser.add_argument("--learning-rate-floor", type=int, default=2)
    parser.add_argument("--learning-rate-schedule", choices=("constant", "linear", "cosine"), default="cosine")
    parser.add_argument("--rejected-buffer-limit", type=int, default=20)
    parser.add_argument("--slow-update-sample-size", type=int, default=3)
    parser.add_argument("--disable-slow-update", action="store_true")
    parser.add_argument("--task-timeout", type=int, default=360)
    parser.add_argument("--task-retries", type=int, default=2)
    parser.add_argument("--retry-backoff-seconds", type=float, default=5.0)
    parser.add_argument("--validation-confirmation-rounds", type=int, default=2)
    parser.add_argument("--validation-required-wins", type=int, default=2)
    parser.add_argument("--validation-mean-delta", type=float, default=0.05)
    parser.add_argument("--development-gate-required-wins", type=int)
    parser.add_argument("--development-gate-mean-delta", type=float)
    parser.add_argument("--early-stop-rejection-limit", type=int, default=3)
    parser.add_argument("--early-stop-validation-score", type=float)
    parser.add_argument("--editor-timeout", type=int, default=300)
    parser.add_argument("--task-limit", type=int, default=0)
    parser.add_argument("--train-task-ids", default="")
    parser.add_argument("--selection-task-ids", default="")
    parser.add_argument("--task-contracts", default="")
    parser.add_argument("--conditions", default="no_skill,human_skill,one_shot,executive")
    parser.add_argument("--baseline-summary", type=Path)
    parser.add_argument(
        "--experiment-stage",
        choices=("mechanism_smoke", "full_selection_development", "same_run_baseline_matrix"),
        default="",
    )
    parser.add_argument("--resume", action="store_true")
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
