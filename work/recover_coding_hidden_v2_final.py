#!/usr/bin/env python3
"""Recover transient Coco failures in completed coding-hidden-v2 final reports."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "work"))

from textskill_optimizer.executive_optimizer import ExecutiveOptimizerConfig, ExecutiveSkillOptimizer
from textskill_optimizer.io import load_tasks_jsonl, write_json
from textskill_optimizer.models import EvaluationReport, Score, Task, TaskOutput, TaskResult
from textskill_optimizer.plugins.coding import (
    CodingRunner,
    CodingScorer,
    coding_retryable_anomaly_reasons,
)
from work.run_coco_hidden_eval import build_coco_tasks
from work.run_coding_hidden_v2_matrix import (
    aggregate_rows,
    contract_breakdown,
    contract_macro_accuracy,
    detect_coco_model,
    family_macro_accuracy,
)


BENCHMARK = ROOT / "examples/coding-hidden-v2"
COCO_WRAPPER = ROOT / "examples/coding/coco_agent_wrapper.py"


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary_path = args.run_dir / "summary.json"
    if not summary_path.exists():
        raise ValueError(f"Missing completed matrix summary: {summary_path}")
    original_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    source_target_model = str(original_summary.get("manifest", {}).get("target_model") or "unknown")
    if args.provenance_only:
        health_path = args.run_dir / "coco_health.json"
        if not health_path.exists():
            raise ValueError(f"Missing recovery-time health record: {health_path}")
        health = json.loads(health_path.read_text(encoding="utf-8"))
        recovery_target_model = str(health.get("model") or "unknown")
        recovered_summary = label_existing_recovery_provenance(
            args.run_dir,
            args.seeds,
            original_summary,
            recovery_target_model=recovery_target_model,
        )
        write_json(args.run_dir / "summary_recovered.json", recovered_summary)
        print(json.dumps(recovered_summary["recovery"], indent=2, sort_keys=True))
        return 0

    health = inspect_coco_model_health(detect_coco_model(), args.coco_bin)
    write_json(args.run_dir / "coco_health.json", health)
    if health["status"] == "blocked":
        print(f"[recovery] blocked: {health['reason']}", file=sys.stderr)
        return 3

    canonical_path = build_coco_tasks(
        BENCHMARK / "selection.jsonl",
        COCO_WRAPPER,
        task_limit=0,
        timeout_seconds=args.task_timeout,
    )
    tasks = {task.id: task for task in load_tasks_jsonl(canonical_path)}
    evaluator = ExecutiveSkillOptimizer(
        CodingRunner(),
        CodingScorer(),
        editor=object(),
        config=ExecutiveOptimizerConfig(
            epochs=1,
            enable_slow_update=False,
            task_retry_limit=args.task_retries,
            task_retry_backoff_seconds=args.retry_backoff_seconds,
        ),
        retry_detector=coding_retryable_anomaly_reasons,
    )

    print(
        f"[recovery] Coco model={detect_coco_model()} (read-only local default); "
        f"task_retries={args.task_retries}",
        file=sys.stderr,
        flush=True,
    )
    recovered_reports: dict[str, dict[str, Any]] = {}
    recovery_durations: dict[str, float] = {}
    for seed in args.seeds:
        report, duration = recover_seed_final(
            args.run_dir,
            seed,
            tasks,
            evaluator,
            source_target_model=source_target_model,
            recovery_target_model=str(health["model"]),
        )
        recovered_reports[seed] = report
        recovery_durations[seed] = duration

    recovered_summary = build_recovered_summary(
        original_summary,
        recovered_reports,
        recovery_durations,
        task_retries=args.task_retries,
        retry_backoff_seconds=args.retry_backoff_seconds,
        recovery_target_model=str(health["model"]),
    )
    write_json(args.run_dir / "summary_recovered.json", recovered_summary)
    print(json.dumps(recovered_summary["aggregate"], indent=2, sort_keys=True))
    return 0


def recover_seed_final(
    run_dir: Path,
    seed: str,
    tasks: dict[str, Task],
    evaluator: ExecutiveSkillOptimizer,
    *,
    source_target_model: str,
    recovery_target_model: str,
) -> tuple[dict[str, Any], float]:
    executive_dir = run_dir / seed / "executive"
    result_path = executive_dir / "result.json"
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    source_report = payload["final_validation_report"]
    anomaly_ids = retryable_task_ids(source_report)
    recovery_path = executive_dir / "final_validation_recovery.json"
    if recovery_path.exists():
        recovery = json.loads(recovery_path.read_text(encoding="utf-8"))
    else:
        recovery = {
            "seed": seed,
            "source_result": str(result_path),
            "source_report": source_report,
            "anomaly_task_ids": anomaly_ids,
            "replacements": {},
            "duration_seconds": 0.0,
        }
    recovery["source_target_model"] = source_target_model
    recovery["recovery_target_model"] = recovery_target_model
    recovery["comparability"] = (
        "same_target_model"
        if source_target_model == recovery_target_model
        else "cross_model_transfer_only"
    )

    replacements = recovery.setdefault("replacements", {})
    skill_text = str(payload["best_skill_text"])
    for task_id in anomaly_ids:
        if task_id in replacements:
            continue
        task = tasks.get(task_id)
        if task is None:
            raise ValueError(f"Missing canonical selection task: {task_id}")
        print(f"[recovery] {seed} task={task_id}", file=sys.stderr, flush=True)
        started = time.monotonic()
        report = evaluator.evaluate(
            skill_text,
            [task],
            name=f"selection:final:recovery:{seed}:{task_id}",
        )
        elapsed = time.monotonic() - started
        replacements[task_id] = report.results[0].to_dict()
        recovery["duration_seconds"] = float(recovery.get("duration_seconds", 0.0)) + elapsed
        write_json(recovery_path, recovery)

    recovered_report = merge_report_replacements(source_report, replacements)
    recovery["completed_at"] = datetime.now(timezone.utc).isoformat()
    recovery["recovered_report"] = recovered_report
    write_json(recovery_path, recovery)
    recovered_result = {
        **payload,
        "final_validation_report": recovered_report,
        "final_validation_recovery": {
            "source_target_model": source_target_model,
            "recovery_target_model": recovery_target_model,
            "comparability": recovery["comparability"],
        },
    }
    write_json(executive_dir / "result_recovered.json", recovered_result)
    return recovered_report, float(recovery.get("duration_seconds", 0.0))


def label_existing_recovery_provenance(
    run_dir: Path,
    seeds: Sequence[str],
    original_summary: dict[str, Any],
    *,
    recovery_target_model: str,
) -> dict[str, Any]:
    """Attach immutable source/recovery model labels without rerunning target agents."""
    existing_summary_path = run_dir / "summary_recovered.json"
    existing_recovery = {}
    if existing_summary_path.exists():
        existing_summary = json.loads(existing_summary_path.read_text(encoding="utf-8"))
        existing_recovery = dict(existing_summary.get("recovery") or {})
    source_target_model = str(original_summary.get("manifest", {}).get("target_model") or "unknown")
    comparability = (
        "same_target_model"
        if source_target_model == recovery_target_model
        else "cross_model_transfer_only"
    )
    reports: dict[str, dict[str, Any]] = {}
    durations: dict[str, float] = {}
    for seed in seeds:
        executive_dir = run_dir / seed / "executive"
        recovery_path = executive_dir / "final_validation_recovery.json"
        if not recovery_path.exists():
            raise ValueError(f"Missing completed recovery artifact: {recovery_path}")
        recovery = json.loads(recovery_path.read_text(encoding="utf-8"))
        report = recovery.get("recovered_report")
        if not isinstance(report, dict):
            raise ValueError(f"Missing recovered_report in {recovery_path}")
        recovery.update(
            {
                "source_target_model": source_target_model,
                "recovery_target_model": recovery_target_model,
                "comparability": comparability,
            }
        )
        write_json(recovery_path, recovery)

        recovered_result_path = executive_dir / "result_recovered.json"
        source_result_path = executive_dir / "result.json"
        result_path = recovered_result_path if recovered_result_path.exists() else source_result_path
        result = json.loads(result_path.read_text(encoding="utf-8"))
        result["final_validation_report"] = report
        result["final_validation_recovery"] = {
            "source_target_model": source_target_model,
            "recovery_target_model": recovery_target_model,
            "comparability": comparability,
        }
        write_json(recovered_result_path, result)
        reports[str(seed)] = report
        durations[str(seed)] = float(recovery.get("duration_seconds", 0.0))

    return build_recovered_summary(
        original_summary,
        reports,
        durations,
        task_retries=int(existing_recovery.get("task_retries", 0)),
        retry_backoff_seconds=float(existing_recovery.get("retry_backoff_seconds", 0.0)),
        recovery_target_model=recovery_target_model,
    )


def retryable_task_ids(report: dict[str, Any]) -> list[str]:
    return [
        result.task.id
        for payload in report.get("results", [])
        if coding_retryable_anomaly_reasons(result := task_result_from_dict(payload))
    ]


def task_result_from_dict(payload: dict[str, Any]) -> TaskResult:
    output_payload = payload.get("output") or {}
    score_payload = payload.get("score") or {}
    return TaskResult(
        task=Task.from_dict(payload.get("task") or {}),
        output=TaskOutput(
            value=output_payload.get("value"),
            trace=list(output_payload.get("trace") or []),
            metadata=dict(output_payload.get("metadata") or {}),
        ),
        score=Score(
            value=float(score_payload.get("value", 0.0)),
            success=bool(score_payload.get("success")),
            message=str(score_payload.get("message") or ""),
            metadata=dict(score_payload.get("metadata") or {}),
        ),
    )


def merge_report_replacements(
    source_report: dict[str, Any],
    replacements: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    results = [
        replacements.get(str(payload.get("task", {}).get("id")), payload)
        for payload in source_report.get("results", [])
    ]
    report = EvaluationReport(
        name="selection:final:recovered",
        results=[task_result_from_dict(payload) for payload in results],
    )
    return report.to_dict()


def build_recovered_summary(
    original: dict[str, Any],
    reports: dict[str, dict[str, Any]],
    durations: dict[str, float],
    *,
    task_retries: int,
    retry_backoff_seconds: float,
    recovery_target_model: str,
) -> dict[str, Any]:
    rows = []
    for original_row in original.get("rows", []):
        row = dict(original_row)
        if row.get("condition") == "executive":
            seed = str(row["seed"])
            report = reports[seed]
            row.update(
                {
                    "task_accuracy": float(report["pass_rate"]),
                    "average_score": float(report["average_score"]),
                    "family_macro_accuracy": family_macro_accuracy(report),
                    "contract_macro_accuracy": contract_macro_accuracy(report),
                    "contract_breakdown": contract_breakdown(report),
                    "duration_seconds": float(row.get("duration_seconds", 0.0)) + durations.get(seed, 0.0),
                    "recovery_duration_seconds": durations.get(seed, 0.0),
                }
            )
        rows.append(row)
    recovery = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scope": "retryable anomalies in executive final reports only",
        "task_retries": task_retries,
        "retry_backoff_seconds": retry_backoff_seconds,
        "seed_workers": 1,
        "source_summary": "summary.json",
        "source_target_model": original.get("manifest", {}).get("target_model"),
        "recovery_target_model": recovery_target_model,
    }
    recovery["comparability"] = (
        "same_target_model"
        if recovery["source_target_model"] == recovery_target_model
        else "cross_model_transfer_only"
    )
    return {
        "manifest": original.get("manifest", {}),
        "recovery": recovery,
        "rows": rows,
        "aggregate": aggregate_rows(rows),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=ROOT / "runs/coding-hidden-v2-matrix")
    parser.add_argument("--seeds", type=lambda raw: [item.strip() for item in raw.split(",") if item.strip()], default=["seed-a", "seed-b", "seed-c"])
    parser.add_argument("--task-timeout", type=int, default=420)
    parser.add_argument("--task-retries", type=int, default=2)
    parser.add_argument("--retry-backoff-seconds", type=float, default=10.0)
    parser.add_argument("--coco-bin", default="/Users/bytedance/.local/bin/coco")
    parser.add_argument(
        "--provenance-only",
        action="store_true",
        help="Relabel completed recovery artifacts from the recorded health cache without rerunning agents.",
    )
    return parser


def inspect_coco_model_health(model_name: str, coco_bin: str) -> dict[str, Any]:
    completed = subprocess.run(
        [coco_bin, "models", "--json"],
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )
    checked_at = datetime.now(timezone.utc).isoformat()
    if completed.returncode != 0:
        return {
            "checked_at": checked_at,
            "model": model_name,
            "status": "unknown",
            "reason": "coco_models_command_failed",
            "returncode": completed.returncode,
        }
    try:
        models = json.loads(completed.stdout)
    except json.JSONDecodeError:
        return {
            "checked_at": checked_at,
            "model": model_name,
            "status": "unknown",
            "reason": "coco_models_invalid_json",
        }
    selected = next(
        (item for item in models if isinstance(item, dict) and item.get("name") == model_name),
        None,
    )
    if selected is None:
        return {
            "checked_at": checked_at,
            "model": model_name,
            "status": "blocked",
            "reason": "configured_model_not_available",
        }
    description = str(selected.get("description") or "")
    if "quota: 100% used" in description.casefold():
        status = "blocked"
        reason = "configured_model_quota_exhausted"
    else:
        status = "available"
        reason = "model_list_reports_available"
    return {
        "checked_at": checked_at,
        "model": model_name,
        "status": status,
        "reason": reason,
        "description": description,
    }


if __name__ == "__main__":
    raise SystemExit(main())
