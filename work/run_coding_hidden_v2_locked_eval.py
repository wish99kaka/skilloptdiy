#!/usr/bin/env python3
"""Evaluate one selected skill on the task file injected by locked_eval."""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "work"))

from textskill_optimizer.io import load_tasks_jsonl, load_text, write_json
from textskill_optimizer.plugins.coding import CodingScorer
from work.run_coco_hidden_eval import build_coco_tasks
from work.run_coding_hidden_v2_matrix import (
    build_baseline_evaluator,
    contract_breakdown,
    contract_macro_accuracy,
    family_macro_accuracy,
)


COCO_WRAPPER = ROOT / "examples/coding/coco_agent_wrapper.py"
LOCKED_TASK_COUNT = 20


def locked_tasks_path(environ: Mapping[str, str] | None = None) -> Path:
    env = os.environ if environ is None else environ
    raw = env.get("CROSS_AGENT_TASKS", "").strip()
    if not raw:
        raise RuntimeError("CROSS_AGENT_TASKS is required; run through textskill_optimizer.locked_eval")
    return Path(raw)


def validate_locked_task_count(tasks: Sequence[Any]) -> None:
    if len(tasks) != LOCKED_TASK_COUNT:
        raise ValueError(f"Locked evaluation expected {LOCKED_TASK_COUNT} tasks, found {len(tasks)}")


def build_locked_result(
    report: dict[str, Any],
    skill_path: str | Path,
    *,
    duration_seconds: float,
) -> dict[str, Any]:
    skill = Path(skill_path)
    skill_bytes = skill.read_bytes()
    task_results = []
    for item in report.get("results") or []:
        task = item.get("task") or {}
        metadata = task.get("metadata") or {}
        score = item.get("score") or {}
        task_results.append(
            {
                "task_id": str(task.get("id") or ""),
                "family": str(metadata.get("benchmark_family") or "unknown"),
                "contracts": list(dict.fromkeys(str(tag) for tag in metadata.get("contract_tags") or [])),
                "score": float(score.get("value") or 0.0),
                "success": bool(score.get("success")),
            }
        )
    return {
        "schema_version": 1,
        "status": "complete",
        "skill_path": str(skill),
        "skill_sha256": hashlib.sha256(skill_bytes).hexdigest(),
        "skill_bytes": len(skill_bytes),
        "task_count": len(report.get("results") or []),
        "task_accuracy": float(report.get("pass_rate") or 0.0),
        "average_score": float(report.get("average_score") or 0.0),
        "family_macro_accuracy": family_macro_accuracy(report),
        "contract_macro_accuracy": contract_macro_accuracy(report),
        "contract_breakdown": contract_breakdown(report),
        "duration_seconds": duration_seconds,
        "task_results": task_results,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skill", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--task-timeout", type=int, default=360)
    parser.add_argument("--task-retries", type=int, default=1)
    parser.add_argument("--retry-backoff-seconds", type=float, default=5.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.out.exists():
        raise FileExistsError(f"Refusing to overwrite locked result: {args.out}")
    if not args.skill.is_file():
        raise FileNotFoundError(f"Selected skill is missing: {args.skill}")

    task_file = locked_tasks_path()
    if not task_file.is_file():
        raise FileNotFoundError(f"Injected locked task file is missing: {task_file}")
    wrapped_tasks = build_coco_tasks(
        task_file,
        COCO_WRAPPER,
        task_limit=0,
        timeout_seconds=args.task_timeout,
    )
    tasks = load_tasks_jsonl(wrapped_tasks)
    validate_locked_task_count(tasks)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    usage_path = args.out.parent / "locked_usage_ledger.jsonl"
    evaluator = build_baseline_evaluator(
        args,
        CodingScorer(),
        usage_path=usage_path,
        usage_context={"benchmark": "coding-hidden-v2", "condition": "locked_test"},
    )
    started = time.monotonic()
    report = evaluator.evaluate(load_text(args.skill), tasks, name="locked:test").to_dict()
    duration = time.monotonic() - started
    payload = build_locked_result(report, args.skill, duration_seconds=duration)
    payload["locked_task_file_sha256"] = hashlib.sha256(task_file.read_bytes()).hexdigest()
    payload["usage_ledger_path"] = str(usage_path)
    write_json(args.out, payload)
    print(
        "locked_eval_result "
        f"status=complete tasks={payload['task_count']} "
        f"task_accuracy={payload['task_accuracy']:.4f} "
        f"family_macro={payload['family_macro_accuracy']:.4f} "
        f"contract_macro={payload['contract_macro_accuracy']:.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
