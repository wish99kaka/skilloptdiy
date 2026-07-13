#!/usr/bin/env python3
"""Build a cached baseline summary for a targeted coding-hidden-v2 selection subset."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from textskill_optimizer.io import write_json
from work.development_gate import build_development_gate
from work.run_coding_hidden_v2_matrix import (
    aggregate_rows,
    contract_breakdown,
    contract_macro_accuracy,
    family_macro_accuracy,
    parse_csv_set,
)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    task_ids = parse_csv_set(args.selection_task_ids)
    if not task_ids:
        raise ValueError("--selection-task-ids is required")
    seeds = parse_required_csv(args.seeds, "--seeds")
    conditions = parse_required_csv(args.conditions, "--conditions")
    summary = build_targeted_baseline_summary(
        args.source_run_dir,
        selection_task_ids=task_ids,
        seeds=seeds,
        conditions=conditions,
    )
    write_json(args.out, summary)
    print(f"targeted_baseline_summary={args.out}")
    return 0


def build_targeted_baseline_summary(
    source_run_dir: Path,
    *,
    selection_task_ids: set[str],
    seeds: list[str],
    conditions: list[str],
) -> dict[str, Any]:
    rows = []
    for seed in seeds:
        for condition in conditions:
            report_path = source_run_dir / seed / condition / "selection.json"
            if not report_path.exists():
                raise FileNotFoundError(f"Missing source report: {report_path}")
            report = filter_report(load_json(report_path), selection_task_ids=selection_task_ids)
            rows.append(build_cached_row(seed, condition, report, report_path))
    aggregate = aggregate_rows(rows)
    return {
        "manifest": {
            "created_by": "work/build_coding_hidden_v2_targeted_baseline.py",
            "source_run_dir": str(source_run_dir),
            "selection_task_ids": sorted(selection_task_ids),
            "seeds": seeds,
            "conditions": conditions,
            "cached_baseline_subset": True,
        },
        "rows": rows,
        "aggregate": aggregate,
        "development_gate": build_development_gate(rows, aggregate, {}),
        "locked_test_recommended": False,
    }


def filter_report(report: dict[str, Any], *, selection_task_ids: set[str]) -> dict[str, Any]:
    results = [
        result
        for result in report.get("results", [])
        if str(result.get("task", {}).get("id")) in selection_task_ids
    ]
    found = {str(result.get("task", {}).get("id")) for result in results}
    missing = selection_task_ids - found
    if missing:
        raise ValueError(f"Source report is missing selected task ids: {', '.join(sorted(missing))}")
    return {
        "name": f"{report.get('name', 'selection')}:targeted-subset",
        "results": results,
        "average_score": mean_score(results),
        "pass_rate": mean_success(results),
    }


def build_cached_row(seed: str, condition: str, report: dict[str, Any], report_path: Path) -> dict[str, Any]:
    task_ids = [str(result.get("task", {}).get("id")) for result in report.get("results", [])]
    return {
        "seed": seed,
        "condition": condition,
        "task_accuracy": float(report["pass_rate"]),
        "average_score": float(report["average_score"]),
        "family_macro_accuracy": family_macro_accuracy(report),
        "contract_macro_accuracy": contract_macro_accuracy(report),
        "contract_breakdown": contract_breakdown(report),
        "duration_seconds": 0.0,
        "run_dir": str(report_path.parent),
        "usage_ledger_path": str(report_path.parent / "usage_ledger.jsonl"),
        "experiment_internal_usage_summary": {},
        "source_report_path": str(report_path),
        "targeted_task_ids": task_ids,
    }


def mean_score(results: list[dict[str, Any]]) -> float:
    if not results:
        return 0.0
    return sum(float(result.get("score", {}).get("value") or 0.0) for result in results) / len(results)


def mean_success(results: list[dict[str, Any]]) -> float:
    if not results:
        return 0.0
    return sum(1 for result in results if result.get("score", {}).get("success")) / len(results)


def parse_required_csv(raw: str, label: str) -> list[str]:
    items = [item.strip() for item in raw.split(",") if item.strip()]
    if not items:
        raise ValueError(f"{label} must not be empty")
    return items


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--selection-task-ids", required=True)
    parser.add_argument("--seeds", default="seed-a,seed-b,seed-c")
    parser.add_argument("--conditions", default="no_skill,human_skill,one_shot")
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
