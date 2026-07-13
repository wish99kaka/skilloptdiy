#!/usr/bin/env python3
"""Recover a partial coding-hidden-v2 matrix summary from interrupted run artifacts."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from work.run_coding_hidden_v2_matrix import aggregate_rows, build_row, build_usage_report


CONDITIONS = ("no_skill", "human_skill", "one_shot")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.write_executive_results:
        written = write_recovered_executive_results(
            args.run_dir,
            early_stop_rejection_limit=args.early_stop_rejection_limit,
        )
        print(json.dumps({"written_results": written}, indent=2, sort_keys=True))
    summary = recover_partial_summary(args.run_dir)
    write_json(args.out or (args.run_dir / "summary_recovered_partial.json"), summary)
    print(json.dumps(summary["recovery"], indent=2, sort_keys=True))
    return 0


def recover_partial_summary(run_dir: Path) -> dict[str, Any]:
    manifest = load_manifest(run_dir)
    seeds = list(manifest.get("seeds") or discover_seeds(run_dir))
    rows: list[dict[str, Any]] = []
    usage_paths: list[Path] = []
    partial_executive_seeds: list[str] = []

    for seed in seeds:
        seed_dir = run_dir / str(seed)
        for condition in CONDITIONS:
            condition_dir = seed_dir / condition
            report_path = condition_dir / "selection.json"
            if not report_path.exists():
                continue
            rows.append(
                build_row(
                    str(seed),
                    condition,
                    load_json(report_path),
                    load_duration(condition_dir),
                    condition_dir,
                )
            )
            usage_paths.append(condition_dir / "usage_ledger.jsonl")

        executive = recover_executive_row(seed_dir / "executive", str(seed))
        if executive is not None:
            rows.append(executive)
            usage_paths.append(seed_dir / "executive" / "usage_ledger.jsonl")
            if executive.get("partial_recovery"):
                partial_executive_seeds.append(str(seed))

    return {
        "manifest": manifest,
        "recovery": {
            "created_at_unix": time.time(),
            "partial": True,
            "source": "interrupted coding-hidden-v2 matrix artifacts",
            "partial_executive_seeds": partial_executive_seeds,
            "complete_rows": len(rows),
            "locked_test_recommended": False,
            "reason": "partial recovery is diagnostic only; rerun or repair before locked test",
        },
        "rows": rows,
        "aggregate": aggregate_rows(rows),
        "usage": build_usage_report(usage_paths, aggregate_stdout_chars=0),
    }


def recover_executive_row(executive_dir: Path, seed: str) -> dict[str, Any] | None:
    result_path = executive_dir / "result.json"
    if result_path.exists():
        result = load_json(result_path)
        return build_row(
            seed,
            "executive",
            result["final_validation_report"],
            load_duration(executive_dir),
            executive_dir,
            extra={
                "accepted_steps": result.get("accepted_steps"),
                "total_steps": result.get("total_steps"),
                "best_validation_score": result.get("best_validation_score"),
                "partial_recovery": False,
            },
        )

    selected_report_path, accepted_steps = select_best_available_report(executive_dir)
    if selected_report_path is None:
        return None
    report = load_json(selected_report_path)
    gate_files = sorted(executive_dir.glob("selection_*_gate.json"))
    return build_row(
        seed,
        "executive",
        report,
        usage_duration_sum(executive_dir / "usage_ledger.jsonl"),
        executive_dir,
        extra={
            "accepted_steps": accepted_steps,
            "total_steps": len(gate_files),
            "best_validation_score": float(report.get("average_score") or 0.0),
            "partial_recovery": True,
            "partial_recovery_source": selected_report_path.name,
            "partial_recovery_reason": "optimizer run stopped before result.json was written",
        },
    )


def write_recovered_executive_results(
    run_dir: Path,
    *,
    early_stop_rejection_limit: int,
) -> list[str]:
    if early_stop_rejection_limit <= 0:
        raise ValueError("early_stop_rejection_limit must be positive")
    written = []
    for seed in discover_seeds(run_dir):
        executive_dir = run_dir / seed / "executive"
        result_path = executive_dir / "result.json"
        if result_path.exists():
            result = load_json(result_path)
            if str(result.get("stop_reason", "")).startswith("recovered_"):
                write_recovered_timing(executive_dir)
            continue
        selected_report_path, accepted_steps = select_best_available_report(executive_dir)
        if selected_report_path is None:
            continue
        gates = sorted(
            executive_dir.glob("selection_*_gate.json"),
            key=lambda path: path.stat().st_mtime,
        )
        if not reached_early_stop(gates, early_stop_rejection_limit):
            continue
        report = load_json(selected_report_path)
        result = {
            "best_skill_text": (executive_dir / "best_skill.md").read_text(encoding="utf-8")
            if (executive_dir / "best_skill.md").exists()
            else "",
            "best_validation_score": float(report.get("average_score") or 0.0),
            "history": recovered_history(gates),
            "final_validation_report": report,
            "rejected_buffer": recovered_rejected_buffer(gates),
            "meta_skill_text": latest_meta_skill(executive_dir),
            "accepted_steps": accepted_steps,
            "total_steps": len(gates),
            "stop_reason": "recovered_early_stop_validation_rejection_limit",
            "checkpoint": {
                "early_stop_rejection_limit": early_stop_rejection_limit,
                "validation_rejection_streak": early_stop_rejection_limit,
                "recovered_from_artifacts": True,
                "source_report": selected_report_path.name,
            },
        }
        write_json(result_path, result)
        write_json(executive_dir / "result_checkpoint.json", result)
        write_recovered_timing(executive_dir)
        written.append(seed)
    return written


def write_recovered_timing(executive_dir: Path) -> None:
    timing_path = executive_dir / "timing.json"
    if not timing_path.exists():
        write_json(timing_path, {"duration_seconds": usage_duration_sum(executive_dir / "usage_ledger.jsonl")})


def reached_early_stop(gates: list[Path], limit: int) -> bool:
    if len(gates) < limit:
        return False
    recent = gates[-limit:]
    return all(load_json(path).get("accepted") is False for path in recent)


def recovered_history(gates: list[Path]) -> list[dict[str, Any]]:
    history = [
        {
            "epoch": 0,
            "candidate": "initial",
            "accepted": True,
            "validation_score": None,
            "rationale": "Recovered initial skill baseline.",
            "metadata": {"recovered_from_artifacts": True},
        }
    ]
    for index, gate_path in enumerate(gates, 1):
        gate = load_json(gate_path)
        candidate = gate_path.name.removeprefix("selection_").removesuffix("_gate.json")
        history.append(
            {
                "epoch": index,
                "candidate": candidate,
                "accepted": bool(gate.get("accepted")),
                "validation_score": gate.get("candidate_mean"),
                "rationale": "Recovered validation gate evidence.",
                "metadata": {
                    "validation_gate": gate,
                    "recovered_from_artifacts": True,
                },
            }
        )
    return history


def recovered_rejected_buffer(gates: list[Path]) -> list[dict[str, Any]]:
    rejected = []
    for index, gate_path in enumerate(gates, 1):
        gate = load_json(gate_path)
        if gate.get("accepted") is not False:
            continue
        candidate = gate_path.name.removeprefix("selection_").removesuffix("_gate.json")
        rejected.append(
            {
                "epoch": index,
                "candidate": candidate,
                "reason": "validation_gate_rejected",
                "rationale": "Recovered validation gate evidence.",
                "validation_score": gate.get("candidate_mean"),
                "failed_task_ids": failed_task_ids_from_gate(gate),
                "metadata": {
                    "validation_gate": gate,
                    "recovered_from_artifacts": True,
                },
            }
        )
    return rejected


def failed_task_ids_from_gate(gate: dict[str, Any]) -> list[str]:
    report = gate.get("candidate_report") if isinstance(gate.get("candidate_report"), dict) else {}
    results = report.get("results") if isinstance(report.get("results"), list) else []
    failed = []
    for item in results:
        if not isinstance(item, dict):
            continue
        score = item.get("score") if isinstance(item.get("score"), dict) else {}
        if score.get("success"):
            continue
        task = item.get("task") if isinstance(item.get("task"), dict) else {}
        task_id = task.get("id")
        if task_id is not None:
            failed.append(str(task_id))
    return failed


def latest_meta_skill(executive_dir: Path) -> str:
    metas = sorted(executive_dir.glob("meta_skill_epoch_*.md"), key=lambda path: path.stat().st_mtime)
    if metas:
        return metas[-1].read_text(encoding="utf-8")
    initial = executive_dir / "meta_skill_initial.md"
    return initial.read_text(encoding="utf-8") if initial.exists() else ""


def select_best_available_report(executive_dir: Path) -> tuple[Path | None, int]:
    initial = executive_dir / "selection_initial.json"
    selected = initial if initial.exists() else None
    accepted_steps = 0
    for gate_path in sorted(executive_dir.glob("selection_*_gate.json"), key=lambda path: path.stat().st_mtime):
        gate = load_json(gate_path)
        candidate_name = gate_path.name.removeprefix("selection_").removesuffix("_gate.json")
        candidate_report = executive_dir / f"selection_{candidate_name}.json"
        if gate.get("accepted") and candidate_report.exists():
            selected = candidate_report
            accepted_steps += 1
    return selected, accepted_steps


def usage_duration_sum(path: Path) -> float:
    if not path.exists():
        return 0.0
    total = 0.0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        duration = event.get("duration_seconds")
        if isinstance(duration, (int, float)):
            total += float(duration)
    return total


def load_manifest(run_dir: Path) -> dict[str, Any]:
    for name in ("experiment_manifest.json", "runner_manifest.json"):
        path = run_dir / name
        if path.exists():
            payload = load_json(path)
            if name == "runner_manifest.json":
                return dict(payload.get("manifest") or payload)
            return dict(payload)
    return {}


def discover_seeds(run_dir: Path) -> list[str]:
    return sorted(path.name for path in run_dir.glob("seed-*") if path.is_dir())


def load_duration(run_dir: Path) -> float:
    path = run_dir / "timing.json"
    if not path.exists():
        return 0.0
    return float(load_json(path).get("duration_seconds") or 0.0)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--write-executive-results", action="store_true")
    parser.add_argument("--early-stop-rejection-limit", type=int, default=3)
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
