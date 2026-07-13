#!/usr/bin/env python3
"""Print a compact status summary for an existing SkillOpt run directory.

The script is artifact-only: it does not inspect system processes, call agents,
or rerun scorers. It is intended for low-token progress checks while a run is
active or right after it completes.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    status = build_compact_status(args.run_dir, top_gates=args.top_gates)
    if args.json:
        print(json.dumps(status, sort_keys=True))
    else:
        print(render_text(status))
    return 0


def build_compact_status(run_dir: str | Path, *, top_gates: int = 5) -> dict[str, Any]:
    run_path = Path(run_dir)
    summary = load_json_if_exists(run_path / "summary.json")
    runner_report = load_json_if_exists(run_path / "runner_report.json")
    runner_execution = load_json_if_exists(run_path / "runner_execution.json")
    smoke_gate = load_json_if_exists(run_path / "smoke_gate_report.json")
    failure_delta = load_json_if_exists(run_path / "failure_delta_report.json")
    seeds = [seed_status(seed_dir) for seed_dir in sorted(run_path.glob("seed-*")) if seed_dir.is_dir()]
    slow_gates = sorted(
        [gate for seed in seeds for gate in seed["gate_durations"]],
        key=lambda item: (-float(item["seconds"]), item["seed"], item["candidate"]),
    )[:top_gates]
    artifact_span = artifact_span_seconds(run_path)
    summary_usage = nested_dict(summary, "aggregate", "executive", "experiment_internal_usage_summary")
    return {
        "run_dir": str(run_path),
        "summary_present": bool(summary),
        "runner_status": runner_report.get("status"),
        "runner_duration_seconds": runner_execution.get("duration_seconds"),
        "runner_returncode": runner_execution.get("returncode"),
        "smoke_status": smoke_gate.get("status"),
        "smoke_reason": smoke_gate.get("reason"),
        "contract_effect_status": nested_dict(smoke_gate, "contract_effect_audit").get("status"),
        "contract_effect": {
            "protected_regression_count": nested_dict(smoke_gate, "contract_effect_audit").get(
                "protected_regression_count"
            ),
            "rejected_protected_regression_count": nested_dict(smoke_gate, "contract_effect_audit").get(
                "rejected_protected_regression_count"
            ),
            "effective_step_count": nested_dict(smoke_gate, "contract_effect_audit").get("effective_step_count"),
            "accepted_required_step_count": nested_dict(smoke_gate, "contract_effect_audit").get(
                "accepted_required_step_count"
            ),
            "rejected_required_step_count": nested_dict(smoke_gate, "contract_effect_audit").get(
                "rejected_required_step_count"
            ),
        },
        "failure_blocker": nested_dict(failure_delta, "diagnosis").get("primary_blocker"),
        "development_gate": compact_development_gate(summary, smoke_gate, runner_report),
        "usage": {
            "optimizer_actual_total_tokens": summary_usage.get("actual_total_tokens"),
            "optimizer_prompt_tokens": summary_usage.get("actual_prompt_tokens"),
            "optimizer_completion_tokens": summary_usage.get("actual_completion_tokens"),
            "optimizer_api_seconds": nested_dict(summary_usage, "by_kind", "optimizer_model_api").get(
                "duration_seconds_total"
            ),
            "optimizer_command_seconds": nested_dict(summary_usage, "by_kind", "optimizer_command").get(
                "duration_seconds_total"
            ),
        },
        "artifact_span_seconds": artifact_span,
        "seeds": seeds,
        "slow_gates": slow_gates,
    }


def seed_status(seed_dir: Path) -> dict[str, Any]:
    executive_dir = seed_dir / "executive"
    proposals = executive_dir / "proposals.jsonl"
    timing_events = load_jsonl(executive_dir / "timing_events.jsonl")
    usage = usage_from_ledger(executive_dir / "usage_ledger.jsonl")
    return {
        "seed": seed_dir.name,
        "result": (executive_dir / "result.json").exists(),
        "checkpoint": (executive_dir / "result_checkpoint.json").exists(),
        "initial": (executive_dir / "selection_initial.json").exists(),
        "proposal_records": count_lines(proposals),
        "candidates": len(list(executive_dir.glob("candidate_*.md"))),
        "gates": len(list(executive_dir.glob("selection_*_gate.json"))),
        "usage_ledger_records": count_lines(executive_dir / "usage_ledger.jsonl"),
        "timing_event_records": len(timing_events),
        "optimizer_tokens": usage["optimizer_tokens"],
        "optimizer_api_seconds": usage["optimizer_api_seconds"],
        "last_artifact": latest_artifact_name(executive_dir),
        "gate_durations": gate_durations(seed_dir.name, executive_dir, timing_events),
    }


def usage_from_ledger(path: Path) -> dict[str, Any]:
    tokens = 0
    seconds = 0.0
    if not path.exists():
        return {"optimizer_tokens": 0, "optimizer_api_seconds": 0.0}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("kind") != "optimizer_model_api":
            continue
        tokens += int(record.get("actual_total_tokens") or 0)
        seconds += float(record.get("duration_seconds") or 0.0)
    return {"optimizer_tokens": tokens, "optimizer_api_seconds": seconds}


def gate_durations(seed: str, executive_dir: Path, timing_events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    event_durations = [
        {
            "seed": seed,
            "candidate": str(event.get("candidate_name") or ""),
            "seconds": round(float(event.get("duration_seconds") or 0.0), 3),
            "source": "timing_events",
        }
        for event in timing_events
        if event.get("event") == "validation_finished" and event.get("candidate_name")
    ]
    if event_durations:
        return event_durations

    output = []
    for candidate in sorted(executive_dir.glob("candidate_*.md")):
        suffix = candidate.name[len("candidate_") : -len(".md")]
        gate = executive_dir / f"selection_{suffix}_gate.json"
        if not gate.exists():
            continue
        seconds = gate.stat().st_mtime - candidate.stat().st_mtime
        output.append(
            {
                "seed": seed,
                "candidate": suffix,
                "seconds": round(seconds, 3),
                "source": "mtime_gap",
            }
        )
    return output


def compact_development_gate(*sources: dict[str, Any]) -> dict[str, Any]:
    gate: dict[str, Any] = {}
    for source in sources:
        candidate = source.get("development_gate") if isinstance(source.get("development_gate"), dict) else {}
        if candidate:
            gate = dict(candidate)
            break
    criteria = gate.get("criteria") if isinstance(gate.get("criteria"), dict) else {}
    required_contract_macro_delta = gate.get("required_contract_macro_delta")
    if required_contract_macro_delta is None:
        required_contract_macro_delta = criteria.get("contract_macro_margin")
    return {
        "passed": gate.get("passed"),
        "executive_mean": gate.get("executive_mean"),
        "mean_delta": gate.get("mean_delta"),
        "seed_wins_vs_best_baseline": gate.get("seed_wins_vs_best_baseline"),
        "required_seed_wins": gate.get("required_seed_wins") or criteria.get("min_seed_wins"),
        "contract_macro_delta": gate.get("contract_macro_delta"),
        "required_contract_macro_delta": required_contract_macro_delta,
        "critical_contract_regression_count": len(gate.get("critical_contract_regressions") or []),
        "blocked_reason": gate.get("blocked_reason"),
    }


def artifact_span_seconds(run_path: Path) -> float | None:
    files = [path for path in run_path.glob("seed-*/executive/*") if path.is_file()]
    if not files:
        return None
    mtimes = [path.stat().st_mtime for path in files]
    return round(max(mtimes) - min(mtimes), 3)


def latest_artifact_name(path: Path) -> str | None:
    files = [item for item in path.glob("*") if item.is_file()]
    if not files:
        return None
    latest = max(files, key=lambda item: item.stat().st_mtime)
    return latest.name


def count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    return len(path.read_text(encoding="utf-8").splitlines())


def render_text(status: dict[str, Any]) -> str:
    gate = dict(status.get("development_gate") or {})
    usage = dict(status.get("usage") or {})
    effect = dict(status.get("contract_effect") or {})
    lines = [
        "skillopt_status",
        (
            f"run={status.get('run_dir')} summary={int(bool(status.get('summary_present')))} "
            f"runner={status.get('runner_status')} smoke={status.get('smoke_status')} "
            f"effect={status.get('contract_effect_status')} blocker={status.get('failure_blocker')} "
            f"span_min={minutes(status.get('artifact_span_seconds'))}"
        ),
        (
            f"gate_passed={gate.get('passed')} mean={gate.get('executive_mean')} "
            f"delta={gate.get('mean_delta')} wins={gate.get('seed_wins_vs_best_baseline')}/{gate.get('required_seed_wins')} "
            f"contract_delta={gate.get('contract_macro_delta')}/{gate.get('required_contract_macro_delta')} "
            f"critical_regressions={gate.get('critical_contract_regression_count')}"
        ),
        (
            f"effect_accepted={effect.get('effective_step_count')}/{effect.get('accepted_required_step_count')} "
            f"accepted_protected_regressions={effect.get('protected_regression_count')} "
            f"rejected_required={effect.get('rejected_required_step_count')} "
            f"rejected_protected_regressions={effect.get('rejected_protected_regression_count')}"
        ),
        (
            f"optimizer_tokens={usage.get('optimizer_actual_total_tokens')} "
            f"prompt={usage.get('optimizer_prompt_tokens')} completion={usage.get('optimizer_completion_tokens')} "
            f"api_min={minutes(usage.get('optimizer_api_seconds'))} command_min={minutes(usage.get('optimizer_command_seconds'))}"
        ),
    ]
    for seed in status.get("seeds", []):
        lines.append(
            f"{seed['seed']} result={int(seed['result'])} props={seed['proposal_records']} "
            f"cand={seed['candidates']} gates={seed['gates']} tokens={seed['optimizer_tokens']} "
            f"api_min={minutes(seed['optimizer_api_seconds'])} timing={seed['timing_event_records']} "
            f"last={seed['last_artifact']}"
        )
    slow = status.get("slow_gates") or []
    if slow:
        formatted = ", ".join(
            f"{item['seed']}/{item['candidate']}={minutes(item['seconds'])}m" for item in slow
        )
        lines.append(f"slow_gates {formatted}")
    return "\n".join(lines)


def minutes(value: Any) -> str:
    if value is None:
        return "NA"
    return f"{float(value) / 60.0:.1f}"


def nested_dict(payload: dict[str, Any], *keys: str) -> dict[str, Any]:
    current: Any = payload
    for key in keys:
        if not isinstance(current, dict):
            return {}
        current = current.get(key)
    return current if isinstance(current, dict) else {}


def load_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if isinstance(payload, dict):
            records.append(payload)
    return records


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--top-gates", type=int, default=5)
    parser.add_argument("--json", action="store_true", help="Print compact JSON instead of text")
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
