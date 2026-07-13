#!/usr/bin/env python3
"""Gate a low-cost SkillOpt executive smoke run.

The script reads existing artifacts only. It does not call external models,
target agents, or scorers.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from work.experiment_runner import build_runner_report
from work.skillopt_contract_effect_audit import build_contract_effect_audit


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = build_smoke_gate_report(
        args.run_dir,
        min_accepted_steps=args.min_accepted_steps,
    )
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    print(render_compact_summary(report) if args.quiet else text)
    return 0 if report["status"] in {"pass", "stop", "inconclusive"} else 1


def build_smoke_gate_report(
    run_dir: str | Path,
    *,
    min_accepted_steps: int = 1,
) -> dict[str, Any]:
    run_path = Path(run_dir)
    runner_report = load_runner_report(run_path)
    summary_path = run_path / "summary.json"
    summary = load_json(summary_path) if summary_path.exists() else {}
    executive_rows = [
        row for row in summary.get("rows", []) if isinstance(row, dict) and row.get("condition") == "executive"
    ]
    accepted_steps_total = sum(int(row.get("accepted_steps") or 0) for row in executive_rows)
    proposal_audit = summarize_proposal_audits(run_path)
    contract_effect_audit = build_contract_effect_audit(run_path)
    checks = {
        "summary_present": bool(summary),
        "runner_complete": runner_report.get("status") == "complete",
        "no_persistent_anomaly": (
            int(runner_report.get("anomaly_summary", {}).get("persistent_anomaly_count") or 0) == 0
        ),
        "proposal_targeting_audit_passed": proposal_audit["status"] == "pass",
        "contract_effect_audit_passed": contract_effect_audit["status"] == "pass",
        "accepted_step_present": accepted_steps_total >= min_accepted_steps,
        "development_gate_passed": bool(runner_report.get("development_gate", {}).get("passed")),
    }
    status, reason, next_action = decide(checks, proposal_audit, contract_effect_audit)
    return {
        "schema_version": 1,
        "run_dir": str(run_path),
        "status": status,
        "reason": reason,
        "next_action": next_action,
        "scale_up_recommended": status == "pass",
        "checks": checks,
        "min_accepted_steps": min_accepted_steps,
        "accepted_steps_total": accepted_steps_total,
        "executive_row_count": len(executive_rows),
        "proposal_audit": proposal_audit,
        "contract_effect_audit": contract_effect_audit,
        "development_gate": runner_report.get("development_gate", {}),
        "anomaly_summary": runner_report.get("anomaly_summary", {}),
        "artifact_paths": {
            "summary": str(summary_path),
            "runner_report": str(run_path / "runner_report.json"),
            "proposal_logs": [str(path) for path in sorted(run_path.rglob("proposals.jsonl"))],
        },
    }


def decide(
    checks: dict[str, bool],
    proposal_audit: dict[str, Any],
    contract_effect_audit: dict[str, Any],
) -> tuple[str, str, str]:
    if not checks["summary_present"]:
        return "missing_artifacts", "summary.json is missing", "run or recover the smoke experiment first"
    if not checks["runner_complete"]:
        return "stop", "runner report is not complete", "inspect runner_report.json and execution logs"
    if not checks["no_persistent_anomaly"]:
        return "stop", "persistent task anomalies were found", "inspect anomaly summary before rerunning"
    if proposal_audit["status"] == "fail":
        return (
            "stop",
            "proposal targeting audit failed",
            "fix editor prompt/proposal metadata before running a larger experiment",
        )
    if proposal_audit["status"] == "missing":
        return (
            "stop",
            "proposal logs are missing",
            "inspect the executive run directory before scaling up",
        )
    if not checks["accepted_step_present"]:
        return "stop", "no accepted executive optimization step", "do not scale up; inspect rejected contract deltas"
    if not checks["development_gate_passed"]:
        return "stop", "development gate failed", "inspect contract deltas before rerunning"
    if proposal_audit["status"] == "not_triggered":
        return (
            "inconclusive",
            "proposal targeting audit was not triggered because no proposal saw contract evidence",
            "run a smoke configuration that produces at least one rejected candidate before a later proposal",
        )
    if contract_effect_audit["status"] == "fail":
        return (
            "stop",
            "contract effect audit failed",
            "fix targeted proposal effectiveness before running a larger experiment",
        )
    if contract_effect_audit["status"] in {"missing", "not_triggered"}:
        return (
            "stop",
            "contract effect audit did not produce evidence",
            "inspect selection gate artifacts and proposal logs before scaling up",
        )
    return (
        "pass",
        "smoke gate passed",
        "run a small executive-only multi-seed experiment before any locked test",
    )


def load_runner_report(run_path: Path) -> dict[str, Any]:
    path = run_path / "runner_report.json"
    if path.exists():
        return load_json(path)
    return build_runner_report(run_path)


def summarize_proposal_audits(run_path: Path) -> dict[str, Any]:
    records = []
    required_records = 0
    failed_records = 0
    missing_audit_records = 0
    evidence_available_records = 0
    for path in sorted(run_path.rglob("proposals.jsonl")):
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            payload = json.loads(line)
            audit = payload.get("proposal_targeting_audit") if isinstance(payload, dict) else None
            if not isinstance(audit, dict):
                missing_audit_records += 1
                records.append({"path": str(path), "line": line_number, "status": "missing_audit"})
                continue
            required = bool(audit.get("required"))
            if audit.get("contract_rejection_evidence_available"):
                evidence_available_records += 1
            missing_count = int(audit.get("missing_targeted_contract_count") or 0)
            if required:
                required_records += 1
            if required and missing_count:
                failed_records += 1
            records.append(
                {
                    "path": str(path),
                    "line": line_number,
                    "status": "fail" if required and missing_count else "pass",
                    "required": required,
                    "missing_targeted_contract_count": missing_count,
                    "priority_contracts": audit.get("priority_contracts", []),
                    "proposal_count": audit.get("proposal_count", 0),
                }
            )
    if failed_records:
        status = "fail"
    elif required_records:
        status = "pass"
    elif records:
        status = "not_triggered"
    else:
        status = "missing"
    return {
        "status": status,
        "proposal_log_count": len(list(run_path.rglob("proposals.jsonl"))),
        "record_count": len(records),
        "required_record_count": required_records,
        "failed_required_record_count": failed_records,
        "missing_audit_record_count": missing_audit_records,
        "evidence_available_record_count": evidence_available_records,
        "records": records,
    }


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def render_compact_summary(report: dict[str, Any]) -> str:
    development_gate = dict(report.get("development_gate") or {})
    proposal_audit = dict(report.get("proposal_audit") or {})
    contract_effect = dict(report.get("contract_effect_audit") or {})
    criteria = development_gate.get("criteria") if isinstance(development_gate.get("criteria"), dict) else {}
    required_wins = development_gate.get("required_seed_wins") or criteria.get("min_seed_wins")
    parts = [
        "smoke_gate",
        f"status={report.get('status')}",
        f"reason={compact_token(report.get('reason'))}",
        f"accepted={report.get('accepted_steps_total')}",
        f"rows={report.get('executive_row_count')}",
        f"audit={proposal_audit.get('status')}",
        f"audit_failed={proposal_audit.get('failed_required_record_count')}/{proposal_audit.get('required_record_count')}",
        f"effect={contract_effect.get('status')}",
        f"effect_effective={contract_effect.get('effective_step_count')}/{contract_effect.get('required_step_count')}",
        f"gate_passed={development_gate.get('passed')}",
        f"mean={development_gate.get('executive_mean')}",
        f"delta={development_gate.get('mean_delta')}",
        f"wins={development_gate.get('seed_wins_vs_best_baseline')}/{required_wins}",
    ]
    return " ".join(parts)


def compact_token(value: Any) -> str:
    text = str(value or "").strip().replace("\n", " ")
    return json.dumps(text, ensure_ascii=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--min-accepted-steps", type=int, default=1)
    parser.add_argument("--quiet", action="store_true", help="Print a one-line summary instead of the full JSON report")
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
