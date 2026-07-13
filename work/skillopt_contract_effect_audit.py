#!/usr/bin/env python3
"""Audit whether evidence-guided proposals changed their targeted contracts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from work.skillopt_failure_delta_report import build_failure_delta_report


SCHEMA_VERSION = 1


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = build_contract_effect_audit(args.run_dir)
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    print(render_compact_summary(report) if args.quiet else text)
    return 0 if report["status"] in {"pass", "fail", "not_triggered"} else 1


def build_contract_effect_audit(run_dir: str | Path) -> dict[str, Any]:
    run_path = Path(run_dir)
    if not (run_path / "summary.json").exists():
        return {
            "schema_version": SCHEMA_VERSION,
            "run_dir": str(run_path),
            "status": "missing",
            "reason": "summary.json is missing",
            "records": [],
            "required_step_count": 0,
            "effective_step_count": 0,
            "failed_effect_step_count": 0,
            "protected_regression_count": 0,
        }
    failure_report = build_failure_delta_report(run_path)
    records = [
        build_effect_record(step)
        for step in failure_report.get("steps", [])
        if nested_bool(step, "proposal_audit", "required")
    ]
    accepted_records = [record for record in records if record["accepted"]]
    rejected_records = [record for record in records if not record["accepted"]]
    effective_count = sum(1 for record in accepted_records if record["targeted_contract_improved"])
    protected_regression_count = sum(
        1 for record in accepted_records if record["protected_contract_regressed"]
    )
    failed_count = sum(1 for record in accepted_records if not record["passes"])
    rejected_effective_count = sum(1 for record in rejected_records if record["targeted_contract_improved"])
    rejected_protected_regression_count = sum(
        1 for record in rejected_records if record["protected_contract_regressed"]
    )
    rejected_failed_count = sum(1 for record in rejected_records if not record["passes"])
    if not records:
        status = "not_triggered"
        reason = "no evidence-guided candidate had contract-effect evidence"
    elif protected_regression_count:
        status = "fail"
        reason = "accepted candidate regressed a protected or anti-regression contract"
    elif failed_count:
        status = "fail"
        reason = "accepted evidence-guided candidate did not produce a safe targeted effect"
    elif effective_count == 0:
        status = "fail"
        reason = "no accepted evidence-guided candidate improved a targeted contract"
    else:
        status = "pass"
        reason = "accepted evidence-guided candidates improved targets without accepted protected regression"
    return {
        "schema_version": SCHEMA_VERSION,
        "run_dir": str(run_path),
        "status": status,
        "reason": reason,
        "required_step_count": len(records),
        "accepted_required_step_count": len(accepted_records),
        "rejected_required_step_count": len(rejected_records),
        "effective_step_count": effective_count,
        "failed_effect_step_count": failed_count,
        "protected_regression_count": protected_regression_count,
        "rejected_effective_step_count": rejected_effective_count,
        "rejected_failed_effect_step_count": rejected_failed_count,
        "rejected_protected_regression_count": rejected_protected_regression_count,
        "records": records,
    }


def build_effect_record(step: dict[str, Any]) -> dict[str, Any]:
    targeted_outcomes = dict(step.get("targeted_contract_outcomes") or {})
    protected_outcomes = dict(step.get("protected_contract_outcomes") or {})
    targeted_improved = sorted(
        contract
        for contract, outcome in targeted_outcomes.items()
        if isinstance(outcome, dict) and maybe_float(outcome.get("delta")) is not None and maybe_float(outcome.get("delta")) > 0
    )
    targeted_evaluated = sorted(
        contract
        for contract, outcome in targeted_outcomes.items()
        if isinstance(outcome, dict) and maybe_float(outcome.get("delta")) is not None
    )
    protected_regressed = sorted(
        contract
        for contract, outcome in protected_outcomes.items()
        if isinstance(outcome, dict) and maybe_float(outcome.get("delta")) is not None and maybe_float(outcome.get("delta")) < 0
    )
    issues = []
    if not targeted_evaluated:
        issues.append("targeted_contract_not_evaluated")
    elif not targeted_improved:
        issues.append("targeted_contract_not_improved")
    if protected_regressed:
        issues.append("protected_contract_regressed")
    return {
        "seed": step.get("seed"),
        "candidate": step.get("candidate"),
        "accepted": bool(step.get("accepted")),
        "targeted_contracts": list(step.get("targeted_contracts") or []),
        "protected_contracts": list(step.get("protected_contracts") or []),
        "targeted_contract_improved": bool(targeted_improved),
        "protected_contract_regressed": bool(protected_regressed),
        "targeted_improved_contracts": targeted_improved,
        "protected_regressed_contracts": protected_regressed,
        "targeted_contract_outcomes": targeted_outcomes,
        "protected_contract_outcomes": protected_outcomes,
        "passes": not issues,
        "issues": issues,
    }


def nested_bool(payload: dict[str, Any], *keys: str) -> bool:
    current: Any = payload
    for key in keys:
        if not isinstance(current, dict):
            return False
        current = current.get(key)
    return bool(current)


def maybe_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def render_compact_summary(report: dict[str, Any]) -> str:
    return " ".join(
        [
            "contract_effect",
            f"status={report.get('status')}",
            f"reason={compact_token(report.get('reason'))}",
            f"effective={report.get('effective_step_count')}/{report.get('required_step_count')}",
            f"failed={report.get('failed_effect_step_count')}",
            f"protected_regressions={report.get('protected_regression_count')}",
        ]
    )


def compact_token(value: Any) -> str:
    text = str(value or "").strip().replace("\n", " ")
    return json.dumps(text, ensure_ascii=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--quiet", action="store_true", help="Print a one-line summary instead of full JSON")
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
