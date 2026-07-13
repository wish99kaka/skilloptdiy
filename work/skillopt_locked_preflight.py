#!/usr/bin/env python3
"""Read-only preflight gate before a one-attempt locked test."""

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
    report = build_locked_preflight_report(args.run_dir, receipt=args.receipt)
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    print(render_compact_summary(report) if args.quiet else text)
    return 0 if report["status"] == "allowed" else 1


def build_locked_preflight_report(run_dir: str | Path, *, receipt: str | Path | None = None) -> dict[str, Any]:
    run_path = Path(run_dir)
    receipt_path = Path(receipt) if receipt is not None else run_path / "locked_receipt.json"
    runner = load_json_if_exists(run_path / "runner_report.json")
    summary = load_json_if_exists(run_path / "summary.json")
    smoke_gate = load_json_if_exists(run_path / "smoke_gate_report.json")
    contract_effect = load_json_if_exists(run_path / "contract_effect_audit.json")
    if not contract_effect:
        contract_effect = load_json_if_exists(run_path / "contract_effect_report.json")

    checks = [
        check("runner_report_present", bool(runner), str(run_path / "runner_report.json")),
        check("runner_complete", runner.get("status") == "complete", str(runner.get("status"))),
        check(
            "development_gate_passed",
            bool(nested_dict(runner, "development_gate").get("passed")),
            str(nested_dict(runner, "development_gate").get("blocked_reason") or ""),
        ),
        check("smoke_gate_passed", smoke_gate.get("status") == "pass", str(smoke_gate.get("reason") or "")),
        check(
            "contract_effect_passed",
            contract_effect.get("status") == "pass",
            str(contract_effect.get("reason") or ""),
        ),
        check(
            "no_persistent_anomaly",
            int(nested_dict(runner, "anomaly_summary").get("persistent_anomaly_count") or 0) == 0,
            str(nested_dict(runner, "anomaly_summary").get("persistent_anomaly_count") or 0),
        ),
        check("actual_optimizer_usage_present", actual_optimizer_usage_present(summary), usage_detail(summary)),
        check("locked_receipt_absent", not receipt_path.exists(), str(receipt_path)),
    ]
    failed = [item for item in checks if not item["passed"]]
    return {
        "schema_version": 1,
        "run_dir": str(run_path),
        "receipt": str(receipt_path),
        "status": "blocked" if failed else "allowed",
        "allowed": not failed,
        "failed_check_count": len(failed),
        "missing_evidence": [item["name"] for item in failed],
        "checks": checks,
        "next_action": (
            "run locked test exactly once through textskill_optimizer.locked_eval"
            if not failed
            else "produce the missing development evidence before locked test"
        ),
    }


def actual_optimizer_usage_present(summary: dict[str, Any]) -> bool:
    usage = nested_dict(summary, "aggregate", "executive", "experiment_internal_usage_summary")
    return int(usage.get("actual_token_events") or 0) > 0 or int(usage.get("actual_total_tokens") or 0) > 0


def usage_detail(summary: dict[str, Any]) -> str:
    usage = nested_dict(summary, "aggregate", "executive", "experiment_internal_usage_summary")
    return (
        f"actual_token_events={usage.get('actual_token_events')} "
        f"actual_total_tokens={usage.get('actual_total_tokens')}"
    )


def check(name: str, passed: bool, detail: str) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "detail": detail}


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


def render_compact_summary(report: dict[str, Any]) -> str:
    return " ".join(
        [
            "locked_preflight",
            f"status={report.get('status')}",
            f"failed={report.get('failed_check_count')}",
            "missing=" + (",".join(report.get("missing_evidence") or []) or "none"),
        ]
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--quiet", action="store_true")
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
