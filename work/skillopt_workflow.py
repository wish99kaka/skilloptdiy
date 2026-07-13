#!/usr/bin/env python3
"""Mechanical SkillOpt workflow orchestration."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from work.experiment_runner import (
    build_runner_report,
    load_json,
    main as experiment_runner_main,
    resolve_workspace_path,
    runner_background_status,
    start_manifest_run,
    wait_for_runner,
    write_json,
)
from work.skillopt_compact_status import build_compact_status, render_text as render_compact_text
from work.skillopt_contract_effect_audit import build_contract_effect_audit
from work.skillopt_failure_delta_report import build_failure_delta_report, render_markdown
from work.skillopt_preflight import build_preflight_report
from work.skillopt_smoke_gate import build_smoke_gate_report


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "report":
        report = write_post_run_artifacts(args.run_dir)
        print(render_workflow_summary(report))
        return 0 if report["decision"]["status"] in {"scale_up_candidate", "stop", "inconclusive"} else 1
    if args.command == "run-smoke":
        report = run_smoke_workflow(
            args.manifest,
            background=args.background,
            wait_timeout_seconds=args.wait_timeout_seconds,
            poll_seconds=args.poll_seconds,
            skip_preflight=args.skip_preflight,
        )
        print(render_workflow_summary(report))
        return 0 if report["decision"]["status"] == "scale_up_candidate" else 1
    raise ValueError(f"Unsupported command: {args.command}")


def run_smoke_workflow(
    manifest_path: str | Path,
    *,
    background: bool = False,
    wait_timeout_seconds: float = 7200.0,
    poll_seconds: float = 30.0,
    skip_preflight: bool = False,
) -> dict[str, Any]:
    manifest = load_json(manifest_path)
    run_dir = resolve_workspace_path(manifest["out_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    if not skip_preflight:
        preflight = build_preflight_report(manifest_path)
        write_json(run_dir / "preflight_report.json", preflight)
        if preflight["status"] != "pass":
            decision = {
                "status": "preflight_failed",
                "scale_up_allowed": False,
                "next_action": "fix preflight failures before running smoke",
            }
            workflow = {
                "schema_version": 1,
                "run_dir": str(run_dir),
                "preflight": preflight,
                "decision": decision,
                "artifact_paths": {"preflight": str(run_dir / "preflight_report.json")},
            }
            write_json(run_dir / "decision.json", workflow)
            return workflow
    if background:
        background_payload = start_manifest_run(manifest_path)
        write_json(background_payload["background_path"], background_payload)
        wait_for_runner(
            run_dir,
            timeout_seconds=wait_timeout_seconds,
            poll_seconds=poll_seconds,
        )
    else:
        experiment_runner_main(["run", "--manifest", str(manifest_path)])
    return write_post_run_artifacts(run_dir)


def write_post_run_artifacts(run_dir: str | Path) -> dict[str, Any]:
    run_path = Path(run_dir)
    runner_report_path = run_path / "runner_report.json"
    if runner_report_path.exists():
        runner_report = load_json(runner_report_path)
    else:
        runner_report = build_runner_report(run_path)
        write_json(runner_report_path, runner_report)

    smoke_gate = build_smoke_gate_report(run_path)
    contract_effect = build_contract_effect_audit(run_path)
    write_json(run_path / "smoke_gate_report.json", smoke_gate)
    write_json(run_path / "contract_effect_audit.json", contract_effect)
    failure_delta = build_failure_delta_report(run_path) if (run_path / "summary.json").exists() else {}
    if failure_delta:
        write_json(run_path / "failure_delta_report.json", failure_delta)
        (run_path / "failure_delta_report.md").write_text(render_markdown(failure_delta), encoding="utf-8")
    compact_status = build_compact_status(run_path)
    compact_text = render_compact_text(compact_status)
    write_json(run_path / "compact_status.json", compact_status)
    (run_path / "compact_status.txt").write_text(compact_text + "\n", encoding="utf-8")
    decision = build_decision(runner_report, smoke_gate, contract_effect)
    workflow = {
        "schema_version": 1,
        "run_dir": str(run_path),
        "runner_status": runner_report.get("status"),
        "smoke_gate": {"status": smoke_gate.get("status"), "reason": smoke_gate.get("reason")},
        "contract_effect": {"status": contract_effect.get("status"), "reason": contract_effect.get("reason")},
        "decision": decision,
        "artifact_paths": {
            "runner_report": str(runner_report_path),
            "smoke_gate_report": str(run_path / "smoke_gate_report.json"),
            "contract_effect_audit": str(run_path / "contract_effect_audit.json"),
            "failure_delta_report": str(run_path / "failure_delta_report.json") if failure_delta else None,
            "failure_delta_markdown": str(run_path / "failure_delta_report.md") if failure_delta else None,
            "compact_status_json": str(run_path / "compact_status.json"),
            "compact_status_text": str(run_path / "compact_status.txt"),
            "decision": str(run_path / "decision.json"),
        },
        "background_status": runner_background_status(run_path),
    }
    write_json(run_path / "decision.json", workflow)
    return workflow


def build_decision(
    runner_report: dict[str, Any],
    smoke_gate: dict[str, Any],
    contract_effect: dict[str, Any],
) -> dict[str, Any]:
    if runner_report.get("status") != "complete":
        return {
            "status": "stop",
            "scale_up_allowed": False,
            "next_action": "inspect runner_report.json and execution logs",
            "reason": "runner did not complete",
        }
    if smoke_gate.get("status") == "pass" and contract_effect.get("status") == "pass":
        return {
            "status": "scale_up_candidate",
            "scale_up_allowed": True,
            "next_action": smoke_gate.get("next_action"),
            "reason": "smoke, development gate, and contract effect audit passed",
        }
    if smoke_gate.get("status") == "inconclusive":
        return {
            "status": "inconclusive",
            "scale_up_allowed": False,
            "next_action": smoke_gate.get("next_action"),
            "reason": smoke_gate.get("reason"),
        }
    return {
        "status": "stop",
        "scale_up_allowed": False,
        "next_action": smoke_gate.get("next_action") or "inspect generated reports",
        "reason": smoke_gate.get("reason") or contract_effect.get("reason"),
    }


def render_workflow_summary(report: dict[str, Any]) -> str:
    decision = dict(report.get("decision") or {})
    return " ".join(
        [
            "skillopt_workflow",
            f"status={decision.get('status')}",
            f"scale_up={decision.get('scale_up_allowed')}",
            f"run={report.get('run_dir')}",
            f"reason={json.dumps(str(decision.get('reason') or ''), ensure_ascii=True)}",
        ]
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    report = subparsers.add_parser("report", help="Generate smoke/effect/failure/compact reports for an existing run")
    report.add_argument("--run-dir", type=Path, required=True)
    run_smoke = subparsers.add_parser("run-smoke", help="Preflight, run a smoke manifest, and generate reports")
    run_smoke.add_argument("--manifest", type=Path, required=True)
    run_smoke.add_argument("--background", action="store_true")
    run_smoke.add_argument("--wait-timeout-seconds", type=float, default=7200.0)
    run_smoke.add_argument("--poll-seconds", type=float, default=30.0)
    run_smoke.add_argument("--skip-preflight", action="store_true")
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
