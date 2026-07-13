#!/usr/bin/env python3
"""Mechanical experiment runner and compact report generator.

The control agent owns the protocol and decisions. This runner only executes a
declared command and writes a small report for the control agent to inspect.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from work.development_gate import (
    DEFAULT_DEVELOPMENT_GATE_CRITERIA,
    build_development_gate,
    count_seed_wins,
    normalize_development_gate_criteria,
)


REPORT_SCHEMA_VERSION = 1
MANIFEST_SCHEMA_VERSION = 1
DEFAULT_ACCEPTANCE = dict(DEFAULT_DEVELOPMENT_GATE_CRITERIA)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "report":
        report = build_runner_report(
            args.run_dir,
            acceptance=load_acceptance(args.acceptance_json),
        )
        out = args.out or (args.run_dir / "runner_report.json")
        write_json(out, report)
        print_report_pointer(out)
        return 0 if report["status"] == "complete" else 1
    if args.command == "run":
        manifest = load_json(args.manifest)
        validate_runner_manifest(manifest)
        run_dir = resolve_workspace_path(manifest["out_dir"])
        run_dir.mkdir(parents=True, exist_ok=True)
        manifest_copy = run_dir / "runner_manifest.json"
        write_json(manifest_copy, manifest)
        execution = execute_manifest_command(manifest, run_dir)
        execution_path = run_dir / "runner_execution.json"
        write_json(execution_path, execution)
        report = build_runner_report(
            run_dir,
            execution=execution,
            acceptance=dict(manifest.get("acceptance") or DEFAULT_ACCEPTANCE),
        )
        report["artifact_paths"]["runner_manifest"] = str(manifest_copy)
        report["artifact_paths"]["runner_execution"] = str(execution_path)
        report_path = run_dir / "runner_report.json"
        write_json(report_path, report)
        print_report_pointer(report_path)
        return int(execution.get("returncode") or 0)
    if args.command == "start":
        background = start_manifest_run(args.manifest)
        background_path = Path(background["background_path"])
        write_json(background_path, background)
        print(f"runner_background={background_path}")
        return 0
    if args.command == "status":
        status = runner_background_status(args.run_dir)
        write_json(args.out, status) if args.out else None
        print(json.dumps(status, indent=2, sort_keys=True))
        return 0 if status["status"] in {"running", "complete"} else 1
    if args.command == "wait":
        status = wait_for_runner(
            args.run_dir,
            timeout_seconds=args.timeout_seconds,
            poll_seconds=args.poll_seconds,
        )
        write_json(args.out, status) if args.out else None
        print(json.dumps(status, indent=2, sort_keys=True))
        if status["status"] == "complete":
            return 0
        return 124 if status.get("timed_out") else 1
    raise ValueError(f"Unsupported command: {args.command}")


def build_runner_report(
    run_dir: str | Path,
    *,
    execution: dict[str, Any] | None = None,
    acceptance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    run_path = Path(run_dir)
    summary_path = run_path / "summary.json"
    report: dict[str, Any] = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "missing_summary",
        "run_dir": str(run_path),
        "artifact_paths": {
            "summary": str(summary_path),
            "runner_report": str(run_path / "runner_report.json"),
        },
        "execution": compact_execution(execution),
        "decision": {
            "locked_test_recommended": False,
            "reason": "summary.json is missing",
        },
    }
    if not summary_path.exists():
        if execution and int(execution.get("returncode") or 0) != 0:
            report["status"] = "failed"
            report["decision"]["reason"] = "runner command failed before summary.json was produced"
        return report

    summary = load_json(summary_path)
    rows = list(summary.get("rows") or [])
    aggregate = dict(summary.get("aggregate") or {})
    manifest = dict(summary.get("manifest") or {})
    development_gate = resolve_development_gate(summary, rows, aggregate, acceptance)
    anomaly_summary = summarize_anomalies(run_path)
    status = "complete"
    if execution and int(execution.get("returncode") or 0) != 0:
        status = "failed"
    report.update(
        {
            "status": status,
            "manifest": compact_manifest(manifest),
            "scores_by_condition": compact_scores(aggregate),
            "seed_rows": compact_rows(rows),
            "development_gate": development_gate,
            "executive_decision": development_gate,
            "anomaly_summary": anomaly_summary,
            "usage": summary.get("usage", {}),
            "artifact_paths": {
                **report["artifact_paths"],
                "summary": str(summary_path),
            },
        }
    )
    locked_ok = (
        status == "complete"
        and development_gate["passed"]
        and anomaly_summary["persistent_anomaly_count"] == 0
    )
    if locked_ok:
        reason = "executive passed development criteria and no persistent anomaly was found"
    elif status != "complete":
        reason = "runner command did not complete cleanly"
    elif anomaly_summary["persistent_anomaly_count"]:
        reason = "persistent task anomalies require control-layer review"
    else:
        reason = development_gate["blocked_reason"]
    report["decision"] = {
        "locked_test_recommended": locked_ok,
        "reason": reason,
    }
    return report


def validate_runner_manifest(manifest: dict[str, Any]) -> None:
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ValueError("manifest.schema_version must be 1")
    if manifest.get("runner_role") != "mechanical_execution_only":
        raise ValueError("manifest.runner_role must be mechanical_execution_only")
    if manifest.get("experiment_type") != "coding_hidden_v2_matrix":
        raise ValueError("Only coding_hidden_v2_matrix is supported by this runner")
    command = manifest.get("command")
    if not isinstance(command, list) or not command or not all(isinstance(item, str) for item in command):
        raise ValueError("manifest.command must be a non-empty list of strings")
    if not manifest.get("out_dir"):
        raise ValueError("manifest.out_dir is required")
    immutable = manifest.get("immutable_controls") if isinstance(manifest.get("immutable_controls"), dict) else {}
    if immutable.get("do_not_change_coco_model") is not True:
        raise ValueError("manifest must explicitly set immutable_controls.do_not_change_coco_model=true")
    forbidden_args = {"--target-model", "--target_model", "--coco-model", "--coco_model"}
    if any(item in forbidden_args for item in command):
        raise ValueError("manifest.command must not override the target Coco model")
    if any(
        item.startswith("--target-model=")
        or item.startswith("--target_model=")
        or item.startswith("--coco-model=")
        or item.startswith("--coco_model=")
        for item in command
    ):
        raise ValueError("manifest.command must not override the target Coco model")
    rounds = command_int_option(command, "--validation-confirmation-rounds")
    stage = str(manifest.get("experiment_stage") or "")
    if rounds == 0 and stage != "mechanism_smoke":
        raise ValueError(
            "manifest with --validation-confirmation-rounds 0 must set "
            "experiment_stage=mechanism_smoke"
        )


def execute_manifest_command(manifest: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    command = [str(item) for item in manifest["command"]]
    timeout = manifest.get("timeout_seconds")
    env = os.environ.copy()
    env.update({str(k): str(v) for k, v in dict(manifest.get("env") or {}).items()})
    for key in manifest.get("env_passthrough", []) or []:
        key = str(key)
        if key not in os.environ:
            raise ValueError(f"Required passthrough env var is missing: {key}")
        env[key] = os.environ[key]

    stdout_path = run_dir / "runner_stdout.txt"
    stderr_path = run_dir / "runner_stderr.txt"
    started = time.monotonic()
    try:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            env=env,
            text=True,
            capture_output=True,
            timeout=float(timeout) if timeout is not None else None,
            check=False,
        )
        stdout = completed.stdout
        stderr = completed.stderr
        returncode = completed.returncode
        timed_out = False
    except subprocess.TimeoutExpired as exc:
        stdout = safe_text(exc.stdout)
        stderr = safe_text(exc.stderr) or f"Command timed out after {timeout}s"
        returncode = 124
        timed_out = True
    duration = time.monotonic() - started
    stdout_path.write_text(stdout, encoding="utf-8")
    stderr_path.write_text(stderr, encoding="utf-8")
    return {
        "command": command,
        "returncode": returncode,
        "timed_out": timed_out,
        "duration_seconds": duration,
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
        "stdout_chars": len(stdout),
        "stderr_chars": len(stderr),
    }


def start_manifest_run(manifest_path: str | Path) -> dict[str, Any]:
    manifest_file = Path(manifest_path)
    manifest = load_json(manifest_file)
    validate_runner_manifest(manifest)
    run_dir = resolve_workspace_path(manifest["out_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)

    existing = runner_background_status(run_dir)
    if existing["status"] == "running":
        raise RuntimeError(f"Runner is already running for {run_dir}: pid={existing.get('pid')}")
    if existing["status"] == "complete":
        raise RuntimeError(f"Runner report already exists for {run_dir}")

    stdout_path = run_dir / "runner_process_stdout.txt"
    stderr_path = run_dir / "runner_process_stderr.txt"
    background_path = run_dir / "runner_background.json"
    command = [
        sys.executable,
        str(ROOT / "work/experiment_runner.py"),
        "run",
        "--manifest",
        str(manifest_file.resolve()),
    ]
    stdout_handle = stdout_path.open("a", encoding="utf-8")
    stderr_handle = stderr_path.open("a", encoding="utf-8")
    try:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=os.environ.copy(),
            stdout=stdout_handle,
            stderr=stderr_handle,
            text=True,
            start_new_session=True,
        )
        pid = process.pid
        detach_popen(process)
    finally:
        stdout_handle.close()
        stderr_handle.close()
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "status": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "pid": pid,
        "run_dir": str(run_dir),
        "manifest": str(manifest_file.resolve()),
        "command": command,
        "background_path": str(background_path),
        "runner_report": str(run_dir / "runner_report.json"),
        "runner_execution": str(run_dir / "runner_execution.json"),
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
    }


def detach_popen(process: subprocess.Popen) -> None:
    # The child is intentionally long-lived; suppress Popen's destructor warning in the short-lived launcher.
    if hasattr(process, "_child_created"):
        process._child_created = False


def runner_background_status(run_dir: str | Path) -> dict[str, Any]:
    run_path = Path(run_dir)
    report_path = run_path / "runner_report.json"
    execution_path = run_path / "runner_execution.json"
    background_path = run_path / "runner_background.json"
    if background_path.exists():
        background = load_json(background_path)
        pid = as_int(background.get("pid"))
        if is_process_running(pid) and (
            not report_path.exists() or background_path.stat().st_mtime > report_path.stat().st_mtime
        ):
            return {
                "status": "running",
                "pid": pid,
                "run_dir": str(run_path),
                "runner_report": str(report_path),
                "runner_execution": str(execution_path) if execution_path.exists() else "missing",
                "stdout_path": background.get("stdout_path"),
                "stderr_path": background.get("stderr_path"),
                "started_at": background.get("started_at"),
            }
    if report_path.exists():
        report = load_json(report_path)
        execution = load_json(execution_path) if execution_path.exists() else {}
        return {
            "status": "complete" if report.get("status") == "complete" else "failed",
            "run_dir": str(run_path),
            "runner_report": str(report_path),
            "runner_execution": str(execution_path) if execution_path.exists() else "missing",
            "returncode": execution.get("returncode"),
            "report_status": report.get("status"),
            "locked_test_recommended": report.get("decision", {}).get("locked_test_recommended"),
        }
    if not background_path.exists():
        return {
            "status": "not_started",
            "run_dir": str(run_path),
            "runner_report": str(report_path),
            "runner_execution": str(execution_path) if execution_path.exists() else "missing",
        }
    background = load_json(background_path)
    pid = as_int(background.get("pid"))
    return {
        "status": "stopped_without_report",
        "pid": pid,
        "run_dir": str(run_path),
        "runner_report": str(report_path),
        "runner_execution": str(execution_path) if execution_path.exists() else "missing",
        "stdout_path": background.get("stdout_path"),
        "stderr_path": background.get("stderr_path"),
        "started_at": background.get("started_at"),
    }


def wait_for_runner(
    run_dir: str | Path,
    *,
    timeout_seconds: float,
    poll_seconds: float,
) -> dict[str, Any]:
    if timeout_seconds < 0:
        raise ValueError("timeout_seconds must be non-negative")
    if poll_seconds <= 0:
        raise ValueError("poll_seconds must be positive")
    started = time.monotonic()
    terminal = {"complete", "failed", "stopped_without_report"}
    while True:
        status = runner_background_status(run_dir)
        elapsed = time.monotonic() - started
        status["elapsed_seconds"] = elapsed
        status["timed_out"] = False
        if status["status"] in terminal:
            return status
        if elapsed >= timeout_seconds:
            status["timed_out"] = True
            return status
        time.sleep(min(poll_seconds, timeout_seconds - elapsed))


def is_process_running(pid: int | None) -> bool:
    if pid is None or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def score_executive_against_baselines(
    rows: list[dict[str, Any]],
    aggregate: dict[str, Any],
    criteria: dict[str, Any],
) -> dict[str, Any]:
    return build_development_gate(rows, aggregate, criteria)


def resolve_development_gate(
    summary: dict[str, Any],
    rows: list[dict[str, Any]],
    aggregate: dict[str, Any],
    acceptance: dict[str, Any] | None,
) -> dict[str, Any]:
    if acceptance is None and isinstance(summary.get("development_gate"), dict):
        return normalize_development_gate_payload(summary["development_gate"])
    return build_development_gate(rows, aggregate, normalize_acceptance(acceptance))


def normalize_development_gate_payload(payload: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(payload)
    criteria = normalize_development_gate_criteria(
        normalized.get("criteria") if isinstance(normalized.get("criteria"), dict) else None
    )
    normalized["criteria"] = criteria
    normalized.setdefault("schema_version", 1)
    normalized.setdefault("score_metric", "task_accuracy_mean")
    normalized.setdefault("best_baseline_condition", None)
    normalized.setdefault("best_baseline_score", float(normalized.get("best_baseline_mean") or 0.0))
    normalized.setdefault("executive_score", float(normalized.get("executive_mean") or 0.0))
    normalized.setdefault("score_delta", float(normalized.get("mean_delta") or 0.0))
    normalized.setdefault("required_delta", criteria["best_baseline_margin"])
    normalized.setdefault("seed_wins_vs_best_baseline", 0)
    normalized.setdefault("required_seed_wins", criteria["min_seed_wins"])
    normalized.setdefault("passed", bool(normalized.get("criteria_met")))
    normalized.setdefault("criteria_met", bool(normalized.get("passed")))
    normalized.setdefault("locked_test_recommended", bool(normalized.get("passed")))
    blocked_reason = str(normalized.get("blocked_reason") or "")
    blocked_reasons = normalized.get("blocked_reasons")
    if not isinstance(blocked_reasons, list):
        blocked_reasons = [blocked_reason] if blocked_reason else []
    normalized["blocked_reasons"] = [str(item) for item in blocked_reasons if str(item)]
    normalized["blocked_reason"] = "; ".join(normalized["blocked_reasons"])
    return normalized


def summarize_anomalies(run_dir: Path) -> dict[str, Any]:
    retry_policy_entries = 0
    retry_count = 0
    persistent = 0
    timeout_markers = 0
    json_files = 0
    for path in run_dir.rglob("*.json"):
        if path.name in {"runner_report.json", "runner_execution.json", "runner_manifest.json"}:
            continue
        try:
            payload = load_json(path)
        except (OSError, json.JSONDecodeError):
            continue
        json_files += 1
        for item in walk_json(payload):
            if not isinstance(item, dict):
                continue
            retry_policy = item.get("retry_policy")
            if isinstance(retry_policy, dict):
                retry_policy_entries += 1
                attempt_count = as_int(retry_policy.get("attempt_count"))
                retry_count += max(0, (attempt_count or 1) - 1)
                if retry_policy.get("persistent_anomaly"):
                    persistent += 1
            if item.get("status") == "agent_timeout" or item.get("error") == "agent_timeout":
                timeout_markers += 1
    return {
        "json_files_scanned": json_files,
        "retry_policy_entries": retry_policy_entries,
        "retry_count": retry_count,
        "persistent_anomaly_count": persistent,
        "agent_timeout_markers": timeout_markers,
    }


def compact_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "experiment_stage",
        "benchmark",
        "development_only",
        "target_harness",
        "target_model",
        "target_model_policy",
        "optimizer_harness",
        "optimizer_model",
        "seeds",
        "seed_workers",
        "conditions",
        "optimizer_config",
        "development_gate_criteria",
        "task_limit",
        "locked_test_sha256",
    )
    return {key: manifest.get(key) for key in keys if key in manifest}


def compact_scores(aggregate: dict[str, Any]) -> dict[str, Any]:
    output = {}
    for condition, payload in aggregate.items():
        if not isinstance(payload, dict):
            continue
        output[condition] = {
            "runs": payload.get("runs"),
            "task_accuracy_mean": payload.get("task_accuracy_mean"),
            "task_accuracy_stddev": payload.get("task_accuracy_stddev"),
            "family_macro_mean": payload.get("family_macro_mean"),
            "family_macro_stddev": payload.get("family_macro_stddev"),
            "contract_macro_mean": payload.get("contract_macro_mean"),
            "contract_macro_stddev": payload.get("contract_macro_stddev"),
            "contract_breakdown": payload.get("contract_breakdown"),
            "duration_seconds_total": payload.get("duration_seconds_total"),
        }
    return output


def compact_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keys = (
        "seed",
        "condition",
        "task_accuracy",
        "family_macro_accuracy",
        "contract_macro_accuracy",
        "contract_breakdown",
        "duration_seconds",
        "accepted_steps",
        "total_steps",
        "best_validation_score",
        "run_dir",
    )
    return [{key: row.get(key) for key in keys if key in row} for row in rows]


def compact_execution(execution: dict[str, Any] | None) -> dict[str, Any] | None:
    if not execution:
        return None
    keys = (
        "returncode",
        "timed_out",
        "duration_seconds",
        "stdout_path",
        "stderr_path",
        "stdout_chars",
        "stderr_chars",
    )
    return {key: execution.get(key) for key in keys if key in execution}


def normalize_acceptance(acceptance: dict[str, Any] | None) -> dict[str, Any]:
    return normalize_development_gate_criteria({**DEFAULT_ACCEPTANCE, **dict(acceptance or {})})


def command_int_option(command: list[str], option: str) -> int | None:
    prefix = option + "="
    for index, item in enumerate(command):
        if item == option and index + 1 < len(command):
            return int(command[index + 1])
        if item.startswith(prefix):
            return int(item[len(prefix):])
    return None


def load_acceptance(raw: str | None) -> dict[str, Any] | None:
    if not raw:
        return None
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise ValueError("--acceptance-json must decode to an object")
    return parsed


def walk_json(value: Any) -> Iterable[Any]:
    yield value
    if isinstance(value, dict):
        for item in value.values():
            yield from walk_json(item)
    elif isinstance(value, list):
        for item in value:
            yield from walk_json(item)


def resolve_workspace_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return ROOT / path


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, payload: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def as_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def safe_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def print_report_pointer(path: Path) -> None:
    print(f"runner_report={path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    report = subparsers.add_parser("report", help="Build runner_report.json from an existing run directory")
    report.add_argument("--run-dir", type=Path, required=True)
    report.add_argument("--out", type=Path)
    report.add_argument("--acceptance-json")

    run = subparsers.add_parser("run", help="Execute a runner manifest and write runner_report.json")
    run.add_argument("--manifest", type=Path, required=True)

    start = subparsers.add_parser("start", help="Start a runner manifest in the background")
    start.add_argument("--manifest", type=Path, required=True)

    status = subparsers.add_parser("status", help="Inspect a background runner directory")
    status.add_argument("--run-dir", type=Path, required=True)
    status.add_argument("--out", type=Path)

    wait = subparsers.add_parser("wait", help="Poll a background runner until it finishes or times out")
    wait.add_argument("--run-dir", type=Path, required=True)
    wait.add_argument("--timeout-seconds", type=float, required=True)
    wait.add_argument("--poll-seconds", type=float, default=30.0)
    wait.add_argument("--out", type=Path)
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
