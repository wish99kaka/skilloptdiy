#!/usr/bin/env python3
"""Preflight checks for SkillOpt runner manifests."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from work.experiment_runner import load_json, validate_runner_manifest
from work.skillopt_stage_policy import (
    command_option_value,
    parse_csv,
    validate_manifest_stage_policy,
    validate_task_ids,
)


MIN_PYTHON = (3, 10)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = build_preflight_report(args.manifest)
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    print(render_compact_summary(report) if args.quiet else text)
    return 0 if report["status"] == "pass" else 1


def build_preflight_report(
    manifest_path: str | Path,
    *,
    environ: Mapping[str, str] | None = None,
    current_python_version: tuple[int, int, int] | None = None,
    command_python_version: tuple[int, int, int] | None = None,
) -> dict[str, Any]:
    manifest_file = Path(manifest_path)
    env = dict(os.environ if environ is None else environ)
    checks: list[dict[str, Any]] = []
    manifest: dict[str, Any] = {}
    if not manifest_file.exists():
        checks.append(check("manifest_exists", False, f"manifest is missing: {manifest_file}"))
        return build_report(manifest_file, checks, manifest)
    checks.append(check("manifest_exists", True, str(manifest_file)))
    try:
        loaded = load_json(manifest_file)
        if not isinstance(loaded, dict):
            raise ValueError("manifest must be a JSON object")
        manifest = loaded
        validate_runner_manifest(manifest)
        checks.append(check("runner_manifest_valid", True, "runner manifest contract is valid"))
    except Exception as exc:
        checks.append(check("runner_manifest_valid", False, str(exc)))
        return build_report(manifest_file, checks, manifest)

    current_version = current_python_version or sys.version_info[:3]
    checks.append(
        check(
            "current_python_version",
            python_at_least(current_version),
            ".".join(str(item) for item in current_version),
        )
    )
    command = [str(item) for item in manifest.get("command") or []]
    command_version = command_python_version
    if command_version is None and command and Path(command[0]).name.startswith("python"):
        command_version = detect_python_version(command[0])
    if command_version is not None:
        checks.append(
            check(
                "command_python_version",
                python_at_least(command_version),
                ".".join(str(item) for item in command_version),
            )
        )

    stage_issues = validate_manifest_stage_policy(manifest)
    checks.append(
        check(
            "stage_policy",
            not stage_issues,
            "ok" if not stage_issues else "; ".join(item["message"] for item in stage_issues),
            issues=stage_issues,
        )
    )

    manifest_env = {str(key): str(value) for key, value in dict(manifest.get("env") or {}).items()}
    combined_env = {**env, **manifest_env}
    for key in manifest.get("env_passthrough") or []:
        key = str(key)
        present = bool(env.get(key, "").strip())
        checks.append(check(f"env_passthrough:{key}", present, "present" if present else "missing"))
    for key in ("EXTERNAL_LLM_BASE_URL", "EXTERNAL_LLM_MODEL"):
        present = bool(combined_env.get(key, "").strip())
        checks.append(check(f"external_llm:{key}", present, "present" if present else "missing"))

    out_dir = str(manifest.get("out_dir") or "")
    command_out = command_option_value(command, "--out")
    checks.append(
        check(
            "out_dir_matches_command",
            bool(out_dir and command_out and normalize_path(out_dir) == normalize_path(command_out)),
            f"manifest.out_dir={out_dir} command.--out={command_out}",
        )
    )

    baseline_summary = command_option_value(command, "--baseline-summary")
    if baseline_summary:
        baseline_path = resolve_workspace_path(baseline_summary)
        checks.append(check("baseline_summary_exists", baseline_path.exists(), str(baseline_path)))
    task_issues = validate_task_ids(
        train_task_ids=set(parse_csv(command_option_value(command, "--train-task-ids"))),
        selection_task_ids=set(parse_csv(command_option_value(command, "--selection-task-ids"))),
    )
    checks.append(
        check(
            "task_ids_exist",
            not task_issues,
            "ok" if not task_issues else "; ".join(item["message"] for item in task_issues),
            issues=task_issues,
        )
    )
    return build_report(manifest_file, checks, manifest)


def build_report(manifest_path: Path, checks: list[dict[str, Any]], manifest: dict[str, Any]) -> dict[str, Any]:
    failed = [item for item in checks if not item["passed"]]
    return {
        "schema_version": 1,
        "manifest": str(manifest_path),
        "status": "fail" if failed else "pass",
        "experiment_stage": manifest.get("experiment_stage"),
        "run_dir": manifest.get("out_dir"),
        "failed_check_count": len(failed),
        "checks": checks,
    }


def detect_python_version(executable: str) -> tuple[int, int, int] | None:
    try:
        completed = subprocess.run(
            [
                executable,
                "-c",
                "import sys,json; print(json.dumps(list(sys.version_info[:3])))",
            ],
            cwd=ROOT,
            text=True,
            capture_output=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if completed.returncode != 0:
        return None
    try:
        payload = json.loads(completed.stdout.strip())
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, list) or len(payload) < 3:
        return None
    return int(payload[0]), int(payload[1]), int(payload[2])


def python_at_least(version: tuple[int, int, int]) -> bool:
    return version[:2] >= MIN_PYTHON


def check(name: str, passed: bool, detail: str, **extra: Any) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "detail": detail, **extra}


def resolve_workspace_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def normalize_path(value: str) -> str:
    return str(resolve_workspace_path(value).resolve())


def render_compact_summary(report: dict[str, Any]) -> str:
    failed = [
        str(item.get("name"))
        for item in report.get("checks", [])
        if isinstance(item, dict) and not item.get("passed")
    ]
    return " ".join(
        [
            "skillopt_preflight",
            f"status={report.get('status')}",
            f"stage={report.get('experiment_stage')}",
            f"failed={report.get('failed_check_count')}",
            f"checks={','.join(failed) if failed else 'none'}",
        ]
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--quiet", action="store_true")
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
