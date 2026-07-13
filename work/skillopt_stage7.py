#!/usr/bin/env python3
"""Prepare, validate, and consume the one-attempt coding-hidden-v2 locked test."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from work.skillopt_locked_preflight import build_locked_preflight_report


LOCKED_RESULT_NAME = "locked_evaluation.json"
LOCKED_RECEIPT_NAME = "locked_receipt.json"
LOCKED_FINAL_REPORT_NAME = "locked_final_report.json"
LOCKED_ATTEMPT_NAME = "locked_attempt.json"
LOCKED_USAGE_LEDGER_NAME = "locked_usage_ledger.jsonl"
CONSUME_CONFIRMATION = "CONSUME_LOCKED_TEST_ONCE"
PYTHON_EXECUTABLE = str(Path(sys.executable).resolve())
PYTHON_VERSION = list(sys.version_info[:3])
LOCKED_ENV_SET = {"COCO_AGENT_TIMEOUT": "360"}
LOCKED_ENV_UNSET = (
    "COCO_AGENT_BIN",
    "COCO_AGENT_DRY_RUN",
    "COCO_AGENT_EXTRA_ARGS",
    "COCO_AGENT_QUERY_TIMEOUT",
    "COCO_AGENT_BASH_TIMEOUT",
    "COCO_AGENT_YOLO",
    "COCO_TASK_LIMIT",
    "CROSS_AGENT_TASKS",
    "PYTHONHOME",
    "PYTHONPATH",
)
CODE_COMMITMENT_PATHS = (
    "work/skillopt_stage7.py",
    "textskill_optimizer/locked_eval.py",
    "work/run_coding_hidden_v2_locked_eval.py",
    "work/run_coco_hidden_eval.py",
    "work/run_coding_hidden_v2_matrix.py",
    "textskill_optimizer/executive_optimizer.py",
    "textskill_optimizer/plugins/coding.py",
    "examples/coding/coco_agent_wrapper.py",
)
USAGE_SCOPE = {
    "optimizer_api": "actual_same_run_usage",
    "target_agent_tokens": "out_of_scope",
}
EXECUTION_POLICY = {"attempts": 1, "task_retries": 1, "whole_command_timeout": None}

SELECTION_RULE = [
    "task_accuracy_desc",
    "contract_macro_accuracy_desc",
    "skill_bytes_asc",
    "seed_asc",
]


def select_locked_candidate(run_dir: str | Path) -> dict[str, Any]:
    run_path = Path(run_dir)
    summary_path = run_path / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    candidates = []
    for row in summary.get("rows") or []:
        if row.get("condition") != "executive":
            continue
        seed = str(row.get("seed") or "")
        if not seed:
            raise ValueError("Executive row is missing seed")
        skill_path = run_path / seed / "executive" / "best_skill.md"
        if not skill_path.is_file():
            raise FileNotFoundError(f"Executive best skill is missing: {skill_path}")
        skill_bytes = skill_path.read_bytes()
        task_accuracy = float(row.get("task_accuracy", row.get("average_score", 0.0)))
        contract_accuracy = float(row.get("contract_macro_accuracy", 0.0))
        candidates.append(
            {
                "seed": seed,
                "task_accuracy": task_accuracy,
                "average_score": float(row.get("average_score", task_accuracy)),
                "contract_macro_accuracy": contract_accuracy,
                "skill_path": str(skill_path),
                "skill_sha256": hashlib.sha256(skill_bytes).hexdigest(),
                "skill_bytes": len(skill_bytes),
                "selection_rule": list(SELECTION_RULE),
            }
        )
    if not candidates:
        raise ValueError(f"No executive rows found in {summary_path}")
    return min(
        candidates,
        key=lambda item: (
            -item["task_accuracy"],
            -item["contract_macro_accuracy"],
            item["skill_bytes"],
            item["seed"],
        ),
    )


def sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_locked_environment(environ: Mapping[str, str] | None = None) -> dict[str, str]:
    environment = dict(os.environ if environ is None else environ)
    for name in LOCKED_ENV_UNSET:
        environment.pop(name, None)
    environment.update(LOCKED_ENV_SET)
    return environment


def build_code_commitments() -> dict[str, str]:
    return {relative: sha256_file(ROOT / relative) for relative in CODE_COMMITMENT_PATHS}


def build_development_evidence(summary: dict[str, Any]) -> dict[str, Any]:
    aggregate = summary.get("aggregate") or {}
    development_gate = summary.get("development_gate") or {}
    usage = ((summary.get("usage") or {}).get("experiment_internal_usage") or {})
    executive_usage = ((aggregate.get("executive") or {}).get("experiment_internal_usage_summary") or {})
    one_shot_usage = ((aggregate.get("one_shot") or {}).get("experiment_internal_usage_summary") or {})
    return {
        "development_gate": {
            "passed": bool(development_gate.get("passed")),
            "executive_mean": development_gate.get("executive_mean"),
            "best_baseline_condition": development_gate.get("best_baseline_condition"),
            "best_baseline_mean": development_gate.get("best_baseline_mean"),
            "mean_delta": development_gate.get("mean_delta"),
            "seed_wins": development_gate.get("seed_wins_vs_best_baseline"),
        },
        "condition_means": {
            condition: (aggregate.get(condition) or {}).get("task_accuracy_mean")
            for condition in ("executive", "human_skill", "no_skill", "one_shot")
        },
        "optimizer_api_tokens": {
            "total": int(usage.get("actual_total_tokens") or 0),
            "executive": int(executive_usage.get("actual_total_tokens") or 0),
            "one_shot": int(one_shot_usage.get("actual_total_tokens") or 0),
        },
    }


def build_locked_command(
    *,
    archive: str | Path,
    key_file: str | Path,
    lock_file: str | Path,
    receipt: str | Path,
    skill: str | Path,
    result: str | Path,
) -> list[str]:
    return [
        PYTHON_EXECUTABLE,
        "-m",
        "textskill_optimizer.locked_eval",
        "run",
        "--archive",
        str(archive),
        "--key-file",
        str(key_file),
        "--lock",
        str(lock_file),
        "--receipt",
        str(receipt),
        "--cwd",
        str(ROOT),
        "--",
        PYTHON_EXECUTABLE,
        "work/run_coding_hidden_v2_locked_eval.py",
        "--skill",
        str(skill),
        "--out",
        str(result),
        "--task-timeout",
        "360",
        "--task-retries",
        "1",
        "--retry-backoff-seconds",
        "5.0",
    ]


def build_stage7_manifest(
    run_dir: str | Path,
    *,
    key_file: str | Path,
    archive: str | Path = ROOT / "examples/coding-hidden-v2/test.enc",
    lock_file: str | Path = ROOT / "examples/coding-hidden-v2/test.lock.json",
) -> dict[str, Any]:
    run_path = Path(run_dir).resolve()
    key_path = Path(key_file).expanduser().resolve()
    archive_path = Path(archive).resolve()
    lock_path = Path(lock_file).resolve()
    for required in (key_path, archive_path, lock_path):
        if not required.is_file():
            raise FileNotFoundError(f"Stage 7 input is missing: {required}")

    preflight = build_locked_preflight_report(run_path)
    if preflight["status"] != "allowed":
        raise ValueError(
            "Locked preflight is not allowed: " + ",".join(preflight.get("missing_evidence") or [])
        )
    selected = select_locked_candidate(run_path)
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    archive_sha256 = sha256_file(archive_path)
    if lock.get("archive_sha256") != archive_sha256:
        raise ValueError("Locked archive does not match test.lock.json")
    lock_details = lock.get("details") or {}
    if lock_details.get("task_count") != 20 or lock_details.get("task_file") != "test.jsonl":
        raise ValueError("Locked metadata must declare test.jsonl with exactly 20 tasks")
    summary = json.loads((run_path / "summary.json").read_text(encoding="utf-8"))
    committed_sha256 = str((summary.get("manifest") or {}).get("locked_test_sha256") or "")
    if committed_sha256 != archive_sha256:
        raise ValueError("Stage 5 summary does not commit to the current locked archive")

    receipt = run_path / LOCKED_RECEIPT_NAME
    result = run_path / LOCKED_RESULT_NAME
    final_report = run_path / LOCKED_FINAL_REPORT_NAME
    attempt = run_path / LOCKED_ATTEMPT_NAME
    usage_ledger = run_path / LOCKED_USAGE_LEDGER_NAME
    for output in (attempt, receipt, result, final_report, usage_ledger):
        if output.exists():
            raise FileExistsError(f"Refusing stale or consumed Stage 7 output: {output}")

    command = build_locked_command(
        archive=archive_path,
        key_file=key_path,
        lock_file=lock_path,
        receipt=receipt,
        skill=selected["skill_path"],
        result=result,
    )
    return {
        "schema_version": 1,
        "experiment_stage": "locked_test_once",
        "runner_role": "mechanical_execution_only",
        "runtime": {
            "python_executable": PYTHON_EXECUTABLE,
            "python_version": list(PYTHON_VERSION),
        },
        "execution_policy": dict(EXECUTION_POLICY),
        "source_run_dir": str(run_path),
        "selected_candidate": selected,
        "locked_test": {
            "archive": str(archive_path),
            "archive_sha256": archive_sha256,
            "lock_file": str(lock_path),
            "key_file": str(key_path),
            "key_file_sha256": sha256_file(key_path),
            "attempt": str(attempt),
            "receipt": str(receipt),
            "result": str(result),
            "final_report": str(final_report),
            "usage_ledger": str(usage_ledger),
            "expected_task_count": 20,
            "task_file": "test.jsonl",
        },
        "development_evidence": build_development_evidence(summary),
        "usage_scope": dict(USAGE_SCOPE),
        "environment_policy": {
            "set": dict(LOCKED_ENV_SET),
            "unset": list(LOCKED_ENV_UNSET),
        },
        "code_commitments": build_code_commitments(),
        "command": command,
    }


def validate_stage7_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    def check(name: str, passed: bool, detail: str = "") -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    check("stage_locked_test_once", manifest.get("experiment_stage") == "locked_test_once")
    check("mechanical_runner_role", manifest.get("runner_role") == "mechanical_execution_only")
    check(
        "python_runtime_unchanged",
        manifest.get("runtime")
        == {"python_executable": PYTHON_EXECUTABLE, "python_version": list(PYTHON_VERSION)},
    )
    check("execution_policy_unchanged", manifest.get("execution_policy") == EXECUTION_POLICY)
    check(
        "environment_policy_unchanged",
        manifest.get("environment_policy")
        == {"set": dict(LOCKED_ENV_SET), "unset": list(LOCKED_ENV_UNSET)},
    )
    try:
        check(
            "code_commitments_unchanged",
            manifest.get("code_commitments") == build_code_commitments(),
        )
    except Exception as exc:
        check("code_commitments_unchanged", False, f"{type(exc).__name__}: {exc}")
    run_path = Path(str(manifest.get("source_run_dir") or ""))
    try:
        preflight = build_locked_preflight_report(run_path)
        check(
            "locked_preflight_allowed",
            preflight.get("status") == "allowed",
            ",".join(preflight.get("missing_evidence") or []),
        )
    except Exception as exc:
        check("locked_preflight_allowed", False, f"{type(exc).__name__}: {exc}")

    selected_manifest = manifest.get("selected_candidate") or {}
    selected_actual: dict[str, Any] = {}
    try:
        selected_actual = select_locked_candidate(run_path)
        identity_fields = (
            "seed",
            "task_accuracy",
            "average_score",
            "contract_macro_accuracy",
            "skill_path",
            "skill_sha256",
            "skill_bytes",
            "selection_rule",
        )
        unchanged = all(selected_manifest.get(field) == selected_actual.get(field) for field in identity_fields)
        check("selected_candidate_unchanged", unchanged, str(selected_actual.get("seed") or ""))
    except Exception as exc:
        check("selected_candidate_unchanged", False, f"{type(exc).__name__}: {exc}")

    locked = manifest.get("locked_test") or {}
    archive = Path(str(locked.get("archive") or ""))
    lock_file = Path(str(locked.get("lock_file") or ""))
    key_file = Path(str(locked.get("key_file") or ""))
    attempt = Path(str(locked.get("attempt") or ""))
    receipt = Path(str(locked.get("receipt") or ""))
    result = Path(str(locked.get("result") or ""))
    final_report = Path(str(locked.get("final_report") or ""))
    usage_ledger = Path(str(locked.get("usage_ledger") or ""))

    actual_archive_sha256 = ""
    try:
        actual_archive_sha256 = sha256_file(archive)
        check(
            "archive_unchanged",
            actual_archive_sha256 == locked.get("archive_sha256"),
            actual_archive_sha256,
        )
    except Exception as exc:
        check("archive_unchanged", False, f"{type(exc).__name__}: {exc}")
    try:
        lock = json.loads(lock_file.read_text(encoding="utf-8"))
        check(
            "lock_matches_archive",
            lock.get("archive_sha256") == actual_archive_sha256,
            str(lock.get("archive_sha256") or ""),
        )
        details = lock.get("details") or {}
        check(
            "lock_metadata_complete",
            details.get("task_count") == 20
            and details.get("task_file") == "test.jsonl"
            and locked.get("expected_task_count") == 20
            and locked.get("task_file") == "test.jsonl",
            json.dumps(details, sort_keys=True),
        )
    except Exception as exc:
        check("lock_matches_archive", False, f"{type(exc).__name__}: {exc}")
        check("lock_metadata_complete", False, f"{type(exc).__name__}: {exc}")
    summary: dict[str, Any] = {}
    try:
        summary = json.loads((run_path / "summary.json").read_text(encoding="utf-8"))
        committed = str((summary.get("manifest") or {}).get("locked_test_sha256") or "")
        check("stage5_archive_commitment", committed == actual_archive_sha256, committed)
    except Exception as exc:
        check("stage5_archive_commitment", False, f"{type(exc).__name__}: {exc}")
    check(
        "development_evidence_unchanged",
        manifest.get("development_evidence") == build_development_evidence(summary),
    )
    check("usage_scope_unchanged", manifest.get("usage_scope") == USAGE_SCOPE)

    check("key_file_present", key_file.is_file(), str(key_file))
    try:
        check(
            "key_file_unchanged",
            sha256_file(key_file) == locked.get("key_file_sha256"),
        )
    except Exception as exc:
        check("key_file_unchanged", False, f"{type(exc).__name__}: {exc}")
    check("locked_attempt_absent", bool(str(attempt)) and not attempt.exists(), str(attempt))
    check("locked_receipt_absent", bool(str(receipt)) and not receipt.exists(), str(receipt))
    check("locked_result_absent", bool(str(result)) and not result.exists(), str(result))
    check("final_report_absent", bool(str(final_report)) and not final_report.exists(), str(final_report))
    check("usage_ledger_absent", bool(str(usage_ledger)) and not usage_ledger.exists(), str(usage_ledger))
    check(
        "usage_ledger_path_fixed",
        usage_ledger == result.parent / LOCKED_USAGE_LEDGER_NAME,
        str(usage_ledger),
    )
    check("expected_task_count_20", locked.get("expected_task_count") == 20)

    expected_command = build_locked_command(
        archive=archive,
        key_file=key_file,
        lock_file=lock_file,
        receipt=receipt,
        skill=selected_actual.get("skill_path", selected_manifest.get("skill_path", "")),
        result=result,
    )
    check("command_unchanged", manifest.get("command") == expected_command)
    check(
        "locked_evaluator_present",
        (ROOT / "work/run_coding_hidden_v2_locked_eval.py").is_file(),
    )
    failed = [item["name"] for item in checks if not item["passed"]]
    return {
        "schema_version": 1,
        "status": "ready" if not failed else "blocked",
        "ready_to_consume": not failed,
        "failed_checks": failed,
        "checks": checks,
    }


def execute_stage7_manifest(
    manifest: dict[str, Any],
    *,
    confirmation: str,
    run_command: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    if confirmation != CONSUME_CONFIRMATION:
        raise ValueError(f"Confirmation must be exactly {CONSUME_CONFIRMATION}")
    readiness = validate_stage7_manifest(manifest)
    if readiness["status"] != "ready":
        raise ValueError("Stage 7 manifest is blocked: " + ",".join(readiness["failed_checks"]))

    locked = manifest["locked_test"]
    attempt_path = Path(locked["attempt"])
    attempt_path.parent.mkdir(parents=True, exist_ok=True)
    with attempt_path.open("x", encoding="utf-8") as handle:
        json.dump(
            {
                "schema_version": 1,
                "status": "started",
                "started_at": datetime.now(timezone.utc).isoformat(),
                "source_run_dir": manifest["source_run_dir"],
                "selected_seed": manifest["selected_candidate"]["seed"],
                "selected_skill_sha256": manifest["selected_candidate"]["skill_sha256"],
                "archive_sha256": locked["archive_sha256"],
                "execution_policy": manifest["execution_policy"],
            },
            handle,
            indent=2,
            sort_keys=True,
        )
        handle.write("\n")

    runner = subprocess.run if run_command is None else run_command
    returncode: int | None = None
    execution_error = ""
    try:
        completed = runner(
            manifest["command"],
            cwd=ROOT,
            check=False,
            env=build_locked_environment(),
        )
        returncode = int(completed.returncode)
    except Exception as exc:
        execution_error = f"{type(exc).__name__}: {exc}"

    receipt = load_json(Path(locked["receipt"]))
    result = load_json(Path(locked["result"]))
    selected = manifest["selected_candidate"]
    checks = {
        "command_returncode_zero": returncode == 0,
        "receipt_present": bool(receipt),
        "receipt_archive_matches": receipt.get("archive_sha256") == locked["archive_sha256"],
        "locked_result_complete": result.get("status") == "complete",
        "selected_skill_matches": result.get("skill_sha256") == selected["skill_sha256"],
        "full_task_count": result.get("task_count") == locked["expected_task_count"],
        "usage_ledger_matches": result.get("usage_ledger_path") == locked["usage_ledger"],
        "usage_ledger_present": Path(locked["usage_ledger"]).is_file(),
    }
    status = "complete" if all(checks.values()) else "failed"
    final = {
        "schema_version": 1,
        "status": status,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_run_dir": manifest["source_run_dir"],
        "selected_candidate": selected,
        "locked_attempt": load_json(attempt_path),
        "locked_receipt": receipt,
        "locked_result": compact_locked_result(result),
        "execution": {
            "returncode": returncode,
            "error": execution_error,
            "checks": checks,
            "attempts": EXECUTION_POLICY["attempts"],
        },
        "same_run_development_evidence": manifest["development_evidence"],
        "usage_scope": manifest["usage_scope"],
        "remaining_limitations": [
            "target-agent token usage is out of scope",
            "paper baselines Trace2Skill, TextGrad, GEPA, and EvoSkill are not in the local matrix",
            "cross-model, cross-harness, and cross-benchmark evidence remains future work",
        ],
    }
    final_path = Path(locked["final_report"])
    final_path.parent.mkdir(parents=True, exist_ok=True)
    final_path.write_text(json.dumps(final, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return final


def compact_locked_result(result: dict[str, Any]) -> dict[str, Any]:
    fields = (
        "status",
        "skill_path",
        "skill_sha256",
        "skill_bytes",
        "task_count",
        "task_accuracy",
        "average_score",
        "family_macro_accuracy",
        "contract_macro_accuracy",
        "contract_breakdown",
        "duration_seconds",
        "locked_task_file_sha256",
        "usage_ledger_path",
    )
    return {field: result[field] for field in fields if field in result}


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)

    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--run-dir", type=Path, required=True)
    prepare.add_argument("--key-file", type=Path, required=True)
    prepare.add_argument("--archive", type=Path, default=ROOT / "examples/coding-hidden-v2/test.enc")
    prepare.add_argument(
        "--lock",
        type=Path,
        default=ROOT / "examples/coding-hidden-v2/test.lock.json",
    )
    prepare.add_argument("--out", type=Path, required=True)

    check_parser = subparsers.add_parser("check")
    check_parser.add_argument("--manifest", type=Path, required=True)
    check_parser.add_argument("--out", type=Path)
    check_parser.add_argument("--quiet", action="store_true")

    run = subparsers.add_parser("run")
    run.add_argument("--manifest", type=Path, required=True)
    run.add_argument("--confirmation", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.action == "prepare":
        if args.out.exists():
            raise FileExistsError(f"Refusing to overwrite Stage 7 manifest: {args.out}")
        manifest = build_stage7_manifest(
            args.run_dir,
            key_file=args.key_file,
            archive=args.archive,
            lock_file=args.lock,
        )
        write_json(args.out, manifest)
        print(
            "stage7_prepare "
            f"status=ready seed={manifest['selected_candidate']['seed']} manifest={args.out}"
        )
        return 0

    manifest = load_json(args.manifest)
    if args.action == "check":
        report = validate_stage7_manifest(manifest)
        if args.out:
            write_json(args.out, report)
        if args.quiet:
            print(
                "stage7_check "
                f"status={report['status']} failed={len(report['failed_checks'])} "
                "missing=" + (",".join(report["failed_checks"]) or "none")
            )
        else:
            print(json.dumps(report, indent=2, sort_keys=True))
        return 0 if report["status"] == "ready" else 1

    final = execute_stage7_manifest(
        manifest,
        confirmation=args.confirmation,
    )
    print(
        "stage7_run "
        f"status={final['status']} attempts={final['execution']['attempts']} "
        f"returncode={final['execution']['returncode']}"
    )
    return 0 if final["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
