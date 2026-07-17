#!/usr/bin/env python3
"""Run the complete paper M6 gate without credentials or external calls."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from textskill_optimizer.paper.zero_cost_network import (  # noqa: E402
    install_zero_cost_network_guard,
    zero_cost_network_guard_active,
)


install_zero_cost_network_guard()

from textskill_optimizer.paper import assess_paper_provenance  # noqa: E402
from textskill_optimizer.paper import assess_zero_cost_gate  # noqa: E402
from textskill_optimizer.paper import canonical_json_sha256  # noqa: E402
from textskill_optimizer.paper import CodeIdentity  # noqa: E402
from textskill_optimizer.paper import ZeroCostGateEvidence  # noqa: E402


TEST_TARGETS = ("tests/conformance", "tests/provenance")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the zero-external-call paper conformance gate."
    )
    parser.add_argument(
        "--audit-only",
        action="store_true",
        help="Validate locked provenance without invoking pytest.",
    )
    parser.add_argument(
        "--receipt-path",
        type=Path,
        help="also persist the exact machine-readable receipt at this path",
    )
    args = parser.parse_args(argv)

    source_lock = _read_json(ROOT / "docs/papers/source-lock.json")
    prompt_snapshot = _read_json(ROOT / "docs/papers/prompt-snapshot-v1.json")
    assessment = assess_paper_provenance(
        source_lock=source_lock,
        prompt_snapshot=prompt_snapshot,
        paper_bytes=(ROOT / source_lock["paper"]["tracked_path"]).read_bytes(),
    ).require()
    code_commit, worktree_clean = _git_identity()
    receipt = {
        "schema_version": "paper-zero-cost-gate-v1",
        "status": "passed",
        "external_calls": 0,
        "network_guard_active": zero_cost_network_guard_active(),
        "paid_experiment_executed": False,
        "paid_development_authorized": False,
        "code_commit": code_commit,
        "worktree_clean": worktree_clean,
        "prompt_count": assessment.prompt_count,
        "prompt_snapshot_sha256": assessment.prompt_snapshot_sha256,
        "source_lock_sha256": canonical_json_sha256(source_lock),
        "golden_trace_sha256": _sha256_file(
            ROOT
            / "tests/conformance/golden/algorithm1-fast-loop-v1.json"
        ),
        "test_targets": list(TEST_TARGETS),
    }
    if args.audit_only:
        _emit_receipt(receipt, args.receipt_path)
        return 0

    completed = subprocess.run(
        (sys.executable, "-m", "pytest", "-q", *TEST_TARGETS),
        cwd=ROOT,
        env=_zero_cost_environment(),
        check=False,
    )
    final_commit, final_worktree_clean = _git_identity()
    decision = assess_zero_cost_gate(
        ZeroCostGateEvidence(
            test_returncode=completed.returncode,
            before=CodeIdentity(code_commit, clean=worktree_clean),
            after=CodeIdentity(final_commit, clean=final_worktree_clean),
            network_guard_active=zero_cost_network_guard_active(),
            external_calls=0,
        )
    )
    receipt["code_commit"] = final_commit
    receipt["worktree_clean"] = final_worktree_clean
    receipt["status"] = decision.status
    receipt["paid_development_authorized"] = decision.authorized
    receipt["violations"] = [
        {"code": item.code, "message": item.message}
        for item in decision.violations
    ]
    _emit_receipt(receipt, args.receipt_path)
    if decision.authorized:
        return 0
    return completed.returncode or 2


def _read_json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if type(payload) is not dict:
        raise ValueError(f"zero-cost gate requires an object: {path}")
    return payload


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _emit_receipt(receipt: dict, path: Path | None) -> None:
    encoded = json.dumps(receipt, sort_keys=True, separators=(",", ":"))
    if path is not None:
        destination = path.resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)


def _zero_cost_environment() -> dict[str, str]:
    environment = dict(os.environ)
    for key in tuple(environment):
        if key.endswith("_API_KEY") or key.endswith("_ACCESS_TOKEN"):
            environment.pop(key)
    environment.pop("PYTHONPATH", None)
    environment.pop("PYTEST_ADDOPTS", None)
    environment.pop("PYTEST_PLUGINS", None)
    environment["PYTHONPATH"] = os.pathsep.join(
        (str(ROOT / "scripts/zero_cost_guard"), str(ROOT))
    )
    environment["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    environment["TEXTSKILL_ZERO_COST_GATE"] = "1"
    return environment


def _git_identity() -> tuple[str | None, bool]:
    commit = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    status = subprocess.run(
        ("git", "status", "--porcelain", "--untracked-files=normal"),
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    if commit.returncode or status.returncode:
        return None, False
    return commit.stdout.strip(), not status.stdout.strip()


if __name__ == "__main__":
    raise SystemExit(main())
