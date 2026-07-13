#!/usr/bin/env python3
"""Run public tests plus hidden scorer-only tests for a copied fixture repo."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) != 3:
        print("Usage: run_hidden_tests.py <fixture-name> <repo-dir>", file=sys.stderr)
        return 2

    fixture_name = sys.argv[1]
    repo_dir = Path(sys.argv[2]).expanduser().resolve()
    hidden_dir = Path(__file__).resolve().parent / "hidden" / fixture_name
    if not repo_dir.is_dir():
        print(f"Repo does not exist: {repo_dir}", file=sys.stderr)
        return 2
    if not hidden_dir.is_dir():
        print(f"Hidden tests do not exist: {hidden_dir}", file=sys.stderr)
        return 2

    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_dir) + os.pathsep + env.get("PYTHONPATH", "")
    public_result = run_unittest(repo_dir, repo_dir / "tests", env, "public")
    hidden_result = run_unittest(repo_dir, hidden_dir, env, "hidden")
    return 0 if public_result == 0 and hidden_result == 0 else 1


def run_unittest(repo_dir: Path, test_dir: Path, env: dict[str, str], label: str) -> int:
    command = [sys.executable, "-m", "unittest", "discover", "-s", str(test_dir)]
    print(f"== {label} tests ==")
    completed = subprocess.run(
        command,
        cwd=repo_dir,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.stdout:
        print(completed.stdout, end="")
    if completed.stderr:
        print(completed.stderr, end="", file=sys.stderr)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
