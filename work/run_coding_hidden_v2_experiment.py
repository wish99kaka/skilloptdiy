#!/usr/bin/env python3
"""Run coding-hidden-v2 with Coco unchanged and a separate external skill editor."""

from __future__ import annotations

import getpass
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Sequence


ROOT = Path(__file__).resolve().parent.parent
MATRIX = ROOT / "work/run_coding_hidden_v2_matrix.py"
EDITOR = ROOT / "examples/coding/openai_compatible_skill_editor.py"
DEFAULT_BASE_URL = "https://ark-cn-beijing.bytedance.net/api/v3"
DEFAULT_MODEL = "ep-20260507113406-9h6cz"


def main(argv: Sequence[str] | None = None) -> int:
    args = list(argv if argv is not None else sys.argv[1:])
    api_key = read_api_key()
    if not api_key:
        print("Missing external editor API key", file=sys.stderr)
        return 2

    env = build_external_editor_env(os.environ, api_key)
    if args == ["--editor-health-check"]:
        return run_editor_health_check(env)

    print(
        f"[coding-hidden-v2] external optimizer={env['EXTERNAL_LLM_MODEL']}; "
        "Coco model=local configured default (no override)",
        file=sys.stderr,
        flush=True,
    )
    completed = subprocess.run(
        [sys.executable, str(MATRIX), *args],
        cwd=ROOT,
        env=env,
        text=True,
        check=False,
    )
    return completed.returncode


def read_api_key() -> str:
    existing = os.environ.get("BYTEDANCE_API_KEY", "").strip()
    if existing:
        return existing
    if sys.stdin.isatty():
        return getpass.getpass("External optimizer API key: ").strip()
    return sys.stdin.readline().strip()


def build_external_editor_env(source: dict[str, str], api_key: str) -> dict[str, str]:
    env = dict(source)
    env.update(
        {
            "EXTERNAL_LLM_BASE_URL": source.get("BYTEDANCE_MODEL_BASE_URL", DEFAULT_BASE_URL),
            "EXTERNAL_LLM_MODEL": source.get("BYTEDANCE_MODEL_ID", DEFAULT_MODEL),
            "EXTERNAL_LLM_API_KEY": api_key,
            "EXTERNAL_LLM_JSON_MODE": source.get("EXTERNAL_LLM_JSON_MODE", "0"),
            "EXTERNAL_LLM_TIMEOUT": source.get("EXTERNAL_LLM_TIMEOUT", "240"),
        }
    )
    return env


def run_editor_health_check(env: dict[str, str]) -> int:
    payload = {
        "operation": "reflect",
        "epoch": 1,
        "skill_text": "# Skill\nInspect the contract, implement the smallest complete fix, and verify it.",
        "train_results": [],
        "optimizer_controls": {"atomic_edit_budget": 1},
    }
    completed = subprocess.run(
        [sys.executable, str(EDITOR)],
        cwd=ROOT,
        env=env,
        input=json.dumps(payload),
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        print(completed.stderr, file=sys.stderr, end="")
        return completed.returncode
    try:
        response = json.loads(completed.stdout)
    except json.JSONDecodeError:
        print("External editor health check returned invalid JSON", file=sys.stderr)
        return 1
    if not isinstance(response, dict) or not isinstance(response.get("proposals"), list):
        print("External editor health check returned an invalid proposal envelope", file=sys.stderr)
        return 1
    print("external-editor-health=ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
