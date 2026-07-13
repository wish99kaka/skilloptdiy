"""Run the Volcengine external editor experiment without storing secrets."""

from __future__ import annotations

import os
import subprocess
import sys
import getpass


def main() -> int:
    if sys.stdin.isatty():
        api_key = getpass.getpass("Volcengine API key: ").strip()
    else:
        api_key = sys.stdin.readline().strip()
    if not api_key:
        print("Missing API key on stdin", file=sys.stderr)
        return 2

    env = os.environ.copy()
    env.update(
        {
            "EXTERNAL_LLM_BASE_URL": "https://ark.cn-beijing.volces.com/api/v3/chat/completions",
            "EXTERNAL_LLM_MODEL": "ep-20260205202435-m8msm",
            "EXTERNAL_LLM_API_KEY": api_key,
            "EXTERNAL_LLM_JSON_MODE": os.environ.get("VOLCENGINE_JSON_MODE", "0"),
        }
    )
    output_dir = os.environ.get(
        "VOLCENGINE_RUN_OUT",
        "runs/coding-volcengine-holdout-rerun",
    )
    command = [
        sys.executable,
        "-m",
        "textskill_optimizer.cli",
        "optimize",
        "--plugin",
        "coding",
        "--skill",
        "examples/coding/skill.md",
        "--train",
        "examples/coding/train.jsonl",
        "--valid",
        "examples/coding/valid.jsonl",
        "--holdout",
        "examples/coding/holdout.jsonl",
        "--epochs",
        "1",
        "--editor-command",
        f"{sys.executable} examples/coding/openai_compatible_skill_editor.py",
        "--out",
        output_dir,
    ]
    completed = subprocess.run(command, env=env, text=True, check=False)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
