"""Run external-model coding agent eval without storing secrets."""

from __future__ import annotations

import getpass
import os
import subprocess
import sys


def main() -> int:
    if sys.stdin.isatty():
        api_key = getpass.getpass("External agent API key: ").strip()
    else:
        api_key = sys.stdin.readline().strip()
    if not api_key:
        print("Missing API key", file=sys.stderr)
        return 2

    env = os.environ.copy()
    env.update(
        {
            "EXTERNAL_AGENT_BASE_URL": "https://ark-cn-beijing.bytedance.net/api/v3",
            "EXTERNAL_AGENT_MODEL": "ep-20260507113406-9h6cz",
            "EXTERNAL_AGENT_API_KEY": api_key,
            "EXTERNAL_AGENT_JSON_MODE": "0",
            "EXTERNAL_AGENT_TIMEOUT": "180",
        }
    )
    tasks = os.environ.get(
        "EXTERNAL_AGENT_TASKS",
        "examples/coding-external-agent/valid.jsonl",
    )
    skill = os.environ.get(
        "EXTERNAL_AGENT_SKILL",
        "examples/coding-external-agent/skill.md",
    )
    command = [
        sys.executable,
        "-m",
        "textskill_optimizer.cli",
        "evaluate",
        "--plugin",
        "coding",
        "--skill",
        skill,
        "--tasks",
        tasks,
    ]
    completed = subprocess.run(command, env=env, text=True, check=False)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
