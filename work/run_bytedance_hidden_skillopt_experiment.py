"""Run hidden-test SkillOpt experiment with one external model and no stored key."""

from __future__ import annotations

import getpass
import os
import subprocess
import sys


def main() -> int:
    if sys.stdin.isatty():
        api_key = getpass.getpass("External API key: ").strip()
    else:
        api_key = sys.stdin.readline().strip()
    if not api_key:
        print("Missing API key", file=sys.stderr)
        return 2

    base_url = os.environ.get(
        "BYTEDANCE_MODEL_BASE_URL",
        "https://ark-cn-beijing.bytedance.net/api/v3",
    )
    model = os.environ.get("BYTEDANCE_MODEL_ID", "ep-20260507113406-9h6cz")
    env = os.environ.copy()
    env.update(
        {
            "EXTERNAL_AGENT_BASE_URL": base_url,
            "EXTERNAL_AGENT_MODEL": model,
            "EXTERNAL_AGENT_API_KEY": api_key,
            "EXTERNAL_AGENT_JSON_MODE": os.environ.get("EXTERNAL_AGENT_JSON_MODE", "0"),
            "EXTERNAL_AGENT_TIMEOUT": os.environ.get("EXTERNAL_AGENT_TIMEOUT", "180"),
            "EXTERNAL_LLM_BASE_URL": base_url,
            "EXTERNAL_LLM_MODEL": model,
            "EXTERNAL_LLM_API_KEY": api_key,
            "EXTERNAL_LLM_JSON_MODE": os.environ.get("EXTERNAL_LLM_JSON_MODE", "0"),
            "EXTERNAL_LLM_TIMEOUT": os.environ.get("EXTERNAL_LLM_TIMEOUT", "180"),
        }
    )

    output_dir = os.environ.get(
        "HIDDEN_SKILLOPT_OUT",
        "runs/coding-hidden-bytedance-skillopt",
    )
    command = [
        sys.executable,
        "-m",
        "textskill_optimizer.cli",
        "optimize",
        "--plugin",
        "coding",
        "--skill",
        os.environ.get("HIDDEN_SKILLOPT_SKILL", "examples/coding-hidden/skill.md"),
        "--train",
        os.environ.get("HIDDEN_SKILLOPT_TRAIN", "examples/coding-hidden/train.jsonl"),
        "--valid",
        os.environ.get("HIDDEN_SKILLOPT_VALID", "examples/coding-hidden/valid.jsonl"),
        "--holdout",
        os.environ.get("HIDDEN_SKILLOPT_HOLDOUT", "examples/coding-hidden/holdout.jsonl"),
        "--epochs",
        os.environ.get("HIDDEN_SKILLOPT_EPOCHS", "1"),
        "--editor-command",
        f"{sys.executable} examples/coding/openai_compatible_skill_editor.py",
        "--editor-timeout",
        os.environ.get("HIDDEN_SKILLOPT_EDITOR_TIMEOUT", "240"),
        "--out",
        output_dir,
    ]
    completed = subprocess.run(command, env=env, text=True, check=False)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
