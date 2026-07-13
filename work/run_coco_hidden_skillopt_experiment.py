"""Run hidden-test SkillOpt with Coco as agent and external model as editor."""

from __future__ import annotations

import getpass
import os
import subprocess
import sys
from pathlib import Path

from run_coco_hidden_eval import build_coco_tasks


ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    log_stage("waiting for external editor API key")
    if sys.stdin.isatty():
        api_key = getpass.getpass("External editor API key: ").strip()
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
    log_stage(f"configured external editor model={model}")
    env = os.environ.copy()
    env.update(
        {
            "EXTERNAL_LLM_BASE_URL": base_url,
            "EXTERNAL_LLM_MODEL": model,
            "EXTERNAL_LLM_API_KEY": api_key,
            "EXTERNAL_LLM_JSON_MODE": os.environ.get("EXTERNAL_LLM_JSON_MODE", "0"),
            "EXTERNAL_LLM_TIMEOUT": os.environ.get("EXTERNAL_LLM_TIMEOUT", "240"),
        }
    )

    log_stage("building Coco train task file")
    train = build_coco_tasks(
        Path(os.environ.get("HIDDEN_COCO_TRAIN", "examples/coding-hidden/train.jsonl")),
        ROOT / "examples/coding/coco_agent_wrapper.py",
    )
    log_stage(f"train tasks={train}")
    log_stage("building Coco validation task file")
    valid = build_coco_tasks(
        Path(os.environ.get("HIDDEN_COCO_VALID", "examples/coding-hidden/valid.jsonl")),
        ROOT / "examples/coding/coco_agent_wrapper.py",
    )
    log_stage(f"validation tasks={valid}")
    log_stage("building Coco holdout task file")
    holdout = build_coco_tasks(
        Path(os.environ.get("HIDDEN_COCO_HOLDOUT", "examples/coding-hidden/holdout.jsonl")),
        ROOT / "examples/coding/coco_agent_wrapper.py",
    )
    log_stage(f"holdout tasks={holdout}")
    output_dir = os.environ.get(
        "HIDDEN_COCO_SKILLOPT_OUT",
        "runs/coding-hidden-coco-skillopt",
    )
    command = [
        sys.executable,
        "-m",
        "textskill_optimizer.cli",
        "optimize",
        "--plugin",
        "coding",
        "--skill",
        os.environ.get("HIDDEN_COCO_SKILL", "examples/coding-hidden/skill.md"),
        "--train",
        str(train),
        "--valid",
        str(valid),
        "--holdout",
        str(holdout),
        "--epochs",
        os.environ.get("HIDDEN_COCO_EPOCHS", "1"),
        "--editor-command",
        f"{sys.executable} examples/coding/openai_compatible_skill_editor.py",
        "--editor-timeout",
        os.environ.get("HIDDEN_COCO_EDITOR_TIMEOUT", "300"),
        "--out",
        output_dir,
    ]
    log_stage(f"starting SkillOpt output_dir={output_dir}")
    completed = subprocess.run(command, env=env, text=True, check=False)
    log_stage(f"SkillOpt finished returncode={completed.returncode}")
    return completed.returncode


def log_stage(message: str) -> None:
    print(f"[coco-skillopt] {message}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
