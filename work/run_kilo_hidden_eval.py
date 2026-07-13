"""Run hidden-test evaluation with Kilo as the coding agent."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    task_file = Path(os.environ.get("KILO_TASKS", "examples/coding-hidden/valid.jsonl"))
    skill_file = os.environ.get("KILO_SKILL", "examples/coding-hidden/skill.md")
    wrapper = ROOT / "examples/coding/kilo_agent_wrapper.py"
    tasks_path = build_kilo_tasks(task_file, wrapper)
    command = [
        sys.executable,
        "-m",
        "textskill_optimizer.cli",
        "evaluate",
        "--plugin",
        "coding",
        "--skill",
        skill_file,
        "--tasks",
        str(tasks_path),
    ]
    completed = subprocess.run(command, text=True, check=False)
    return completed.returncode


def build_kilo_tasks(task_file: Path, wrapper: Path) -> Path:
    source = (ROOT / task_file).resolve() if not task_file.is_absolute() else task_file.resolve()
    task_dir = source.parent
    task_limit = int(os.environ.get("KILO_TASK_LIMIT", "0") or "0")
    lines = []
    for line in source.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        if task_limit and len(lines) >= task_limit:
            break
        payload = json.loads(line)
        metadata = dict(payload.get("metadata") or {})
        metadata["agent_command"] = f"{sys.executable} {wrapper}"
        metadata["_task_dir"] = str(task_dir)
        metadata["timeout_seconds"] = int(
            os.environ.get("KILO_TASK_TIMEOUT", str(metadata.get("timeout_seconds", 900)))
        )
        payload["metadata"] = metadata
        lines.append(json.dumps(payload, separators=(",", ":")))

    tmp = tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        prefix="textskill-kilo-tasks-",
        suffix=".jsonl",
        delete=False,
    )
    with tmp:
        tmp.write("\n".join(lines))
        tmp.write("\n")
    return Path(tmp.name)


if __name__ == "__main__":
    raise SystemExit(main())
