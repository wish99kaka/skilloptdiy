"""Run hidden-test evaluation with Coco as the coding agent."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Sequence


ROOT = Path(__file__).resolve().parent.parent


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    wrapper = ROOT / "examples/coding/coco_agent_wrapper.py"
    tasks_path = build_coco_tasks(
        args.tasks,
        wrapper,
        task_limit=args.task_limit,
        timeout_seconds=args.task_timeout,
    )
    command = [
        sys.executable,
        "-m",
        "textskill_optimizer.cli",
        "evaluate",
        "--plugin",
        "coding",
        "--skill",
        str(args.skill),
        "--tasks",
        str(tasks_path),
    ]
    completed = subprocess.run(command, text=True, check=False)
    return completed.returncode


def build_coco_tasks(
    task_file: Path,
    wrapper: Path,
    *,
    task_limit: int | None = None,
    timeout_seconds: int | None = None,
) -> Path:
    source = (ROOT / task_file).resolve() if not task_file.is_absolute() else task_file.resolve()
    task_dir = source.parent
    if task_limit is None:
        task_limit = int(os.environ.get("COCO_TASK_LIMIT", "0") or "0")
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
        configured_timeout = timeout_seconds
        if configured_timeout is None:
            configured_timeout = int(
                os.environ.get("COCO_TASK_TIMEOUT", str(metadata.get("timeout_seconds", 900)))
            )
        metadata["timeout_seconds"] = configured_timeout
        payload["metadata"] = metadata
        lines.append(json.dumps(payload, separators=(",", ":")))

    tmp = tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        prefix="textskill-coco-tasks-",
        suffix=".jsonl",
        delete=False,
    )
    with tmp:
        tmp.write("\n".join(lines))
        tmp.write("\n")
    return Path(tmp.name)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tasks",
        type=Path,
        default=Path(os.environ.get("COCO_TASKS", "examples/coding-hidden/valid.jsonl")),
    )
    parser.add_argument(
        "--skill",
        type=Path,
        default=Path(os.environ.get("COCO_SKILL", "examples/coding-hidden/skill.md")),
    )
    parser.add_argument(
        "--task-limit",
        type=int,
        default=int(os.environ.get("COCO_TASK_LIMIT", "0") or "0"),
    )
    timeout = os.environ.get("COCO_TASK_TIMEOUT", "").strip()
    parser.add_argument("--task-timeout", type=int, default=int(timeout) if timeout else None)
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
