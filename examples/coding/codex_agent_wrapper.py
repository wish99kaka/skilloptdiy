"""Codex CLI wrapper for TextSkill coding-agent experiments.

The coding runner sets these environment variables before invoking this script:
  TEXTSKILL_REPO_DIR
  TEXTSKILL_SKILL_PATH
  TEXTSKILL_TASK_PATH
  TEXTSKILL_INSTRUCTION

Optional controls:
  CODEX_AGENT_BIN          defaults to "codex"
  CODEX_AGENT_MODEL        passed as --model when set
  CODEX_AGENT_SANDBOX      defaults to "workspace-write"
  CODEX_AGENT_APPROVAL     defaults to "never"
  CODEX_AGENT_EXTRA_ARGS   extra codex exec args, shell-split
  CODEX_AGENT_TIMEOUT      defaults to 600 seconds
  CODEX_AGENT_DRY_RUN      set 1 to print argv/prompt JSON instead of running Codex
  CODEX_AGENT_PROMPT_MODE  minimal or guided. Defaults to minimal.
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any


DEFAULT_TIMEOUT_SECONDS = 600


def main() -> int:
    try:
        context = load_context_from_env()
        prompt = build_prompt(context)
        argv = build_codex_argv(context)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    if os.environ.get("CODEX_AGENT_DRY_RUN") == "1":
        print(json.dumps({"argv": argv, "prompt": prompt}, indent=2))
        return 0

    timeout = int(os.environ.get("CODEX_AGENT_TIMEOUT", str(DEFAULT_TIMEOUT_SECONDS)))
    try:
        completed = subprocess.run(
            argv,
            input=prompt,
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
    except FileNotFoundError:
        print(
            f"Codex binary not found: {argv[0]!r}. Set CODEX_AGENT_BIN.",
            file=sys.stderr,
        )
        return 127
    except subprocess.TimeoutExpired as exc:
        if exc.stdout:
            print(exc.stdout, end="")
        if exc.stderr:
            print(exc.stderr, end="", file=sys.stderr)
        print(f"Codex agent timed out after {timeout}s", file=sys.stderr)
        return 124

    if completed.stdout:
        print(completed.stdout, end="")
    if completed.stderr:
        print(completed.stderr, end="", file=sys.stderr)
    return completed.returncode


def load_context_from_env() -> dict[str, Any]:
    repo_dir = required_path_env("TEXTSKILL_REPO_DIR", must_be_dir=True)
    skill_path = required_path_env("TEXTSKILL_SKILL_PATH", must_be_file=True)
    task_path = required_path_env("TEXTSKILL_TASK_PATH", must_be_file=True)
    instruction = os.environ.get("TEXTSKILL_INSTRUCTION", "").strip()
    if not instruction:
        raise ValueError("TEXTSKILL_INSTRUCTION is required")

    task = json.loads(task_path.read_text(encoding="utf-8"))
    if not isinstance(task, dict):
        raise ValueError("TEXTSKILL_TASK_PATH must contain a JSON object")

    return {
        "repo_dir": repo_dir,
        "skill_path": skill_path,
        "task_path": task_path,
        "instruction": instruction,
        "skill_text": skill_path.read_text(encoding="utf-8"),
        "task": task,
    }


def required_path_env(
    name: str,
    *,
    must_be_dir: bool = False,
    must_be_file: bool = False,
) -> Path:
    raw = os.environ.get(name, "").strip()
    if not raw:
        raise ValueError(f"{name} is required")
    path = Path(raw).expanduser().resolve()
    if must_be_dir and not path.is_dir():
        raise ValueError(f"{name} must point to a directory: {path}")
    if must_be_file and not path.is_file():
        raise ValueError(f"{name} must point to a file: {path}")
    return path


def build_prompt(context: dict[str, Any]) -> str:
    task = context["task"]
    metadata = task.get("metadata") if isinstance(task.get("metadata"), dict) else {}
    test_command = metadata.get("test_command", "")
    task_id = task.get("id", "unknown")

    mode = os.environ.get("CODEX_AGENT_PROMPT_MODE", "minimal").strip().lower()
    if mode not in {"minimal", "guided"}:
        raise ValueError("CODEX_AGENT_PROMPT_MODE must be 'minimal' or 'guided'")

    constraints = build_constraints(mode)

    return f"""You are the coding agent inside a TextSkill optimization task.

Repository:
{context["repo_dir"]}

Task id:
{task_id}

Task instruction:
{context["instruction"]}

Test command:
{test_command}

Skill document to follow:
```markdown
{context["skill_text"].strip()}
```

Operational constraints:
{constraints}
"""


def build_constraints(mode: str) -> str:
    base = [
        "- Work only inside the repository path above.",
        "- Follow the skill document as the primary process guidance.",
        "- Do not modify files under `.textskill/`.",
        "- Do not edit tests unless the task explicitly asks for test changes.",
        "- The test command above is available for verification.",
        "- Finish with a concise summary of changed files and whether tests pass.",
    ]
    if mode == "minimal":
        return "\n".join(base)

    guided = [
        "- Use the test command above as the source of truth.",
        "- Prefer the smallest implementation change that explains the failing test.",
        "- Before finishing, run the test command when feasible.",
    ]
    return "\n".join(base + guided)


def build_codex_argv(context: dict[str, Any]) -> list[str]:
    binary = os.environ.get("CODEX_AGENT_BIN", "codex")
    sandbox = os.environ.get("CODEX_AGENT_SANDBOX", "workspace-write")
    approval = os.environ.get("CODEX_AGENT_APPROVAL", "never")
    argv = [
        binary,
        "--ask-for-approval",
        approval,
        "exec",
        "--cd",
        str(context["repo_dir"]),
        "--skip-git-repo-check",
        "--ephemeral",
        "--sandbox",
        sandbox,
    ]
    model = os.environ.get("CODEX_AGENT_MODEL", "").strip()
    if model:
        argv.extend(["--model", model])

    extra = os.environ.get("CODEX_AGENT_EXTRA_ARGS", "").strip()
    if extra:
        argv.extend(shlex.split(extra))

    argv.append("-")
    return argv


if __name__ == "__main__":
    raise SystemExit(main())
