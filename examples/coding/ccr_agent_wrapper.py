"""Claude Code Router wrapper for TextSkill coding-agent experiments.

The coding runner sets these environment variables before invoking this script:
  TEXTSKILL_REPO_DIR
  TEXTSKILL_SKILL_PATH
  TEXTSKILL_TASK_PATH
  TEXTSKILL_INSTRUCTION

Optional controls:
  CCR_AGENT_BIN            defaults to the local nvm ccr binary when present
  CCR_AGENT_TIMEOUT        subprocess timeout, defaults to 900 seconds
  CCR_AGENT_EXTRA_ARGS     extra ccr code args, shell-split
  CCR_AGENT_DRY_RUN        set 1 to print argv/stdin prompt JSON instead of running CCR
  CCR_AGENT_PERMISSION     defaults to bypassPermissions
"""

from __future__ import annotations

import json
import os
import signal
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any


DEFAULT_CCR_BIN = Path("/Users/bytedance/.nvm/versions/node/v22.22.2/bin/ccr")
DEFAULT_TIMEOUT_SECONDS = 900
DEFAULT_PATH_PREFIXES = (
    "/Users/bytedance/.local/bin",
    "/Users/bytedance/.nvm/versions/node/v22.22.2/bin",
)


def main() -> int:
    try:
        context = load_context_from_env()
        prompt = build_prompt(context)
        argv = build_ccr_argv(prompt)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    if os.environ.get("CCR_AGENT_DRY_RUN") == "1":
        print(json.dumps({"argv": argv, "stdin": prompt}, indent=2))
        return 0

    timeout = resolve_agent_timeout(context, "CCR_AGENT_TIMEOUT")
    try:
        completed = run_agent_process(
            argv,
            cwd=context["repo_dir"],
            env=build_subprocess_env(),
            input=prompt,
            timeout=timeout,
        )
    except FileNotFoundError:
        print(f"CCR binary not found: {argv[0]!r}. Set CCR_AGENT_BIN.", file=sys.stderr)
        return 127

    if completed.stdout:
        print(completed.stdout, end="")
    if completed.stderr:
        print(completed.stderr, end="", file=sys.stderr)
    if completed.timed_out:
        print(f"CCR agent timed out after {timeout}s", file=sys.stderr)
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

    return f"""You are the coding agent inside a TextSkill optimization task.

Work only inside this repository:
{context["repo_dir"]}

Task id:
{task_id}

Task instruction:
{context["instruction"]}

Public test command available to you:
{test_command}

Skill document to follow as process guidance:
```markdown
{context["skill_text"].strip()}
```

Operational constraints:
- Modify implementation/source files only.
- Do not edit tests unless the task explicitly asks for test changes.
- Do not modify files under `.textskill/`.
- Treat the skill document as binding requirements, not optional advice.
- Do not stop just because public tests pass; implement concrete skill rules even when public tests do not cover them.
- Before finishing, audit the changed code against every concrete skill rule and adjust the implementation if a rule is only partially satisfied.
- Run or reason through the public test command before finishing.
- Finish with a concise summary of changed files and test status.
"""


def build_ccr_argv(prompt: str) -> list[str]:
    binary = os.environ.get("CCR_AGENT_BIN", default_ccr_binary())
    argv = [
        binary,
        "code",
        "-p",
        "--output-format",
        "text",
        "--no-session-persistence",
    ]

    permission = os.environ.get("CCR_AGENT_PERMISSION", "bypassPermissions").strip()
    if permission:
        argv.extend(["--permission-mode", permission])

    extra = os.environ.get("CCR_AGENT_EXTRA_ARGS", "").strip()
    if extra:
        argv.extend(shlex.split(extra))

    return argv


class AgentProcessResult:
    def __init__(self, returncode: int, stdout: str, stderr: str, *, timed_out: bool) -> None:
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr
        self.timed_out = timed_out


def resolve_agent_timeout(context: dict[str, Any], env_name: str) -> int:
    override = os.environ.get(env_name, "").strip()
    if override:
        return int(override)
    task = context.get("task") if isinstance(context.get("task"), dict) else {}
    metadata = task.get("metadata") if isinstance(task.get("metadata"), dict) else {}
    return int(metadata.get("timeout_seconds", DEFAULT_TIMEOUT_SECONDS))


def run_agent_process(
    argv: list[str],
    *,
    cwd: Path,
    timeout: int,
    env: dict[str, str] | None = None,
    input: str | None = None,
) -> AgentProcessResult:
    process = subprocess.Popen(
        argv,
        cwd=cwd,
        env=env,
        stdin=subprocess.PIPE if input is not None else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(input=input, timeout=timeout)
        return AgentProcessResult(process.returncode, stdout or "", stderr or "", timed_out=False)
    except subprocess.TimeoutExpired:
        terminate_process_group(process)
        stdout, stderr = process.communicate()
        return AgentProcessResult(124, stdout or "", stderr or "", timed_out=True)


def terminate_process_group(process: subprocess.Popen) -> None:
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            return
        process.wait()


def default_ccr_binary() -> str:
    if DEFAULT_CCR_BIN.exists():
        return str(DEFAULT_CCR_BIN)
    return "ccr"


def build_subprocess_env() -> dict[str, str]:
    env = os.environ.copy()
    existing_path = env.get("PATH", "")
    prefixes = [p for p in DEFAULT_PATH_PREFIXES if p and p not in existing_path.split(os.pathsep)]
    if prefixes:
        env["PATH"] = os.pathsep.join([*prefixes, existing_path]) if existing_path else os.pathsep.join(prefixes)
    return env


if __name__ == "__main__":
    raise SystemExit(main())
