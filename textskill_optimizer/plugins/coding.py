"""Command-driven coding-agent optimization plugin."""

from __future__ import annotations

import difflib
import json
import os
import shlex
import shutil
import subprocess
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from textskill_optimizer.interfaces import EDITOR_CAPABILITY_FULL_REPLACEMENT
from textskill_optimizer.models import EditProposal, Score, Task, TaskOutput, TaskResult
from textskill_optimizer.usage_ledger import append_usage_event, estimate_tokens_from_chars

DEFAULT_TIMEOUT_SECONDS = 60
SKILL_MARKER = "test-guided root-cause loop"


class CodingRunner:
    """Runs a coding task by copying a repo, invoking an agent, then testing."""

    def __init__(
        self,
        *,
        usage_ledger_path: str | Path | None = None,
        usage_context: dict[str, Any] | None = None,
    ) -> None:
        self.usage_ledger_path = Path(usage_ledger_path) if usage_ledger_path is not None else None
        self.usage_context = dict(usage_context or {})
        self._usage_scope: dict[str, Any] = {}

    @contextmanager
    def usage_scope(self, **extra: Any) -> Iterator[None]:
        previous = self._usage_scope
        self._usage_scope = {**previous, **extra}
        try:
            yield
        finally:
            self._usage_scope = previous

    def run(self, skill_text: str, task: Task) -> TaskOutput:
        metadata = task.metadata
        repo = metadata.get("repo")
        test_command = metadata.get("test_command")
        if not repo:
            raise ValueError(f"Task {task.id!r} metadata.repo is required")
        if not test_command:
            raise ValueError(f"Task {task.id!r} metadata.test_command is required")
        score_test_command = str(test_command)
        agent_test_command = str(metadata.get("agent_test_command") or score_test_command)

        source_repo = resolve_path(str(repo), metadata)
        if not source_repo.exists():
            raise ValueError(f"Task {task.id!r} repo does not exist: {source_repo}")

        timeout = int(metadata.get("timeout_seconds", DEFAULT_TIMEOUT_SECONDS))
        with tempfile.TemporaryDirectory(prefix=f"textskill-{task.id}-") as tmp:
            workdir = Path(tmp) / "repo"
            shutil.copytree(source_repo, workdir, ignore=shutil.ignore_patterns("__pycache__"))
            state_dir = workdir / ".textskill"
            state_dir.mkdir()

            skill_path = state_dir / "skill.md"
            task_path = state_dir / "task.json"
            skill_path.write_text(skill_text, encoding="utf-8")
            task_payload_text = json.dumps(
                build_agent_task_payload(task, agent_test_command),
                indent=2,
            )
            task_path.write_text(task_payload_text, encoding="utf-8")

            before_files = snapshot_files(workdir)
            pre_test = run_task_command(
                score_test_command,
                workdir,
                skill_path,
                task_path,
                task,
                timeout,
            )
            agent_command = resolve_agent_command(metadata)
            agent_result = run_agent_command(
                agent_command,
                workdir,
                skill_path,
                task_path,
                task,
                timeout,
            )
            post_test = run_task_command(
                score_test_command,
                workdir,
                skill_path,
                task_path,
                task,
                timeout,
            )
            after_files = snapshot_files(workdir)

            diff = unified_repo_diff(before_files, after_files)
            self.record_agent_usage(
                task=task,
                command=agent_command,
                agent_result=agent_result,
                skill_text=skill_text,
                task_payload_text=task_payload_text,
                repo_snapshot_chars=sum(len(value) for value in before_files.values()),
                diff=diff,
            )
            value = {
                "tests_passed": post_test["returncode"] == 0,
                "pre_tests_passed": pre_test["returncode"] == 0,
                "agent_returncode": agent_result["returncode"],
            }
            trace = [
                f"repo={source_repo}",
                f"pre_test_returncode={pre_test['returncode']}",
                f"agent_returncode={agent_result['returncode']}",
                f"post_test_returncode={post_test['returncode']}",
            ]
            return TaskOutput(
                value=value,
                trace=trace,
                metadata={
                    "repo": str(source_repo),
                    "test_command": score_test_command,
                    "agent_test_command": agent_test_command,
                    "agent_command": agent_command,
                    "pre_test": pre_test,
                    "agent": agent_result,
                    "post_test": post_test,
                    "diff": diff,
                },
            )

    def record_agent_usage(
        self,
        *,
        task: Task,
        command: str,
        agent_result: dict[str, Any],
        skill_text: str,
        task_payload_text: str,
        repo_snapshot_chars: int,
        diff: str,
    ) -> None:
        input_chars = len(skill_text) + len(task_payload_text) + repo_snapshot_chars
        output_chars = (
            len(str(agent_result.get("stdout") or ""))
            + len(str(agent_result.get("stderr") or ""))
            + len(diff)
        )
        context = {**self.usage_context, **self._usage_scope}
        append_usage_event(
            self.usage_ledger_path,
            {
                "kind": "target_agent_cli",
                "operation": "run_task",
                "context": context,
                "task_id": task.id,
                "benchmark_family": task.metadata.get("benchmark_family"),
                "command": command,
                "returncode": agent_result.get("returncode"),
                "timed_out": bool(agent_result.get("timed_out")),
                "duration_seconds": float(agent_result.get("duration_seconds") or 0.0),
                "input_chars": input_chars,
                "output_chars": output_chars,
                "skill_chars": len(skill_text),
                "task_payload_chars": len(task_payload_text),
                "repo_snapshot_chars": repo_snapshot_chars,
                "stdout_chars": len(str(agent_result.get("stdout") or "")),
                "stderr_chars": len(str(agent_result.get("stderr") or "")),
                "diff_chars": len(diff),
                "estimated_prompt_tokens": estimate_tokens_from_chars(input_chars),
                "estimated_completion_tokens": estimate_tokens_from_chars(output_chars),
            },
        )


class CodingScorer:
    """Scores coding tasks by post-agent test success."""

    def score(self, task: Task, output: TaskOutput) -> Score:
        expected = task.expected if isinstance(task.expected, dict) else {}
        expected_pass = expected.get("tests_passed", True)
        tests_passed = bool(output.value.get("tests_passed")) if isinstance(output.value, dict) else False
        success = tests_passed == expected_pass
        if success:
            return Score(1.0, True, "post-agent tests matched expectation")
        return Score(
            0.0,
            False,
            "post-agent tests failed",
            {
                "post_test_returncode": output.metadata.get("post_test", {}).get("returncode"),
            },
        )


def coding_retryable_anomaly_reasons(result: TaskResult) -> list[str]:
    """Classify agent/harness failures that must not be treated as skill failures."""

    if result.score.success:
        return []
    metadata = result.output.metadata
    value = result.output.value if isinstance(result.output.value, dict) else {}
    agent = metadata.get("agent") if isinstance(metadata.get("agent"), dict) else {}
    post_test = metadata.get("post_test") if isinstance(metadata.get("post_test"), dict) else {}
    agent_returncode = _as_int(value.get("agent_returncode", agent.get("returncode")))
    post_test_returncode = _as_int(post_test.get("returncode"))
    timed_out = bool(agent.get("timed_out")) or agent_returncode == 124
    diff = str(metadata.get("diff") or "")
    stdout = str(agent.get("stdout") or "")
    stderr = str(agent.get("stderr") or "")
    agent_text = f"{stdout}\n{stderr}"
    reasons: list[str] = []

    if timed_out:
        reasons.append("agent_timeout")
    elif agent_returncode is not None and agent_returncode != 0:
        reasons.append("agent_nonzero_returncode")
    if post_test_returncode not in (None, 0) and not diff.strip():
        reasons.append("failed_post_test_without_repo_change")
    if "<seed:tool" in agent_text and "</seed:tool_call>" in agent_text and not diff.strip():
        reasons.append("malformed_tool_call_without_repo_change")
    if not stdout.strip() and not stderr.strip() and not diff.strip():
        reasons.append("empty_agent_output_without_repo_change")
    return reasons


def _as_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


class CodingSkillEditor:
    """Adds a compact coding loop rule after failed training trajectories."""

    capabilities = frozenset({EDITOR_CAPABILITY_FULL_REPLACEMENT})

    def propose(
        self,
        skill_text: str,
        train_results: list[TaskResult],
        *,
        epoch: int,
        rejected_buffer: list[dict[str, Any]] | None = None,
        meta_skill: str = "",
        optimizer_controls: dict[str, Any] | None = None,
    ) -> list[EditProposal]:
        failures = [result for result in train_results if not result.score.success]
        if not failures:
            return []
        if SKILL_MARKER in skill_text.casefold():
            return []

        edited = append_coding_loop(skill_text)
        failed_ids = ", ".join(result.task.id for result in failures)
        return [
            EditProposal(
                name=f"coding-root-cause-loop-epoch-{epoch}",
                skill_text=edited,
                rationale=(
                    "Added a test-guided root-cause loop because training tasks "
                    f"failed after the agent ran: {failed_ids}"
                ),
                metadata={"failed_task_ids": [result.task.id for result in failures]},
            )
        ]


def build_runner() -> CodingRunner:
    return CodingRunner()


def build_scorer() -> CodingScorer:
    return CodingScorer()


def build_editor() -> CodingSkillEditor:
    return CodingSkillEditor()


def resolve_agent_command(metadata: dict[str, Any]) -> str:
    command = metadata.get("agent_command") or os.environ.get("TEXTSKILL_CODING_AGENT_CMD")
    if not command:
        raise ValueError(
            "Coding tasks require metadata.agent_command or TEXTSKILL_CODING_AGENT_CMD"
        )
    return str(command)


def build_agent_task_payload(task: Task, agent_test_command: str) -> dict[str, Any]:
    payload = task.to_dict()
    metadata = dict(payload.get("metadata") or {})
    metadata["test_command"] = agent_test_command
    for hidden_key in ("agent_test_command", "hidden_test_command", "score_test_command"):
        metadata.pop(hidden_key, None)
    payload["metadata"] = metadata
    return payload


def run_task_command(
    command: str,
    workdir: Path,
    skill_path: Path,
    task_path: Path,
    task: Task,
    timeout: int,
    *,
    env: dict[str, str] | None = None,
) -> dict[str, Any]:
    formatted = format_task_command(command, workdir, skill_path, task_path, task)
    return run_command(formatted, workdir, timeout, env=env)


def run_agent_command(
    command: str,
    workdir: Path,
    skill_path: Path,
    task_path: Path,
    task: Task,
    timeout: int,
) -> dict[str, Any]:
    env = os.environ.copy()
    env.update(
        {
            "TEXTSKILL_REPO_DIR": str(workdir),
            "TEXTSKILL_SKILL_PATH": str(skill_path),
            "TEXTSKILL_TASK_PATH": str(task_path),
            "TEXTSKILL_INSTRUCTION": task.input,
        }
    )
    formatted = format_task_command(command, workdir, skill_path, task_path, task)
    return run_command(formatted, workdir, timeout, env=env)


def format_task_command(
    command: str,
    workdir: Path,
    skill_path: Path,
    task_path: Path,
    task: Task,
) -> str:
    return command.format(
        repo=shlex.quote(str(workdir)),
        skill=shlex.quote(str(skill_path)),
        task=shlex.quote(str(task_path)),
        task_dir=shlex.quote(str(Path(task.metadata.get("_task_dir", ".")).resolve())),
        instruction=shlex.quote(task.input),
    )


def resolve_path(value: str, metadata: dict[str, Any]) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    task_dir = metadata.get("_task_dir")
    if task_dir:
        return (Path(str(task_dir)) / path).resolve()
    return path.resolve()


def run_command(
    command: str,
    cwd: Path,
    timeout: int,
    *,
    env: dict[str, str] | None = None,
) -> dict[str, Any]:
    started = time.monotonic()
    try:
        completed = subprocess.run(
            shlex.split(command),
            cwd=cwd,
            env=env,
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
        return {
            "command": command,
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
            "timed_out": False,
            "duration_seconds": time.monotonic() - started,
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "command": command,
            "returncode": 124,
            "stdout": exc.stdout or "",
            "stderr": exc.stderr or f"Command timed out after {timeout}s",
            "timed_out": True,
            "duration_seconds": time.monotonic() - started,
        }


def snapshot_files(root: Path) -> dict[str, str]:
    files: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if path.is_dir() or should_ignore(path, root):
            continue
        try:
            files[str(path.relative_to(root))] = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
    return files


def should_ignore(path: Path, root: Path) -> bool:
    relative_parts = path.relative_to(root).parts
    ignored = {".textskill", "__pycache__", ".pytest_cache", ".git"}
    return any(part in ignored for part in relative_parts)


def unified_repo_diff(before: dict[str, str], after: dict[str, str]) -> str:
    chunks: list[str] = []
    for name in sorted(set(before) | set(after)):
        if before.get(name) == after.get(name):
            continue
        before_lines = before.get(name, "").splitlines(keepends=True)
        after_lines = after.get(name, "").splitlines(keepends=True)
        chunks.extend(
            difflib.unified_diff(
                before_lines,
                after_lines,
                fromfile=f"a/{name}",
                tofile=f"b/{name}",
            )
        )
    return "".join(chunks)


def append_coding_loop(skill_text: str) -> str:
    block = """## Coding Agent Loop

- Use a test-guided root-cause loop: run or inspect the provided test command, read the exact failure, change implementation files instead of tests, then rerun the same command.
- Prefer the smallest implementation change that explains the failing assertion.
- Do not declare success until the provided test command passes.
"""
    return skill_text.rstrip() + "\n\n" + block
