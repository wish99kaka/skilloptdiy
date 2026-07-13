"""OpenAI-compatible coding agent wrapper for TextSkill experiments.

This wrapper lets an external Chat Completions-compatible model edit a copied
repo. It does not give the model shell access. The model returns JSON file
edits; this script applies them, then the runner executes the task test command.

Required environment:
  TEXTSKILL_REPO_DIR
  TEXTSKILL_SKILL_PATH
  TEXTSKILL_TASK_PATH
  TEXTSKILL_INSTRUCTION
  EXTERNAL_AGENT_BASE_URL   e.g. https://ark-cn-beijing.bytedance.net/api/v3
  EXTERNAL_AGENT_MODEL      e.g. ep-...

Optional environment:
  EXTERNAL_AGENT_API_KEY      defaults to "not-needed"
  EXTERNAL_AGENT_JSON_MODE    defaults to 1; set 0 if response_format is unsupported
  EXTERNAL_AGENT_TEMPERATURE  defaults to 0.1
  EXTERNAL_AGENT_TIMEOUT      defaults to 120
  EXTERNAL_AGENT_DRY_RUN      set 1 to print request JSON and skip API call
  EXTERNAL_AGENT_MAX_FILE_CHARS defaults to 12000
"""

from __future__ import annotations

import json
import os
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


SYSTEM_PROMPT = """You are a coding agent fixing a small repository.

Return only JSON with this shape:
{
  "edits": [
    {
      "path": "relative/path.py",
      "content": "full replacement file content"
    }
  ],
  "summary": "short explanation"
}

Rules:
- The user payload includes skill_text; treat it as the process instructions for this experiment.
- Edit implementation/source files only.
- Do not edit tests unless the task explicitly asks for test changes.
- Do not edit files under .textskill/.
- Return full replacement file content, not a patch.
- Make the smallest source change that satisfies the public tests and the inferred full behavior.
- Do not hard-code only the public examples when the function name and code imply broader behavior.
- If no edit is needed, return {"edits": [], "summary": "..."}.
"""


def main() -> int:
    try:
        context = load_context_from_env()
        request_payload, url, api_key, timeout = build_request_from_env(context)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    if os.environ.get("EXTERNAL_AGENT_DRY_RUN") == "1":
        print(
            json.dumps(
                {
                    "url": url,
                    "model": request_payload.get("model"),
                    "uses_json_mode": "response_format" in request_payload,
                    "request_payload": request_payload,
                },
                indent=2,
            )
        )
        return 0

    try:
        response_payload = call_chat_completions(url, api_key, request_payload, timeout)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        print(f"External agent API error {exc.code}: {body}", file=sys.stderr)
        return 1
    except urllib.error.URLError as exc:
        print(f"External agent connection error: {exc.reason}", file=sys.stderr)
        return 1

    content = extract_chat_message_content(response_payload)
    if not content:
        print("External agent response did not contain assistant content", file=sys.stderr)
        return 1

    try:
        edit_payload = json.loads(extract_json_text(content))
    except json.JSONDecodeError as exc:
        print(f"External agent output was not JSON: {exc}", file=sys.stderr)
        print(content, file=sys.stderr)
        return 1

    try:
        applied = apply_edits(context["repo_dir"], edit_payload)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print(json.dumps({"applied": applied, "summary": edit_payload.get("summary", "")}))
    return 0


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
        "skill_text": skill_path.read_text(encoding="utf-8"),
        "task": task,
        "instruction": instruction,
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


def build_request_from_env(
    context: dict[str, Any],
) -> tuple[dict[str, Any], str, str, float]:
    base_url = os.environ.get("EXTERNAL_AGENT_BASE_URL", "").rstrip("/")
    model = os.environ.get("EXTERNAL_AGENT_MODEL", "").strip()
    if not base_url:
        raise ValueError("EXTERNAL_AGENT_BASE_URL is required")
    if not model:
        raise ValueError("EXTERNAL_AGENT_MODEL is required")
    api_key = os.environ.get("EXTERNAL_AGENT_API_KEY", "not-needed")
    timeout = float(os.environ.get("EXTERNAL_AGENT_TIMEOUT", "120"))
    url = normalize_chat_completions_url(base_url)
    return build_chat_request_payload(context, model=model), url, api_key, timeout


def normalize_chat_completions_url(base_url: str) -> str:
    cleaned = base_url.rstrip("/")
    if cleaned.endswith("/chat/completions"):
        return cleaned
    return cleaned + "/chat/completions"


def build_chat_request_payload(context: dict[str, Any], *, model: str) -> dict[str, Any]:
    task = context["task"]
    metadata = task.get("metadata") if isinstance(task.get("metadata"), dict) else {}
    test_command = metadata.get("test_command", "")
    repo_snapshot = collect_repo_snapshot(context["repo_dir"])
    user_payload = {
        "task_id": task.get("id"),
        "instruction": context["instruction"],
        "test_command": test_command,
        "repo_files": repo_snapshot,
        "initial_test_result": run_test_command(context["repo_dir"], test_command),
    }
    request: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    "Skill instructions for this experiment:\n\n"
                    f"{context['skill_text'].strip()}\n\n"
                    "Task payload:\n"
                    f"{json.dumps(user_payload, indent=2)}"
                ),
            },
        ],
        "temperature": float(os.environ.get("EXTERNAL_AGENT_TEMPERATURE", "0.1")),
    }
    if os.environ.get("EXTERNAL_AGENT_JSON_MODE", "1") != "0":
        request["response_format"] = {"type": "json_object"}
    return request


def collect_repo_snapshot(repo_dir: Path) -> list[dict[str, str]]:
    max_chars = int(os.environ.get("EXTERNAL_AGENT_MAX_FILE_CHARS", "12000"))
    files: list[dict[str, str]] = []
    for path in sorted(repo_dir.rglob("*")):
        if path.is_dir() or should_skip(path, repo_dir):
            continue
        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if len(content) > max_chars:
            content = content[:max_chars] + "\n...[truncated]"
        files.append({"path": str(path.relative_to(repo_dir)), "content": content})
    return files


def should_skip(path: Path, repo_dir: Path) -> bool:
    parts = path.relative_to(repo_dir).parts
    return any(part in {".git", ".textskill", "__pycache__", ".pytest_cache"} for part in parts)


def run_test_command(repo_dir: Path, test_command: str) -> dict[str, Any]:
    if not test_command:
        return {"returncode": None, "stdout": "", "stderr": "No test command provided"}
    import shlex
    import subprocess

    completed = subprocess.run(
        shlex.split(test_command),
        cwd=repo_dir,
        text=True,
        capture_output=True,
        check=False,
        timeout=60,
    )
    return {
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def call_chat_completions(
    url: str,
    api_key: str,
    request_payload: dict[str, Any],
    timeout: float,
) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(request_payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def extract_chat_message_content(response_payload: dict[str, Any]) -> str:
    choices = response_payload.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    first = choices[0]
    message = first.get("message") if isinstance(first, dict) else None
    if not isinstance(message, dict):
        return ""
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            item.get("text", "")
            for item in content
            if isinstance(item, dict) and isinstance(item.get("text"), str)
        )
    return ""


def extract_json_text(content: str) -> str:
    stripped = content.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", stripped, re.DOTALL)
    if fenced:
        return fenced.group(1).strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        return stripped
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start >= 0 and end > start:
        return stripped[start : end + 1]
    return stripped


def apply_edits(repo_dir: Path, payload: dict[str, Any]) -> list[str]:
    repo_dir = repo_dir.resolve()
    edits = payload.get("edits")
    if not isinstance(edits, list):
        raise ValueError("External agent JSON must contain an edits list")
    applied: list[str] = []
    for index, edit in enumerate(edits, 1):
        if not isinstance(edit, dict):
            raise ValueError(f"Edit {index} must be a JSON object")
        relative = edit.get("path")
        content = edit.get("content")
        if not isinstance(relative, str) or not relative.strip():
            raise ValueError(f"Edit {index} is missing path")
        if not isinstance(content, str):
            raise ValueError(f"Edit {index} is missing string content")
        path = safe_repo_path(repo_dir, relative)
        if ".textskill" in path.relative_to(repo_dir).parts:
            raise ValueError("External agent cannot edit .textskill files")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        applied.append(str(path.relative_to(repo_dir)))
    return applied


def safe_repo_path(repo_dir: Path, relative: str) -> Path:
    if relative.startswith("/"):
        raise ValueError(f"Edit path must be relative: {relative}")
    path = (repo_dir / relative).resolve()
    try:
        path.relative_to(repo_dir)
    except ValueError as exc:
        raise ValueError(f"Edit path escapes repo: {relative}") from exc
    return path


if __name__ == "__main__":
    raise SystemExit(main())
