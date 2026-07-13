"""OpenAI-backed command editor for TextSkill Optimizer.

Usage:
  export OPENAI_API_KEY=...
  export OPENAI_MODEL=gpt-5.2
  python3 examples/coding/openai_skill_editor.py < optimizer-payload.json

The optimizer normally invokes this via:
  --editor-command "python3 examples/coding/openai_skill_editor.py"
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request
from typing import Any

from editor_io import load_optimizer_payload_from_stdin


DEFAULT_MODEL = "gpt-5.2"
OPENAI_RESPONSES_URL = "https://api.openai.com/v1/responses"


DEVELOPER_PROMPT = """You are editing a reusable agent skill document.

First principles:
- The skill document is the only thing you may change.
- The runner, scorer, tests, benchmarks, expected answers, and task metadata are fixed.
- A proposal is useful only if it can improve validation tasks, not just the observed training tasks.

Editing rules:
- Return the full replacement skill document in skill_text.
- Preserve useful existing instructions.
- Add small, general, testable behavior rules derived from failure evidence.
- Do not mention specific task IDs, fixture names, exact hidden answers, or benchmark internals in the skill.
- Do not instruct the agent to edit tests, weaken tests, skip tests, change scorers, or fake outputs.
- Prefer one strong proposal over many weak proposals.
- If there is no clear generalizable improvement, return an empty proposals list.
"""


PROPOSAL_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["proposals"],
    "properties": {
        "proposals": {
            "type": "array",
            "maxItems": 2,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["name", "skill_text", "rationale"],
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Short kebab-case candidate name.",
                    },
                    "skill_text": {
                        "type": "string",
                        "description": "Full replacement skill document.",
                    },
                    "rationale": {
                        "type": "string",
                        "description": "Why this edit should improve future tasks.",
                    },
                },
            },
        }
    },
}


def main() -> int:
    try:
        optimizer_payload = load_optimizer_payload_from_stdin()
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("OPENAI_API_KEY is required", file=sys.stderr)
        return 2

    request_payload = build_request_payload(
        optimizer_payload,
        model=os.environ.get("OPENAI_MODEL", DEFAULT_MODEL),
    )
    try:
        response_payload = call_openai_responses(api_key, request_payload)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        print(f"OpenAI API error {exc.code}: {body}", file=sys.stderr)
        return 1

    output_text = extract_output_text(response_payload)
    if not output_text:
        print("OpenAI response did not contain output text", file=sys.stderr)
        return 1

    try:
        proposals = json.loads(output_text)
    except json.JSONDecodeError as exc:
        print(f"OpenAI output was not JSON: {exc}", file=sys.stderr)
        print(output_text, file=sys.stderr)
        return 1

    print(json.dumps(proposals))
    return 0


def build_request_payload(optimizer_payload: dict[str, Any], *, model: str) -> dict[str, Any]:
    user_payload = {
        "epoch": optimizer_payload.get("epoch"),
        "current_skill_text": optimizer_payload.get("skill_text"),
        "failed_training_results": [
            result
            for result in optimizer_payload.get("train_results", [])
            if not result.get("score", {}).get("success")
        ],
        "successful_training_results": [
            {
                "task": result.get("task", {}),
                "score": result.get("score", {}),
                "trace": result.get("output", {}).get("trace", []),
            }
            for result in optimizer_payload.get("train_results", [])
            if result.get("score", {}).get("success")
        ],
    }
    return {
        "model": model,
        "input": [
            {
                "role": "developer",
                "content": [{"type": "input_text", "text": DEVELOPER_PROMPT}],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": json.dumps(user_payload, indent=2),
                    }
                ],
            },
        ],
        "text": {
            "format": {
                "type": "json_schema",
                "name": "skill_edit_proposals",
                "strict": True,
                "schema": PROPOSAL_SCHEMA,
            }
        },
    }


def call_openai_responses(
    api_key: str,
    request_payload: dict[str, Any],
) -> dict[str, Any]:
    request = urllib.request.Request(
        OPENAI_RESPONSES_URL,
        data=json.dumps(request_payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    timeout = float(os.environ.get("OPENAI_TIMEOUT_SECONDS", "120"))
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def extract_output_text(response_payload: dict[str, Any]) -> str:
    direct = response_payload.get("output_text")
    if isinstance(direct, str):
        return direct

    texts: list[str] = []
    for item in response_payload.get("output", []):
        if not isinstance(item, dict):
            continue
        for content in item.get("content", []):
            if not isinstance(content, dict):
                continue
            if isinstance(content.get("text"), str):
                texts.append(content["text"])
    return "\n".join(texts)


if __name__ == "__main__":
    raise SystemExit(main())
