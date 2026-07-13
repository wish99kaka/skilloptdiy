"""A tiny command editor for the offline coding example.

It reads the optimizer payload from stdin and prints a JSON proposal to stdout.
Replace this with an LLM-backed script in real use.
"""

from __future__ import annotations

import json
import sys

from editor_io import load_optimizer_payload_from_stdin


CODING_LOOP = """## Coding Agent Loop

- Use a test-guided root-cause loop: run or inspect the provided test command, read the exact failure, change implementation files instead of tests, then rerun the same command.
- Prefer the smallest implementation change that explains the failing assertion.
- Do not declare success until the provided test command passes.
"""


def main() -> int:
    try:
        payload = load_optimizer_payload_from_stdin()
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    skill_text = payload["skill_text"]
    failures = [
        result
        for result in payload["train_results"]
        if not result.get("score", {}).get("success")
    ]
    if not failures or "test-guided root-cause loop" in skill_text.casefold():
        print(json.dumps({"proposals": []}))
        return 0

    failed_ids = ", ".join(result["task"]["id"] for result in failures)
    proposal = {
        "name": "demo-llm-coding-loop",
        "skill_text": skill_text.rstrip() + "\n\n" + CODING_LOOP,
        "rationale": (
            "Training failures show the agent did not use the test failure as "
            f"the source of truth: {failed_ids}"
        ),
        "metadata": {"failed_task_ids": [result["task"]["id"] for result in failures]},
    }
    print(json.dumps({"proposals": [proposal]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
