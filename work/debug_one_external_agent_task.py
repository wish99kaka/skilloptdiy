"""Debug one external-agent coding task."""

from __future__ import annotations

import getpass
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from textskill_optimizer.io import load_tasks_jsonl, load_text
from textskill_optimizer.plugins.coding import CodingRunner, CodingScorer


def main() -> int:
    task_file = os.environ.get(
        "EXTERNAL_AGENT_TASKS",
        "examples/coding-external-agent/holdout.jsonl",
    )
    skill_file = os.environ.get(
        "EXTERNAL_AGENT_SKILL",
        "examples/coding-external-agent/skill.md",
    )
    task_index = int(os.environ.get("EXTERNAL_AGENT_TASK_INDEX", "0"))
    if sys.stdin.isatty():
        api_key = getpass.getpass("External agent API key: ").strip()
    else:
        api_key = sys.stdin.readline().strip()
    if not api_key:
        print("Missing API key", file=sys.stderr)
        return 2
    os.environ.update(
        {
            "EXTERNAL_AGENT_BASE_URL": "https://ark-cn-beijing.bytedance.net/api/v3",
            "EXTERNAL_AGENT_MODEL": "ep-20260507113406-9h6cz",
            "EXTERNAL_AGENT_API_KEY": api_key,
            "EXTERNAL_AGENT_JSON_MODE": "0",
            "EXTERNAL_AGENT_TIMEOUT": "180",
        }
    )
    task = load_tasks_jsonl(task_file)[task_index]
    skill = load_text(skill_file)
    output = CodingRunner().run(skill, task)
    score = CodingScorer().score(task, output)
    payload = {
        "task_id": task.id,
        "score": score.to_dict(),
        "trace": output.trace,
        "agent": output.metadata.get("agent"),
        "post_test": output.metadata.get("post_test"),
        "diff": output.metadata.get("diff"),
    }
    json.dump(payload, sys.stdout, indent=2)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
