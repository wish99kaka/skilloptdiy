"""Print one real coding task runner result for debugging."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from textskill_optimizer.io import load_tasks_jsonl, load_text
from textskill_optimizer.plugins.coding import CodingRunner, CodingScorer


def main() -> int:
    task = load_tasks_jsonl("examples/coding-real/valid.jsonl")[0]
    skill = load_text("examples/coding-real/skill.md")
    runner = CodingRunner()
    scorer = CodingScorer()
    output = runner.run(skill, task)
    score = scorer.score(task, output)
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
