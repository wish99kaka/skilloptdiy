import unittest
import tempfile
from pathlib import Path

from work.run_coding_hidden_v2_locked_eval import (
    build_locked_result,
    locked_tasks_path,
    validate_locked_task_count,
)


class CodingHiddenV2LockedEvalTests(unittest.TestCase):
    def test_requires_locked_wrapper_task_path(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "CROSS_AGENT_TASKS"):
            locked_tasks_path({})

        self.assertEqual(
            locked_tasks_path({"CROSS_AGENT_TASKS": "/tmp/locked/test.jsonl"}),
            Path("/tmp/locked/test.jsonl"),
        )

    def test_locked_result_records_skill_identity_and_all_score_levels(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            skill = Path(tmp) / "skill.md"
            skill.write_text("Always validate inputs.\n", encoding="utf-8")
            report = {
                "name": "locked:test",
                "average_score": 0.5,
                "pass_rate": 0.5,
                "results": [
                    result("task-a", "family-a", ["contract-a"], True),
                    result("task-b", "family-b", ["contract-b"], False),
                ],
            }

            payload = build_locked_result(report, skill, duration_seconds=12.5)

        self.assertEqual(payload["task_count"], 2)
        self.assertEqual(payload["task_accuracy"], 0.5)
        self.assertEqual(payload["family_macro_accuracy"], 0.5)
        self.assertEqual(payload["contract_macro_accuracy"], 0.5)
        self.assertEqual(payload["skill_bytes"], 24)
        self.assertEqual(len(payload["skill_sha256"]), 64)
        self.assertNotIn("evaluation_report", payload)
        self.assertEqual(
            payload["task_results"][0],
            {
                "task_id": "task-a",
                "family": "family-a",
                "contracts": ["contract-a"],
                "score": 1.0,
                "success": True,
            },
        )

    def test_requires_complete_twenty_task_locked_split(self) -> None:
        validate_locked_task_count([object()] * 20)

        with self.assertRaisesRegex(ValueError, "expected 20"):
            validate_locked_task_count([object()] * 19)


def result(task_id: str, family: str, contracts: list[str], success: bool) -> dict:
    return {
        "task": {
            "id": task_id,
            "input": "must not persist",
            "metadata": {"benchmark_family": family, "contract_tags": contracts},
        },
        "output": {"value": "must not persist"},
        "score": {"success": success, "value": float(success)},
    }


if __name__ == "__main__":
    unittest.main()
