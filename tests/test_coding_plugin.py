import unittest
import subprocess
from pathlib import Path
import tempfile
from unittest.mock import patch

from textskill_optimizer.io import load_tasks_jsonl, load_text
from textskill_optimizer.models import Task
from textskill_optimizer.optimizer import OptimizerConfig, SkillOptimizer
from textskill_optimizer.cli import main as cli_main
from textskill_optimizer.plugins.coding import (
    CodingRunner,
    CodingScorer,
    CodingSkillEditor,
    SKILL_MARKER,
    build_agent_task_payload,
)


ROOT = Path(__file__).resolve().parent.parent


class CodingPluginTests(unittest.TestCase):
    def test_coding_plugin_optimizes_skill_against_tests(self) -> None:
        skill = load_text(ROOT / "examples/coding/skill.md")
        train_tasks = load_tasks_jsonl(ROOT / "examples/coding/train.jsonl")
        valid_tasks = load_tasks_jsonl(ROOT / "examples/coding/valid.jsonl")
        optimizer = SkillOptimizer(
            runner=CodingRunner(),
            scorer=CodingScorer(),
            editor=CodingSkillEditor(),
            config=OptimizerConfig(epochs=1),
        )

        baseline = optimizer.evaluate(skill, valid_tasks, name="baseline")
        result = optimizer.optimize(skill, train_tasks, valid_tasks)

        self.assertEqual(baseline.average_score, 0.0)
        self.assertEqual(result.best_validation_score, 1.0)
        self.assertIn(SKILL_MARKER, result.best_skill_text.casefold())
        self.assertTrue(any(item.accepted for item in result.history[1:]))

    def test_cli_writes_holdout_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            exit_code = cli_main(
                [
                    "optimize",
                    "--plugin",
                    "coding",
                    "--skill",
                    str(ROOT / "examples/coding/skill.md"),
                    "--train",
                    str(ROOT / "examples/coding/train.jsonl"),
                    "--valid",
                    str(ROOT / "examples/coding/valid.jsonl"),
                    "--holdout",
                    str(ROOT / "examples/coding/holdout.jsonl"),
                    "--epochs",
                    "1",
                    "--out",
                    tmp,
                ]
            )

            holdout_path = Path(tmp) / "holdout_final.json"
            holdout_exists = holdout_path.exists()

        self.assertEqual(exit_code, 0)
        self.assertTrue(holdout_exists)

    def test_agent_task_payload_replaces_hidden_test_command(self) -> None:
        task = Task(
            id="hidden",
            input="Fix it",
            expected={"tests_passed": True},
            metadata={
                "repo": "fixtures/hidden",
                "test_command": "python3 hidden.py {repo}",
                "agent_test_command": "python3 -m unittest discover -s tests",
                "hidden_test_command": "python3 hidden.py",
                "score_test_command": "python3 scorer.py",
            },
        )

        payload = build_agent_task_payload(task, "python3 -m unittest discover -s tests")

        self.assertEqual(
            payload["metadata"]["test_command"],
            "python3 -m unittest discover -s tests",
        )
        self.assertNotIn("agent_test_command", payload["metadata"])
        self.assertNotIn("hidden_test_command", payload["metadata"])
        self.assertNotIn("score_test_command", payload["metadata"])

    def test_hidden_fixture_dry_run_scores_with_hidden_tests(self) -> None:
        skill = load_text(ROOT / "examples/coding-hidden/skill.md")
        tasks = load_tasks_jsonl(ROOT / "examples/coding-hidden/valid.jsonl")
        optimizer = SkillOptimizer(
            runner=CodingRunner(),
            scorer=CodingScorer(),
            editor=CodingSkillEditor(),
        )

        with patch.dict(
            "os.environ",
            {
                "EXTERNAL_AGENT_DRY_RUN": "1",
                "EXTERNAL_AGENT_BASE_URL": "https://example.invalid/api/v3",
                "EXTERNAL_AGENT_MODEL": "dry-run-model",
            },
        ):
            report = optimizer.evaluate(skill, tasks, name="hidden-dry-run")

        self.assertEqual(report.average_score, 0.0)
        output = report.results[0].output
        self.assertIn("run_hidden_tests.py", output.metadata["test_command"])
        self.assertEqual(
            output.metadata["agent_test_command"],
            "python3 -m unittest discover -s tests",
        )
        self.assertIn(
            "python3 -m unittest discover -s tests",
            output.metadata["agent"]["stdout"],
        )

    def test_hidden_fixtures_start_with_failing_public_tests(self) -> None:
        hidden_root = ROOT / "examples/coding-hidden"
        unexpectedly_passing = []
        for fixture in sorted((hidden_root / "fixtures").iterdir()):
            completed = subprocess.run(
                ["python3", "-m", "unittest", "discover", "-s", "tests"],
                cwd=fixture,
                text=True,
                capture_output=True,
                check=False,
            )
            if completed.returncode == 0:
                unexpectedly_passing.append(fixture.name)

        self.assertEqual(unexpectedly_passing, [])

    def test_hidden_train_covers_collection_and_range_edges(self) -> None:
        tasks = load_tasks_jsonl(ROOT / "examples/coding-hidden/train.jsonl")
        task_ids = {task.id for task in tasks}

        self.assertIn("coding-hidden-train-dedupe-by-email", task_ids)
        self.assertIn("coding-hidden-train-number-range", task_ids)
        self.assertIn("coding-hidden-train-nested-default", task_ids)
        self.assertIn("coding-hidden-train-stable-sort", task_ids)
        self.assertIn("coding-hidden-train-parse-duration", task_ids)
        self.assertIn("coding-hidden-train-round-cents", task_ids)
        self.assertEqual(len(tasks), 8)

    def test_hidden_valid_and_holdout_have_minimum_breadth(self) -> None:
        valid_tasks = load_tasks_jsonl(ROOT / "examples/coding-hidden/valid.jsonl")
        holdout_tasks = load_tasks_jsonl(ROOT / "examples/coding-hidden/holdout.jsonl")
        valid_ids = {task.id for task in valid_tasks}
        holdout_ids = {task.id for task in holdout_tasks}

        self.assertEqual(len(valid_tasks), 4)
        self.assertEqual(len(holdout_tasks), 4)
        self.assertIn("coding-hidden-valid-nested-pluck", valid_ids)
        self.assertIn("coding-hidden-valid-stable-sort-events", valid_ids)
        self.assertIn("coding-hidden-valid-parse-int-list", valid_ids)
        self.assertIn("coding-hidden-holdout-safe-nested-get", holdout_ids)
        self.assertIn("coding-hidden-holdout-round-tax", holdout_ids)
        self.assertIn("coding-hidden-holdout-dedupe-casefold", holdout_ids)


if __name__ == "__main__":
    unittest.main()
