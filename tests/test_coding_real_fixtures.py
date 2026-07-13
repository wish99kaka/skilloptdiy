import os
import subprocess
import unittest
from pathlib import Path
from unittest.mock import patch

from textskill_optimizer.io import load_tasks_jsonl, load_text
from textskill_optimizer.optimizer import SkillOptimizer
from textskill_optimizer.plugins.coding import CodingRunner, CodingScorer, CodingSkillEditor


ROOT = Path(__file__).resolve().parent.parent
REAL_ROOT = ROOT / "examples/coding-real"
HARD_ROOT = ROOT / "examples/coding-real-hard"


class CodingRealFixtureTests(unittest.TestCase):
    def test_real_fixtures_have_no_answer_markers(self) -> None:
        markers = []
        for fixture_root in [REAL_ROOT / "fixtures", HARD_ROOT / "fixtures"]:
            for path in fixture_root.rglob("*.py"):
                text = path.read_text(encoding="utf-8")
                if "TEXTSKILL_FIX" in text:
                    markers.append(str(path.relative_to(ROOT)))

        self.assertEqual(markers, [])

    def test_hard_fixtures_have_no_answer_markers(self) -> None:
        markers = []
        for path in (HARD_ROOT / "fixtures").rglob("*.py"):
            text = path.read_text(encoding="utf-8")
            if "TEXTSKILL_FIX" in text:
                markers.append(str(path.relative_to(ROOT)))

        self.assertEqual(markers, [])

    def test_all_real_fixture_tests_fail_before_agent_runs(self) -> None:
        failures = []
        fixtures = sorted((REAL_ROOT / "fixtures").iterdir())
        for fixture in fixtures:
            completed = subprocess.run(
                ["python3", "-m", "unittest", "discover", "-s", "tests"],
                cwd=fixture,
                text=True,
                capture_output=True,
                check=False,
            )
            if completed.returncode == 0:
                failures.append(fixture.name)

        self.assertEqual(failures, [])

    def test_all_hard_fixture_tests_fail_before_agent_runs(self) -> None:
        failures = []
        fixtures = sorted((HARD_ROOT / "fixtures").iterdir())
        for fixture in fixtures:
            completed = subprocess.run(
                ["python3", "-m", "unittest", "discover", "-s", "tests"],
                cwd=fixture,
                text=True,
                capture_output=True,
                check=False,
            )
            if completed.returncode == 0:
                failures.append(fixture.name)

        self.assertEqual(failures, [])

    def test_codex_wrapper_dry_run_does_not_pass_real_tasks(self) -> None:
        skill = load_text(REAL_ROOT / "skill.md")
        tasks = load_tasks_jsonl(REAL_ROOT / "train.jsonl")
        optimizer = SkillOptimizer(
            runner=CodingRunner(),
            scorer=CodingScorer(),
            editor=CodingSkillEditor(),
        )

        with patch.dict(os.environ, {"CODEX_AGENT_DRY_RUN": "1"}):
            report = optimizer.evaluate(skill, tasks, name="real-dry-run")

        self.assertEqual(report.average_score, 0.0)
        self.assertTrue(
            all("argv" in result.output.metadata["agent"]["stdout"] for result in report.results)
        )

    def test_codex_wrapper_dry_run_does_not_pass_hard_tasks(self) -> None:
        skill = load_text(HARD_ROOT / "skill.md")
        tasks = load_tasks_jsonl(HARD_ROOT / "train.jsonl")
        optimizer = SkillOptimizer(
            runner=CodingRunner(),
            scorer=CodingScorer(),
            editor=CodingSkillEditor(),
        )

        with patch.dict(os.environ, {"CODEX_AGENT_DRY_RUN": "1"}):
            report = optimizer.evaluate(skill, tasks, name="hard-dry-run")

        self.assertEqual(report.average_score, 0.0)


if __name__ == "__main__":
    unittest.main()
