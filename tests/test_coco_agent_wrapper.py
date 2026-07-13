import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parent.parent
MODULE_PATH = ROOT / "examples/coding/coco_agent_wrapper.py"


def load_wrapper_module():
    spec = importlib.util.spec_from_file_location("coco_agent_wrapper", MODULE_PATH)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class CocoAgentWrapperTests(unittest.TestCase):
    def test_builds_coco_print_argv(self) -> None:
        module = load_wrapper_module()
        with patch.dict(
            os.environ,
            {
                "COCO_AGENT_BIN": "/path/to/coco",
                "COCO_AGENT_QUERY_TIMEOUT": "3m",
                "COCO_AGENT_BASH_TIMEOUT": "2m",
                "COCO_AGENT_EXTRA_ARGS": "--output-format text",
            },
            clear=True,
        ):
            argv = module.build_coco_argv({"repo_dir": Path("/tmp/repo")}, "Fix tests")

        self.assertEqual(argv[0], "/path/to/coco")
        self.assertIn("--print", argv)
        self.assertIn("--yolo", argv)
        self.assertIn("3m", argv)
        self.assertIn("2m", argv)
        self.assertEqual(argv[-1], "Fix tests")

    def test_build_prompt_includes_skill_and_test_command(self) -> None:
        module = load_wrapper_module()
        context = {
            "repo_dir": Path("/tmp/repo"),
            "instruction": "Fix the failing tests.",
            "skill_text": "# Skill\nUse robust edge checks.",
            "task": {
                "id": "task-1",
                "metadata": {"test_command": "python3 -m unittest"},
            },
        }

        prompt = module.build_prompt(context)

        self.assertIn("Fix the failing tests.", prompt)
        self.assertIn("python3 -m unittest", prompt)
        self.assertIn("Use robust edge checks.", prompt)
        self.assertIn("Do not modify files under `.textskill/`.", prompt)
        self.assertIn("binding requirements", prompt)
        self.assertIn("public tests do not cover", prompt)

    def test_build_prompt_omits_skill_binding_for_no_skill_baseline(self) -> None:
        module = load_wrapper_module()
        context = {
            "repo_dir": Path("/tmp/repo"),
            "instruction": "Fix the failing tests.",
            "skill_text": "",
            "task": {"id": "task-1", "metadata": {"test_command": "python3 -m unittest"}},
        }

        prompt = module.build_prompt(context)

        self.assertIn("No additional skill document", prompt)
        self.assertNotIn("binding requirements", prompt)

    def test_dry_run_prints_argv_and_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "repo"
            repo.mkdir()
            skill = Path(tmp) / "skill.md"
            task = Path(tmp) / "task.json"
            skill.write_text("# Skill\n", encoding="utf-8")
            task.write_text(
                json.dumps(
                    {
                        "id": "task-1",
                        "metadata": {"test_command": "python3 -m unittest"},
                    }
                ),
                encoding="utf-8",
            )
            env = os.environ.copy()
            env.update(
                {
                    "TEXTSKILL_REPO_DIR": str(repo),
                    "TEXTSKILL_SKILL_PATH": str(skill),
                    "TEXTSKILL_TASK_PATH": str(task),
                    "TEXTSKILL_INSTRUCTION": "Fix tests.",
                    "COCO_AGENT_DRY_RUN": "1",
                    "COCO_AGENT_BIN": "coco-test",
                }
            )

            completed = subprocess.run(
                [sys.executable, str(MODULE_PATH)],
                env=env,
                text=True,
                capture_output=True,
                check=False,
            )

        self.assertEqual(completed.returncode, 0)
        payload = json.loads(completed.stdout)
        self.assertEqual(payload["argv"][0], "coco-test")
        self.assertIn("Fix tests.", payload["prompt"])

    def test_missing_env_returns_actionable_error(self) -> None:
        completed = subprocess.run(
            [sys.executable, str(MODULE_PATH)],
            env={},
            text=True,
            capture_output=True,
            check=False,
        )

        self.assertEqual(completed.returncode, 2)
        self.assertIn("TEXTSKILL_REPO_DIR is required", completed.stderr)

    def test_timeout_uses_task_metadata_unless_env_overrides(self) -> None:
        module = load_wrapper_module()
        context = {"task": {"metadata": {"timeout_seconds": 45}}}

        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(module.resolve_agent_timeout(context, "COCO_AGENT_TIMEOUT"), 45)
        with patch.dict(os.environ, {"COCO_AGENT_TIMEOUT": "12"}, clear=True):
            self.assertEqual(module.resolve_agent_timeout(context, "COCO_AGENT_TIMEOUT"), 12)


if __name__ == "__main__":
    unittest.main()
