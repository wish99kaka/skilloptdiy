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
MODULE_PATH = ROOT / "examples/coding/codex_agent_wrapper.py"


def load_wrapper_module():
    spec = importlib.util.spec_from_file_location("codex_agent_wrapper", MODULE_PATH)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class CodexAgentWrapperTests(unittest.TestCase):
    def test_builds_codex_exec_argv(self) -> None:
        module = load_wrapper_module()
        with tempfile.TemporaryDirectory() as tmp:
            context = {"repo_dir": Path(tmp)}
            env = {
                "CODEX_AGENT_BIN": "/path/to/codex",
                "CODEX_AGENT_MODEL": "gpt-test",
                "CODEX_AGENT_EXTRA_ARGS": "--json --output-last-message result.txt",
            }
            with patch.dict(os.environ, env, clear=True):
                argv = module.build_codex_argv(context)

        self.assertEqual(
            argv[:6],
            ["/path/to/codex", "--ask-for-approval", "never", "exec", "--cd", tmp],
        )
        self.assertIn("--skip-git-repo-check", argv)
        self.assertIn("--ephemeral", argv)
        self.assertIn("--model", argv)
        self.assertIn("gpt-test", argv)
        self.assertEqual(argv[-1], "-")

    def test_build_prompt_includes_skill_and_test_command(self) -> None:
        module = load_wrapper_module()
        context = {
            "repo_dir": Path("/tmp/repo"),
            "instruction": "Fix the failing tests.",
            "skill_text": "# Skill\nUse root cause analysis.",
            "task": {
                "id": "task-1",
                "metadata": {"test_command": "python3 -m unittest"},
            },
        }

        with patch.dict(os.environ, {}, clear=True):
            prompt = module.build_prompt(context)

        self.assertIn("Fix the failing tests.", prompt)
        self.assertIn("python3 -m unittest", prompt)
        self.assertIn("Use root cause analysis.", prompt)
        self.assertIn("Do not modify files under `.textskill/`.", prompt)
        self.assertIn("Follow the skill document as the primary process guidance.", prompt)
        self.assertNotIn("Prefer the smallest implementation change", prompt)

    def test_guided_prompt_mode_adds_method_guidance(self) -> None:
        module = load_wrapper_module()
        context = {
            "repo_dir": Path("/tmp/repo"),
            "instruction": "Fix the failing tests.",
            "skill_text": "# Skill",
            "task": {
                "id": "task-1",
                "metadata": {"test_command": "python3 -m unittest"},
            },
        }

        with patch.dict(os.environ, {"CODEX_AGENT_PROMPT_MODE": "guided"}, clear=True):
            prompt = module.build_prompt(context)

        self.assertIn("Use the test command above as the source of truth.", prompt)
        self.assertIn("Prefer the smallest implementation change", prompt)

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
                    "CODEX_AGENT_DRY_RUN": "1",
                    "CODEX_AGENT_BIN": "codex-test",
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
        self.assertEqual(payload["argv"][0], "codex-test")
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


if __name__ == "__main__":
    unittest.main()
