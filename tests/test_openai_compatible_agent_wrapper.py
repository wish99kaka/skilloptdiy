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
MODULE_PATH = ROOT / "examples/coding/openai_compatible_agent_wrapper.py"


def load_agent_module():
    spec = importlib.util.spec_from_file_location("openai_compatible_agent_wrapper", MODULE_PATH)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class OpenAICompatibleAgentWrapperTests(unittest.TestCase):
    def test_normalizes_base_and_full_endpoint(self) -> None:
        module = load_agent_module()

        self.assertEqual(
            module.normalize_chat_completions_url("https://example.com/v1"),
            "https://example.com/v1/chat/completions",
        )
        self.assertEqual(
            module.normalize_chat_completions_url("https://example.com/v1/chat/completions"),
            "https://example.com/v1/chat/completions",
        )

    def test_extract_json_text_from_fenced_content(self) -> None:
        module = load_agent_module()
        payload = module.extract_json_text('```json\n{"edits": [], "summary": "ok"}\n```')

        self.assertEqual(json.loads(payload)["summary"], "ok")

    def test_system_prompt_makes_skill_text_binding(self) -> None:
        module = load_agent_module()

        self.assertIn("skill_text", module.SYSTEM_PROMPT)
        self.assertIn("inferred full behavior", module.SYSTEM_PROMPT)
        self.assertIn("Do not hard-code only the public examples", module.SYSTEM_PROMPT)

    def test_build_request_sends_skill_as_separate_message(self) -> None:
        module = load_agent_module()
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "repo"
            (repo / "app").mkdir(parents=True)
            (repo / "tests").mkdir()
            (repo / "app/main.py").write_text("def f():\n    return 1\n", encoding="utf-8")
            (repo / "tests/test_main.py").write_text(
                "import unittest\n\nclass T(unittest.TestCase):\n    def test_ok(self):\n        self.assertTrue(True)\n",
                encoding="utf-8",
            )
            payload = module.build_chat_request_payload(
                {
                    "repo_dir": repo,
                    "task": {"id": "task-1", "metadata": {"test_command": "python3 -m unittest discover -s tests"}},
                    "skill_text": "# Skill\nFollow me.",
                    "instruction": "Fix tests.",
                },
                model="model-x",
            )

        self.assertIn("Skill instructions", payload["messages"][1]["content"])
        user_message = payload["messages"][1]["content"]
        self.assertIn("Task payload", user_message)
        self.assertLess(user_message.index("Skill instructions"), user_message.index("Task payload"))
        task_payload = user_message.split("Task payload:", 1)[1]
        self.assertNotIn("skill_text", task_payload)

    def test_apply_edits_writes_files_and_rejects_escape(self) -> None:
        module = load_agent_module()
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            applied = module.apply_edits(
                repo,
                {
                    "edits": [
                        {"path": "app/main.py", "content": "def ok():\n    return True\n"}
                    ]
                },
            )

            self.assertEqual(applied, ["app/main.py"])
            self.assertTrue((repo / "app/main.py").exists())
            with self.assertRaises(ValueError):
                module.apply_edits(
                    repo,
                    {"edits": [{"path": "../escape.py", "content": ""}]},
                )

    def test_dry_run_prints_request(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "repo"
            (repo / "app").mkdir(parents=True)
            (repo / "tests").mkdir()
            (repo / "app/main.py").write_text("def f():\n    return 1\n", encoding="utf-8")
            (repo / "tests/test_main.py").write_text(
                "import unittest\n\nclass T(unittest.TestCase):\n    def test_ok(self):\n        self.assertTrue(True)\n",
                encoding="utf-8",
            )
            skill = Path(tmp) / "skill.md"
            task = Path(tmp) / "task.json"
            skill.write_text("# Skill\n", encoding="utf-8")
            task.write_text(
                json.dumps(
                    {
                        "id": "task-1",
                        "metadata": {"test_command": "python3 -m unittest discover -s tests"},
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
                    "EXTERNAL_AGENT_BASE_URL": "https://example.com/v1",
                    "EXTERNAL_AGENT_MODEL": "model-x",
                    "EXTERNAL_AGENT_DRY_RUN": "1",
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
        self.assertEqual(payload["model"], "model-x")
        self.assertTrue(payload["uses_json_mode"])

    def test_build_request_requires_model_config(self) -> None:
        module = load_agent_module()
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ValueError):
                module.build_request_from_env({"repo_dir": Path("."), "task": {}, "skill_text": "", "instruction": ""})


if __name__ == "__main__":
    unittest.main()
