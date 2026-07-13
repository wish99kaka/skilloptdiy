import importlib.util
import os
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parent.parent
MODULE_PATH = ROOT / "examples/coding/ccr_agent_wrapper.py"


def load_wrapper_module():
    spec = importlib.util.spec_from_file_location("ccr_agent_wrapper", MODULE_PATH)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class CcrAgentWrapperTests(unittest.TestCase):
    def test_prompt_is_not_passed_as_argv(self) -> None:
        module = load_wrapper_module()
        prompt = "Prompt with ```markdown fences``` and `inline` backticks"

        argv = module.build_ccr_argv(prompt)

        self.assertIn("code", argv)
        self.assertIn("-p", argv)
        self.assertNotIn(prompt, argv)

    def test_subprocess_env_adds_ccr_dependencies_to_path(self) -> None:
        module = load_wrapper_module()

        with patch.dict(os.environ, {"PATH": "/usr/bin"}, clear=False):
            env = module.build_subprocess_env()

        self.assertTrue(env["PATH"].startswith("/Users/bytedance/.local/bin"))

    def test_timeout_uses_task_metadata_unless_env_overrides(self) -> None:
        module = load_wrapper_module()
        context = {"task": {"metadata": {"timeout_seconds": 45}}}

        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(module.resolve_agent_timeout(context, "CCR_AGENT_TIMEOUT"), 45)
        with patch.dict(os.environ, {"CCR_AGENT_TIMEOUT": "12"}, clear=True):
            self.assertEqual(module.resolve_agent_timeout(context, "CCR_AGENT_TIMEOUT"), 12)


if __name__ == "__main__":
    unittest.main()
