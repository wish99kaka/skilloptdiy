import importlib.util
import os
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parent.parent
MODULE_PATH = ROOT / "examples/coding/kilo_agent_wrapper.py"


def load_wrapper_module():
    spec = importlib.util.spec_from_file_location("kilo_agent_wrapper", MODULE_PATH)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class KiloAgentWrapperTests(unittest.TestCase):
    def test_builds_non_interactive_run_command(self) -> None:
        module = load_wrapper_module()
        context = {"repo_dir": Path("/tmp/repo")}
        prompt = "Fix the task"

        argv = module.build_kilo_argv(context, prompt)

        self.assertEqual(argv[1], "run")
        self.assertIn("--dir", argv)
        self.assertIn("/tmp/repo", argv)
        self.assertIn("--auto", argv)
        self.assertEqual(argv[-1], prompt)

    def test_subprocess_env_adds_kilo_dependencies_to_path(self) -> None:
        module = load_wrapper_module()

        with patch.dict(os.environ, {"PATH": "/usr/bin"}, clear=False):
            env = module.build_subprocess_env()

        self.assertTrue(env["PATH"].startswith("/Users/bytedance/.local/bin"))

    def test_timeout_uses_task_metadata_unless_env_overrides(self) -> None:
        module = load_wrapper_module()
        context = {"task": {"metadata": {"timeout_seconds": 45}}}

        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(module.resolve_agent_timeout(context, "KILO_AGENT_TIMEOUT"), 45)
        with patch.dict(os.environ, {"KILO_AGENT_TIMEOUT": "12"}, clear=True):
            self.assertEqual(module.resolve_agent_timeout(context, "KILO_AGENT_TIMEOUT"), 12)


if __name__ == "__main__":
    unittest.main()
