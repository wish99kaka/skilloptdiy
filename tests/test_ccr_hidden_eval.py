import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parent.parent
MODULE_PATH = ROOT / "work/run_ccr_hidden_eval.py"


def load_eval_module():
    spec = importlib.util.spec_from_file_location("run_ccr_hidden_eval", MODULE_PATH)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class CcrHiddenEvalTests(unittest.TestCase):
    def test_rewrites_agent_command_and_preserves_task_dir(self) -> None:
        module = load_eval_module()
        with tempfile.TemporaryDirectory() as tmp:
            task_file = Path(tmp) / "tasks.jsonl"
            task_file.write_text(
                json.dumps(
                    {
                        "id": "t1",
                        "input": "Fix",
                        "metadata": {
                            "repo": "fixtures/x",
                            "test_command": "python3 {task_dir}/run_hidden_tests.py x {repo}",
                            "agent_command": "old-agent",
                            "timeout_seconds": 180,
                        },
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            with patch.dict("os.environ", {"CCR_TASK_TIMEOUT": "777"}):
                rewritten = module.build_ccr_tasks(task_file, Path("/tmp/ccr_agent_wrapper.py"))

            payload = json.loads(rewritten.read_text(encoding="utf-8"))

        metadata = payload["metadata"]
        self.assertIn("ccr_agent_wrapper.py", metadata["agent_command"])
        self.assertEqual(Path(metadata["_task_dir"]).resolve(), Path(tmp).resolve())
        self.assertEqual(metadata["timeout_seconds"], 777)

    def test_can_limit_task_count(self) -> None:
        module = load_eval_module()
        with tempfile.TemporaryDirectory() as tmp:
            task_file = Path(tmp) / "tasks.jsonl"
            task_file.write_text(
                "\n".join(
                    [
                        json.dumps({"id": "t1", "input": "Fix", "metadata": {"repo": "fixtures/x", "test_command": "true"}}),
                        json.dumps({"id": "t2", "input": "Fix", "metadata": {"repo": "fixtures/y", "test_command": "true"}}),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            with patch.dict("os.environ", {"CCR_TASK_LIMIT": "1"}):
                rewritten = module.build_ccr_tasks(task_file, Path("/tmp/ccr_agent_wrapper.py"))

            payloads = [json.loads(line) for line in rewritten.read_text(encoding="utf-8").splitlines()]

        self.assertEqual([payload["id"] for payload in payloads], ["t1"])


if __name__ == "__main__":
    unittest.main()
