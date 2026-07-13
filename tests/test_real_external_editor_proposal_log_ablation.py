import argparse
import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parent.parent
MODULE_PATH = ROOT / "work/run_real_external_editor_proposal_log_ablation.py"


def load_module():
    spec = importlib.util.spec_from_file_location("run_real_external_editor_proposal_log_ablation", MODULE_PATH)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class RealExternalEditorProposalLogAblationTests(unittest.TestCase):
    def test_parse_csv_requires_values(self) -> None:
        module = load_module()

        self.assertEqual(module.parse_csv("seed-a, seed-b"), ["seed-a", "seed-b"])
        with self.assertRaises(ValueError):
            module.parse_csv(" , ")

    def test_build_external_env_dry_run_uses_non_secret_placeholder(self) -> None:
        module = load_module()
        args = argparse.Namespace(
            base_url="https://example.test/v1",
            model="model-a",
            dry_run=True,
            json_mode="0",
            llm_timeout=123,
            temperature="0.1",
        )

        with patch.dict(os.environ, {}, clear=True):
            env = module.build_external_env(args)

        self.assertEqual(env["EXTERNAL_LLM_API_KEY"], "not-needed")
        self.assertEqual(env["EXTERNAL_LLM_BASE_URL"], "https://example.test/v1")
        self.assertEqual(env["EXTERNAL_LLM_MODEL"], "model-a")

    def test_dry_run_writes_summary_and_manifest(self) -> None:
        module = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "capture"
            summary = module.capture_and_replay(
                out_dir=out_dir,
                replay_out_dir=Path(tmp) / "replay",
                proposal_log_path=out_dir / "proposals.jsonl",
                seeds=["seed-a"],
                cases=["gate_lr"],
                timeout_seconds=5,
                editor_command="python editor.py",
                editor_timeout=10,
                env={
                    "EXTERNAL_LLM_BASE_URL": "https://example.test/v1",
                    "EXTERNAL_LLM_MODEL": "model-a",
                    "EXTERNAL_LLM_API_KEY": "not-needed",
                    "EXTERNAL_LLM_JSON_MODE": "0",
                    "EXTERNAL_LLM_TEMPERATURE": "0.2",
                },
                agent_path=Path("work/coding_hidden_text_sensitive_agent.py"),
                dry_run=True,
            )

            saved = json.loads((out_dir / "dry_run_summary.json").read_text(encoding="utf-8"))
            manifest = (out_dir / "dry_run_manifest.md").read_text(encoding="utf-8")

        self.assertEqual(summary["planned_runs"], [{"seed": "seed-a", "case": "gate_lr"}])
        self.assertEqual(summary["agent_path"], "work/coding_hidden_text_sensitive_agent.py")
        self.assertEqual(saved["external_model"]["model"], "model-a")
        self.assertTrue(saved["external_model"]["has_api_key"])
        self.assertEqual(saved["agent_path"], "work/coding_hidden_text_sensitive_agent.py")
        self.assertIn("| seed-a | gate_lr |", manifest)
        self.assertIn("Agent path: `work/coding_hidden_text_sensitive_agent.py`", manifest)


if __name__ == "__main__":
    unittest.main()
