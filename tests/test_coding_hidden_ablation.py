import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
AGENT_PATH = ROOT / "work/coding_hidden_deterministic_agent.py"
TEXT_AGENT_PATH = ROOT / "work/coding_hidden_text_sensitive_agent.py"
ABLATION_PATH = ROOT / "work/run_coding_hidden_ablation.py"


def load_module(path: Path, name: str):
    if str(path.parent) not in sys.path:
        sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class CodingHiddenAblationTests(unittest.TestCase):
    def test_agent_identifies_fixture_from_task_metadata(self) -> None:
        module = load_module(AGENT_PATH, "coding_hidden_deterministic_agent")
        with tempfile.TemporaryDirectory() as tmp:
            task = Path(tmp) / "task.json"
            task.write_text(
                json.dumps({"id": "t1", "metadata": {"repo": "fixtures/unique-by-id"}}),
                encoding="utf-8",
            )

            self.assertEqual(module.fixture_name_from_task(task), "unique-by-id")

    def test_tasks_are_rewritten_to_use_deterministic_agent(self) -> None:
        module = load_module(ABLATION_PATH, "run_coding_hidden_ablation")

        tasks = module.load_coding_hidden_tasks(
            ROOT / "examples/coding-hidden/valid.jsonl",
            timeout_seconds=7,
        )

        self.assertEqual(tasks[0].metadata["timeout_seconds"], 7)
        self.assertIn("coding_hidden_deterministic_agent.py", tasks[0].metadata["agent_command"])

    def test_tasks_can_use_text_sensitive_agent(self) -> None:
        module = load_module(ABLATION_PATH, "run_coding_hidden_ablation")

        tasks = module.load_coding_hidden_tasks(
            ROOT / "examples/coding-hidden/valid.jsonl",
            timeout_seconds=7,
            agent_path=TEXT_AGENT_PATH,
        )

        self.assertIn("coding_hidden_text_sensitive_agent.py", tasks[0].metadata["agent_command"])

    def test_text_sensitive_agent_infers_capabilities_from_natural_language(self) -> None:
        module = load_module(TEXT_AGENT_PATH, "coding_hidden_text_sensitive_agent")

        capabilities = module.infer_capabilities(
            "For keyed dedupe keep records missing the key as independent entries. "
            "For inclusive ranges normalize lower/upper bounds. "
            "Handle malformed inputs and separators."
        )

        self.assertIn("keyed_dedupe", capabilities)
        self.assertIn("range_bounds", capabilities)
        self.assertIn("token_parser", capabilities)

    def test_text_sensitive_agent_treats_missing_keys_last_as_sort_capability(self) -> None:
        module = load_module(TEXT_AGENT_PATH, "coding_hidden_text_sensitive_agent")

        capabilities = module.infer_capabilities(
            "Sort-by-key: keep stable and place missing keys last."
        )

        self.assertIn("sort_missing_last", capabilities)

    def test_ablation_configs_can_use_real_editor_lr_profile(self) -> None:
        module = load_module(ABLATION_PATH, "run_coding_hidden_ablation")

        with tempfile.TemporaryDirectory() as tmp:
            configs = {
                case["name"]: case
                for case in module.ablation_configs(Path(tmp), lr_profile="real-editor")
            }

        self.assertIsNone(configs["gate_only"]["config"].max_skill_delta_chars)
        self.assertEqual(configs["gate_lr"]["config"].max_skill_delta_chars, 520)
        self.assertEqual(configs["gate_lr_rejected"]["config"].max_skill_delta_chars, 520)
        self.assertEqual(configs["gate_lr_rejected_meta"]["config"].max_skill_delta_chars, 520)

    def test_coding_hidden_meta_skill_names_missing_capability_families(self) -> None:
        module = load_module(ABLATION_PATH, "run_coding_hidden_ablation")

        meta = module.CODING_HIDDEN_META_SKILL

        self.assertIn("nested path access", meta)
        self.assertIn("sort-by-key", meta)
        self.assertIn("missing keys last", meta)
        self.assertIn("rounding", meta)
        self.assertIn("Decimal", meta)

    def test_small_real_subset_shows_rejected_buffer_contribution(self) -> None:
        module = load_module(ABLATION_PATH, "run_coding_hidden_ablation")
        train = module.load_coding_hidden_tasks(
            ROOT / "examples/coding-hidden/train.jsonl",
            timeout_seconds=10,
        )[:2]
        valid = module.load_coding_hidden_tasks(
            ROOT / "examples/coding-hidden/valid.jsonl",
            timeout_seconds=10,
        )[:1]
        holdout = module.load_coding_hidden_tasks(
            ROOT / "examples/coding-hidden/holdout.jsonl",
            timeout_seconds=10,
        )[:1]

        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            configs = {case["name"]: case for case in module.ablation_configs(out_dir)}
            gate_lr = module.run_case(
                configs["gate_lr"],
                out_dir=out_dir,
                train_tasks=train,
                validation_tasks=valid,
                holdout_tasks=holdout,
            )
            rejected = module.run_case(
                configs["gate_lr_rejected"],
                out_dir=out_dir,
                train_tasks=train,
                validation_tasks=valid,
                holdout_tasks=holdout,
            )

        self.assertEqual(gate_lr["validation_score"], 0.0)
        self.assertEqual(gate_lr["lr_rejections"], 2)
        self.assertEqual(rejected["validation_score"], 1.0)
        self.assertEqual(rejected["first_success_epoch"], 2)


if __name__ == "__main__":
    unittest.main()
