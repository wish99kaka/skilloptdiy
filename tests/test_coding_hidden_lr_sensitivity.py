import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
MODULE_PATH = ROOT / "work/run_coding_hidden_lr_sensitivity.py"


def load_module():
    spec = importlib.util.spec_from_file_location("run_coding_hidden_lr_sensitivity", MODULE_PATH)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class CodingHiddenLRSensitivityTests(unittest.TestCase):
    def test_parse_seed_list_requires_values(self) -> None:
        module = load_module()

        self.assertEqual(module.parse_seed_list("seed-a, seed-b"), ["seed-a", "seed-b"])
        with self.assertRaises(ValueError):
            module.parse_seed_list(" , ")

    def test_source_case_label(self) -> None:
        module = load_module()

        self.assertEqual(module.source_case_label("gate_lr"), "+LR")
        self.assertEqual(
            module.source_case_label("gate_lr_rejected_meta"),
            "+LR+Rejected Buffer+Meta Skill",
        )

    def test_aggregate_rows_groups_by_source_case_and_budget(self) -> None:
        module = load_module()
        rows = [
            {
                "source_case": "gate_lr",
                "source_case_label": "+LR",
                "budget": "strict",
                "budget_label": "strict",
                "validation_score": 0.0,
                "holdout_score": 0.0,
                "lr_rejections": 2,
            },
            {
                "source_case": "gate_lr",
                "source_case_label": "+LR",
                "budget": "strict",
                "budget_label": "strict",
                "validation_score": 1.0,
                "holdout_score": 0.5,
                "lr_rejections": 0,
            },
        ]

        aggregates = module.aggregate_rows(rows)

        self.assertEqual(len(aggregates), 1)
        self.assertEqual(aggregates[0]["mean_validation_score"], 0.5)
        self.assertEqual(aggregates[0]["mean_holdout_score"], 0.25)
        self.assertEqual(aggregates[0]["mean_lr_rejections"], 1)
        self.assertEqual(aggregates[0]["successes"], 1)


if __name__ == "__main__":
    unittest.main()
