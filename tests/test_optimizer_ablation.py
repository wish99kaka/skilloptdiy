import importlib.util
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
MODULE_PATH = ROOT / "work/run_optimizer_ablation.py"


def load_ablation_module():
    spec = importlib.util.spec_from_file_location("run_optimizer_ablation", MODULE_PATH)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class OptimizerAblationTests(unittest.TestCase):
    def test_controlled_ablation_rows_show_mechanism_contribution(self) -> None:
        module = load_ablation_module()
        with tempfile.TemporaryDirectory() as tmp:
            summary = module.run_ablation(Path(tmp))

        rows = {row["name"]: row for row in summary["rows"]}

        self.assertEqual(rows["gate_only"]["final_validation_score"], 0.0)
        self.assertEqual(rows["gate_only"]["validated_rejections"], 2)

        self.assertEqual(rows["gate_lr"]["final_validation_score"], 0.0)
        self.assertEqual(rows["gate_lr"]["lr_rejections"], 2)
        self.assertEqual(rows["gate_lr"]["validated_rejections"], 0)

        self.assertEqual(rows["gate_lr_rejected"]["final_validation_score"], 1.0)
        self.assertEqual(rows["gate_lr_rejected"]["first_success_epoch"], 2)

        self.assertEqual(rows["gate_lr_rejected_meta"]["final_validation_score"], 1.0)
        self.assertEqual(rows["gate_lr_rejected_meta"]["first_success_epoch"], 1)
        self.assertLess(
            rows["gate_lr_rejected_meta"]["rejected_total"],
            rows["gate_lr_rejected"]["rejected_total"],
        )


if __name__ == "__main__":
    unittest.main()
