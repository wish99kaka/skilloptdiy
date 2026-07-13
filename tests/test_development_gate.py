import unittest

from work.development_gate import build_development_gate


class DevelopmentGateTests(unittest.TestCase):
    def test_contract_macro_regression_blocks_task_accuracy_gain(self) -> None:
        rows = [
            row("seed-a", "human_skill", 0.8),
            row("seed-b", "human_skill", 0.8),
            row("seed-c", "human_skill", 0.8),
            row("seed-a", "executive", 0.9),
            row("seed-b", "executive", 0.9),
            row("seed-c", "executive", 0.9),
        ]
        aggregate = {
            "human_skill": condition(task_mean=0.8, contract_macro=0.8),
            "executive": condition(task_mean=0.9, contract_macro=0.7),
        }

        gate = build_development_gate(rows, aggregate, {"best_baseline_margin": 0.05, "min_seed_wins": 2})

        self.assertFalse(gate["passed"])
        self.assertAlmostEqual(gate["contract_macro_delta"], -0.1)
        self.assertIn("contract macro delta -0.1000", gate["blocked_reason"])

    def test_critical_contract_regression_blocks_scale_up(self) -> None:
        rows = [
            row("seed-a", "human_skill", 0.8),
            row("seed-b", "human_skill", 0.8),
            row("seed-c", "human_skill", 0.8),
            row("seed-a", "executive", 0.9),
            row("seed-b", "executive", 0.9),
            row("seed-c", "executive", 0.9),
        ]
        aggregate = {
            "human_skill": condition(
                task_mean=0.8,
                contract_macro=0.8,
                contracts={
                    "largest_remainder": {"accuracy": 1.0, "passed": 3, "total": 3},
                    "input_validation": {"accuracy": 0.5, "passed": 3, "total": 6},
                },
            ),
            "executive": condition(
                task_mean=0.9,
                contract_macro=0.8,
                contracts={
                    "largest_remainder": {"accuracy": 0.0, "passed": 0, "total": 3},
                    "input_validation": {"accuracy": 1.0, "passed": 6, "total": 6},
                },
            ),
        }

        gate = build_development_gate(
            rows,
            aggregate,
            {
                "best_baseline_margin": 0.05,
                "min_seed_wins": 2,
                "critical_contracts": ["largest_remainder"],
            },
        )

        self.assertFalse(gate["passed"])
        self.assertEqual(gate["critical_contract_regressions"][0]["contract"], "largest_remainder")
        self.assertIn("critical contract regressions", gate["blocked_reason"])


def row(seed: str, condition_name: str, task_accuracy: float) -> dict:
    return {"seed": seed, "condition": condition_name, "task_accuracy": task_accuracy}


def condition(
    *,
    task_mean: float,
    contract_macro: float,
    contracts: dict | None = None,
) -> dict:
    return {
        "task_accuracy_mean": task_mean,
        "contract_macro_mean": contract_macro,
        "contract_breakdown": contracts or {"stable_order": {"accuracy": contract_macro, "passed": 1, "total": 1}},
    }


if __name__ == "__main__":
    unittest.main()
