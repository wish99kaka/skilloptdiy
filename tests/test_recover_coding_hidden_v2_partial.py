import json
import tempfile
import unittest
from pathlib import Path

from work.recover_coding_hidden_v2_partial import (
    recover_partial_summary,
    write_recovered_executive_results,
)


class RecoverCodingHiddenV2PartialTests(unittest.TestCase):
    def test_recovers_partial_executive_row_from_initial_selection(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            write_json(run_dir / "experiment_manifest.json", {"benchmark": "coding-hidden-v2", "seeds": ["seed-a"]})
            for condition in ("no_skill", "human_skill", "one_shot"):
                condition_dir = run_dir / "seed-a" / condition
                write_json(condition_dir / "selection.json", report("baseline", True))
                write_json(condition_dir / "timing.json", {"duration_seconds": 1.0})
            executive_dir = run_dir / "seed-a" / "executive"
            write_json(executive_dir / "selection_initial.json", report("executive", True))
            write_json(
                executive_dir / "selection_atomic-epoch-1-batch-1_gate.json",
                {"accepted": False, "candidate_mean": 0.0, "current_mean": 1.0},
            )

            summary = recover_partial_summary(run_dir)

        executive_rows = [
            row for row in summary["rows"] if row["condition"] == "executive"
        ]
        self.assertEqual(len(executive_rows), 1)
        self.assertTrue(executive_rows[0]["partial_recovery"])
        self.assertEqual(executive_rows[0]["partial_recovery_source"], "selection_initial.json")
        self.assertFalse(summary["recovery"]["locked_test_recommended"])
        self.assertIn("contract_macro_mean", summary["aggregate"]["executive"])

    def test_writes_recovered_result_only_after_early_stop_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            write_json(run_dir / "experiment_manifest.json", {"benchmark": "coding-hidden-v2", "seeds": ["seed-a"]})
            executive_dir = run_dir / "seed-a" / "executive"
            write_json(executive_dir / "selection_initial.json", report("executive", True))
            (executive_dir / "best_skill.md").parent.mkdir(parents=True, exist_ok=True)
            (executive_dir / "best_skill.md").write_text("# Skill\n", encoding="utf-8")
            for index in range(3):
                write_json(
                    executive_dir / f"selection_candidate-{index}_gate.json",
                    gate_payload(index),
                )
            (executive_dir / "usage_ledger.jsonl").write_text(
                json.dumps({"duration_seconds": 2.5}) + "\n",
                encoding="utf-8",
            )

            written = write_recovered_executive_results(run_dir, early_stop_rejection_limit=3)
            result = json.loads((executive_dir / "result.json").read_text(encoding="utf-8"))
            timing = json.loads((executive_dir / "timing.json").read_text(encoding="utf-8"))

        self.assertEqual(written, ["seed-a"])
        self.assertEqual(result["stop_reason"], "recovered_early_stop_validation_rejection_limit")
        self.assertEqual(result["accepted_steps"], 0)
        self.assertEqual(len(result["rejected_buffer"]), 3)
        self.assertEqual(
            result["rejected_buffer"][0]["metadata"]["validation_gate"]["contract_evidence"]["top_no_improvement_contracts"][0]["contract"],
            "immutability",
        )
        self.assertEqual(result["rejected_buffer"][0]["failed_task_ids"], ["candidate-0"])
        self.assertEqual(timing["duration_seconds"], 2.5)


def report(task_id: str, success: bool) -> dict:
    return {
        "name": "selection",
        "average_score": float(success),
        "pass_rate": float(success),
        "results": [
            {
                "task": {
                    "id": task_id,
                    "input": "Fix",
                    "metadata": {
                        "benchmark_family": "family",
                        "contract_tags": ["immutability"],
                    },
                },
                "output": {"value": {}, "metadata": {}},
                "score": {"value": float(success), "success": success, "message": "", "metadata": {}},
            }
        ],
    }


def gate_payload(index: int) -> dict:
    return {
        "accepted": False,
        "candidate_mean": 0.0,
        "current_mean": 1.0,
        "contract_evidence": {
            "top_no_improvement_contracts": [
                {"contract": "immutability", "current_accuracy": 0.0, "candidate_accuracy": 0.0, "delta": 0.0}
            ]
        },
        "candidate_report": report(f"candidate-{index}", False),
    }


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
