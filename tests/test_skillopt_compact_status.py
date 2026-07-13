import json
import os
import tempfile
import unittest
from pathlib import Path

from work.skillopt_compact_status import build_compact_status, render_text


class SkillOptCompactStatusTests(unittest.TestCase):
    def test_reports_seed_usage_and_slow_gates_without_large_logs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            executive_dir = run_dir / "seed-a" / "executive"
            executive_dir.mkdir(parents=True)
            write_json(
                run_dir / "summary.json",
                {
                    "aggregate": {
                        "executive": {
                            "experiment_internal_usage_summary": {
                                "actual_total_tokens": 1234,
                                "actual_prompt_tokens": 1000,
                                "actual_completion_tokens": 234,
                                "by_kind": {
                                    "optimizer_model_api": {"duration_seconds_total": 90.0},
                                    "optimizer_command": {"duration_seconds_total": 120.0},
                                },
                            }
                        }
                    },
                    "development_gate": {
                        "passed": False,
                        "executive_mean": 0.5,
                        "mean_delta": 0.0,
                        "contract_macro_delta": -0.1,
                        "required_contract_macro_delta": 0.0,
                        "critical_contract_regressions": [
                            {"contract": "largest_remainder", "delta": -1.0}
                        ],
                        "seed_wins_vs_best_baseline": 1,
                        "required_seed_wins": 2,
                    },
                },
            )
            write_json(run_dir / "runner_report.json", {"status": "complete"})
            write_json(run_dir / "runner_execution.json", {"duration_seconds": 300.0, "returncode": 0})
            write_json(
                run_dir / "smoke_gate_report.json",
                {
                    "status": "stop",
                    "reason": "development gate failed",
                    "contract_effect_audit": {"status": "fail"},
                },
            )
            write_json(
                run_dir / "failure_delta_report.json",
                {"diagnosis": {"primary_blocker": "proposal_effectiveness_after_contract_evidence"}},
            )
            (executive_dir / "selection_initial.json").write_text("{}", encoding="utf-8")
            (executive_dir / "result.json").write_text("{}", encoding="utf-8")
            (executive_dir / "proposals.jsonl").write_text("{}\n{}\n", encoding="utf-8")
            append_jsonl(
                executive_dir / "usage_ledger.jsonl",
                {
                    "kind": "optimizer_model_api",
                    "actual_total_tokens": 300,
                    "duration_seconds": 30.0,
                },
            )
            candidate = executive_dir / "candidate_atomic-epoch-1-batch-1.md"
            gate = executive_dir / "selection_atomic-epoch-1-batch-1_gate.json"
            candidate.write_text("candidate", encoding="utf-8")
            gate.write_text("{}", encoding="utf-8")
            os.utime(candidate, (1000, 1000))
            os.utime(gate, (1240, 1240))

            status = build_compact_status(run_dir)
            text = render_text(status)

        self.assertEqual(status["usage"]["optimizer_actual_total_tokens"], 1234)
        self.assertEqual(status["seeds"][0]["proposal_records"], 2)
        self.assertEqual(status["seeds"][0]["optimizer_tokens"], 300)
        self.assertEqual(status["slow_gates"][0]["seconds"], 240)
        self.assertIn("skillopt_status", text)
        self.assertIn("effect=fail", text)
        self.assertIn("contract_delta=-0.1/0.0", text)
        self.assertIn("critical_regressions=1", text)
        self.assertIn("seed-a result=1 props=2 cand=1 gates=1", text)
        self.assertIn("timing=0", text)
        self.assertIn("slow_gates seed-a/atomic-epoch-1-batch-1=4.0m", text)

    def test_prefers_timing_events_over_mtime_gap(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            executive_dir = run_dir / "seed-a" / "executive"
            executive_dir.mkdir(parents=True)
            candidate = executive_dir / "candidate_atomic-epoch-1-batch-1.md"
            gate = executive_dir / "selection_atomic-epoch-1-batch-1_gate.json"
            candidate.write_text("candidate", encoding="utf-8")
            gate.write_text("{}", encoding="utf-8")
            os.utime(candidate, (1000, 1000))
            os.utime(gate, (4600, 4600))
            append_jsonl(
                executive_dir / "timing_events.jsonl",
                {
                    "event": "validation_finished",
                    "candidate_name": "atomic-epoch-1-batch-1",
                    "duration_seconds": 180.0,
                },
            )

            status = build_compact_status(run_dir)
            text = render_text(status)

        self.assertEqual(status["seeds"][0]["timing_event_records"], 1)
        self.assertEqual(status["slow_gates"][0]["seconds"], 180.0)
        self.assertEqual(status["slow_gates"][0]["source"], "timing_events")
        self.assertIn("slow_gates seed-a/atomic-epoch-1-batch-1=3.0m", text)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def append_jsonl(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
