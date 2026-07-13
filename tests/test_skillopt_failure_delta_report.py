import json
import tempfile
import unittest
from pathlib import Path

from work.skillopt_failure_delta_report import (
    build_failure_delta_report,
    render_compact_summary,
    render_markdown,
)


class SkillOptFailureDeltaReportTests(unittest.TestCase):
    def test_reports_evidence_guided_rejection_after_audit_passes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            executive_dir = run_dir / "seed-a" / "executive"
            executive_dir.mkdir(parents=True)
            write_json(
                run_dir / "summary.json",
                {
                    "development_gate": {
                        "passed": False,
                        "blocked_reason": "executive won 0 seeds; required 1",
                        "best_baseline_condition": "human_skill",
                        "best_baseline_mean": 1.0,
                        "executive_mean": 0.0,
                        "mean_delta": -1.0,
                        "seed_wins_vs_best_baseline": 0,
                        "required_seed_wins": 1,
                    },
                    "aggregate": {
                        "executive": {
                            "experiment_internal_usage_summary": {"actual_total_tokens": 123}
                        }
                    },
                    "rows": [
                        row("seed-a", "human_skill", 1.0),
                        row("seed-a", "executive", 0.0, accepted_steps=0, total_steps=1),
                    ],
                },
            )
            write_json(
                run_dir / "smoke_gate_report.json",
                {
                    "status": "stop",
                    "reason": "development gate failed",
                    "proposal_audit": {
                        "status": "pass",
                        "required_record_count": 1,
                        "failed_required_record_count": 0,
                    },
                },
            )
            write_json(
                executive_dir / "result.json",
                {
                    "accepted_steps": 0,
                    "total_steps": 1,
                    "best_validation_score": 0.0,
                    "history": [
                        {
                            "candidate": "initial",
                            "epoch": 0,
                            "accepted": True,
                            "validation_score": 1.0,
                            "metadata": {},
                        },
                        {
                            "candidate": "atomic-epoch-1-batch-1",
                            "epoch": 1,
                            "accepted": False,
                            "validation_score": 0.0,
                            "metadata": {
                                "phase": "fast_update",
                                "step": 1,
                                "rejection_reason": "validation_gate_rejected",
                                "selected_edits": [
                                    {
                                        "operation": "add",
                                        "target": "__end__",
                                        "priority": 1.0,
                                        "content": "Check stable order.",
                                    }
                                ],
                            },
                        },
                    ],
                },
            )
            append_jsonl(
                executive_dir / "proposals.jsonl",
                {
                    "optimizer_controls": {
                        "phase": "reflection",
                        "epoch": 1,
                        "batch_index": 1,
                        "step": 1,
                    },
                    "proposal_targeting_audit": {
                        "required": True,
                        "contract_rejection_evidence_available": True,
                        "priority_contracts": ["stable_order"],
                        "missing_targeted_contract_count": 0,
                    },
                    "proposals": [
                        {
                            "name": "target-stable-order",
                            "metadata": {
                                "targeted_contracts": ["stable_order"],
                                "evidence_source": "contract_rejection_evidence",
                            },
                        }
                    ],
                },
            )
            write_json(
                executive_dir / "selection_atomic-epoch-1-batch-1_gate.json",
                {
                    "accepted": False,
                    "current_mean": 1.0,
                    "candidate_mean": 0.0,
                    "contract_evidence": {
                        "summary": {
                            "negative_contract_count": 1,
                            "no_improvement_contract_count": 0,
                        },
                        "contract_deltas": {
                            "stable_order": {
                                "contract": "stable_order",
                                "current_accuracy": 1.0,
                                "candidate_accuracy": 0.0,
                                "delta": -1.0,
                            }
                        },
                    },
                },
            )

            report = build_failure_delta_report(run_dir)
            markdown = render_markdown(report)
            compact = render_compact_summary(report)

        self.assertEqual(report["summary"]["proposal_audit_status"], "pass")
        self.assertEqual(report["summary"]["evidence_required_rejected_count"], 1)
        self.assertEqual(report["diagnosis"]["primary_blocker"], "proposal_effectiveness_after_contract_evidence")
        self.assertIn("priority_contract_regression", report["steps"][0]["failure_labels"])
        self.assertEqual(
            report["contract_summary"]["top_evidence_required_regressed_contracts"],
            [{"contract": "stable_order", "count": 1}],
        )
        self.assertIn("SkillOpt Failure Delta Report", markdown)
        self.assertNotIn("\n", compact)
        self.assertIn('blocker="proposal_effectiveness_after_contract_evidence"', compact)
        self.assertIn("top_regressed=stable_order:1", compact)


def row(seed: str, condition: str, score: float, *, accepted_steps=None, total_steps=None) -> dict:
    payload = {
        "seed": seed,
        "condition": condition,
        "task_accuracy": score,
        "family_macro_accuracy": score,
        "contract_macro_accuracy": score,
        "duration_seconds": 1.0,
    }
    if accepted_steps is not None:
        payload["accepted_steps"] = accepted_steps
    if total_steps is not None:
        payload["total_steps"] = total_steps
    return payload


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def append_jsonl(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
