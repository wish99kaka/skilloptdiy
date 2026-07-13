import json
import tempfile
import unittest
from pathlib import Path

from work.skillopt_smoke_gate import build_smoke_gate_report, render_compact_summary


class SkillOptSmokeGateTests(unittest.TestCase):
    def test_missing_summary_is_missing_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            report = build_smoke_gate_report(Path(tmp))

        self.assertEqual(report["status"], "missing_artifacts")
        self.assertFalse(report["scale_up_recommended"])

    def test_not_triggered_when_no_proposal_saw_contract_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            write_smoke_artifacts(run_dir, development_gate_passed=True, accepted_steps=1)
            write_proposal_audit(run_dir, required=False, evidence_available=False, missing_count=0)

            report = build_smoke_gate_report(run_dir)

        self.assertEqual(report["status"], "inconclusive")
        self.assertIn("not triggered", report["reason"])
        self.assertFalse(report["scale_up_recommended"])

    def test_no_accepted_step_stops_even_when_audit_was_not_triggered(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            write_smoke_artifacts(run_dir, development_gate_passed=True, accepted_steps=0)
            write_proposal_audit(run_dir, required=False, evidence_available=False, missing_count=0)

            report = build_smoke_gate_report(run_dir)

        self.assertEqual(report["status"], "stop")
        self.assertEqual(report["reason"], "no accepted executive optimization step")
        self.assertEqual(report["proposal_audit"]["status"], "not_triggered")
        self.assertFalse(report["scale_up_recommended"])

    def test_missing_proposal_logs_stop_the_gate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            write_smoke_artifacts(run_dir, development_gate_passed=True, accepted_steps=1)

            report = build_smoke_gate_report(run_dir)

        self.assertEqual(report["status"], "stop")
        self.assertEqual(report["reason"], "proposal logs are missing")
        self.assertFalse(report["scale_up_recommended"])

    def test_failed_targeting_audit_stops_the_gate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            write_smoke_artifacts(run_dir, development_gate_passed=True, accepted_steps=1)
            write_proposal_audit(run_dir, required=True, evidence_available=True, missing_count=1)

            report = build_smoke_gate_report(run_dir)

        self.assertEqual(report["status"], "stop")
        self.assertEqual(report["reason"], "proposal targeting audit failed")
        self.assertFalse(report["scale_up_recommended"])

    def test_pass_recommends_scale_up(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            write_smoke_artifacts(run_dir, development_gate_passed=True, accepted_steps=1)
            write_contract_effect_artifacts(run_dir, target_delta=0.5, protected_delta=0.0)

            report = build_smoke_gate_report(run_dir)

        self.assertEqual(report["status"], "pass")
        self.assertTrue(report["scale_up_recommended"])
        self.assertTrue(report["checks"]["proposal_targeting_audit_passed"])
        self.assertTrue(report["checks"]["contract_effect_audit_passed"])

    def test_contract_effect_failure_stops_the_gate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            write_smoke_artifacts(run_dir, development_gate_passed=True, accepted_steps=1)
            write_contract_effect_artifacts(run_dir, target_delta=0.0, protected_delta=0.0)

            report = build_smoke_gate_report(run_dir)

        self.assertEqual(report["status"], "stop")
        self.assertEqual(report["reason"], "contract effect audit failed")
        self.assertFalse(report["scale_up_recommended"])

    def test_compact_summary_is_one_line(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            write_smoke_artifacts(run_dir, development_gate_passed=False, accepted_steps=1)
            write_proposal_audit(run_dir, required=True, evidence_available=True, missing_count=0)

            text = render_compact_summary(build_smoke_gate_report(run_dir))

        self.assertNotIn("\n", text)
        self.assertIn("smoke_gate status=stop", text)
        self.assertIn("audit=pass", text)
        self.assertIn("gate_passed=False", text)


def write_smoke_artifacts(run_dir: Path, *, development_gate_passed: bool, accepted_steps: int) -> None:
    executive_dir = run_dir / "seed-a" / "executive"
    executive_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "manifest": {"benchmark": "coding-hidden-v2"},
        "rows": [
            {"seed": "seed-a", "condition": "human_skill", "contract_macro_accuracy": 0.8},
            {
                "seed": "seed-a",
                "condition": "executive",
                "contract_macro_accuracy": 0.9,
                "accepted_steps": accepted_steps,
            },
        ],
        "aggregate": {},
        "development_gate": {
            "passed": development_gate_passed,
            "locked_test_recommended": development_gate_passed,
            "blocked_reason": "" if development_gate_passed else "development gate failed",
        },
        "locked_test_recommended": development_gate_passed,
    }
    runner_report = {
        "status": "complete",
        "development_gate": summary["development_gate"],
        "anomaly_summary": {"persistent_anomaly_count": 0},
    }
    write_json(run_dir / "summary.json", summary)
    write_json(run_dir / "runner_report.json", runner_report)


def write_proposal_audit(
    run_dir: Path,
    *,
    required: bool,
    evidence_available: bool,
    missing_count: int,
) -> None:
    proposal_log = run_dir / "seed-a" / "executive" / "proposals.jsonl"
    payload = {
        "proposal_targeting_audit": {
            "required": required,
            "contract_rejection_evidence_available": evidence_available,
            "missing_targeted_contract_count": missing_count,
            "priority_contracts": ["stable_order"],
            "proposal_count": 1,
        }
    }
    proposal_log.write_text(json.dumps(payload) + "\n", encoding="utf-8")


def write_contract_effect_artifacts(
    run_dir: Path,
    *,
    target_delta: float,
    protected_delta: float,
) -> None:
    executive_dir = run_dir / "seed-a" / "executive"
    proposal_log = executive_dir / "proposals.jsonl"
    payload = {
        "optimizer_controls": {
            "phase": "reflection",
            "epoch": 1,
            "batch_index": 1,
            "step": 1,
        },
        "proposal_targeting_audit": {
            "required": True,
            "contract_rejection_evidence_available": True,
            "missing_targeted_contract_count": 0,
            "priority_contracts": ["stable_order"],
            "proposal_policy": {
                "anti_regression_contracts": [
                    {"contract": "input_validation", "reason": "protect_against_regression"}
                ],
                "protected_priority_contracts": [],
            },
            "proposals": [
                {
                    "name": "target-stable-order",
                    "targeted_contracts": ["stable_order"],
                    "protected_contracts": ["input_validation"],
                }
            ],
        },
        "proposals": [
            {
                "name": "target-stable-order",
                "metadata": {
                    "targeted_contracts": ["stable_order"],
                    "protected_contracts": ["input_validation"],
                    "evidence_source": "contract_rejection_evidence",
                },
            }
        ],
    }
    proposal_log.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    write_json(
        executive_dir / "result.json",
        {
            "history": [
                {
                    "candidate": "initial",
                    "epoch": 0,
                    "accepted": True,
                    "validation_score": 0.0,
                    "metadata": {},
                },
                {
                    "candidate": "atomic-epoch-1-batch-1",
                    "epoch": 1,
                    "accepted": target_delta > 0 and protected_delta >= 0,
                    "validation_score": 1.0 if target_delta > 0 and protected_delta >= 0 else 0.5,
                    "metadata": {
                        "phase": "fast_update",
                        "step": 1,
                        "selected_edits": [
                            {
                                "operation": "add",
                                "target": "__end__",
                                "priority": 1.0,
                                "content": "Improve stable order while preserving validation.",
                            }
                        ],
                    },
                },
            ]
        },
    )
    write_json(
        executive_dir / "selection_atomic-epoch-1-batch-1_gate.json",
        {
            "accepted": target_delta > 0 and protected_delta >= 0,
            "current_mean": 0.5,
            "candidate_mean": 1.0 if target_delta > 0 and protected_delta >= 0 else 0.5,
            "contract_evidence": {
                "contract_deltas": {
                    "stable_order": contract_delta("stable_order", target_delta),
                    "input_validation": contract_delta("input_validation", protected_delta),
                },
                "summary": {},
            },
        },
    )


def contract_delta(contract: str, delta: float) -> dict:
    current = 0.5
    return {
        "contract": contract,
        "current_accuracy": current,
        "candidate_accuracy": current + delta,
        "delta": delta,
    }


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
