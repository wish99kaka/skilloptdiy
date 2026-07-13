import json
import tempfile
import unittest
from pathlib import Path

from work.skillopt_contract_effect_audit import (
    build_contract_effect_audit,
    render_compact_summary,
)


class SkillOptContractEffectAuditTests(unittest.TestCase):
    def test_passes_when_target_improves_and_protected_contract_is_stable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            write_effect_artifacts(
                run_dir,
                target_delta=0.5,
                protected_delta=0.0,
            )

            report = build_contract_effect_audit(run_dir)
            compact = render_compact_summary(report)

        self.assertEqual(report["status"], "pass")
        self.assertEqual(report["effective_step_count"], 1)
        self.assertEqual(report["protected_regression_count"], 0)
        self.assertIn("contract_effect status=pass", compact)
        self.assertNotIn("\n", compact)

    def test_fails_when_targeted_contract_does_not_improve(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            write_effect_artifacts(
                run_dir,
                target_delta=0.0,
                protected_delta=0.0,
            )

            report = build_contract_effect_audit(run_dir)

        self.assertEqual(report["status"], "fail")
        self.assertEqual(report["effective_step_count"], 0)
        self.assertIn("targeted_contract_not_improved", report["records"][0]["issues"])

    def test_fails_when_protected_contract_regresses(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            write_effect_artifacts(
                run_dir,
                target_delta=0.5,
                protected_delta=-0.5,
                accepted=True,
            )

            report = build_contract_effect_audit(run_dir)

        self.assertEqual(report["status"], "fail")
        self.assertEqual(report["protected_regression_count"], 1)
        self.assertIn("protected_contract_regressed", report["records"][0]["issues"])

    def test_records_rejected_protected_regressions_without_blocking_scale_up(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            write_effect_artifacts(
                run_dir,
                target_delta=0.5,
                protected_delta=-0.5,
                accepted=False,
            )

            report = build_contract_effect_audit(run_dir)

        self.assertEqual(report["status"], "fail")
        self.assertEqual(report["reason"], "no accepted evidence-guided candidate improved a targeted contract")
        self.assertEqual(report["protected_regression_count"], 0)
        self.assertEqual(report["rejected_protected_regression_count"], 1)

    def test_passes_when_accepted_steps_are_safe_even_if_rejected_steps_regress(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            write_effect_artifacts(
                run_dir,
                target_delta=0.5,
                protected_delta=0.0,
                extra_rejected_protected_regression=True,
            )

            report = build_contract_effect_audit(run_dir)

        self.assertEqual(report["status"], "pass")
        self.assertEqual(report["protected_regression_count"], 0)
        self.assertEqual(report["rejected_protected_regression_count"], 1)


def write_effect_artifacts(
    run_dir: Path,
    *,
    target_delta: float,
    protected_delta: float,
    accepted: bool | None = None,
    extra_rejected_protected_regression: bool = False,
) -> None:
    accepted = target_delta > 0 and protected_delta >= 0 if accepted is None else accepted
    executive_dir = run_dir / "seed-a" / "executive"
    executive_dir.mkdir(parents=True)
    write_json(
        run_dir / "summary.json",
        {
            "development_gate": {"best_baseline_condition": "human_skill"},
            "rows": [],
            "aggregate": {},
        },
    )
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
                    "accepted": accepted,
                    "validation_score": 1.0 if accepted else 0.5,
                    "metadata": {
                        "phase": "fast_update",
                        "step": 1,
                        "rejection_reason": None if accepted else "validation_gate_rejected",
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
            + (
                [
                    {
                        "candidate": "atomic-epoch-1-batch-2",
                        "epoch": 1,
                        "accepted": False,
                        "validation_score": 0.5,
                        "metadata": {
                            "phase": "fast_update",
                            "step": 2,
                            "rejection_reason": "contract_policy_rejected",
                            "selected_edits": [
                                {
                                    "operation": "add",
                                    "target": "__end__",
                                    "priority": 1.0,
                                    "content": "Rejected protected regression.",
                                }
                            ],
                        },
                    }
                ]
                if extra_rejected_protected_regression
                else []
            )
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
        },
    )
    if extra_rejected_protected_regression:
        append_jsonl(
            executive_dir / "proposals.jsonl",
            {
                "optimizer_controls": {
                    "phase": "reflection",
                    "epoch": 1,
                    "batch_index": 2,
                    "step": 2,
                },
                "proposal_targeting_audit": {
                    "required": True,
                    "contract_rejection_evidence_available": True,
                    "priority_contracts": ["stable_order"],
                    "missing_targeted_contract_count": 0,
                    "proposal_policy": {
                        "anti_regression_contracts": [
                            {"contract": "input_validation", "reason": "protect_against_regression"}
                        ],
                        "protected_priority_contracts": [],
                    },
                    "proposals": [
                        {
                            "name": "rejected-protected-regression",
                            "targeted_contracts": ["stable_order"],
                            "protected_contracts": ["input_validation"],
                        }
                    ],
                },
                "proposals": [
                    {
                        "name": "rejected-protected-regression",
                        "metadata": {
                            "targeted_contracts": ["stable_order"],
                            "protected_contracts": ["input_validation"],
                            "evidence_source": "contract_rejection_evidence",
                        },
                    }
                ],
            },
        )
    write_json(
        executive_dir / "selection_atomic-epoch-1-batch-1_gate.json",
        {
            "accepted": accepted,
            "current_mean": 0.5,
            "candidate_mean": 1.0 if accepted else 0.5,
            "contract_evidence": {
                "contract_deltas": {
                    "stable_order": contract_delta("stable_order", target_delta),
                    "input_validation": contract_delta("input_validation", protected_delta),
                },
                "summary": {},
            },
        },
    )
    if extra_rejected_protected_regression:
        write_json(
            executive_dir / "selection_atomic-epoch-1-batch-2_gate.json",
            {
                "accepted": False,
                "current_mean": 1.0,
                "candidate_mean": 0.5,
                "contract_evidence": {
                    "contract_deltas": {
                        "stable_order": contract_delta("stable_order", 0.0),
                        "input_validation": contract_delta("input_validation", -0.5),
                    },
                    "summary": {},
                },
            },
        )


def contract_delta(contract: str, delta: float) -> dict:
    current = 0.5
    candidate = current + delta
    return {
        "contract": contract,
        "current_accuracy": current,
        "candidate_accuracy": candidate,
        "delta": delta,
    }


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def append_jsonl(path: Path, payload: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload) + "\n")


if __name__ == "__main__":
    unittest.main()
