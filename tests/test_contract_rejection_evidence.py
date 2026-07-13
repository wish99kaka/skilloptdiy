import unittest

from textskill_optimizer.contract_rejection_evidence import (
    audit_proposal_targeting,
    build_contract_rejection_evidence,
)
from textskill_optimizer.models import AtomicEdit, EditProposal


class ContractRejectionEvidenceTests(unittest.TestCase):
    def test_builds_compact_priority_contracts_from_rejected_buffer(self) -> None:
        evidence = build_contract_rejection_evidence([rejected("bad", "stable_order", -1.0)])

        self.assertTrue(evidence["available"])
        self.assertEqual(evidence["priority_contracts"][0]["contract"], "stable_order")
        self.assertEqual(
            evidence["recent_rejections"][0]["blocking_contracts"][0]["kind"],
            "negative_delta",
        )
        self.assertEqual(
            evidence["proposal_policy"]["anti_regression_contracts"][0]["contract"],
            "stable_order",
        )

    def test_audit_flags_missing_targeted_contract_metadata(self) -> None:
        evidence = build_contract_rejection_evidence([rejected("bad", "stable_order", -1.0)])
        proposal = EditProposal(
            "generic",
            rationale="Generic advice.",
            edits=(AtomicEdit("add", "__end__", "Check every contract."),),
        )

        audit = audit_proposal_targeting([proposal], evidence)

        self.assertTrue(audit["required"])
        self.assertEqual(audit["missing_targeted_contract_count"], 1)
        self.assertIn("missing_targeted_contracts", audit["proposals"][0]["issues"])

    def test_audit_accepts_priority_target_metadata(self) -> None:
        evidence = build_contract_rejection_evidence([rejected("bad", "stable_order", -1.0)])
        proposal = EditProposal(
            "targeted",
            rationale="Targeted advice.",
            metadata={
                "targeted_contracts": ["stable_order"],
                "protected_contracts": ["stable_order"],
                "evidence_source": "contract_rejection_evidence",
                "expected_behavior_change": "Preserve original ordering when fixing code.",
            },
            edits=(AtomicEdit("add", "__end__", "Preserve documented ordering invariants."),),
        )

        audit = audit_proposal_targeting([proposal], evidence)

        self.assertEqual(audit["missing_targeted_contract_count"], 0)
        self.assertTrue(audit["all_proposals_target_priority_contract"])
        self.assertEqual(audit["proposals"][0]["targeted_priority_contracts"], ["stable_order"])

    def test_audit_requires_anti_regression_guard_metadata(self) -> None:
        evidence = build_contract_rejection_evidence([rejected("bad", "stable_order", -1.0)])
        proposal = EditProposal(
            "targeted",
            rationale="Targeted advice.",
            metadata={
                "targeted_contracts": ["stable_order"],
                "evidence_source": "contract_rejection_evidence",
                "expected_behavior_change": "Preserve ordering.",
            },
            edits=(AtomicEdit("add", "__end__", "Preserve documented ordering invariants."),),
        )

        audit = audit_proposal_targeting([proposal], evidence)

        self.assertEqual(audit["missing_targeted_contract_count"], 1)
        self.assertIn("missing_anti_regression_guard", audit["proposals"][0]["issues"])
        self.assertEqual(audit["proposals"][0]["missing_protected_contracts"], ["stable_order"])

    def test_audit_requires_protection_for_currently_passing_priority_contracts(self) -> None:
        evidence = build_contract_rejection_evidence(
            [rejected_no_improvement("bad-1", "input_validation", current_accuracy=0.5)]
        )
        proposal = EditProposal(
            "targeted-without-protection",
            rationale="Target the failing contract.",
            metadata={
                "targeted_contracts": ["input_validation"],
                "evidence_source": "contract_rejection_evidence",
                "expected_behavior_change": "Improve invalid input handling.",
            },
            edits=(AtomicEdit("add", "__end__", "Validate input before mutating data."),),
        )

        audit = audit_proposal_targeting([proposal], evidence)

        self.assertEqual(
            evidence["proposal_policy"]["protected_priority_contracts"][0]["contract"],
            "input_validation",
        )
        self.assertEqual(audit["missing_targeted_contract_count"], 1)
        self.assertIn("missing_priority_contract_protection", audit["proposals"][0]["issues"])
        self.assertEqual(audit["proposals"][0]["missing_protected_contracts"], ["input_validation"])

    def test_audit_accepts_single_contract_when_currently_passing_priority_is_protected(self) -> None:
        evidence = build_contract_rejection_evidence(
            [rejected_no_improvement("bad-1", "input_validation", current_accuracy=0.5)]
        )
        proposal = EditProposal(
            "targeted-with-protection",
            rationale="Target the failing contract without losing partial success.",
            metadata={
                "targeted_contracts": ["input_validation"],
                "protected_contracts": ["input_validation"],
                "evidence_source": "contract_rejection_evidence",
                "expected_behavior_change": "Improve invalid input handling.",
            },
            edits=(AtomicEdit("add", "__end__", "Validate input before mutating data."),),
        )

        audit = audit_proposal_targeting([proposal], evidence)

        self.assertEqual(audit["missing_targeted_contract_count"], 0)

    def test_audit_requires_cooldown_override_for_repeated_no_improvement(self) -> None:
        evidence = build_contract_rejection_evidence(
            [
                rejected_no_improvement("bad-1", "input_validation"),
                rejected_no_improvement("bad-2", "input_validation"),
            ]
        )
        proposal = EditProposal(
            "repeat-input-validation",
            rationale="Try again.",
            metadata={
                "targeted_contracts": ["input_validation"],
                "protected_contracts": [],
                "evidence_source": "contract_rejection_evidence",
                "expected_behavior_change": "Handle invalid inputs.",
            },
            edits=(AtomicEdit("add", "__end__", "Check documented input validation."),),
        )

        audit = audit_proposal_targeting([proposal], evidence)

        self.assertEqual(
            evidence["proposal_policy"]["cooldown_contracts"][0]["contract"],
            "input_validation",
        )
        self.assertEqual(audit["missing_targeted_contract_count"], 1)
        self.assertIn("missing_cooldown_override", audit["proposals"][0]["issues"])
        self.assertEqual(audit["proposals"][0]["targeted_cooldown_contracts"], ["input_validation"])

    def test_audit_accepts_cooldown_override(self) -> None:
        evidence = build_contract_rejection_evidence(
            [
                rejected_no_improvement("bad-1", "input_validation"),
                rejected_no_improvement("bad-2", "input_validation"),
            ]
        )
        proposal = EditProposal(
            "repeat-input-validation",
            rationale="Try with narrower evidence.",
            metadata={
                "targeted_contracts": ["input_validation"],
                "evidence_source": "contract_rejection_evidence",
                "expected_behavior_change": "Handle invalid inputs without changing allocation logic.",
                "cooldown_override": "New failure trace shows exact exception validation was skipped.",
            },
            edits=(AtomicEdit("add", "__end__", "Check documented input validation."),),
        )

        audit = audit_proposal_targeting([proposal], evidence)

        self.assertEqual(audit["missing_targeted_contract_count"], 0)
        self.assertTrue(audit["proposals"][0]["cooldown_override_present"])

    def test_audit_requires_multi_contract_protection_for_single_contract_mode(self) -> None:
        evidence = build_contract_rejection_evidence(
            [
                rejected_no_improvement("bad-1", "input_validation", current_accuracy=0.5),
                rejected_no_improvement("bad-2", "stable_order", current_accuracy=0.5),
            ]
        )
        proposal = EditProposal(
            "broad-audit",
            rationale="Try all contracts.",
            metadata={
                "targeted_contracts": ["input_validation", "stable_order"],
                "evidence_source": "contract_rejection_evidence",
                "expected_behavior_change": "Improve validation and ordering together.",
                "cooldown_override": "New evidence says both checks failed together.",
            },
            edits=(AtomicEdit("add", "__end__", "Check every documented contract."),),
        )

        audit = audit_proposal_targeting([proposal], evidence)

        self.assertEqual(
            evidence["proposal_policy"]["single_contract_targeting"]["max_targeted_priority_contracts"],
            1,
        )
        self.assertIn(
            "multi_contract_without_required_protection",
            audit["proposals"][0]["issues"],
        )
        self.assertEqual(
            audit["proposals"][0]["missing_multi_contract_protections"],
            ["input_validation", "stable_order"],
        )

    def test_audit_accepts_multi_contract_when_prior_passing_contracts_are_protected(self) -> None:
        evidence = build_contract_rejection_evidence(
            [
                rejected_no_improvement("bad-1", "input_validation", current_accuracy=0.5),
                rejected_no_improvement("bad-2", "stable_order", current_accuracy=0.5),
            ]
        )
        proposal = EditProposal(
            "protected-broad-audit",
            rationale="Try all contracts while preserving partial successes.",
            metadata={
                "targeted_contracts": ["input_validation", "stable_order"],
                "protected_contracts": ["input_validation", "stable_order"],
                "evidence_source": "contract_rejection_evidence",
                "expected_behavior_change": "Improve validation and ordering together.",
                "cooldown_override": "New evidence says both checks failed together.",
            },
            edits=(AtomicEdit("add", "__end__", "Check every documented contract."),),
        )

        audit = audit_proposal_targeting([proposal], evidence)

        self.assertEqual(audit["missing_targeted_contract_count"], 0)


def rejected(candidate: str, contract: str, delta: float) -> dict:
    return {
        "candidate": candidate,
        "reason": "validation_gate_rejected",
        "validation_score": 0.5,
        "metadata": {
            "validation_gate": {
                "current_mean": 1.0,
                "candidate_mean": 0.5,
                "contract_evidence": {
                    "top_negative_contracts": [
                        {
                            "contract": contract,
                            "current_accuracy": 1.0,
                            "candidate_accuracy": 0.0,
                            "delta": delta,
                        }
                    ],
                    "top_no_improvement_contracts": [],
                },
            }
        },
    }


def rejected_no_improvement(candidate: str, contract: str, *, current_accuracy: float = 0.0) -> dict:
    return {
        "candidate": candidate,
        "reason": "validation_gate_rejected",
        "validation_score": 0.5,
        "metadata": {
            "validation_gate": {
                "current_mean": 0.5,
                "candidate_mean": 0.5,
                "contract_evidence": {
                    "top_negative_contracts": [],
                    "top_no_improvement_contracts": [
                        {
                            "contract": contract,
                            "current_accuracy": current_accuracy,
                            "candidate_accuracy": 0.0,
                            "delta": 0.0,
                        }
                    ],
                },
            }
        },
    }


if __name__ == "__main__":
    unittest.main()
