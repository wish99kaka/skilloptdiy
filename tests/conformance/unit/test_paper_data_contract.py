import unittest

from textskill_optimizer.paper import (
    DataFirewallViolation,
    PaperDataAccessPolicy,
    RunPhase,
    SelectionScore,
    SplitRole,
    strict_selection_decision,
)


class PaperDataContractTests(unittest.TestCase):
    def test_selection_controller_accepts_only_a_scalar(self) -> None:
        sentinel = object()

        score = SelectionScore.from_payload({"score": 0.75})

        self.assertEqual(score.value, 0.75)
        with self.assertRaisesRegex(DataFirewallViolation, "diagnostics"):
            SelectionScore.from_payload({"score": 0.75, "diagnostics": sentinel})

    def test_strict_gate_rejects_ties(self) -> None:
        current = SelectionScore(0.75)

        tie = strict_selection_decision(current=current, candidate=SelectionScore(0.75))
        improvement = strict_selection_decision(
            current=current,
            candidate=SelectionScore(0.75001),
        )

        self.assertFalse(tie.accepted)
        self.assertTrue(improvement.accepted)

    def test_test_split_is_inaccessible_during_optimization(self) -> None:
        policy = PaperDataAccessPolicy()

        with self.assertRaisesRegex(DataFirewallViolation, "test"):
            policy.require(split=SplitRole.TEST, phase=RunPhase.OPTIMIZATION)

        policy.require(split=SplitRole.TEST, phase=RunPhase.FINAL_EVALUATION)


if __name__ == "__main__":
    unittest.main()
