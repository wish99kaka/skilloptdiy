import unittest

from textskill_optimizer.paper import (
    CodeIdentity,
    ZeroCostGateEvidence,
    assess_zero_cost_gate,
)


COMMIT = "a" * 40


class PaperZeroCostPolicyTests(unittest.TestCase):
    def test_only_same_clean_commit_with_tests_and_network_guard_is_authorized(
        self,
    ) -> None:
        evidence = ZeroCostGateEvidence(
            test_returncode=0,
            before=CodeIdentity(COMMIT, clean=True),
            after=CodeIdentity(COMMIT, clean=True),
            network_guard_active=True,
            external_calls=0,
        )

        decision = assess_zero_cost_gate(evidence)

        self.assertTrue(decision.authorized)
        self.assertEqual(decision.status, "passed")
        self.assertEqual(decision.violations, ())

    def test_failure_dirty_state_or_identity_change_stays_blocked(self) -> None:
        cases = (
            ZeroCostGateEvidence(
                test_returncode=1,
                before=CodeIdentity(COMMIT, clean=True),
                after=CodeIdentity(COMMIT, clean=True),
                network_guard_active=True,
                external_calls=0,
            ),
            ZeroCostGateEvidence(
                test_returncode=0,
                before=CodeIdentity(COMMIT, clean=False),
                after=CodeIdentity(COMMIT, clean=False),
                network_guard_active=True,
                external_calls=0,
            ),
            ZeroCostGateEvidence(
                test_returncode=0,
                before=CodeIdentity(COMMIT, clean=True),
                after=CodeIdentity("b" * 40, clean=True),
                network_guard_active=True,
                external_calls=0,
            ),
            ZeroCostGateEvidence(
                test_returncode=0,
                before=CodeIdentity(COMMIT, clean=True),
                after=CodeIdentity(COMMIT, clean=True),
                network_guard_active=False,
                external_calls=0,
            ),
        )

        for evidence in cases:
            with self.subTest(evidence=evidence):
                decision = assess_zero_cost_gate(evidence)
                self.assertFalse(decision.authorized)
                self.assertNotEqual(decision.status, "passed")


if __name__ == "__main__":
    unittest.main()
