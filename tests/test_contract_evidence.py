import unittest

from textskill_optimizer.contract_evidence import (
    UNKNOWN_CONTRACT,
    contract_breakdown,
    contract_delta_evidence,
)
from textskill_optimizer.models import EvaluationReport, Score, Task, TaskOutput, TaskResult


class ContractEvidenceTests(unittest.TestCase):
    def test_contract_delta_identifies_regression_and_no_improvement(self) -> None:
        current = EvaluationReport(
            "current",
            [
                result("c1", ["stable_order"], True),
                result("c2", ["input_validation"], False),
            ],
        )
        candidate = EvaluationReport(
            "candidate",
            [
                result("c1", ["stable_order"], False),
                result("c2", ["input_validation"], False),
            ],
        )

        evidence = contract_delta_evidence(current, candidate)

        self.assertEqual(evidence["contract_deltas"]["stable_order"]["delta"], -1.0)
        self.assertEqual(evidence["top_negative_contracts"][0]["contract"], "stable_order")
        self.assertEqual(evidence["top_no_improvement_contracts"][0]["contract"], "input_validation")
        self.assertEqual(evidence["summary"]["negative_contract_count"], 1)
        self.assertEqual(evidence["summary"]["no_improvement_contract_count"], 1)

    def test_contract_breakdown_falls_back_to_unknown_contract(self) -> None:
        report = EvaluationReport("candidate", [result("c1", [], False)])

        breakdown = contract_breakdown(report)

        self.assertEqual(breakdown[UNKNOWN_CONTRACT]["total"], 1)
        self.assertEqual(breakdown[UNKNOWN_CONTRACT]["accuracy"], 0.0)


def result(task_id: str, tags: list[str], success: bool) -> TaskResult:
    return TaskResult(
        task=Task(id=task_id, input="", metadata={"contract_tags": tags}),
        output=TaskOutput(value=None),
        score=Score(1.0 if success else 0.0, success=success),
    )


if __name__ == "__main__":
    unittest.main()
