import unittest

from textskill_optimizer.paper import OptimizerStage, PaperEdit, PaperEditOperation
from textskill_optimizer.paper.responses import (
    OptimizerContractViolation,
    epoch_response_schema,
    optimizer_response_schema,
    parse_epoch_response,
    parse_patch_response,
    parse_rank_response,
)


class PaperOptimizerResponseTests(unittest.TestCase):
    def test_slow_and_meta_text_contracts_are_strict(self) -> None:
        cases = (
            (
                OptimizerStage.PROPOSE_SLOW_UPDATE,
                "slow_update_content",
            ),
            (OptimizerStage.UPDATE_META_SKILL, "meta_skill_content"),
        )
        for stage, field in cases:
            with self.subTest(stage=stage):
                payload = {"reasoning": "evidence", field: "durable guidance"}
                parsed = parse_epoch_response(stage=stage, payload=payload)
                self.assertEqual(parsed.content, "durable guidance")
                self.assertEqual(parsed.response_schema, epoch_response_schema(stage))
                with self.assertRaises(OptimizerContractViolation):
                    parse_epoch_response(
                        stage=stage,
                        payload={**payload, "selection_diagnostics": {}},
                    )
                with self.assertRaises(OptimizerContractViolation):
                    parse_epoch_response(
                        stage=stage,
                        payload={"reasoning": "evidence", field: "   "},
                    )

    def test_failure_analysis_rejects_response_side_channels(self) -> None:
        payload = {
            "batch_size": 2,
            "failure_summary": [
                {
                    "failure_type": "tool-use",
                    "count": 2,
                    "description": "The agent trusts an unverified tool result.",
                }
            ],
            "patch": {
                "reasoning": "Require verification.",
                "edits": [{"op": "append", "content": "- Verify tool results."}],
            },
        }

        parsed = parse_patch_response(
            stage=OptimizerStage.REFLECT_FAILURE,
            payload=payload,
            edit_budget=1,
            edit_id_prefix="failure-1",
            expected_batch_size=2,
        )

        self.assertEqual(len(parsed.edits), 1)
        self.assertEqual(parsed.edits[0].operation, PaperEditOperation.APPEND)
        self.assertEqual(parsed.edits[0].source_type, "failure")
        self.assertEqual(
            optimizer_response_schema(OptimizerStage.REFLECT_FAILURE, edit_budget=1),
            parsed.response_schema,
        )
        payload["selection_diagnostics"] = {"private": True}
        with self.assertRaisesRegex(OptimizerContractViolation, "unknown field"):
            parse_patch_response(
                stage=OptimizerStage.REFLECT_FAILURE,
                payload=payload,
                edit_budget=1,
                edit_id_prefix="failure-1",
                expected_batch_size=2,
            )
        payload.pop("selection_diagnostics")
        payload["failure_summary"][0]["failure_type"] = "   "
        with self.assertRaisesRegex(OptimizerContractViolation, "failure_summary"):
            parse_patch_response(
                stage=OptimizerStage.REFLECT_FAILURE,
                payload=payload,
                edit_budget=1,
                edit_id_prefix="failure-1",
                expected_batch_size=2,
            )

    def test_merge_contract_preserves_source_and_support_for_ranking(self) -> None:
        parsed = parse_patch_response(
            stage=OptimizerStage.MERGE_FINAL_FAILURE_PRIORITIZED,
            payload={
                "reasoning": "Failure rule has broader support.",
                "edits": [
                    {
                        "op": "replace",
                        "target": "old",
                        "content": "new",
                        "support_count": 3,
                        "source_type": "failure",
                    }
                ],
            },
            edit_budget=4,
            edit_id_prefix="final",
        )

        self.assertEqual(parsed.edits[0].support_count, 3)
        self.assertEqual(parsed.edits[0].source_type, "failure")

        with self.assertRaisesRegex(OptimizerContractViolation, "source_type"):
            parse_patch_response(
                stage=OptimizerStage.MERGE_FAILURE,
                payload={
                    "reasoning": "bad source",
                    "edits": [
                        {
                            "op": "append",
                            "content": "rule",
                            "support_count": 1,
                            "source_type": "success",
                        }
                    ],
                },
                edit_budget=4,
                edit_id_prefix="failure-merge",
            )

    def test_ranking_is_a_strict_model_decision_with_no_local_fallback(self) -> None:
        candidates = (
            PaperEdit("a", PaperEditOperation.APPEND, content="A"),
            PaperEdit("b", PaperEditOperation.APPEND, content="B"),
            PaperEdit("c", PaperEditOperation.APPEND, content="C"),
        )

        selected = parse_rank_response(
            payload={"reasoning": "B then A", "selected_indices": [1, 0]},
            candidates=candidates,
            edit_budget=2,
        )
        self.assertEqual([edit.edit_id for edit in selected], ["b", "a"])

        for invalid in ([0, 0], [3], [0, 1, 2]):
            with self.subTest(invalid=invalid):
                with self.assertRaises(OptimizerContractViolation):
                    parse_rank_response(
                        payload={"reasoning": "invalid", "selected_indices": invalid},
                        candidates=candidates,
                        edit_budget=2,
                    )


if __name__ == "__main__":
    unittest.main()
