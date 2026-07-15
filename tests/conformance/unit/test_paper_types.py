import unittest

from textskill_optimizer.paper import (
    AlgorithmEvent,
    AlgorithmEventType,
    OptimizerBackend,
    OptimizerRequest,
    OptimizerResponse,
    OptimizerStage,
    PaperEdit,
    PaperEditOperation,
    PaperState,
    SelectionScore,
)


class PaperTypeTests(unittest.TestCase):
    def test_algorithm_event_round_trips_through_its_public_schema(self) -> None:
        event = AlgorithmEvent(
            sequence=7,
            event_type=AlgorithmEventType.MERGE_FINAL_FAILURE_PRIORITIZED,
            epoch=1,
            step=2,
            payload={"suggestion_count": 4},
        )

        restored = AlgorithmEvent.from_dict(event.to_dict())

        self.assertEqual(restored, event)

    def test_optimizer_backend_is_a_single_injected_paper_seam(self) -> None:
        class FakeBackend:
            def complete(self, request: OptimizerRequest) -> OptimizerResponse:
                return OptimizerResponse(
                    call_id=request.call_id,
                    payload={"suggestions": ["preserve stable ordering"]},
                    model_id="scripted-optimizer",
                )

        backend = FakeBackend()
        request = OptimizerRequest(
            call_id="call-1",
            stage=OptimizerStage.REFLECT_FAILURE,
            prompt="reflect on failures",
            response_schema={"type": "object"},
        )

        self.assertIsInstance(backend, OptimizerBackend)
        self.assertEqual(backend.complete(request).call_id, "call-1")

    def test_paper_edits_use_the_four_algorithm_operations(self) -> None:
        operations = {
            PaperEditOperation.APPEND: ("", "new guidance"),
            PaperEditOperation.INSERT_AFTER: ("anchor", "new guidance"),
            PaperEditOperation.REPLACE: ("old", "new"),
            PaperEditOperation.DELETE: ("obsolete", ""),
        }

        edits = [
            PaperEdit(edit_id=f"edit-{index}", operation=operation, target=target, content=content)
            for index, (operation, (target, content)) in enumerate(operations.items(), start=1)
        ]

        self.assertEqual({item.operation for item in edits}, set(PaperEditOperation))

        with self.assertRaisesRegex(ValueError, "exact PaperEditOperation"):
            PaperEdit(edit_id="string-op", operation="append", content="unsafe")

    def test_paper_state_keeps_current_and_best_skill_separate(self) -> None:
        state = PaperState(
            epoch=2,
            step=3,
            current_skill="candidate",
            current_score=SelectionScore(0.6),
            best_skill="best-so-far",
            best_score=SelectionScore(0.8),
            meta_skill="optimizer-only",
        )

        self.assertNotEqual(state.current_skill, state.best_skill)
        self.assertLess(state.current_score.value, state.best_score.value)


if __name__ == "__main__":
    unittest.main()
