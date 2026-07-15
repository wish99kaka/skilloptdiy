from __future__ import annotations

import json
import inspect
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from textskill_optimizer.paper import (
    AlgorithmEventType,
    OptimizerRequest,
    OptimizerResponse,
    OptimizerStage,
    PaperFastLoop,
    PaperProfileViolation,
    load_paper_profile,
)

from _paper_runtime import build_runtime


class GoldenFastLoopBackend:
    def __init__(self) -> None:
        self.requests: list[OptimizerRequest] = []
        self.refinement_round = {"failure": 0, "success": 0}

    def complete(self, request: OptimizerRequest) -> OptimizerResponse:
        self.requests.append(request)
        prompt = json.loads(request.prompt)
        if request.stage is OptimizerStage.REFLECT_FAILURE:
            payload = {
                "batch_size": 1,
                "failure_summary": [
                    {
                        "failure_type": "verification",
                        "count": 1,
                        "description": "An unverified result was trusted.",
                    }
                ],
                "patch": {
                    "reasoning": "Add verification.",
                    "edits": [{"op": "append", "content": "- draft failure rule"}],
                },
            }
        elif request.stage is OptimizerStage.REFLECT_SUCCESS:
            payload = {
                "batch_size": 1,
                "success_patterns": ["verification"],
                "patch": {
                    "reasoning": "Preserve verification.",
                    "edits": [{"op": "append", "content": "- draft success rule"}],
                },
            }
        elif request.stage is OptimizerStage.REFINE:
            source = prompt["source_type"]
            self.refinement_round[source] += 1
            round_number = self.refinement_round[source]
            payload = {
                "reasoning": f"semantic refinement {round_number}",
                "edits": [
                    {
                        "op": "append",
                        "content": f"- {source} refined {round_number}",
                    }
                ],
                "converged": round_number == 3,
            }
        elif request.stage is OptimizerStage.MERGE_FAILURE:
            payload = {
                "reasoning": "Consolidate failure evidence.",
                "edits": [
                    {
                        "op": "append",
                        "content": "- accepted rule",
                        "support_count": 3,
                        "source_type": "failure",
                    }
                ],
            }
        elif request.stage is OptimizerStage.MERGE_SUCCESS:
            payload = {
                "reasoning": "Consolidate success evidence.",
                "edits": [
                    {
                        "op": "append",
                        "content": "- success rule",
                        "support_count": 2,
                        "source_type": "success",
                    }
                ],
            }
        elif request.stage is OptimizerStage.MERGE_FINAL_FAILURE_PRIORITIZED:
            payload = {
                "reasoning": "Keep failure first.",
                "edits": [
                    {
                        "op": "append",
                        "content": "- accepted rule",
                        "support_count": 3,
                        "source_type": "failure",
                    },
                    {
                        "op": "append",
                        "content": "- success rule",
                        "support_count": 2,
                        "source_type": "success",
                    },
                ],
            }
        elif request.stage is OptimizerStage.RANK_TOP_L:
            payload = {"reasoning": "Failure correction wins.", "selected_indices": [0]}
        else:
            raise AssertionError(f"unexpected stage {request.stage}")
        return OptimizerResponse(
            call_id=request.call_id,
            payload=payload,
            model_id="golden-fake",
        )


class EmptyFastLoopBackend:
    def __init__(self) -> None:
        self.requests: list[OptimizerRequest] = []

    def complete(self, request: OptimizerRequest) -> OptimizerResponse:
        self.requests.append(request)
        prompt = json.loads(request.prompt)
        if request.stage is OptimizerStage.REFLECT_FAILURE:
            payload = {
                "batch_size": len(prompt["trajectories"]),
                "failure_summary": [],
                "patch": {"reasoning": "none", "edits": []},
            }
        elif request.stage is OptimizerStage.REFLECT_SUCCESS:
            payload = {
                "batch_size": len(prompt["trajectories"]),
                "success_patterns": [],
                "patch": {"reasoning": "none", "edits": []},
            }
        elif request.stage is OptimizerStage.REFINE:
            payload = {"reasoning": "converged", "edits": [], "converged": True}
        elif request.stage in {
            OptimizerStage.MERGE_FAILURE,
            OptimizerStage.MERGE_SUCCESS,
            OptimizerStage.MERGE_FINAL_FAILURE_PRIORITIZED,
        }:
            payload = {"reasoning": "none", "edits": []}
        elif request.stage is OptimizerStage.RANK_TOP_L:
            payload = {"reasoning": "nothing to rank", "selected_indices": []}
        else:
            raise AssertionError(request.stage)
        return OptimizerResponse(
            call_id=request.call_id,
            payload=payload,
            model_id="empty-fake",
        )


class SemanticFailureBackend(EmptyFastLoopBackend):
    def __init__(self, failed_stage: OptimizerStage) -> None:
        super().__init__()
        self.failed_stage = failed_stage

    def complete(self, request: OptimizerRequest) -> OptimizerResponse:
        if request.stage is self.failed_stage:
            self.requests.append(request)
            return OptimizerResponse(
                call_id=request.call_id,
                payload={
                    "reasoning": "invalid semantic fallback",
                    "edits": [],
                    "local_fallback": True,
                },
                model_id="invalid-fake",
            )
        return super().complete(request)


class PaperFastLoopTests(unittest.TestCase):
    def test_untrusted_callers_cannot_inject_selection_cache(self) -> None:
        parameters = inspect.signature(PaperFastLoop).parameters
        run_parameters = inspect.signature(PaperFastLoop.run_step).parameters

        self.assertNotIn("initial_score_cache", parameters)
        self.assertNotIn("reflection_minibatch_size", parameters)
        self.assertNotIn("max_refinement_rounds", parameters)
        self.assertNotIn("state", run_parameters)

    def test_fast_loop_revalidates_the_frozen_paper_profile(self) -> None:
        backend = EmptyFastLoopBackend()
        with tempfile.TemporaryDirectory() as tmp:
            controller, _ = build_runtime(Path(tmp), backend)
            forged = replace(
                load_paper_profile(),
                reflection_minibatch_size=1,
            )

            with self.assertRaisesRegex(PaperProfileViolation, "not compliant"):
                PaperFastLoop(controller, profile=forged)

    def test_fake_backend_trace_matches_algorithm_one_and_replays_exactly(self) -> None:
        backend = GoldenFastLoopBackend()
        with tempfile.TemporaryDirectory() as tmp:
            controller, train = build_runtime(Path(tmp), backend)
            loop = PaperFastLoop(
                controller,
                profile=load_paper_profile(),
            )
            initialized = loop.initialize("# Skill\n")
            evidence = train.collect(initialized.current_skill)
            result = loop.run_step(
                train_evidence=evidence,
                edit_budget=2,
            )

        self.assertEqual(
            [request.stage for request in backend.requests],
            [
                OptimizerStage.REFLECT_FAILURE,
                OptimizerStage.REFINE,
                OptimizerStage.REFINE,
                OptimizerStage.REFINE,
                OptimizerStage.REFLECT_SUCCESS,
                OptimizerStage.REFINE,
                OptimizerStage.REFINE,
                OptimizerStage.REFINE,
                OptimizerStage.MERGE_FAILURE,
                OptimizerStage.MERGE_SUCCESS,
                OptimizerStage.MERGE_FINAL_FAILURE_PRIORITIZED,
                OptimizerStage.RANK_TOP_L,
            ],
        )
        self.assertEqual(
            [event.event_type for event in result.events],
            [
                AlgorithmEventType.STEP_STARTED,
                AlgorithmEventType.ROLLOUT_COLLECTED,
                AlgorithmEventType.FAILURE_REFLECTED,
                AlgorithmEventType.ANALYST_REFINED,
                AlgorithmEventType.ANALYST_REFINED,
                AlgorithmEventType.ANALYST_REFINED,
                AlgorithmEventType.SUCCESS_REFLECTED,
                AlgorithmEventType.ANALYST_REFINED,
                AlgorithmEventType.ANALYST_REFINED,
                AlgorithmEventType.ANALYST_REFINED,
                AlgorithmEventType.MERGE_FAILURE,
                AlgorithmEventType.MERGE_SUCCESS,
                AlgorithmEventType.MERGE_FINAL_FAILURE_PRIORITIZED,
                AlgorithmEventType.RANK_TOP_L,
                AlgorithmEventType.PATCH_APPLIED,
                AlgorithmEventType.SELECTION_SCORED,
                AlgorithmEventType.CANDIDATE_ACCEPTED,
            ],
        )
        self.assertEqual(result.state.current_score.value, 0.8)
        self.assertEqual(result.state.best_skill, result.state.current_skill)
        self.assertEqual(result.state.best_score.value, 0.8)
        self.assertEqual(loop.state, result.state)
        self.assertIn("- accepted rule", result.state.current_skill)
        self.assertEqual(result.replay(), result.apply_result)
        self.assertEqual(result.ranked_edits[0].support_count, 3)
        self.assertEqual(result.ranked_edits[0].source_type, "failure")
        refinement_prompts = [
            json.loads(request.prompt)["prior_patch"]["edits"]
            for request in backend.requests
            if request.stage is OptimizerStage.REFINE
        ]
        self.assertEqual(
            [items[0]["content"] for items in refinement_prompts],
            [
                "- draft failure rule",
                "- failure refined 1",
                "- failure refined 2",
                "- draft success rule",
                "- success refined 1",
                "- success refined 2",
            ],
        )
        self.assertTrue(
            all(
                request.metadata["data_sources"] == ["train"]
                for request in backend.requests
            )
        )

    def test_unchanged_candidate_uses_hash_cache_and_rejects_tie(self) -> None:
        backend = EmptyFastLoopBackend()
        with tempfile.TemporaryDirectory() as tmp:
            controller, train = build_runtime(
                Path(tmp),
                backend,
                invalid_selection_after_first=True,
            )
            loop = PaperFastLoop(
                controller,
                profile=load_paper_profile(),
            )
            state = loop.initialize("# Skill\n")
            evidence = train.collect(state.current_skill)
            result = loop.run_step(
                train_evidence=evidence,
                edit_budget=2,
            )

        self.assertEqual(result.state.current_skill, state.current_skill)
        self.assertEqual(
            result.events[-1].event_type,
            AlgorithmEventType.CANDIDATE_REJECTED,
        )
        selection_event = next(
            event
            for event in result.events
            if event.event_type is AlgorithmEventType.SELECTION_SCORED
        )
        self.assertTrue(selection_event.payload["cache_hit"])
        self.assertEqual(selection_event.payload["candidate_score"], 0.5)
        self.assertEqual(backend.requests[-1].stage, OptimizerStage.RANK_TOP_L)

    def test_semantic_merge_or_rank_failure_retries_then_skips_unchanged(self) -> None:
        for failed_stage in (
            OptimizerStage.MERGE_FAILURE,
            OptimizerStage.RANK_TOP_L,
        ):
            with self.subTest(failed_stage=failed_stage):
                backend = SemanticFailureBackend(failed_stage)
                with tempfile.TemporaryDirectory() as tmp:
                    controller, train = build_runtime(
                        Path(tmp),
                        backend,
                        invalid_selection_after_first=True,
                    )
                    loop = PaperFastLoop(
                        controller,
                        profile=load_paper_profile(),
                    )
                    state = loop.initialize("# Skill\n")
                    evidence = train.collect(state.current_skill)
                    result = loop.run_step(
                        train_evidence=evidence,
                        edit_budget=2,
                    )

                attempts = [
                    request
                    for request in backend.requests
                    if request.stage is failed_stage
                ]
                self.assertEqual(len(attempts), 2)
                self.assertEqual(result.state.current_skill, state.current_skill)
                self.assertEqual(result.state.step, 1)
                self.assertEqual(loop.state, result.state)
                self.assertEqual(result.skipped_stage, failed_stage)
                self.assertEqual(
                    result.events[-1].event_type,
                    AlgorithmEventType.CANDIDATE_REJECTED,
                )
                self.assertEqual(
                    result.events[-1].payload["reason"],
                    "semantic_stage_exhausted",
                )
                self.assertEqual(result.events[-1].payload["attempts"], 2)
                self.assertEqual(
                    [request.metadata["semantic_attempt"] for request in attempts],
                    [1, 2],
                )
                self.assertTrue(
                    all(
                        request.metadata["retry_policy_id"]
                        == "semantic-retry-once-v1"
                        for request in attempts
                    )
                )
                self.assertFalse(
                    any(
                        event.event_type
                        in {
                            AlgorithmEventType.PATCH_APPLIED,
                            AlgorithmEventType.SELECTION_SCORED,
                        }
                        for event in result.events
                    )
                )

    def test_failure_proposals_merge_in_stable_profile_sized_levels(self) -> None:
        backend = EmptyFastLoopBackend()
        with tempfile.TemporaryDirectory() as tmp:
            controller, train = build_runtime(
                Path(tmp),
                backend,
                invalid_selection_after_first=True,
                failure_count=65,
                success_count=1,
            )
            loop = PaperFastLoop(
                controller,
                profile=load_paper_profile(),
            )
            state = loop.initialize("# Skill\n")
            evidence = train.collect(state.current_skill)
            loop.run_step(
                train_evidence=evidence,
                edit_budget=2,
            )

        failure_merge_requests = [
            request
            for request in backend.requests
            if request.stage is OptimizerStage.MERGE_FAILURE
        ]
        self.assertEqual(len(failure_merge_requests), 3)
        self.assertEqual(
            [
                len(json.loads(request.prompt)["patches"])
                for request in failure_merge_requests
            ],
            [8, 1, 2],
        )

    def test_run_step_requires_selection_owned_initialization(self) -> None:
        backend = EmptyFastLoopBackend()
        with tempfile.TemporaryDirectory() as tmp:
            controller, train = build_runtime(Path(tmp), backend)
            evidence = train.collect("# Forged\n")
            loop = PaperFastLoop(
                controller,
                profile=load_paper_profile(),
            )

            with self.assertRaisesRegex(ValueError, "must be initialized"):
                loop.run_step(
                    train_evidence=evidence,
                    edit_budget=2,
                )


if __name__ == "__main__":
    unittest.main()
