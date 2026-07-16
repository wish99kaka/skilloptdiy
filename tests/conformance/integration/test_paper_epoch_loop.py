from __future__ import annotations

import inspect
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from textskill_optimizer.paper import (
    AlgorithmEventType,
    CheckpointAuthenticator,
    ControllerRole,
    DataFirewallViolation,
    OptimizerRequest,
    OptimizerResponse,
    OptimizerStage,
    PaperEpochLoop,
    PaperEpochPlan,
    load_paper_profile,
)
from textskill_optimizer.paper.responses import OptimizerContractViolation

from _paper_runtime import build_runtime


class EmptyEpochBackend:
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
            payload = {"reasoning": "done", "edits": [], "converged": True}
        elif request.stage in {
            OptimizerStage.MERGE_FAILURE,
            OptimizerStage.MERGE_SUCCESS,
            OptimizerStage.MERGE_FINAL_FAILURE_PRIORITIZED,
        }:
            payload = {"reasoning": "none", "edits": []}
        elif request.stage is OptimizerStage.RANK_TOP_L:
            payload = {"reasoning": "none", "selected_indices": []}
        else:
            raise AssertionError(request.stage)
        return OptimizerResponse(
            call_id=request.call_id,
            payload=payload,
            model_id="epoch-empty-fake",
        )


class RejectedBufferBackend(EmptyEpochBackend):
    def complete(self, request: OptimizerRequest) -> OptimizerResponse:
        self.requests.append(request)
        prompt = json.loads(request.prompt)
        if request.stage is OptimizerStage.REFLECT_FAILURE:
            payload = {
                "batch_size": len(prompt["trajectories"]),
                "failure_summary": [
                    {
                        "failure_type": "verification",
                        "count": 1,
                        "description": "trusted an unverified result",
                    }
                ],
                "patch": {"reasoning": "fix it", "edits": []},
            }
        elif request.stage is OptimizerStage.REFLECT_SUCCESS:
            payload = {
                "batch_size": len(prompt["trajectories"]),
                "success_patterns": [],
                "patch": {"reasoning": "none", "edits": []},
            }
        elif request.stage is OptimizerStage.REFINE:
            payload = {"reasoning": "done", "edits": [], "converged": True}
        elif request.stage is OptimizerStage.MERGE_FAILURE:
            payload = {
                "reasoning": "try a rule",
                "edits": [
                    {
                        "op": "append",
                        "content": "- rejected rule",
                        "support_count": 1,
                        "source_type": "failure",
                    }
                ],
            }
        elif request.stage is OptimizerStage.MERGE_SUCCESS:
            payload = {"reasoning": "none", "edits": []}
        elif request.stage is OptimizerStage.MERGE_FINAL_FAILURE_PRIORITIZED:
            payload = {
                "reasoning": "try a rule",
                "edits": [
                    {
                        "op": "append",
                        "content": "- rejected rule",
                        "support_count": 1,
                        "source_type": "failure",
                    }
                ],
            }
        elif request.stage is OptimizerStage.RANK_TOP_L:
            payload = {"reasoning": "try it", "selected_indices": [0]}
        else:
            raise AssertionError(request.stage)
        return OptimizerResponse(
            call_id=request.call_id,
            payload=payload,
            model_id="epoch-buffer-fake",
        )


class SlowMetaBackend(RejectedBufferBackend):
    META_TOKEN = "OPTIMIZER_ONLY_META_TOKEN"

    def complete(self, request: OptimizerRequest) -> OptimizerResponse:
        if request.stage is OptimizerStage.PROPOSE_SLOW_UPDATE:
            self.requests.append(request)
            return OptimizerResponse(
                call_id=request.call_id,
                payload={
                    "reasoning": "consolidate longitudinal behavior",
                    "slow_update_content": "- durable but tied guidance",
                },
                model_id="slow-meta-fake",
            )
        if request.stage is OptimizerStage.UPDATE_META_SKILL:
            self.requests.append(request)
            return OptimizerResponse(
                call_id=request.call_id,
                payload={
                    "reasoning": "remember optimizer behavior",
                    "meta_skill_content": self.META_TOKEN,
                },
                model_id="slow-meta-fake",
            )
        if request.stage in {
            OptimizerStage.MERGE_FAILURE,
            OptimizerStage.MERGE_FINAL_FAILURE_PRIORITIZED,
        } and request.call_id.startswith(("e2-", "e3-")):
            self.requests.append(request)
            payload = {
                "reasoning": "accept one fast rule",
                "edits": [
                    {
                        "op": "append",
                        "content": "- accepted rule",
                        "support_count": 1,
                        "source_type": "failure",
                    }
                ],
            }
            return OptimizerResponse(
                call_id=request.call_id,
                payload=payload,
                model_id="slow-meta-fake",
            )
        return super().complete(request)


class InvalidMetaBackend(SlowMetaBackend):
    def complete(self, request: OptimizerRequest) -> OptimizerResponse:
        if request.stage is OptimizerStage.UPDATE_META_SKILL:
            self.requests.append(request)
            return OptimizerResponse(
                call_id=request.call_id,
                payload={
                    "reasoning": "invalid",
                    "meta_skill_content": self.META_TOKEN,
                    "selection_diagnostics": {},
                },
                model_id="invalid-meta-fake",
            )
        return super().complete(request)


class PaperEpochLoopTests(unittest.TestCase):
    def _build_loop(
        self,
        root: Path,
        backend,
        *,
        steps_per_epoch: int = 2,
        longitudinal_fixture: bool = False,
        slow_selection_accept: bool = False,
        truncate_scheduled_batch: bool = False,
    ):
        profile = load_paper_profile()
        controller, train = build_runtime(
            root,
            backend,
            failure_count=20,
            success_count=20,
            longitudinal_fixture=longitudinal_fixture,
            slow_selection_accept=slow_selection_accept,
            truncate_scheduled_batch=truncate_scheduled_batch,
        )
        registration = train.registry.require(
            train.controller_id,
            role=ControllerRole.TRAIN,
        )
        plan = PaperEpochPlan.build(
            profile=profile,
            train_split_id=registration.split_id,
            train_split_manifest_sha256=registration.artifact(
                "split_manifest"
            ).sha256,
            steps_per_epoch=steps_per_epoch,
        )
        return (
            PaperEpochLoop(controller, profile=profile, plan=plan),
            train,
            plan,
            controller,
        )

    def test_frozen_plan_owns_scheduler_and_lifecycle_sequence(self) -> None:
        self.assertNotIn(
            "edit_budget",
            inspect.signature(PaperEpochLoop.run_step).parameters,
        )
        backend = EmptyEpochBackend()
        with tempfile.TemporaryDirectory() as tmp:
            loop, train, plan, _ = self._build_loop(Path(tmp), backend)
            initialized = loop.initialize("# Skill\n")

            first_evidence = loop.collect_train_evidence()
            first = loop.run_step(train_evidence=first_evidence)
            second_evidence = loop.collect_train_evidence()
            second = loop.run_step(train_evidence=second_evidence)

            with self.assertRaisesRegex(ValueError, "epoch 1 is complete"):
                loop.run_step(train_evidence=second_evidence)

        self.assertEqual(first.cursor, plan.cursor(epoch=1, step=1))
        self.assertEqual(second.cursor, plan.cursor(epoch=1, step=2))
        self.assertEqual(
            [event.event_type for event in loop.events[:2]],
            [AlgorithmEventType.RUN_STARTED, AlgorithmEventType.EPOCH_STARTED],
        )
        signed_request = json.loads(first_evidence.canonical_request)
        signed_payload = json.loads(first_evidence.canonical_payload)
        self.assertEqual(signed_request["batch_seed"], first.cursor.batch_seed)
        self.assertEqual(signed_request["batch_size"], 40)
        self.assertEqual(signed_payload["batch_seed"], first.cursor.batch_seed)
        self.assertEqual(len(signed_payload["trajectories"]), 40)
        self.assertEqual(
            [event.sequence for event in loop.events],
            list(range(len(loop.events))),
        )
        step_events = [
            event
            for event in loop.events
            if event.event_type is AlgorithmEventType.STEP_STARTED
        ]
        self.assertEqual(
            [event.payload["edit_budget"] for event in step_events],
            [4, 4],
        )
        self.assertEqual(
            [event.payload["train_batch_id"] for event in step_events],
            [
                plan.cursor(epoch=1, step=1).batch_id,
                plan.cursor(epoch=1, step=2).batch_id,
            ],
        )

    def test_plan_cannot_reuse_profile_hash_with_drifted_copied_fields(self) -> None:
        backend = EmptyEpochBackend()
        with tempfile.TemporaryDirectory() as tmp:
            _, _, plan, controller = self._build_loop(Path(tmp), backend)
            forged = replace(plan, rollout_batch_size=1)

            with self.assertRaisesRegex(ValueError, "do not match frozen profile"):
                PaperEpochLoop(
                    controller,
                    profile=load_paper_profile(),
                    plan=forged,
                )

        self.assertEqual(plan.rollout_batch_size, 40)

    def test_rejected_buffer_is_derived_and_visible_only_to_later_steps(self) -> None:
        backend = RejectedBufferBackend()
        with tempfile.TemporaryDirectory() as tmp:
            loop, train, _, _ = self._build_loop(Path(tmp), backend)
            initialized = loop.initialize("# Skill\n")
            first = loop.run_step(train_evidence=loop.collect_train_evidence())
            first_request_count = len(backend.requests)
            second = loop.run_step(train_evidence=loop.collect_train_evidence())

        self.assertEqual(first.state.current_skill, initialized.current_skill)
        self.assertEqual(second.state.current_skill, initialized.current_skill)
        self.assertEqual(len(loop.epoch_buffer), 2)
        first_record = loop.epoch_buffer[0].to_optimizer_payload()
        self.assertEqual(first_record["failure_patterns"][0]["failure_type"], "verification")
        self.assertEqual(first_record["rejected_edits"][0]["content"], "- rejected rule")
        self.assertEqual(first_record["score_delta"], 0.0)
        self.assertTrue(
            all(
                json.loads(request.prompt)["epoch_buffer"] == []
                for request in backend.requests[:first_request_count]
            )
        )
        self.assertTrue(
            all(
                len(json.loads(request.prompt)["epoch_buffer"]) == 1
                for request in backend.requests[first_request_count:]
            )
        )

    def test_step_rejects_signed_train_evidence_not_bound_to_planned_batch(self) -> None:
        backend = EmptyEpochBackend()
        with tempfile.TemporaryDirectory() as tmp:
            loop, train, _, _ = self._build_loop(Path(tmp), backend)
            state = loop.initialize("# Skill\n")
            unbound = train.collect(state.current_skill)

            with self.assertRaisesRegex(DataFirewallViolation, "scheduled batch"):
                loop.run_step(train_evidence=unbound)

            result = loop.run_step(
                train_evidence=loop.collect_train_evidence()
            )

        self.assertEqual(result.cursor.step, 1)

    def test_signed_train_response_must_match_planned_trajectory_count(self) -> None:
        backend = EmptyEpochBackend()
        with tempfile.TemporaryDirectory() as tmp:
            loop, _, _, _ = self._build_loop(
                Path(tmp),
                backend,
                truncate_scheduled_batch=True,
            )
            loop.initialize("# Skill\n")

            with self.assertRaisesRegex(DataFirewallViolation, "trajectory count"):
                loop.collect_train_evidence()

    def test_epoch_one_skips_slow_meta_and_clears_buffer_before_epoch_two(self) -> None:
        backend = RejectedBufferBackend()
        with tempfile.TemporaryDirectory() as tmp:
            loop, train, _, _ = self._build_loop(Path(tmp), backend)
            state = loop.initialize("# Skill\n")
            for _ in range(2):
                result = loop.run_step(
                    train_evidence=loop.collect_train_evidence()
                )
                state = result.state
            self.assertEqual(len(loop.epoch_buffer), 2)

            completion = loop.finish_epoch()
            epoch_two_request_start = len(backend.requests)
            epoch_two = loop.run_step(
                train_evidence=loop.collect_train_evidence()
            )

        self.assertEqual(completion.completed_epoch, 1)
        self.assertEqual(completion.state.epoch, 2)
        self.assertEqual(completion.state.step, 0)
        self.assertEqual(loop.epoch_buffer[0].epoch, 2)
        self.assertEqual(
            [event.event_type for event in completion.events],
            [
                AlgorithmEventType.SLOW_UPDATE_SKIPPED,
                AlgorithmEventType.META_UPDATE_SKIPPED,
                AlgorithmEventType.EPOCH_COMPLETED,
                AlgorithmEventType.EPOCH_STARTED,
            ],
        )
        self.assertTrue(
            all(
                json.loads(request.prompt)["epoch_buffer"] == []
                for request in backend.requests[epoch_two_request_start:]
            )
        )
        self.assertEqual(epoch_two.cursor.epoch, 2)

    def test_authenticated_resume_matches_uninterrupted_run_exactly(self) -> None:
        backend = RejectedBufferBackend()
        authenticator = CheckpointAuthenticator(
            key_id="test-checkpoint-key",
            secret_key=b"k" * 32,
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            continuous, train, plan, controller = self._build_loop(root, backend)
            state = continuous.initialize("# Skill\n")
            for _ in range(2):
                state = continuous.run_step(
                    train_evidence=continuous.collect_train_evidence()
                ).state
            state = continuous.finish_epoch().state
            continuous.run_step(
                train_evidence=continuous.collect_train_evidence()
            )
            expected_requests = [request.call_id for request in backend.requests]

            backend.requests.clear()
            crashed = PaperEpochLoop(
                controller,
                profile=load_paper_profile(),
                plan=plan,
            )
            state = crashed.initialize("# Skill\n")
            state = crashed.run_step(
                train_evidence=crashed.collect_train_evidence()
            ).state
            checkpoint = crashed.checkpoint(authenticator)
            resumed = PaperEpochLoop.resume(
                controller,
                profile=load_paper_profile(),
                plan=plan,
                checkpoint=checkpoint,
                authenticator=authenticator,
            )
            state = resumed.run_step(
                train_evidence=resumed.collect_train_evidence()
            ).state
            state = resumed.finish_epoch().state
            resumed.run_step(
                train_evidence=resumed.collect_train_evidence()
            )

            tampered = checkpoint.to_dict()
            tampered["payload"]["fast_loop"]["next_event_sequence"] += 1
            with self.assertRaisesRegex(ValueError, "authentication failed"):
                PaperEpochLoop.resume(
                    controller,
                    profile=load_paper_profile(),
                    plan=plan,
                    checkpoint=type(checkpoint).from_mapping(tampered),
                    authenticator=authenticator,
                )
            forged_payload = checkpoint.to_dict()["payload"]
            forged_payload["fast_loop"]["state"]["step"] = (
                plan.steps_per_epoch + 1
            )
            with self.assertRaisesRegex(ValueError, "outside frozen plan"):
                PaperEpochLoop.resume(
                    controller,
                    profile=load_paper_profile(),
                    plan=plan,
                    checkpoint=authenticator.sign(forged_payload),
                    authenticator=authenticator,
                )

        self.assertEqual(resumed.state, continuous.state)
        self.assertEqual(resumed.score_cache, continuous.score_cache)
        self.assertEqual(resumed.epoch_buffer, continuous.epoch_buffer)
        self.assertEqual(resumed.epoch_snapshots, continuous.epoch_snapshots)
        self.assertEqual(resumed.events, continuous.events)
        self.assertEqual(
            [request.call_id for request in backend.requests],
            expected_requests,
        )

    def test_resume_after_slow_meta_matches_continuing_epoch_three(self) -> None:
        backend = SlowMetaBackend()
        authenticator = CheckpointAuthenticator(
            key_id="post-slow-checkpoint-key",
            secret_key=b"s" * 32,
        )
        with tempfile.TemporaryDirectory() as tmp:
            loop, _, plan, controller = self._build_loop(
                Path(tmp),
                backend,
                steps_per_epoch=1,
                longitudinal_fixture=True,
            )
            loop.initialize("# Skill\n")
            loop.run_step(train_evidence=loop.collect_train_evidence())
            loop.finish_epoch()
            loop.run_step(train_evidence=loop.collect_train_evidence())
            loop.finish_epoch(
                longitudinal_evidence=loop.collect_longitudinal_evidence()
            )
            checkpoint = loop.checkpoint(authenticator)

            backend.requests.clear()
            loop.run_step(train_evidence=loop.collect_train_evidence())
            expected_requests = [request.call_id for request in backend.requests]
            expected = (
                loop.state,
                loop.score_cache,
                loop.epoch_buffer,
                loop.epoch_snapshots,
                loop.events,
            )

            backend.requests.clear()
            resumed = PaperEpochLoop.resume(
                controller,
                profile=load_paper_profile(),
                plan=plan,
                checkpoint=checkpoint,
                authenticator=authenticator,
            )
            resumed.run_step(train_evidence=resumed.collect_train_evidence())

        self.assertEqual(
            (
                resumed.state,
                resumed.score_cache,
                resumed.epoch_buffer,
                resumed.epoch_snapshots,
                resumed.events,
            ),
            expected,
        )
        self.assertEqual(
            [request.call_id for request in backend.requests],
            expected_requests,
        )
        self.assertEqual(resumed.state.meta_skill, SlowMetaBackend.META_TOKEN)

    def test_epoch_two_slow_is_strictly_gated_and_meta_is_optimizer_only(self) -> None:
        backend = SlowMetaBackend()
        with tempfile.TemporaryDirectory() as tmp:
            loop, train, _, _ = self._build_loop(
                Path(tmp),
                backend,
                steps_per_epoch=1,
                longitudinal_fixture=True,
            )
            state = loop.initialize("# Skill\n")
            state = loop.run_step(
                train_evidence=loop.collect_train_evidence()
            ).state
            state = loop.finish_epoch().state
            state = loop.run_step(
                train_evidence=loop.collect_train_evidence()
            ).state
            completion = loop.finish_epoch(
                longitudinal_evidence=loop.collect_longitudinal_evidence()
            )
            epoch_three_request_start = len(backend.requests)
            loop.run_step(
                train_evidence=loop.collect_train_evidence()
            )

        slow_request = next(
            request
            for request in backend.requests
            if request.stage is OptimizerStage.PROPOSE_SLOW_UPDATE
        )
        longitudinal = json.loads(slow_request.prompt)["longitudinal"]
        self.assertEqual(
            {name: len(items) for name, items in longitudinal.items()},
            {
                "improvements": 5,
                "regressions": 5,
                "persistent_failures": 5,
                "stable_successes": 5,
            },
        )
        self.assertIn(
            AlgorithmEventType.CANDIDATE_REJECTED,
            [event.event_type for event in completion.events],
        )
        self.assertEqual(completion.state.current_score.value, 0.8)
        self.assertNotIn(SlowMetaBackend.META_TOKEN, completion.state.current_skill)
        self.assertNotIn(SlowMetaBackend.META_TOKEN, completion.state.best_skill)
        self.assertEqual(completion.state.meta_skill, SlowMetaBackend.META_TOKEN)
        future_fast_requests = [
            request
            for request in backend.requests[epoch_three_request_start:]
            if request.stage
            not in {
                OptimizerStage.PROPOSE_SLOW_UPDATE,
                OptimizerStage.UPDATE_META_SKILL,
            }
        ]
        self.assertTrue(future_fast_requests)
        self.assertTrue(
            all(
                json.loads(request.prompt)["meta_skill"]
                == SlowMetaBackend.META_TOKEN
                for request in future_fast_requests
            )
        )

    def test_invalid_meta_response_cannot_partially_commit_slow_gate(self) -> None:
        backend = InvalidMetaBackend()
        with tempfile.TemporaryDirectory() as tmp:
            loop, train, _, _ = self._build_loop(
                Path(tmp),
                backend,
                steps_per_epoch=1,
                longitudinal_fixture=True,
            )
            state = loop.initialize("# Skill\n")
            state = loop.run_step(
                train_evidence=loop.collect_train_evidence()
            ).state
            state = loop.finish_epoch().state
            state = loop.run_step(
                train_evidence=loop.collect_train_evidence()
            ).state
            before = (
                loop.state,
                loop.score_cache,
                loop.events,
                loop.epoch_snapshots,
            )
            with self.assertRaises(OptimizerContractViolation):
                loop.finish_epoch(
                    longitudinal_evidence=loop.collect_longitudinal_evidence()
                )

        self.assertEqual(
            (
                loop.state,
                loop.score_cache,
                loop.events,
                loop.epoch_snapshots,
            ),
            before,
        )

    def test_accepted_slow_update_does_not_overwrite_fast_epoch_snapshot(self) -> None:
        backend = SlowMetaBackend()
        with tempfile.TemporaryDirectory() as tmp:
            loop, _, _, _ = self._build_loop(
                Path(tmp),
                backend,
                steps_per_epoch=1,
                longitudinal_fixture=True,
                slow_selection_accept=True,
            )
            state = loop.initialize("# Skill\n")
            state = loop.run_step(
                train_evidence=loop.collect_train_evidence()
            ).state
            state = loop.finish_epoch().state
            fast_epoch_two = loop.run_step(
                train_evidence=loop.collect_train_evidence()
            ).state.current_skill
            epoch_two = loop.finish_epoch(
                longitudinal_evidence=loop.collect_longitudinal_evidence()
            )
            state = loop.run_step(
                train_evidence=loop.collect_train_evidence()
            ).state
            loop.finish_epoch(
                longitudinal_evidence=loop.collect_longitudinal_evidence()
            )

        self.assertEqual(loop.epoch_snapshots[1], fast_epoch_two)
        self.assertNotIn("durable but tied guidance", loop.epoch_snapshots[1])
        self.assertIn("durable but tied guidance", epoch_two.state.current_skill)
        self.assertIn(
            AlgorithmEventType.CANDIDATE_ACCEPTED,
            [event.event_type for event in epoch_two.events],
        )
        epoch_three_slow = next(
            request
            for request in backend.requests
            if request.call_id == "e3-slow-update"
        )
        prompt = json.loads(epoch_three_slow.prompt)
        self.assertNotIn(
            "durable but tied guidance",
            prompt["previous_epoch_skill"],
        )
        self.assertIn(
            "durable but tied guidance",
            prompt["current_epoch_skill"],
        )
        self.assertEqual(
            prompt["previous_slow_update"],
            "- durable but tied guidance",
        )

    def test_final_epoch_closes_run_once_and_checkpoint_preserves_closure(self) -> None:
        backend = SlowMetaBackend()
        authenticator = CheckpointAuthenticator(
            key_id="completed-run-key",
            secret_key=b"c" * 32,
        )
        with tempfile.TemporaryDirectory() as tmp:
            loop, train, plan, controller = self._build_loop(
                Path(tmp),
                backend,
                steps_per_epoch=1,
                longitudinal_fixture=True,
            )
            state = loop.initialize("# Skill\n")
            completion = None
            for epoch in range(1, plan.epochs + 1):
                last_step_evidence = loop.collect_train_evidence()
                state = loop.run_step(train_evidence=last_step_evidence).state
                if epoch == 1:
                    completion = loop.finish_epoch()
                else:
                    completion = loop.finish_epoch(
                        longitudinal_evidence=loop.collect_longitudinal_evidence()
                    )
                state = completion.state
            assert completion is not None
            checkpoint = loop.checkpoint(authenticator)
            resumed = PaperEpochLoop.resume(
                controller,
                profile=load_paper_profile(),
                plan=plan,
                checkpoint=checkpoint,
                authenticator=authenticator,
            )
            with self.assertRaisesRegex(ValueError, "run is complete"):
                resumed.finish_epoch()
            with self.assertRaisesRegex(ValueError, "run is complete"):
                resumed.run_step(
                    train_evidence=last_step_evidence
                )

        self.assertTrue(completion.run_completed)
        self.assertEqual(completion.completed_epoch, plan.epochs)
        self.assertEqual(loop.events[-1].event_type, AlgorithmEventType.RUN_COMPLETED)
        self.assertEqual(len(loop.epoch_snapshots), plan.epochs)
        self.assertEqual(resumed.events, loop.events)


if __name__ == "__main__":
    unittest.main()
