import json
import tempfile
import unittest
from pathlib import Path

from textskill_optimizer.edits import END_TARGET, SLOW_START
from textskill_optimizer.executive_optimizer import (
    ExecutiveOptimizerConfig,
    ExecutiveSkillOptimizer,
    PersistentTaskAnomaly,
    evidence_guided_proposal_filter_issues,
    local_proposal_penalty,
    reflection_minibatches,
    scheduled_learning_rate,
)
from textskill_optimizer.interfaces import EDITOR_CAPABILITY_ATOMIC_EDITS
from textskill_optimizer.models import AtomicEdit, EditProposal, EvaluationReport, OptimizerStateUpdate, RejectedProposal, Score, Task, TaskOutput, TaskResult
from textskill_optimizer.plugins.coding import CodingScorer, coding_retryable_anomaly_reasons


class SkillTextRunner:
    def run(self, skill_text: str, task: Task) -> TaskOutput:
        return TaskOutput(skill_text)


class ExpectedRulesScorer:
    def score(self, task: Task, output: TaskOutput) -> Score:
        expected = list(task.expected or [])
        matched = sum(1 for rule in expected if rule in str(output.value))
        value = matched / len(expected) if expected else 1.0
        return Score(value, value == 1.0)


class SequencedScoreRunner:
    def __init__(self, scores: dict[str, list[float]]) -> None:
        self.scores = {skill: list(values) for skill, values in scores.items()}
        self.calls: list[str] = []

    def run(self, skill_text: str, task: Task) -> TaskOutput:
        self.calls.append(skill_text)
        return TaskOutput(self.scores[skill_text].pop(0))


class TaskScoreRunner:
    def __init__(self, scores: dict[str, dict[str, float]]) -> None:
        self.scores = scores
        self.calls: list[tuple[str, str]] = []

    def run(self, skill_text: str, task: Task) -> TaskOutput:
        self.calls.append((skill_text, task.id))
        return TaskOutput(self.scores[skill_text][task.id])


class NumericScorer:
    def score(self, task: Task, output: TaskOutput) -> Score:
        value = float(output.value)
        return Score(value, value == 1.0)


class AtomicEditor:
    capabilities = frozenset({EDITOR_CAPABILITY_ATOMIC_EDITS})


class ExecutiveOptimizerTests(unittest.TestCase):
    def test_optimize_rejects_editor_without_atomic_capability_before_evaluation(self) -> None:
        class FailIfRun:
            def run(self, skill_text: str, task: Task) -> TaskOutput:
                raise AssertionError("evaluation must not start with an incompatible editor")

        optimizer = ExecutiveSkillOptimizer(
            FailIfRun(),
            ExpectedRulesScorer(),
            object(),
            ExecutiveOptimizerConfig(enable_slow_update=False),
        )

        with self.assertRaisesRegex(ValueError, "atomic_edits"):
            optimizer.optimize(
                "# Skill\n",
                [Task(id="train", input="", expected=[])],
                [Task(id="selection", input="", expected=[])],
            )

    def test_optimize_rejects_full_replacement_from_atomic_editor(self) -> None:
        class MisconfiguredEditor:
            capabilities = frozenset({EDITOR_CAPABILITY_ATOMIC_EDITS})

            def propose(self, skill_text, train_results, *, epoch, **kwargs):
                return [EditProposal(name="whole-document", skill_text="# Replacement\n")]

        optimizer = ExecutiveSkillOptimizer(
            SkillTextRunner(),
            ExpectedRulesScorer(),
            MisconfiguredEditor(),
            ExecutiveOptimizerConfig(
                epochs=1,
                rollout_batch_size=1,
                reflection_minibatch_size=1,
                enable_slow_update=False,
            ),
        )

        with self.assertRaisesRegex(ValueError, "returned a non-atomic proposal"):
            optimizer.optimize(
                "# Skill\n",
                [Task(id="train", input="", expected=[])],
                [Task(id="selection", input="", expected=[])],
            )

    def test_validation_gate_config_rejects_impossible_majority(self) -> None:
        with self.assertRaises(ValueError):
            ExecutiveOptimizerConfig(
                validation_confirmation_rounds=1,
                validation_required_wins=3,
            )

    def test_validation_gate_stops_after_non_improving_initial_round(self) -> None:
        runner = SequencedScoreRunner({"candidate": [0.7]})
        optimizer = ExecutiveSkillOptimizer(
            runner,
            NumericScorer(),
            object(),
            ExecutiveOptimizerConfig(
                validation_confirmation_rounds=2,
                validation_required_wins=2,
                validation_mean_delta=0.05,
            ),
        )

        decision = optimizer.validate_candidate(
            "current",
            "candidate",
            [Task(id="selection", input="")],
            candidate_name="candidate",
            current_score=0.8,
        )

        self.assertFalse(decision.accepted)
        self.assertEqual(decision.total_rounds, 1)
        self.assertEqual(runner.calls, ["candidate"])

    def test_validation_gate_records_contract_evidence_from_existing_current_report(self) -> None:
        runner = SequencedScoreRunner({"candidate": [0.0]})
        optimizer = ExecutiveSkillOptimizer(
            runner,
            NumericScorer(),
            object(),
            ExecutiveOptimizerConfig(validation_confirmation_rounds=0),
        )
        current_report = EvaluationReport(
            "current",
            [
                TaskResult(
                    task=Task(id="selection", input="", metadata={"contract_tags": ["stable_order"]}),
                    output=TaskOutput(1.0),
                    score=Score(1.0, True),
                )
            ],
        )

        with tempfile.TemporaryDirectory() as tmp:
            decision = optimizer.validate_candidate(
                "current",
                "candidate",
                [Task(id="selection", input="", metadata={"contract_tags": ["stable_order"]})],
                candidate_name="candidate",
                current_report=current_report,
                run_path=Path(tmp),
            )
            gate = json.loads((Path(tmp) / "selection_candidate_gate.json").read_text(encoding="utf-8"))

        self.assertEqual(runner.calls, ["candidate"])
        self.assertFalse(decision.accepted)
        self.assertEqual(
            decision.contract_evidence["contract_deltas"]["stable_order"]["delta"],
            -1.0,
        )
        self.assertEqual(
            gate["contract_evidence"]["top_negative_contracts"][0]["contract"],
            "stable_order",
        )

    def test_validation_gate_blocks_evidence_guided_protected_regression(self) -> None:
        tasks = [
            Task(id="target", input="", metadata={"contract_tags": ["target_contract"]}),
            Task(id="neutral", input="", metadata={"contract_tags": ["neutral_contract"]}),
            Task(id="protected", input="", metadata={"contract_tags": ["protected_contract"]}),
        ]
        current_scores = {"target": 0.0, "neutral": 0.0, "protected": 1.0}
        current_report = EvaluationReport(
            "current",
            [
                TaskResult(
                    task=task,
                    output=TaskOutput(current_scores[task.id]),
                    score=Score(current_scores[task.id], current_scores[task.id] == 1.0),
                )
                for task in tasks
            ],
        )
        runner = TaskScoreRunner(
            {
                "candidate": {
                    "target": 1.0,
                    "neutral": 1.0,
                    "protected": 0.0,
                }
            }
        )
        optimizer = ExecutiveSkillOptimizer(
            runner,
            NumericScorer(),
            object(),
            ExecutiveOptimizerConfig(validation_confirmation_rounds=0),
        )

        decision = optimizer.validate_candidate(
            "current",
            "candidate",
            tasks,
            candidate_name="candidate",
            current_report=current_report,
            candidate_contract_policy={
                "required": True,
                "evidence_sources": ["contract_rejection_evidence"],
                "targeted_contracts": ["target_contract"],
                "protected_contracts": ["protected_contract"],
            },
        )

        self.assertGreater(decision.candidate_mean, decision.current_mean)
        self.assertFalse(decision.accepted)
        self.assertIn("protected_contract_regressed", decision.contract_policy_guard["issues"])
        self.assertEqual(
            decision.contract_policy_guard["targeted_improved_contracts"],
            ["target_contract"],
        )

    def test_validation_gate_rejects_one_round_false_positive(self) -> None:
        runner = SequencedScoreRunner(
            {
                "current": [0.9, 0.8],
                "candidate": [1.0, 0.8, 0.8],
            }
        )
        optimizer = ExecutiveSkillOptimizer(
            runner,
            NumericScorer(),
            object(),
            ExecutiveOptimizerConfig(
                validation_confirmation_rounds=2,
                validation_required_wins=2,
                validation_mean_delta=0.05,
            ),
        )

        decision = optimizer.validate_candidate(
            "current",
            "candidate",
            [Task(id="selection", input="")],
            candidate_name="candidate",
            current_score=0.8,
        )

        self.assertFalse(decision.accepted)
        self.assertEqual(decision.wins, 1)
        self.assertEqual(decision.total_rounds, 3)
        self.assertAlmostEqual(decision.candidate_mean - decision.current_mean, 1 / 30)

    def test_validation_gate_accepts_stable_majority_and_aggregates_evidence(self) -> None:
        runner = SequencedScoreRunner(
            {
                "current": [0.8, 0.8],
                "candidate": [1.0, 1.0, 0.9],
            }
        )
        optimizer = ExecutiveSkillOptimizer(
            runner,
            NumericScorer(),
            object(),
            ExecutiveOptimizerConfig(
                validation_confirmation_rounds=2,
                validation_required_wins=2,
                validation_mean_delta=0.05,
            ),
        )

        decision = optimizer.validate_candidate(
            "current",
            "candidate",
            [Task(id="selection", input="")],
            candidate_name="candidate",
            current_score=0.8,
        )

        self.assertTrue(decision.accepted)
        self.assertEqual(decision.wins, 3)
        self.assertEqual(len(decision.candidate_report.results), 3)
        self.assertAlmostEqual(decision.candidate_report.average_score, decision.candidate_mean)

    def test_evaluate_targeted_retry_replaces_transient_agent_failure(self) -> None:
        class FlakyRunner:
            def __init__(self) -> None:
                self.calls = 0

            def run(self, skill_text, task):
                self.calls += 1
                success = self.calls == 2
                return TaskOutput(
                    {"tests_passed": success, "agent_returncode": 0 if success else 1},
                    metadata={
                        "agent": {
                            "returncode": 0 if success else 1,
                            "stdout": "fixed" if success else "",
                            "stderr": "",
                            "timed_out": False,
                        },
                        "post_test": {"returncode": 0 if success else 1},
                        "diff": "patch" if success else "",
                    },
                )

        runner = FlakyRunner()
        optimizer = ExecutiveSkillOptimizer(
            runner,
            CodingScorer(),
            object(),
            ExecutiveOptimizerConfig(
                epochs=1,
                task_retry_limit=1,
                task_retry_backoff_seconds=0,
                enable_slow_update=False,
            ),
            retry_detector=coding_retryable_anomaly_reasons,
        )

        report = optimizer.evaluate("skill", [Task(id="t", input="", expected={"tests_passed": True})])

        self.assertEqual(runner.calls, 2)
        self.assertTrue(report.results[0].score.success)
        policy = report.results[0].output.metadata["retry_policy"]
        self.assertEqual(policy["attempt_count"], 2)
        self.assertFalse(policy["persistent_anomaly"])

    def test_evaluate_raises_on_persistent_agent_failure(self) -> None:
        class BrokenRunner:
            def run(self, skill_text, task):
                return TaskOutput(
                    {"tests_passed": False, "agent_returncode": 124},
                    metadata={
                        "agent": {"returncode": 124, "stdout": "", "stderr": "timeout", "timed_out": True},
                        "post_test": {"returncode": 1},
                        "diff": "",
                    },
                )

        optimizer = ExecutiveSkillOptimizer(
            BrokenRunner(),
            CodingScorer(),
            object(),
            ExecutiveOptimizerConfig(
                epochs=1,
                task_retry_limit=1,
                task_retry_backoff_seconds=0,
                enable_slow_update=False,
            ),
            retry_detector=coding_retryable_anomaly_reasons,
        )

        with self.assertRaises(PersistentTaskAnomaly):
            optimizer.evaluate("skill", [Task(id="t", input="", expected={"tests_passed": True})])

    def test_learning_rate_schedules_decay_to_floor(self) -> None:
        self.assertEqual([scheduled_learning_rate(4, 2, "linear", step, 3) for step in (1, 2, 3)], [4, 3, 2])
        self.assertEqual([scheduled_learning_rate(4, 2, "cosine", step, 3) for step in (1, 2, 3)], [4, 3, 2])
        self.assertEqual(scheduled_learning_rate(4, 2, "constant", 3, 3), 4)

    def test_reflection_minibatches_separate_failures_and_successes(self) -> None:
        tasks = [Task(id=str(index), input="") for index in range(3)]
        results = []
        for index, task in enumerate(tasks):
            success = index == 2
            results.append(
                type("Result", (), {"task": task, "score": Score(float(success), success)})()
            )

        batches = reflection_minibatches(results, 1)

        self.assertEqual([kind for kind, _ in batches], ["failure", "failure", "success"])

    def test_minibatch_edits_merge_and_pass_strict_selection_gate(self) -> None:
        class DuplicateEditor(AtomicEditor):
            def __init__(self) -> None:
                self.calls = 0

            def propose(self, skill_text, train_results, *, epoch, **kwargs):
                self.calls += 1
                return [
                    EditProposal(
                        name=f"rule-{self.calls}",
                        rationale="Repeated failure evidence.",
                        edits=(AtomicEdit("add", END_TARGET, "Rule A"),),
                    )
                ]

        editor = DuplicateEditor()
        train = [Task(id="t1", input="", expected=["Rule A"]), Task(id="t2", input="", expected=["Rule A"])]
        selection = [Task(id="s1", input="", expected=["Rule A"])]
        optimizer = ExecutiveSkillOptimizer(
            SkillTextRunner(),
            ExpectedRulesScorer(),
            editor,
            ExecutiveOptimizerConfig(
                epochs=1,
                rollout_batch_size=2,
                reflection_minibatch_size=1,
                learning_rate=1,
                learning_rate_floor=1,
                enable_slow_update=False,
            ),
        )

        result = optimizer.optimize("# Skill\n", train, selection)

        self.assertEqual(result.best_validation_score, 1.0)
        self.assertIn("Rule A", result.best_skill_text)
        self.assertEqual(editor.calls, 2)
        accepted = [item for item in result.history if item.accepted and item.epoch == 1]
        self.assertEqual(accepted[0].metadata["ranked_edits"][0]["support"], 2)

    def test_early_stop_validation_score_skips_training_when_initial_is_target(self) -> None:
        class FailingEditor(AtomicEditor):
            def propose(self, *args, **kwargs):
                raise AssertionError("editor should not be called after target validation score")

        train = [Task(id="t1", input="", expected=["Rule B"])]
        selection = [Task(id="s1", input="", expected=["Rule A"])]
        optimizer = ExecutiveSkillOptimizer(
            SkillTextRunner(),
            ExpectedRulesScorer(),
            FailingEditor(),
            ExecutiveOptimizerConfig(
                epochs=2,
                rollout_batch_size=1,
                reflection_minibatch_size=1,
                learning_rate=1,
                learning_rate_floor=1,
                enable_slow_update=False,
                early_stop_validation_score=1.0,
            ),
        )

        with tempfile.TemporaryDirectory() as tmp:
            result = optimizer.optimize("# Skill\nRule A\n", train, selection, run_dir=tmp)
            written = json.loads((Path(tmp) / "result.json").read_text(encoding="utf-8"))

        self.assertEqual(result.stop_reason, "early_stop_validation_score_target")
        self.assertEqual(result.total_steps, 0)
        self.assertEqual(written["stop_reason"], "early_stop_validation_score_target")

    def test_early_stop_validation_score_stops_after_accepted_target_candidate(self) -> None:
        class CountingEditor(AtomicEditor):
            def __init__(self) -> None:
                self.calls = 0

            def propose(self, skill_text, train_results, *, epoch, **kwargs):
                self.calls += 1
                return [
                    EditProposal(
                        name="add-rule-a",
                        rationale="Selection target can be reached.",
                        edits=(AtomicEdit("add", END_TARGET, "Rule A"),),
                    )
                ]

        editor = CountingEditor()
        train = [
            Task(id="t1", input="", expected=["Rule A"]),
            Task(id="t2", input="", expected=["Rule A"]),
        ]
        selection = [Task(id="s1", input="", expected=["Rule A"])]
        optimizer = ExecutiveSkillOptimizer(
            SkillTextRunner(),
            ExpectedRulesScorer(),
            editor,
            ExecutiveOptimizerConfig(
                epochs=2,
                rollout_batch_size=1,
                reflection_minibatch_size=1,
                learning_rate=1,
                learning_rate_floor=1,
                enable_slow_update=False,
                early_stop_validation_score=1.0,
            ),
        )

        result = optimizer.optimize("# Skill\n", train, selection)

        self.assertEqual(result.stop_reason, "early_stop_validation_score_target")
        self.assertEqual(result.best_validation_score, 1.0)
        self.assertEqual(result.total_steps, 1)
        self.assertEqual(editor.calls, 1)

    def test_rejected_buffer_is_available_to_later_batch_in_same_epoch(self) -> None:
        class BufferEditor(AtomicEditor):
            def __init__(self) -> None:
                self.buffers = []

            def propose(self, skill_text, train_results, *, epoch, rejected_buffer=None, **kwargs):
                self.buffers.append(list(rejected_buffer or []))
                content = "Bad Rule" if not rejected_buffer else "Good Rule"
                return [EditProposal(content, rationale=content, edits=(AtomicEdit("add", END_TARGET, content),))]

        editor = BufferEditor()
        train = [Task(id="t1", input="", expected=["Good Rule"]), Task(id="t2", input="", expected=["Good Rule"])]
        selection = [Task(id="s1", input="", expected=["Good Rule"])]
        optimizer = ExecutiveSkillOptimizer(
            SkillTextRunner(),
            ExpectedRulesScorer(),
            editor,
            ExecutiveOptimizerConfig(
                epochs=1,
                rollout_batch_size=1,
                reflection_minibatch_size=1,
                learning_rate=1,
                learning_rate_floor=1,
                enable_slow_update=False,
                seed=1,
            ),
        )

        result = optimizer.optimize("# Skill\n", train, selection)

        self.assertEqual(editor.buffers[0], [])
        self.assertEqual(editor.buffers[1][0]["reason"], "validation_gate_rejected")
        self.assertIn("contract_evidence", editor.buffers[1][0]["metadata"]["validation_gate"])
        self.assertIn("Good Rule", result.best_skill_text)

    def test_evidence_guided_guard_only_proposal_is_rejected_before_selection_gate(self) -> None:
        class GuardOnlyEditor(AtomicEditor):
            def propose(self, skill_text, train_results, *, epoch, **kwargs):
                return [
                    EditProposal(
                        "guard-only",
                        rationale="Only asks the agent to preserve previous behavior.",
                        metadata={
                            "evidence_source": "contract_rejection_evidence",
                            "targeted_contracts": ["unicode_casefold"],
                            "protected_contracts": ["largest_remainder", "input_validation"],
                            "expected_behavior_change": "Improve unicode handling without regressions.",
                        },
                        edits=(
                            AtomicEdit(
                                "add",
                                END_TARGET,
                                (
                                    "Before modifying any implementation logic, first run all existing public tests "
                                    "and confirm every already passing test still passes."
                                ),
                            ),
                        ),
                    )
                ]

        optimizer = ExecutiveSkillOptimizer(
            SkillTextRunner(),
            ExpectedRulesScorer(),
            GuardOnlyEditor(),
            ExecutiveOptimizerConfig(
                epochs=1,
                rollout_batch_size=1,
                reflection_minibatch_size=1,
                learning_rate=1,
                learning_rate_floor=1,
                enable_slow_update=False,
            ),
        )

        result = optimizer.optimize(
            "# Skill\n",
            [Task(id="train", input="", expected=["unicode casefold rule"])],
            [Task(id="selection", input="", expected=["unicode casefold rule"])],
        )

        self.assertEqual(result.accepted_steps, 0)
        self.assertEqual(result.history[-1].candidate, "guard-only")
        self.assertEqual(result.history[-1].metadata["rejection_reason"], "proposal_policy_rejected")
        self.assertIn(
            "target_mechanism_missing",
            result.history[-1].metadata["proposal_policy_issues"],
        )
        self.assertFalse(any(item.candidate.startswith("atomic-") for item in result.history))

    def test_evidence_guided_proposal_requires_protected_mechanisms_in_edit_text(self) -> None:
        weak = EditProposal(
            "largest-remainder-only",
            rationale="Targets largest remainder but only declares protected contracts in metadata.",
            metadata={
                "evidence_source": "contract_rejection_evidence",
                "targeted_contracts": ["largest_remainder"],
                "protected_contracts": ["input_validation", "stable_order"],
                "expected_behavior_change": "Fix leftover allocation without regressions.",
            },
            edits=(
                AtomicEdit(
                    "add",
                    END_TARGET,
                    (
                        "For proportional allocation, confirm the remainder distribution sort key uses "
                        "(-remainder, original index) before finalizing."
                    ),
                ),
            ),
        )
        grounded = EditProposal(
            "largest-remainder-with-guards",
            rationale="Targets largest remainder and preserves the protected mechanisms in the skill text.",
            metadata={
                "evidence_source": "contract_rejection_evidence",
                "targeted_contracts": ["largest_remainder"],
                "protected_contracts": ["input_validation", "stable_order"],
                "expected_behavior_change": "Fix leftover allocation without regressions.",
            },
            edits=(
                AtomicEdit(
                    "add",
                    END_TARGET,
                    (
                        "For proportional allocation, raise ValueError for negative totals or weights, "
                        "return all zeros for zero-sum weights, preserve stable output order, and "
                        "compute quotas, floor them, then distribute leftover units by largest "
                        "fractional remainder with original-index tie breaks."
                    ),
                ),
            ),
        )
        wrong_zero_sum = EditProposal(
            "largest-remainder-wrong-zero-sum",
            rationale="Names the right target but gives the target agent the wrong zero-sum branch.",
            metadata={
                "evidence_source": "contract_rejection_evidence",
                "targeted_contracts": ["largest_remainder"],
                "protected_contracts": ["input_validation", "stable_order"],
                "expected_behavior_change": "Fix leftover allocation without regressions.",
            },
            edits=(
                AtomicEdit(
                    "add",
                    END_TARGET,
                    (
                        "For proportional allocation, raise ValueError for negative totals or weights, "
                        "raise ValueError for zero-sum weights, preserve stable output order, and "
                        "compute quotas, floor them, then distribute leftover units by largest "
                        "fractional remainder with original-index tie breaks."
                    ),
                ),
            ),
        )
        generic_validation = EditProposal(
            "largest-remainder-generic-validation",
            rationale="Uses generic validation language without naming a concrete guard.",
            metadata={
                "evidence_source": "contract_rejection_evidence",
                "targeted_contracts": ["largest_remainder"],
                "protected_contracts": ["input_validation", "stable_order"],
                "expected_behavior_change": "Fix leftover allocation without regressions.",
            },
            edits=(
                AtomicEdit(
                    "add",
                    END_TARGET,
                    (
                        "When implementing largest-remainder proportional allocation, return all zeros "
                        "for zero-sum weights, preserve existing invalid input rejection rules, and "
                        "break ties by ascending index."
                    ),
                ),
            ),
        )

        weak_issues = evidence_guided_proposal_filter_issues(weak)
        wrong_zero_sum_issues = evidence_guided_proposal_filter_issues(wrong_zero_sum)
        generic_issues = evidence_guided_proposal_filter_issues(generic_validation)
        grounded_issues = evidence_guided_proposal_filter_issues(grounded)

        self.assertIn("protected_mechanism_missing:input_validation", weak_issues)
        self.assertIn("target_mechanism_missing", wrong_zero_sum_issues)
        self.assertIn("protected_mechanism_missing:input_validation", generic_issues)
        self.assertEqual(grounded_issues, [])

    def test_rejected_buffer_persists_to_next_epoch(self) -> None:
        class BufferEditor(AtomicEditor):
            def __init__(self) -> None:
                self.buffers = []

            def propose(self, skill_text, train_results, *, epoch, rejected_buffer=None, **kwargs):
                self.buffers.append(list(rejected_buffer or []))
                content = "Bad Rule" if not rejected_buffer else "Good Rule"
                return [EditProposal(content, rationale=content, edits=(AtomicEdit("add", END_TARGET, content),))]

        editor = BufferEditor()
        train = [Task(id="t1", input="", expected=["Good Rule"])]
        selection = [Task(id="s1", input="", expected=["Good Rule"])]
        optimizer = ExecutiveSkillOptimizer(
            SkillTextRunner(),
            ExpectedRulesScorer(),
            editor,
            ExecutiveOptimizerConfig(
                epochs=2,
                rollout_batch_size=1,
                reflection_minibatch_size=1,
                learning_rate=1,
                learning_rate_floor=1,
                enable_slow_update=False,
                seed=1,
            ),
        )

        result = optimizer.optimize("# Skill\n", train, selection)

        self.assertEqual(editor.buffers[0], [])
        self.assertEqual(editor.buffers[1][0]["reason"], "validation_gate_rejected")
        self.assertIn("Good Rule", result.best_skill_text)
        self.assertEqual(result.accepted_steps, 1)

    def test_repeated_generic_contract_audit_is_penalized_in_ranking(self) -> None:
        class RepeatingEditor(AtomicEditor):
            def __init__(self) -> None:
                self.calls = 0

            def propose(self, skill_text, train_results, *, epoch, rejected_buffer=None, **kwargs):
                self.calls += 1
                generic = EditProposal(
                    f"generic-{self.calls}",
                    rationale="Generic audit advice.",
                    metadata={"priority": 0.5},
                    edits=(
                        AtomicEdit(
                            "add",
                            END_TARGET,
                            "Verify all documented contract requirements before finalizing.",
                        ),
                    ),
                )
                if self.calls == 1:
                    return [generic]
                return [
                    generic,
                    EditProposal(
                        "specific-good",
                        rationale="Specific observed missing rule.",
                        edits=(AtomicEdit("add", END_TARGET, "Good Rule"),),
                    ),
                ]

        editor = RepeatingEditor()
        train = [Task(id="t1", input="", expected=["Good Rule"])]
        selection = [Task(id="s1", input="", expected=["Good Rule"])]
        optimizer = ExecutiveSkillOptimizer(
            SkillTextRunner(),
            ExpectedRulesScorer(),
            editor,
            ExecutiveOptimizerConfig(
                epochs=2,
                rollout_batch_size=1,
                reflection_minibatch_size=1,
                learning_rate=1,
                learning_rate_floor=1,
                enable_slow_update=False,
                seed=1,
            ),
        )

        result = optimizer.optimize("# Skill\n", train, selection)

        self.assertIn("Good Rule", result.best_skill_text)
        accepted = [item for item in result.history if item.accepted and item.candidate != "initial"]
        self.assertEqual(len(accepted), 1)
        self.assertEqual(accepted[0].metadata["selected_edits"][0]["content"], "Good Rule")
        generic_ranked = [
            item
            for item in accepted[0].metadata["ranked_edits"]
            if "Verify all documented" in item["content"]
        ][0]
        self.assertLess(generic_ranked["priority"], 0.0)

    def test_retargeting_recently_failed_contract_without_new_mechanism_is_penalized(self) -> None:
        rejected_all = [
            RejectedProposal(
                epoch=1,
                candidate="atomic-epoch-1-batch-1",
                reason="validation_gate_rejected",
                rationale="No improvement.",
                metadata={
                    "validation_gate": {
                        "contract_evidence": {
                            "top_negative_contracts": [],
                            "top_no_improvement_contracts": [
                                {
                                    "contract": "input_validation",
                                    "current_accuracy": 0.0,
                                    "candidate_accuracy": 0.0,
                                    "delta": 0.0,
                                }
                            ],
                        }
                    }
                },
            )
        ]
        repeat = EditProposal(
            "repeat-input-validation",
            rationale="Try the same target again.",
            metadata={
                "targeted_contracts": ["input_validation"],
                "evidence_source": "contract_rejection_evidence",
                "expected_behavior_change": "Handle invalid inputs.",
            },
            edits=(AtomicEdit("add", END_TARGET, "Validate inputs before processing."),),
        )
        new_mechanism = EditProposal(
            "new-input-validation-mechanism",
            rationale="Use a new observed mechanism.",
            metadata={
                "targeted_contracts": ["input_validation"],
                "evidence_source": "contract_rejection_evidence",
                "expected_behavior_change": "Handle invalid inputs before list mutation.",
                "cooldown_override": "New evidence shows the prior rule missed mutation before validation.",
            },
            edits=(AtomicEdit("add", END_TARGET, "Validate inputs before mutating lists."),),
        )

        repeat_penalty = local_proposal_penalty(repeat, rejected_all)
        new_mechanism_penalty = local_proposal_penalty(new_mechanism, rejected_all)

        self.assertGreaterEqual(repeat_penalty, 1.25)
        self.assertLess(new_mechanism_penalty, repeat_penalty)

    def test_early_stop_rejection_limit_writes_checkpoint_and_stops_batches(self) -> None:
        class BadEditor(AtomicEditor):
            def __init__(self) -> None:
                self.calls = 0

            def propose(self, skill_text, train_results, *, epoch, **kwargs):
                self.calls += 1
                return [
                    EditProposal(
                        f"bad-{self.calls}",
                        rationale="Repeated unhelpful edit.",
                        edits=(AtomicEdit("add", END_TARGET, f"Bad Rule {self.calls}"),),
                    )
                ]

        editor = BadEditor()
        train = [
            Task(id="t1", input="", expected=["Bad Rule 1"]),
            Task(id="t2", input="", expected=["Bad Rule 2"]),
        ]
        selection = [Task(id="s1", input="", expected=["Good Rule"])]
        optimizer = ExecutiveSkillOptimizer(
            SkillTextRunner(),
            ExpectedRulesScorer(),
            editor,
            ExecutiveOptimizerConfig(
                epochs=1,
                rollout_batch_size=1,
                reflection_minibatch_size=1,
                learning_rate=1,
                learning_rate_floor=1,
                enable_slow_update=False,
                early_stop_rejection_limit=1,
            ),
        )

        with tempfile.TemporaryDirectory() as tmp:
            result = optimizer.optimize("# Skill\n", train, selection, run_dir=tmp)
            checkpoint = json.loads((Path(tmp) / "result_checkpoint.json").read_text(encoding="utf-8"))
            written = json.loads((Path(tmp) / "result.json").read_text(encoding="utf-8"))

        self.assertEqual(editor.calls, 1)
        self.assertEqual(result.stop_reason, "early_stop_validation_rejection_limit")
        self.assertEqual(result.checkpoint["validation_rejection_streak"], 1)
        self.assertEqual(checkpoint["stop_reason"], "early_stop_validation_rejection_limit")
        self.assertEqual(written["stop_reason"], "early_stop_validation_rejection_limit")

    def test_optimize_writes_timing_events_for_candidate_gate(self) -> None:
        class BadEditor(AtomicEditor):
            def propose(self, skill_text, train_results, *, epoch, **kwargs):
                return [
                    EditProposal(
                        "bad",
                        rationale="Training-only edit.",
                        edits=(AtomicEdit("add", END_TARGET, "Bad Rule"),),
                    )
                ]

        train = [Task(id="train", input="", expected=["Bad Rule"])]
        selection = [Task(id="selection", input="", expected=["Good Rule"])]
        optimizer = ExecutiveSkillOptimizer(
            SkillTextRunner(),
            ExpectedRulesScorer(),
            BadEditor(),
            ExecutiveOptimizerConfig(
                epochs=1,
                rollout_batch_size=1,
                reflection_minibatch_size=1,
                learning_rate=1,
                learning_rate_floor=1,
                enable_slow_update=False,
            ),
        )

        with tempfile.TemporaryDirectory() as tmp:
            optimizer.optimize("# Skill\n", train, selection, run_dir=tmp)
            events = [
                json.loads(line)
                for line in (Path(tmp) / "timing_events.jsonl").read_text(encoding="utf-8").splitlines()
            ]

        event_names = [event["event"] for event in events]
        self.assertIn("candidate_written", event_names)
        self.assertIn("validation_started", event_names)
        self.assertIn("validation_gate_written", event_names)
        self.assertIn("validation_finished", event_names)
        self.assertIn("task_started", event_names)
        self.assertIn("task_finished", event_names)
        validation_finished = [
            event for event in events if event["event"] == "validation_finished"
        ][0]
        self.assertEqual(validation_finished["candidate_name"], "atomic-epoch-1-batch-1")
        self.assertIn("duration_seconds", validation_finished)

    def test_final_report_reuses_gate_evidence_without_extra_evaluation(self) -> None:
        class CountingRunner(SkillTextRunner):
            def __init__(self) -> None:
                self.task_ids = []

            def run(self, skill_text: str, task: Task) -> TaskOutput:
                self.task_ids.append(task.id)
                return super().run(skill_text, task)

        class RuleEditor(AtomicEditor):
            def propose(self, skill_text, train_results, *, epoch, **kwargs):
                return [
                    EditProposal(
                        "rule",
                        rationale="Verified rule.",
                        edits=(AtomicEdit("add", END_TARGET, "Rule A"),),
                    )
                ]

        runner = CountingRunner()
        optimizer = ExecutiveSkillOptimizer(
            runner,
            ExpectedRulesScorer(),
            RuleEditor(),
            ExecutiveOptimizerConfig(
                epochs=1,
                rollout_batch_size=1,
                reflection_minibatch_size=1,
                learning_rate=1,
                learning_rate_floor=1,
                enable_slow_update=False,
                validation_confirmation_rounds=2,
                validation_required_wins=2,
                validation_mean_delta=0.05,
            ),
        )

        result = optimizer.optimize(
            "# Skill\n",
            [Task(id="train", input="", expected=["Rule A"])],
            [Task(id="selection", input="", expected=["Rule A"])],
        )

        self.assertEqual(runner.task_ids.count("selection"), 6)
        self.assertEqual(len(result.final_validation_report.results), 3)
        self.assertEqual(result.best_validation_score, 1.0)

    def test_epoch_state_update_can_pass_gate_and_update_meta_skill(self) -> None:
        class SlowEditor(AtomicEditor):
            def propose(self, skill_text, train_results, *, epoch, **kwargs):
                return [
                    EditProposal(
                        "fast",
                        rationale="Fast evidence.",
                        edits=(AtomicEdit("add", END_TARGET, "Fast Rule"),),
                    )
                ]

            def update_state(self, **kwargs):
                return OptimizerStateUpdate(
                    meta_skill="Preserve repeated verified improvements.",
                    slow_update="Slow Rule",
                    rationale="Longitudinal comparison found a durable improvement.",
                )

        train = [Task(id="t1", input="", expected=["Fast Rule"])]
        selection = [Task(id="s1", input="", expected=["Fast Rule", "Slow Rule"])]
        optimizer = ExecutiveSkillOptimizer(
            SkillTextRunner(),
            ExpectedRulesScorer(),
            SlowEditor(),
            ExecutiveOptimizerConfig(
                epochs=1,
                rollout_batch_size=1,
                reflection_minibatch_size=1,
                learning_rate=1,
                learning_rate_floor=1,
                enable_slow_update=True,
            ),
        )

        with tempfile.TemporaryDirectory() as tmp:
            result = optimizer.optimize("# Skill\n", train, selection, run_dir=tmp)
            self.assertTrue((Path(tmp) / "meta_skill_epoch_1.md").is_file())

        self.assertEqual(result.best_validation_score, 1.0)
        self.assertIn(SLOW_START, result.best_skill_text)
        self.assertIn("Slow Rule", result.best_skill_text)
        self.assertIn("verified improvements", result.meta_skill_text)


if __name__ == "__main__":
    unittest.main()
