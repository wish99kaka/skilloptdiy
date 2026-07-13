import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from textskill_optimizer.models import EvaluationReport, Score, Task, TaskOutput, TaskResult


ROOT = Path(__file__).resolve().parent.parent
MODULE_PATH = ROOT / "work/run_cross_agent_skill_eval.py"


def load_eval_module():
    spec = importlib.util.spec_from_file_location("run_cross_agent_skill_eval", MODULE_PATH)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def make_result(
    *,
    success: bool,
    diff: str = "",
    agent_returncode: int = 0,
    post_test_returncode: int = 0,
    stdout: str = "done",
    stderr: str = "",
    timed_out: bool = False,
) -> TaskResult:
    task = Task(id="task-1", input="Fix it", expected={"tests_passed": True})
    output = TaskOutput(
        value={"tests_passed": success, "agent_returncode": agent_returncode},
        metadata={
            "agent": {
                "returncode": agent_returncode,
                "stdout": stdout,
                "stderr": stderr,
                "timed_out": timed_out,
            },
            "post_test": {"returncode": post_test_returncode},
            "diff": diff,
        },
    )
    score = Score(1.0 if success else 0.0, success)
    return TaskResult(task=task, output=output, score=score)


class CrossAgentSkillEvalTests(unittest.TestCase):
    def test_kilo_tool_fragment_without_diff_is_retryable(self) -> None:
        module = load_eval_module()
        result = make_result(
            success=False,
            post_test_returncode=1,
            stdout='<seed:tool name="read"></seed:tool_call>',
        )

        reasons = module.retryable_anomaly_reasons(result)

        self.assertIn("failed_post_test_without_repo_change", reasons)
        self.assertIn("malformed_tool_call_without_repo_change", reasons)

    def test_nonempty_diff_hidden_failure_is_not_retryable(self) -> None:
        module = load_eval_module()
        result = make_result(
            success=False,
            post_test_returncode=1,
            diff="--- a/app.py\n+++ b/app.py\n@@\n-return []\n+return [1]\n",
            stdout="changed app.py",
        )

        self.assertEqual(module.retryable_anomaly_reasons(result), [])

    def test_successful_result_is_not_retryable_even_when_agent_times_out(self) -> None:
        module = load_eval_module()
        result = make_result(
            success=True,
            agent_returncode=124,
            post_test_returncode=0,
            diff="--- a/app.py\n+++ b/app.py\n@@\n-return []\n+return [1]\n",
            timed_out=True,
        )

        self.assertEqual(module.retryable_anomaly_reasons(result), [])

    def test_evaluate_with_retries_selects_second_attempt_after_anomaly(self) -> None:
        module = load_eval_module()
        task = Task(id="task-1", input="Fix it", expected={"tests_passed": True})

        class FakeOptimizer:
            def __init__(self) -> None:
                self.calls = 0

            def evaluate(self, skill_text, tasks, *, name):
                self.calls += 1
                if self.calls == 1:
                    return EvaluationReport(
                        name=name,
                        results=[
                            make_result(
                                success=False,
                                post_test_returncode=1,
                                stdout='<seed:tool name="read"></seed:tool_call>',
                            )
                        ],
                    )
                return EvaluationReport(
                    name=name,
                    results=[
                        make_result(
                            success=True,
                            post_test_returncode=0,
                            diff="--- a/app.py\n+++ b/app.py\n@@\n-return []\n+return [1]\n",
                        )
                    ],
                )

        fake = FakeOptimizer()

        report = module.evaluate_with_retries(
            fake,
            "skill",
            [task],
            name="kilo:revised",
            max_retries=1,
        )

        self.assertEqual(fake.calls, 2)
        self.assertTrue(report.results[0].score.success)
        retry_policy = report.results[0].output.metadata["retry_policy"]
        self.assertEqual(retry_policy["attempt_count"], 2)
        self.assertEqual(retry_policy["selected_attempt"], 1)
        self.assertTrue(retry_policy["attempts"][0]["retryable"])
        self.assertFalse(retry_policy["attempts"][1]["retryable"])

    def test_health_check_flags_persistent_retryable_anomaly(self) -> None:
        module = load_eval_module()
        task = Task(id="task-1", input="Fix it", expected={"tests_passed": True})

        class FakeOptimizer:
            seen_task = None

            def evaluate(self, skill_text, tasks, *, name):
                self.seen_task = tasks[0]
                return EvaluationReport(
                    name=name,
                    results=[
                        make_result(
                            success=False,
                            agent_returncode=1,
                            post_test_returncode=1,
                            stdout="",
                            stderr="service unavailable",
                        )
                    ],
                )

        fake = FakeOptimizer()

        report, reasons = module.run_agent_health_check(
            fake,
            "skill",
            task,
            name="ccr:revised:health",
            max_retries=0,
            timeout_seconds=45,
        )

        self.assertEqual(fake.seen_task.metadata["timeout_seconds"], 45)
        self.assertIn("agent_nonzero_returncode", reasons)
        self.assertIn("failed_post_test_without_repo_change", reasons)

    def test_health_check_accepts_nonempty_diff_failure_as_agent_available(self) -> None:
        module = load_eval_module()
        task = Task(id="task-1", input="Fix it", expected={"tests_passed": True})

        class FakeOptimizer:
            def evaluate(self, skill_text, tasks, *, name):
                return EvaluationReport(
                    name=name,
                    results=[
                        make_result(
                            success=False,
                            post_test_returncode=1,
                            diff="--- a/app.py\n+++ b/app.py\n@@\n-return []\n+return [1]\n",
                        )
                    ],
                )

        _, reasons = module.run_agent_health_check(
            FakeOptimizer(),
            "skill",
            task,
            name="kilo:revised:health",
            max_retries=0,
            timeout_seconds=None,
        )

        self.assertEqual(reasons, [])

    def test_adaptive_task_stops_after_first_two_agents_agree(self) -> None:
        module = load_eval_module()
        task_by_agent = {
            agent: {"task-1": Task(id="task-1", input="Fix it")}
            for agent in ("coco", "ccr", "kilo")
        }

        class FakeOptimizer:
            def __init__(self) -> None:
                self.calls: list[str] = []

            def evaluate(self, skill_text, tasks, *, name):
                agent = name.split(":", 1)[0]
                self.calls.append(agent)
                return EvaluationReport(name=name, results=[make_result(success=True)])

        fake = FakeOptimizer()

        results, vote = module.evaluate_adaptive_task(
            fake,
            "skill",
            task_by_agent,
            ["coco", "ccr", "kilo"],
            skill_label="skill-a",
            task_id="task-1",
            max_retries=0,
            force_full_audit=False,
        )

        self.assertEqual(fake.calls, ["coco", "ccr"])
        self.assertEqual(set(results), {"coco", "ccr"})
        self.assertEqual(vote["decision_reason"], "first_two_agree")
        self.assertEqual(vote["passed"], 2)
        self.assertEqual(vote["total"], 2)
        self.assertEqual(vote["skipped_agents"], ["kilo"])

    def test_adaptive_task_runs_third_agent_on_disagreement(self) -> None:
        module = load_eval_module()
        task_by_agent = {
            agent: {"task-1": Task(id="task-1", input="Fix it")}
            for agent in ("coco", "ccr", "kilo")
        }

        class FakeOptimizer:
            def __init__(self) -> None:
                self.calls: list[str] = []

            def evaluate(self, skill_text, tasks, *, name):
                agent = name.split(":", 1)[0]
                self.calls.append(agent)
                return EvaluationReport(
                    name=name,
                    results=[make_result(success=agent != "ccr", diff="changed")],
                )

        fake = FakeOptimizer()

        _, vote = module.evaluate_adaptive_task(
            fake,
            "skill",
            task_by_agent,
            ["coco", "ccr", "kilo"],
            skill_label="skill-a",
            task_id="task-1",
            max_retries=0,
            force_full_audit=False,
        )

        self.assertEqual(fake.calls, ["coco", "ccr", "kilo"])
        self.assertEqual(vote["decision_reason"], "third_agent_breaker")
        self.assertEqual(vote["passed"], 2)
        self.assertEqual(vote["total"], 3)

    def test_adaptive_task_full_audit_runs_third_agent_even_when_first_two_agree(self) -> None:
        module = load_eval_module()
        task_by_agent = {
            agent: {"task-1": Task(id="task-1", input="Fix it")}
            for agent in ("coco", "ccr", "kilo")
        }

        class FakeOptimizer:
            def __init__(self) -> None:
                self.calls: list[str] = []

            def evaluate(self, skill_text, tasks, *, name):
                agent = name.split(":", 1)[0]
                self.calls.append(agent)
                return EvaluationReport(name=name, results=[make_result(success=True)])

        fake = FakeOptimizer()

        _, vote = module.evaluate_adaptive_task(
            fake,
            "skill",
            task_by_agent,
            ["coco", "ccr", "kilo"],
            skill_label="skill-a",
            task_id="task-1",
            max_retries=0,
            force_full_audit=True,
        )

        self.assertEqual(fake.calls, ["coco", "ccr", "kilo"])
        self.assertEqual(vote["decision_reason"], "full_audit")
        self.assertEqual(vote["passed"], 3)
        self.assertEqual(vote["total"], 3)

    def test_adaptive_task_does_not_count_retryable_anomaly_as_vote(self) -> None:
        module = load_eval_module()
        task_by_agent = {
            agent: {"task-1": Task(id="task-1", input="Fix it")}
            for agent in ("coco", "ccr", "kilo")
        }

        class FakeOptimizer:
            def __init__(self) -> None:
                self.calls: list[str] = []

            def evaluate(self, skill_text, tasks, *, name):
                agent = name.split(":", 1)[0]
                self.calls.append(agent)
                if agent == "coco":
                    result = make_result(
                        success=False,
                        agent_returncode=124,
                        post_test_returncode=1,
                        stdout="",
                        stderr="timeout",
                        timed_out=True,
                    )
                else:
                    result = make_result(success=True, diff="changed")
                return EvaluationReport(name=name, results=[result])

        fake = FakeOptimizer()

        _, vote = module.evaluate_adaptive_task(
            fake,
            "skill",
            task_by_agent,
            ["coco", "ccr", "kilo"],
            skill_label="skill-a",
            task_id="task-1",
            max_retries=0,
            force_full_audit=False,
        )

        self.assertEqual(fake.calls, ["coco", "ccr", "kilo"])
        self.assertEqual(vote["passed"], 2)
        self.assertEqual(vote["total"], 2)
        self.assertEqual(vote["invalid_votes"][0]["agent"], "coco")

    def test_stable_agent_order_is_reproducible(self) -> None:
        module = load_eval_module()

        first = module.stable_agent_order(["coco", "ccr", "kilo"], "seed-1", "skill", "task")
        second = module.stable_agent_order(["coco", "ccr", "kilo"], "seed-1", "skill", "task")

        self.assertEqual(first, second)
        self.assertEqual(sorted(first), ["ccr", "coco", "kilo"])

    def test_full_audit_selection_uses_reproducible_nonzero_quota(self) -> None:
        module = load_eval_module()
        task_ids = ["task-1", "task-2", "task-3", "task-4"]

        first = module.select_full_audit_task_ids("seed-1", "skill", task_ids, 0.25)
        second = module.select_full_audit_task_ids("seed-1", "skill", task_ids, 0.25)

        self.assertEqual(first, second)
        self.assertEqual(len(first), 1)
        self.assertTrue(first.issubset(set(task_ids)))

    def test_full_audit_selection_rounds_fractional_quota_up(self) -> None:
        module = load_eval_module()
        task_ids = ["task-1", "task-2", "task-3"]

        selected = module.select_full_audit_task_ids("seed-1", "skill", task_ids, 0.5)

        self.assertEqual(len(selected), 2)
        self.assertEqual(
            module.select_full_audit_task_ids("seed-1", "skill", task_ids, 0),
            set(),
        )
        self.assertEqual(
            module.select_full_audit_task_ids("seed-1", "skill", task_ids, 1),
            set(task_ids),
        )

    def test_load_target_task_ids_supports_any_failed_and_majority_failed(self) -> None:
        module = load_eval_module()
        payload = {
            "rows": [
                {"failed": ["task-agent-only"]},
                {"failed": ["agent_health_check"]},
            ],
            "votes": [
                {"task": "task-majority-failed", "majority_success": False, "tie": False},
                {"task": "task-passed", "majority_success": True, "tie": False},
            ],
        }

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "summary.json"
            path.write_text(json.dumps(payload), encoding="utf-8")

            any_failed = module.load_target_task_ids(path, "any_failed")
            majority_failed = module.load_target_task_ids(path, "majority_failed")

        self.assertEqual(any_failed, {"task-agent-only", "task-majority-failed"})
        self.assertEqual(majority_failed, {"task-majority-failed"})

    def test_skipped_adaptive_agent_has_no_score_in_summary_row(self) -> None:
        module = load_eval_module()

        row = module.build_agent_summary_row(
            agent_name="ccr",
            skill_label="skill-a",
            health_status="not_run",
            health_reasons=[],
            health_report="",
            results=[],
            task_scores={},
            skipped_tasks=["task-1"],
            task_anomalies={},
            report_path=None,
        )

        self.assertIsNone(row["average_score"])
        self.assertIsNone(row["pass_rate"])
        self.assertEqual(row["failed"], [])
        self.assertEqual(module.format_optional_float(row["average_score"]), "-")

    def test_call_savings_counts_skipped_agents(self) -> None:
        module = load_eval_module()
        rows = [
            {"agent": "coco"},
            {"agent": "ccr"},
            {"agent": "kilo"},
        ]
        votes = [
            {
                "skill": "skill-a",
                "task": "task-1",
                "agents_run": ["coco", "ccr"],
                "skipped_agents": ["kilo"],
            },
            {
                "skill": "skill-a",
                "task": "task-2",
                "agents_run": ["kilo", "coco"],
                "skipped_agents": ["ccr"],
            },
        ]

        savings = module.build_call_savings(rows, votes)

        self.assertEqual(savings["planned_agent_calls"], 6)
        self.assertEqual(savings["actual_agent_calls"], 4)
        self.assertEqual(savings["skipped_agent_calls"], 2)
        self.assertEqual(savings["saved_agent_calls"], 2)
        self.assertAlmostEqual(savings["saved_rate"], 1 / 3)
        self.assertEqual(savings["by_skill"]["skill-a"]["tasks"], 2)

    def test_call_savings_full_votes_have_zero_savings(self) -> None:
        module = load_eval_module()
        rows = [
            {"agent": "coco"},
            {"agent": "ccr"},
            {"agent": "kilo"},
        ]
        votes = [
            {
                "skill": "skill-a",
                "task": "task-1",
                "agents_run": ["coco", "ccr", "kilo"],
                "skipped_agents": [],
            }
        ]

        savings = module.build_call_savings(rows, votes)

        self.assertEqual(savings["planned_agent_calls"], 3)
        self.assertEqual(savings["actual_agent_calls"], 3)
        self.assertEqual(savings["saved_agent_calls"], 0)
        self.assertEqual(savings["saved_rate"], 0.0)


if __name__ == "__main__":
    unittest.main()
