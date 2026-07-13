import os
import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

from work.run_coding_hidden_v2_matrix import (
    aggregate_rows,
    build_baseline_evaluator,
    build_executive_config,
    build_manifest,
    build_summary,
    build_usage_report,
    contract_breakdown,
    contract_macro_accuracy,
    detect_coco_model,
    family_macro_accuracy,
    filter_tasks,
    merge_cached_baseline_rows,
    parse_conditions,
    parse_csv_set,
    parse_seeds,
    require_external_editor_config,
    run_seeds,
)
from textskill_optimizer.models import Task
from textskill_optimizer.usage_ledger import append_usage_event


class CodingHiddenV2MatrixTests(unittest.TestCase):
    def test_family_macro_accuracy_weights_families_equally(self) -> None:
        report = {
            "results": [
                {"task": {"metadata": {"benchmark_family": "a"}}, "score": {"success": True}},
                {"task": {"metadata": {"benchmark_family": "a"}}, "score": {"success": False}},
                {"task": {"metadata": {"benchmark_family": "b"}}, "score": {"success": True}},
            ]
        }

        self.assertEqual(family_macro_accuracy(report), 0.75)

    def test_contract_macro_accuracy_weights_contracts_equally(self) -> None:
        report = {
            "results": [
                {
                    "task": {"metadata": {"contract_tags": ["immutability", "numeric_filtering"]}},
                    "score": {"success": True},
                },
                {
                    "task": {"metadata": {"contract_tags": ["immutability"]}},
                    "score": {"success": False},
                },
                {
                    "task": {"metadata": {"contract_tags": ["stable_order"]}},
                    "score": {"success": True},
                },
            ]
        }

        breakdown = contract_breakdown(report)

        self.assertEqual(breakdown["immutability"]["passed"], 1)
        self.assertEqual(breakdown["immutability"]["total"], 2)
        self.assertAlmostEqual(contract_macro_accuracy(report), (0.5 + 1.0 + 1.0) / 3)

    def test_aggregate_reports_mean_and_sample_stddev(self) -> None:
        rows = [
            {
                "condition": "executive",
                "task_accuracy": 0.5,
                "family_macro_accuracy": 0.4,
                "contract_macro_accuracy": 0.25,
                "contract_breakdown": {"immutability": {"passed": 1, "total": 2, "accuracy": 0.5}},
                "duration_seconds": 2,
            },
            {
                "condition": "executive",
                "task_accuracy": 1.0,
                "family_macro_accuracy": 0.8,
                "contract_macro_accuracy": 0.75,
                "contract_breakdown": {"immutability": {"passed": 2, "total": 2, "accuracy": 1.0}},
                "duration_seconds": 3,
            },
        ]

        aggregate = aggregate_rows(rows)["executive"]

        self.assertEqual(aggregate["runs"], 2)
        self.assertEqual(aggregate["task_accuracy_mean"], 0.75)
        self.assertGreater(aggregate["task_accuracy_stddev"], 0)
        self.assertEqual(aggregate["contract_macro_mean"], 0.5)
        self.assertEqual(aggregate["contract_breakdown"]["immutability"]["passed"], 3)
        self.assertEqual(aggregate["contract_breakdown"]["immutability"]["total"], 4)
        self.assertEqual(aggregate["duration_seconds_total"], 5)

    def test_usage_report_excludes_target_agent_from_primary_scope(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ledger = Path(tmp) / "usage.jsonl"
            append_usage_event(
                ledger,
                {
                    "kind": "optimizer_model_api",
                    "operation": "reflect",
                    "actual_total_tokens": 12,
                },
            )
            append_usage_event(
                ledger,
                {
                    "kind": "target_agent_cli",
                    "operation": "run_task",
                    "estimated_prompt_tokens": 100,
                },
            )

            report = build_usage_report([ledger], aggregate_stdout_chars=80)

        self.assertEqual(report["primary_scope"], "executor_io_proxy")
        self.assertFalse(report["actual_executor_tokens_available"])
        self.assertEqual(report["experiment_internal_usage"]["calls"], 1)
        self.assertEqual(report["experiment_internal_usage"]["actual_total_tokens"], 12)
        self.assertEqual(report["experiment_internal_usage"]["estimated_total_tokens"], 0)
        self.assertEqual(report["excluded_from_primary_usage"]["target_agent_event_count"], 1)

    def test_build_summary_persists_development_gate(self) -> None:
        manifest = {
            "benchmark": "coding-hidden-v2",
            "development_gate_criteria": {
                "best_baseline_margin": 0.05,
                "min_seed_wins": 2,
            },
        }
        rows = [
            summary_row("seed-a", "human_skill", 0.8),
            summary_row("seed-b", "human_skill", 0.8),
            summary_row("seed-c", "human_skill", 0.8),
            summary_row("seed-a", "executive", 0.9),
            summary_row("seed-b", "executive", 0.9),
            summary_row("seed-c", "executive", 0.8),
        ]

        summary, aggregate_stdout = build_summary(manifest, rows)

        self.assertIn('"executive"', aggregate_stdout)
        self.assertIn("development_gate", summary)
        self.assertTrue(summary["development_gate"]["passed"])
        self.assertTrue(summary["locked_test_recommended"])
        self.assertEqual(summary["development_gate"]["best_baseline_condition"], "human_skill")
        self.assertEqual(summary["development_gate"]["required_seed_wins"], 2)

    def test_parse_conditions_rejects_unknown_condition(self) -> None:
        self.assertEqual(parse_conditions("executive,human_skill"), {"executive", "human_skill"})
        with self.assertRaises(ValueError):
            parse_conditions("executive,unknown")

    def test_cached_baselines_merge_for_executive_only_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            baseline_rows = [
                summary_row("seed-a", "human_skill", 0.8),
                summary_row("seed-b", "human_skill", 0.8),
                summary_row("seed-c", "human_skill", 0.8),
                summary_row("seed-a", "one_shot", 0.7),
                summary_row("seed-b", "one_shot", 0.7),
                summary_row("seed-c", "one_shot", 0.7),
                summary_row("seed-a", "executive", 0.1),
            ]
            baseline_summary = tmp_path / "baseline_summary.json"
            baseline_summary.write_text(json.dumps({"rows": baseline_rows}), encoding="utf-8")
            current_rows = [
                summary_row("seed-a", "executive", 0.9),
                summary_row("seed-b", "executive", 0.9),
                summary_row("seed-c", "executive", 0.8),
            ]

            rows = merge_cached_baseline_rows(
                current_rows,
                baseline_summary,
                ["seed-a", "seed-b", "seed-c"],
                {"executive"},
            )

        self.assertEqual(len(rows), 9)
        self.assertTrue(all(row["cached_baseline"] for row in rows[:6]))
        self.assertEqual({row["condition"] for row in rows}, {"human_skill", "one_shot", "executive"})
        self.assertTrue(any(row["condition"] == "executive" and not row.get("cached_baseline") for row in rows))

    def test_manifest_keeps_candidate_gate_and_development_gate_criteria_separate(self) -> None:
        args = Namespace(
            seed_workers=3,
            conditions={"executive"},
            baseline_summary=Path("runs/baseline/summary.json"),
            train_task_ids={"coding-hidden-v2-train-allocation-1"},
            selection_task_ids={"coding-hidden-v2-selection-allocation-2"},
            task_contracts={"largest_remainder"},
            epochs=1,
            rollout_batch_size=1,
            reflection_minibatch_size=1,
            learning_rate=2,
            learning_rate_floor=1,
            learning_rate_schedule="constant",
            slow_update_sample_size=1,
            disable_slow_update=True,
            task_retries=1,
            retry_backoff_seconds=5,
            validation_confirmation_rounds=0,
            validation_required_wins=1,
            validation_mean_delta=0.05,
            development_gate_required_wins=2,
            development_gate_mean_delta=0.05,
            early_stop_rejection_limit=1,
            task_limit=1,
        )

        manifest = build_manifest(args, ["seed-a", "seed-b", "seed-c"], optimizer_model="editor")

        self.assertEqual(manifest["conditions"], ["executive"])
        self.assertEqual(manifest["optimizer_config"]["validation_required_wins"], 1)
        self.assertFalse(manifest["optimizer_config"]["enable_slow_update"])
        self.assertEqual(manifest["development_gate_criteria"]["min_seed_wins"], 2)
        self.assertEqual(manifest["experiment_stage"], "mechanism_smoke")
        self.assertEqual(manifest["baseline_summary"], "runs/baseline/summary.json")
        self.assertEqual(manifest["task_filter"]["train_task_ids"], ["coding-hidden-v2-train-allocation-1"])
        self.assertEqual(manifest["task_filter"]["selection_task_ids"], ["coding-hidden-v2-selection-allocation-2"])
        self.assertEqual(manifest["task_filter"]["task_contracts"], ["largest_remainder"])

    def test_filter_tasks_by_ids_and_contracts(self) -> None:
        tasks = [
            task("a", ["stable_order"]),
            task("b", ["largest_remainder", "input_validation"]),
            task("c", ["immutability"]),
        ]

        self.assertEqual(parse_csv_set("a,b,, "), {"a", "b"})
        by_id = filter_tasks(tasks, task_ids={"b"}, contract_tags=set(), split_name="train")
        by_contract = filter_tasks(
            tasks,
            task_ids=set(),
            contract_tags={"largest_remainder"},
            split_name="train",
        )

        self.assertEqual([item.id for item in by_id], ["b"])
        self.assertEqual([item.id for item in by_contract], ["b"])
        with self.assertRaises(ValueError):
            filter_tasks(tasks, task_ids={"missing"}, contract_tags=set(), split_name="train")
        with self.assertRaises(ValueError):
            filter_tasks(tasks, task_ids=set(), contract_tags={"unicode_casefold"}, split_name="train")

    def test_requires_at_least_three_seed_labels(self) -> None:
        with self.assertRaises(ValueError):
            parse_seeds("a,b")
        self.assertEqual(parse_seeds("a,b,c"), ["a", "b", "c"])

    def test_detects_coco_model_without_mutating_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp) / "traecli.yaml"
            original = "model:\n  name: local-coco-model\nother: value\n"
            config.write_text(original, encoding="utf-8")

            self.assertEqual(detect_coco_model(config), "local-coco-model")
            self.assertEqual(config.read_text(encoding="utf-8"), original)

    def test_external_optimizer_requires_base_url_and_model(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ValueError):
                require_external_editor_config()
        with patch.dict(
            os.environ,
            {"EXTERNAL_LLM_BASE_URL": "https://example.test/v1", "EXTERNAL_LLM_MODEL": "editor"},
            clear=True,
        ):
            self.assertEqual(require_external_editor_config(), "editor")

    def test_run_seeds_preserves_seed_order_with_parallel_workers(self) -> None:
        args = Namespace(seed_workers=3)

        with patch(
            "work.run_coding_hidden_v2_matrix.run_seed",
            side_effect=lambda _args, seed, _train, _selection: [{"seed": seed}],
        ):
            rows = run_seeds(args, ["a", "b", "c"], [], [])

        self.assertEqual([row["seed"] for row in rows], ["a", "b", "c"])

    def test_run_seeds_clamps_worker_count(self) -> None:
        args = Namespace(seed_workers=99)

        with patch(
            "work.run_coding_hidden_v2_matrix.run_seed",
            side_effect=lambda _args, seed, _train, _selection: [{"seed": seed}],
        ):
            rows = run_seeds(args, ["a", "b", "c"], [], [])

        self.assertEqual(len(rows), 3)

    def test_matrix_records_persistent_anomalies_instead_of_aborting(self) -> None:
        args = Namespace(
            task_retries=2,
            retry_backoff_seconds=0,
            epochs=2,
            rollout_batch_size=10,
            reflection_minibatch_size=5,
            learning_rate=4,
            learning_rate_floor=2,
            learning_rate_schedule="cosine",
            rejected_buffer_limit=20,
            slow_update_sample_size=3,
            disable_slow_update=False,
            validation_confirmation_rounds=2,
            validation_required_wins=2,
            validation_mean_delta=0.05,
            early_stop_rejection_limit=3,
        )

        baseline = build_baseline_evaluator(
            args,
            scorer=object(),
            usage_path=Path("usage.jsonl"),
            usage_context={},
        )
        executive = build_executive_config(args, "seed-a")

        self.assertFalse(baseline.config.fail_on_persistent_task_anomaly)
        self.assertFalse(executive.fail_on_persistent_task_anomaly)
        self.assertTrue(executive.enable_slow_update)
        self.assertEqual(executive.early_stop_rejection_limit, 3)


def summary_row(seed: str, condition: str, score: float) -> dict:
    return {
        "seed": seed,
        "condition": condition,
        "task_accuracy": score,
        "family_macro_accuracy": score,
        "contract_macro_accuracy": score,
        "contract_breakdown": {"immutability": {"passed": int(score > 0), "total": 1, "accuracy": score}},
        "duration_seconds": 1.0,
    }


def task(task_id: str, contract_tags: list[str]) -> Task:
    return Task(
        id=task_id,
        input="fix it",
        expected={"tests_passed": True},
        metadata={"contract_tags": contract_tags},
    )


if __name__ == "__main__":
    unittest.main()
