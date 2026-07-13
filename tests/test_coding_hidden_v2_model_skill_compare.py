import json
import shlex
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from work.run_coding_hidden_v2_model_skill_compare import (
    aggregate_rows,
    build_agent_extra_args,
    build_comparisons,
    main,
    parse_seeds,
)


class CodingHiddenV2ModelSkillCompareTests(unittest.TestCase):
    def test_parse_seeds_requires_at_least_one_label(self) -> None:
        self.assertEqual(parse_seeds(" seed-a , seed-b "), ["seed-a", "seed-b"])
        with self.assertRaises(ValueError):
            parse_seeds(" , ")

    def test_build_agent_extra_args_appends_model_and_rejects_ambiguous_override(self) -> None:
        extra_args = build_agent_extra_args("--output-format text", "strong-model")

        self.assertEqual(
            shlex.split(extra_args),
            ["--output-format", "text", "--model", "strong-model"],
        )
        with self.assertRaises(ValueError):
            build_agent_extra_args("--model old-model", "strong-model")
        with self.assertRaises(ValueError):
            build_agent_extra_args("--model=old-model", "strong-model")

    def test_aggregate_rows_and_comparisons_report_proxy_cost_deltas(self) -> None:
        rows = [
            {
                "model_label": "strong",
                "condition": "no_skill",
                "pass_rate": 0.6,
                "duration_seconds": 10.0,
                "calls": 5,
                "estimated_prompt_tokens": 60,
                "estimated_completion_tokens": 40,
                "estimated_total_tokens": 100,
            },
            {
                "model_label": "strong",
                "condition": "no_skill",
                "pass_rate": 0.8,
                "duration_seconds": 14.0,
                "calls": 5,
                "estimated_prompt_tokens": 90,
                "estimated_completion_tokens": 50,
                "estimated_total_tokens": 140,
            },
            {
                "model_label": "weak",
                "condition": "no_skill",
                "pass_rate": 0.2,
                "duration_seconds": 20.0,
                "calls": 5,
                "estimated_prompt_tokens": 120,
                "estimated_completion_tokens": 80,
                "estimated_total_tokens": 200,
            },
        ]

        aggregate = aggregate_rows(rows)
        comparisons = build_comparisons(aggregate, strong_label="strong", weak_label="weak")

        self.assertEqual(aggregate["strong"]["no_skill"]["runs"], 2)
        self.assertAlmostEqual(aggregate["strong"]["no_skill"]["pass_rate_mean"], 0.7)
        self.assertAlmostEqual(aggregate["strong"]["no_skill"]["duration_seconds_total"], 24.0)
        self.assertAlmostEqual(aggregate["strong"]["no_skill"]["calls_mean"], 5.0)
        self.assertEqual(aggregate["strong"]["no_skill"]["estimated_total_tokens_total"], 240)
        self.assertAlmostEqual(comparisons["no_skill"]["pass_rate_mean_delta"], 0.5)
        self.assertAlmostEqual(comparisons["no_skill"]["duration_seconds_mean_delta"], -8.0)
        self.assertEqual(comparisons["no_skill"]["estimated_total_tokens_mean_delta"], -80.0)

    def test_main_writes_summary_with_manifest_aggregate_and_comparisons(self) -> None:
        fake_rows = [
            {
                "seed": "seed-a",
                "model_label": "strong",
                "model_name": "strong-model",
                "condition": "no_skill",
                "pass_rate": 0.9,
                "average_score": 0.9,
                "duration_seconds": 11.0,
                "calls": 10,
                "estimated_prompt_tokens": 100,
                "estimated_completion_tokens": 40,
                "estimated_total_tokens": 140,
                "run_dir": "/tmp/strong/no_skill",
                "report_path": "/tmp/strong/no_skill/report.json",
                "usage_ledger_path": "/tmp/strong/no_skill/usage_ledger.jsonl",
            },
            {
                "seed": "seed-a",
                "model_label": "weak",
                "model_name": "weak-model",
                "condition": "no_skill",
                "pass_rate": 0.4,
                "average_score": 0.4,
                "duration_seconds": 17.0,
                "calls": 10,
                "estimated_prompt_tokens": 140,
                "estimated_completion_tokens": 60,
                "estimated_total_tokens": 200,
                "run_dir": "/tmp/weak/no_skill",
                "report_path": "/tmp/weak/no_skill/report.json",
                "usage_ledger_path": "/tmp/weak/no_skill/usage_ledger.jsonl",
            },
        ]

        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "run"
            prepared_tasks = Path(tmp) / "prepared_tasks.jsonl"
            prepared_tasks.write_text("", encoding="utf-8")
            with patch(
                "work.run_coding_hidden_v2_model_skill_compare.prepare_tasks",
                return_value=(prepared_tasks, []),
            ), patch(
                "work.run_coding_hidden_v2_model_skill_compare.run_evaluations",
                return_value=fake_rows,
            ):
                exit_code = main(
                    [
                        "--out",
                        str(out_dir),
                        "--strong-model",
                        "strong-model",
                        "--weak-model",
                        "weak-model",
                    ]
                )

            summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))

        self.assertEqual(exit_code, 0)
        self.assertEqual(summary["manifest"]["models"]["strong"]["name"], "strong-model")
        self.assertEqual(summary["aggregate"]["strong"]["no_skill"]["runs"], 1)
        self.assertAlmostEqual(summary["comparisons"]["no_skill"]["pass_rate_mean_delta"], 0.5)


if __name__ == "__main__":
    unittest.main()
