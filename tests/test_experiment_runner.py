import json
import sys
import tempfile
import textwrap
import time
import unittest
from pathlib import Path

from work.experiment_runner import (
    build_runner_report,
    main,
    validate_runner_manifest,
)


class ExperimentRunnerTests(unittest.TestCase):
    def test_runner_report_blocks_locked_test_when_dev_criteria_fail(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            write_summary(
                run_dir,
                rows=[
                    row("seed-a", "human_skill", 0.8),
                    row("seed-b", "human_skill", 0.8),
                    row("seed-c", "human_skill", 0.8),
                    row("seed-a", "one_shot", 0.8),
                    row("seed-b", "one_shot", 0.8),
                    row("seed-c", "one_shot", 0.8),
                    row("seed-a", "executive", 0.8, accepted_steps=0),
                    row("seed-b", "executive", 0.8, accepted_steps=0),
                    row("seed-c", "executive", 0.9, accepted_steps=1),
                ],
            )
            write_json(
                run_dir / "seed-c" / "executive" / "result.json",
                {
                    "final_validation_report": {
                        "results": [
                            {
                                "output": {
                                    "metadata": {
                                        "retry_policy": {
                                            "attempt_count": 2,
                                            "persistent_anomaly": False,
                                        }
                                    }
                                }
                            }
                        ]
                    }
                },
            )

            report = build_runner_report(run_dir)

        self.assertEqual(report["status"], "complete")
        self.assertIn("development_gate", report)
        self.assertFalse(report["executive_decision"]["criteria_met"])
        self.assertFalse(report["development_gate"]["passed"])
        self.assertFalse(report["decision"]["locked_test_recommended"])
        self.assertEqual(report["executive_decision"]["seed_wins_vs_best_baseline"], 1)
        self.assertEqual(report["anomaly_summary"]["retry_count"], 1)
        self.assertEqual(report["anomaly_summary"]["persistent_anomaly_count"], 0)
        self.assertIn("contract_macro_accuracy", report["seed_rows"][0])
        self.assertIn("contract_macro_mean", report["scores_by_condition"]["executive"])
        self.assertIn("contract_breakdown", report["scores_by_condition"]["executive"])

    def test_runner_report_allows_locked_test_when_criteria_pass_and_no_anomaly(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            write_summary(
                run_dir,
                rows=[
                    row("seed-a", "human_skill", 0.8),
                    row("seed-b", "human_skill", 0.8),
                    row("seed-c", "human_skill", 0.8),
                    row("seed-a", "executive", 0.9, accepted_steps=1),
                    row("seed-b", "executive", 0.9, accepted_steps=1),
                    row("seed-c", "executive", 0.8, accepted_steps=0),
                ],
            )

            report = build_runner_report(run_dir)

        self.assertTrue(report["executive_decision"]["criteria_met"])
        self.assertTrue(report["development_gate"]["passed"])
        self.assertTrue(report["decision"]["locked_test_recommended"])

    def test_runner_report_uses_summary_development_gate_when_present(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            summary = minimal_summary(
                [
                    row("seed-a", "human_skill", 0.8),
                    row("seed-b", "human_skill", 0.8),
                    row("seed-c", "human_skill", 0.8),
                    row("seed-a", "executive", 0.9),
                    row("seed-b", "executive", 0.9),
                    row("seed-c", "executive", 0.9),
                ]
            )
            summary["development_gate"] = {
                "schema_version": 1,
                "criteria": {"best_baseline_margin": 0.05, "min_seed_wins": 2},
                "best_baseline_condition": "human_skill",
                "best_baseline_score": 0.8,
                "executive_score": 0.9,
                "score_delta": 0.1,
                "required_delta": 0.05,
                "seed_wins_vs_best_baseline": 3,
                "required_seed_wins": 2,
                "passed": False,
                "criteria_met": False,
                "locked_test_recommended": False,
                "blocked_reasons": ["forced summary gate for compatibility test"],
                "blocked_reason": "forced summary gate for compatibility test",
            }
            write_json(run_dir / "summary.json", summary)

            report = build_runner_report(run_dir)

        self.assertFalse(report["development_gate"]["passed"])
        self.assertFalse(report["decision"]["locked_test_recommended"])
        self.assertEqual(report["decision"]["reason"], "forced summary gate for compatibility test")

    def test_runner_report_blocks_on_persistent_anomaly(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            write_summary(
                run_dir,
                rows=[
                    row("seed-a", "human_skill", 0.8),
                    row("seed-b", "human_skill", 0.8),
                    row("seed-c", "human_skill", 0.8),
                    row("seed-a", "executive", 0.9),
                    row("seed-b", "executive", 0.9),
                    row("seed-c", "executive", 0.8),
                ],
            )
            write_json(
                run_dir / "seed-a" / "executive" / "result.json",
                {
                    "output": {
                        "metadata": {
                            "retry_policy": {
                                "attempt_count": 3,
                                "persistent_anomaly": True,
                            }
                        }
                    }
                },
            )

            report = build_runner_report(run_dir)

        self.assertTrue(report["executive_decision"]["criteria_met"])
        self.assertFalse(report["decision"]["locked_test_recommended"])
        self.assertEqual(report["anomaly_summary"]["persistent_anomaly_count"], 1)

    def test_manifest_rejects_target_model_override(self) -> None:
        manifest = valid_manifest(command=["python3", "work/run_coding_hidden_v2_matrix.py", "--target_model", "x"])

        with self.assertRaises(ValueError):
            validate_runner_manifest(manifest)

    def test_manifest_rejects_zero_confirmation_without_mechanism_smoke_stage(self) -> None:
        manifest = valid_manifest(
            command=[
                "python3",
                "work/run_coding_hidden_v2_matrix.py",
                "--validation-confirmation-rounds",
                "0",
            ]
        )

        with self.assertRaises(ValueError):
            validate_runner_manifest(manifest)

    def test_manifest_allows_zero_confirmation_for_mechanism_smoke(self) -> None:
        manifest = valid_manifest(
            command=[
                "python3",
                "work/run_coding_hidden_v2_matrix.py",
                "--validation-confirmation-rounds",
                "0",
            ],
            experiment_stage="mechanism_smoke",
        )

        validate_runner_manifest(manifest)

    def test_coco_runner_task_is_mechanical_and_secret_free(self) -> None:
        cases = [
            (
                "work/coco_manifest_runner_task.json",
                "work/experiment_runner_manifest.example.json",
                "work/experiment_runner_manifest.example.json",
                "start",
                False,
            ),
            (
                "work/coco_manifest_runner_smoke_task.json",
                "work/experiment_runner_manifest.smoke.json",
                "work/experiment_runner_manifest.smoke.json",
                "run",
                True,
            ),
        ]
        validate_runner_manifest(
            json.loads(Path("work/experiment_runner_manifest.executive_smoke.json").read_text(encoding="utf-8"))
        )
        validate_runner_manifest(
            json.loads(
                Path("work/experiment_runner_manifest.targeted_executive_smoke.json").read_text(encoding="utf-8")
            )
        )
        for task_path, manifest_path, expected_manifest_arg, expected_verb, expect_task_limit in cases:
            with self.subTest(task_path=task_path):
                task = json.loads(Path(task_path).read_text(encoding="utf-8"))
                manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))

                validate_runner_manifest(manifest)
                self.assertEqual(task["role"], "mechanical_execution_only")
                self.assertEqual(
                    task["allowed_command"],
                    [
                        "python3",
                        "work/experiment_runner.py",
                        expected_verb,
                        "--manifest",
                        expected_manifest_arg,
                    ],
                )
                combined = json.dumps(task) + json.dumps(manifest)
                self.assertNotIn("b9e2", combined)
                self.assertNotIn("apiKey", combined)
                self.assertIn("EXTERNAL_LLM_API_KEY", manifest["env_passthrough"])
                self.assertEqual("--task-limit" in manifest["command"], expect_task_limit)
                self.assertIn("--resume", manifest["command"])

    def test_run_manifest_writes_report_and_captures_stdout(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            summary_payload = minimal_summary([row("seed-a", "human_skill", 0.8), row("seed-a", "executive", 0.9)])
            summary_text = json.dumps(summary_payload)
            script = (
                "import json; from pathlib import Path; "
                f"out=Path({str(run_dir)!r}); out.mkdir(parents=True, exist_ok=True); "
                f"(out/'summary.json').write_text({summary_text!r}); "
                "print('runner finished')"
            )
            manifest_path = Path(tmp) / "manifest.json"
            write_json(
                manifest_path,
                valid_manifest(
                    out_dir=str(run_dir),
                    command=[sys.executable, "-c", script],
                ),
            )

            exit_code = main(["run", "--manifest", str(manifest_path)])

            report = json.loads((run_dir / "runner_report.json").read_text(encoding="utf-8"))
            execution = json.loads((run_dir / "runner_execution.json").read_text(encoding="utf-8"))
            stdout = (run_dir / "runner_stdout.txt").read_text(encoding="utf-8")

        self.assertEqual(exit_code, 0)
        self.assertEqual(report["status"], "complete")
        self.assertEqual(execution["returncode"], 0)
        self.assertIn("runner finished", stdout)

    def test_start_manifest_returns_background_state_and_status_completes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            summary_payload = minimal_summary([row("seed-a", "human_skill", 0.8), row("seed-a", "executive", 0.9)])
            summary_text = json.dumps(summary_payload)
            script = (
                "from pathlib import Path; "
                f"out=Path({str(run_dir)!r}); out.mkdir(parents=True, exist_ok=True); "
                f"(out/'summary.json').write_text({summary_text!r}); "
                "print('background runner finished')"
            )
            manifest_path = Path(tmp) / "manifest.json"
            write_json(
                manifest_path,
                valid_manifest(
                    out_dir=str(run_dir),
                    command=[sys.executable, "-c", script],
                ),
            )

            exit_code = main(["start", "--manifest", str(manifest_path)])
            self.assertEqual(exit_code, 0)
            background = json.loads((run_dir / "runner_background.json").read_text(encoding="utf-8"))
            self.assertEqual(background["status"], "running")
            self.assertNotIn("EXTERNAL_LLM_API_KEY", json.dumps(background))

            report_path = run_dir / "runner_report.json"
            for _ in range(50):
                if report_path.exists():
                    break
                time.sleep(0.05)
            self.assertTrue(report_path.exists())

            status_path = Path(tmp) / "status.json"
            status_code = main(["status", "--run-dir", str(run_dir), "--out", str(status_path)])
            status = json.loads(status_path.read_text(encoding="utf-8"))

        self.assertEqual(status_code, 0)
        self.assertEqual(status["status"], "complete")
        self.assertEqual(status["returncode"], 0)

    def test_wait_manifest_returns_complete_for_background_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            summary_payload = minimal_summary([row("seed-a", "human_skill", 0.8), row("seed-a", "executive", 0.9)])
            summary_text = json.dumps(summary_payload)
            script = (
                "from pathlib import Path; "
                f"out=Path({str(run_dir)!r}); out.mkdir(parents=True, exist_ok=True); "
                f"(out/'summary.json').write_text({summary_text!r}); "
                "print('background runner finished')"
            )
            manifest_path = Path(tmp) / "manifest.json"
            write_json(
                manifest_path,
                valid_manifest(
                    out_dir=str(run_dir),
                    command=[sys.executable, "-c", script],
                ),
            )

            self.assertEqual(main(["start", "--manifest", str(manifest_path)]), 0)
            status_path = Path(tmp) / "wait-status.json"
            wait_code = main(
                [
                    "wait",
                    "--run-dir",
                    str(run_dir),
                    "--timeout-seconds",
                    "5",
                    "--poll-seconds",
                    "0.05",
                    "--out",
                    str(status_path),
                ]
            )
            status = json.loads(status_path.read_text(encoding="utf-8"))

        self.assertEqual(wait_code, 0)
        self.assertEqual(status["status"], "complete")
        self.assertFalse(status["timed_out"])

    def test_wait_manifest_times_out_when_not_started(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            status_path = Path(tmp) / "wait-status.json"

            wait_code = main(
                [
                    "wait",
                    "--run-dir",
                    str(run_dir),
                    "--timeout-seconds",
                    "0.05",
                    "--poll-seconds",
                    "0.01",
                    "--out",
                    str(status_path),
                ]
            )
            status = json.loads(status_path.read_text(encoding="utf-8"))

        self.assertEqual(wait_code, 124)
        self.assertEqual(status["status"], "not_started")
        self.assertTrue(status["timed_out"])

    def test_status_prefers_newer_running_background_over_stale_failed_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            write_json(
                run_dir / "runner_report.json",
                {"status": "failed", "decision": {"locked_test_recommended": False}},
            )
            write_json(run_dir / "runner_execution.json", {"returncode": 1})
            time.sleep(0.01)
            write_json(
                run_dir / "runner_background.json",
                {
                    "pid": 1,
                    "started_at": "2026-06-26T00:00:00+00:00",
                    "stdout_path": str(run_dir / "stdout.txt"),
                    "stderr_path": str(run_dir / "stderr.txt"),
                },
            )
            status_path = Path(tmp) / "status.json"

            status_code = main(["status", "--run-dir", str(run_dir), "--out", str(status_path)])
            status = json.loads(status_path.read_text(encoding="utf-8"))

        self.assertEqual(status_code, 0)
        self.assertEqual(status["status"], "running")
        self.assertEqual(status["pid"], 1)


def row(seed: str, condition: str, score: float, *, accepted_steps: int | None = None) -> dict:
    payload = {
        "seed": seed,
        "condition": condition,
        "task_accuracy": score,
        "average_score": score,
        "family_macro_accuracy": score,
        "contract_macro_accuracy": score,
        "contract_breakdown": {"immutability": {"passed": int(score > 0), "total": 1, "accuracy": score}},
        "duration_seconds": 1.0,
        "run_dir": f"runs/{seed}/{condition}",
    }
    if condition == "executive":
        payload["accepted_steps"] = accepted_steps
        payload["total_steps"] = 2
        payload["best_validation_score"] = score
    return payload


def write_summary(run_dir: Path, *, rows: list[dict]) -> None:
    write_json(run_dir / "summary.json", minimal_summary(rows))


def minimal_summary(rows: list[dict]) -> dict:
    return {
        "manifest": {
            "benchmark": "coding-hidden-v2",
            "development_only": True,
            "target_harness": "coco",
            "target_model": "DeepSeek-V4-Pro",
            "target_model_policy": "read-local-default-without-override",
            "optimizer_harness": "openai-compatible-external-editor",
            "optimizer_model": "optimizer-model",
            "seeds": sorted({item["seed"] for item in rows}),
            "conditions": sorted({item["condition"] for item in rows}),
        },
        "rows": rows,
        "aggregate": aggregate(rows),
        "usage": {"primary_scope": "executor_io_proxy"},
    }


def aggregate(rows: list[dict]) -> dict:
    output = {}
    for condition in sorted({item["condition"] for item in rows}):
        selected = [item for item in rows if item["condition"] == condition]
        mean = sum(item["task_accuracy"] for item in selected) / len(selected)
        output[condition] = {
            "runs": len(selected),
            "task_accuracy_mean": mean,
            "task_accuracy_stddev": 0.0,
            "family_macro_mean": mean,
            "family_macro_stddev": 0.0,
            "contract_macro_mean": mean,
            "contract_macro_stddev": 0.0,
            "contract_breakdown": {"immutability": {"passed": sum(1 for item in selected if item["task_accuracy"] > 0), "total": len(selected), "accuracy": mean}},
            "duration_seconds_total": len(selected),
        }
    return output


def valid_manifest(
    *,
    out_dir: str = "runs/test",
    command: list[str] | None = None,
    experiment_stage: str | None = None,
) -> dict:
    return {
        "schema_version": 1,
        "runner_role": "mechanical_execution_only",
        "experiment_type": "coding_hidden_v2_matrix",
        **({"experiment_stage": experiment_stage} if experiment_stage else {}),
        "out_dir": out_dir,
        "command": command or ["python3", "work/run_coding_hidden_v2_matrix.py", "--resume"],
        "immutable_controls": {"do_not_change_coco_model": True},
        "acceptance": {"best_baseline_margin": 0.05, "min_seed_wins": 2},
    }


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
