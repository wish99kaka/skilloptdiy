import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from work.skillopt_stage7 import (
    build_locked_environment,
    build_stage7_manifest,
    execute_stage7_manifest,
    main,
    select_locked_candidate,
    validate_stage7_manifest,
)


class SkillOptStage7Tests(unittest.TestCase):
    def test_locked_environment_removes_operator_overrides(self) -> None:
        environment = build_locked_environment(
            {
                "PATH": "/bin",
                "COCO_AGENT_BIN": "/tmp/not-coco",
                "COCO_AGENT_DRY_RUN": "1",
                "COCO_AGENT_EXTRA_ARGS": "--model changed",
                "COCO_TASK_LIMIT": "1",
            }
        )

        self.assertEqual(environment["PATH"], "/bin")
        self.assertEqual(environment["COCO_AGENT_TIMEOUT"], "360")
        self.assertNotIn("COCO_AGENT_BIN", environment)
        self.assertNotIn("COCO_AGENT_DRY_RUN", environment)
        self.assertNotIn("COCO_AGENT_EXTRA_ARGS", environment)
        self.assertNotIn("COCO_TASK_LIMIT", environment)

    def test_selects_shortest_skill_after_development_scores_tie(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            rows = []
            for seed, skill_text in (
                ("seed-a", "a" * 30),
                ("seed-b", "b" * 20),
                ("seed-c", "c" * 10),
            ):
                skill = run_dir / seed / "executive" / "best_skill.md"
                skill.parent.mkdir(parents=True)
                skill.write_text(skill_text, encoding="utf-8")
                rows.append(
                    {
                        "condition": "executive",
                        "seed": seed,
                        "average_score": 1.0,
                        "contract_macro_accuracy": 1.0,
                    }
                )
            (run_dir / "summary.json").write_text(
                json.dumps({"rows": rows}),
                encoding="utf-8",
            )

            selected = select_locked_candidate(run_dir)

        self.assertEqual(selected["seed"], "seed-c")
        self.assertEqual(selected["skill_bytes"], 10)
        self.assertEqual(len(selected["skill_sha256"]), 64)

    def test_manifest_pins_candidate_commitment_and_one_attempt_command(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = build_manifest_fixture(root)
            archive_sha256 = manifest["locked_test"]["archive_sha256"]
            key_sha256 = hashlib.sha256(
                Path(manifest["locked_test"]["key_file"]).read_bytes()
            ).hexdigest()

        self.assertEqual(manifest["experiment_stage"], "locked_test_once")
        self.assertEqual(manifest["selected_candidate"]["seed"], "seed-c")
        self.assertEqual(manifest["locked_test"]["archive_sha256"], archive_sha256)
        self.assertEqual(
            manifest["locked_test"]["key_file_sha256"],
            key_sha256,
        )
        self.assertEqual(manifest["locked_test"]["task_file"], "test.jsonl")
        self.assertTrue(manifest["locked_test"]["usage_ledger"].endswith("locked_usage_ledger.jsonl"))
        self.assertTrue(manifest["locked_test"]["receipt"].endswith("locked_receipt.json"))
        self.assertEqual(
            manifest["runtime"]["python_executable"],
            str(Path(sys.executable).resolve()),
        )
        self.assertEqual(manifest["command"][0], manifest["runtime"]["python_executable"])
        self.assertEqual(
            manifest["execution_policy"],
            {"attempts": 1, "task_retries": 1, "whole_command_timeout": None},
        )
        self.assertIn("work/run_coding_hidden_v2_locked_eval.py", manifest["command"])
        self.assertEqual(
            manifest["code_commitments"]["work/run_coding_hidden_v2_locked_eval.py"],
            hashlib.sha256(
                Path("work/run_coding_hidden_v2_locked_eval.py").read_bytes()
            ).hexdigest(),
        )
        self.assertEqual(
            manifest["development_evidence"],
            {
                "development_gate": {
                    "passed": True,
                    "executive_mean": 1.0,
                    "best_baseline_condition": "human_skill",
                    "best_baseline_mean": 0.8,
                    "mean_delta": 0.2,
                    "seed_wins": 3,
                },
                "condition_means": {
                    "executive": 1.0,
                    "human_skill": 0.8,
                    "no_skill": 0.6,
                    "one_shot": 0.7,
                },
                "optimizer_api_tokens": {"total": 110, "executive": 100, "one_shot": 10},
            },
        )

    def test_validation_blocks_candidate_changed_after_manifest_creation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manifest = build_manifest_fixture(Path(tmp))
            selected_skill = Path(manifest["selected_candidate"]["skill_path"])
            selected_skill.write_text("changed after approval", encoding="utf-8")

            report = validate_stage7_manifest(manifest)

        self.assertEqual(report["status"], "blocked")
        self.assertIn("selected_candidate_unchanged", report["failed_checks"])

    def test_validation_blocks_changed_development_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manifest = build_manifest_fixture(Path(tmp))
            manifest["development_evidence"]["development_gate"]["mean_delta"] = 999

            report = validate_stage7_manifest(manifest)

        self.assertEqual(report["status"], "blocked")
        self.assertIn("development_evidence_unchanged", report["failed_checks"])

    def test_consumes_once_only_after_exact_confirmation_and_writes_final_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manifest = build_manifest_fixture(Path(tmp))
            calls = []

            def run_command(command, *, cwd, check, env):
                self.assertTrue(Path(manifest["locked_test"]["attempt"]).is_file())
                self.assertEqual(env["COCO_AGENT_TIMEOUT"], "360")
                calls.append((command, cwd, check))
                locked = manifest["locked_test"]
                Path(locked["receipt"]).write_text(
                    json.dumps({"archive_sha256": locked["archive_sha256"], "returncode": 0}),
                    encoding="utf-8",
                )
                Path(locked["usage_ledger"]).write_text("{}\n", encoding="utf-8")
                Path(locked["result"]).write_text(
                    json.dumps(
                        {
                            "status": "complete",
                            "skill_sha256": manifest["selected_candidate"]["skill_sha256"],
                            "task_count": 20,
                            "task_accuracy": 0.9,
                            "family_macro_accuracy": 0.9,
                            "contract_macro_accuracy": 0.8,
                            "usage_ledger_path": locked["usage_ledger"],
                            "task_results": [{"task_id": "locked-task", "success": True}],
                        }
                    ),
                    encoding="utf-8",
                )
                return SimpleNamespace(returncode=0)

            with self.assertRaisesRegex(ValueError, "CONSUME_LOCKED_TEST_ONCE"):
                execute_stage7_manifest(manifest, confirmation="", run_command=run_command)
            final = execute_stage7_manifest(
                manifest,
                confirmation="CONSUME_LOCKED_TEST_ONCE",
                run_command=run_command,
            )
            attempt = json.loads(Path(manifest["locked_test"]["attempt"]).read_text(encoding="utf-8"))

        self.assertEqual(len(calls), 1)
        self.assertEqual(attempt["status"], "started")
        self.assertEqual(final["status"], "complete")
        self.assertTrue(final["execution"]["checks"]["usage_ledger_present"])
        self.assertEqual(final["locked_result"]["task_accuracy"], 0.9)
        self.assertNotIn("task_results", final["locked_result"])

    def test_check_cli_writes_ready_report_without_consuming_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = build_manifest_fixture(root)
            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            report_path = root / "readiness.json"

            returncode = main(
                ["check", "--manifest", str(manifest_path), "--out", str(report_path), "--quiet"]
            )

            report = json.loads(report_path.read_text(encoding="utf-8"))
            receipt_exists = Path(manifest["locked_test"]["receipt"]).exists()

        self.assertEqual(returncode, 0)
        self.assertEqual(report["status"], "ready")
        self.assertFalse(receipt_exists)


def write_ready_run(run_dir: Path) -> None:
    run_dir.mkdir(parents=True)
    rows = []
    for seed, size in (("seed-a", 30), ("seed-b", 20), ("seed-c", 10)):
        skill = run_dir / seed / "executive" / "best_skill.md"
        skill.parent.mkdir(parents=True)
        skill.write_text(seed[-1] * size, encoding="utf-8")
        rows.append(
            {
                "condition": "executive",
                "seed": seed,
                "task_accuracy": 1.0,
                "average_score": 1.0,
                "contract_macro_accuracy": 1.0,
            }
        )
    (run_dir / "summary.json").write_text(
        json.dumps(
            {
                "rows": rows,
                "aggregate": {
                    "executive": {
                        "task_accuracy_mean": 1.0,
                        "experiment_internal_usage_summary": {
                            "actual_token_events": 1,
                            "actual_total_tokens": 100,
                        }
                    },
                    "human_skill": {"task_accuracy_mean": 0.8},
                    "no_skill": {"task_accuracy_mean": 0.6},
                    "one_shot": {
                        "task_accuracy_mean": 0.7,
                        "experiment_internal_usage_summary": {"actual_total_tokens": 10},
                    },
                },
                "development_gate": {
                    "passed": True,
                    "executive_mean": 1.0,
                    "best_baseline_condition": "human_skill",
                    "best_baseline_mean": 0.8,
                    "mean_delta": 0.2,
                    "seed_wins_vs_best_baseline": 3,
                },
                "usage": {"experiment_internal_usage": {"actual_total_tokens": 110}},
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "runner_report.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "development_gate": {"passed": True, "blocked_reason": ""},
                "anomaly_summary": {"persistent_anomaly_count": 0},
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "smoke_gate_report.json").write_text(
        json.dumps({"status": "pass", "reason": "ok"}),
        encoding="utf-8",
    )
    (run_dir / "contract_effect_audit.json").write_text(
        json.dumps({"status": "pass", "reason": "ok"}),
        encoding="utf-8",
    )


def build_manifest_fixture(root: Path) -> dict:
    run_dir = root / "run"
    write_ready_run(run_dir)
    archive = root / "test.enc"
    archive.write_bytes(b"sealed-test")
    archive_sha256 = hashlib.sha256(archive.read_bytes()).hexdigest()
    lock = root / "test.lock.json"
    lock.write_text(
        json.dumps(
            {
                "format": "textskill-fernet-tar-gz-v1",
                "archive_sha256": archive_sha256,
                "details": {"task_count": 20, "task_file": "test.jsonl"},
            }
        ),
        encoding="utf-8",
    )
    key = root / "test.key"
    key.write_text("external-key", encoding="utf-8")
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    summary["manifest"] = {"locked_test_sha256": archive_sha256}
    (run_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    return build_stage7_manifest(
        run_dir,
        key_file=key,
        archive=archive,
        lock_file=lock,
    )


if __name__ == "__main__":
    unittest.main()
