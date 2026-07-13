import json
import tempfile
import unittest
from pathlib import Path

from work.skillopt_locked_preflight import build_locked_preflight_report
from work.skillopt_manifest_builder import build_manifest, build_parser, validate_manifest
from work.skillopt_preflight import build_preflight_report
from work.skillopt_stage_policy import validate_manifest_stage_policy
from work.skillopt_workflow import write_post_run_artifacts


class SkillOptMechanicalToolsTests(unittest.TestCase):
    def test_stage_policy_requires_confirmation_rounds_for_scale_up(self) -> None:
        manifest = valid_manifest(
            stage="full_selection_development",
            command=[
                "python3",
                "work/run_coding_hidden_v2_matrix.py",
                "--out",
                "runs/full",
                "--conditions",
                "executive",
                "--baseline-summary",
                "runs/baseline/summary.json",
                "--validation-confirmation-rounds",
                "0",
            ],
        )

        issues = validate_manifest_stage_policy(manifest)

        self.assertIn("confirmation_rounds_required", {item["code"] for item in issues})

    def test_manifest_builder_derives_targeted_critical_contracts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            baseline = Path(tmp) / "summary.json"
            baseline.write_text('{"rows":[],"aggregate":{}}\n', encoding="utf-8")
            args = build_parser().parse_args(
                [
                    "--stage",
                    "mechanism_smoke",
                    "--out",
                    str(Path(tmp) / "manifest.json"),
                    "--run-dir",
                    str(Path(tmp) / "run"),
                    "--selection-task-ids",
                    "coding-hidden-v2-selection-allocation-2",
                    "--train-task-ids",
                    "coding-hidden-v2-train-allocation-1",
                    "--baseline-summary",
                    str(baseline),
                    "--external-llm-base-url",
                    "https://example.invalid",
                    "--external-llm-model",
                    "model",
                ]
            )

            manifest = build_manifest(args)
            validate_manifest(manifest)

        self.assertEqual(manifest["experiment_stage"], "mechanism_smoke")
        self.assertEqual(
            manifest["acceptance"]["critical_contracts"],
            ["input_validation", "largest_remainder", "stable_order"],
        )
        self.assertIn("EXTERNAL_LLM_API_KEY", manifest["env_passthrough"])

    def test_manifest_builder_uses_stage_timeout_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            baseline = Path(tmp) / "summary.json"
            baseline.write_text('{"rows":[],"aggregate":{}}\n', encoding="utf-8")
            full_args = build_parser().parse_args(
                [
                    "--stage",
                    "full_selection_development",
                    "--out",
                    str(Path(tmp) / "full_manifest.json"),
                    "--run-dir",
                    str(Path(tmp) / "full_run"),
                    "--baseline-summary",
                    str(baseline),
                ]
            )
            smoke_args = build_parser().parse_args(
                [
                    "--stage",
                    "mechanism_smoke",
                    "--out",
                    str(Path(tmp) / "smoke_manifest.json"),
                    "--run-dir",
                    str(Path(tmp) / "smoke_run"),
                    "--baseline-summary",
                    str(baseline),
                ]
            )

            full_manifest = build_manifest(full_args)
            smoke_manifest = build_manifest(smoke_args)

        self.assertEqual(full_manifest["timeout_seconds"], 43200)
        self.assertEqual(smoke_manifest["timeout_seconds"], 7200)
        self.assertIn("--early-stop-validation-score", full_manifest["command"])
        self.assertIn("1.0", full_manifest["command"])
        self.assertNotIn("--early-stop-validation-score", smoke_manifest["command"])

    def test_preflight_reports_missing_key_and_old_command_python(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            baseline = Path(tmp) / "summary.json"
            baseline.write_text('{"rows":[],"aggregate":{}}\n', encoding="utf-8")
            manifest = valid_manifest(
                stage="mechanism_smoke",
                command=[
                    "python3",
                    "work/run_coding_hidden_v2_matrix.py",
                    "--out",
                    str(Path(tmp) / "run"),
                    "--conditions",
                    "executive",
                    "--baseline-summary",
                    str(baseline),
                    "--validation-confirmation-rounds",
                    "0",
                ],
                out_dir=str(Path(tmp) / "run"),
            )
            manifest["env"] = {
                "EXTERNAL_LLM_BASE_URL": "https://example.invalid",
                "EXTERNAL_LLM_MODEL": "model",
            }
            manifest_path = Path(tmp) / "manifest.json"
            write_json(manifest_path, manifest)

            report = build_preflight_report(
                manifest_path,
                environ={},
                current_python_version=(3, 10, 20),
                command_python_version=(3, 9, 6),
            )

        self.assertEqual(report["status"], "fail")
        failed = {item["name"] for item in report["checks"] if not item["passed"]}
        self.assertIn("env_passthrough:EXTERNAL_LLM_API_KEY", failed)
        self.assertIn("command_python_version", failed)

    def test_locked_preflight_allows_only_complete_evidence_without_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            write_locked_ready_artifacts(run_dir)

            allowed = build_locked_preflight_report(run_dir)
            (run_dir / "locked_receipt.json").write_text("{}", encoding="utf-8")
            blocked = build_locked_preflight_report(run_dir)

        self.assertEqual(allowed["status"], "allowed")
        self.assertEqual(blocked["status"], "blocked")
        self.assertIn("locked_receipt_absent", blocked["missing_evidence"])

    def test_workflow_report_writes_decision_and_compact_status(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            write_smoke_pass_artifacts(run_dir)

            report = write_post_run_artifacts(run_dir)
            decision = json.loads((run_dir / "decision.json").read_text(encoding="utf-8"))
            smoke_exists = (run_dir / "smoke_gate_report.json").exists()
            effect_exists = (run_dir / "contract_effect_audit.json").exists()
            failure_exists = (run_dir / "failure_delta_report.json").exists()
            compact_exists = (run_dir / "compact_status.txt").exists()

        self.assertEqual(report["decision"]["status"], "scale_up_candidate")
        self.assertTrue(smoke_exists)
        self.assertTrue(effect_exists)
        self.assertTrue(failure_exists)
        self.assertTrue(compact_exists)
        self.assertTrue(decision["decision"]["scale_up_allowed"])


def valid_manifest(
    *,
    stage: str,
    command: list[str],
    out_dir: str = "runs/full",
) -> dict:
    return {
        "schema_version": 1,
        "experiment_type": "coding_hidden_v2_matrix",
        "experiment_stage": stage,
        "runner_role": "mechanical_execution_only",
        "out_dir": out_dir,
        "command": command,
        "immutable_controls": {"do_not_change_coco_model": True},
        "env_passthrough": ["EXTERNAL_LLM_API_KEY"],
    }


def write_locked_ready_artifacts(run_dir: Path) -> None:
    write_json(
        run_dir / "runner_report.json",
        {
            "status": "complete",
            "development_gate": {"passed": True, "blocked_reason": ""},
            "anomaly_summary": {"persistent_anomaly_count": 0},
        },
    )
    write_json(
        run_dir / "summary.json",
        {
            "aggregate": {
                "executive": {
                    "experiment_internal_usage_summary": {
                        "actual_token_events": 1,
                        "actual_total_tokens": 100,
                    }
                }
            }
        },
    )
    write_json(run_dir / "smoke_gate_report.json", {"status": "pass", "reason": "ok"})
    write_json(run_dir / "contract_effect_audit.json", {"status": "pass", "reason": "ok"})


def write_smoke_pass_artifacts(run_dir: Path) -> None:
    executive_dir = run_dir / "seed-a" / "executive"
    executive_dir.mkdir(parents=True)
    summary = {
        "manifest": {"benchmark": "coding-hidden-v2"},
        "rows": [
            {"seed": "seed-a", "condition": "human_skill", "task_accuracy": 0.5},
            {
                "seed": "seed-a",
                "condition": "executive",
                "task_accuracy": 1.0,
                "accepted_steps": 1,
            },
        ],
        "aggregate": {
            "executive": {
                "task_accuracy_mean": 1.0,
                "experiment_internal_usage_summary": {
                    "actual_token_events": 1,
                    "actual_total_tokens": 100,
                },
            },
            "human_skill": {"task_accuracy_mean": 0.5},
        },
        "development_gate": {
            "passed": True,
            "locked_test_recommended": True,
            "best_baseline_condition": "human_skill",
            "blocked_reason": "",
        },
    }
    write_json(run_dir / "summary.json", summary)
    write_json(
        run_dir / "runner_report.json",
        {
            "status": "complete",
            "development_gate": summary["development_gate"],
            "anomaly_summary": {"persistent_anomaly_count": 0},
        },
    )
    append_jsonl(
        executive_dir / "proposals.jsonl",
        {
            "optimizer_controls": {"phase": "reflection", "epoch": 1, "batch_index": 1},
            "proposal_targeting_audit": {
                "required": True,
                "contract_rejection_evidence_available": True,
                "priority_contracts": ["stable_order"],
                "missing_targeted_contract_count": 0,
                "proposals": [
                    {
                        "name": "target-stable-order",
                        "targeted_contracts": ["stable_order"],
                        "protected_contracts": ["input_validation"],
                    }
                ],
            },
            "proposals": [
                {
                    "name": "target-stable-order",
                    "metadata": {
                        "targeted_contracts": ["stable_order"],
                        "protected_contracts": ["input_validation"],
                        "evidence_source": "contract_rejection_evidence",
                    },
                }
            ],
        },
    )
    write_json(
        executive_dir / "result.json",
        {
            "history": [
                {"candidate": "initial", "epoch": 0, "accepted": True, "metadata": {}},
                {
                    "candidate": "atomic-epoch-1-batch-1",
                    "epoch": 1,
                    "accepted": True,
                    "validation_score": 1.0,
                    "metadata": {"phase": "fast_update", "step": 1, "selected_edits": []},
                },
            ]
        },
    )
    write_json(
        executive_dir / "selection_atomic-epoch-1-batch-1_gate.json",
        {
            "accepted": True,
            "current_mean": 0.5,
            "candidate_mean": 1.0,
            "contract_evidence": {
                "contract_deltas": {
                    "stable_order": contract_delta("stable_order", 0.5),
                    "input_validation": contract_delta("input_validation", 0.0),
                }
            },
        },
    )


def contract_delta(contract: str, delta: float) -> dict:
    return {
        "contract": contract,
        "current_accuracy": 0.5,
        "candidate_accuracy": 0.5 + delta,
        "delta": delta,
    }


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def append_jsonl(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
