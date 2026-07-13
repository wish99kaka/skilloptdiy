import unittest
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

from work.recover_coding_hidden_v2_final import (
    build_recovered_summary,
    merge_report_replacements,
    retryable_task_ids,
    inspect_coco_model_health,
    label_existing_recovery_provenance,
)


def result(task_id: str, *, success: bool, agent_returncode: int, diff: str = "") -> dict:
    return {
        "task": {
            "id": task_id,
            "input": "Fix",
            "metadata": {"benchmark_family": task_id, "contract_tags": ["immutability"]},
        },
        "output": {
            "value": {"tests_passed": success, "agent_returncode": agent_returncode},
            "trace": [],
            "metadata": {
                "agent": {
                    "returncode": agent_returncode,
                    "stdout": "ok" if success else "",
                    "stderr": "",
                    "timed_out": agent_returncode == 124,
                },
                "post_test": {"returncode": 0 if success else 1},
                "diff": diff,
            },
        },
        "score": {"value": float(success), "success": success, "message": "", "metadata": {}},
    }


class RecoverCodingHiddenV2FinalTests(unittest.TestCase):
    def test_labels_existing_recovery_without_rerunning_agents(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            executive_dir = run_dir / "a" / "executive"
            executive_dir.mkdir(parents=True)
            report = {
                "name": "recovered",
                "results": [result("family", success=True, agent_returncode=0, diff="patch")],
                "average_score": 1.0,
                "pass_rate": 1.0,
            }
            (executive_dir / "final_validation_recovery.json").write_text(
                json.dumps({"recovered_report": report, "duration_seconds": 3.0}),
                encoding="utf-8",
            )
            (executive_dir / "result.json").write_text(
                json.dumps({"best_skill_text": "skill", "final_validation_report": report}),
                encoding="utf-8",
            )
            original = {
                "manifest": {"target_model": "old-model"},
                "rows": [
                    {
                        "seed": "a",
                        "condition": "executive",
                        "task_accuracy": 0.0,
                        "average_score": 0.0,
                        "family_macro_accuracy": 0.0,
                        "contract_macro_accuracy": 0.0,
                        "contract_breakdown": {"immutability": {"passed": 0, "total": 1, "accuracy": 0.0}},
                        "duration_seconds": 2.0,
                    }
                ],
            }

            summary = label_existing_recovery_provenance(
                run_dir,
                ["a"],
                original,
                recovery_target_model="new-model",
            )

            recovery = json.loads(
                (executive_dir / "final_validation_recovery.json").read_text(encoding="utf-8")
            )
            self.assertEqual(recovery["source_target_model"], "old-model")
            self.assertEqual(recovery["recovery_target_model"], "new-model")
            self.assertEqual(summary["recovery"]["comparability"], "cross_model_transfer_only")

    def test_health_check_blocks_exhausted_configured_model(self) -> None:
        completed = type(
            "Completed",
            (),
            {
                "returncode": 0,
                "stdout": '[{"name":"openrouter-2o","description":"Quota: 100% used, resets weekly"}]',
                "stderr": "",
            },
        )()
        with patch("work.recover_coding_hidden_v2_final.subprocess.run", return_value=completed):
            health = inspect_coco_model_health("openrouter-2o", "coco")

        self.assertEqual(health["status"], "blocked")
        self.assertEqual(health["reason"], "configured_model_quota_exhausted")

    def test_health_check_keeps_configured_model_unchanged(self) -> None:
        completed = type(
            "Completed",
            (),
            {
                "returncode": 0,
                "stdout": '[{"name":"openrouter-2o","description":"Context window: 936k"}]',
                "stderr": "",
            },
        )()
        with patch("work.recover_coding_hidden_v2_final.subprocess.run", return_value=completed) as run:
            health = inspect_coco_model_health("openrouter-2o", "coco")

        self.assertEqual(health["status"], "available")
        self.assertEqual(run.call_args.args[0], ["coco", "models", "--json"])

    def test_retryable_ids_exclude_semantic_failure_with_diff(self) -> None:
        report = {
            "results": [
                result("infra", success=False, agent_returncode=124),
                result("semantic", success=False, agent_returncode=0, diff="patch"),
            ]
        }

        self.assertEqual(retryable_task_ids(report), ["infra"])

    def test_merge_replaces_only_targeted_task(self) -> None:
        source = {
            "name": "raw",
            "results": [
                result("infra", success=False, agent_returncode=124),
                result("semantic", success=False, agent_returncode=0, diff="patch"),
            ],
        }
        recovered = merge_report_replacements(
            source,
            {"infra": result("infra", success=True, agent_returncode=0, diff="patch")},
        )

        self.assertEqual(recovered["pass_rate"], 0.5)
        self.assertFalse(recovered["results"][1]["score"]["success"])

    def test_recovered_summary_updates_only_executive_rows(self) -> None:
        original = {
            "manifest": {"benchmark": "v2"},
            "rows": [
                {
                    "seed": "a",
                    "condition": "no_skill",
                    "task_accuracy": 0.8,
                    "average_score": 0.8,
                    "family_macro_accuracy": 0.8,
                    "contract_macro_accuracy": 0.8,
                    "contract_breakdown": {"immutability": {"passed": 4, "total": 5, "accuracy": 0.8}},
                    "duration_seconds": 1,
                },
                {
                    "seed": "a",
                    "condition": "executive",
                    "task_accuracy": 0.4,
                    "average_score": 0.4,
                    "family_macro_accuracy": 0.4,
                    "contract_macro_accuracy": 0.4,
                    "contract_breakdown": {"immutability": {"passed": 2, "total": 5, "accuracy": 0.4}},
                    "duration_seconds": 2,
                },
            ],
        }
        report = merge_report_replacements(
            {"name": "raw", "results": [result("family", success=False, agent_returncode=124)]},
            {"family": result("family", success=True, agent_returncode=0, diff="patch")},
        )

        recovered = build_recovered_summary(
            original,
            {"a": report},
            {"a": 3.0},
            task_retries=2,
            retry_backoff_seconds=1.0,
            recovery_target_model="new-model",
        )

        self.assertEqual(recovered["rows"][0]["task_accuracy"], 0.8)
        self.assertEqual(recovered["rows"][1]["task_accuracy"], 1.0)
        self.assertEqual(recovered["rows"][1]["contract_macro_accuracy"], 1.0)
        self.assertEqual(recovered["rows"][1]["contract_breakdown"]["immutability"]["accuracy"], 1.0)
        self.assertEqual(recovered["rows"][1]["duration_seconds"], 5.0)
        self.assertEqual(recovered["recovery"]["comparability"], "cross_model_transfer_only")


if __name__ == "__main__":
    unittest.main()
