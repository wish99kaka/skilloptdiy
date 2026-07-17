from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from textskill_optimizer.paper.preregistration import (
    PaperPreregistrationViolation,
    load_paper_preregistration,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class PaperPreregistrationTests(unittest.TestCase):
    def _valid_payload(self, root: Path) -> dict:
        train = root / "train.json"
        selection = root / "selection.json"
        runner = root / "runner.py"
        plan = root / "plan.json"
        materialization = root / "materialization.json"
        official_train_ids = root / "official-train-ids.json"
        official_selection_ids = root / "official-selection-ids.json"
        for path, content in (
            (train, "[]"),
            (selection, "[]"),
            (runner, "# frozen runner\n"),
            (plan, "{}"),
            (materialization, "{}"),
            (official_train_ids, "[]"),
            (official_selection_ids, "[]"),
        ):
            path.write_text(content, encoding="utf-8")
        artifacts = [
            {"artifact_id": path.stem, "path": str(path), "sha256": _sha256(path)}
            for path in (train, selection, runner, plan)
        ]
        artifacts.extend(
            {
                "artifact_id": artifact_id,
                "path": str(path),
                "sha256": _sha256(path),
            }
            for artifact_id, path in (
                ("materialization_receipt", materialization),
                ("official_train_id_manifest", official_train_ids),
                ("official_selection_id_manifest", official_selection_ids),
            )
        )
        return {
            "schema_version": "paper-development-preregistration-v1",
            "protocol_id": "paper-faithful-v1",
            "stage": "zero_call_dry_run",
            "authorization": None,
            "benchmark": {
                "id": "searchqa",
                "source_repo": "lucadiliello/searchqa",
                "source_revision": "c1a979068ba118d85467179b704031d113d689cc",
                "train_split_id": "searchqa-smoke-train-v1",
                "selection_split_id": "searchqa-smoke-selection-v1",
                "train_count": 40,
                "selection_count": 5,
                "official_test_id_manifest_sha256": "b" * 64,
                "test_payload_status": "not_materialized",
            },
            "models": {
                "target_model": "scripted-searchqa-v1",
                "target_reasoning": "none",
                "optimizer_model": "scripted-optimizer-v1",
                "optimizer_reasoning": "none",
            },
            "execution": {
                "seed": 42,
                "retry_policy": "semantic-retry-once-v1",
                "target_backend": "scripted",
                "optimizer_backend": "scripted",
                "profile_sha256": "a" * 64,
                "plan_artifact_id": "plan",
            },
            "budgets": {
                "target_calls": 480,
                "target_tokens": 1,
                "optimizer_calls": 240,
                "optimizer_tokens": 1,
                "wall_time_seconds": 3600.0,
                "safety_factor": 1.5,
            },
            "stop_conditions": [
                "budget_breach",
                "controller_failure",
                "data_firewall_violation",
                "selection_saturation",
            ],
            "test_access": {"allowed": False, "attempt": 0},
            "artifacts": artifacts,
        }

    def test_accepts_a_hash_bound_zero_call_preregistration(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "preregistration.json"
            path.write_text(json.dumps(self._valid_payload(root)), encoding="utf-8")

            prereg = load_paper_preregistration(path)

        self.assertEqual(prereg.stage, "zero_call_dry_run")
        self.assertFalse(prereg.test_access_allowed)

    def test_rejects_any_development_test_payload_access(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            payload = self._valid_payload(root)
            payload["benchmark"]["test_payload_status"] = "materialized"
            path = root / "preregistration.json"
            path.write_text(json.dumps(payload), encoding="utf-8")

            with self.assertRaisesRegex(PaperPreregistrationViolation, "test payload"):
                load_paper_preregistration(path)

    def test_rejects_unresolved_model_identity_and_artifact_drift(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            payload = self._valid_payload(root)
            payload["models"]["target_model"] = "configured-default"
            path = root / "preregistration.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(PaperPreregistrationViolation, "model identity"):
                load_paper_preregistration(path)

            payload = self._valid_payload(root)
            path.write_text(json.dumps(payload), encoding="utf-8")
            (root / "runner.py").write_text("# drifted\n", encoding="utf-8")
            with self.assertRaisesRegex(PaperPreregistrationViolation, "hash"):
                load_paper_preregistration(path)

    def test_rejects_missing_or_unbounded_stop_budgets(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            payload = self._valid_payload(root)
            payload["budgets"]["target_calls"] = 0
            path = root / "preregistration.json"
            path.write_text(json.dumps(payload), encoding="utf-8")

            with self.assertRaisesRegex(PaperPreregistrationViolation, "budget"):
                load_paper_preregistration(path)

    def test_paid_stage_requires_a_hash_bound_receipt_for_the_same_clean_commit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            payload = self._valid_payload(root)
            receipt_path = root / "zero-cost-receipt.json"
            receipt_path.write_text(
                json.dumps(
                    {
                        "schema_version": "paper-zero-cost-gate-v1",
                        "status": "passed",
                        "external_calls": 0,
                        "network_guard_active": True,
                        "paid_experiment_executed": False,
                        "paid_development_authorized": True,
                        "worktree_clean": True,
                        "code_commit": "c" * 40,
                        "prompt_count": 18,
                        "prompt_snapshot_sha256": "a" * 64,
                        "source_lock_sha256": "b" * 64,
                        "golden_trace_sha256": "c" * 64,
                        "test_targets": ["tests/conformance", "tests/provenance"],
                        "violations": [],
                    }
                ),
                encoding="utf-8",
            )
            payload["stage"] = "mechanism_smoke"
            payload["authorization"] = {
                "local_code_commit": "d" * 40,
                "zero_cost_receipt_artifact_id": "zero_cost_receipt",
                "mechanism_dry_run_receipt_artifact_id": (
                    "mechanism_dry_run_receipt"
                ),
                "mechanism_dry_run_preregistration_artifact_id": (
                    "mechanism_dry_run_preregistration"
                ),
                "paid_development_authorized": True,
            }
            payload["execution"]["target_backend"] = "coco"
            payload["execution"]["optimizer_backend"] = "openai_compatible"
            payload["artifacts"].append(
                {
                    "artifact_id": "zero_cost_receipt",
                    "path": str(receipt_path),
                    "sha256": _sha256(receipt_path),
                }
            )
            path = root / "preregistration.json"
            path.write_text(json.dumps(payload), encoding="utf-8")

            with self.assertRaisesRegex(
                PaperPreregistrationViolation, "does not authorize"
            ):
                load_paper_preregistration(path)


if __name__ == "__main__":
    unittest.main()
