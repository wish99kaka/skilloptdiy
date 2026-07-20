from __future__ import annotations

import json
import hashlib
import math
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from textskill_optimizer.paper.searchqa_experiment import (
    prepare_zero_call_searchqa_experiment,
    prepare_searchqa_mechanism_smoke,
    run_searchqa_experiment,
)
from textskill_optimizer.paper.searchqa import SearchQAMaterialization
from textskill_optimizer.paper.provenance import canonical_json_sha256
from textskill_optimizer.paper.preregistration import (
    PaperPreregistrationViolation,
    load_paper_preregistration,
)


ROOT = Path(__file__).parents[3]


def _items(prefix: str, count: int) -> list[dict]:
    return [
        {
            "id": f"{prefix}-{index:03d}",
            "question": f"What is answer {index}?",
            "context": f"The answer is value-{index}.",
            "answers": [f"value-{index}"],
        }
        for index in range(count)
    ]


def _fake_materialization(root: Path) -> tuple[Path, SearchQAMaterialization]:
    receipt = root / "materialization-receipt.json"
    train_manifest = root / "official-train-ids.json"
    selection_manifest = root / "official-selection-ids.json"
    receipt.write_text("{}", encoding="utf-8")
    train_manifest.write_text("[]", encoding="utf-8")
    selection_manifest.write_text("[]", encoding="utf-8")
    return receipt, SearchQAMaterialization(
        receipt_path=receipt.resolve(),
        train_manifest_path=train_manifest.resolve(),
        selection_manifest_path=selection_manifest.resolve(),
    )


def _authorized_zero_cost_receipt(commit: str) -> dict:
    return {
        "schema_version": "paper-zero-cost-gate-v1",
        "status": "passed",
        "external_calls": 0,
        "network_guard_active": True,
        "paid_experiment_executed": False,
        "paid_development_authorized": True,
        "code_commit": commit,
        "worktree_clean": True,
        "prompt_count": 18,
        "prompt_snapshot_sha256": canonical_json_sha256(
            json.loads(
                (ROOT / "docs/papers/prompt-snapshot-v1.json").read_text(
                    encoding="utf-8"
                )
            )
        ),
        "source_lock_sha256": canonical_json_sha256(
            json.loads(
                (ROOT / "docs/papers/source-lock.json").read_text(encoding="utf-8")
            )
        ),
        "golden_trace_sha256": hashlib.sha256(
            (ROOT / "tests/conformance/golden/algorithm1-fast-loop-v1.json").read_bytes()
        ).hexdigest(),
        "test_targets": ["tests/conformance", "tests/provenance"],
        "violations": [],
    }


class PaperSearchQAExperimentTests(unittest.TestCase):
    def test_zero_call_run_executes_the_full_epoch_graph_without_test_access(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train_path = root / "open-train.json"
            selection_path = root / "open-selection.json"
            train_path.write_text(json.dumps(_items("train", 40)), encoding="utf-8")
            selection_path.write_text(
                json.dumps(_items("selection", 5)), encoding="utf-8"
            )
            materialization_receipt, materialization = _fake_materialization(root)
            with patch(
                "textskill_optimizer.paper.searchqa_experiment."
                "verify_searchqa_materialization_receipt",
                return_value=materialization,
            ):
                preregistration_path = prepare_zero_call_searchqa_experiment(
                    run_dir=root / "run",
                    train_path=train_path,
                    selection_path=selection_path,
                    materialization_receipt_path=materialization_receipt,
                )

            receipt_path = run_searchqa_experiment(preregistration_path)
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            with self.assertRaisesRegex(ValueError, "single-use"):
                run_searchqa_experiment(preregistration_path)

        self.assertEqual(receipt["status"], "completed")
        self.assertEqual(receipt["completed_epochs"], 4)
        self.assertEqual(receipt["completed_steps"], 4)
        self.assertTrue(receipt["full_call_graph_complete"])
        self.assertTrue(receipt["selection_unsaturated"])
        self.assertEqual(receipt["usage"]["external_target_calls"], 0)
        self.assertEqual(receipt["usage"]["external_optimizer_calls"], 0)
        self.assertGreater(receipt["usage"]["logical_target_calls"], 0)
        self.assertGreater(receipt["usage"]["logical_optimizer_calls"], 0)
        self.assertGreater(receipt["usage"]["estimated_target_tokens"], 0)
        self.assertGreater(receipt["usage"]["estimated_optimizer_tokens"], 0)
        self.assertEqual(receipt["test_access"], {"allowed": False, "attempt": 0})
        self.assertEqual(receipt["test_payload_status"], "not_materialized")
        for required in (
            "run_started",
            "failure_reflected",
            "success_reflected",
            "analyst_refined",
            "merge_final_failure_prioritized",
            "candidate_accepted",
            "candidate_rejected",
            "slow_update_proposed",
            "meta_update_completed",
            "run_completed",
        ):
            self.assertGreater(receipt["event_counts"].get(required, 0), 0, required)

    def test_selection_saturation_writes_a_single_use_stop_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train_path = root / "open-train.json"
            selection_path = root / "open-selection.json"
            train_path.write_text(json.dumps(_items("train", 40)), encoding="utf-8")
            saturated = _items("selection", 5)
            for index, item in enumerate(saturated):
                item["id"] = f"selection-{index * 2:03d}"
            selection_path.write_text(json.dumps(saturated), encoding="utf-8")
            materialization_receipt, materialization = _fake_materialization(root)
            with patch(
                "textskill_optimizer.paper.searchqa_experiment."
                "verify_searchqa_materialization_receipt",
                return_value=materialization,
            ):
                preregistration_path = prepare_zero_call_searchqa_experiment(
                    run_dir=root / "run",
                    train_path=train_path,
                    selection_path=selection_path,
                    materialization_receipt_path=materialization_receipt,
                )

            with self.assertRaisesRegex(RuntimeError, "selection_saturation"):
                run_searchqa_experiment(preregistration_path)
            receipt_path = root / "run" / "receipt.json"
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            with self.assertRaisesRegex(ValueError, "single-use"):
                run_searchqa_experiment(preregistration_path)

        self.assertEqual(
            receipt["schema_version"],
            "paper-searchqa-development-stop-receipt-v1",
        )
        self.assertEqual(receipt["status"], "stopped")
        self.assertEqual(receipt["stop_reason"], "selection_saturation")
        self.assertEqual(receipt["initial_selection_score"], 1.0)
        self.assertFalse(receipt["selection_unsaturated"])
        self.assertEqual(receipt["completed_epochs"], 0)
        self.assertEqual(receipt["completed_steps"], 0)
        self.assertEqual(receipt["usage"]["logical_target_calls"], 5)
        self.assertEqual(receipt["usage"]["logical_optimizer_calls"], 0)
        self.assertEqual(receipt["test_access"], {"allowed": False, "attempt": 0})
        self.assertEqual(receipt["test_payload_status"], "not_materialized")
        self.assertIsNone(receipt["claim_class"])

    def test_prepare_rejects_a_saturated_or_too_small_smoke_slice(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train_path = root / "train.json"
            selection_path = root / "selection.json"
            train_path.write_text(json.dumps(_items("train", 39)), encoding="utf-8")
            selection_path.write_text(json.dumps(_items("selection", 1)), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "40"):
                prepare_zero_call_searchqa_experiment(
                    run_dir=root / "run",
                    train_path=train_path,
                    selection_path=selection_path,
                    materialization_receipt_path=root / "missing-receipt.json",
                )

    def test_paid_smoke_preparation_freezes_models_and_budgets_without_calls(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train_path = root / "train.json"
            selection_path = root / "selection.json"
            config_path = root / "traecli.yaml"
            zero_cost_receipt = root / "zero-cost-receipt.json"
            coco_binary = root / "coco"
            train_path.write_text(json.dumps(_items("train", 40)), encoding="utf-8")
            selection_path.write_text(
                json.dumps(_items("selection", 5)), encoding="utf-8"
            )
            config_path.write_text(
                "model:\n    name: exact-coco-model\n",
                encoding="utf-8",
            )
            coco_binary.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
            os.chmod(coco_binary, 0o700)
            materialization_receipt, materialization = _fake_materialization(root)
            with patch(
                "textskill_optimizer.paper.searchqa_experiment."
                "verify_searchqa_materialization_receipt",
                return_value=materialization,
            ):
                dry_preregistration = prepare_zero_call_searchqa_experiment(
                    run_dir=root / "dry-run",
                    train_path=train_path,
                    selection_path=selection_path,
                    materialization_receipt_path=materialization_receipt,
                    mechanism_smoke_scope=True,
                )
            dry_receipt = run_searchqa_experiment(dry_preregistration)
            dry_usage = json.loads(dry_receipt.read_text(encoding="utf-8"))["usage"]
            zero_cost_receipt.write_text(
                json.dumps(_authorized_zero_cost_receipt("c" * 40)),
                encoding="utf-8",
            )
            with patch.dict("os.environ", {"EXTERNAL_LLM_MODEL": "optimizer-v1"}):
                with patch(
                    "textskill_optimizer.paper.searchqa_experiment._git_identity",
                    return_value=("c" * 40, True),
                ), patch(
                    "textskill_optimizer.paper.searchqa_experiment."
                    "_default_coco_config_path",
                    return_value=config_path,
                ), patch(
                    "textskill_optimizer.paper.searchqa_experiment.resolve_coco_binary",
                    return_value=coco_binary,
                ), patch(
                    "textskill_optimizer.paper.searchqa_experiment."
                    "verify_searchqa_materialization_receipt",
                    return_value=materialization,
                ):
                    preregistration_path = prepare_searchqa_mechanism_smoke(
                        run_dir=root / "paid-run",
                        train_path=train_path,
                        selection_path=selection_path,
                        target_model="exact-coco-model",
                        target_reasoning="not_configured",
                        optimizer_model="optimizer-v1",
                        optimizer_reasoning="medium",
                        safety_factor=1.5,
                        zero_cost_receipt_path=zero_cost_receipt,
                        materialization_receipt_path=materialization_receipt,
                        mechanism_dry_run_receipt_path=dry_receipt,
                    )
            payload = json.loads(preregistration_path.read_text(encoding="utf-8"))
            tampered = json.loads(preregistration_path.read_text(encoding="utf-8"))
            tampered["budgets"]["target_calls"] += 1
            tampered_path = root / "tampered-preregistration.json"
            tampered_path.write_text(json.dumps(tampered), encoding="utf-8")
            with self.assertRaisesRegex(
                PaperPreregistrationViolation, "mechanically derived"
            ):
                load_paper_preregistration(tampered_path)
            plan = json.loads(
                (root / "paid-run" / "paper-epoch-plan.json").read_text(
                    encoding="utf-8"
                )
            )
            with patch(
                "textskill_optimizer.paper.searchqa_experiment._git_identity",
                return_value=("d" * 40, True),
            ):
                with self.assertRaisesRegex(ValueError, "preregistered clean Git commit"):
                    run_searchqa_experiment(preregistration_path)

        self.assertEqual(payload["stage"], "mechanism_smoke")
        self.assertEqual(payload["models"]["target_model"], "exact-coco-model")
        self.assertEqual(payload["models"]["optimizer_model"], "optimizer-v1")
        self.assertEqual(payload["execution"]["target_backend"], "coco")
        self.assertEqual(
            payload["execution"]["optimizer_backend"], "openai_compatible"
        )
        self.assertEqual(plan["epochs"], 2)
        self.assertEqual(plan["mechanisms"]["claim_scope"], "mechanism_test")
        self.assertEqual(payload["authorization"]["local_code_commit"], "c" * 40)
        self.assertEqual(
            payload["budgets"]["target_calls"],
            math.ceil(dry_usage["logical_target_calls"] * 1.5),
        )
        self.assertEqual(
            payload["budgets"]["optimizer_tokens"],
            math.ceil(dry_usage["estimated_optimizer_tokens"] * 1.5),
        )
        self.assertEqual(payload["budgets"]["token_policy"], "audit_only")
        self.assertIn(
            "zero_cost_receipt",
            {artifact["artifact_id"] for artifact in payload["artifacts"]},
        )


if __name__ == "__main__":
    unittest.main()
