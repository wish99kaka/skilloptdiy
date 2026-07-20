from __future__ import annotations

import json
import hashlib
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from textskill_optimizer.paper.searchqa import (
    SEARCHQA_DATASET_REVISION,
    SEARCHQA_DATASET_SERVER_ENDPOINT,
    SearchQAContractViolation,
    SearchQAItem,
    extract_searchqa_answer,
    fetch_searchqa_rows_by_id,
    load_searchqa_items,
    normalize_searchqa_answer,
    score_searchqa_response,
    select_searchqa_development_rows,
    verify_searchqa_materialization_receipt,
)
from textskill_optimizer.paper.backend import OptimizerRequest, OptimizerStage
from textskill_optimizer.paper.searchqa_experiment import (
    OpenAICompatiblePaperOptimizerBackend,
    PaidBudgetGuard,
    _require_within_budgets,
)
from textskill_optimizer.paper.searchqa_controller_runtime import TargetBudgetGuard


class PaperSearchQAContractTests(unittest.TestCase):
    def test_pins_the_materialization_source_revision(self) -> None:
        self.assertEqual(
            SEARCHQA_DATASET_REVISION,
            "c1a979068ba118d85467179b704031d113d689cc",
        )

    def test_matches_official_answer_extraction_and_exact_match(self) -> None:
        response = "analysis\n<answer>wrong</answer>\n<answer>The, Apollo 11!</answer>"

        self.assertEqual(extract_searchqa_answer(response), "The, Apollo 11!")
        self.assertEqual(normalize_searchqa_answer("The, Apollo 11!"), "apollo 11")
        score = score_searchqa_response(response, ("Apollo 11", "Apollo Eleven"))
        self.assertEqual(score.exact_match, 1.0)
        self.assertEqual(score.predicted_answer, "The, Apollo 11!")

    def test_falls_back_to_the_last_non_empty_line(self) -> None:
        score = score_searchqa_response("reasoning\n\nParis", ("Paris",))

        self.assertEqual(score.exact_match, 1.0)
        self.assertEqual(score.predicted_answer, "Paris")

    def test_item_schema_is_exact_and_answers_are_non_empty(self) -> None:
        with self.assertRaisesRegex(SearchQAContractViolation, "exactly"):
            SearchQAItem.from_mapping(
                {
                    "id": "q1",
                    "question": "Question?",
                    "context": "Context",
                    "answers": ["Answer"],
                    "split": "test",
                }
            )
        with self.assertRaisesRegex(SearchQAContractViolation, "answers"):
            SearchQAItem.from_mapping(
                {
                    "id": "q1",
                    "question": "Question?",
                    "context": "Context",
                    "answers": [],
                }
            )

    def test_loader_rejects_duplicate_ids(self) -> None:
        item = {
            "id": "q1",
            "question": "Question?",
            "context": "Context",
            "answers": ["Answer"],
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "items.json"
            path.write_text(json.dumps([item, item]), encoding="utf-8")

            with self.assertRaisesRegex(SearchQAContractViolation, "duplicate"):
                load_searchqa_items(path)

    def test_development_materialization_selects_only_train_and_selection_ids(self) -> None:
        rows = [
            {
                "key": f"id-{index}",
                "question": f"Question {index}?",
                "context": f"Context {index}",
                "answers": [f"answer-{index}"],
            }
            for index in range(8)
        ]

        selected = select_searchqa_development_rows(
            rows,
            train_ids=("id-0", "id-1", "id-2", "id-3"),
            selection_ids=("id-4", "id-5"),
            train_limit=4,
            selection_limit=2,
            seed=42,
        )

        self.assertEqual(len(selected["train"]), 4)
        self.assertEqual(len(selected["selection"]), 2)
        self.assertNotIn("test", selected)
        self.assertEqual(
            {item.item_id for item in selected["train"]},
            {"id-0", "id-1", "id-2", "id-3"},
        )

    def test_development_materialization_fails_if_a_selected_id_is_missing(self) -> None:
        with self.assertRaisesRegex(SearchQAContractViolation, "missing"):
            select_searchqa_development_rows(
                [],
                train_ids=("missing-train",),
                selection_ids=("missing-selection",),
                train_limit=1,
                selection_limit=1,
                seed=42,
            )

    def test_filtered_fetch_requests_only_explicit_ids(self) -> None:
        item_id = "a" * 32
        captured: list[str] = []

        def fake_fetch(url, *, timeout_seconds):
            captured.append(url)
            if "revision/main" in url:
                return {"sha": SEARCHQA_DATASET_REVISION}
            rows = (
                [
                    {
                        "row": {
                            "key": item_id,
                            "question": "Question?",
                            "context": "Context",
                            "answers": ["Answer"],
                        },
                        "truncated_cells": [],
                    }
                ]
                if "split=train" in url
                else []
            )
            return {
                "rows": rows,
                "partial": False,
                "num_rows_total": len(rows),
            }

        with patch(
            "textskill_optimizer.paper.searchqa._fetch_json",
            side_effect=fake_fetch,
        ):
            fetched = fetch_searchqa_rows_by_id((item_id,))

        self.assertEqual([row["key"] for row in fetched.rows], [item_id])
        filter_urls = [url for url in captured if url.startswith(SEARCHQA_DATASET_SERVER_ENDPOINT)]
        self.assertEqual(len(filter_urls), 2)
        self.assertTrue(all("where=" in url and item_id in url for url in filter_urls))

    def test_materialization_receipt_binds_official_sample_and_output_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train_ids = [f"{index:032x}" for index in range(40)]
            selection_ids = [f"{100 + index:032x}" for index in range(5)]
            train_manifest = root / "official-train-ids.json"
            selection_manifest = root / "official-selection-ids.json"
            train_path = root / "train.json"
            selection_path = root / "selection.json"
            train_manifest.write_text(
                json.dumps([{"id": item_id} for item_id in train_ids]),
                encoding="utf-8",
            )
            selection_manifest.write_text(
                json.dumps([{"id": item_id} for item_id in selection_ids]),
                encoding="utf-8",
            )

            def items(ids):
                return [
                    {
                        "id": item_id,
                        "question": "Question?",
                        "context": "Context",
                        "answers": ["Answer"],
                    }
                    for item_id in ids
                ]

            train_path.write_text(json.dumps(items(train_ids)), encoding="utf-8")
            selection_path.write_text(
                json.dumps(items(selection_ids)), encoding="utf-8"
            )

            def sha(path):
                return hashlib.sha256(path.read_bytes()).hexdigest()

            hashes = {
                "train": sha(train_manifest),
                "selection": sha(selection_manifest),
                "test": "f" * 64,
            }
            receipt = {
                "schema_version": "searchqa-development-materialization-v2",
                "source_repo": "lucadiliello/searchqa",
                "source_revision": SEARCHQA_DATASET_REVISION,
                "source_access": {
                    "method": "hf_dataset_server_filter_v1",
                    "endpoint": SEARCHQA_DATASET_SERVER_ENDPOINT,
                    "source_main_revision": SEARCHQA_DATASET_REVISION,
                    "queried_splits": ["train", "validation"],
                    "requested_id_count": 45,
                    "received_id_count": 45,
                },
                "official_manifest_sha256": {
                    "train": hashes["train"],
                    "selection": hashes["selection"],
                    "test_commitment": hashes["test"],
                },
                "manifest_files": {
                    "train": {"path": str(train_manifest), "sha256": hashes["train"]},
                    "selection": {
                        "path": str(selection_manifest),
                        "sha256": hashes["selection"],
                    },
                },
                "sample": {"seed": 42, "train_limit": 40, "selection_limit": 5},
                "counts": {"train": 40, "selection": 5},
                "outputs": {
                    "train": {"path": str(train_path), "sha256": sha(train_path)},
                    "selection": {
                        "path": str(selection_path),
                        "sha256": sha(selection_path),
                    },
                },
                "test_payload_status": "not_materialized",
            }
            receipt_path = root / "materialization-receipt.json"
            receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
            with patch.dict(
                "textskill_optimizer.paper.searchqa."
                "OFFICIAL_SEARCHQA_ID_MANIFEST_SHA256",
                hashes,
                clear=True,
            ), patch.dict(
                "textskill_optimizer.paper.searchqa."
                "OFFICIAL_SEARCHQA_DEVELOPMENT_OUTPUT_SHA256",
                {"train": sha(train_path), "selection": sha(selection_path)},
                clear=True,
            ):
                verified = verify_searchqa_materialization_receipt(
                    receipt_path,
                    train_path=train_path,
                    selection_path=selection_path,
                )
                train_path.write_text("[]", encoding="utf-8")
                with self.assertRaisesRegex(SearchQAContractViolation, "hash drift"):
                    verify_searchqa_materialization_receipt(
                        receipt_path,
                        train_path=train_path,
                        selection_path=selection_path,
                    )

        self.assertEqual(verified.receipt_path.name, "materialization-receipt.json")

    def test_external_optimizer_requires_json_and_actual_usage(self) -> None:
        provider_response = {
            "choices": [{"message": {"content": '{"reasoning":"ok","edits":[]}'}}],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 4,
                "total_tokens": 14,
            },
        }

        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self):
                return json.dumps(provider_response).encode()

        captured = {}

        def fake_urlopen(request, timeout):
            captured["body"] = json.loads(request.data)
            captured["timeout"] = timeout
            return FakeResponse()

        with patch.dict(
            "os.environ",
            {
                "EXTERNAL_LLM_BASE_URL": "https://example.invalid/v1",
                "EXTERNAL_LLM_API_KEY": "secret",
                "EXTERNAL_LLM_MODEL": "optimizer-v1",
            },
        ), patch("urllib.request.urlopen", side_effect=fake_urlopen):
            guard = PaidBudgetGuard(
                {
                    "optimizer_calls": 1,
                    "optimizer_tokens": 100,
                },
                deadline=time.monotonic() + 60,
            )
            backend = OpenAICompatiblePaperOptimizerBackend(
                model_id="optimizer-v1",
                reasoning_effort="medium",
                budget_guard=guard,
            )
            request = OptimizerRequest(
                call_id="call-1",
                stage=OptimizerStage.MERGE_FAILURE,
                system_prompt="system",
                prompt="{}",
                response_schema={"type": "object"},
            )
            response = backend.complete(request)
            with self.assertRaisesRegex(RuntimeError, "optimizer_calls"):
                backend.complete(request)

        self.assertEqual(captured["body"]["response_format"], {"type": "json_object"})
        self.assertEqual(captured["body"]["reasoning_effort"], "medium")
        self.assertEqual(response.usage["total_tokens"], 14)

    def test_model_tokens_are_audit_only_but_call_caps_still_stop(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            guard = TargetBudgetGuard(
                SimpleNamespace(
                    usage_ledger=root / "train.jsonl",
                    peer_usage_ledger=root / "selection.jsonl",
                    target_call_cap=1,
                    target_token_cap=100,
                    deadline_monotonic=time.monotonic() + 60,
                )
            )

            guard.reserve(estimated_prompt_tokens=10)
            with self.assertRaisesRegex(RuntimeError, "target_calls"):
                guard.reserve(estimated_prompt_tokens=10)

            token_guard = TargetBudgetGuard(
                SimpleNamespace(
                    usage_ledger=root / "train-2.jsonl",
                    peer_usage_ledger=root / "selection-2.jsonl",
                    target_call_cap=2,
                    target_token_cap=1,
                    deadline_monotonic=time.monotonic() + 60,
                )
            )
            token_guard.reserve(estimated_prompt_tokens=10)
            token_guard.settle(1_000, reservation=0)
            token_guard.reserve(estimated_prompt_tokens=10)
            token_guard.settle(1_000, reservation=0)
            with self.assertRaisesRegex(RuntimeError, "target_calls"):
                token_guard.reserve(estimated_prompt_tokens=10)

            optimizer_guard = PaidBudgetGuard(
                {"optimizer_calls": 2, "optimizer_tokens": 1},
                deadline=time.monotonic() + 60,
            )
            reservation = optimizer_guard.reserve_optimizer_call(estimated_tokens=10)
            optimizer_guard.settle_optimizer_tokens(1_000, reservation=reservation)
            reservation = optimizer_guard.reserve_optimizer_call(estimated_tokens=10)
            optimizer_guard.settle_optimizer_tokens(1_000, reservation=reservation)
            with self.assertRaisesRegex(RuntimeError, "optimizer_calls"):
                optimizer_guard.reserve_optimizer_call(estimated_tokens=10)

            budgets = {
                "target_calls": 2,
                "target_tokens": 1,
                "optimizer_calls": 2,
                "optimizer_tokens": 1,
                "wall_time_seconds": 60.0,
            }
            usage = {
                "logical_target_calls": 2,
                "target_tokens": 2_000,
                "logical_optimizer_calls": 2,
                "optimizer_tokens": 2_000,
            }
            _require_within_budgets(budgets, usage, 1.0)
            usage["logical_target_calls"] = 3
            with self.assertRaisesRegex(RuntimeError, "target_calls"):
                _require_within_budgets(budgets, usage, 1.0)


if __name__ == "__main__":
    unittest.main()
    fetch_searchqa_rows_by_id,
