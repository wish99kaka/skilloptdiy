from __future__ import annotations

import json
import hashlib
import os
import tempfile
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
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
    get_searchqa_development_materialization_policy,
    load_searchqa_items,
    normalize_searchqa_answer,
    sample_searchqa_development_ids,
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
from textskill_optimizer.paper.searchqa_controller_runtime import (
    CocoACPWorkerPool,
    TargetBudgetGuard,
    TargetResult,
    _CocoACPWorker,
    _append_failed_usage,
)


class PaperSearchQAContractTests(unittest.TestCase):
    def test_coco_acp_pool_starts_five_workers_serially_then_runs_in_parallel(self) -> None:
        starts: list[int] = []
        active_by_worker: dict[int, int] = {}
        active_total = 0
        max_active_total = 0
        lock = threading.Lock()
        first_wave = threading.Barrier(5)

        class FakeWorker:
            def __init__(self, worker_id: int) -> None:
                self.worker_id = worker_id

            def prompt(self, prompt: str, *, timeout: float) -> TargetResult:
                nonlocal active_total, max_active_total
                with lock:
                    active_by_worker[self.worker_id] = (
                        active_by_worker.get(self.worker_id, 0) + 1
                    )
                    self.assert_worker_is_serial()
                    active_total += 1
                    max_active_total = max(max_active_total, active_total)
                if prompt.startswith("first-"):
                    first_wave.wait(timeout=1.0)
                time.sleep(0.01)
                with lock:
                    active_by_worker[self.worker_id] -= 1
                    active_total -= 1
                return TargetResult(prompt, 1, 1, True, 0.01)

            def assert_worker_is_serial(self) -> None:
                if active_by_worker[self.worker_id] != 1:
                    raise AssertionError("one ACP worker received concurrent prompts")

            def close(self) -> None:
                return None

        def start_worker(*, worker_id: int, **_kwargs):
            starts.append(worker_id)
            self.assertEqual(active_total, 0)
            return FakeWorker(worker_id)

        pool = CocoACPWorkerPool.start(
            binary=Path("/tmp/fake-coco"),
            cwd=Path("/tmp"),
            task_count=10,
            startup_timeout=lambda: 1.0,
            worker_factory=start_worker,
        )
        try:
            with ThreadPoolExecutor(max_workers=5) as executor:
                first = list(
                    executor.map(
                        lambda index: pool.prompt(f"first-{index}", timeout=1.0),
                        range(5),
                    )
                )
            with ThreadPoolExecutor(max_workers=5) as executor:
                second = list(
                    executor.map(
                        lambda index: pool.prompt(f"second-{index}", timeout=1.0),
                        range(5),
                    )
                )
        finally:
            pool.close()

        self.assertEqual(starts, [0, 1, 2, 3, 4])
        self.assertEqual(max_active_total, 5)
        self.assertEqual([item.response for item in first], [f"first-{i}" for i in range(5)])
        self.assertEqual(
            [item.response for item in second], [f"second-{i}" for i in range(5)]
        )

    def test_coco_acp_pool_closes_started_workers_when_serial_startup_fails(self) -> None:
        closed: list[int] = []

        class FakeWorker:
            def __init__(self, worker_id: int) -> None:
                self.worker_id = worker_id

            def close(self) -> None:
                closed.append(self.worker_id)

        def start_worker(*, worker_id: int, **_kwargs):
            if worker_id == 2:
                raise RuntimeError("startup failed")
            return FakeWorker(worker_id)

        with self.assertRaisesRegex(RuntimeError, "startup failed"):
            CocoACPWorkerPool.start(
                binary=Path("/tmp/fake-coco"),
                cwd=Path("/tmp"),
                task_count=5,
                startup_timeout=lambda: 1.0,
                worker_factory=start_worker,
            )

        self.assertEqual(closed, [1, 0])

    def test_coco_acp_worker_uses_a_fresh_session_and_actual_usage_per_prompt(self) -> None:
        fake_server = """#!/usr/bin/env python3
import json
import sys

sessions = 0
used = set()
for line in sys.stdin:
    request = json.loads(line)
    method = request["method"]
    if method == "initialize":
        result = {"protocolVersion": 1, "agentCapabilities": {}, "authMethods": []}
    elif method == "session/new":
        sessions += 1
        result = {"sessionId": f"session-{sessions}"}
    elif method == "session/prompt":
        session_id = request["params"]["sessionId"]
        if session_id in used:
            response = {"jsonrpc": "2.0", "id": request["id"], "error": {"code": -32602, "message": "session reused"}}
            print(json.dumps(response), flush=True)
            continue
        used.add(session_id)
        update = {
            "jsonrpc": "2.0",
            "method": "session/update",
            "params": {
                "sessionId": session_id,
                "update": {
                    "sessionUpdate": "agent_message_chunk",
                    "content": {"type": "text", "text": f"reply-{session_id}"},
                },
            },
        }
        print(json.dumps(update), flush=True)
        result = {
            "stopReason": "end_turn",
            "usage": {"inputTokens": sessions * 10, "outputTokens": sessions, "totalTokens": sessions * 11},
        }
    else:
        response = {"jsonrpc": "2.0", "id": request["id"], "error": {"code": -32601, "message": "unknown"}}
        print(json.dumps(response), flush=True)
        continue
    print(json.dumps({"jsonrpc": "2.0", "id": request["id"], "result": result}), flush=True)
"""
        with tempfile.TemporaryDirectory() as tmp:
            binary = Path(tmp) / "fake-coco"
            binary.write_text(fake_server, encoding="utf-8")
            os.chmod(binary, 0o700)
            worker = _CocoACPWorker.start(
                worker_id=0,
                binary=binary,
                cwd=Path(tmp),
                timeout=2.0,
            )
            try:
                first = worker.prompt("first", timeout=2.0)
                second = worker.prompt("second", timeout=2.0)
            finally:
                worker.close()

        self.assertEqual(first.response, "reply-session-1")
        self.assertEqual(first.prompt_tokens, 10)
        self.assertEqual(first.completion_tokens, 1)
        self.assertEqual(second.response, "reply-session-2")
        self.assertEqual(second.prompt_tokens, 20)
        self.assertEqual(second.completion_tokens, 2)

    def test_coco_acp_pool_fails_closed_without_restarting_a_worker(self) -> None:
        starts: list[int] = []

        class FailingWorker:
            def prompt(self, prompt: str, *, timeout: float) -> TargetResult:
                raise RuntimeError("prompt failed")

            def close(self) -> None:
                return None

        def start_worker(*, worker_id: int, **_kwargs):
            starts.append(worker_id)
            return FailingWorker()

        pool = CocoACPWorkerPool.start(
            binary=Path("/tmp/fake-coco"),
            cwd=Path("/tmp"),
            task_count=1,
            startup_timeout=lambda: 1.0,
            worker_factory=start_worker,
        )
        try:
            with self.assertRaisesRegex(RuntimeError, "prompt failed"):
                pool.prompt("first", timeout=1.0)
            with self.assertRaisesRegex(RuntimeError, "stopped after a worker failure"):
                pool.prompt("second", timeout=1.0)
        finally:
            pool.close()

        self.assertEqual(starts, [0])

    def test_failed_target_usage_retains_bounded_root_cause(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ledger = Path(tmp) / "usage.jsonl"
            args = SimpleNamespace(
                usage_ledger=ledger,
                target_model="target-v1",
                target_reasoning="not_configured",
            )
            item = SearchQAItem("item-1", "Question?", "Context", ("Answer",))

            _append_failed_usage(
                args,
                item,
                prompt="prompt",
                error=RuntimeError("ACP_ROOT_CAUSE" + "x" * 2000),
            )
            record = json.loads(ledger.read_text(encoding="utf-8"))

        self.assertTrue(record["error_message"].startswith("ACP_ROOT_CAUSE"))
        self.assertLessEqual(len(record["error_message"]), 1000)

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

    def test_materialization_policy_versions_the_larger_smoke_selection(self) -> None:
        legacy = get_searchqa_development_materialization_policy(
            train_limit=40,
            selection_limit=5,
            seed=42,
        )
        current = get_searchqa_development_materialization_policy(
            train_limit=40,
            selection_limit=20,
            seed=42,
        )

        self.assertEqual(
            legacy.schema_version,
            "searchqa-development-materialization-v2",
        )
        self.assertEqual(
            current.schema_version,
            "searchqa-development-materialization-v3",
        )
        self.assertEqual(current.selection_limit, 20)
        with self.assertRaisesRegex(SearchQAContractViolation, "unsupported"):
            get_searchqa_development_materialization_policy(
                train_limit=40,
                selection_limit=10,
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
            selection_ids = [f"{100 + index:032x}" for index in range(20)]
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

            def sha(path):
                return hashlib.sha256(path.read_bytes()).hexdigest()

            hashes = {
                "train": sha(train_manifest),
                "selection": sha(selection_manifest),
                "test": "f" * 64,
            }
            receipt_path = root / "materialization-receipt.json"
            output_hashes = {}
            for schema_version, selection_limit in (
                ("searchqa-development-materialization-v2", 5),
                ("searchqa-development-materialization-v3", 20),
            ):
                with self.subTest(schema_version=schema_version):
                    sampled = sample_searchqa_development_ids(
                        train_ids=train_ids,
                        selection_ids=selection_ids,
                        train_limit=40,
                        selection_limit=selection_limit,
                        seed=42,
                    )
                    train_path.write_text(
                        json.dumps(items(sampled["train"])), encoding="utf-8"
                    )
                    selection_path.write_text(
                        json.dumps(items(sampled["selection"])), encoding="utf-8"
                    )
                    output_hashes[schema_version] = {
                        "train": sha(train_path),
                        "selection": sha(selection_path),
                    }
                    receipt = {
                        "schema_version": schema_version,
                        "source_repo": "lucadiliello/searchqa",
                        "source_revision": SEARCHQA_DATASET_REVISION,
                        "source_access": {
                            "method": "hf_dataset_server_filter_v1",
                            "endpoint": SEARCHQA_DATASET_SERVER_ENDPOINT,
                            "source_main_revision": SEARCHQA_DATASET_REVISION,
                            "queried_splits": ["train", "validation"],
                            "requested_id_count": 40 + selection_limit,
                            "received_id_count": 40 + selection_limit,
                        },
                        "official_manifest_sha256": {
                            "train": hashes["train"],
                            "selection": hashes["selection"],
                            "test_commitment": hashes["test"],
                        },
                        "manifest_files": {
                            "train": {
                                "path": str(train_manifest),
                                "sha256": hashes["train"],
                            },
                            "selection": {
                                "path": str(selection_manifest),
                                "sha256": hashes["selection"],
                            },
                        },
                        "sample": {
                            "seed": 42,
                            "train_limit": 40,
                            "selection_limit": selection_limit,
                        },
                        "counts": {"train": 40, "selection": selection_limit},
                        "outputs": {
                            "train": {
                                "path": str(train_path),
                                "sha256": sha(train_path),
                            },
                            "selection": {
                                "path": str(selection_path),
                                "sha256": sha(selection_path),
                            },
                        },
                        "test_payload_status": "not_materialized",
                    }
                    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
                    with patch.dict(
                        "textskill_optimizer.paper.searchqa."
                        "OFFICIAL_SEARCHQA_ID_MANIFEST_SHA256",
                        hashes,
                        clear=True,
                    ), patch.dict(
                        "textskill_optimizer.paper.searchqa."
                        "OFFICIAL_SEARCHQA_DEVELOPMENT_OUTPUT_SHA256_BY_SCHEMA",
                        output_hashes,
                        clear=True,
                    ):
                        verified = verify_searchqa_materialization_receipt(
                            receipt_path,
                            train_path=train_path,
                            selection_path=selection_path,
                        )

            train_path.write_text("[]", encoding="utf-8")
            with patch.dict(
                "textskill_optimizer.paper.searchqa."
                "OFFICIAL_SEARCHQA_ID_MANIFEST_SHA256",
                hashes,
                clear=True,
            ), patch.dict(
                "textskill_optimizer.paper.searchqa."
                "OFFICIAL_SEARCHQA_DEVELOPMENT_OUTPUT_SHA256_BY_SCHEMA",
                output_hashes,
                clear=True,
            ):
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
