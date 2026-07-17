#!/usr/bin/env python3
"""Materialize only SearchQA train/selection rows from pinned official IDs."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from textskill_optimizer.paper.searchqa import (
    OFFICIAL_SEARCHQA_ID_MANIFEST_SHA256,
    SEARCHQA_DATASET_SERVER_ENDPOINT,
    SEARCHQA_DATASET_REPO,
    SEARCHQA_DATASET_REVISION,
    fetch_searchqa_rows_by_id,
    sample_searchqa_development_ids,
    select_searchqa_development_rows,
    verify_searchqa_materialization_receipt,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-manifest-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-size", type=int, default=40)
    parser.add_argument("--selection-size", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    try:
        train_manifest = args.official_manifest_dir / "train" / "items.json"
        selection_manifest = args.official_manifest_dir / "val" / "items.json"
        _require_hash(train_manifest, OFFICIAL_SEARCHQA_ID_MANIFEST_SHA256["train"])
        _require_hash(
            selection_manifest,
            OFFICIAL_SEARCHQA_ID_MANIFEST_SHA256["selection"],
        )
        train_ids = _load_ids(train_manifest)
        selection_ids = _load_ids(selection_manifest)
        sampled_ids = sample_searchqa_development_ids(
            train_ids=train_ids,
            selection_ids=selection_ids,
            train_limit=args.train_size,
            selection_limit=args.selection_size,
            seed=args.seed,
        )
        requested_ids = (*sampled_ids["train"], *sampled_ids["selection"])
        source = fetch_searchqa_rows_by_id(requested_ids)
        selected = select_searchqa_development_rows(
            source.rows,
            train_ids=sampled_ids["train"],
            selection_ids=sampled_ids["selection"],
            train_limit=args.train_size,
            selection_limit=args.selection_size,
            seed=args.seed,
        )
        output_dir = args.output_dir.resolve()
        output_dir.mkdir(parents=True, exist_ok=False)
        train_path = output_dir / "train.json"
        selection_path = output_dir / "selection.json"
        copied_train_manifest = output_dir / "official-train-ids.json"
        copied_selection_manifest = output_dir / "official-selection-ids.json"
        copied_train_manifest.write_bytes(train_manifest.read_bytes())
        copied_selection_manifest.write_bytes(selection_manifest.read_bytes())
        _write_items(train_path, selected["train"])
        _write_items(selection_path, selected["selection"])
        receipt = {
            "schema_version": "searchqa-development-materialization-v2",
            "source_repo": SEARCHQA_DATASET_REPO,
            "source_revision": SEARCHQA_DATASET_REVISION,
            "source_access": {
                "method": "hf_dataset_server_filter_v1",
                "endpoint": SEARCHQA_DATASET_SERVER_ENDPOINT,
                "source_main_revision": source.source_main_revision,
                "queried_splits": list(source.queried_splits),
                "requested_id_count": len(requested_ids),
                "received_id_count": len(source.rows),
            },
            "official_manifest_sha256": {
                "train": OFFICIAL_SEARCHQA_ID_MANIFEST_SHA256["train"],
                "selection": OFFICIAL_SEARCHQA_ID_MANIFEST_SHA256["selection"],
                "test_commitment": OFFICIAL_SEARCHQA_ID_MANIFEST_SHA256["test"],
            },
            "manifest_files": {
                "train": {
                    "path": str(copied_train_manifest),
                    "sha256": _sha256(copied_train_manifest),
                },
                "selection": {
                    "path": str(copied_selection_manifest),
                    "sha256": _sha256(copied_selection_manifest),
                },
            },
            "sample": {
                "seed": args.seed,
                "train_limit": args.train_size,
                "selection_limit": args.selection_size,
            },
            "counts": {
                "train": len(selected["train"]),
                "selection": len(selected["selection"]),
            },
            "outputs": {
                "train": {"path": str(train_path), "sha256": _sha256(train_path)},
                "selection": {
                    "path": str(selection_path),
                    "sha256": _sha256(selection_path),
                },
            },
            "test_payload_status": "not_materialized",
        }
        receipt_path = output_dir / "materialization-receipt.json"
        receipt_path.write_text(
            json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        verify_searchqa_materialization_receipt(
            receipt_path,
            train_path=train_path,
            selection_path=selection_path,
        )
    except Exception as error:
        print(str(error), file=sys.stderr)
        return 2
    print(str(args.output_dir.resolve() / "materialization-receipt.json"))
    return 0


def _load_ids(path: Path) -> tuple[str, ...]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if type(payload) is not list:
        raise ValueError(f"official SearchQA manifest must be a list: {path}")
    ids = tuple(str(item["id"]) for item in payload)
    if len(ids) != len(set(ids)):
        raise ValueError(f"official SearchQA manifest contains duplicate ids: {path}")
    return ids


def _require_hash(path: Path, expected: str) -> None:
    if not path.is_file() or _sha256(path) != expected:
        raise ValueError(f"official SearchQA manifest hash mismatch: {path}")


def _write_items(path: Path, items) -> None:
    path.write_text(
        json.dumps(
            [item.to_mapping() for item in items],
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
