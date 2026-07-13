#!/usr/bin/env python3
"""Validate coding-hidden-v2 development artifacts without opening locked test data."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BENCHMARK = ROOT / "examples/coding-hidden-v2"
LOWER_SNAKE_CASE = re.compile(r"^[a-z][a-z0-9]*(?:_[a-z0-9]+)*$")


def validate_benchmark(benchmark: Path) -> dict[str, Any]:
    protocol = json.loads((benchmark / "protocol.json").read_text(encoding="utf-8"))
    contract_vocab = set(protocol.get("contract_tags") or [])
    if not contract_vocab:
        raise ValueError("protocol.json must declare non-empty contract_tags")
    lock = json.loads((benchmark / "test.lock.json").read_text(encoding="utf-8"))
    archive = benchmark / "test.enc"
    actual_hash = hashlib.sha256(archive.read_bytes()).hexdigest()
    if actual_hash != lock.get("archive_sha256"):
        raise ValueError("test.enc does not match test.lock.json")
    if (benchmark / "test.jsonl").exists():
        raise ValueError("Plaintext test.jsonl must not exist")

    family_counts: dict[str, Counter[str]] = {}
    checked_tasks = 0
    for split in ("train", "selection"):
        counts: Counter[str] = Counter()
        manifest = benchmark / f"{split}.jsonl"
        for line in manifest.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            task = json.loads(line)
            metadata = task["metadata"]
            contract_tags = metadata.get("contract_tags")
            if not isinstance(contract_tags, list) or not contract_tags:
                raise ValueError(f"Task is missing contract_tags: {task['id']}")
            for tag in contract_tags:
                if not isinstance(tag, str) or not LOWER_SNAKE_CASE.match(tag):
                    raise ValueError(f"Invalid contract tag format in {task['id']}: {tag!r}")
                if tag not in contract_vocab:
                    raise ValueError(f"Unknown contract tag in {task['id']}: {tag}")
            fixture = Path(metadata["repo"]).name
            repo = benchmark / metadata["repo"]
            hidden = benchmark / "hidden" / fixture
            if not repo.is_dir() or not hidden.is_dir():
                raise ValueError(f"Task paths are incomplete: {task['id']}")
            completed = subprocess.run(
                [sys.executable, str(benchmark / "run_hidden_tests.py"), fixture, str(repo)],
                text=True,
                capture_output=True,
                check=False,
            )
            if completed.returncode == 0:
                raise ValueError(f"Initial fixture unexpectedly passes: {task['id']}")
            counts[str(metadata["benchmark_family"])] += 1
            checked_tasks += 1
        family_counts[split] = counts

    expected_families = set(protocol["families"])
    for split, counts in family_counts.items():
        if set(counts) != expected_families or set(counts.values()) != {1}:
            raise ValueError(f"Unbalanced {split} family coverage: {dict(counts)}")

    return {
        "benchmark": protocol["benchmark"],
        "checked_development_tasks": checked_tasks,
        "train_families": len(family_counts["train"]),
        "selection_families": len(family_counts["selection"]),
        "locked_test_tasks": lock["details"]["task_count"],
        "contract_tags": sorted(contract_vocab),
        "archive_sha256": actual_hash,
        "plaintext_test_absent": True,
        "all_initial_fixtures_fail": True,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = validate_benchmark(args.benchmark.resolve())
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
