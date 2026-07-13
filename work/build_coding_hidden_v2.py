#!/usr/bin/env python3
"""Build coding-hidden-v2 and seal its final test split."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from textskill_optimizer.locked_eval import seal_directory


DEFAULT_OUT = ROOT / "examples/coding-hidden-v2"
RUNNER_SOURCE = ROOT / "examples/coding-hidden/run_hidden_tests.py"


@dataclass(frozen=True)
class RenderedTask:
    family: str
    split: str
    variant: int
    fixture: str
    module: str
    function: str
    contract_tags: tuple[str, ...]
    source: str
    public_test: str
    hidden_test: str


Renderer = Callable[[str, int], tuple[str, str, str, str]]


FAMILY_FUNCTIONS: dict[str, tuple[str, str, str, str]] = {
    "batching": ("partition_items", "chunk_records", "batch_jobs", "group_pages"),
    "overlay": ("merge_settings", "overlay_config", "apply_overrides", "resolve_options"),
    "ledger": ("running_balance", "apply_ledger", "account_total", "inventory_level"),
    "intervals": ("merge_windows", "coalesce_ranges", "combine_spans", "compact_intervals"),
    "dependencies": ("dependency_order", "build_order", "install_order", "release_order"),
    "allocation": ("allocate_units", "split_capacity", "apportion_seats", "distribute_tokens"),
    "headers": ("unique_headers", "normalize_columns", "prepare_headings", "dedupe_labels"),
    "grouping": ("group_sums", "rollup_amounts", "category_totals", "bucket_values"),
    "backoff": ("backoff_delays", "retry_schedule", "poll_delays", "cooldown_steps"),
    "moving-average": ("moving_average", "rolling_mean", "window_average", "sliding_mean"),
}

CONTRACT_TAG_VOCAB: tuple[str, ...] = (
    "falsy_preservation",
    "immutability",
    "input_validation",
    "largest_remainder",
    "numeric_filtering",
    "stable_order",
    "topological_order",
    "unicode_casefold",
    "window_bounds",
)

FAMILY_CONTRACT_TAGS: dict[str, tuple[str, ...]] = {
    "allocation": ("largest_remainder", "input_validation", "stable_order"),
    "backoff": ("input_validation", "window_bounds"),
    "batching": ("input_validation", "immutability", "stable_order"),
    "dependencies": ("topological_order", "stable_order", "input_validation"),
    "grouping": ("numeric_filtering", "stable_order", "immutability"),
    "headers": ("unicode_casefold", "stable_order", "input_validation"),
    "intervals": ("stable_order", "window_bounds", "immutability"),
    "ledger": ("numeric_filtering", "immutability"),
    "moving-average": ("input_validation", "window_bounds", "immutability"),
    "overlay": ("falsy_preservation", "immutability"),
}


def clean(text: str) -> str:
    return textwrap.dedent(text).lstrip()


def render_benchmark_tasks() -> dict[str, list[RenderedTask]]:
    tasks = {"train": [], "selection": [], "test": []}
    for family, functions in FAMILY_FUNCTIONS.items():
        renderer = RENDERERS[family]
        split_variants = (("train", 0), ("selection", 1), ("test", 2), ("test", 3))
        for split, variant in split_variants:
            function = functions[variant]
            module, source, public_test, hidden_test = renderer(function, variant)
            tasks[split].append(
                RenderedTask(
                    family=family,
                    split=split,
                    variant=variant,
                    fixture=f"{split}-{family}-{variant + 1}",
                    module=module,
                    function=function,
                    contract_tags=FAMILY_CONTRACT_TAGS[family],
                    source=source,
                    public_test=public_test,
                    hidden_test=hidden_test,
                )
            )
    return tasks


def render_batching(function: str, variant: int) -> tuple[str, str, str, str]:
    module = "batches"
    count = 5 + variant
    size = 2 + variant % 2
    values = list(range(1, count + 1))
    expected = [values[index : index + size] for index in range(0, len(values), size)]
    source = clean(
        f'''\
        def {function}(items: list, size: int) -> list[list]:
            """Return fresh consecutive chunks; reject non-positive sizes and never mutate items."""
            return [list(items)]
        '''
    )
    public = clean(
        f'''\
        import unittest
        from app.{module} import {function}


        class BatchingTests(unittest.TestCase):
            def test_splits_consecutive_items(self):
                self.assertEqual({function}({values!r}, {size}), {expected!r})


        if __name__ == "__main__":
            unittest.main()
        '''
    )
    hidden = clean(
        f'''\
        import unittest
        from app.{module} import {function}


        class HiddenBatchingTests(unittest.TestCase):
            def test_empty_and_oversized_chunks(self):
                self.assertEqual({function}([], 3), [])
                self.assertEqual({function}([1, 2], 9), [[1, 2]])

            def test_rejects_non_positive_size(self):
                with self.assertRaises(ValueError):
                    {function}([1], 0)

            def test_does_not_mutate_input(self):
                values = [1, 2, 3]
                {function}(values, 2)
                self.assertEqual(values, [1, 2, 3])
        '''
    )
    return module, source, public, hidden


def render_overlay(function: str, variant: int) -> tuple[str, str, str, str]:
    module = "config"
    source = clean(
        f'''\
        def {function}(base: dict, overrides: dict) -> dict:
            """Return a new mapping where non-None overrides win without mutating either input."""
            result = base
            result.update(overrides)
            return result
        '''
    )
    public = clean(
        f'''\
        import unittest
        from app.{module} import {function}


        class OverlayTests(unittest.TestCase):
            def test_non_none_overrides_win(self):
                self.assertEqual(
                    {function}({{"host": "a", "port": 80}}, {{"port": {8080 + variant}, "host": None}}),
                    {{"host": "a", "port": {8080 + variant}}},
                )


        if __name__ == "__main__":
            unittest.main()
        '''
    )
    hidden = clean(
        f'''\
        import unittest
        from app.{module} import {function}


        class HiddenOverlayTests(unittest.TestCase):
            def test_keeps_falsey_values_except_none(self):
                self.assertEqual(
                    {function}({{"enabled": True, "count": 3}}, {{"enabled": False, "count": 0, "extra": None}}),
                    {{"enabled": False, "count": 0}},
                )

            def test_does_not_mutate_inputs(self):
                base = {{"a": 1}}
                overrides = {{"a": 2, "b": 3}}
                result = {function}(base, overrides)
                self.assertEqual(base, {{"a": 1}})
                self.assertEqual(overrides, {{"a": 2, "b": 3}})
                self.assertIsNot(result, base)
        '''
    )
    return module, source, public, hidden


def render_ledger(function: str, variant: int) -> tuple[str, str, str, str]:
    module = "ledger"
    start = 10 * (variant + 1)
    source = clean(
        f'''\
        def {function}(start: int | float, events: list[dict]) -> int | float:
            """Add numeric amount fields, skipping missing, boolean, or non-numeric amounts."""
            return start
        '''
    )
    public = clean(
        f'''\
        import unittest
        from app.{module} import {function}


        class LedgerTests(unittest.TestCase):
            def test_applies_positive_and_negative_amounts(self):
                events = [{{"amount": 5}}, {{"amount": -2}}, {{"note": "skip"}}]
                self.assertEqual({function}({start}, events), {start + 3})


        if __name__ == "__main__":
            unittest.main()
        '''
    )
    hidden = clean(
        f'''\
        import unittest
        from app.{module} import {function}


        class HiddenLedgerTests(unittest.TestCase):
            def test_skips_boolean_and_malformed_amounts(self):
                events = [{{"amount": True}}, {{"amount": "5"}}, {{}}, {{"amount": 1.5}}]
                self.assertEqual({function}(2, events), 3.5)

            def test_does_not_mutate_events(self):
                events = [{{"amount": -3}}, {{"amount": 8}}]
                snapshot = [dict(item) for item in events]
                {function}(4, events)
                self.assertEqual(events, snapshot)
        '''
    )
    return module, source, public, hidden


def render_intervals(function: str, variant: int) -> tuple[str, str, str, str]:
    module = "intervals"
    shift = variant * 2
    source = clean(
        f'''\
        def {function}(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
            """Normalize endpoints, sort, and merge overlapping or touching closed intervals."""
            return list(intervals)
        '''
    )
    public = clean(
        f'''\
        import unittest
        from app.{module} import {function}


        class IntervalTests(unittest.TestCase):
            def test_merges_unsorted_overlaps(self):
                values = [({5 + shift}, {8 + shift}), ({1 + shift}, {3 + shift}), ({2 + shift}, {6 + shift})]
                self.assertEqual({function}(values), [({1 + shift}, {8 + shift})])


        if __name__ == "__main__":
            unittest.main()
        '''
    )
    hidden = clean(
        f'''\
        import unittest
        from app.{module} import {function}


        class HiddenIntervalTests(unittest.TestCase):
            def test_normalizes_and_merges_touching_intervals(self):
                self.assertEqual({function}([(5, 3), (5, 7), (10, 9), (7, 9)]), [(3, 10)])

            def test_empty_and_input_preservation(self):
                self.assertEqual({function}([]), [])
                values = [(3, 1), (8, 9)]
                {function}(values)
                self.assertEqual(values, [(3, 1), (8, 9)])
        '''
    )
    return module, source, public, hidden


def render_dependencies(function: str, variant: int) -> tuple[str, str, str, str]:
    module = "dependencies"
    root = chr(ord("a") + variant)
    child = chr(ord("k") + variant)
    source = clean(
        f'''\
        def {function}(graph: dict[str, list[str]]) -> list[str]:
            """Return lexicographically stable dependency-first order; include referenced nodes and reject cycles."""
            return list(graph)
        '''
    )
    public = clean(
        f'''\
        import unittest
        from app.{module} import {function}


        class DependencyTests(unittest.TestCase):
            def test_dependencies_come_first(self):
                graph = {{"{child}": ["{root}"], "{root}": []}}
                self.assertEqual({function}(graph), ["{root}", "{child}"])


        if __name__ == "__main__":
            unittest.main()
        '''
    )
    hidden = clean(
        f'''\
        import unittest
        from app.{module} import {function}


        class HiddenDependencyTests(unittest.TestCase):
            def test_includes_referenced_only_nodes_and_stable_ties(self):
                graph = {{"build": ["core"], "docs": [], "test": ["core"]}}
                self.assertEqual({function}(graph), ["core", "build", "docs", "test"])

            def test_rejects_cycles(self):
                with self.assertRaises(ValueError):
                    {function}({{"a": ["b"], "b": ["a"]}})
        '''
    )
    return module, source, public, hidden


def render_allocation(function: str, variant: int) -> tuple[str, str, str, str]:
    module = "allocation"
    total = 9 + variant
    expected = largest_remainder(total, [1, 2, 1])
    source = clean(
        f'''\
        def {function}(total: int, weights: list[int | float]) -> list[int]:
            """Allocate all non-negative integer units by largest remainder, breaking ties by lower index."""
            if not weights:
                return []
            share = total // len(weights)
            return [share for _ in weights]
        '''
    )
    public = clean(
        f'''\
        import unittest
        from app.{module} import {function}


        class AllocationTests(unittest.TestCase):
            def test_allocates_proportionally(self):
                self.assertEqual({function}({total}, [1, 2, 1]), {expected!r})


        if __name__ == "__main__":
            unittest.main()
        '''
    )
    hidden = clean(
        f'''\
        import unittest
        from app.{module} import {function}


        class HiddenAllocationTests(unittest.TestCase):
            def test_distributes_tied_remainders_by_index(self):
                self.assertEqual({function}(2, [1, 1, 1]), [1, 1, 0])

            def test_zero_weights_and_invalid_inputs(self):
                self.assertEqual({function}(5, [0, 0]), [0, 0])
                with self.assertRaises(ValueError):
                    {function}(-1, [1])
                with self.assertRaises(ValueError):
                    {function}(3, [1, -1])
        '''
    )
    return module, source, public, hidden


def largest_remainder(total: int, weights: list[int]) -> list[int]:
    weight_sum = sum(weights)
    quotas = [total * weight / weight_sum for weight in weights]
    result = [int(quota) for quota in quotas]
    remaining = total - sum(result)
    order = sorted(range(len(weights)), key=lambda index: (-(quotas[index] - result[index]), index))
    for index in order[:remaining]:
        result[index] += 1
    return result


def render_headers(function: str, variant: int) -> tuple[str, str, str, str]:
    module = "headers"
    base = ("Name", "Region", "Status", "Owner")[variant]
    source = clean(
        f'''\
        def {function}(headers: list[str]) -> list[str]:
            """Trim labels, replace blanks with column, and add _2/_3 suffixes for case-insensitive duplicates."""
            return [header.strip() for header in headers]
        '''
    )
    public = clean(
        f'''\
        import unittest
        from app.{module} import {function}


        class HeaderTests(unittest.TestCase):
            def test_suffixes_duplicate_headers(self):
                self.assertEqual({function}(["{base}", " {base.lower()} "]), ["{base}", "{base.lower()}_2"])


        if __name__ == "__main__":
            unittest.main()
        '''
    )
    hidden = clean(
        f'''\
        import unittest
        from app.{module} import {function}


        class HiddenHeaderTests(unittest.TestCase):
            def test_handles_blanks_and_third_duplicates(self):
                self.assertEqual(
                    {function}(["", "  ", "Name", "name", "NAME"]),
                    ["column", "column_2", "Name", "name_2", "NAME_3"],
                )

            def test_uses_unicode_casefold(self):
                self.assertEqual({function}(["Straße", "STRASSE"]), ["Straße", "STRASSE_2"])
        '''
    )
    return module, source, public, hidden


def render_grouping(function: str, variant: int) -> tuple[str, str, str, str]:
    module = "grouping"
    first = 2 + variant
    source = clean(
        f'''\
        def {function}(rows: list[dict], group_key: str, value_key: str) -> dict:
            """Sum numeric non-boolean values by group in first-seen order, skipping malformed rows."""
            return {{}}
        '''
    )
    public = clean(
        f'''\
        import unittest
        from app.{module} import {function}


        class GroupingTests(unittest.TestCase):
            def test_groups_numeric_values(self):
                rows = [{{"team": "a", "points": {first}}}, {{"team": "b", "points": 3}}, {{"team": "a", "points": 4}}]
                self.assertEqual({function}(rows, "team", "points"), {{"a": {first + 4}, "b": 3}})


        if __name__ == "__main__":
            unittest.main()
        '''
    )
    hidden = clean(
        f'''\
        import unittest
        from app.{module} import {function}


        class HiddenGroupingTests(unittest.TestCase):
            def test_skips_malformed_and_boolean_values(self):
                rows = [{{"g": "x", "v": 1.5}}, {{"g": "x", "v": True}}, {{"v": 9}}, {{"g": "y", "v": -2}}]
                self.assertEqual({function}(rows, "g", "v"), {{"x": 1.5, "y": -2}})

            def test_preserves_inputs_and_first_seen_order(self):
                rows = [{{"g": "z", "v": 1}}, {{"g": "a", "v": 2}}]
                snapshot = [dict(row) for row in rows]
                result = {function}(rows, "g", "v")
                self.assertEqual(list(result), ["z", "a"])
                self.assertEqual(rows, snapshot)
        '''
    )
    return module, source, public, hidden


def render_backoff(function: str, variant: int) -> tuple[str, str, str, str]:
    module = "backoff"
    base = variant + 1
    cap = base * 3
    source = clean(
        f'''\
        def {function}(base: int | float, attempts: int, cap: int | float) -> list[int | float]:
            """Return capped exponential delays; non-positive attempts are empty and negative base/cap is invalid."""
            return [base * (2 ** index) for index in range(attempts)]
        '''
    )
    public = clean(
        f'''\
        import unittest
        from app.{module} import {function}


        class BackoffTests(unittest.TestCase):
            def test_caps_exponential_delays(self):
                self.assertEqual({function}({base}, 4, {cap}), [{base}, {base * 2}, {cap}, {cap}])


        if __name__ == "__main__":
            unittest.main()
        '''
    )
    hidden = clean(
        f'''\
        import unittest
        from app.{module} import {function}


        class HiddenBackoffTests(unittest.TestCase):
            def test_empty_attempts_and_cap_below_base(self):
                self.assertEqual({function}(2, 0, 8), [])
                self.assertEqual({function}(5, 3, 2), [2, 2, 2])

            def test_rejects_negative_base_or_cap(self):
                with self.assertRaises(ValueError):
                    {function}(-1, 2, 4)
                with self.assertRaises(ValueError):
                    {function}(1, 2, -4)
        '''
    )
    return module, source, public, hidden


def render_moving_average(function: str, variant: int) -> tuple[str, str, str, str]:
    module = "averages"
    values = [variant + 1, variant + 3, variant + 5, variant + 7]
    expected = [(values[index] + values[index + 1]) / 2 for index in range(3)]
    source = clean(
        f'''\
        def {function}(values: list[int | float], width: int) -> list[float]:
            """Return averages for complete consecutive windows; require a positive width and preserve input."""
            if not values:
                return []
            return [sum(values) / len(values)]
        '''
    )
    public = clean(
        f'''\
        import unittest
        from app.{module} import {function}


        class MovingAverageTests(unittest.TestCase):
            def test_returns_complete_window_averages(self):
                self.assertEqual({function}({values!r}, 2), {expected!r})


        if __name__ == "__main__":
            unittest.main()
        '''
    )
    hidden = clean(
        f'''\
        import unittest
        from app.{module} import {function}


        class HiddenMovingAverageTests(unittest.TestCase):
            def test_width_one_and_width_larger_than_input(self):
                self.assertEqual({function}([2, 4], 1), [2.0, 4.0])
                self.assertEqual({function}([2, 4], 3), [])

            def test_rejects_non_positive_width_and_preserves_input(self):
                with self.assertRaises(ValueError):
                    {function}([1], 0)
                values = [1, 2, 3]
                {function}(values, 2)
                self.assertEqual(values, [1, 2, 3])
        '''
    )
    return module, source, public, hidden


RENDERERS: dict[str, Renderer] = {
    "batching": render_batching,
    "overlay": render_overlay,
    "ledger": render_ledger,
    "intervals": render_intervals,
    "dependencies": render_dependencies,
    "allocation": render_allocation,
    "headers": render_headers,
    "grouping": render_grouping,
    "backoff": render_backoff,
    "moving-average": render_moving_average,
}


def write_split(root: Path, split: str, tasks: list[RenderedTask], *, write_fixtures: bool = True) -> None:
    lines = []
    for task in tasks:
        if write_fixtures:
            write_fixture(root, task)
        payload = {
            "id": f"coding-hidden-v2-{split}-{task.family}-{task.variant + 1}",
            "input": "Fix the failing public tests without editing tests. Implement the full documented contract.",
            "expected": {"tests_passed": True},
            "metadata": {
                "repo": f"fixtures/{task.fixture}",
                "test_command": f"python3 {{task_dir}}/run_hidden_tests.py {task.fixture} {{repo}}",
                "agent_test_command": "python3 -m unittest discover -s tests",
                "timeout_seconds": 360,
                "benchmark_family": task.family,
                "benchmark_split": split,
                "benchmark_variant": task.variant + 1,
                "contract_tags": list(task.contract_tags),
            },
        }
        lines.append(json.dumps(payload, separators=(",", ":")))
    (root / f"{split}.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_fixture(root: Path, task: RenderedTask) -> None:
    fixture = root / "fixtures" / task.fixture
    app = fixture / "app"
    tests = fixture / "tests"
    hidden = root / "hidden" / task.fixture
    app.mkdir(parents=True)
    tests.mkdir(parents=True)
    hidden.mkdir(parents=True)
    (app / "__init__.py").write_text("", encoding="utf-8")
    (app / f"{task.module}.py").write_text(task.source, encoding="utf-8")
    (tests / f"test_{task.module}.py").write_text(task.public_test, encoding="utf-8")
    (hidden / f"test_{task.module}_hidden.py").write_text(task.hidden_test, encoding="utf-8")


def write_protocol_files(root: Path, tasks: dict[str, list[RenderedTask]]) -> None:
    shutil.copy2(RUNNER_SOURCE, root / "run_hidden_tests.py")
    (root / "skill.md").write_text(
        clean(
            '''\
            # Coding Repair Skill

            Fix implementation defects without editing tests. Read the documented contract, inspect the current implementation, run public tests, and preserve unrelated behavior.
            '''
        ),
        encoding="utf-8",
    )
    protocol = {
        "benchmark": "coding-hidden-v2",
        "protocol_version": 1,
        "scoring_unit": ["task_accuracy", "family_macro_accuracy", "contract_macro_accuracy"],
        "contract_tags": list(CONTRACT_TAG_VOCAB),
        "family_count": len(FAMILY_FUNCTIONS),
        "train_tasks": len(tasks["train"]),
        "selection_tasks": len(tasks["selection"]),
        "locked_test_tasks": len(tasks["test"]),
        "test_access_policy": "single final attempt through textskill_optimizer.locked_eval",
        "families": sorted(FAMILY_FUNCTIONS),
    }
    (root / "protocol.json").write_text(
        json.dumps(protocol, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (root / "README.md").write_text(
        clean(
            '''\
            # coding-hidden-v2

            This benchmark has ten capability families. Development exposes one train and one selection task per family. Two additional variants per family are stored only in `test.enc`.

            Rules:

            - Optimize only with `train.jsonl`.
            - Accept or reject edits only with `selection.jsonl`.
            - Do not decrypt or evaluate `test.enc` during development.
            - Run harness health checks on development tasks before final evaluation.
            - Final evaluation must use `python3 -m textskill_optimizer.locked_eval run` and writes a one-attempt receipt even when the child command fails.
            - Report task accuracy, family macro accuracy, and contract macro accuracy.

            The key file is intentionally outside the repository. The lock prevents accidental evaluation and creates an auditable commitment; it does not defend against a malicious workspace owner.
            '''
        ),
        encoding="utf-8",
    )


def build_benchmark(output: Path, key_file: Path) -> dict:
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite benchmark: {output}")
    output.mkdir(parents=True)
    tasks = render_benchmark_tasks()
    write_split(output, "train", tasks["train"])
    write_split(output, "selection", tasks["selection"])
    write_protocol_files(output, tasks)

    with tempfile.TemporaryDirectory(prefix="coding-hidden-v2-test-") as tmp:
        test_root = Path(tmp)
        write_split(test_root, "test", tasks["test"])
        shutil.copy2(RUNNER_SOURCE, test_root / "run_hidden_tests.py")
        lock = seal_directory(
            test_root,
            output / "test.enc",
            key_file,
            output / "test.lock.json",
            details={
                "benchmark": "coding-hidden-v2",
                "task_file": "test.jsonl",
                "task_count": len(tasks["test"]),
                "family_count": len(FAMILY_FUNCTIONS),
                "tasks_per_family": 2,
            },
        )
    return lock


def sync_development_metadata(output: Path) -> None:
    if not output.exists():
        raise FileNotFoundError(f"Benchmark does not exist: {output}")
    tasks = render_benchmark_tasks()
    write_split(output, "train", tasks["train"], write_fixtures=False)
    write_split(output, "selection", tasks["selection"], write_fixtures=False)
    write_protocol_files(output, tasks)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--key-file", type=Path)
    parser.add_argument(
        "--sync-development-metadata",
        action="store_true",
        help="Refresh train/selection manifests and protocol metadata without resealing locked test data.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.sync_development_metadata:
        sync_development_metadata(args.out.resolve())
        print(f"synced_development_metadata={args.out.resolve()}")
        return 0
    if args.key_file is None:
        raise SystemExit("--key-file is required unless --sync-development-metadata is set")
    lock = build_benchmark(args.out.resolve(), args.key_file.expanduser().resolve())
    print(json.dumps(lock, indent=2, sort_keys=True))
    print(f"key_file={args.key_file.expanduser().resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
