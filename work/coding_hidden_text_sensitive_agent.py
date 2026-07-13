"""Text-sensitive deterministic agent for coding-hidden replay experiments."""

from __future__ import annotations

import json
import os
from pathlib import Path

from coding_hidden_deterministic_agent import (
    FULL_MARKER,
    SOLUTIONS,
    TRAIN_FIXTURES,
    TRAIN_MARKER,
    apply_solution,
    fixture_name_from_task,
)


CAPABILITY_FIXTURES = {
    "keyed_dedupe": {"dedupe-by-email", "unique-by-id", "dedupe-casefold"},
    "range_bounds": {"number-range", "date-range"},
    "nested_path": {"nested-default", "nested-pluck", "safe-nested-get"},
    "token_parser": {"money-parser", "parse-duration", "parse-int-list"},
    "rounding": {"round-cents", "round-tax"},
    "sort_missing_last": {"stable-sort-events"},
    "stable_numeric_sort": {"stable-sort"},
    "slug_normalization": {"slug-normalizer"},
}


def main() -> int:
    repo = Path(os.environ["TEXTSKILL_REPO_DIR"])
    skill = Path(os.environ["TEXTSKILL_SKILL_PATH"]).read_text(encoding="utf-8")
    fixture = fixture_name_from_task(Path(os.environ["TEXTSKILL_TASK_PATH"]))

    if FULL_MARKER in skill:
        apply_solution(repo, fixture)
        return 0
    if TRAIN_MARKER in skill and fixture in TRAIN_FIXTURES:
        apply_solution(repo, fixture)
        return 0

    capabilities = infer_capabilities(skill)
    for capability in sorted(capabilities):
        if fixture in CAPABILITY_FIXTURES[capability]:
            apply_solution(repo, fixture)
            print(f"text-sensitive capability={capability}")
            return 0

    print(
        "no text-sensitive deterministic rule "
        f"for fixture={fixture} capabilities={json.dumps(sorted(capabilities))}"
    )
    return 0


def infer_capabilities(skill_text: str) -> set[str]:
    text = normalize(skill_text)
    capabilities: set[str] = set()

    if contains_any(text, "dedupe", "de duplication", "de-duplication", "unique by key", "unique-by-key"):
        if contains_any(text, "missing key", "missing the key", "no key", "no-key", "missing dedupe key"):
            if contains_any(text, "independent", "preserve", "keep"):
                capabilities.add("keyed_dedupe")
    if "records missing the deduplication key" in text:
        capabilities.add("keyed_dedupe")

    if contains_any(
        text,
        "reversed bounds",
        "normalize bounds",
        "normalize lower",
        "lower upper bounds",
    ):
        capabilities.add("range_bounds")
    if "inclusive bounds" in text and "range" in text:
        capabilities.add("range_bounds")

    if contains_any(text, "nested", "path segment", "path utility", "list index", "list indexes"):
        if contains_any(text, "missing", "defensive", "bounds", "skip", "default"):
            capabilities.add("nested_path")

    if contains_any(text, "separator", "separators", "malformed input", "malformed inputs"):
        capabilities.add("token_parser")
    if "units" in text and contains_any(text, "parse", "process", "conversion", "conversions"):
        capabilities.add("token_parser")

    if contains_any(text, "rounding", "round ", "round."):
        capabilities.add("rounding")

    if contains_any(
        text,
        "missing sort key",
        "sort key",
        "missing key last",
        "missing keys last",
        "missing values last",
        "missing timestamps",
        "timestamps last",
    ):
        if contains_any(text, "last", "stable", "ascending", "safe key"):
            capabilities.add("sort_missing_last")
    if "sort" in text and "stable" in text:
        capabilities.add("stable_numeric_sort")

    if contains_any(text, "slug", "separator normalization"):
        capabilities.add("slug_normalization")

    return capabilities


def normalize(value: str) -> str:
    return (
        value.casefold()
        .replace("_", " ")
        .replace("-", " ")
        .replace("/", " ")
        .replace("null", "none")
    )


def contains_any(text: str, *needles: str) -> bool:
    return any(needle in text for needle in needles)


if __name__ == "__main__":
    raise SystemExit(main())
