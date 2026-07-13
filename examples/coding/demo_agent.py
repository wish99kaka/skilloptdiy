"""A tiny skill-sensitive coding agent for offline examples.

Real use should pass a Codex, Claude Code, or custom agent command instead.
This script exists only to make the optimization loop testable without network
access or external credentials.
"""

from __future__ import annotations

import os
from pathlib import Path


REQUIRED_SKILL_MARKER = "test-guided root-cause loop"
FIX_MARKER = "# TEXTSKILL_FIX:"


def main() -> int:
    repo = Path(os.environ["TEXTSKILL_REPO_DIR"])
    skill = Path(os.environ["TEXTSKILL_SKILL_PATH"]).read_text(encoding="utf-8").casefold()
    if not should_apply_skill(skill):
        print("No test/root-cause/source-fix loop in skill; leaving repo unchanged.")
        return 0

    fixed = 0
    for path in sorted((repo / "app").rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        rewritten, count = rewrite_marked_fixes(text)
        if count:
            path.write_text(rewritten, encoding="utf-8")
            fixed += count
    print(f"fixed_markers={fixed}")
    return 0


def rewrite_marked_fixes(text: str) -> tuple[str, int]:
    output: list[str] = []
    fixed = 0
    for line in text.splitlines(keepends=True):
        if FIX_MARKER not in line:
            output.append(line)
            continue
        prefix, replacement = line.split(FIX_MARKER, 1)
        indentation = prefix[: len(prefix) - len(prefix.lstrip())]
        newline = "\n" if line.endswith("\n") else ""
        output.append(indentation + replacement.strip() + newline)
        fixed += 1
    return "".join(output), fixed


def should_apply_skill(skill: str) -> bool:
    normalized = skill.casefold()
    if REQUIRED_SKILL_MARKER in normalized:
        return True
    has_test_signal = "test" in normalized and (
        "fail" in normalized or "failure" in normalized
    )
    has_root_cause_signal = (
        "root cause" in normalized
        or "diagnose" in normalized
        or "responsible" in normalized
        or "directly addresses" in normalized
        or "discrepancy" in normalized
    )
    has_source_fix_signal = (
        "source code" in normalized
        or "implementation" in normalized
        or "without editing tests" in normalized
    )
    has_minimal_change_signal = (
        "minimal" in normalized
        or "targeted" in normalized
        or "small" in normalized
    )
    return (
        has_test_signal
        and has_root_cause_signal
        and has_source_fix_signal
        and has_minimal_change_signal
    )


if __name__ == "__main__":
    raise SystemExit(main())
