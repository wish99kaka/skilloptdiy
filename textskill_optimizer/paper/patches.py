"""Sequential, replayable patch application for paper-mode skill updates."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from .types import PaperEdit, PaperEditOperation


SLOW_UPDATE_START = "<!-- SLOW_UPDATE_START -->"
SLOW_UPDATE_END = "<!-- SLOW_UPDATE_END -->"


@dataclass(frozen=True)
class EditApplyReport:
    index: int
    edit_id: str
    operation: PaperEditOperation
    status: str
    before_sha256: str
    after_sha256: str


@dataclass(frozen=True)
class PatchApplyResult:
    input_sha256: str
    output_skill: str
    output_sha256: str
    reports: tuple[EditApplyReport, ...]


def apply_paper_patch(
    skill_text: str,
    edits: tuple[PaperEdit, ...],
) -> PatchApplyResult:
    """Apply exact paper edits in order and return a hash-chained report."""

    if type(skill_text) is not str or not skill_text.strip():
        raise ValueError("paper patch requires skill_text")
    if type(edits) is not tuple or any(type(item) is not PaperEdit for item in edits):
        raise ValueError("paper patch requires a tuple of exact PaperEdit values")
    input_sha256 = _sha256(skill_text)
    current = skill_text
    reports: list[EditApplyReport] = []
    for index, edit in enumerate(edits, 1):
        before_sha256 = _sha256(current)
        current, status = _apply_edit(current, edit)
        reports.append(
            EditApplyReport(
                index=index,
                edit_id=edit.edit_id,
                operation=edit.operation,
                status=status,
                before_sha256=before_sha256,
                after_sha256=_sha256(current),
            )
        )
    return PatchApplyResult(
        input_sha256=input_sha256,
        output_skill=current,
        output_sha256=_sha256(current),
        reports=tuple(reports),
    )


def write_slow_update_field(skill_text: str, content: str = "") -> str:
    """Create or exclusively replace the single protected slow-update field."""

    if type(skill_text) is not str or not skill_text.strip():
        raise ValueError("slow update requires skill_text")
    if type(content) is not str:
        raise ValueError("slow update content must be a string")
    if content.count(SLOW_UPDATE_START) or content.count(SLOW_UPDATE_END):
        raise ValueError("slow update content cannot contain protected markers")
    start_count = skill_text.count(SLOW_UPDATE_START)
    end_count = skill_text.count(SLOW_UPDATE_END)
    if start_count == 0 and end_count == 0:
        body = content.strip()
        return (
            skill_text.rstrip()
            + f"\n\n{SLOW_UPDATE_START}\n"
            + (body + "\n" if body else "")
            + f"{SLOW_UPDATE_END}\n"
        )
    if start_count != 1 or end_count != 1:
        raise ValueError("skill must contain exactly one complete slow-update field")
    start = skill_text.index(SLOW_UPDATE_START)
    end = skill_text.index(SLOW_UPDATE_END)
    if end < start + len(SLOW_UPDATE_START):
        raise ValueError("slow-update field markers are out of order")
    body = content.strip()
    return (
        skill_text[: start + len(SLOW_UPDATE_START)]
        + "\n"
        + (body + "\n" if body else "")
        + skill_text[end:]
    )


def read_slow_update_field(skill_text: str) -> str:
    """Return the exact guidance inside the one valid protected field."""

    normalized = write_slow_update_field(skill_text, "")
    if normalized.count(SLOW_UPDATE_START) != 1:
        raise AssertionError("normalized slow-update field is missing")
    start = skill_text.find(SLOW_UPDATE_START)
    end = skill_text.find(SLOW_UPDATE_END)
    if start == -1 or end == -1:
        return ""
    return skill_text[start + len(SLOW_UPDATE_START) : end].strip()


def _apply_edit(skill_text: str, edit: PaperEdit) -> tuple[str, str]:
    content = _strip_slow_markers(edit.content).strip()
    if edit.target and _target_is_protected(skill_text, edit.target):
        return skill_text, "skipped_protected_region"
    if edit.operation is PaperEditOperation.APPEND:
        return _append_before_slow_region(skill_text, content, "applied_append")
    if edit.operation is PaperEditOperation.INSERT_AFTER:
        if edit.target not in skill_text:
            return _append_before_slow_region(
                skill_text,
                content,
                "applied_insert_after_fallback_append",
            )
        target_end = skill_text.index(edit.target) + len(edit.target)
        if target_end > 0 and skill_text[target_end - 1] == "\n":
            insert_at = target_end
        else:
            newline = skill_text.find("\n", target_end)
            insert_at = newline + 1 if newline != -1 else len(skill_text)
        if _position_is_protected(skill_text, insert_at):
            return skill_text, "skipped_protected_region"
        return (
            skill_text[:insert_at]
            + "\n"
            + content
            + "\n"
            + skill_text[insert_at:],
            "applied_insert_after",
        )
    if edit.operation is PaperEditOperation.REPLACE:
        if edit.target not in skill_text:
            return skill_text, "skipped_replace_target_not_found"
        return skill_text.replace(edit.target, content, 1), "applied_replace"
    if edit.target not in skill_text:
        return skill_text, "skipped_delete_target_not_found"
    return skill_text.replace(edit.target, "", 1), "applied_delete"


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _append_before_slow_region(
    skill_text: str,
    content: str,
    status_without_region: str,
) -> tuple[str, str]:
    start = skill_text.find(SLOW_UPDATE_START)
    if start == -1:
        return skill_text.rstrip() + "\n\n" + content + "\n", status_without_region
    status = (
        "applied_append_before_protected_region"
        if status_without_region == "applied_append"
        else "applied_insert_after_fallback_before_protected_region"
    )
    return (
        skill_text[:start].rstrip()
        + "\n\n"
        + content
        + "\n\n"
        + skill_text[start:],
        status,
    )


def _target_is_protected(skill_text: str, target: str) -> bool:
    target_start = skill_text.find(target)
    if target_start == -1:
        return False
    target_end = target_start + len(target)
    return any(
        target_start < protected_end and target_end > protected_start
        for protected_start, protected_end in _protected_ranges(skill_text)
    )


def _position_is_protected(skill_text: str, position: int) -> bool:
    return any(
        protected_start < position < protected_end
        for protected_start, protected_end in _protected_ranges(skill_text)
    )


def _protected_ranges(skill_text: str) -> tuple[tuple[int, int], ...]:
    ranges: list[tuple[int, int]] = []
    cursor = 0
    while True:
        protected_start = skill_text.find(SLOW_UPDATE_START, cursor)
        if protected_start == -1:
            return tuple(ranges)
        protected_end = skill_text.find(
            SLOW_UPDATE_END,
            protected_start + len(SLOW_UPDATE_START),
        )
        if protected_end == -1:
            ranges.append((protected_start, len(skill_text)))
            return tuple(ranges)
        protected_end += len(SLOW_UPDATE_END)
        ranges.append((protected_start, protected_end))
        cursor = protected_end


def _strip_slow_markers(value: str) -> str:
    return value.replace(SLOW_UPDATE_START, "").replace(SLOW_UPDATE_END, "")
