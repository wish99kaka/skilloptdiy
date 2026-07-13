"""Atomic skill-edit application, aggregation, and learning-rate clipping."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .models import AtomicEdit, EditProposal


SLOW_START = "<!-- SKILLOPT:SLOW_UPDATE:START -->"
SLOW_END = "<!-- SKILLOPT:SLOW_UPDATE:END -->"
END_TARGET = "__end__"


@dataclass(frozen=True)
class RankedEdit:
    edit: AtomicEdit
    support: int
    first_seen: int


@dataclass(frozen=True)
class MergeResult:
    selected: tuple[AtomicEdit, ...]
    ranked: tuple[RankedEdit, ...]
    duplicate_count: int
    conflict_count: int


def apply_atomic_edits(skill_text: str, edits: Iterable[AtomicEdit]) -> str:
    updated = skill_text
    for edit in edits:
        updated = apply_atomic_edit(updated, edit)
    return normalize_final_newline(updated)


def apply_atomic_edit(skill_text: str, edit: AtomicEdit) -> str:
    reject_protected_edit(skill_text, edit)
    if edit.operation == "add":
        content = edit.content.strip("\n")
        if edit.target in {"", END_TARGET}:
            return skill_text.rstrip() + "\n\n" + content + "\n"
        require_unique_target(skill_text, edit.target)
        return skill_text.replace(edit.target, edit.target + "\n" + content, 1)

    require_unique_target(skill_text, edit.target)
    if edit.operation == "delete":
        return skill_text.replace(edit.target, "", 1)
    return skill_text.replace(edit.target, edit.content, 1)


def require_unique_target(skill_text: str, target: str) -> None:
    count = skill_text.count(target)
    if count != 1:
        raise ValueError(f"Atomic edit target must occur exactly once; found {count}: {target!r}")


def reject_protected_edit(skill_text: str, edit: AtomicEdit) -> None:
    if SLOW_START in edit.target or SLOW_END in edit.target:
        raise ValueError("Atomic edits cannot target slow-update markers")
    if SLOW_START in edit.content or SLOW_END in edit.content:
        raise ValueError("Atomic edits cannot write slow-update markers")
    start = skill_text.find(SLOW_START)
    end = skill_text.find(SLOW_END)
    if start < 0 and end < 0:
        return
    if start < 0 or end < start:
        raise ValueError("Skill contains a malformed protected slow-update field")
    end += len(SLOW_END)
    if edit.target and edit.target not in {END_TARGET}:
        target_start = skill_text.find(edit.target)
        target_end = target_start + len(edit.target)
        if target_start >= 0 and target_start < end and target_end > start:
            raise ValueError("Atomic edits cannot modify the protected slow-update field")


def merge_and_rank_atomic_edits(
    proposals: Iterable[EditProposal],
    *,
    budget: int,
) -> MergeResult:
    if budget < 0:
        raise ValueError("Atomic edit budget must be non-negative")
    grouped: dict[tuple[str, str, str], RankedEdit] = {}
    duplicate_count = 0
    seen_index = 0
    for proposal in proposals:
        proposal_priority = float(proposal.metadata.get("priority", 0.0))
        for edit in proposal.edits:
            key = canonical_edit_key(edit)
            existing = grouped.get(key)
            effective = AtomicEdit(
                operation=edit.operation,
                target=edit.target,
                content=edit.content,
                rationale=edit.rationale,
                priority=max(edit.priority, proposal_priority),
            )
            if existing is None:
                grouped[key] = RankedEdit(effective, support=1, first_seen=seen_index)
                seen_index += 1
            else:
                duplicate_count += 1
                grouped[key] = RankedEdit(
                    edit=AtomicEdit(
                        operation=existing.edit.operation,
                        target=existing.edit.target,
                        content=existing.edit.content,
                        rationale=existing.edit.rationale or effective.rationale,
                        priority=max(existing.edit.priority, effective.priority),
                    ),
                    support=existing.support + 1,
                    first_seen=existing.first_seen,
                )

    ranked = sorted(
        grouped.values(),
        key=lambda item: (-item.support, -item.edit.priority, item.first_seen),
    )
    selected: list[AtomicEdit] = []
    claimed_targets: set[str] = set()
    conflict_count = 0
    for item in ranked:
        edit = item.edit
        conflict_key = edit.target if edit.operation in {"delete", "replace"} else ""
        if conflict_key and conflict_key in claimed_targets:
            conflict_count += 1
            continue
        selected.append(edit)
        if conflict_key:
            claimed_targets.add(conflict_key)
        if len(selected) >= budget:
            break
    return MergeResult(tuple(selected), tuple(ranked), duplicate_count, conflict_count)


def canonical_edit_key(edit: AtomicEdit) -> tuple[str, str, str]:
    return edit.operation, edit.target.strip(), edit.content.strip()


def set_slow_update(skill_text: str, content: str) -> str:
    block = f"{SLOW_START}\n{content.strip()}\n{SLOW_END}"
    start = skill_text.find(SLOW_START)
    end = skill_text.find(SLOW_END)
    if start < 0 and end < 0:
        return normalize_final_newline(skill_text.rstrip() + "\n\n" + block)
    if start < 0 or end < start:
        raise ValueError("Skill contains a malformed protected slow-update field")
    end += len(SLOW_END)
    return normalize_final_newline(skill_text[:start] + block + skill_text[end:])


def normalize_final_newline(value: str) -> str:
    return value.rstrip() + "\n"
