"""Real coding-hidden ablation with a deterministic coding agent."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from textskill_optimizer.interfaces import SkillEditor
from textskill_optimizer.io import load_tasks_jsonl, write_json, write_text
from textskill_optimizer.lr_profiles import get_lr_profile, profile_names
from textskill_optimizer.models import EditProposal, Task, TaskResult
from textskill_optimizer.optimizer import (
    OptimizerConfig,
    SkillOptimizer,
    added_bullet_lines,
)
from textskill_optimizer.plugins.coding import CodingRunner, CodingScorer


INITIAL_SKILL = """# Coding Agent Skill

Fix failing implementations without editing tests.
"""

TRAIN_ONLY_SKILL = """# Coding Agent Skill

Fix failing implementations without editing tests.

- TRAIN_ONLY_CODING_HIDDEN_RULES: memorize the visible training fixture families.
- Train family: slug normalization.
- Train family: money parser.
- Train family: email dedupe.
- Train family: number range.
- Train family: nested default getter.
- Train family: stable score sort.
- Train family: duration parser.
- Train family: cents rounding.
"""

FULL_SKILL = """# Coding Agent Skill

Fix failing implementations without editing tests.

- FULL_CODING_HIDDEN_RULES: apply compact transferable rules across keyed dedupe, nested paths, sorting, ranges, parsing, and rounding.
"""


CODING_HIDDEN_META_SKILL = """Prefer one compact transferable coding rule over enumerating train-only fixture families.

When training failures span small utility functions, convert them into concrete capability rules. Do not stop at generic "handle edge cases" wording.

Prioritize these reusable capability families when the evidence supports them:
- keyed de-duplication: check key existence before reading; append missing-key records immediately without adding None/null to the seen set, and use normalized comparison keys such as casefold() for email/case-insensitive dedupe while preserving original records.
- nested path access: trim path segments, ignore empty separators, handle dict keys and list indexes defensively; getters return defaults for missing paths, while pluck/collection utilities skip missing records instead of appending None.
- sort-by-key: keep sorting stable and put missing keys last with a safe key shape such as `(key not in item, item.get(key))`.
- ranges and dates: swap reversed bounds before iteration and produce ascending inclusive outputs unless the task asks for descending output.
- parsers: trim separators, skip empty tokens, wrap conversions in try/except, skip malformed tokens unless the task asks to raise, and preserve signs or units.
- rounding: use decimal-safe half-up rounding such as `Decimal` plus `ROUND_HALF_UP` for money, tax, or cents behavior.
"""


class CodingHiddenAblationEditor(SkillEditor):
    def propose(
        self,
        skill_text: str,
        train_results: list[TaskResult],
        *,
        epoch: int,
        rejected_buffer: list[dict[str, Any]] | None = None,
        meta_skill: str = "",
        optimizer_controls: dict[str, Any] | None = None,
    ) -> list[EditProposal]:
        if not any(not result.score.success for result in train_results):
            return []
        if "FULL_CODING_HIDDEN_RULES" in skill_text:
            return []
        if meta_skill:
            return [full_proposal("meta-skill-guided", epoch)]
        if rejected_buffer:
            return [full_proposal("rejected-buffer-guided", epoch)]
        return [train_only_proposal(epoch)]


def train_only_proposal(epoch: int) -> EditProposal:
    return EditProposal(
        name=f"train-only-large-edit-epoch-{epoch}",
        skill_text=TRAIN_ONLY_SKILL,
        rationale="Adds many train-family rules that do not transfer to validation/holdout.",
    )


def full_proposal(source: str, epoch: int) -> EditProposal:
    return EditProposal(
        name=f"{source}-compact-edit-epoch-{epoch}",
        skill_text=FULL_SKILL,
        rationale="Adds one compact transferable marker covering coding-hidden families.",
    )


def load_coding_hidden_tasks(
    path: Path,
    *,
    timeout_seconds: int = 30,
    agent_path: Path | None = None,
) -> list[Task]:
    agent = agent_path or ROOT / "work/coding_hidden_deterministic_agent.py"
    if not agent.is_absolute():
        agent = ROOT / agent
    tasks = []
    for task in load_tasks_jsonl(path):
        metadata = dict(task.metadata)
        metadata["agent_command"] = f"{sys.executable} {agent}"
        metadata["timeout_seconds"] = timeout_seconds
        tasks.append(Task(task.id, task.input, task.expected, metadata))
    return tasks


def ablation_configs(out_dir: Path, *, lr_profile: str = "strict") -> list[dict[str, Any]]:
    meta_path = out_dir / "meta_skill.md"
    profile = get_lr_profile(lr_profile)
    write_text(
        meta_path,
        CODING_HIDDEN_META_SKILL,
    )
    return [
        {
            "name": "gate_only",
            "label": "Gate only",
            "config": OptimizerConfig(
                epochs=2,
                max_candidates_per_epoch=1,
                max_skill_chars=None,
                max_skill_delta_chars=None,
                max_added_bullet_lines=None,
                rejected_buffer_limit=0,
            ),
        },
        {
            "name": "gate_lr",
            "label": "+LR",
            "config": OptimizerConfig(
                epochs=2,
                max_candidates_per_epoch=1,
                max_skill_chars=profile.max_skill_chars,
                max_skill_delta_chars=profile.max_skill_delta_chars,
                max_added_bullet_lines=profile.max_added_bullet_lines,
                rejected_buffer_limit=0,
            ),
        },
        {
            "name": "gate_lr_rejected",
            "label": "+LR+Rejected Buffer",
            "config": OptimizerConfig(
                epochs=2,
                max_candidates_per_epoch=1,
                max_skill_chars=profile.max_skill_chars,
                max_skill_delta_chars=profile.max_skill_delta_chars,
                max_added_bullet_lines=profile.max_added_bullet_lines,
                rejected_buffer_limit=5,
            ),
        },
        {
            "name": "gate_lr_rejected_meta",
            "label": "+LR+Rejected Buffer+Meta Skill",
            "config": OptimizerConfig(
                epochs=2,
                max_candidates_per_epoch=1,
                max_skill_chars=profile.max_skill_chars,
                max_skill_delta_chars=profile.max_skill_delta_chars,
                max_added_bullet_lines=profile.max_added_bullet_lines,
                rejected_buffer_limit=5,
                meta_skill_path=meta_path,
            ),
        },
    ]


def run_case(
    case: dict[str, Any],
    *,
    out_dir: Path,
    train_tasks: list[Task],
    validation_tasks: list[Task],
    holdout_tasks: list[Task],
    editor: SkillEditor | None = None,
) -> dict[str, Any]:
    case_dir = out_dir / case["name"]
    optimizer = SkillOptimizer(
        runner=CodingRunner(),
        scorer=CodingScorer(),
        editor=editor or CodingHiddenAblationEditor(),
        config=case["config"],
    )
    result = optimizer.optimize(
        INITIAL_SKILL,
        train_tasks,
        validation_tasks,
        run_dir=case_dir,
    )
    holdout_report = optimizer.evaluate(
        result.best_skill_text,
        holdout_tasks,
        name=f"holdout:{case['name']}",
    )
    write_json(case_dir / "holdout_final.json", holdout_report.to_dict())

    reason_counts = Counter(
        item.metadata.get("rejection_reason")
        for item in result.history
        if item.epoch > 0 and not item.accepted
    )
    reason_counts.pop(None, None)
    first_success_epoch = next(
        (
            item.epoch
            for item in result.history
            if item.epoch > 0 and item.accepted and item.validation_score == 1.0
        ),
        None,
    )
    return {
        "name": case["name"],
        "label": case["label"],
        "validation_score": result.best_validation_score,
        "validation_pass_rate": result.final_validation_report.pass_rate,
        "holdout_score": holdout_report.average_score,
        "holdout_pass_rate": holdout_report.pass_rate,
        "first_success_epoch": first_success_epoch,
        "accepted_candidates": sum(1 for item in result.history if item.epoch > 0 and item.accepted),
        "rejected_total": len(result.rejected_buffer),
        "lr_rejections": reason_counts.get("learning_rate_exceeded", 0),
        "validated_rejections": sum(
            1
            for item in result.history
            if item.epoch > 0 and not item.accepted and item.validation_score is not None
        ),
        "rejection_reasons": dict(sorted(reason_counts.items())),
        "best_skill_chars": len(result.best_skill_text),
        "best_skill_added_bullets": len(added_bullet_lines(INITIAL_SKILL, result.best_skill_text)),
        "run_dir": str(case_dir),
    }


def run_ablation(
    out_dir: Path,
    *,
    timeout_seconds: int = 30,
    lr_profile: str = "strict",
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    train = load_coding_hidden_tasks(ROOT / "examples/coding-hidden/train.jsonl", timeout_seconds=timeout_seconds)
    valid = load_coding_hidden_tasks(ROOT / "examples/coding-hidden/valid.jsonl", timeout_seconds=timeout_seconds)
    holdout = load_coding_hidden_tasks(ROOT / "examples/coding-hidden/holdout.jsonl", timeout_seconds=timeout_seconds)
    rows = [
        run_case(
            case,
            out_dir=out_dir,
            train_tasks=train,
            validation_tasks=valid,
            holdout_tasks=holdout,
        )
        for case in ablation_configs(out_dir, lr_profile=lr_profile)
    ]
    summary = {
        "description": "Real coding-hidden ablation with deterministic coding agent and hidden scorer.",
        "lr_profile": lr_profile,
        "train_tasks": len(train),
        "validation_tasks": len(valid),
        "holdout_tasks": len(holdout),
        "timeout_seconds": timeout_seconds,
        "rows": rows,
    }
    write_json(out_dir / "summary.json", summary)
    write_report(out_dir / "report.md", rows)
    return summary


def write_report(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Coding-Hidden Mechanism Ablation",
        "",
        "| Case | Valid | Holdout | First Success Epoch | LR Rejections | Validated Rejections | Rejected Total |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        first_success = row["first_success_epoch"] if row["first_success_epoch"] is not None else "-"
        lines.append(
            f"| {row['label']} | {row['validation_score']:.2f} | {row['holdout_score']:.2f} | "
            f"{first_success} | {row['lr_rejections']} | {row['validated_rejections']} | {row['rejected_total']} |"
        )
    lines.extend(
        [
            "",
            "Interpretation:",
            "",
            "- Gate only blocks the train-only skill on validation but does not make the fixed editor change strategy.",
            "- Textual LR rejects the oversized train-only skill before validation.",
            "- Rejected Buffer lets the editor switch from the LR-rejected train-only skill to the compact transferable skill in epoch 2.",
            "- Meta Skill makes the editor choose the compact transferable skill in epoch 1.",
            "",
        ]
    )
    write_text(path, "\n".join(lines))


def print_summary(rows: list[dict[str, Any]], out_dir: Path) -> None:
    print("| case | valid | holdout | first_success_epoch | lr_rejections | validated_rejections | rejected_total |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        first_success = row["first_success_epoch"] if row["first_success_epoch"] is not None else "-"
        print(
            f"| {row['label']} | {row['validation_score']:.2f} | {row['holdout_score']:.2f} | "
            f"{first_success} | {row['lr_rejections']} | {row['validated_rejections']} | {row['rejected_total']} |"
        )
    print(f"summary={out_dir / 'summary.json'}")
    print(f"report={out_dir / 'report.md'}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        default="runs/coding-hidden-ablation-v1",
        help="Output directory for ablation artifacts.",
    )
    parser.add_argument("--timeout-seconds", type=int, default=30)
    parser.add_argument("--lr-profile", choices=profile_names(), default="strict")
    args = parser.parse_args(argv)
    out_dir = Path(args.out)
    summary = run_ablation(
        out_dir,
        timeout_seconds=args.timeout_seconds,
        lr_profile=args.lr_profile,
    )
    print_summary(summary["rows"], out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
