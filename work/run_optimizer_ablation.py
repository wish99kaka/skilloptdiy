"""Controlled ablation for minimal SkillOpt stability mechanisms."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from textskill_optimizer.interfaces import SkillEditor, SkillRunner, SkillScorer
from textskill_optimizer.io import write_json, write_text
from textskill_optimizer.models import EditProposal, Score, Task, TaskOutput, TaskResult
from textskill_optimizer.optimizer import OptimizerConfig, SkillOptimizer


INITIAL_SKILL = """# Controlled Skill

Start with no reusable edge-case rule.
"""

GOOD_RULE = "- GENERALIZE_EDGE_CASES: use a reusable rule that covers unseen validation cases."
BAD_RULE = "- MEMORIZE_TRAIN_ONLY: copy the visible training pattern without a reusable rule."


class ControlledRunner(SkillRunner):
    def run(self, skill_text: str, task: Task) -> TaskOutput:
        return TaskOutput(
            value={
                "has_general_rule": "GENERALIZE_EDGE_CASES" in skill_text,
                "has_train_marker": "MEMORIZE_TRAIN_ONLY" in skill_text,
            },
            metadata={"task_kind": task.metadata.get("kind")},
        )


class ControlledScorer(SkillScorer):
    def score(self, task: Task, output: TaskOutput) -> Score:
        value = output.value if isinstance(output.value, dict) else {}
        kind = task.metadata.get("kind")
        if kind == "train":
            success = bool(value.get("has_general_rule") or value.get("has_train_marker"))
        else:
            success = bool(value.get("has_general_rule"))
        return Score(1.0 if success else 0.0, success, "controlled score")


class ContextAwareEditor(SkillEditor):
    """Scripted editor that reacts only to optimizer-provided mechanisms."""

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
        if "GENERALIZE_EDGE_CASES" in skill_text:
            return []
        if meta_skill:
            return [good_proposal("meta-skill-guided", epoch)]
        if rejected_buffer:
            return [good_proposal("rejected-buffer-guided", epoch)]
        return [bad_proposal(epoch)]


def bad_proposal(epoch: int) -> EditProposal:
    memorized_examples = "\n".join(
        f"- memorized visible example {index}: keep this train-only pattern."
        for index in range(1, 7)
    )
    skill_text = f"{INITIAL_SKILL.rstrip()}\n\n{BAD_RULE}\n{memorized_examples}\n"
    return EditProposal(
        name=f"train-only-large-edit-epoch-{epoch}",
        skill_text=skill_text,
        rationale="Overfits to training-visible behavior and adds too many bullets.",
    )


def good_proposal(source: str, epoch: int) -> EditProposal:
    skill_text = f"{INITIAL_SKILL.rstrip()}\n\n{GOOD_RULE}\n"
    return EditProposal(
        name=f"{source}-compact-edit-epoch-{epoch}",
        skill_text=skill_text,
        rationale="Adds one compact general rule that transfers to validation.",
    )


def train_tasks() -> list[Task]:
    return [
        Task(
            id="train-visible-edge",
            input="Visible training edge case",
            expected={"success": True},
            metadata={"kind": "train"},
        )
    ]


def validation_tasks() -> list[Task]:
    return [
        Task(
            id="valid-unseen-edge",
            input="Unseen validation edge case",
            expected={"success": True},
            metadata={"kind": "validation"},
        )
    ]


def ablation_configs(out_dir: Path) -> list[dict[str, Any]]:
    meta_path = out_dir / "meta_skill.md"
    write_text(
        meta_path,
        "Prefer one compact general rule over memorizing visible examples.\n",
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
                max_skill_chars=400,
                max_skill_delta_chars=180,
                max_added_bullet_lines=1,
                rejected_buffer_limit=0,
            ),
        },
        {
            "name": "gate_lr_rejected",
            "label": "+LR+Rejected Buffer",
            "config": OptimizerConfig(
                epochs=2,
                max_candidates_per_epoch=1,
                max_skill_chars=400,
                max_skill_delta_chars=180,
                max_added_bullet_lines=1,
                rejected_buffer_limit=5,
            ),
        },
        {
            "name": "gate_lr_rejected_meta",
            "label": "+LR+Rejected Buffer+Meta Skill",
            "config": OptimizerConfig(
                epochs=2,
                max_candidates_per_epoch=1,
                max_skill_chars=400,
                max_skill_delta_chars=180,
                max_added_bullet_lines=1,
                rejected_buffer_limit=5,
                meta_skill_path=meta_path,
            ),
        },
    ]


def run_case(case: dict[str, Any], out_dir: Path) -> dict[str, Any]:
    case_dir = out_dir / case["name"]
    optimizer = SkillOptimizer(
        runner=ControlledRunner(),
        scorer=ControlledScorer(),
        editor=ContextAwareEditor(),
        config=case["config"],
    )
    result = optimizer.optimize(
        INITIAL_SKILL,
        train_tasks(),
        validation_tasks(),
        run_dir=case_dir,
    )
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
        "final_validation_score": result.best_validation_score,
        "final_pass_rate": result.final_validation_report.pass_rate,
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
        "run_dir": str(case_dir),
    }


def run_ablation(out_dir: Path) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = [run_case(case, out_dir) for case in ablation_configs(out_dir)]
    summary = {
        "description": "Controlled ablation of validation gate, textual learning rate, rejected buffer, and meta skill.",
        "rows": rows,
    }
    write_json(out_dir / "summary.json", summary)
    write_report(out_dir / "report.md", rows)
    return summary


def write_report(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# SkillOpt Minimal Stability Ablation",
        "",
        "| Case | Final Score | First Success Epoch | LR Rejections | Validated Rejections | Rejected Total |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        first_success = row["first_success_epoch"] if row["first_success_epoch"] is not None else "-"
        lines.append(
            f"| {row['label']} | {row['final_validation_score']:.2f} | {first_success} | "
            f"{row['lr_rejections']} | {row['validated_rejections']} | {row['rejected_total']} |"
        )
    lines.extend(
        [
            "",
            "Interpretation:",
            "",
            "- Gate only rejects the train-only candidate but cannot make the editor stop repeating it.",
            "- Textual LR rejects the oversized edit before validation, reducing unsafe validation attempts.",
            "- Rejected Buffer lets the editor react to the prior rejection and emit the compact transferable rule.",
            "- Meta Skill biases the editor toward the compact transferable rule in the first epoch.",
            "",
        ]
    )
    write_text(path, "\n".join(lines))


def print_summary(rows: list[dict[str, Any]], out_dir: Path) -> None:
    print("| case | final | first_success_epoch | lr_rejections | validated_rejections | rejected_total |")
    print("|---|---:|---:|---:|---:|---:|")
    for row in rows:
        first_success = row["first_success_epoch"] if row["first_success_epoch"] is not None else "-"
        print(
            f"| {row['label']} | {row['final_validation_score']:.2f} | {first_success} | "
            f"{row['lr_rejections']} | {row['validated_rejections']} | {row['rejected_total']} |"
        )
    print(f"summary={out_dir / 'summary.json'}")
    print(f"report={out_dir / 'report.md'}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        default="runs/optimizer-ablation-minimal-v1",
        help="Output directory for ablation artifacts.",
    )
    args = parser.parse_args(argv)
    out_dir = Path(args.out)
    summary = run_ablation(out_dir)
    print_summary(summary["rows"], out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
