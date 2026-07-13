"""Run LR-budget sensitivity over a fixed coding-hidden proposal log."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
WORK = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(WORK))

from textskill_optimizer.io import write_json, write_text
from textskill_optimizer.lr_profiles import profile_budget_dict
from textskill_optimizer.optimizer import OptimizerConfig

import run_coding_hidden_ablation as base
import run_coding_hidden_proposal_log_ablation as replay


LR_SOURCE_CASES = [
    "gate_lr",
    "gate_lr_rejected",
    "gate_lr_rejected_meta",
]


BUDGETS = [
    profile_budget_dict("strict"),
    {
        "name": "delta_320",
        "label": "delta 320",
        "max_skill_chars": 600,
        "max_skill_delta_chars": 320,
        "max_added_bullet_lines": 1,
    },
    profile_budget_dict("real-editor"),
    profile_budget_dict("loose-diagnostic"),
]


def run_lr_sensitivity(
    out_dir: Path,
    *,
    proposal_log_path: Path,
    seeds: list[str],
    timeout_seconds: int,
    agent_path: Path,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    proposal_log = replay.load_proposal_log(proposal_log_path)
    train = base.load_coding_hidden_tasks(
        ROOT / "examples/coding-hidden/train.jsonl",
        timeout_seconds=timeout_seconds,
        agent_path=agent_path,
    )
    valid = base.load_coding_hidden_tasks(
        ROOT / "examples/coding-hidden/valid.jsonl",
        timeout_seconds=timeout_seconds,
        agent_path=agent_path,
    )
    holdout = base.load_coding_hidden_tasks(
        ROOT / "examples/coding-hidden/holdout.jsonl",
        timeout_seconds=timeout_seconds,
        agent_path=agent_path,
    )

    rows: list[dict[str, Any]] = []
    for seed in seeds:
        seed_dir = out_dir / seed
        meta_path = seed_dir / "meta_skill.md"
        write_text(
            meta_path,
            "Prefer one compact transferable coding rule over enumerating train-only fixture families.\n",
        )
        rows.append(
            run_source_case(
                seed=seed,
                seed_dir=seed_dir,
                source_case="gate_only",
                budget=None,
                proposal_log=proposal_log,
                train=train,
                valid=valid,
                holdout=holdout,
            )
        )
        for budget in BUDGETS:
            for source_case in LR_SOURCE_CASES:
                rows.append(
                    run_source_case(
                        seed=seed,
                        seed_dir=seed_dir,
                        source_case=source_case,
                        budget=budget,
                        proposal_log=proposal_log,
                        train=train,
                        valid=valid,
                        holdout=holdout,
                        meta_path=meta_path,
                    )
                )

    summary = {
        "description": "LR-budget sensitivity over fixed real external-editor proposal logs.",
        "proposal_log": str(proposal_log_path),
        "agent_path": str(agent_path),
        "seeds": seeds,
        "timeout_seconds": timeout_seconds,
        "budgets": BUDGETS,
        "rows": rows,
        "aggregates": aggregate_rows(rows),
    }
    write_json(out_dir / "summary.json", summary)
    write_report(out_dir / "report.md", summary)
    return summary


def run_source_case(
    *,
    seed: str,
    seed_dir: Path,
    source_case: str,
    budget: dict[str, Any] | None,
    proposal_log: replay.ProposalLog,
    train: list[Any],
    valid: list[Any],
    holdout: list[Any],
    meta_path: Path | None = None,
) -> dict[str, Any]:
    if source_case == "gate_only":
        case = {
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
        }
        budget_name = "none"
        budget_label = "no LR"
    else:
        if budget is None:
            raise ValueError("LR source cases require a budget")
        rejected_limit = 0 if source_case == "gate_lr" else 5
        case = {
            "name": f"{source_case}__{budget['name']}",
            "label": f"{source_case_label(source_case)} @ {budget['label']}",
            "config": OptimizerConfig(
                epochs=2,
                max_candidates_per_epoch=1,
                max_skill_chars=int(budget["max_skill_chars"]),
                max_skill_delta_chars=int(budget["max_skill_delta_chars"]),
                max_added_bullet_lines=int(budget["max_added_bullet_lines"]),
                rejected_buffer_limit=rejected_limit,
                meta_skill_path=meta_path if source_case == "gate_lr_rejected_meta" else None,
            ),
        }
        budget_name = str(budget["name"])
        budget_label = str(budget["label"])

    editor = replay.ProposalLogEditor(
        proposal_log,
        seed=seed,
        case_name=source_case,
    )
    row = base.run_case(
        case,
        out_dir=seed_dir,
        train_tasks=train,
        validation_tasks=valid,
        holdout_tasks=holdout,
        editor=editor,
    )
    row["seed"] = seed
    row["source_case"] = source_case
    row["source_case_label"] = source_case_label(source_case)
    row["budget"] = budget_name
    row["budget_label"] = budget_label
    return row


def source_case_label(source_case: str) -> str:
    return {
        "gate_only": "Gate only",
        "gate_lr": "+LR",
        "gate_lr_rejected": "+LR+Rejected Buffer",
        "gate_lr_rejected_meta": "+LR+Rejected Buffer+Meta Skill",
    }[source_case]


def aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((str(row["source_case"]), str(row["budget"])), []).append(row)
    aggregates = []
    for (source_case, budget), group_rows in sorted(groups.items()):
        aggregates.append(
            {
                "source_case": source_case,
                "source_case_label": group_rows[0]["source_case_label"],
                "budget": budget,
                "budget_label": group_rows[0]["budget_label"],
                "seeds": len(group_rows),
                "mean_validation_score": statistics.mean(
                    float(row["validation_score"]) for row in group_rows
                ),
                "mean_holdout_score": statistics.mean(
                    float(row["holdout_score"]) for row in group_rows
                ),
                "mean_lr_rejections": statistics.mean(
                    int(row["lr_rejections"]) for row in group_rows
                ),
                "successes": sum(1 for row in group_rows if float(row["validation_score"]) == 1.0),
            }
        )
    return aggregates


def write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Coding-Hidden LR Sensitivity",
        "",
        f"Proposal log: `{summary['proposal_log']}`",
        f"Agent path: `{summary['agent_path']}`",
        "",
        "## Aggregate Results",
        "",
        "| Source Case | Budget | Seeds | Mean Valid | Mean Holdout | Mean LR Rejections | Successes |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary["aggregates"]:
        lines.append(
            f"| {row['source_case_label']} | {row['budget_label']} | {row['seeds']} | "
            f"{row['mean_validation_score']:.2f} | {row['mean_holdout_score']:.2f} | "
            f"{row['mean_lr_rejections']:.1f} | {row['successes']} |"
        )
    lines.extend(
        [
            "",
            "## Per-Seed Results",
            "",
            "| Seed | Source Case | Budget | Valid | Holdout | LR Rejections | Accepted Candidates |",
            "|---|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in summary["rows"]:
        lines.append(
            f"| {row['seed']} | {row['source_case_label']} | {row['budget_label']} | "
            f"{row['validation_score']:.2f} | {row['holdout_score']:.2f} | "
            f"{row['lr_rejections']} | {row['accepted_candidates']} |"
        )
    write_text(path, "\n".join(lines) + "\n")


def parse_seed_list(raw: str) -> list[str]:
    seeds = [item.strip() for item in raw.split(",") if item.strip()]
    if not seeds:
        raise ValueError("At least one seed is required")
    return seeds


def print_summary(summary: dict[str, Any], out_dir: Path) -> None:
    print("| source_case | budget | mean_valid | mean_holdout | mean_lr_rejections | successes |")
    print("|---|---|---:|---:|---:|---:|")
    for row in summary["aggregates"]:
        print(
            f"| {row['source_case_label']} | {row['budget_label']} | "
            f"{row['mean_validation_score']:.2f} | {row['mean_holdout_score']:.2f} | "
            f"{row['mean_lr_rejections']:.1f} | {row['successes']} |"
        )
    print(f"summary={out_dir / 'summary.json'}")
    print(f"report={out_dir / 'report.md'}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="runs/coding-hidden-lr-sensitivity-v1")
    parser.add_argument(
        "--proposal-log",
        default="runs/coding-hidden-real-external-editor-capture-v1/proposals.jsonl",
    )
    parser.add_argument("--seeds", default="seed-a")
    parser.add_argument("--timeout-seconds", type=int, default=30)
    parser.add_argument("--agent-path", default="work/coding_hidden_text_sensitive_agent.py")
    args = parser.parse_args(argv)

    out_dir = Path(args.out)
    summary = run_lr_sensitivity(
        out_dir,
        proposal_log_path=Path(args.proposal_log),
        seeds=parse_seed_list(args.seeds),
        timeout_seconds=args.timeout_seconds,
        agent_path=Path(args.agent_path),
    )
    print_summary(summary, out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
