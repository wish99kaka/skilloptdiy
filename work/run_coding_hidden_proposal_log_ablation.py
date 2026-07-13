"""Replay coding-hidden ablations from fixed proposal logs."""

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

from textskill_optimizer.interfaces import SkillEditor
from textskill_optimizer.io import write_json, write_text
from textskill_optimizer.models import EditProposal, TaskResult

import run_coding_hidden_ablation as base


ProposalLog = dict[tuple[str, str, int], list[EditProposal]]


class ProposalLogEditor(SkillEditor):
    def __init__(self, proposal_log: ProposalLog, *, seed: str, case_name: str) -> None:
        self.proposal_log = proposal_log
        self.seed = seed
        self.case_name = case_name

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
        return list(self.proposal_log.get((self.seed, self.case_name, epoch), []))


def load_proposal_log(path: Path) -> ProposalLog:
    proposal_log: ProposalLog = {}
    for index, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        payload = json.loads(line)
        seed = str(payload["seed"])
        case = str(payload["case"])
        epoch = int(payload["epoch"])
        proposals = []
        for proposal in payload.get("proposals", []):
            proposals.append(
                EditProposal(
                    name=str(proposal.get("name") or f"proposal-{index}"),
                    skill_text=str(proposal["skill_text"]),
                    rationale=str(proposal.get("rationale") or "Proposal log replay."),
                    metadata={
                        **(
                            proposal.get("metadata")
                            if isinstance(proposal.get("metadata"), dict)
                            else {}
                        ),
                        "proposal_log": str(path),
                        "proposal_log_line": index,
                        "proposal_log_seed": seed,
                    },
                )
            )
        proposal_log[(seed, case, epoch)] = proposals
    return proposal_log


def write_sample_proposal_log(path: Path, seeds: list[str]) -> None:
    lines = []
    for seed in seeds:
        for case in ("gate_only", "gate_lr"):
            for epoch in (1, 2):
                lines.append(
                    proposal_log_record(
                        seed,
                        case,
                        epoch,
                        base.train_only_proposal(epoch),
                    )
                )
        lines.append(
            proposal_log_record(
                seed,
                "gate_lr_rejected",
                1,
                base.train_only_proposal(1),
            )
        )
        lines.append(
            proposal_log_record(
                seed,
                "gate_lr_rejected",
                2,
                base.full_proposal("proposal-log-rejected-buffer-guided", 2),
            )
        )
        for epoch in (1, 2):
            lines.append(
                proposal_log_record(
                    seed,
                    "gate_lr_rejected_meta",
                    epoch,
                    base.full_proposal("proposal-log-meta-guided", epoch),
                )
            )
    write_text(path, "\n".join(json.dumps(line, sort_keys=True) for line in lines) + "\n")


def proposal_log_record(seed: str, case: str, epoch: int, proposal: EditProposal) -> dict[str, Any]:
    return {
        "seed": seed,
        "case": case,
        "epoch": epoch,
        "proposals": [
            {
                "name": proposal.name,
                "skill_text": proposal.skill_text,
                "rationale": proposal.rationale,
                "metadata": proposal.metadata,
            }
        ],
    }


def run_replay_ablation(
    out_dir: Path,
    *,
    proposal_log_path: Path,
    seeds: list[str],
    timeout_seconds: int,
    agent_path: Path | None = None,
    lr_profile: str = "strict",
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    proposal_log = load_proposal_log(proposal_log_path)
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
    rows = []
    for seed in seeds:
        seed_dir = out_dir / seed
        for case in base.ablation_configs(seed_dir, lr_profile=lr_profile):
            editor = ProposalLogEditor(proposal_log, seed=seed, case_name=case["name"])
            row = base.run_case(
                case,
                out_dir=seed_dir,
                train_tasks=train,
                validation_tasks=valid,
                holdout_tasks=holdout,
                editor=editor,
            )
            row["seed"] = seed
            rows.append(row)

    summary = {
        "description": "Coding-hidden ablation replayed from fixed proposal logs.",
        "proposal_log": str(proposal_log_path),
        "agent_path": str(agent_path) if agent_path is not None else "",
        "lr_profile": lr_profile,
        "seeds": seeds,
        "timeout_seconds": timeout_seconds,
        "rows": rows,
        "aggregates": aggregate_rows(rows),
    }
    write_json(out_dir / "summary.json", summary)
    write_report(out_dir / "report.md", summary)
    return summary


def aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_case: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_case.setdefault(str(row["name"]), []).append(row)
    aggregates = []
    for case_name, case_rows in sorted(by_case.items()):
        aggregates.append(
            {
                "name": case_name,
                "label": str(case_rows[0]["label"]),
                "seeds": len(case_rows),
                "mean_validation_score": statistics.mean(
                    float(row["validation_score"]) for row in case_rows
                ),
                "mean_holdout_score": statistics.mean(
                    float(row["holdout_score"]) for row in case_rows
                ),
                "successes": sum(1 for row in case_rows if float(row["validation_score"]) == 1.0),
                "mean_first_success_epoch": mean_optional(
                    row["first_success_epoch"] for row in case_rows
                ),
                "mean_lr_rejections": statistics.mean(
                    int(row["lr_rejections"]) for row in case_rows
                ),
            }
        )
    return aggregates


def mean_optional(values: Any) -> float | None:
    present = [float(value) for value in values if value is not None]
    if not present:
        return None
    return statistics.mean(present)


def write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Coding-Hidden Proposal-Log Ablation",
        "",
        f"Proposal log: `{summary['proposal_log']}`",
        f"Agent path: `{summary.get('agent_path') or 'default marker-only deterministic agent'}`",
        f"LR profile: `{summary.get('lr_profile') or 'strict'}`",
        "",
        "## Aggregate Results",
        "",
        "| Case | Seeds | Mean Valid | Mean Holdout | Successes | Mean First Success Epoch | Mean LR Rejections |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["aggregates"]:
        first = row["mean_first_success_epoch"]
        first_text = "-" if first is None else f"{first:.1f}"
        lines.append(
            f"| {row['label']} | {row['seeds']} | {row['mean_validation_score']:.2f} | "
            f"{row['mean_holdout_score']:.2f} | {row['successes']} | {first_text} | "
            f"{row['mean_lr_rejections']:.1f} |"
        )
    lines.extend(
        [
            "",
            "## Per-Seed Results",
            "",
            "| Seed | Case | Valid | Holdout | First Success Epoch | LR Rejections |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in summary["rows"]:
        first = row["first_success_epoch"] if row["first_success_epoch"] is not None else "-"
        lines.append(
            f"| {row['seed']} | {row['label']} | {row['validation_score']:.2f} | "
            f"{row['holdout_score']:.2f} | {first} | {row['lr_rejections']} |"
        )
    write_text(path, "\n".join(lines) + "\n")


def parse_seed_list(raw: str) -> list[str]:
    seeds = [item.strip() for item in raw.split(",") if item.strip()]
    if not seeds:
        raise ValueError("At least one seed is required")
    return seeds


def print_summary(summary: dict[str, Any], out_dir: Path) -> None:
    print("| case | seeds | mean_valid | mean_holdout | successes | mean_first_success_epoch |")
    print("|---|---:|---:|---:|---:|---:|")
    for row in summary["aggregates"]:
        first = row["mean_first_success_epoch"]
        first_text = "-" if first is None else f"{first:.1f}"
        print(
            f"| {row['label']} | {row['seeds']} | {row['mean_validation_score']:.2f} | "
            f"{row['mean_holdout_score']:.2f} | {row['successes']} | {first_text} |"
        )
    print(f"summary={out_dir / 'summary.json'}")
    print(f"report={out_dir / 'report.md'}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="runs/coding-hidden-proposal-log-ablation-v1")
    parser.add_argument("--proposal-log", default="")
    parser.add_argument(
        "--agent-path",
        default="",
        help="Optional coding agent path for replay. Defaults to the marker-only deterministic agent.",
    )
    parser.add_argument("--seeds", default="sample-a,sample-b")
    parser.add_argument("--timeout-seconds", type=int, default=30)
    parser.add_argument("--lr-profile", choices=base.profile_names(), default="strict")
    parser.add_argument(
        "--write-sample-log",
        default="",
        help="Write a sample proposal log and exit unless --run-after-write is set.",
    )
    parser.add_argument("--run-after-write", action="store_true")
    args = parser.parse_args(argv)

    seeds = parse_seed_list(args.seeds)
    if args.write_sample_log:
        sample_path = Path(args.write_sample_log)
        write_sample_proposal_log(sample_path, seeds)
        print(f"sample_proposal_log={sample_path}")
        if not args.run_after_write:
            return 0
        proposal_log_path = sample_path
    elif args.proposal_log:
        proposal_log_path = Path(args.proposal_log)
    else:
        proposal_log_path = Path(args.out) / "sample_proposals.jsonl"
        write_sample_proposal_log(proposal_log_path, seeds)

    out_dir = Path(args.out)
    summary = run_replay_ablation(
        out_dir,
        proposal_log_path=proposal_log_path,
        seeds=seeds,
        timeout_seconds=args.timeout_seconds,
        agent_path=Path(args.agent_path) if args.agent_path else None,
        lr_profile=args.lr_profile,
    )
    print_summary(summary, out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
