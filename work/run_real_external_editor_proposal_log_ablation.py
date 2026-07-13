"""Capture real external-editor proposal logs, then replay the ablation."""

from __future__ import annotations

import argparse
import getpass
import json
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
WORK = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(WORK))

from textskill_optimizer.command_editor import CommandEditorConfig, CommandSkillEditor
from textskill_optimizer.io import write_json, write_text

import run_coding_hidden_ablation as base
import run_coding_hidden_proposal_log_ablation as replay


DEFAULT_BASE_URL = "https://ark-cn-beijing.bytedance.net/api/v3"
DEFAULT_MODEL = "ep-20260507113406-9h6cz"
DEFAULT_CASES = [
    "gate_only",
    "gate_lr",
    "gate_lr_rejected",
    "gate_lr_rejected_meta",
]


def capture_and_replay(
    *,
    out_dir: Path,
    replay_out_dir: Path,
    proposal_log_path: Path,
    seeds: list[str],
    cases: list[str],
    timeout_seconds: int,
    editor_command: str,
    editor_timeout: int,
    env: dict[str, str],
    lr_profile: str = "strict",
    agent_path: Path | None = None,
    skip_replay: bool = False,
    append_log: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    if proposal_log_path.exists() and not append_log and not dry_run:
        raise FileExistsError(
            f"Proposal log already exists: {proposal_log_path}. "
            "Use --append-log or choose a new --proposal-log."
        )
    if dry_run:
        summary = dry_run_summary(
            out_dir=out_dir,
            replay_out_dir=replay_out_dir,
            proposal_log_path=proposal_log_path,
            seeds=seeds,
            cases=cases,
            timeout_seconds=timeout_seconds,
            editor_command=editor_command,
            editor_timeout=editor_timeout,
            env=env,
            lr_profile=lr_profile,
            agent_path=agent_path,
            skip_replay=skip_replay,
        )
        write_json(out_dir / "dry_run_summary.json", summary)
        write_manifest(out_dir / "dry_run_manifest.md", summary)
        return summary

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

    old_env = os.environ.copy()
    os.environ.update(env)
    try:
        rows = []
        for seed in seeds:
            seed_dir = out_dir / seed
            configs = {
                case["name"]: case
                for case in base.ablation_configs(seed_dir, lr_profile=lr_profile)
            }
            for case_name in cases:
                if case_name not in configs:
                    raise ValueError(f"Unknown case: {case_name}")
                print(f"[capture] seed={seed} case={case_name}", flush=True)
                editor = CommandSkillEditor(
                    CommandEditorConfig(
                        command=editor_command,
                        timeout_seconds=editor_timeout,
                        proposal_log_path=proposal_log_path,
                        proposal_log_seed=seed,
                        proposal_log_case=case_name,
                    )
                )
                row = base.run_case(
                    configs[case_name],
                    out_dir=seed_dir,
                    train_tasks=train,
                    validation_tasks=valid,
                    holdout_tasks=holdout,
                    editor=editor,
                )
                row["seed"] = seed
                rows.append(row)
    finally:
        os.environ.clear()
        os.environ.update(old_env)

    summary: dict[str, Any] = {
        "description": "Real external-editor proposal-log capture for coding-hidden ablation.",
        "proposal_log": str(proposal_log_path),
        "replay_out_dir": str(replay_out_dir),
        "lr_profile": lr_profile,
        "agent_path": str(agent_path) if agent_path is not None else "",
        "seeds": seeds,
        "cases": cases,
        "timeout_seconds": timeout_seconds,
        "editor_command": editor_command,
        "external_model": {
            "base_url": env.get("EXTERNAL_LLM_BASE_URL", ""),
            "model": env.get("EXTERNAL_LLM_MODEL", ""),
            "json_mode": env.get("EXTERNAL_LLM_JSON_MODE", ""),
            "temperature": env.get("EXTERNAL_LLM_TEMPERATURE", ""),
        },
        "rows": rows,
    }
    write_json(out_dir / "capture_summary.json", summary)

    if not skip_replay:
        print("[replay] running fixed proposal-log ablation", flush=True)
        summary["replay"] = replay.run_replay_ablation(
            replay_out_dir,
            proposal_log_path=proposal_log_path,
            seeds=seeds,
            timeout_seconds=timeout_seconds,
            agent_path=agent_path,
            lr_profile=lr_profile,
        )
        write_json(out_dir / "capture_summary.json", summary)
    write_report(out_dir / "report.md", summary)
    return summary


def dry_run_summary(
    *,
    out_dir: Path,
    replay_out_dir: Path,
    proposal_log_path: Path,
    seeds: list[str],
    cases: list[str],
    timeout_seconds: int,
    editor_command: str,
    editor_timeout: int,
    env: dict[str, str],
    lr_profile: str,
    agent_path: Path | None,
    skip_replay: bool,
) -> dict[str, Any]:
    return {
        "description": "Dry run for real external-editor proposal-log capture.",
        "out_dir": str(out_dir),
        "replay_out_dir": str(replay_out_dir),
        "proposal_log": str(proposal_log_path),
        "lr_profile": lr_profile,
        "agent_path": str(agent_path) if agent_path is not None else "",
        "seeds": seeds,
        "cases": cases,
        "timeout_seconds": timeout_seconds,
        "editor_command": editor_command,
        "editor_timeout": editor_timeout,
        "skip_replay": skip_replay,
        "external_model": {
            "base_url": env.get("EXTERNAL_LLM_BASE_URL", ""),
            "model": env.get("EXTERNAL_LLM_MODEL", ""),
            "json_mode": env.get("EXTERNAL_LLM_JSON_MODE", ""),
            "temperature": env.get("EXTERNAL_LLM_TEMPERATURE", ""),
            "has_api_key": bool(env.get("EXTERNAL_LLM_API_KEY")),
        },
        "planned_runs": [
            {"seed": seed, "case": case_name}
            for seed in seeds
            for case_name in cases
        ],
    }


def write_manifest(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Real External Editor Proposal-Log Capture Dry Run",
        "",
        f"Proposal log: `{summary['proposal_log']}`",
        f"Replay out: `{summary['replay_out_dir']}`",
        f"LR profile: `{summary['lr_profile']}`",
        f"Agent path: `{summary.get('agent_path') or 'default marker-only deterministic agent'}`",
        f"Model: `{summary['external_model']['model']}`",
        "",
        "| Seed | Case |",
        "|---|---|",
    ]
    for row in summary["planned_runs"]:
        lines.append(f"| {row['seed']} | {row['case']} |")
    write_text(path, "\n".join(lines) + "\n")


def write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Real External Editor Proposal-Log Capture",
        "",
        f"Proposal log: `{summary['proposal_log']}`",
        f"Model: `{summary['external_model']['model']}`",
        f"LR profile: `{summary['lr_profile']}`",
        f"Agent path: `{summary.get('agent_path') or 'default marker-only deterministic agent'}`",
        "",
        "## Capture Results",
        "",
        "| Seed | Case | Valid | Holdout | First Success Epoch | LR Rejections |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in summary["rows"]:
        first = row["first_success_epoch"] if row["first_success_epoch"] is not None else "-"
        lines.append(
            f"| {row['seed']} | {row['label']} | {row['validation_score']:.2f} | "
            f"{row['holdout_score']:.2f} | {first} | {row['lr_rejections']} |"
        )
    if "replay" in summary:
        lines.extend(
            [
                "",
                "## Replay Aggregates",
                "",
                "| Case | Seeds | Mean Valid | Mean Holdout | Successes | Mean First Success Epoch |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for row in summary["replay"]["aggregates"]:
            first = row["mean_first_success_epoch"]
            first_text = "-" if first is None else f"{first:.1f}"
            lines.append(
                f"| {row['label']} | {row['seeds']} | {row['mean_validation_score']:.2f} | "
                f"{row['mean_holdout_score']:.2f} | {row['successes']} | {first_text} |"
            )
    write_text(path, "\n".join(lines) + "\n")


def build_external_env(args: argparse.Namespace) -> dict[str, str]:
    base_url = (
        args.base_url
        or os.environ.get("EXTERNAL_LLM_BASE_URL")
        or os.environ.get("BYTEDANCE_MODEL_BASE_URL")
        or DEFAULT_BASE_URL
    )
    model = (
        args.model
        or os.environ.get("EXTERNAL_LLM_MODEL")
        or os.environ.get("BYTEDANCE_MODEL_ID")
        or DEFAULT_MODEL
    )
    api_key = os.environ.get("EXTERNAL_LLM_API_KEY", "")
    if not api_key and args.dry_run:
        api_key = "not-needed"
    if not api_key and not args.dry_run:
        api_key = read_api_key()
    if not api_key:
        raise ValueError("EXTERNAL_LLM_API_KEY is required for a real capture run")

    return {
        "EXTERNAL_LLM_BASE_URL": base_url,
        "EXTERNAL_LLM_MODEL": model,
        "EXTERNAL_LLM_API_KEY": api_key,
        "EXTERNAL_LLM_JSON_MODE": args.json_mode,
        "EXTERNAL_LLM_TIMEOUT": str(args.llm_timeout),
        "EXTERNAL_LLM_TEMPERATURE": str(args.temperature),
    }


def read_api_key() -> str:
    if sys.stdin.isatty():
        return getpass.getpass("External LLM API key: ").strip()
    return sys.stdin.readline().strip()


def parse_csv(raw: str) -> list[str]:
    values = [item.strip() for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("Expected at least one comma-separated value")
    return values


def print_summary(summary: dict[str, Any], out_dir: Path) -> None:
    if "rows" not in summary:
        print(f"dry_run_summary={out_dir / 'dry_run_summary.json'}")
        print(f"dry_run_manifest={out_dir / 'dry_run_manifest.md'}")
        return
    print("| seed | case | valid | holdout | first_success_epoch | lr_rejections |")
    print("|---|---|---:|---:|---:|---:|")
    for row in summary["rows"]:
        first = row["first_success_epoch"] if row["first_success_epoch"] is not None else "-"
        print(
            f"| {row['seed']} | {row['label']} | {row['validation_score']:.2f} | "
            f"{row['holdout_score']:.2f} | {first} | {row['lr_rejections']} |"
        )
    print(f"proposal_log={summary['proposal_log']}")
    print(f"capture_summary={out_dir / 'capture_summary.json'}")
    if "replay" in summary:
        print(f"replay_summary={Path(summary['replay_out_dir']) / 'summary.json'}")
        print(f"replay_report={Path(summary['replay_out_dir']) / 'report.md'}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="runs/coding-hidden-real-external-editor-capture-v1")
    parser.add_argument("--replay-out", default="runs/coding-hidden-real-proposal-log-ablation-v1")
    parser.add_argument("--proposal-log", default="")
    parser.add_argument("--seeds", default="seed-a")
    parser.add_argument("--cases", default=",".join(DEFAULT_CASES))
    parser.add_argument("--timeout-seconds", type=int, default=30)
    parser.add_argument("--lr-profile", choices=base.profile_names(), default="strict")
    parser.add_argument(
        "--agent-path",
        default="",
        help="Optional coding agent path for capture and replay.",
    )
    parser.add_argument(
        "--editor-command",
        default=f"{sys.executable} examples/coding/openai_compatible_skill_editor.py",
    )
    parser.add_argument("--editor-timeout", type=int, default=240)
    parser.add_argument("--base-url", default="")
    parser.add_argument("--model", default="")
    parser.add_argument("--json-mode", default=os.environ.get("EXTERNAL_LLM_JSON_MODE", "0"))
    parser.add_argument("--temperature", default=os.environ.get("EXTERNAL_LLM_TEMPERATURE", "0.2"))
    parser.add_argument("--llm-timeout", type=int, default=int(os.environ.get("EXTERNAL_LLM_TIMEOUT", "180")))
    parser.add_argument("--skip-replay", action="store_true")
    parser.add_argument("--append-log", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    out_dir = Path(args.out)
    proposal_log_path = Path(args.proposal_log) if args.proposal_log else out_dir / "proposals.jsonl"
    summary = capture_and_replay(
        out_dir=out_dir,
        replay_out_dir=Path(args.replay_out),
        proposal_log_path=proposal_log_path,
        seeds=parse_csv(args.seeds),
        cases=parse_csv(args.cases),
        timeout_seconds=args.timeout_seconds,
        editor_command=args.editor_command,
        editor_timeout=args.editor_timeout,
        env=build_external_env(args),
        lr_profile=args.lr_profile,
        agent_path=Path(args.agent_path) if args.agent_path else None,
        skip_replay=args.skip_replay,
        append_log=args.append_log,
        dry_run=args.dry_run,
    )
    print_summary(summary, out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
