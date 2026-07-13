#!/usr/bin/env python3
"""Compare Coco target models under no-skill and explicit-skill baselines."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import statistics
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Sequence


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from textskill_optimizer.io import load_tasks_jsonl, load_text, write_json
from textskill_optimizer.optimizer import SkillOptimizer
from textskill_optimizer.plugins.coding import CodingRunner, CodingScorer
from textskill_optimizer.usage_ledger import summarize_usage_file
from work.run_coco_hidden_eval import build_coco_tasks


DEFAULT_BENCHMARK = ROOT / "examples/coding-hidden-v2"
DEFAULT_WRAPPER = ROOT / "examples/coding/coco_agent_wrapper.py"
TARGET_AGENT_KINDS = ("target_agent_cli",)


class NoopEditor:
    def propose(self, *args, **kwargs):
        return []


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    benchmark = resolve_root_relative(args.benchmark)
    tasks_path = resolve_root_relative(args.tasks) if args.tasks else benchmark / "selection.jsonl"
    no_skill_path = resolve_root_relative(args.no_skill) if args.no_skill else benchmark / "no_skill.md"
    explicit_skill_path = (
        resolve_root_relative(args.explicit_skill) if args.explicit_skill else benchmark / "skill.md"
    )
    agent_wrapper = resolve_root_relative(args.agent_wrapper)
    out_dir = resolve_root_relative(args.out)
    seeds = parse_seeds(args.seeds)
    if args.strong_label == args.weak_label:
        raise ValueError("strong and weak labels must be different")

    out_dir.mkdir(parents=True, exist_ok=True)
    prepared_tasks_path, tasks = prepare_tasks(
        tasks_path=tasks_path,
        wrapper=agent_wrapper,
        out_dir=out_dir,
        task_limit=args.task_limit,
        task_timeout=args.task_timeout,
    )
    manifest = build_manifest(
        benchmark=benchmark,
        prepared_tasks_path=prepared_tasks_path,
        no_skill_path=no_skill_path,
        explicit_skill_path=explicit_skill_path,
        agent_wrapper=agent_wrapper,
        strong_label=args.strong_label,
        strong_model=args.strong_model,
        weak_label=args.weak_label,
        weak_model=args.weak_model,
        base_agent_extra_args=args.agent_extra_args,
        seeds=seeds,
    )
    write_json(out_dir / "manifest.json", manifest)

    rows = run_evaluations(
        tasks=tasks,
        no_skill_path=no_skill_path,
        explicit_skill_path=explicit_skill_path,
        out_dir=out_dir,
        strong_label=args.strong_label,
        strong_model=args.strong_model,
        weak_label=args.weak_label,
        weak_model=args.weak_model,
        seeds=seeds,
        base_agent_extra_args=args.agent_extra_args,
    )
    aggregate = aggregate_rows(rows)
    comparisons = build_comparisons(
        aggregate,
        strong_label=args.strong_label,
        weak_label=args.weak_label,
    )
    summary = {
        "manifest": manifest,
        "rows": rows,
        "aggregate": aggregate,
        "comparisons": comparisons,
    }
    write_json(out_dir / "summary.json", summary)
    print(json.dumps({"aggregate": aggregate, "comparisons": comparisons}, indent=2, sort_keys=True))
    return 0


def resolve_root_relative(path: str | Path) -> Path:
    value = Path(path)
    if value.is_absolute():
        return value.resolve()
    return (ROOT / value).resolve()


def parse_seeds(raw: str) -> list[str]:
    seeds = [item.strip() for item in raw.split(",") if item.strip()]
    if not seeds:
        raise ValueError("At least one seed label is required")
    return seeds


def build_agent_extra_args(base_extra_args: str, model: str) -> str:
    tokens = shlex.split(base_extra_args) if base_extra_args.strip() else []
    for index, token in enumerate(tokens):
        if token == "--model" or token.startswith("--model="):
            raise ValueError("agent extra args already contain --model; remove it and use --strong-model/--weak-model")
        if token in {"-m", "--m"}:
            continue
        if token == "--model" and index + 1 < len(tokens):
            raise ValueError("agent extra args already contain --model; remove it and use --strong-model/--weak-model")
    tokens.extend(["--model", model])
    return shlex.join(tokens)


def prepare_tasks(
    *,
    tasks_path: Path,
    wrapper: Path,
    out_dir: Path,
    task_limit: int,
    task_timeout: int | None,
) -> tuple[Path, list]:
    generated = build_coco_tasks(
        tasks_path,
        wrapper,
        task_limit=task_limit or None,
        timeout_seconds=task_timeout,
    )
    prepared = out_dir / "prepared_tasks.jsonl"
    shutil.copy2(generated, prepared)
    return prepared, load_tasks_jsonl(prepared)


def run_evaluations(
    *,
    tasks: list,
    no_skill_path: Path,
    explicit_skill_path: Path,
    out_dir: Path,
    strong_label: str,
    strong_model: str,
    weak_label: str,
    weak_model: str,
    seeds: list[str],
    base_agent_extra_args: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    skill_conditions = (
        ("no_skill", no_skill_path),
        ("explicit_skill", explicit_skill_path),
    )
    model_conditions = (
        (strong_label, strong_model),
        (weak_label, weak_model),
    )
    for seed in seeds:
        for model_label, model_name in model_conditions:
            for condition, skill_path in skill_conditions:
                run_dir = out_dir / seed / model_label / condition
                rows.append(
                    run_single_evaluation(
                        tasks=tasks,
                        skill_path=skill_path,
                        run_dir=run_dir,
                        seed=seed,
                        model_label=model_label,
                        model_name=model_name,
                        condition=condition,
                        base_agent_extra_args=base_agent_extra_args,
                    )
                )
    return rows


def run_single_evaluation(
    *,
    tasks: list,
    skill_path: Path,
    run_dir: Path,
    seed: str,
    model_label: str,
    model_name: str,
    condition: str,
    base_agent_extra_args: str,
) -> dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=True)
    usage_ledger_path = run_dir / "usage_ledger.jsonl"
    report_path = run_dir / "report.json"
    timing_path = run_dir / "timing.json"
    usage_summary_path = run_dir / "usage_summary.json"

    optimizer = SkillOptimizer(
        runner=CodingRunner(
            usage_ledger_path=usage_ledger_path,
            usage_context={
                "seed": seed,
                "model_label": model_label,
                "model_name": model_name,
                "condition": condition,
                "comparison": "model_skill_proxy_cost",
            },
        ),
        scorer=CodingScorer(),
        editor=NoopEditor(),
    )
    agent_extra_args = build_agent_extra_args(base_agent_extra_args, model_name)
    started = time.monotonic()
    with patched_env({"COCO_AGENT_EXTRA_ARGS": agent_extra_args}):
        report = optimizer.evaluate(
            load_text(skill_path),
            tasks,
            name=f"{model_label}:{condition}:{seed}",
        )
    duration = time.monotonic() - started
    usage_summary = summarize_usage_file(usage_ledger_path, include_kinds=TARGET_AGENT_KINDS)

    write_json(report_path, report.to_dict())
    write_json(timing_path, {"duration_seconds": duration})
    write_json(usage_summary_path, usage_summary)

    return {
        "seed": seed,
        "model_label": model_label,
        "model_name": model_name,
        "condition": condition,
        "pass_rate": float(report.pass_rate),
        "average_score": float(report.average_score),
        "duration_seconds": duration,
        "calls": int(usage_summary.get("calls") or 0),
        "estimated_prompt_tokens": int(usage_summary.get("estimated_prompt_tokens") or 0),
        "estimated_completion_tokens": int(usage_summary.get("estimated_completion_tokens") or 0),
        "estimated_total_tokens": int(usage_summary.get("estimated_total_tokens") or 0),
        "run_dir": str(run_dir),
        "report_path": str(report_path),
        "usage_ledger_path": str(usage_ledger_path),
    }


def aggregate_rows(rows: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, Any]]]:
    grouped: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for row in rows:
        grouped.setdefault(str(row["model_label"]), {}).setdefault(str(row["condition"]), []).append(row)

    aggregate: dict[str, dict[str, dict[str, Any]]] = {}
    for model_label, by_condition in grouped.items():
        aggregate[model_label] = {}
        for condition, selected in by_condition.items():
            aggregate[model_label][condition] = {
                "runs": len(selected),
                "pass_rate_mean": mean(float(row["pass_rate"]) for row in selected),
                "pass_rate_stddev": stddev(float(row["pass_rate"]) for row in selected),
                "duration_seconds_mean": mean(float(row["duration_seconds"]) for row in selected),
                "duration_seconds_stddev": stddev(float(row["duration_seconds"]) for row in selected),
                "duration_seconds_total": sum(float(row["duration_seconds"]) for row in selected),
                "calls_mean": mean(float(row["calls"]) for row in selected),
                "calls_total": sum(int(row["calls"]) for row in selected),
                "estimated_prompt_tokens_mean": mean(float(row["estimated_prompt_tokens"]) for row in selected),
                "estimated_prompt_tokens_total": sum(int(row["estimated_prompt_tokens"]) for row in selected),
                "estimated_completion_tokens_mean": mean(
                    float(row["estimated_completion_tokens"]) for row in selected
                ),
                "estimated_completion_tokens_total": sum(
                    int(row["estimated_completion_tokens"]) for row in selected
                ),
                "estimated_total_tokens_mean": mean(float(row["estimated_total_tokens"]) for row in selected),
                "estimated_total_tokens_total": sum(int(row["estimated_total_tokens"]) for row in selected),
            }
    return aggregate


def build_comparisons(
    aggregate: dict[str, dict[str, dict[str, Any]]],
    *,
    strong_label: str,
    weak_label: str,
) -> dict[str, dict[str, Any]]:
    strong = aggregate.get(strong_label, {})
    weak = aggregate.get(weak_label, {})
    conditions = sorted(set(strong) & set(weak))
    comparisons: dict[str, dict[str, Any]] = {}
    for condition in conditions:
        strong_metrics = strong[condition]
        weak_metrics = weak[condition]
        comparisons[condition] = {
            "strong_label": strong_label,
            "weak_label": weak_label,
            "pass_rate_mean_delta": float(strong_metrics["pass_rate_mean"]) - float(weak_metrics["pass_rate_mean"]),
            "duration_seconds_mean_delta": float(strong_metrics["duration_seconds_mean"])
            - float(weak_metrics["duration_seconds_mean"]),
            "calls_mean_delta": float(strong_metrics["calls_mean"]) - float(weak_metrics["calls_mean"]),
            "estimated_prompt_tokens_mean_delta": float(strong_metrics["estimated_prompt_tokens_mean"])
            - float(weak_metrics["estimated_prompt_tokens_mean"]),
            "estimated_completion_tokens_mean_delta": float(strong_metrics["estimated_completion_tokens_mean"])
            - float(weak_metrics["estimated_completion_tokens_mean"]),
            "estimated_total_tokens_mean_delta": float(strong_metrics["estimated_total_tokens_mean"])
            - float(weak_metrics["estimated_total_tokens_mean"]),
        }
    return comparisons


def build_manifest(
    *,
    benchmark: Path,
    prepared_tasks_path: Path,
    no_skill_path: Path,
    explicit_skill_path: Path,
    agent_wrapper: Path,
    strong_label: str,
    strong_model: str,
    weak_label: str,
    weak_model: str,
    base_agent_extra_args: str,
    seeds: list[str],
) -> dict[str, Any]:
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "benchmark": str(benchmark),
        "prepared_tasks_path": str(prepared_tasks_path),
        "agent_wrapper": str(agent_wrapper),
        "conditions": ["no_skill", "explicit_skill"],
        "skills": {
            "no_skill": str(no_skill_path),
            "explicit_skill": str(explicit_skill_path),
        },
        "models": {
            strong_label: {"name": strong_model},
            weak_label: {"name": weak_model},
        },
        "seeds": list(seeds),
        "proxy_cost_scope": {
            "duration_seconds": "end-to-end wall-clock evaluation time",
            "calls": "target_agent_cli usage event count",
            "estimated_tokens": "char-based token estimate from target_agent_cli usage events",
        },
        "base_agent_extra_args": base_agent_extra_args,
    }


def mean(values) -> float:
    items = list(values)
    return statistics.mean(items) if items else 0.0


def stddev(values) -> float:
    items = list(values)
    return statistics.stdev(items) if len(items) > 1 else 0.0


@contextmanager
def patched_env(updates: dict[str, str]) -> Iterator[None]:
    previous = {key: os.environ.get(key) for key in updates}
    os.environ.update(updates)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument("--tasks", type=Path, default=None, help="Defaults to <benchmark>/selection.jsonl")
    parser.add_argument("--no-skill", type=Path, default=None, help="Defaults to <benchmark>/no_skill.md")
    parser.add_argument("--explicit-skill", type=Path, default=None, help="Defaults to <benchmark>/skill.md")
    parser.add_argument("--agent-wrapper", type=Path, default=DEFAULT_WRAPPER)
    parser.add_argument("--out", type=Path, default=ROOT / "runs/coding-hidden-v2-model-skill-compare")
    parser.add_argument("--strong-label", default="strong")
    parser.add_argument("--strong-model", required=True)
    parser.add_argument("--weak-label", default="weak")
    parser.add_argument("--weak-model", required=True)
    parser.add_argument("--seeds", default="seed-a,seed-b,seed-c")
    parser.add_argument("--task-limit", type=int, default=0)
    timeout = os.environ.get("COCO_TASK_TIMEOUT", "").strip()
    parser.add_argument("--task-timeout", type=int, default=int(timeout) if timeout else None)
    parser.add_argument(
        "--agent-extra-args",
        default=os.environ.get("COCO_AGENT_EXTRA_ARGS", ""),
        help="Base Coco CLI args; the script appends --model for each comparison model.",
    )
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
