"""Evaluate skill variants across real coding-agent wrappers."""

from __future__ import annotations

import os
import re
import hashlib
import json
import math
import random
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator

WORK = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(WORK))

from textskill_optimizer.cli import build_optimizer, load_plugin
from textskill_optimizer.io import load_tasks_jsonl, load_text, write_json
from textskill_optimizer.models import EvaluationReport, Task, TaskResult

from run_coco_hidden_eval import build_coco_tasks
from run_ccr_hidden_eval import build_ccr_tasks
from run_kilo_hidden_eval import build_kilo_tasks


TaskBuilder = Callable[[Path, Path], Path]


DEFAULT_SKILLS = (
    ("initial", ROOT / "examples/coding-hidden/skill.md"),
    ("coco_best_expanded", ROOT / "runs/coding-hidden-coco-skillopt-expanded-8-4-4/best_skill.md"),
    ("revised", ROOT / "work/coding_revised_skill.md"),
)

AGENTS: dict[str, tuple[TaskBuilder, Path, str]] = {
    "coco": (build_coco_tasks, ROOT / "examples/coding/coco_agent_wrapper.py", "COCO_TASK_LIMIT"),
    "ccr": (build_ccr_tasks, ROOT / "examples/coding/ccr_agent_wrapper.py", "CCR_TASK_LIMIT"),
    "kilo": (build_kilo_tasks, ROOT / "examples/coding/kilo_agent_wrapper.py", "KILO_TASK_LIMIT"),
}


def main() -> int:
    tasks = Path(os.environ.get("CROSS_AGENT_TASKS", "examples/coding-hidden/valid.jsonl"))
    out_dir = Path(os.environ.get("CROSS_AGENT_OUT", "runs/coding-hidden-cross-agent-skill-eval"))
    task_limit = os.environ.get("CROSS_AGENT_TASK_LIMIT", "").strip()
    agent_names = parse_agent_names(os.environ.get("CROSS_AGENT_AGENTS", "coco,ccr"))
    skills = parse_skills(os.environ.get("CROSS_AGENT_SKILLS", ""))
    max_retries = parse_nonnegative_int(os.environ.get("CROSS_AGENT_RETRIES", "1"), "CROSS_AGENT_RETRIES")
    health_check = parse_bool(os.environ.get("CROSS_AGENT_HEALTH_CHECK", "1"))
    health_retries = parse_nonnegative_int(
        os.environ.get("CROSS_AGENT_HEALTH_RETRIES", str(max_retries)),
        "CROSS_AGENT_HEALTH_RETRIES",
    )
    health_timeout_seconds = parse_optional_positive_int(
        os.environ.get("CROSS_AGENT_HEALTH_TIMEOUT", "120"),
        "CROSS_AGENT_HEALTH_TIMEOUT",
    )
    voting_mode = parse_voting_mode(os.environ.get("CROSS_AGENT_VOTING_MODE", "full"))
    random_seed = os.environ.get("CROSS_AGENT_RANDOM_SEED", "0")
    full_audit_rate = parse_probability(
        os.environ.get("CROSS_AGENT_FULL_AUDIT_RATE", "0"),
        "CROSS_AGENT_FULL_AUDIT_RATE",
    )
    target_failed_from = os.environ.get("CROSS_AGENT_TARGET_FAILED_FROM", "").strip()
    target_scope = os.environ.get("CROSS_AGENT_TARGET_SCOPE", "any_failed").strip()
    target_task_ids = (
        load_target_task_ids(Path(target_failed_from), target_scope)
        if target_failed_from
        else None
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    plugin = load_plugin("coding")
    optimizer = build_optimizer(plugin)

    if voting_mode == "adaptive-majority":
        rows, vote_rows = run_adaptive_majority(
            optimizer,
            tasks=tasks,
            out_dir=out_dir,
            task_limit=task_limit,
            agent_names=agent_names,
            skills=skills,
            max_retries=max_retries,
            health_check=health_check,
            health_retries=health_retries,
            health_timeout_seconds=health_timeout_seconds,
            random_seed=random_seed,
            full_audit_rate=full_audit_rate,
            target_task_ids=target_task_ids,
        )
        write_summary(
            out_dir / "summary.json",
            tasks=tasks,
            task_limit=task_limit,
            max_retries=max_retries,
            health_check=health_check,
            health_retries=health_retries,
            health_timeout_seconds=health_timeout_seconds,
            voting_mode=voting_mode,
            random_seed=random_seed,
            full_audit_rate=full_audit_rate,
            target_failed_from=target_failed_from,
            target_scope=target_scope,
            target_task_ids=target_task_ids,
            rows=rows,
            vote_rows=vote_rows,
        )
        print_summary(rows, vote_rows, out_dir / "summary.json")
        return 0

    rows = []
    for agent_name in agent_names:
        builder, wrapper, limit_env = AGENTS[agent_name]
        with optional_env(limit_env, task_limit):
            rewritten_tasks = builder(tasks, wrapper)
        loaded_tasks = load_tasks_jsonl(rewritten_tasks)
        loaded_tasks = filter_tasks_by_ids(loaded_tasks, target_task_ids)
        if not loaded_tasks:
            raise ValueError(f"No tasks loaded for agent {agent_name!r}")
        for skill_label, skill_path in skills:
            skill_text = load_text(skill_path)
            if health_check:
                health_report, health_reasons = run_agent_health_check(
                    optimizer,
                    skill_text,
                    loaded_tasks[0],
                    name=f"{agent_name}:{skill_label}:health",
                    max_retries=health_retries,
                    timeout_seconds=health_timeout_seconds,
                )
                health_path = out_dir / f"{safe_name(agent_name)}__{safe_name(skill_label)}__health.json"
                write_json(health_path, health_report.to_dict())
                if health_reasons:
                    rows.append(
                        {
                            "agent": agent_name,
                            "skill": skill_label,
                            "health_status": "failed",
                            "health_reasons": health_reasons,
                            "average_score": 0.0,
                            "pass_rate": 0.0,
                            "failed": ["agent_health_check"],
                            "task_scores": {},
                            "report": str(health_path),
                        }
                    )
                    print(
                        f"[cross-agent] skip agent={agent_name} skill={skill_label} "
                        f"health=failed reasons={','.join(health_reasons)}",
                        file=sys.stderr,
                        flush=True,
                    )
                    continue

            print(
                f"[cross-agent] start agent={agent_name} skill={skill_label} "
                f"tasks={len(loaded_tasks)} retries={max_retries}",
                file=sys.stderr,
                flush=True,
            )
            report = evaluate_with_retries(
                optimizer,
                skill_text,
                loaded_tasks,
                name=f"{agent_name}:{skill_label}",
                max_retries=max_retries,
            )
            report_path = out_dir / f"{safe_name(agent_name)}__{safe_name(skill_label)}.json"
            write_json(report_path, report.to_dict())
            failed = [result.task.id for result in report.results if not result.score.success]
            task_scores = {result.task.id: result.score.success for result in report.results}
            rows.append(
                {
                    "agent": agent_name,
                    "skill": skill_label,
                    "health_status": "passed" if health_check else "not_run",
                    "average_score": report.average_score,
                    "pass_rate": report.pass_rate,
                    "failed": failed,
                    "task_scores": task_scores,
                    "report": str(report_path),
                }
            )
            print(
                f"[cross-agent] done agent={agent_name} skill={skill_label} "
                f"avg={report.average_score:.4f} pass={report.pass_rate:.4f}",
                file=sys.stderr,
                flush=True,
            )

    summary_path = out_dir / "summary.json"
    vote_rows = build_vote_rows(rows)
    write_summary(
        summary_path,
        tasks=tasks,
        task_limit=task_limit,
        max_retries=max_retries,
        health_check=health_check,
        health_retries=health_retries,
        health_timeout_seconds=health_timeout_seconds,
        voting_mode=voting_mode,
        random_seed=random_seed,
        full_audit_rate=full_audit_rate,
        target_failed_from=target_failed_from,
        target_scope=target_scope,
        target_task_ids=target_task_ids,
        rows=rows,
        vote_rows=vote_rows,
    )
    print_summary(rows, vote_rows, summary_path)
    return 0


def parse_agent_names(raw: str) -> list[str]:
    names = [item.strip() for item in raw.split(",") if item.strip()]
    if not names:
        raise ValueError("CROSS_AGENT_AGENTS must include at least one agent")
    unknown = [name for name in names if name not in AGENTS]
    if unknown:
        raise ValueError(f"Unknown agent(s): {', '.join(unknown)}")
    return names


def parse_skills(raw: str) -> list[tuple[str, Path]]:
    if not raw.strip():
        return list(DEFAULT_SKILLS)
    skills = []
    for item in raw.split(","):
        if not item.strip():
            continue
        if ":" not in item:
            raise ValueError("CROSS_AGENT_SKILLS entries must be label:path")
        label, path = item.split(":", 1)
        skills.append((label.strip(), Path(path.strip())))
    if not skills:
        raise ValueError("CROSS_AGENT_SKILLS must include at least one skill")
    return skills


def parse_nonnegative_int(raw: str, name: str) -> int:
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a non-negative integer") from exc
    if value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def parse_optional_positive_int(raw: str, name: str) -> int | None:
    value = raw.strip()
    if not value:
        return None
    parsed = parse_nonnegative_int(value, name)
    if parsed == 0:
        raise ValueError(f"{name} must be positive or empty")
    return parsed


def parse_bool(raw: str) -> bool:
    return raw.strip().casefold() not in {"0", "false", "no", "off"}


def parse_voting_mode(raw: str) -> str:
    value = raw.strip().casefold() or "full"
    if value not in {"full", "adaptive-majority"}:
        raise ValueError("CROSS_AGENT_VOTING_MODE must be 'full' or 'adaptive-majority'")
    return value


def parse_probability(raw: str, name: str) -> float:
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number between 0 and 1") from exc
    if value < 0.0 or value > 1.0:
        raise ValueError(f"{name} must be between 0 and 1")
    return value


def load_target_task_ids(summary_path: Path, scope: str) -> set[str]:
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    scope = scope.strip().casefold() or "any_failed"
    if scope not in {"any_failed", "majority_failed"}:
        raise ValueError("CROSS_AGENT_TARGET_SCOPE must be 'any_failed' or 'majority_failed'")

    target_ids: set[str] = set()
    for vote in payload.get("votes", []):
        if not isinstance(vote, dict):
            continue
        majority_success = bool(vote.get("majority_success"))
        tie = bool(vote.get("tie"))
        if tie or not majority_success:
            task = vote.get("task")
            if task:
                target_ids.add(str(task))

    if scope == "any_failed":
        for row in payload.get("rows", []):
            if not isinstance(row, dict):
                continue
            for task in row.get("failed", []):
                task_id = str(task)
                if task_id != "agent_health_check":
                    target_ids.add(task_id)

    if not target_ids:
        raise ValueError(f"No targeted tasks found in {summary_path} with scope {scope!r}")
    return target_ids


def filter_tasks_by_ids(tasks: list[Task], target_task_ids: set[str] | None) -> list[Task]:
    if target_task_ids is None:
        return tasks
    return [task for task in tasks if task.id in target_task_ids]


@contextmanager
def optional_env(name: str, value: str) -> Iterator[None]:
    previous = os.environ.get(name)
    if value:
        os.environ[name] = value
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = previous


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "item"


def run_adaptive_majority(
    optimizer: Any,
    *,
    tasks: Path,
    out_dir: Path,
    task_limit: str,
    agent_names: list[str],
    skills: list[tuple[str, Path]],
    max_retries: int,
    health_check: bool,
    health_retries: int,
    health_timeout_seconds: int | None,
    random_seed: str,
    full_audit_rate: float,
    target_task_ids: set[str] | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if len(agent_names) != 3:
        raise ValueError("adaptive-majority requires exactly three agents")

    tasks_by_agent = load_tasks_for_agents(
        tasks,
        task_limit=task_limit,
        agent_names=agent_names,
        target_task_ids=target_task_ids,
    )
    canonical_tasks = tasks_by_agent[agent_names[0]]
    if not canonical_tasks:
        raise ValueError("No tasks loaded for adaptive-majority")

    rows: list[dict[str, Any]] = []
    vote_rows: list[dict[str, Any]] = []
    health_cache: dict[tuple[str, str], dict[str, Any]] = {}

    for skill_label, skill_path in skills:
        skill_text = load_text(skill_path)
        healthy_agents: list[str] = []
        health_status_by_agent: dict[str, str] = {}
        health_reasons_by_agent: dict[str, list[str]] = {}
        health_report_by_agent: dict[str, str] = {}

        for agent_name in agent_names:
            health_record = get_health_record(
                optimizer,
                skill_text,
                tasks_by_agent[agent_name][0],
                out_dir=out_dir,
                agent_name=agent_name,
                skill_label=skill_label,
                max_retries=health_retries,
                timeout_seconds=health_timeout_seconds,
                enabled=health_check,
                cache=health_cache,
            )
            health_status_by_agent[agent_name] = str(health_record["status"])
            health_reasons_by_agent[agent_name] = list(health_record.get("reasons", []))
            health_report_by_agent[agent_name] = str(health_record.get("report") or "")
            if health_record["status"] != "failed":
                healthy_agents.append(agent_name)

        if len(healthy_agents) < 2:
            for agent_name in agent_names:
                rows.append(
                    build_agent_summary_row(
                        agent_name=agent_name,
                        skill_label=skill_label,
                        health_status=health_status_by_agent[agent_name],
                        health_reasons=health_reasons_by_agent[agent_name],
                        health_report=health_report_by_agent[agent_name],
                        results=[],
                        task_scores={},
                        skipped_tasks=[],
                        task_anomalies={},
                        report_path=None,
                    )
                )
            continue

        print(
            f"[cross-agent] adaptive start skill={skill_label} "
            f"agents={','.join(healthy_agents)} tasks={len(canonical_tasks)} "
            f"audit_rate={full_audit_rate:.2f}",
            file=sys.stderr,
            flush=True,
        )

        full_audit_task_ids = select_full_audit_task_ids(
            random_seed,
            skill_label,
            [task.id for task in canonical_tasks],
            full_audit_rate,
        )

        results_by_agent: dict[str, list[TaskResult]] = {agent: [] for agent in agent_names}
        task_scores_by_agent: dict[str, dict[str, bool]] = {agent: {} for agent in agent_names}
        skipped_by_agent: dict[str, list[str]] = {agent: [] for agent in agent_names}
        anomalies_by_agent: dict[str, dict[str, list[str]]] = {agent: {} for agent in agent_names}
        task_by_agent = {
            agent: {task.id: task for task in loaded_tasks}
            for agent, loaded_tasks in tasks_by_agent.items()
        }

        for task in canonical_tasks:
            order = stable_agent_order(healthy_agents, random_seed, skill_label, task.id)
            force_full_audit = task.id in full_audit_task_ids
            task_results, vote_row = evaluate_adaptive_task(
                optimizer,
                skill_text,
                task_by_agent,
                order,
                skill_label=skill_label,
                task_id=task.id,
                max_retries=max_retries,
                force_full_audit=force_full_audit,
            )
            vote_rows.append(vote_row)
            run_agents = set(task_results)
            for agent_name, result in task_results.items():
                results_by_agent[agent_name].append(result)
                reasons = retryable_anomaly_reasons(result)
                if reasons:
                    anomalies_by_agent[agent_name][task.id] = reasons
                    continue
                task_scores_by_agent[agent_name][task.id] = result.score.success
            for agent_name in healthy_agents:
                if agent_name not in run_agents:
                    skipped_by_agent[agent_name].append(task.id)
            print(
                f"[cross-agent] adaptive task={task.id} skill={skill_label} "
                f"vote={vote_row['passed']}/{vote_row['total']} "
                f"reason={vote_row['decision_reason']} "
                f"agents={','.join(vote_row['agents_run'])}",
                file=sys.stderr,
                flush=True,
            )

        for agent_name in agent_names:
            report_path = None
            if results_by_agent[agent_name]:
                report = EvaluationReport(
                    name=f"{agent_name}:{skill_label}:adaptive",
                    results=results_by_agent[agent_name],
                )
                report_path = out_dir / f"{safe_name(agent_name)}__{safe_name(skill_label)}.json"
                write_json(report_path, report.to_dict())
            rows.append(
                build_agent_summary_row(
                    agent_name=agent_name,
                    skill_label=skill_label,
                    health_status=health_status_by_agent[agent_name],
                    health_reasons=health_reasons_by_agent[agent_name],
                    health_report=health_report_by_agent[agent_name],
                    results=results_by_agent[agent_name],
                    task_scores=task_scores_by_agent[agent_name],
                    skipped_tasks=skipped_by_agent[agent_name],
                    task_anomalies=anomalies_by_agent[agent_name],
                    report_path=report_path,
                )
            )

    return rows, vote_rows


def load_tasks_for_agents(
    tasks: Path,
    *,
    task_limit: str,
    agent_names: list[str],
    target_task_ids: set[str] | None,
) -> dict[str, list[Task]]:
    tasks_by_agent = {}
    for agent_name in agent_names:
        builder, wrapper, limit_env = AGENTS[agent_name]
        with optional_env(limit_env, task_limit):
            rewritten_tasks = builder(tasks, wrapper)
        loaded_tasks = filter_tasks_by_ids(load_tasks_jsonl(rewritten_tasks), target_task_ids)
        if not loaded_tasks:
            raise ValueError(f"No tasks loaded for agent {agent_name!r}")
        tasks_by_agent[agent_name] = loaded_tasks
    return tasks_by_agent


def get_health_record(
    optimizer: Any,
    skill_text: str,
    task: Task,
    *,
    out_dir: Path,
    agent_name: str,
    skill_label: str,
    max_retries: int,
    timeout_seconds: int | None,
    enabled: bool,
    cache: dict[tuple[str, str], dict[str, Any]],
) -> dict[str, Any]:
    cache_key = (agent_name, skill_label)
    if cache_key in cache:
        return cache[cache_key]
    if not enabled:
        record = {"status": "not_run", "reasons": [], "report": ""}
        cache[cache_key] = record
        return record

    health_report, health_reasons = run_agent_health_check(
        optimizer,
        skill_text,
        task,
        name=f"{agent_name}:{skill_label}:health",
        max_retries=max_retries,
        timeout_seconds=timeout_seconds,
    )
    health_path = out_dir / f"{safe_name(agent_name)}__{safe_name(skill_label)}__health.json"
    write_json(health_path, health_report.to_dict())
    record = {
        "status": "failed" if health_reasons else "passed",
        "reasons": health_reasons,
        "report": str(health_path),
    }
    cache[cache_key] = record
    if health_reasons:
        print(
            f"[cross-agent] skip agent={agent_name} skill={skill_label} "
            f"health=failed reasons={','.join(health_reasons)}",
            file=sys.stderr,
            flush=True,
        )
    return record


def evaluate_adaptive_task(
    optimizer: Any,
    skill_text: str,
    task_by_agent: dict[str, dict[str, Task]],
    agent_order: list[str],
    *,
    skill_label: str,
    task_id: str,
    max_retries: int,
    force_full_audit: bool,
) -> tuple[dict[str, TaskResult], dict[str, Any]]:
    task_results: dict[str, TaskResult] = {}
    valid_votes: list[tuple[str, bool]] = []
    invalid_votes: list[dict[str, Any]] = []

    for agent_name in agent_order:
        task = task_by_agent[agent_name][task_id]
        result = evaluate_one_task_with_retries(
            optimizer,
            skill_text,
            task,
            name=f"{agent_name}:{skill_label}:adaptive",
            max_retries=max_retries,
        )
        task_results[agent_name] = result
        reasons = retryable_anomaly_reasons(result)
        if reasons:
            invalid_votes.append({"agent": agent_name, "reasons": reasons})
        else:
            valid_votes.append((agent_name, result.score.success))

        if should_stop_adaptive_votes(valid_votes, force_full_audit):
            break

    return task_results, build_adaptive_vote_row(
        skill_label=skill_label,
        task_id=task_id,
        agent_order=agent_order,
        task_results=task_results,
        valid_votes=valid_votes,
        invalid_votes=invalid_votes,
        force_full_audit=force_full_audit,
    )


def should_stop_adaptive_votes(
    valid_votes: list[tuple[str, bool]],
    force_full_audit: bool,
) -> bool:
    if force_full_audit:
        return len(valid_votes) >= 3
    if len(valid_votes) < 2:
        return False
    if valid_votes[0][1] == valid_votes[1][1]:
        return True
    return len(valid_votes) >= 3


def build_adaptive_vote_row(
    *,
    skill_label: str,
    task_id: str,
    agent_order: list[str],
    task_results: dict[str, TaskResult],
    valid_votes: list[tuple[str, bool]],
    invalid_votes: list[dict[str, Any]],
    force_full_audit: bool,
) -> dict[str, Any]:
    passed = sum(1 for _, success in valid_votes if success)
    total = len(valid_votes)
    tie = total % 2 == 0 and total > 0 and passed == total / 2
    majority_success = total > 0 and passed > total / 2
    if total < 2:
        decision_reason = "insufficient_valid_votes"
    elif force_full_audit and len(task_results) == len(agent_order):
        decision_reason = "full_audit"
    elif len(valid_votes) == 2 and valid_votes[0][1] == valid_votes[1][1]:
        decision_reason = "first_two_agree"
    elif total >= 3:
        decision_reason = "third_agent_breaker"
    else:
        decision_reason = "inconclusive"

    agents_run = list(task_results)
    return {
        "skill": skill_label,
        "task": task_id,
        "passed": passed,
        "total": total,
        "majority_success": majority_success,
        "tie": tie,
        "decision_reason": decision_reason,
        "agents_run": agents_run,
        "skipped_agents": [agent for agent in agent_order if agent not in task_results],
        "valid_votes": [
            {"agent": agent, "success": success}
            for agent, success in valid_votes
        ],
        "invalid_votes": invalid_votes,
        "force_full_audit": force_full_audit,
    }


def stable_agent_order(
    agent_names: list[str],
    random_seed: str,
    skill_label: str,
    task_id: str,
) -> list[str]:
    ordered = list(agent_names)
    seed = stable_int(f"{random_seed}:order:{skill_label}:{task_id}")
    random.Random(seed).shuffle(ordered)
    return ordered


def select_full_audit_task_ids(
    random_seed: str,
    skill_label: str,
    task_ids: list[str],
    full_audit_rate: float,
) -> set[str]:
    if full_audit_rate <= 0 or not task_ids:
        return set()
    if full_audit_rate >= 1:
        return set(task_ids)

    audit_count = min(len(task_ids), math.ceil(len(task_ids) * full_audit_rate))
    ranked = sorted(
        task_ids,
        key=lambda task_id: stable_int(f"{random_seed}:audit:{skill_label}:{task_id}"),
    )
    return set(ranked[:audit_count])


def stable_int(value: str) -> int:
    return int.from_bytes(hashlib.sha256(value.encode("utf-8")).digest()[:8], "big")


def stable_unit_interval(value: str) -> float:
    return stable_int(value) / float(2**64 - 1)


def build_agent_summary_row(
    *,
    agent_name: str,
    skill_label: str,
    health_status: str,
    health_reasons: list[str],
    health_report: str,
    results: list[TaskResult],
    task_scores: dict[str, bool],
    skipped_tasks: list[str],
    task_anomalies: dict[str, list[str]],
    report_path: Path | None,
) -> dict[str, Any]:
    valid_total = len(task_scores)
    valid_passed = sum(1 for success in task_scores.values() if success)
    failed = [task_id for task_id, success in task_scores.items() if not success]
    if health_status == "failed":
        failed = ["agent_health_check"]
    average_score = valid_passed / valid_total if valid_total else None
    pass_rate = valid_passed / valid_total if valid_total else None
    if health_status == "failed":
        average_score = 0.0
        pass_rate = 0.0
    return {
        "agent": agent_name,
        "skill": skill_label,
        "health_status": health_status,
        "health_reasons": health_reasons,
        "average_score": average_score,
        "pass_rate": pass_rate,
        "failed": failed,
        "task_scores": task_scores,
        "skipped_tasks": skipped_tasks,
        "task_anomalies": task_anomalies,
        "report": str(report_path) if report_path is not None else health_report,
    }


def run_agent_health_check(
    optimizer: Any,
    skill_text: str,
    task: Task,
    *,
    name: str,
    max_retries: int,
    timeout_seconds: int | None,
) -> tuple[EvaluationReport, list[str]]:
    health_task = task_with_timeout(task, timeout_seconds)
    report = evaluate_with_retries(
        optimizer,
        skill_text,
        [health_task],
        name=name,
        max_retries=max_retries,
    )
    result = report.results[0]
    return report, retryable_anomaly_reasons(result)


def task_with_timeout(task: Task, timeout_seconds: int | None) -> Task:
    if timeout_seconds is None:
        return task
    metadata = dict(task.metadata)
    metadata["timeout_seconds"] = timeout_seconds
    return Task(id=task.id, input=task.input, expected=task.expected, metadata=metadata)


def evaluate_with_retries(
    optimizer: Any,
    skill_text: str,
    tasks: list[Task],
    *,
    name: str,
    max_retries: int,
) -> EvaluationReport:
    results: list[TaskResult] = []
    for task in tasks:
        selected = evaluate_one_task_with_retries(
            optimizer,
            skill_text,
            task,
            name=name,
            max_retries=max_retries,
        )
        results.append(selected)
    return EvaluationReport(name=name, results=results)


def evaluate_one_task_with_retries(
    optimizer: Any,
    skill_text: str,
    task: Task,
    *,
    name: str,
    max_retries: int,
) -> TaskResult:
    attempts: list[dict[str, Any]] = []
    selected: TaskResult | None = None
    for attempt_index in range(max_retries + 1):
        report = optimizer.evaluate(
            skill_text,
            [task],
            name=f"{name}:{task.id}:attempt:{attempt_index}",
        )
        result = report.results[0]
        reasons = retryable_anomaly_reasons(result)
        attempts.append(build_attempt_record(result, attempt_index, reasons))
        selected = result
        if not reasons:
            break

    if selected is None:
        raise RuntimeError(f"No evaluation result produced for task {task.id!r}")

    selected.output.metadata["retry_policy"] = {
        "max_retries": max_retries,
        "attempt_count": len(attempts),
        "attempts": attempts,
        "selected_attempt": len(attempts) - 1,
    }
    return selected


def retryable_anomaly_reasons(result: TaskResult) -> list[str]:
    metadata = result.output.metadata
    value = result.output.value if isinstance(result.output.value, dict) else {}
    agent = metadata.get("agent") if isinstance(metadata.get("agent"), dict) else {}
    post_test = metadata.get("post_test") if isinstance(metadata.get("post_test"), dict) else {}

    reasons: list[str] = []
    agent_returncode = as_int(value.get("agent_returncode", agent.get("returncode")))
    post_test_returncode = as_int(post_test.get("returncode"))
    agent_timed_out = bool(agent.get("timed_out")) or agent_returncode == 124
    diff = str(metadata.get("diff") or "")
    stdout = str(agent.get("stdout") or "")
    stderr = str(agent.get("stderr") or "")
    agent_text = f"{stdout}\n{stderr}"

    if result.score.success:
        return reasons

    if agent_timed_out:
        reasons.append("agent_timeout")
    elif agent_returncode is not None and agent_returncode != 0:
        reasons.append("agent_nonzero_returncode")

    if (
        post_test_returncode is not None
        and post_test_returncode != 0
        and not diff.strip()
        and not result.score.success
    ):
        reasons.append("failed_post_test_without_repo_change")

    if "<seed:tool" in agent_text and "</seed:tool_call>" in agent_text and not diff.strip():
        reasons.append("malformed_tool_call_without_repo_change")

    if not stdout.strip() and not stderr.strip() and not diff.strip() and not result.score.success:
        reasons.append("empty_agent_output_without_repo_change")

    return reasons


def build_attempt_record(result: TaskResult, attempt_index: int, reasons: list[str]) -> dict[str, Any]:
    metadata = result.output.metadata
    value = result.output.value if isinstance(result.output.value, dict) else {}
    agent = metadata.get("agent") if isinstance(metadata.get("agent"), dict) else {}
    post_test = metadata.get("post_test") if isinstance(metadata.get("post_test"), dict) else {}
    diff = str(metadata.get("diff") or "")
    return {
        "attempt": attempt_index,
        "success": result.score.success,
        "score": result.score.value,
        "retryable": bool(reasons),
        "reasons": list(reasons),
        "agent_returncode": as_int(value.get("agent_returncode", agent.get("returncode"))),
        "post_test_returncode": as_int(post_test.get("returncode")),
        "diff_empty": not diff.strip(),
    }


def as_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def build_vote_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    by_skill: dict[str, dict[str, list[bool]]] = {}
    for row in rows:
        skill = str(row["skill"])
        task_scores = row.get("task_scores")
        if not isinstance(task_scores, dict):
            continue
        skill_votes = by_skill.setdefault(skill, {})
        for task_id, success in task_scores.items():
            skill_votes.setdefault(str(task_id), []).append(bool(success))

    vote_rows = []
    for skill, task_votes in sorted(by_skill.items()):
        for task_id, votes in sorted(task_votes.items()):
            passed = sum(1 for vote in votes if vote)
            total = len(votes)
            vote_rows.append(
                {
                    "skill": skill,
                    "task": task_id,
                    "passed": passed,
                    "total": total,
                    "majority_success": passed > total / 2,
                    "tie": total % 2 == 0 and passed == total / 2,
                }
            )
    return vote_rows


def write_summary(
    summary_path: Path,
    *,
    tasks: Path,
    task_limit: str,
    max_retries: int,
    health_check: bool,
    health_retries: int,
    health_timeout_seconds: int | None,
    voting_mode: str,
    random_seed: str,
    full_audit_rate: float,
    target_failed_from: str,
    target_scope: str,
    target_task_ids: set[str] | None,
    rows: list[dict[str, Any]],
    vote_rows: list[dict[str, Any]],
) -> None:
    call_savings = build_call_savings(rows, vote_rows)
    full_audit_tasks = sorted(
        str(row["task"])
        for row in vote_rows
        if row.get("force_full_audit")
    )
    write_json(
        summary_path,
        {
            "tasks": str(tasks),
            "task_limit": task_limit or None,
            "max_retries": max_retries,
            "health_check": health_check,
            "health_retries": health_retries if health_check else None,
            "health_timeout_seconds": health_timeout_seconds if health_check else None,
            "voting_mode": voting_mode,
            "random_seed": random_seed,
            "full_audit_rate": full_audit_rate,
            "full_audit_tasks": full_audit_tasks,
            "full_audit_task_count": len(full_audit_tasks),
            "target_failed_from": target_failed_from or None,
            "target_scope": target_scope if target_failed_from else None,
            "target_task_ids": sorted(target_task_ids) if target_task_ids is not None else None,
            "call_savings": call_savings,
            "rows": rows,
            "votes": vote_rows,
        },
    )


def build_call_savings(
    rows: list[dict[str, Any]],
    vote_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    agent_names = sorted({str(row["agent"]) for row in rows})
    total_agents = len(agent_names)
    planned_calls = len(vote_rows) * total_agents
    actual_calls = sum(len(row.get("agents_run", [])) for row in vote_rows)
    skipped_calls = sum(len(row.get("skipped_agents", [])) for row in vote_rows)
    saved_calls = planned_calls - actual_calls
    by_skill: dict[str, dict[str, Any]] = {}
    for row in vote_rows:
        skill = str(row["skill"])
        skill_summary = by_skill.setdefault(
            skill,
            {
                "planned_agent_calls": 0,
                "actual_agent_calls": 0,
                "skipped_agent_calls": 0,
                "saved_agent_calls": 0,
                "saved_rate": 0.0,
                "tasks": 0,
            },
        )
        skill_summary["tasks"] += 1
        skill_summary["planned_agent_calls"] += total_agents
        skill_summary["actual_agent_calls"] += len(row.get("agents_run", []))
        skill_summary["skipped_agent_calls"] += len(row.get("skipped_agents", []))
    for skill_summary in by_skill.values():
        skill_summary["saved_agent_calls"] = (
            skill_summary["planned_agent_calls"] - skill_summary["actual_agent_calls"]
        )
        planned = int(skill_summary["planned_agent_calls"])
        skill_summary["saved_rate"] = (
            float(skill_summary["saved_agent_calls"]) / planned if planned else 0.0
        )

    return {
        "agent_count": total_agents,
        "task_votes": len(vote_rows),
        "planned_agent_calls": planned_calls,
        "actual_agent_calls": actual_calls,
        "skipped_agent_calls": skipped_calls,
        "saved_agent_calls": saved_calls,
        "saved_rate": saved_calls / planned_calls if planned_calls else 0.0,
        "by_skill": by_skill,
    }


def print_summary(rows: list[dict[str, object]], vote_rows: list[dict[str, object]], summary_path: Path) -> None:
    print("| agent | skill | health | avg | pass | failed |")
    print("|---|---|---|---:|---:|---|")
    for row in rows:
        failed = row["failed"]
        failed_text = ", ".join(failed) if isinstance(failed, list) and failed else "-"
        health = str(row.get("health_status", "unknown"))
        average_score = format_optional_float(row.get("average_score"))
        pass_rate = format_optional_float(row.get("pass_rate"))
        print(
            f"| {row['agent']} | {row['skill']} | {health} | "
            f"{average_score} | {pass_rate} | {failed_text} |"
        )
    if vote_rows:
        print()
        print("| skill | task | vote | majority |")
        print("|---|---|---:|---|")
        for row in vote_rows:
            majority = "tie" if row["tie"] else ("pass" if row["majority_success"] else "fail")
            print(f"| {row['skill']} | {row['task']} | {row['passed']}/{row['total']} | {majority} |")
        savings = build_call_savings([dict(row) for row in rows], [dict(row) for row in vote_rows])
        print()
        print("| planned calls | actual calls | saved calls | saved rate |")
        print("|---:|---:|---:|---:|")
        print(
            f"| {savings['planned_agent_calls']} | {savings['actual_agent_calls']} | "
            f"{savings['saved_agent_calls']} | {float(savings['saved_rate']):.2%} |"
        )
    print(f"summary={summary_path}")


def format_optional_float(value: object) -> str:
    if value is None:
        return "-"
    return f"{float(value):.4f}"


if __name__ == "__main__":
    raise SystemExit(main())
