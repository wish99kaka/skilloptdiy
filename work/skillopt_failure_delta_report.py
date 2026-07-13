#!/usr/bin/env python3
"""Build a compact failure-delta report from an existing SkillOpt smoke run.

The script is read-only with respect to experiment execution: it reads run
artifacts, summarizes proposal/gate/contract deltas, and writes report files.
It does not call external models, target agents, or scorers.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


SCHEMA_VERSION = 1


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = build_failure_delta_report(args.run_dir)
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    if args.markdown_out:
        args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_out.write_text(render_markdown(report), encoding="utf-8")
    print(render_compact_summary(report) if args.quiet else text)
    return 0


def build_failure_delta_report(run_dir: str | Path) -> dict[str, Any]:
    run_path = Path(run_dir)
    summary = load_json_if_exists(run_path / "summary.json")
    smoke_gate = load_json_if_exists(run_path / "smoke_gate_report.json")
    development_gate = dict(summary.get("development_gate") or smoke_gate.get("development_gate") or {})
    rows = [row for row in summary.get("rows", []) if isinstance(row, dict)]
    best_baseline_condition = str(development_gate.get("best_baseline_condition") or "human_skill")
    seed_rows = build_seed_rows(rows, best_baseline_condition)

    steps: list[dict[str, Any]] = []
    for seed_dir in sorted(path for path in run_path.glob("seed-*") if path.is_dir()):
        executive_dir = seed_dir / "executive"
        if not executive_dir.exists():
            continue
        proposals_by_candidate = load_proposals_by_candidate(executive_dir)
        gates_by_candidate = load_gates_by_candidate(executive_dir)
        result = load_json_if_exists(executive_dir / "result.json")
        history = [item for item in result.get("history", []) if isinstance(item, dict)]
        for item in history:
            candidate = str(item.get("candidate") or "")
            if candidate == "initial" or not candidate:
                continue
            proposal_records = proposals_by_candidate.get(candidate, [])
            gate = gates_by_candidate.get(candidate) or nested_dict(item, "metadata", "validation_gate")
            steps.append(
                build_step_record(
                    seed=seed_dir.name,
                    candidate=candidate,
                    history_item=item,
                    proposal_records=proposal_records,
                    gate=gate,
                )
            )

    report_summary = build_report_summary(
        run_path=run_path,
        summary=summary,
        smoke_gate=smoke_gate,
        development_gate=development_gate,
        steps=steps,
        seed_rows=seed_rows,
    )
    contract_summary = build_contract_summary(steps)
    failure_modes = build_failure_modes(steps)
    diagnosis = build_diagnosis(
        report_summary,
        contract_summary,
        failure_modes,
        seed_rows,
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "run_dir": str(run_path),
        "summary": report_summary,
        "seed_rows": seed_rows,
        "steps": steps,
        "contract_summary": contract_summary,
        "failure_modes": failure_modes,
        "diagnosis": diagnosis,
    }


def build_step_record(
    *,
    seed: str,
    candidate: str,
    history_item: dict[str, Any],
    proposal_records: list[dict[str, Any]],
    gate: dict[str, Any],
) -> dict[str, Any]:
    metadata = dict(history_item.get("metadata") or {})
    selected_edits = [edit for edit in metadata.get("selected_edits", []) if isinstance(edit, dict)]
    audits = [record.get("proposal_targeting_audit") for record in proposal_records]
    audits = [audit for audit in audits if isinstance(audit, dict)]
    proposals = [
        proposal
        for record in proposal_records
        for proposal in record.get("proposals", [])
        if isinstance(proposal, dict)
    ]
    priority_contracts = sorted(
        {
            str(contract)
            for audit in audits
            for contract in audit.get("priority_contracts", [])
            if str(contract)
        }
    )
    targeted_contracts = sorted(
        {
            contract
            for proposal in proposals
            for contract in normalized_string_list(nested_dict(proposal, "metadata").get("targeted_contracts"))
        }
    )
    protected_contracts = sorted(
        {
            contract
            for proposal in proposals
            for contract in normalized_string_list(nested_dict(proposal, "metadata").get("protected_contracts"))
        }
    )
    audit_protected_contracts = sorted(
        {
            contract
            for audit in audits
            for audit_proposal in audit.get("proposals", [])
            if isinstance(audit_proposal, dict)
            for contract in normalized_string_list(audit_proposal.get("protected_contracts"))
        }
    )
    anti_regression_contracts = sorted(
        {
            str(item.get("contract"))
            for audit in audits
            for item in nested_list(audit, "proposal_policy", "anti_regression_contracts")
            if isinstance(item, dict) and item.get("contract")
        }
    )
    protected_priority_contracts = sorted(
        {
            str(item.get("contract"))
            for audit in audits
            for item in nested_list(audit, "proposal_policy", "protected_priority_contracts")
            if isinstance(item, dict) and item.get("contract")
        }
    )
    guarded_contracts = sorted(
        set(protected_contracts)
        .union(audit_protected_contracts)
        .union(anti_regression_contracts)
        .union(protected_priority_contracts)
    )
    evidence_sources = sorted(
        {
            str(nested_dict(proposal, "metadata").get("evidence_source") or "")
            for proposal in proposals
            if str(nested_dict(proposal, "metadata").get("evidence_source") or "")
        }
    )
    audit_required = any(bool(audit.get("required")) for audit in audits)
    audit_failed = any(int(audit.get("missing_targeted_contract_count") or 0) > 0 for audit in audits if audit.get("required"))
    evidence_available = any(bool(audit.get("contract_rejection_evidence_available")) for audit in audits)
    contract_evidence = dict(gate.get("contract_evidence") or {})
    deltas = {
        str(contract): dict(payload)
        for contract, payload in dict(contract_evidence.get("contract_deltas") or {}).items()
        if isinstance(payload, dict)
    }
    regressed = sorted(contract for contract, payload in deltas.items() if float(payload.get("delta") or 0.0) < 0)
    improved = sorted(contract for contract, payload in deltas.items() if float(payload.get("delta") or 0.0) > 0)
    unchanged_failed = sorted(
        contract
        for contract, payload in deltas.items()
        if float(payload.get("delta") or 0.0) == 0.0 and float(payload.get("candidate_accuracy") or 0.0) < 1.0
    )
    current_mean = maybe_float(gate.get("current_mean"))
    candidate_mean = maybe_float(gate.get("candidate_mean"))
    if candidate_mean is None:
        candidate_mean = maybe_float(history_item.get("validation_score"))
    mean_delta = (
        None
        if current_mean is None or candidate_mean is None
        else candidate_mean - current_mean
    )
    targeted_outcomes = build_contract_outcomes(targeted_contracts, deltas)
    priority_outcomes = build_contract_outcomes(priority_contracts, deltas)
    guarded_outcomes = build_contract_outcomes(guarded_contracts, deltas)
    failure_labels = classify_step(
        accepted=bool(history_item.get("accepted")),
        current_mean=current_mean,
        candidate_mean=candidate_mean,
        audit_required=audit_required,
        targeted_contracts=targeted_contracts,
        priority_contracts=priority_contracts,
        targeted_outcomes=targeted_outcomes,
        priority_outcomes=priority_outcomes,
        regressed=regressed,
        unchanged_failed=unchanged_failed,
    )
    return {
        "seed": seed,
        "candidate": candidate,
        "epoch": history_item.get("epoch"),
        "step": metadata.get("step"),
        "phase": metadata.get("phase"),
        "accepted": bool(history_item.get("accepted")),
        "current_mean": current_mean,
        "candidate_mean": candidate_mean,
        "mean_delta": mean_delta,
        "validation_score": history_item.get("validation_score"),
        "rejection_reason": metadata.get("rejection_reason"),
        "proposal_audit": {
            "required": audit_required,
            "evidence_available": evidence_available,
            "failed": audit_failed,
            "missing_targeted_contract_count": sum(
                int(audit.get("missing_targeted_contract_count") or 0)
                for audit in audits
                if audit.get("required")
            ),
            "priority_contracts": priority_contracts,
            "evidence_sources": evidence_sources,
        },
        "targeted_contracts": targeted_contracts,
        "protected_contracts": guarded_contracts,
        "anti_regression_contracts": anti_regression_contracts,
        "protected_priority_contracts": protected_priority_contracts,
        "selected_edit_count": len(selected_edits),
        "selected_edit_summaries": summarize_edits(selected_edits),
        "contract_delta_summary": {
            "improved": improved,
            "regressed": regressed,
            "unchanged_failed": unchanged_failed,
            "summary": contract_evidence.get("summary", {}),
        },
        "targeted_contract_outcomes": targeted_outcomes,
        "priority_contract_outcomes": priority_outcomes,
        "protected_contract_outcomes": guarded_outcomes,
        "failure_labels": failure_labels,
    }


def classify_step(
    *,
    accepted: bool,
    current_mean: float | None,
    candidate_mean: float | None,
    audit_required: bool,
    targeted_contracts: list[str],
    priority_contracts: list[str],
    targeted_outcomes: dict[str, dict[str, Any]],
    priority_outcomes: dict[str, dict[str, Any]],
    regressed: list[str],
    unchanged_failed: list[str],
) -> list[str]:
    labels = []
    if accepted and candidate_mean is not None and candidate_mean < 1.0:
        labels.append("accepted_partial_improvement")
    if not accepted:
        labels.append("rejected")
        if candidate_mean is None or current_mean is None:
            labels.append("rejected_without_validation")
        elif candidate_mean <= current_mean:
            labels.append("no_selection_gain")
        if regressed:
            labels.append("contract_regression")
        if unchanged_failed:
            labels.append("persistent_contract_failure")
    if audit_required:
        labels.append("contract_evidence_used")
        if not accepted:
            labels.append("evidence_guided_rejected")
        evaluated_targeted = [outcome for outcome in targeted_outcomes.values() if outcome["delta"] is not None]
        evaluated_priority = [outcome for outcome in priority_outcomes.values() if outcome["delta"] is not None]
        if targeted_contracts and not evaluated_targeted:
            labels.append("targeted_contracts_not_evaluated")
        elif evaluated_targeted and not any(outcome["delta"] > 0 for outcome in evaluated_targeted):
            labels.append("targeted_contracts_not_improved")
        if priority_contracts and not evaluated_priority:
            labels.append("priority_contracts_not_evaluated")
        elif evaluated_priority and not any(outcome["delta"] > 0 for outcome in evaluated_priority):
            labels.append("priority_contracts_not_improved")
        if any(outcome["delta"] is not None and outcome["delta"] < 0 for outcome in priority_outcomes.values()):
            labels.append("priority_contract_regression")
    return labels


def build_contract_outcomes(
    contracts: Iterable[str],
    deltas: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    outcomes = {}
    for contract in sorted(set(contracts)):
        if contract not in deltas:
            outcomes[contract] = {
                "current_accuracy": None,
                "candidate_accuracy": None,
                "delta": None,
                "status": "not_evaluated",
            }
            continue
        payload = dict(deltas.get(contract) or {})
        current_accuracy = float(payload.get("current_accuracy") or 0.0)
        candidate_accuracy = float(payload.get("candidate_accuracy") or 0.0)
        delta = float(payload.get("delta") or 0.0)
        if delta > 0:
            status = "improved"
        elif delta < 0:
            status = "regressed"
        elif candidate_accuracy < 1.0:
            status = "unchanged_failed"
        else:
            status = "unchanged_passed"
        outcomes[contract] = {
            "current_accuracy": current_accuracy,
            "candidate_accuracy": candidate_accuracy,
            "delta": delta,
            "status": status,
        }
    return outcomes


def build_report_summary(
    *,
    run_path: Path,
    summary: dict[str, Any],
    smoke_gate: dict[str, Any],
    development_gate: dict[str, Any],
    steps: list[dict[str, Any]],
    seed_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    required_steps = [step for step in steps if step["proposal_audit"]["required"]]
    accepted_steps = [step for step in steps if step["accepted"]]
    rejected_steps = [step for step in steps if not step["accepted"]]
    usage = nested_dict(summary, "aggregate", "executive", "experiment_internal_usage_summary")
    return {
        "run_dir": str(run_path),
        "smoke_status": smoke_gate.get("status"),
        "smoke_reason": smoke_gate.get("reason"),
        "development_gate_passed": bool(development_gate.get("passed")),
        "development_gate_blocked_reason": development_gate.get("blocked_reason"),
        "executive_mean": development_gate.get("executive_mean"),
        "best_baseline_condition": development_gate.get("best_baseline_condition"),
        "best_baseline_mean": development_gate.get("best_baseline_mean"),
        "mean_delta": development_gate.get("mean_delta"),
        "seed_wins_vs_best_baseline": development_gate.get("seed_wins_vs_best_baseline"),
        "required_seed_wins": development_gate.get("required_seed_wins"),
        "step_count": len(steps),
        "accepted_step_count": len(accepted_steps),
        "rejected_step_count": len(rejected_steps),
        "evidence_required_step_count": len(required_steps),
        "evidence_required_accepted_count": sum(1 for step in required_steps if step["accepted"]),
        "evidence_required_rejected_count": sum(1 for step in required_steps if not step["accepted"]),
        "proposal_audit_status": nested_dict(smoke_gate, "proposal_audit").get("status"),
        "proposal_audit_required_records": nested_dict(smoke_gate, "proposal_audit").get("required_record_count"),
        "proposal_audit_failed_required_records": nested_dict(smoke_gate, "proposal_audit").get("failed_required_record_count"),
        "optimizer_actual_total_tokens": usage.get("actual_total_tokens"),
        "seed_count": len(seed_rows),
    }


def build_seed_rows(rows: list[dict[str, Any]], best_baseline_condition: str) -> list[dict[str, Any]]:
    by_seed: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        seed = str(row.get("seed") or "")
        condition = str(row.get("condition") or "")
        if seed and condition:
            by_seed[seed][condition] = row
    output = []
    for seed in sorted(by_seed):
        executive = by_seed[seed].get("executive", {})
        baseline = by_seed[seed].get(best_baseline_condition, {})
        executive_score = maybe_float(executive.get("task_accuracy"))
        baseline_score = maybe_float(baseline.get("task_accuracy"))
        output.append(
            {
                "seed": seed,
                "executive_task_accuracy": executive_score,
                "baseline_condition": best_baseline_condition,
                "baseline_task_accuracy": baseline_score,
                "executive_minus_baseline": (
                    None if executive_score is None or baseline_score is None else executive_score - baseline_score
                ),
                "accepted_steps": executive.get("accepted_steps"),
                "total_steps": executive.get("total_steps"),
            }
        )
    return output


def build_contract_summary(steps: list[dict[str, Any]]) -> dict[str, Any]:
    all_contracts: dict[str, Counter] = defaultdict(Counter)
    evidence_contracts: dict[str, Counter] = defaultdict(Counter)
    for step in steps:
        target = evidence_contracts if step["proposal_audit"]["required"] else None
        for contract in step["proposal_audit"]["priority_contracts"]:
            all_contracts[contract]["priority_count"] += 1
            if target is not None:
                target[contract]["priority_count"] += 1
        for contract in step["targeted_contracts"]:
            all_contracts[contract]["targeted_count"] += 1
            if target is not None:
                target[contract]["targeted_count"] += 1
        outcomes_by_contract = {
            **dict(step.get("priority_contract_outcomes") or {}),
            **dict(step.get("targeted_contract_outcomes") or {}),
        }
        for contract, outcome in outcomes_by_contract.items():
            key = str(outcome.get("status") or "unknown")
            all_contracts[contract][key] += 1
            if not step["accepted"]:
                all_contracts[contract][f"rejected_{key}"] += 1
            else:
                all_contracts[contract][f"accepted_{key}"] += 1
            if target is not None:
                target[contract][key] += 1
                if not step["accepted"]:
                    target[contract][f"rejected_{key}"] += 1
                else:
                    target[contract][f"accepted_{key}"] += 1
    return {
        "all_steps": counters_to_dict(all_contracts),
        "evidence_required_steps": counters_to_dict(evidence_contracts),
        "top_regressed_contracts": top_contracts(all_contracts, "regressed"),
        "top_unchanged_failed_contracts": top_contracts(all_contracts, "unchanged_failed"),
        "top_evidence_required_regressed_contracts": top_contracts(evidence_contracts, "regressed"),
        "top_evidence_required_unchanged_failed_contracts": top_contracts(evidence_contracts, "unchanged_failed"),
    }


def build_failure_modes(steps: list[dict[str, Any]]) -> dict[str, Any]:
    labels = Counter(label for step in steps for label in step["failure_labels"])
    by_seed: dict[str, Counter] = defaultdict(Counter)
    for step in steps:
        by_seed[step["seed"]].update(step["failure_labels"])
    return {
        "counts": dict(sorted(labels.items())),
        "by_seed": {seed: dict(sorted(counter.items())) for seed, counter in sorted(by_seed.items())},
    }


def build_diagnosis(
    summary: dict[str, Any],
    contract_summary: dict[str, Any],
    failure_modes: dict[str, Any],
    seed_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    observations = []
    recommendations = []
    if summary["proposal_audit_status"] == "pass" and not summary["development_gate_passed"]:
        primary = "proposal_effectiveness_after_contract_evidence"
        observations.append(
            "Contract-evidence proposal audit passed, but the development gate failed; metadata compliance is no longer the main blocker."
        )
    elif summary["proposal_audit_status"] != "pass":
        primary = "proposal_audit_or_evidence_usage"
        observations.append("Proposal audit did not pass; fix evidence usage before evaluating proposal quality.")
    else:
        primary = "development_gate_ready"
        observations.append("Proposal audit and development gate passed.")

    evidence_rejected = int(summary.get("evidence_required_rejected_count") or 0)
    evidence_total = int(summary.get("evidence_required_step_count") or 0)
    if evidence_total:
        observations.append(
            f"{evidence_rejected}/{evidence_total} evidence-guided steps were rejected by validation."
        )
    top_regressed = contract_summary.get("top_evidence_required_regressed_contracts") or contract_summary.get("top_regressed_contracts")
    if top_regressed:
        observations.append(
            "Top regressed contracts after evidence-guided edits: "
            + ", ".join(f"{item['contract']}({item['count']})" for item in top_regressed[:3])
            + "."
        )
        recommendations.append(
            "Add an anti-regression guard for the top regressed contracts before accepting broader contract-targeted edits."
        )
    top_unchanged = (
        contract_summary.get("top_evidence_required_unchanged_failed_contracts")
        or contract_summary.get("top_unchanged_failed_contracts")
    )
    if top_unchanged:
        observations.append(
            "Top unchanged failed contracts: "
            + ", ".join(f"{item['contract']}({item['count']})" for item in top_unchanged[:3])
            + "."
        )
        recommendations.append(
            "Require the editor to state the expected contract-level direction and reject/penalize proposals that repeat targets without improving those contracts."
        )
    losing_seeds = [
        row["seed"]
        for row in seed_rows
        if row.get("executive_minus_baseline") is not None and row["executive_minus_baseline"] <= 0
    ]
    if losing_seeds:
        observations.append(f"Seeds not beating the cached baseline: {', '.join(losing_seeds)}.")
    if not recommendations:
        recommendations.append("Run the next smoke only after this report identifies a concrete policy change.")
    recommendations.append("Do not scale up until a smoke passes both proposal audit and development gate.")
    return {
        "primary_blocker": primary,
        "observations": observations,
        "recommended_next_actions": recommendations,
    }


def load_proposals_by_candidate(executive_dir: Path) -> dict[str, list[dict[str, Any]]]:
    output: dict[str, list[dict[str, Any]]] = defaultdict(list)
    path = executive_dir / "proposals.jsonl"
    if not path.exists():
        return output
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        record = json.loads(line)
        if not isinstance(record, dict):
            continue
        candidate = candidate_from_proposal_record(record)
        if candidate:
            copied = dict(record)
            copied["proposal_log_line"] = line_number
            output[candidate].append(copied)
    return output


def candidate_from_proposal_record(record: dict[str, Any]) -> str:
    controls = dict(record.get("optimizer_controls") or {})
    if controls.get("phase") != "reflection":
        return ""
    epoch = controls.get("epoch")
    batch_index = controls.get("batch_index")
    if epoch is None or batch_index is None:
        return ""
    return f"atomic-epoch-{epoch}-batch-{batch_index}"


def load_gates_by_candidate(executive_dir: Path) -> dict[str, dict[str, Any]]:
    output = {}
    pattern = re.compile(r"selection_(.+)_gate\.json$")
    for path in sorted(executive_dir.glob("selection_*_gate.json")):
        match = pattern.fullmatch(path.name)
        if not match:
            continue
        output[match.group(1)] = load_json_if_exists(path)
    return output


def summarize_edits(edits: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for edit in edits:
        content = str(edit.get("content") or "").strip().replace("\n", " ")
        output.append(
            {
                "operation": edit.get("operation"),
                "target": edit.get("target"),
                "priority": edit.get("priority"),
                "content_preview": content[:180],
            }
        )
    return output


def render_markdown(report: dict[str, Any]) -> str:
    summary = report["summary"]
    diagnosis = report["diagnosis"]
    lines = [
        "# SkillOpt Failure Delta Report",
        "",
        f"Run: `{report['run_dir']}`",
        "",
        "## Verdict",
        "",
        f"- primary blocker: `{diagnosis['primary_blocker']}`",
        f"- smoke status: `{summary.get('smoke_status')}`",
        f"- proposal audit: `{summary.get('proposal_audit_status')}`",
        f"- development gate passed: `{summary.get('development_gate_passed')}`",
        f"- executive mean: `{summary.get('executive_mean')}`",
        f"- mean delta: `{summary.get('mean_delta')}`",
        f"- seed wins: `{summary.get('seed_wins_vs_best_baseline')}/{summary.get('required_seed_wins')}`",
        f"- accepted steps: `{summary.get('accepted_step_count')}/{summary.get('step_count')}`",
        f"- evidence-guided rejected steps: `{summary.get('evidence_required_rejected_count')}/{summary.get('evidence_required_step_count')}`",
        "",
        "## Observations",
        "",
        *[f"- {item}" for item in diagnosis["observations"]],
        "",
        "## Seed Outcomes",
        "",
        "| seed | executive | baseline | delta | accepted/total |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in report["seed_rows"]:
        lines.append(
            f"| {row['seed']} | {fmt(row.get('executive_task_accuracy'))} | "
            f"{fmt(row.get('baseline_task_accuracy'))} | {fmt(row.get('executive_minus_baseline'))} | "
            f"{row.get('accepted_steps')}/{row.get('total_steps')} |"
        )
    lines.extend(
        [
            "",
            "## Top Contract Blockers",
            "",
            "| category | contracts |",
            "| --- | --- |",
            f"| evidence regressions | {format_contract_list(report['contract_summary'].get('top_evidence_required_regressed_contracts'))} |",
            f"| evidence unchanged failures | {format_contract_list(report['contract_summary'].get('top_evidence_required_unchanged_failed_contracts'))} |",
            "",
            "## Next Actions",
            "",
            *[f"- {item}" for item in diagnosis["recommended_next_actions"]],
            "",
        ]
    )
    return "\n".join(lines)


def render_compact_summary(report: dict[str, Any]) -> str:
    summary = dict(report.get("summary") or {})
    diagnosis = dict(report.get("diagnosis") or {})
    contracts = dict(report.get("contract_summary") or {})
    parts = [
        "failure_delta",
        f"blocker={compact_token(diagnosis.get('primary_blocker'))}",
        f"smoke={summary.get('smoke_status')}",
        f"audit={summary.get('proposal_audit_status')}",
        f"gate_passed={summary.get('development_gate_passed')}",
        f"mean={summary.get('executive_mean')}",
        f"delta={summary.get('mean_delta')}",
        f"wins={summary.get('seed_wins_vs_best_baseline')}/{summary.get('required_seed_wins')}",
        f"accepted={summary.get('accepted_step_count')}/{summary.get('step_count')}",
        f"evidence_rejected={summary.get('evidence_required_rejected_count')}/{summary.get('evidence_required_step_count')}",
        f"tokens={summary.get('optimizer_actual_total_tokens')}",
        f"top_regressed={format_compact_contracts(contracts.get('top_evidence_required_regressed_contracts'))}",
        f"top_unchanged={format_compact_contracts(contracts.get('top_evidence_required_unchanged_failed_contracts'))}",
    ]
    return " ".join(parts)


def compact_token(value: Any) -> str:
    text = str(value or "").strip().replace("\n", " ")
    return json.dumps(text, ensure_ascii=True)


def format_compact_contracts(items: list[dict[str, Any]] | None) -> str:
    if not items:
        return "none"
    return ",".join(f"{item['contract']}:{item['count']}" for item in items[:3])


def format_contract_list(items: list[dict[str, Any]] | None) -> str:
    if not items:
        return "none"
    return ", ".join(f"{item['contract']} ({item['count']})" for item in items[:5])


def fmt(value: Any) -> str:
    if value is None:
        return ""
    return f"{float(value):.4f}"


def counters_to_dict(counters: dict[str, Counter]) -> dict[str, dict[str, int]]:
    return {contract: dict(sorted(counter.items())) for contract, counter in sorted(counters.items())}


def top_contracts(counters: dict[str, Counter], key: str, *, limit: int = 5) -> list[dict[str, Any]]:
    items = [
        {"contract": contract, "count": int(counter.get(key) or 0)}
        for contract, counter in counters.items()
        if int(counter.get(key) or 0) > 0
    ]
    return sorted(items, key=lambda item: (-item["count"], item["contract"]))[:limit]


def normalized_string_list(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value] if value else []
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if str(item)]


def nested_dict(payload: dict[str, Any], *keys: str) -> dict[str, Any]:
    current: Any = payload
    for key in keys:
        if not isinstance(current, dict):
            return {}
        current = current.get(key)
    return current if isinstance(current, dict) else {}


def nested_list(payload: dict[str, Any], *keys: str) -> list[Any]:
    current: Any = payload
    for key in keys:
        if not isinstance(current, dict):
            return []
        current = current.get(key)
    return current if isinstance(current, list) else []


def maybe_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def load_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--markdown-out", type=Path)
    parser.add_argument("--quiet", action="store_true", help="Print a one-line summary instead of the full JSON report")
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
