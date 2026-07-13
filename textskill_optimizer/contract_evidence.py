"""Contract-level scoring evidence for validation gates."""

from __future__ import annotations

from typing import Any

from .models import EvaluationReport, TaskResult


UNKNOWN_CONTRACT = "unknown_contract"


def contract_breakdown(report: EvaluationReport | None) -> dict[str, dict[str, float | int]]:
    """Return pass/total/accuracy grouped by task contract tags."""
    if report is None:
        return {}
    totals: dict[str, dict[str, int]] = {}
    for result in report.results:
        for tag in contract_tags(result):
            bucket = totals.setdefault(tag, {"passed": 0, "total": 0})
            bucket["total"] += 1
            if result.score.success:
                bucket["passed"] += 1
    return {
        tag: {
            "passed": counts["passed"],
            "total": counts["total"],
            "accuracy": counts["passed"] / counts["total"] if counts["total"] else 0.0,
        }
        for tag, counts in sorted(totals.items())
    }


def contract_tags(result: TaskResult) -> list[str]:
    raw = result.task.metadata.get("contract_tags") or [UNKNOWN_CONTRACT]
    tags = [str(item) for item in raw if str(item)] if isinstance(raw, list) else [str(raw)]
    return list(dict.fromkeys(tags or [UNKNOWN_CONTRACT]))


def contract_delta_evidence(
    current_report: EvaluationReport | None,
    candidate_report: EvaluationReport,
    *,
    limit: int = 5,
) -> dict[str, Any]:
    current = contract_breakdown(current_report)
    candidate = contract_breakdown(candidate_report)
    deltas = contract_deltas(current, candidate)
    negative = [
        payload
        for payload in sorted(deltas.values(), key=lambda item: (item["delta"], item["contract"]))
        if payload["delta"] < 0
    ][:limit]
    no_improvement = [
        payload
        for payload in sorted(
            deltas.values(),
            key=lambda item: (item["candidate_accuracy"], item["delta"], item["contract"]),
        )
        if payload["delta"] == 0 and payload["candidate_accuracy"] < 1.0
    ][:limit]
    improved = [payload for payload in deltas.values() if payload["delta"] > 0]
    return {
        "current_contract_breakdown": current,
        "candidate_contract_breakdown": candidate,
        "contract_deltas": deltas,
        "top_negative_contracts": negative,
        "top_no_improvement_contracts": no_improvement,
        "summary": {
            "negative_contract_count": len([item for item in deltas.values() if item["delta"] < 0]),
            "no_improvement_contract_count": len(
                [
                    item
                    for item in deltas.values()
                    if item["delta"] == 0 and item["candidate_accuracy"] < 1.0
                ]
            ),
            "improved_contract_count": len(improved),
        },
    }


def contract_deltas(
    current: dict[str, dict[str, float | int]],
    candidate: dict[str, dict[str, float | int]],
) -> dict[str, dict[str, float | int | str]]:
    output: dict[str, dict[str, float | int | str]] = {}
    for contract in sorted(set(current) | set(candidate)):
        current_payload = current.get(contract, {})
        candidate_payload = candidate.get(contract, {})
        current_accuracy = float(current_payload.get("accuracy") or 0.0)
        candidate_accuracy = float(candidate_payload.get("accuracy") or 0.0)
        output[contract] = {
            "contract": contract,
            "current_accuracy": current_accuracy,
            "candidate_accuracy": candidate_accuracy,
            "delta": candidate_accuracy - current_accuracy,
            "current_passed": int(current_payload.get("passed") or 0),
            "current_total": int(current_payload.get("total") or 0),
            "candidate_passed": int(candidate_payload.get("passed") or 0),
            "candidate_total": int(candidate_payload.get("total") or 0),
        }
    return output
