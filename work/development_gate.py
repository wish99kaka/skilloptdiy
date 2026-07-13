"""Development-gate verdicts for coding-hidden-v2 experiments."""

from __future__ import annotations

from typing import Any


DEVELOPMENT_GATE_SCHEMA_VERSION = 2
DEFAULT_DEVELOPMENT_GATE_CRITERIA = {
    "best_baseline_margin": 0.05,
    "min_seed_wins": 2,
    "contract_macro_margin": 0.0,
    "critical_contracts": [],
    "critical_contract_regression_epsilon": 0.0,
}


def normalize_development_gate_criteria(criteria: dict[str, Any] | None) -> dict[str, Any]:
    merged = {**DEFAULT_DEVELOPMENT_GATE_CRITERIA, **dict(criteria or {})}
    critical_contracts = normalized_string_list(
        merged.get("critical_contracts") or merged.get("priority_contracts")
    )
    policy = merged.get("critical_contract_regression_policy")
    if isinstance(policy, dict):
        critical_contracts = normalized_string_list(policy.get("contracts")) or critical_contracts
        epsilon = float(policy.get("epsilon") or merged["critical_contract_regression_epsilon"])
        policy_mode = str(policy.get("mode") or "configured_contracts")
    else:
        epsilon = float(merged["critical_contract_regression_epsilon"])
        policy_mode = str(policy or "configured_contracts")
    return {
        "best_baseline_margin": float(merged["best_baseline_margin"]),
        "min_seed_wins": int(merged["min_seed_wins"]),
        "contract_macro_margin": float(merged["contract_macro_margin"]),
        "critical_contracts": critical_contracts,
        "critical_contract_regression_policy": {
            "mode": policy_mode,
            "contracts": critical_contracts,
            "epsilon": epsilon,
        },
    }


def build_development_gate(
    rows: list[dict[str, Any]],
    aggregate: dict[str, Any],
    criteria: dict[str, Any] | None = None,
) -> dict[str, Any]:
    normalized = normalize_development_gate_criteria(criteria)
    executive = aggregate.get("executive") if isinstance(aggregate.get("executive"), dict) else None
    baseline_conditions = [
        str(name)
        for name, payload in aggregate.items()
        if name != "executive" and isinstance(payload, dict)
    ]
    best_baseline_condition = best_condition_by_mean(aggregate, baseline_conditions)
    best_baseline = (
        aggregate.get(best_baseline_condition)
        if best_baseline_condition and isinstance(aggregate.get(best_baseline_condition), dict)
        else None
    )
    best_baseline_score = float((best_baseline or {}).get("task_accuracy_mean") or 0.0)
    executive_score = float((executive or {}).get("task_accuracy_mean") or 0.0)
    score_delta = executive_score - best_baseline_score
    best_baseline_contract_macro = metric_with_fallback(
        best_baseline,
        "contract_macro_mean",
        best_baseline_score,
    )
    executive_contract_macro = metric_with_fallback(
        executive,
        "contract_macro_mean",
        executive_score,
    )
    contract_macro_delta = executive_contract_macro - best_baseline_contract_macro
    required_contract_macro_delta = float(normalized["contract_macro_margin"])
    critical_regressions = critical_contract_regressions(
        best_baseline,
        executive,
        normalized["critical_contracts"],
        epsilon=float(normalized["critical_contract_regression_policy"]["epsilon"]),
    )
    seed_wins = count_seed_wins(rows, best_baseline_condition)
    required_delta = float(normalized["best_baseline_margin"])
    required_seed_wins = int(normalized["min_seed_wins"])

    blocked_reasons: list[str] = []
    if executive is None:
        blocked_reasons.append("executive aggregate is missing")
    if best_baseline_condition is None:
        blocked_reasons.append("no baseline condition is available")
    if score_delta < required_delta:
        blocked_reasons.append(
            f"executive mean delta {score_delta:.4f} is below required margin {required_delta:.4f}"
        )
    if contract_macro_delta < required_contract_macro_delta:
        blocked_reasons.append(
            "contract macro delta "
            f"{contract_macro_delta:.4f} is below required margin {required_contract_macro_delta:.4f}"
        )
    if critical_regressions:
        contracts = ", ".join(item["contract"] for item in critical_regressions)
        blocked_reasons.append(f"critical contract regressions: {contracts}")
    if seed_wins < required_seed_wins:
        blocked_reasons.append(f"executive won {seed_wins} seeds; required {required_seed_wins}")
    passed = not blocked_reasons
    return {
        "schema_version": DEVELOPMENT_GATE_SCHEMA_VERSION,
        "score_metric": "task_accuracy_mean",
        "criteria": normalized,
        "best_baseline_condition": best_baseline_condition,
        "best_baseline_score": best_baseline_score,
        "best_baseline_mean": best_baseline_score,
        "executive_score": executive_score,
        "executive_mean": executive_score,
        "score_delta": score_delta,
        "mean_delta": score_delta,
        "required_delta": required_delta,
        "contract_metric": "contract_macro_mean",
        "best_baseline_contract_macro_mean": best_baseline_contract_macro,
        "executive_contract_macro_mean": executive_contract_macro,
        "contract_macro_delta": contract_macro_delta,
        "required_contract_macro_delta": required_contract_macro_delta,
        "critical_contract_regressions": critical_regressions,
        "seed_wins_vs_best_baseline": seed_wins,
        "required_seed_wins": required_seed_wins,
        "passed": passed,
        "criteria_met": passed,
        "locked_test_recommended": passed,
        "blocked_reasons": blocked_reasons,
        "blocked_reason": "; ".join(blocked_reasons),
    }


def best_condition_by_mean(aggregate: dict[str, Any], conditions: list[str]) -> str | None:
    best_condition = None
    best_score = 0.0
    for condition in conditions:
        payload = aggregate.get(condition) if isinstance(aggregate.get(condition), dict) else {}
        score = float(payload.get("task_accuracy_mean") or 0.0)
        if best_condition is None or score > best_score:
            best_condition = condition
            best_score = score
    return best_condition


def count_seed_wins(rows: list[dict[str, Any]], baseline_condition: str | None) -> int:
    if not baseline_condition:
        return 0
    by_seed: dict[str, dict[str, float]] = {}
    for row in rows:
        seed = str(row.get("seed") or "")
        condition = str(row.get("condition") or "")
        if not seed or not condition:
            continue
        by_seed.setdefault(seed, {})[condition] = float(row.get("task_accuracy") or 0.0)
    wins = 0
    for scores in by_seed.values():
        if scores.get("executive", 0.0) > scores.get(baseline_condition, 0.0):
            wins += 1
    return wins


def critical_contract_regressions(
    baseline: dict[str, Any] | None,
    executive: dict[str, Any] | None,
    critical_contracts: list[str],
    *,
    epsilon: float = 0.0,
) -> list[dict[str, Any]]:
    if not baseline or not executive or not critical_contracts:
        return []
    baseline_breakdown = baseline.get("contract_breakdown")
    executive_breakdown = executive.get("contract_breakdown")
    if not isinstance(baseline_breakdown, dict) or not isinstance(executive_breakdown, dict):
        return []
    regressions = []
    for contract in critical_contracts:
        base_payload = baseline_breakdown.get(contract)
        exec_payload = executive_breakdown.get(contract)
        if not isinstance(base_payload, dict) or not isinstance(exec_payload, dict):
            continue
        baseline_accuracy = float(base_payload.get("accuracy") or 0.0)
        executive_accuracy = float(exec_payload.get("accuracy") or 0.0)
        delta = executive_accuracy - baseline_accuracy
        if delta < -epsilon:
            regressions.append(
                {
                    "contract": contract,
                    "baseline_accuracy": baseline_accuracy,
                    "executive_accuracy": executive_accuracy,
                    "delta": delta,
                }
            )
    return regressions


def metric_with_fallback(payload: dict[str, Any] | None, key: str, fallback: float) -> float:
    if not isinstance(payload, dict) or key not in payload:
        return fallback
    return float(payload.get(key) or 0.0)


def normalized_string_list(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value] if value else []
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if str(item)]
