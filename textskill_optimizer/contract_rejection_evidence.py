"""Compact rejected-buffer contract evidence and proposal targeting audit."""

from __future__ import annotations

from typing import Any

from .models import EditProposal


def build_contract_rejection_evidence(
    rejected_buffer: Any,
    *,
    recent_limit: int = 5,
    contract_limit: int = 8,
) -> dict[str, Any]:
    if not isinstance(rejected_buffer, list):
        return {"available": False, "priority_contracts": [], "recent_rejections": []}
    recent_rejections = []
    contract_counts: dict[str, dict[str, Any]] = {}
    for item in rejected_buffer[-recent_limit:]:
        if not isinstance(item, dict):
            continue
        gate = nested_dict(item, "metadata", "validation_gate")
        evidence = gate.get("contract_evidence") if isinstance(gate, dict) else None
        if not isinstance(evidence, dict):
            continue
        blockers = []
        for kind, key in (
            ("negative_delta", "top_negative_contracts"),
            ("no_improvement", "top_no_improvement_contracts"),
        ):
            for contract_item in evidence.get(key) or []:
                if not isinstance(contract_item, dict):
                    continue
                contract = str(contract_item.get("contract") or "").strip()
                if not contract:
                    continue
                compact = compact_contract_item(contract_item, kind)
                blockers.append(compact)
                bucket = contract_counts.setdefault(
                    contract,
                    {
                        "contract": contract,
                        "negative_delta_count": 0,
                        "no_improvement_count": 0,
                        "max_current_accuracy": 0.0,
                        "worst_delta": 0.0,
                    },
                )
                if kind == "negative_delta":
                    bucket["negative_delta_count"] += 1
                else:
                    bucket["no_improvement_count"] += 1
                bucket["max_current_accuracy"] = max(
                    float(bucket["max_current_accuracy"]),
                    float(compact["current_accuracy"]),
                )
                bucket["worst_delta"] = min(float(bucket["worst_delta"]), float(compact["delta"]))
        if blockers:
            recent_rejections.append(
                {
                    "candidate": item.get("candidate"),
                    "reason": item.get("reason"),
                    "validation_score": item.get("validation_score"),
                    "candidate_mean": gate.get("candidate_mean") if isinstance(gate, dict) else None,
                    "current_mean": gate.get("current_mean") if isinstance(gate, dict) else None,
                    "blocking_contracts": blockers[:contract_limit],
                }
            )
    priority_contracts = sorted(
        contract_counts.values(),
        key=lambda item: (
            -int(item["negative_delta_count"]),
            -int(item["no_improvement_count"]),
            float(item["worst_delta"]),
            str(item["contract"]),
        ),
    )[:contract_limit]
    proposal_policy = build_proposal_policy(priority_contracts)
    return {
        "available": bool(recent_rejections),
        "priority_contracts": priority_contracts,
        "recent_rejections": recent_rejections,
        "proposal_policy": proposal_policy,
    }


def audit_proposal_targeting(
    proposals: list[EditProposal],
    contract_rejection_evidence: dict[str, Any],
) -> dict[str, Any]:
    priority_contracts = [
        str(item.get("contract"))
        for item in contract_rejection_evidence.get("priority_contracts", [])
        if isinstance(item, dict) and item.get("contract")
    ]
    priority = set(priority_contracts)
    required = bool(contract_rejection_evidence.get("available") and priority)
    policy = contract_rejection_evidence.get("proposal_policy")
    policy = policy if isinstance(policy, dict) else {}
    anti_regression_contracts = policy_contract_set(policy, "anti_regression_contracts")
    protected_priority_contracts = policy_contract_set(policy, "protected_priority_contracts")
    cooldown_contracts = policy_contract_set(policy, "cooldown_contracts")
    single_contract = policy.get("single_contract_targeting")
    single_contract = single_contract if isinstance(single_contract, dict) else {}
    max_targets = int(single_contract.get("max_targeted_priority_contracts") or 0)
    multi_requires = policy_contract_set(single_contract, "multi_contract_requires_protected_contracts")
    proposal_records = []
    missing_count = 0
    for proposal in proposals:
        metadata = proposal.metadata if isinstance(proposal.metadata, dict) else {}
        targeted = normalized_string_list(metadata.get("targeted_contracts"))
        targeted_priority = sorted(priority.intersection(targeted))
        protected = normalized_string_list(metadata.get("protected_contracts"))
        protected_set = set(protected)
        missing_protected = sorted(anti_regression_contracts - protected_set)
        missing_priority_protected = sorted(protected_priority_contracts - protected_set)
        missing_multi_protected = sorted(multi_requires - protected_set)
        targeted_cooldown = sorted(cooldown_contracts.intersection(targeted))
        issues = []
        if required and not targeted:
            issues.append("missing_targeted_contracts")
        elif required and not targeted_priority:
            issues.append("no_priority_contract_targeted")
        evidence_source = str(metadata.get("evidence_source") or "")
        if required and evidence_source != "contract_rejection_evidence":
            issues.append("evidence_source_not_contract_rejection_evidence")
        if required and not str(metadata.get("expected_behavior_change") or "").strip():
            issues.append("missing_expected_behavior_change")
        if required and anti_regression_contracts and missing_protected:
            issues.append("missing_anti_regression_guard")
        if required and protected_priority_contracts and missing_priority_protected:
            issues.append("missing_priority_contract_protection")
        if required and max_targets > 0 and len(targeted_priority) > max_targets and missing_multi_protected:
            issues.append("multi_contract_without_required_protection")
        if required and targeted_cooldown and not str(metadata.get("cooldown_override") or "").strip():
            issues.append("missing_cooldown_override")
        if issues:
            missing_count += 1
        proposal_records.append(
            {
                "name": proposal.name,
                "targeted_contracts": targeted,
                "targeted_priority_contracts": targeted_priority,
                "protected_contracts": protected,
                "missing_protected_contracts": sorted(
                    set(missing_protected).union(missing_priority_protected)
                )
                if required
                else [],
                "missing_priority_protected_contracts": missing_priority_protected if required else [],
                "missing_multi_contract_protections": missing_multi_protected if required else [],
                "targeted_cooldown_contracts": targeted_cooldown if required else [],
                "evidence_source": evidence_source,
                "expected_behavior_change_present": bool(
                    str(metadata.get("expected_behavior_change") or "").strip()
                ),
                "cooldown_override_present": bool(str(metadata.get("cooldown_override") or "").strip()),
                "passes": not issues,
                "issues": issues,
            }
        )
    return {
        "required": required,
        "contract_rejection_evidence_available": bool(contract_rejection_evidence.get("available")),
        "priority_contracts": priority_contracts,
        "proposal_policy": policy,
        "proposal_count": len(proposals),
        "missing_targeted_contract_count": missing_count,
        "all_proposals_target_priority_contract": required and missing_count == 0,
        "proposals": proposal_records,
    }


def build_proposal_policy(priority_contracts: list[dict[str, Any]]) -> dict[str, Any]:
    anti_regression = [
        compact_policy_contract(item, "protect_against_regression")
        for item in priority_contracts
        if int(item.get("negative_delta_count") or 0) > 0
    ]
    cooldown = [
        compact_policy_contract(item, "cooldown_repeated_no_improvement")
        for item in priority_contracts
        if int(item.get("no_improvement_count") or 0) >= 2
    ]
    multi_contract_requires_protection = [
        compact_policy_contract(item, "protect_previously_passing_priority_contract")
        for item in priority_contracts
        if float(item.get("max_current_accuracy") or 0.0) > 0.0
    ]
    protected_priority_contracts = [
        compact_policy_contract(item, "protect_currently_passing_priority_contract")
        for item in priority_contracts
        if float(item.get("max_current_accuracy") or 0.0) > 0.0
    ]
    return {
        "anti_regression_contracts": anti_regression,
        "protected_priority_contracts": protected_priority_contracts,
        "cooldown_contracts": cooldown,
        "single_contract_targeting": {
            "max_targeted_priority_contracts": 1,
            "multi_contract_requires_protected_contracts": multi_contract_requires_protection,
        },
        "rules": [
            "Any proposal after contract evidence must preserve anti_regression_contracts.",
            "Any proposal after contract evidence must protect protected_priority_contracts.",
            "A proposal that targets cooldown_contracts must include a new, evidence-backed cooldown_override.",
            "Target one priority contract per proposal unless all previously passing priority contracts are protected.",
        ],
    }


def compact_policy_contract(payload: dict[str, Any], reason: str) -> dict[str, Any]:
    return {
        "contract": str(payload.get("contract") or ""),
        "reason": reason,
        "negative_delta_count": int(payload.get("negative_delta_count") or 0),
        "no_improvement_count": int(payload.get("no_improvement_count") or 0),
        "max_current_accuracy": float(payload.get("max_current_accuracy") or 0.0),
        "worst_delta": float(payload.get("worst_delta") or 0.0),
    }


def policy_contract_set(policy: dict[str, Any], key: str) -> set[str]:
    values = policy.get(key)
    if not isinstance(values, list):
        return set()
    return {
        str(item.get("contract"))
        for item in values
        if isinstance(item, dict) and item.get("contract")
    }


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


def compact_contract_item(payload: dict[str, Any], kind: str) -> dict[str, Any]:
    return {
        "contract": str(payload.get("contract") or ""),
        "kind": kind,
        "current_accuracy": float(payload.get("current_accuracy") or 0.0),
        "candidate_accuracy": float(payload.get("candidate_accuracy") or 0.0),
        "delta": float(payload.get("delta") or 0.0),
    }
