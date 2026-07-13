"""OpenAI-compatible external model editor for TextSkill Optimizer.

This script targets providers that expose a Chat Completions-compatible API,
including many private gateways, LiteLLM, vLLM, and local model servers.

Required environment:
  EXTERNAL_LLM_BASE_URL   e.g. http://localhost:4000/v1
                          or a full .../chat/completions endpoint
  EXTERNAL_LLM_MODEL      e.g. qwen2.5-coder-32b-instruct

Optional environment:
  EXTERNAL_LLM_API_KEY    defaults to "not-needed"
  EXTERNAL_LLM_TIMEOUT    defaults to 120
  EXTERNAL_LLM_JSON_MODE  defaults to 1. Set 0 if your endpoint rejects response_format.
  EXTERNAL_LLM_TEMPERATURE defaults to 0.2
  EXTERNAL_LLM_DRY_RUN    set 1 to validate config and print the request without calling the API
"""

from __future__ import annotations

import json
import os
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from editor_io import load_optimizer_payload_from_stdin
from textskill_optimizer.contract_rejection_evidence import build_contract_rejection_evidence
from textskill_optimizer.usage_ledger import (
    append_usage_event,
    estimate_tokens_from_chars,
    extract_chat_usage,
    usage_context_from_env,
)


SYSTEM_PROMPT = """You are editing a reusable agent skill document from scored trajectory evidence.

Return only JSON with this shape:
{
  "proposals": [
    {
      "name": "short-kebab-case-name",
      "rationale": "why this evidence supports the update",
      "metadata": {
        "targeted_contracts": ["contract labels targeted by this proposal"],
        "protected_contracts": ["priority contracts this proposal must preserve"],
        "evidence_source": "contract_rejection_evidence | trajectory_comparison | meta_skill",
        "expected_behavior_change": "specific validation behavior this edit should improve",
        "cooldown_override": "new evidence-backed mechanism for retargeting a recently failed or repeatedly non-improving contract; empty otherwise"
      },
      "edits": [
        {
          "operation": "add | delete | replace",
          "target": "exact text from the current skill, or __end__ for append",
          "content": "text to add or replace with; empty only for delete",
          "rationale": "evidence for this atomic edit",
          "priority": 0.0
        }
      ]
    }
  ]
}

First principles:
- The skill document is the only thing you may change.
- The runner, scorer, tests, benchmarks, expected answers, and task metadata are fixed.
- A proposal is useful only if it can improve validation tasks, not just observed training tasks.

Editing rules:
- Return localized add, delete, or replace edits, never a full rewritten skill document.
- Quote replace/delete targets exactly from current_skill_text. Use `__end__` only for an append.
- Follow optimizer_controls.atomic_edit_budget. Rank edits with priority, but let the optimizer perform final merging and clipping.
- Preserve useful existing instructions.
- Use meta_skill as optimizer-side guidance about how to write durable skill rules.
- Use rejected_buffer as negative feedback. Do not repeat rejected directions unless new evidence clearly fixes the prior failure.
- Use contract_rejection_evidence as the primary rejection signal when available. Target at least one listed priority contract or return no proposals.
- When contract_rejection_evidence is available, every proposal must include metadata.targeted_contracts, metadata.evidence_source, and metadata.expected_behavior_change.
- In that case metadata.evidence_source must be exactly the literal string "contract_rejection_evidence"; do not use synonyms such as trajectory_comparison, validation_gate, rejected_buffer, or meta_skill.
- The target agent reads only edit.content, not proposal metadata. Convert each targeted and protected contract into executable rule text in edit.content.
- When contract_rejection_evidence.proposal_policy.anti_regression_contracts is non-empty, metadata.protected_contracts must include every listed contract and the edit must preserve those behaviors while improving the target.
- When contract_rejection_evidence.proposal_policy.protected_priority_contracts is non-empty, metadata.protected_contracts must include every listed contract, even if the proposal targets only one contract.
- For each metadata.protected_contracts item, include a concrete preserve/check mechanism in edit.content; metadata alone does not protect behavior.
- For input validation, name the invalid-input guard to preserve or add. Do not replace invalid-input rejection with a silent fallback output.
- Do not write generic phrases such as "preserve invalid input rejection rules"; spell out concrete guards supported by the contract or verifier feedback, such as raising for negative totals, negative weights, malformed input, or other named invalid classes.
- For largest-remainder proportional allocation evidence, do not merely name the method. In edit.content state the executable branches supported by evidence: raise ValueError for negative totals or weights, return an all-zero output for zero-sum weights rather than raising or distributing units, compute quotas and floors, distribute remaining units by largest fractional remainders, and break ties by original index.
- When targeting any contract_rejection_evidence.proposal_policy.cooldown_contracts or any recently failed target, include metadata.cooldown_override explaining the new evidence-backed mechanism that makes this different from the rejected direction; otherwise do not retarget it.
- When contract_rejection_evidence.proposal_policy.single_contract_targeting is present, target exactly one priority contract per proposal unless metadata.protected_contracts includes every listed multi_contract_requires_protected_contracts item.
- Treat contract names as optimizer-side failure labels. Do not copy them into the deployed skill as benchmark answers; translate them into general executable checks.
- Every new rule must be justified by evidence in the supplied trajectories, verifier feedback, or longitudinal optimizer state. Do not inject domain knowledge merely because it sounds prudent.
- Compare failures with successes before editing. Preserve procedures associated with successful trajectories and target recurring causes rather than isolated symptoms.
- Prefer the smallest procedural change that can explain multiple observations. Treat a one-example diagnosis as a hypothesis, not a universal rule.
- Keep skill_text concise: no more than 120 words, no nested lists, and no explanatory prose.
- Do not mention specific task IDs, fixture names, exact hidden answers, or benchmark internals in the skill.
- Do not turn observed training failures into narrow task-specific examples; abstract them into reusable checks.
- Convert evidence into actions the target agent can execute: what to inspect, what invariant to preserve, and how to verify the repair. Do not enumerate unrelated robustness cases.
- Do not repeat generic full-contract-audit advice unless contract_rejection_evidence shows that a more specific contract-targeted rule is unsupported.
- Do not propose rules that only ask the agent to run public tests, snapshot passing behavior, verify every requirement, or audit all contracts.
- If the evidence does not support a generalizable improvement, return {"proposals": []}.
- Do not instruct the agent to edit tests, weaken tests, skip tests, change scorers, or fake outputs.
- Prefer one strong proposal over many weak proposals.
"""


SLOW_META_SYSTEM_PROMPT = """You maintain optimizer-only longitudinal state for a reusable agent skill.

Return only JSON:
{
  "meta_skill": "concise optimizer guidance for future reflection and ranking",
  "slow_update": "concise durable guidance proposed for the protected slow-update field",
  "rationale": "what changed across epochs and why"
}

Use only the paired previous/current trajectories, contract_rejection_evidence, rejection history, and existing meta guidance. Prioritize contract deltas over generic repeated advice, but do not copy contract labels into target-side benchmark answers. Summarize improvements, regressions, persistent failures, and stable successes. Do not mention task IDs, fixture names, hidden answers, or preload domain rules. Keep both fields compact. Return an empty slow_update when no durable target-side guidance is supported.
"""


ONE_SHOT_SYSTEM_PROMPT = """Write one compact reusable coding-agent skill from visible development contracts and public tests only.

Return only JSON: {"skill_text": "complete skill markdown", "rationale": "why these procedures generalize"}.

Do not mention task IDs, fixture names, hidden tests, exact expected outputs, or invent domain rules unsupported by the supplied public context. Focus on reusable investigation, implementation, and verification procedures. Keep the skill under 160 words.
"""


def main() -> int:
    try:
        optimizer_payload = load_optimizer_payload_from_stdin()
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    if optimizer_payload.get("operation") == "capabilities":
        print(
            json.dumps(
                {
                    "capabilities": [
                        "atomic_edits",
                        "full_skill_replacement",
                    ]
                }
            )
        )
        return 0
    try:
        request_payload, url, api_key, timeout = build_chat_request_from_env(optimizer_payload)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    if os.environ.get("EXTERNAL_LLM_DRY_RUN") == "1":
        print(
            json.dumps(
                {
                    "url": url,
                    "model": request_payload.get("model"),
                    "uses_json_mode": "response_format" in request_payload,
                    "request_payload": request_payload,
                }
            )
        )
        return 0

    try:
        response_payload = call_chat_completions(url, api_key, request_payload, timeout)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        print(f"External model API error {exc.code}: {body}", file=sys.stderr)
        return 1
    except urllib.error.URLError as exc:
        print(f"External model connection error: {exc.reason}", file=sys.stderr)
        return 1

    content = extract_chat_message_content(response_payload)
    record_model_api_usage(
        request_payload,
        response_payload,
        operation=str(optimizer_payload.get("operation") or "reflect"),
        content=content,
        url=url,
        duration_seconds=response_payload.pop("_duration_seconds", 0.0),
    )
    if not content:
        print("External model response did not contain assistant content", file=sys.stderr)
        return 1

    try:
        proposals = parse_model_json(content)
    except json.JSONDecodeError as exc:
        print(f"External model output was not JSON: {exc}", file=sys.stderr)
        print(content, file=sys.stderr)
        if str(optimizer_payload.get("operation") or "reflect") == "reflect":
            print(json.dumps({"proposals": []}))
            return 0
        return 1

    proposals = enforce_contract_evidence_source(proposals, optimizer_payload)
    print(json.dumps(proposals))
    return 0


def build_chat_request_from_env(
    optimizer_payload: dict[str, Any],
) -> tuple[dict[str, Any], str, str, float]:
    base_url = os.environ.get("EXTERNAL_LLM_BASE_URL", "").rstrip("/")
    model = os.environ.get("EXTERNAL_LLM_MODEL", "")
    if not base_url:
        raise ValueError("EXTERNAL_LLM_BASE_URL is required")
    if not model:
        raise ValueError("EXTERNAL_LLM_MODEL is required")

    api_key = os.environ.get("EXTERNAL_LLM_API_KEY", "not-needed")
    timeout = float(os.environ.get("EXTERNAL_LLM_TIMEOUT", "120"))
    url = normalize_chat_completions_url(base_url)
    return build_chat_request_payload(optimizer_payload, model=model), url, api_key, timeout


def normalize_chat_completions_url(base_url: str) -> str:
    cleaned = base_url.rstrip("/")
    if cleaned.endswith("/chat/completions"):
        return cleaned
    return cleaned + "/chat/completions"


def build_chat_request_payload(
    optimizer_payload: dict[str, Any],
    *,
    model: str,
) -> dict[str, Any]:
    operation = str(optimizer_payload.get("operation") or "reflect")
    if operation == "slow_meta_update":
        user_payload = {
            "operation": operation,
            "epoch": optimizer_payload.get("epoch"),
            "current_skill_text": optimizer_payload.get("current_skill_text"),
            "meta_skill": optimizer_payload.get("meta_skill", ""),
            "comparison": optimizer_payload.get("comparison", {}),
            "contract_rejection_evidence": build_contract_rejection_evidence(
                optimizer_payload.get("rejected_buffer", [])
            ),
            "rejected_buffer": optimizer_payload.get("rejected_buffer", []),
            "optimizer_controls": optimizer_payload.get("optimizer_controls", {}),
        }
        system_prompt = SLOW_META_SYSTEM_PROMPT
    elif operation == "one_shot_skill":
        user_payload = {
            "operation": operation,
            "seed_label": optimizer_payload.get("seed_label"),
            "development_context": optimizer_payload.get("development_context", []),
        }
        system_prompt = ONE_SHOT_SYSTEM_PROMPT
    else:
        user_payload = build_reflection_user_payload(optimizer_payload)
        system_prompt = SYSTEM_PROMPT
    request: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(user_payload, indent=2)},
        ],
        "temperature": float(os.environ.get("EXTERNAL_LLM_TEMPERATURE", "0.2")),
    }
    if os.environ.get("EXTERNAL_LLM_JSON_MODE", "1") != "0":
        request["response_format"] = {"type": "json_object"}
    return request


def build_reflection_user_payload(optimizer_payload: dict[str, Any]) -> dict[str, Any]:
    user_payload = {
        "operation": "reflect",
        "epoch": optimizer_payload.get("epoch"),
        "current_skill_text": optimizer_payload.get("skill_text"),
        "meta_skill": optimizer_payload.get("meta_skill", ""),
        "optimizer_controls": optimizer_payload.get("optimizer_controls", {}),
        "contract_rejection_evidence": build_contract_rejection_evidence(
            optimizer_payload.get("rejected_buffer", [])
        ),
        "rejected_buffer": optimizer_payload.get("rejected_buffer", []),
        "failed_training_results": [
            result
            for result in optimizer_payload.get("train_results", [])
            if not result.get("score", {}).get("success")
        ],
        "successful_training_results": [
            {
                "task": result.get("task", {}),
                "score": result.get("score", {}),
                "trace": result.get("output", {}).get("trace", []),
            }
            for result in optimizer_payload.get("train_results", [])
            if result.get("score", {}).get("success")
        ],
    }
    return user_payload


def call_chat_completions(
    url: str,
    api_key: str,
    request_payload: dict[str, Any],
    timeout: float,
) -> dict[str, Any]:
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    request = urllib.request.Request(
        url,
        data=json.dumps(request_payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    import time

    started = time.monotonic()
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if isinstance(payload, dict):
        payload["_duration_seconds"] = time.monotonic() - started
    return payload


def record_model_api_usage(
    request_payload: dict[str, Any],
    response_payload: dict[str, Any],
    *,
    operation: str,
    content: str,
    url: str,
    duration_seconds: float,
) -> None:
    ledger_path = os.environ.get("TEXTSKILL_USAGE_LEDGER_PATH")
    if not ledger_path:
        return
    messages = request_payload.get("messages") if isinstance(request_payload, dict) else []
    prompt_chars = 0
    if isinstance(messages, list):
        for message in messages:
            if isinstance(message, dict):
                prompt_chars += len(str(message.get("content") or ""))
    completion_chars = len(content)
    usage = extract_chat_usage(response_payload)
    append_usage_event(
        ledger_path,
        {
            "kind": "optimizer_model_api",
            "operation": operation,
            "context": usage_context_from_env(),
            "url": url,
            "model": request_payload.get("model"),
            "duration_seconds": duration_seconds,
            "input_chars": prompt_chars,
            "output_chars": completion_chars,
            "actual_prompt_tokens": usage["prompt_tokens"],
            "actual_completion_tokens": usage["completion_tokens"],
            "actual_total_tokens": usage["total_tokens"],
            "estimated_prompt_tokens": estimate_tokens_from_chars(prompt_chars),
            "estimated_completion_tokens": estimate_tokens_from_chars(completion_chars),
        },
    )


def extract_chat_message_content(response_payload: dict[str, Any]) -> str:
    choices = response_payload.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    message = choices[0].get("message") if isinstance(choices[0], dict) else None
    if not isinstance(message, dict):
        return ""
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [
            item.get("text", "")
            for item in content
            if isinstance(item, dict) and isinstance(item.get("text"), str)
        ]
        return "\n".join(parts)
    return ""


def extract_json_text(content: str) -> str:
    stripped = content.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", stripped, re.DOTALL)
    if fenced:
        return fenced.group(1).strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        return stripped

    start = stripped.find("{")
    end = stripped.rfind("}")
    if start >= 0 and end > start:
        return stripped[start : end + 1]
    return stripped


def parse_model_json(content: str) -> dict[str, Any]:
    text = extract_json_text(content)
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = json.loads(escape_control_chars_in_json_strings(text))
    if not isinstance(parsed, dict):
        raise json.JSONDecodeError("Expected a JSON object", text, 0)
    return parsed


def enforce_contract_evidence_source(
    payload: dict[str, Any],
    optimizer_payload: dict[str, Any],
) -> dict[str, Any]:
    """Normalize proposal metadata when rejected-buffer contract evidence is active."""
    operation = str(optimizer_payload.get("operation") or "reflect")
    if operation != "reflect":
        return payload
    evidence = build_contract_rejection_evidence(optimizer_payload.get("rejected_buffer", []))
    priority_contracts = evidence.get("priority_contracts") if isinstance(evidence, dict) else []
    if not evidence.get("available") or not priority_contracts:
        return payload
    policy = evidence.get("proposal_policy") if isinstance(evidence, dict) else {}
    required_protected = policy_contract_names(policy, "anti_regression_contracts")
    required_protected.extend(policy_contract_names(policy, "protected_priority_contracts"))
    proposals = payload.get("proposals") if isinstance(payload, dict) else None
    if not isinstance(proposals, list):
        return payload
    copied = dict(payload)
    normalized_proposals = []
    for proposal in proposals:
        if not isinstance(proposal, dict):
            normalized_proposals.append(proposal)
            continue
        normalized = dict(proposal)
        metadata = normalized.get("metadata") if isinstance(normalized.get("metadata"), dict) else {}
        protected = normalized_string_list(metadata.get("protected_contracts"))
        normalized["metadata"] = {
            **metadata,
            "evidence_source": "contract_rejection_evidence",
            "protected_contracts": dedupe_strings([*protected, *required_protected]),
        }
        normalized_proposals.append(normalized)
    copied["proposals"] = normalized_proposals
    return copied


def policy_contract_names(policy: Any, key: str) -> list[str]:
    if not isinstance(policy, dict):
        return []
    names = []
    for item in policy.get(key) or []:
        if isinstance(item, dict):
            name = str(item.get("contract") or "").strip()
            if name:
                names.append(name)
    return names


def normalized_string_list(value: Any) -> list[str]:
    if isinstance(value, str):
        values = [value]
    elif isinstance(value, list):
        values = value
    else:
        values = []
    return dedupe_strings(str(item).strip() for item in values if str(item).strip())


def dedupe_strings(values: Any) -> list[str]:
    result = []
    seen = set()
    for value in values:
        text = str(value).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result


def escape_control_chars_in_json_strings(text: str) -> str:
    output: list[str] = []
    in_string = False
    escaped = False
    replacements = {"\b": "\\b", "\t": "\\t", "\n": "\\n", "\f": "\\f", "\r": "\\r"}
    for char in text:
        if not in_string:
            output.append(char)
            if char == '"':
                in_string = True
            continue
        if escaped:
            output.append(char)
            escaped = False
        elif char == "\\":
            output.append(char)
            escaped = True
        elif char == '"':
            output.append(char)
            in_string = False
        elif ord(char) < 0x20:
            output.append(replacements.get(char, f"\\u{ord(char):04x}"))
        else:
            output.append(char)
    return "".join(output)


if __name__ == "__main__":
    raise SystemExit(main())
