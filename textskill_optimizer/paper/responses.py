"""Strict optimizer response contracts for the paper fast loop."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .backend import OptimizerStage
from .schema_validation import validate_schema
from .types import (
    ObservedFailurePattern,
    PaperEdit,
    PaperEditOperation,
    PaperEditSource,
    PaperSuggestion,
    PaperSuggestionPriority,
    PaperSuggestionType,
)


class OptimizerContractViolation(ValueError):
    """Raised when a model response cannot drive a paper optimizer stage."""


@dataclass(frozen=True)
class ParsedPatchResponse:
    response_schema: Mapping[str, Any]
    reasoning: str
    edits: tuple[PaperEdit, ...]
    converged: bool = False
    failure_patterns: tuple[ObservedFailurePattern, ...] = ()


@dataclass(frozen=True)
class ParsedTextUpdate:
    response_schema: Mapping[str, Any]
    reasoning: str
    content: str


@dataclass(frozen=True)
class ParsedLearningRate:
    response_schema: Mapping[str, Any]
    reasoning: str
    raw_learning_rate: int
    learning_rate: int
    confidence: str
    risk_notes: tuple[str, ...]


@dataclass(frozen=True)
class ParsedSuggestionResponse:
    response_schema: Mapping[str, Any]
    reasoning: str
    suggestions: tuple[PaperSuggestion, ...]
    converged: bool = False
    failure_patterns: tuple[ObservedFailurePattern, ...] = ()


@dataclass(frozen=True)
class ParsedRewriteResponse:
    response_schema: Mapping[str, Any]
    reasoning: str
    change_summary: tuple[str, ...]
    new_skill: str


_PATCH_STAGES = frozenset(
    {
        OptimizerStage.REFLECT_FAILURE,
        OptimizerStage.REFLECT_SUCCESS,
        OptimizerStage.REFINE,
        OptimizerStage.MERGE_FAILURE,
        OptimizerStage.MERGE_SUCCESS,
        OptimizerStage.MERGE_FINAL_FAILURE_PRIORITIZED,
    }
)


def optimizer_response_schema(
    stage: OptimizerStage,
    *,
    edit_budget: int,
    candidate_count: int | None = None,
    update_mode: str = "patch",
) -> dict[str, Any]:
    """Build the strict JSON contract matching one optimizer prompt."""

    _require_stage_and_budget(stage, edit_budget)
    if candidate_count is not None and (
        type(candidate_count) is not int or candidate_count < 0
    ):
        raise ValueError("candidate_count must be a non-negative integer")
    if update_mode not in {"patch", "rewrite_from_suggestions"}:
        raise ValueError("unsupported optimizer response update mode")
    if update_mode == "rewrite_from_suggestions":
        if stage is OptimizerStage.REFLECT_FAILURE:
            return _failure_suggestion_schema(edit_budget)
        if stage is OptimizerStage.REFLECT_SUCCESS:
            return _success_suggestion_schema(edit_budget)
        if stage is OptimizerStage.REFINE:
            return _refinement_suggestion_schema(edit_budget)
        if stage is OptimizerStage.MERGE_FAILURE:
            return _merge_suggestion_schema((PaperEditSource.FAILURE.value,))
        if stage is OptimizerStage.MERGE_SUCCESS:
            return _merge_suggestion_schema((PaperEditSource.SUCCESS.value,))
        if stage is OptimizerStage.MERGE_FINAL_FAILURE_PRIORITIZED:
            return _merge_suggestion_schema(
                (PaperEditSource.FAILURE.value, PaperEditSource.SUCCESS.value)
            )
    if stage is OptimizerStage.REFLECT_FAILURE:
        return _failure_schema(edit_budget)
    if stage is OptimizerStage.REFLECT_SUCCESS:
        return _success_schema(edit_budget)
    if stage is OptimizerStage.REFINE:
        return _refinement_schema(edit_budget)
    if stage is OptimizerStage.MERGE_FAILURE:
        return _merge_schema((PaperEditSource.FAILURE.value,))
    if stage is OptimizerStage.MERGE_SUCCESS:
        return _merge_schema((PaperEditSource.SUCCESS.value,))
    if stage is OptimizerStage.MERGE_FINAL_FAILURE_PRIORITIZED:
        return _merge_schema(
            (PaperEditSource.FAILURE.value, PaperEditSource.SUCCESS.value)
        )
    if stage is OptimizerStage.RANK_TOP_L:
        index_schema: dict[str, Any] = {"type": "integer", "minimum": 0}
        if candidate_count:
            index_schema["maximum"] = candidate_count - 1
        return _object_schema(
            required=("reasoning", "selected_indices"),
            properties={
                "reasoning": {"type": "string"},
                "selected_indices": {
                    "type": "array",
                    "maxItems": edit_budget,
                    "items": index_schema,
                },
            },
        )
    raise ValueError(f"stage has no fast-loop response contract: {stage.value}")


def rewrite_response_schema() -> dict[str, Any]:
    return _object_schema(
        required=("reasoning", "change_summary", "new_skill"),
        properties={
            "reasoning": {"type": "string", "minLength": 1},
            "change_summary": {
                "type": "array",
                "items": {"type": "string", "minLength": 1},
            },
            "new_skill": {"type": "string", "minLength": 1},
        },
    )


def parse_patch_response(
    *,
    stage: OptimizerStage,
    payload: Mapping[str, Any],
    edit_budget: int,
    edit_id_prefix: str,
    expected_batch_size: int | None = None,
    source_type: PaperEditSource | str | None = None,
) -> ParsedPatchResponse:
    """Validate and normalize one reflection/refinement/merge response."""

    if stage not in _PATCH_STAGES:
        raise ValueError(f"not a patch response stage: {stage.value}")
    if type(edit_id_prefix) is not str or not edit_id_prefix.strip():
        raise ValueError("edit_id_prefix is required")
    schema = optimizer_response_schema(stage, edit_budget=edit_budget)
    _require_valid_payload(payload, schema)
    if expected_batch_size is not None:
        if type(expected_batch_size) is not int or expected_batch_size < 1:
            raise ValueError("expected_batch_size must be a positive integer")
        if payload.get("batch_size") != expected_batch_size:
            raise OptimizerContractViolation(
                "optimizer batch_size does not match the reflection minibatch"
            )

    if stage in {
        OptimizerStage.REFLECT_FAILURE,
        OptimizerStage.REFLECT_SUCCESS,
    }:
        patch = payload["patch"]
        reasoning = patch["reasoning"]
        raw_edits = patch["edits"]
        resolved_source = (
            PaperEditSource.FAILURE
            if stage is OptimizerStage.REFLECT_FAILURE
            else PaperEditSource.SUCCESS
        )
        converged = False
    elif stage is OptimizerStage.REFINE:
        reasoning = payload["reasoning"]
        raw_edits = payload["edits"]
        resolved_source = _coerce_source(source_type)
        converged = payload["converged"]
    else:
        reasoning = payload["reasoning"]
        raw_edits = payload["edits"]
        resolved_source = None
        converged = False

    edits: list[PaperEdit] = []
    for index, raw_edit in enumerate(raw_edits, 1):
        item_source = (
            _coerce_source(raw_edit["source_type"])
            if "source_type" in raw_edit
            else resolved_source
        )
        try:
            edits.append(
                PaperEdit(
                    edit_id=f"{edit_id_prefix}-{index}",
                    operation=PaperEditOperation(raw_edit["op"]),
                    target=raw_edit.get("target", ""),
                    content=raw_edit.get("content", ""),
                    rationale=reasoning,
                    support_count=raw_edit.get("support_count", 1),
                    source_type=item_source,
                )
            )
        except (TypeError, ValueError) as error:
            raise OptimizerContractViolation(
                f"invalid edit at index {index - 1}: {error}"
            ) from error
    try:
        failure_patterns = (
            tuple(
                ObservedFailurePattern(
                    failure_type=item["failure_type"],
                    count=item["count"],
                    description=item["description"],
                )
                for item in payload["failure_summary"]
            )
            if stage is OptimizerStage.REFLECT_FAILURE
            else ()
        )
    except ValueError as error:
        raise OptimizerContractViolation(
            f"invalid failure_summary: {error}"
        ) from error
    return ParsedPatchResponse(
        response_schema=schema,
        reasoning=reasoning,
        edits=tuple(edits),
        converged=converged,
        failure_patterns=failure_patterns,
    )


def parse_rank_response(
    *,
    payload: Mapping[str, Any],
    candidates: tuple[PaperEdit, ...],
    edit_budget: int,
) -> tuple[PaperEdit, ...]:
    """Accept only the optimizer's valid, ordered top-L selection."""

    if type(candidates) is not tuple or any(
        type(candidate) is not PaperEdit for candidate in candidates
    ):
        raise ValueError("rank candidates must be a tuple of exact PaperEdit values")
    schema = optimizer_response_schema(
        OptimizerStage.RANK_TOP_L,
        edit_budget=edit_budget,
        candidate_count=len(candidates),
    )
    _require_valid_payload(payload, schema)
    indices = payload["selected_indices"]
    if len(indices) != len(set(indices)):
        raise OptimizerContractViolation("selected_indices must be unique")
    if any(index >= len(candidates) for index in indices):
        raise OptimizerContractViolation("selected_indices contains an unknown edit")
    return tuple(candidates[index] for index in indices)


def parse_suggestion_response(
    *,
    stage: OptimizerStage,
    payload: Mapping[str, Any],
    edit_budget: int,
    suggestion_id_prefix: str,
    expected_batch_size: int | None = None,
    source_type: PaperEditSource | str | None = None,
) -> ParsedSuggestionResponse:
    if stage not in _PATCH_STAGES:
        raise ValueError(f"not a suggestion response stage: {stage.value}")
    if type(suggestion_id_prefix) is not str or not suggestion_id_prefix.strip():
        raise ValueError("suggestion_id_prefix is required")
    schema = optimizer_response_schema(
        stage,
        edit_budget=edit_budget,
        update_mode="rewrite_from_suggestions",
    )
    _require_valid_payload(payload, schema)
    if expected_batch_size is not None and payload.get(
        "batch_size"
    ) != expected_batch_size:
        raise OptimizerContractViolation(
            "optimizer batch_size does not match the reflection minibatch"
        )
    if stage in {
        OptimizerStage.REFLECT_FAILURE,
        OptimizerStage.REFLECT_SUCCESS,
    }:
        container = payload["patch"]
        reasoning = container["reasoning"]
        raw_suggestions = container["revise_suggestions"]
        resolved_source = (
            PaperEditSource.FAILURE
            if stage is OptimizerStage.REFLECT_FAILURE
            else PaperEditSource.SUCCESS
        )
        converged = False
    else:
        reasoning = payload["reasoning"]
        raw_suggestions = payload["revise_suggestions"]
        resolved_source = (
            _coerce_source(source_type)
            if stage is OptimizerStage.REFINE
            else None
        )
        converged = payload.get("converged", False)
    suggestions: list[PaperSuggestion] = []
    for index, item in enumerate(raw_suggestions, 1):
        item_source = (
            _coerce_source(item["source_type"])
            if "source_type" in item
            else resolved_source
        )
        try:
            suggestions.append(
                PaperSuggestion(
                    suggestion_id=f"{suggestion_id_prefix}-{index}",
                    suggestion_type=PaperSuggestionType(item["type"]),
                    title=item["title"],
                    motivation=item["motivation"],
                    instruction=item["instruction"],
                    priority_hint=PaperSuggestionPriority(
                        item["priority_hint"]
                    ),
                    support_count=item.get("support_count", 1),
                    source_type=item_source,
                )
            )
        except (TypeError, ValueError) as error:
            raise OptimizerContractViolation(
                f"invalid suggestion at index {index - 1}: {error}"
            ) from error
    failure_patterns = _parse_failure_patterns(stage, payload)
    return ParsedSuggestionResponse(
        response_schema=schema,
        reasoning=reasoning,
        suggestions=tuple(suggestions),
        converged=converged,
        failure_patterns=failure_patterns,
    )


def parse_suggestion_rank_response(
    *,
    payload: Mapping[str, Any],
    candidates: tuple[PaperSuggestion, ...],
    edit_budget: int,
) -> tuple[PaperSuggestion, ...]:
    if type(candidates) is not tuple or any(
        type(candidate) is not PaperSuggestion for candidate in candidates
    ):
        raise ValueError("rank candidates must be exact PaperSuggestion values")
    schema = optimizer_response_schema(
        OptimizerStage.RANK_TOP_L,
        edit_budget=edit_budget,
        candidate_count=len(candidates),
        update_mode="rewrite_from_suggestions",
    )
    _require_valid_payload(payload, schema)
    indices = payload["selected_indices"]
    if len(indices) != len(set(indices)):
        raise OptimizerContractViolation("selected_indices must be unique")
    if any(index >= len(candidates) for index in indices):
        raise OptimizerContractViolation("selected_indices contains an unknown item")
    return tuple(candidates[index] for index in indices)


def parse_rewrite_response(payload: Mapping[str, Any]) -> ParsedRewriteResponse:
    schema = rewrite_response_schema()
    _require_valid_payload(payload, schema)
    if (
        not payload["reasoning"].strip()
        or not payload["new_skill"].strip()
        or any(not item.strip() for item in payload["change_summary"])
    ):
        raise OptimizerContractViolation(
            "rewrite response text fields must contain content"
        )
    return ParsedRewriteResponse(
        response_schema=schema,
        reasoning=payload["reasoning"],
        change_summary=tuple(payload["change_summary"]),
        new_skill=payload["new_skill"].rstrip() + "\n",
    )


def learning_rate_response_schema() -> dict[str, Any]:
    return _object_schema(
        required=("learning_rate", "reasoning", "confidence", "risk_notes"),
        properties={
            "learning_rate": {"type": "integer", "minimum": 0},
            "reasoning": {"type": "string", "minLength": 1},
            "confidence": {
                "type": "string",
                "enum": ["low", "medium", "high"],
            },
            "risk_notes": {
                "type": "array",
                "items": {"type": "string", "minLength": 1},
            },
        },
    )


def parse_learning_rate_response(
    *,
    payload: Mapping[str, Any],
    candidate_count: int,
) -> ParsedLearningRate:
    if type(candidate_count) is not int or candidate_count < 0:
        raise ValueError("candidate_count must be a non-negative integer")
    schema = learning_rate_response_schema()
    _require_valid_payload(payload, schema)
    if not payload["reasoning"].strip() or any(
        not note.strip() for note in payload["risk_notes"]
    ):
        raise OptimizerContractViolation(
            "autonomous learning-rate text fields must contain content"
        )
    raw = payload["learning_rate"]
    return ParsedLearningRate(
        response_schema=schema,
        reasoning=payload["reasoning"],
        raw_learning_rate=raw,
        learning_rate=min(raw, candidate_count),
        confidence=payload["confidence"],
        risk_notes=tuple(payload["risk_notes"]),
    )


def epoch_response_schema(stage: OptimizerStage) -> dict[str, Any]:
    if stage is OptimizerStage.PROPOSE_SLOW_UPDATE:
        content_field = "slow_update_content"
    elif stage is OptimizerStage.UPDATE_META_SKILL:
        content_field = "meta_skill_content"
    else:
        raise ValueError(f"stage has no epoch response contract: {stage.value}")
    return _object_schema(
        required=("reasoning", content_field),
        properties={
            "reasoning": {"type": "string"},
            content_field: {"type": "string", "minLength": 1},
        },
    )


def parse_epoch_response(
    *,
    stage: OptimizerStage,
    payload: Mapping[str, Any],
) -> ParsedTextUpdate:
    schema = epoch_response_schema(stage)
    _require_valid_payload(payload, schema)
    field = (
        "slow_update_content"
        if stage is OptimizerStage.PROPOSE_SLOW_UPDATE
        else "meta_skill_content"
    )
    if not payload[field].strip():
        raise OptimizerContractViolation(
            f"invalid optimizer response: {field} must contain guidance"
        )
    return ParsedTextUpdate(
        response_schema=schema,
        reasoning=payload["reasoning"],
        content=payload[field],
    )


def _require_stage_and_budget(stage: OptimizerStage, edit_budget: int) -> None:
    if type(stage) is not OptimizerStage:
        raise ValueError("stage must be an exact OptimizerStage")
    minimum = 0 if stage is OptimizerStage.RANK_TOP_L else 1
    if type(edit_budget) is not int or edit_budget < minimum:
        raise ValueError(f"edit_budget must be an integer >= {minimum}")


def _require_valid_payload(
    payload: Mapping[str, Any], schema: Mapping[str, Any]
) -> None:
    if type(payload) is not dict:
        raise OptimizerContractViolation(
            "optimizer payload must be an exact JSON object"
        )
    violations = validate_schema(payload, schema)
    if violations:
        details = "; ".join(
            f"{violation.path}: {violation.message}" for violation in violations
        )
        raise OptimizerContractViolation(f"invalid optimizer response: {details}")


def _coerce_source(value: PaperEditSource | str | None) -> PaperEditSource:
    try:
        return PaperEditSource(value)
    except (TypeError, ValueError) as error:
        raise OptimizerContractViolation(
            "source_type must be failure or success"
        ) from error


def _failure_schema(edit_budget: int) -> dict[str, Any]:
    return _object_schema(
        required=("batch_size", "failure_summary", "patch"),
        properties={
            "batch_size": {"type": "integer", "minimum": 1},
            "failure_summary": {
                "type": "array",
                "items": _object_schema(
                    required=("failure_type", "count", "description"),
                    properties={
                        "failure_type": {"type": "string", "minLength": 1},
                        "count": {"type": "integer", "minimum": 1},
                        "description": {"type": "string", "minLength": 1},
                    },
                ),
            },
            "patch": _analyst_patch_schema(edit_budget),
        },
    )


def _parse_failure_patterns(
    stage: OptimizerStage,
    payload: Mapping[str, Any],
) -> tuple[ObservedFailurePattern, ...]:
    if stage is not OptimizerStage.REFLECT_FAILURE:
        return ()
    try:
        return tuple(
            ObservedFailurePattern(
                failure_type=item["failure_type"],
                count=item["count"],
                description=item["description"],
            )
            for item in payload["failure_summary"]
        )
    except ValueError as error:
        raise OptimizerContractViolation(
            f"invalid failure_summary: {error}"
        ) from error


def _failure_suggestion_schema(edit_budget: int) -> dict[str, Any]:
    return _object_schema(
        required=("batch_size", "failure_summary", "patch"),
        properties={
            "batch_size": {"type": "integer", "minimum": 1},
            "failure_summary": {
                "type": "array",
                "items": _object_schema(
                    required=("failure_type", "count", "description"),
                    properties={
                        "failure_type": {"type": "string", "minLength": 1},
                        "count": {"type": "integer", "minimum": 1},
                        "description": {"type": "string", "minLength": 1},
                    },
                ),
            },
            "patch": _analyst_suggestion_schema(edit_budget),
        },
    )


def _success_suggestion_schema(edit_budget: int) -> dict[str, Any]:
    return _object_schema(
        required=("batch_size", "success_patterns", "patch"),
        properties={
            "batch_size": {"type": "integer", "minimum": 1},
            "success_patterns": {
                "type": "array",
                "items": {"type": "string", "minLength": 1},
            },
            "patch": _analyst_suggestion_schema(edit_budget),
        },
    )


def _refinement_suggestion_schema(edit_budget: int) -> dict[str, Any]:
    return _object_schema(
        required=("reasoning", "revise_suggestions", "converged"),
        properties={
            "reasoning": {"type": "string"},
            "revise_suggestions": _suggestions_schema(
                max_items=edit_budget,
                require_merge_fields=False,
                source_values=(),
            ),
            "converged": {"type": "boolean"},
        },
    )


def _merge_suggestion_schema(source_values: tuple[str, ...]) -> dict[str, Any]:
    return _object_schema(
        required=("reasoning", "revise_suggestions"),
        properties={
            "reasoning": {"type": "string"},
            "revise_suggestions": _suggestions_schema(
                max_items=None,
                require_merge_fields=True,
                source_values=source_values,
            ),
        },
    )


def _analyst_suggestion_schema(edit_budget: int) -> dict[str, Any]:
    return _object_schema(
        required=("reasoning", "revise_suggestions"),
        properties={
            "reasoning": {"type": "string"},
            "revise_suggestions": _suggestions_schema(
                max_items=edit_budget,
                require_merge_fields=False,
                source_values=(),
            ),
        },
    )


def _suggestions_schema(
    *,
    max_items: int | None,
    require_merge_fields: bool,
    source_values: tuple[str, ...],
) -> dict[str, Any]:
    properties: dict[str, Any] = {
        "type": {
            "type": "string",
            "enum": [item.value for item in PaperSuggestionType],
        },
        "title": {"type": "string", "minLength": 1},
        "motivation": {"type": "string", "minLength": 1},
        "instruction": {"type": "string", "minLength": 1},
        "priority_hint": {
            "type": "string",
            "enum": [item.value for item in PaperSuggestionPriority],
        },
    }
    required = [
        "type",
        "title",
        "motivation",
        "instruction",
        "priority_hint",
    ]
    if require_merge_fields:
        properties.update(
            {
                "support_count": {"type": "integer", "minimum": 1},
                "source_type": {"type": "string", "enum": list(source_values)},
            }
        )
        required.extend(("support_count", "source_type"))
    schema: dict[str, Any] = {
        "type": "array",
        "items": _object_schema(required=tuple(required), properties=properties),
    }
    if max_items is not None:
        schema["maxItems"] = max_items
    return schema


def _success_schema(edit_budget: int) -> dict[str, Any]:
    return _object_schema(
        required=("batch_size", "success_patterns", "patch"),
        properties={
            "batch_size": {"type": "integer", "minimum": 1},
            "success_patterns": {
                "type": "array",
                "items": {"type": "string", "minLength": 1},
            },
            "patch": _analyst_patch_schema(edit_budget),
        },
    )


def _refinement_schema(edit_budget: int) -> dict[str, Any]:
    return _object_schema(
        required=("reasoning", "edits", "converged"),
        properties={
            "reasoning": {"type": "string"},
            "edits": _edits_schema(
                max_items=edit_budget,
                require_merge_fields=False,
                source_values=(),
            ),
            "converged": {"type": "boolean"},
        },
    )


def _merge_schema(source_values: tuple[str, ...]) -> dict[str, Any]:
    return _object_schema(
        required=("reasoning", "edits"),
        properties={
            "reasoning": {"type": "string"},
            "edits": _edits_schema(
                max_items=None,
                require_merge_fields=True,
                source_values=source_values,
            ),
        },
    )


def _analyst_patch_schema(edit_budget: int) -> dict[str, Any]:
    return _object_schema(
        required=("reasoning", "edits"),
        properties={
            "reasoning": {"type": "string"},
            "edits": _edits_schema(
                max_items=edit_budget,
                require_merge_fields=False,
                source_values=(),
            ),
        },
    )


def _edits_schema(
    *,
    max_items: int | None,
    require_merge_fields: bool,
    source_values: tuple[str, ...],
) -> dict[str, Any]:
    properties: dict[str, Any] = {
        "op": {"type": "string", "enum": [item.value for item in PaperEditOperation]},
        "target": {"type": "string"},
        "content": {"type": "string"},
    }
    required = ["op"]
    if require_merge_fields:
        properties.update(
            {
                "support_count": {"type": "integer", "minimum": 1},
                "source_type": {"type": "string", "enum": list(source_values)},
            }
        )
        required.extend(("support_count", "source_type"))
    schema: dict[str, Any] = {
        "type": "array",
        "items": _object_schema(required=tuple(required), properties=properties),
    }
    if max_items is not None:
        schema["maxItems"] = max_items
    return schema


def _object_schema(
    *, required: tuple[str, ...], properties: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "type": "object",
        "required": list(required),
        "additionalProperties": False,
        "properties": dict(properties),
    }
