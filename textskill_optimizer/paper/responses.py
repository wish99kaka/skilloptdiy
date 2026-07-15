"""Strict optimizer response contracts for the paper fast loop."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .backend import OptimizerStage
from .schema_validation import validate_schema
from .types import PaperEdit, PaperEditOperation, PaperEditSource


class OptimizerContractViolation(ValueError):
    """Raised when a model response cannot drive a paper optimizer stage."""


@dataclass(frozen=True)
class ParsedPatchResponse:
    response_schema: Mapping[str, Any]
    reasoning: str
    edits: tuple[PaperEdit, ...]
    converged: bool = False


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
) -> dict[str, Any]:
    """Build the strict JSON contract matching one optimizer prompt."""

    _require_stage_and_budget(stage, edit_budget)
    if candidate_count is not None and (
        type(candidate_count) is not int or candidate_count < 0
    ):
        raise ValueError("candidate_count must be a non-negative integer")
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
    return ParsedPatchResponse(
        response_schema=schema,
        reasoning=reasoning,
        edits=tuple(edits),
        converged=converged,
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


def _require_stage_and_budget(stage: OptimizerStage, edit_budget: int) -> None:
    if type(stage) is not OptimizerStage:
        raise ValueError("stage must be an exact OptimizerStage")
    if type(edit_budget) is not int or edit_budget < 1:
        raise ValueError("edit_budget must be a positive integer")


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
