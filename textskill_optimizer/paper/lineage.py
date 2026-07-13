"""Lineage validation for paper claim artifacts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from importlib.resources import files
from typing import Any, Mapping

from .claims import ClaimClass, EvidenceLevel
from .schema_validation import SchemaViolation, validate_schema


@dataclass(frozen=True)
class RunLineage:
    claim_class: ClaimClass
    evidence_level: EvidenceLevel | None
    protocol_id: str
    split_id: str
    test_exposure_status: str
    payload: Mapping[str, Any]


@dataclass(frozen=True)
class LineageAssessment:
    lineage: RunLineage | None
    violations: tuple[SchemaViolation, ...]

    @property
    def compliant(self) -> bool:
        return self.lineage is not None and not self.violations


def assess_lineage(payload: Mapping[str, Any]) -> LineageAssessment:
    schema_path = files("textskill_optimizer.paper").joinpath(
        "schemas", "lineage-v1.schema.json"
    )
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    violations = validate_schema(payload, schema)
    if violations:
        return LineageAssessment(lineage=None, violations=violations)
    try:
        claim_class = ClaimClass(payload["claim_class"])
    except ValueError:
        return LineageAssessment(
            lineage=None,
            violations=(SchemaViolation("$.claim_class", "unknown claim class"),),
        )
    evidence_level = (
        EvidenceLevel(payload["evidence_level"])
        if payload["evidence_level"] is not None
        else None
    )
    return LineageAssessment(
        lineage=RunLineage(
            claim_class=claim_class,
            evidence_level=evidence_level,
            protocol_id=payload["protocol_id"],
            split_id=payload["data"]["split_id"],
            test_exposure_status=payload["test_exposure"]["status"],
            payload=payload,
        ),
        violations=(),
    )
