"""Public zero-call conformance classifier for proposed paper runs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .claims import ClaimClass, EvidenceLevel
from .config import assess_paper_profile
from .lineage import assess_lineage
from .provenance import canonical_json_sha256
from .registry import ConsumedSplitRegistry


@dataclass(frozen=True)
class ContractViolation:
    code: str
    path: str
    message: str


@dataclass(frozen=True)
class PaperRunAssessment:
    claim_class: ClaimClass | None
    evidence_level: EvidenceLevel | None
    violations: tuple[ContractViolation, ...]

    @property
    def eligible(self) -> bool:
        return self.claim_class is not None and not self.violations


def assess_paper_run(
    *,
    profile: Mapping[str, Any],
    lineage: Mapping[str, Any],
    consumed_splits: ConsumedSplitRegistry | None = None,
) -> PaperRunAssessment:
    """Classify protocol and provenance without constructing a backend or engine."""

    violations: list[ContractViolation] = []
    profile_assessment = assess_paper_profile(profile)
    violations.extend(
        ContractViolation(item.code, f"profile.{item.path}", item.message)
        for item in profile_assessment.violations
    )
    lineage_assessment = assess_lineage(lineage)
    violations.extend(
        ContractViolation("invalid_lineage", item.path, item.message)
        for item in lineage_assessment.violations
    )
    claim_class = (
        lineage_assessment.lineage.claim_class
        if lineage_assessment.lineage is not None
        else None
    )
    evidence_level = (
        lineage_assessment.lineage.evidence_level
        if lineage_assessment.lineage is not None
        else None
    )
    if lineage_assessment.lineage is not None:
        allowed_evidence = {
            ClaimClass.MECHANISM_TEST: {None, EvidenceLevel.PAPER_MECHANISM_CONFORMANT},
            ClaimClass.DEVELOPMENT_RESULT: {None},
            ClaimClass.CONTRACT_AWARE_EXTENSION: {None},
            ClaimClass.PAPER_FAITHFUL_DEVELOPMENT: {None},
            ClaimClass.PAPER_FAITHFUL_HELDOUT: {None, EvidenceLevel.FRESH_LOCAL_EFFICACY},
            ClaimClass.PAPER_SCALE_REPRODUCTION: {
                None,
                EvidenceLevel.PARTIAL_PAPER_REPRODUCTION,
                EvidenceLevel.PAPER_SCOPE_REPLICATION,
            },
        }
        if evidence_level not in allowed_evidence[lineage_assessment.lineage.claim_class]:
            violations.append(
                ContractViolation(
                    "claim_evidence_mismatch",
                    "lineage.evidence_level",
                    "evidence level is not supported by this claim class",
                )
            )
        if evidence_level is not None:
            violations.append(
                ContractViolation(
                    "unverified_evidence_level",
                    "lineage.evidence_level",
                    "M1 accepts only null; a later measured gate must assign evidence levels",
                )
            )
        if profile_assessment.profile is not None:
            effective_profile_sha256 = canonical_json_sha256(
                profile_assessment.profile.to_dict()
            )
            recorded_profile_sha256 = lineage_assessment.lineage.payload["artifacts"][
                "profile_sha256"
            ]
            if recorded_profile_sha256 != effective_profile_sha256:
                violations.append(
                    ContractViolation(
                        "profile_hash_mismatch",
                        "lineage.artifacts.profile_sha256",
                        "lineage hash does not identify the effective profile",
                    )
                )
        if lineage_assessment.lineage.claim_class is ClaimClass.CONTRACT_AWARE_EXTENSION:
            violations.append(
                ContractViolation(
                    "claim_protocol_mismatch",
                    "lineage.claim_class",
                    "paper-faithful-v1 cannot emit a contract-aware extension claim",
                )
            )
        if lineage_assessment.lineage.claim_class in {
            ClaimClass.PAPER_FAITHFUL_HELDOUT,
            ClaimClass.PAPER_SCALE_REPRODUCTION,
        }:
            exposure = lineage_assessment.lineage.payload["test_exposure"]
            if (
                exposure["status"] != "untouched"
                or exposure["attempt"] != 0
                or exposure["receipt_sha256"] is not None
                or bool(exposure["history"])
            ):
                violations.append(
                    ContractViolation(
                        "heldout_not_untouched",
                        "lineage.test_exposure",
                        "held-out claims require untouched status, attempt 0, no receipt, and empty history",
                    )
                )
        if lineage_assessment.lineage.protocol_id != "paper-faithful-v1":
            violations.append(
                ContractViolation(
                    "protocol_mismatch",
                    "lineage.protocol_id",
                    "paper runs require protocol_id 'paper-faithful-v1'",
                )
            )
        if (
            profile_assessment.profile is not None
            and lineage_assessment.lineage.protocol_id
            != profile_assessment.profile.protocol_id
        ):
            violations.append(
                ContractViolation(
                    "protocol_mismatch",
                    "lineage.protocol_id",
                    "lineage and profile protocol identifiers differ",
                )
            )
        registry = consumed_splits or ConsumedSplitRegistry.load()
        consumption = registry.find(lineage_assessment.lineage.split_id)
        if consumption is not None:
            if lineage_assessment.lineage.test_exposure_status != "consumed":
                violations.append(
                    ContractViolation(
                        "test_exposure_mismatch",
                        "lineage.test_exposure.status",
                        "registry records this split as consumed",
                    )
                )
            if (
                lineage_assessment.lineage.claim_class
                is ClaimClass.PAPER_FAITHFUL_HELDOUT
                and consumption.protocol_id != lineage_assessment.lineage.protocol_id
            ):
                violations.append(
                    ContractViolation(
                        "consumed_split",
                        "lineage.data.split_id",
                        f"split was consumed by protocol {consumption.protocol_id!r}",
                    )
                )
    return PaperRunAssessment(
        claim_class=claim_class,
        evidence_level=evidence_level,
        violations=tuple(violations),
    )
