"""Claim taxonomy for protocol-scoped SkillOpt evidence."""

from __future__ import annotations

from enum import Enum


class ClaimClass(str, Enum):
    MECHANISM_TEST = "mechanism_test"
    DEVELOPMENT_RESULT = "development_result"
    CONTRACT_AWARE_EXTENSION = "contract_aware_extension"
    PAPER_FAITHFUL_DEVELOPMENT = "paper_faithful_development"
    PAPER_FAITHFUL_HELDOUT = "paper_faithful_heldout"
    PAPER_SCALE_REPRODUCTION = "paper_scale_reproduction"

    @property
    def requires_paper_profile(self) -> bool:
        return self in {
            ClaimClass.PAPER_FAITHFUL_DEVELOPMENT,
            ClaimClass.PAPER_FAITHFUL_HELDOUT,
            ClaimClass.PAPER_SCALE_REPRODUCTION,
        }


class EvidenceLevel(str, Enum):
    """Program-level conclusions, distinct from result-bundle claim classes."""

    PAPER_MECHANISM_CONFORMANT = "paper_mechanism_conformant"
    FRESH_LOCAL_EFFICACY = "fresh_local_efficacy"
    PARTIAL_PAPER_REPRODUCTION = "partial_paper_reproduction"
    PAPER_SCOPE_REPLICATION = "paper_scope_replication"
