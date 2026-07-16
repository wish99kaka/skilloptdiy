"""Authorization policy for the clean-commit M6 zero-cost gate."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CodeIdentity:
    commit_sha: str | None
    clean: bool

    def __post_init__(self) -> None:
        if self.commit_sha is not None and (
            type(self.commit_sha) is not str
            or len(self.commit_sha) != 40
            or any(character not in "0123456789abcdef" for character in self.commit_sha)
        ):
            raise ValueError("code identity requires a full lowercase Git SHA")
        if type(self.clean) is not bool:
            raise ValueError("code identity clean flag must be exact bool")


@dataclass(frozen=True)
class ZeroCostGateEvidence:
    test_returncode: int
    before: CodeIdentity
    after: CodeIdentity
    network_guard_active: bool
    external_calls: int

    def __post_init__(self) -> None:
        if type(self.test_returncode) is not int or self.test_returncode < 0:
            raise ValueError("zero-cost test return code must be non-negative")
        if type(self.before) is not CodeIdentity or type(self.after) is not CodeIdentity:
            raise ValueError("zero-cost evidence requires exact code identities")
        if type(self.network_guard_active) is not bool:
            raise ValueError("network guard flag must be exact bool")
        if type(self.external_calls) is not int or self.external_calls < 0:
            raise ValueError("external call count must be a non-negative integer")


@dataclass(frozen=True)
class ZeroCostGateViolation:
    code: str
    message: str


@dataclass(frozen=True)
class ZeroCostGateDecision:
    status: str
    authorized: bool
    violations: tuple[ZeroCostGateViolation, ...]


def assess_zero_cost_gate(evidence: ZeroCostGateEvidence) -> ZeroCostGateDecision:
    """Authorize development only for one guarded test run on one clean commit."""

    if type(evidence) is not ZeroCostGateEvidence:
        raise ValueError("zero-cost gate requires exact evidence")
    evidence.__post_init__()
    violations: list[ZeroCostGateViolation] = []
    if evidence.test_returncode:
        violations.append(
            ZeroCostGateViolation("tests_failed", "the default test gate failed")
        )
    if not evidence.network_guard_active:
        violations.append(
            ZeroCostGateViolation(
                "network_guard_missing",
                "external network access was not mechanically blocked",
            )
        )
    if evidence.external_calls:
        violations.append(
            ZeroCostGateViolation(
                "external_calls_detected",
                "zero-cost acceptance cannot contain external calls",
            )
        )
    if (
        evidence.before.commit_sha is None
        or evidence.after.commit_sha is None
        or not evidence.before.clean
        or not evidence.after.clean
        or evidence.before.commit_sha != evidence.after.commit_sha
    ):
        violations.append(
            ZeroCostGateViolation(
                "code_identity_changed",
                "the same clean Git commit must surround the complete test run",
            )
        )
    if not violations:
        return ZeroCostGateDecision("passed", True, ())
    priority = {
        "tests_failed": "blocked_tests_failed",
        "network_guard_missing": "blocked_network_guard",
        "external_calls_detected": "blocked_external_calls",
        "code_identity_changed": "blocked_uncommitted",
    }
    return ZeroCostGateDecision(
        priority[violations[0].code],
        False,
        tuple(violations),
    )
