"""Final-test controller isolated from every optimization-side module."""

from __future__ import annotations

import hashlib
import re
import subprocess
from dataclasses import asdict, dataclass

from .controller_process import (
    ControllerRegistry,
    ControllerRole,
    canonical_json,
    parse_signed_response,
    require_exact_keys,
    require_finite_scalar,
)
from .errors import DataFirewallViolation
from .provenance import canonical_json_sha256


def _require_hash(name: str, value: str, length: int) -> None:
    if type(value) is not str or re.fullmatch(rf"[0-9a-f]{{{length}}}", value) is None:
        raise DataFirewallViolation(f"{name} must be {length} lowercase hex characters")


@dataclass(frozen=True)
class FrozenCandidate:
    candidate_id: str
    skill_text: str
    skill_sha256: str

    def __post_init__(self) -> None:
        if type(self.candidate_id) is not str or not self.candidate_id.strip():
            raise DataFirewallViolation("frozen candidate requires candidate_id")
        if type(self.skill_text) is not str or not self.skill_text.strip():
            raise DataFirewallViolation("frozen candidate requires skill_text")
        expected = hashlib.sha256(self.skill_text.encode("utf-8")).hexdigest()
        if self.skill_sha256 != expected:
            raise DataFirewallViolation(
                f"candidate {self.candidate_id!r} skill hash does not match its text"
            )

    @classmethod
    def from_skill(cls, candidate_id: str, skill_text: str) -> "FrozenCandidate":
        return cls(
            candidate_id=candidate_id,
            skill_text=skill_text,
            skill_sha256=hashlib.sha256(skill_text.encode("utf-8")).hexdigest(),
        )


@dataclass(frozen=True)
class FinalEvaluationPolicy:
    test_split_id: str
    split_manifest_sha256: str
    controller_registry_sha256: str
    runner_sha256: str
    scorer_sha256: str
    harness_sha256: str
    environment_sha256: str
    profile_sha256: str
    local_code_commit: str
    protocol_id: str = "paper-faithful-v1"

    def __post_init__(self) -> None:
        if type(self.test_split_id) is not str or not self.test_split_id.strip():
            raise DataFirewallViolation("final policy requires test_split_id")
        if self.protocol_id != "paper-faithful-v1":
            raise DataFirewallViolation("final policy requires paper-faithful-v1")
        for name in (
            "split_manifest_sha256",
            "controller_registry_sha256",
            "runner_sha256",
            "scorer_sha256",
            "harness_sha256",
            "environment_sha256",
            "profile_sha256",
        ):
            _require_hash(name, getattr(self, name), 64)
        _require_hash("local_code_commit", self.local_code_commit, 40)


@dataclass(frozen=True)
class FinalEvaluationPlan:
    candidates: tuple[FrozenCandidate, ...]
    policy: FinalEvaluationPolicy

    def __post_init__(self) -> None:
        if type(self.candidates) is not tuple or not self.candidates:
            raise DataFirewallViolation("final evaluation requires frozen candidates")
        if any(type(item) is not FrozenCandidate for item in self.candidates):
            raise DataFirewallViolation("final evaluation accepts only FrozenCandidate")
        candidate_ids = [item.candidate_id for item in self.candidates]
        if len(candidate_ids) != len(set(candidate_ids)):
            raise DataFirewallViolation("final candidate identifiers must be unique")
        if type(self.policy) is not FinalEvaluationPolicy:
            raise DataFirewallViolation("final evaluation requires frozen policy")

    @classmethod
    def freeze(
        cls,
        *,
        candidates: tuple[FrozenCandidate, ...],
        policy: FinalEvaluationPolicy,
    ) -> "FinalEvaluationPlan":
        return cls(candidates=candidates, policy=policy)

    def to_manifest(self) -> dict[str, object]:
        return {
            "protocol_id": self.policy.protocol_id,
            "candidates": [
                {
                    "candidate_id": item.candidate_id,
                    "skill_sha256": item.skill_sha256,
                }
                for item in self.candidates
            ],
            "policy": asdict(self.policy),
        }

    @property
    def sha256(self) -> str:
        return canonical_json_sha256(self.to_manifest())


@dataclass(frozen=True)
class FinalCandidateScore:
    candidate_id: str
    skill_sha256: str
    score: float


@dataclass(frozen=True)
class FinalTestReport:
    plan_sha256: str
    scores: tuple[FinalCandidateScore, ...]


@dataclass(frozen=True)
class FinalTestController:
    """Invoke a test-owned process only after revalidating a frozen plan."""

    registry: ControllerRegistry
    controller_id: str

    def __post_init__(self) -> None:
        if type(self.registry) is not ControllerRegistry:
            raise DataFirewallViolation("final test requires exact controller registry")
        self.registry.require(self.controller_id, role=ControllerRole.TEST)

    def evaluate(self, plan: FinalEvaluationPlan) -> FinalTestReport:
        if type(plan) is not FinalEvaluationPlan:
            qualifier = "exact " if isinstance(plan, FinalEvaluationPlan) else "frozen "
            raise DataFirewallViolation(
                f"test access requires an {qualifier}FinalEvaluationPlan"
            )
        self.__post_init__()
        validated = _revalidate_plan(plan)
        registration = self.registry.require(
            self.controller_id, role=ControllerRole.TEST
        )
        if validated.policy.controller_registry_sha256 != self.registry.sha256:
            raise DataFirewallViolation("final policy registry hash does not match")
        if validated.policy.test_split_id != registration.split_id:
            raise DataFirewallViolation("final policy test split does not match registry")
        if (
            validated.policy.split_manifest_sha256
            != registration.artifact("split_manifest").sha256
        ):
            raise DataFirewallViolation(
                "final policy split manifest hash does not match registry"
            )
        for artifact_id, expected in (
            ("runner", validated.policy.runner_sha256),
            ("scorer", validated.policy.scorer_sha256),
            ("harness", validated.policy.harness_sha256),
        ):
            if registration.artifact(artifact_id).sha256 != expected:
                raise DataFirewallViolation(
                    f"final policy {artifact_id} hash does not match registered artifact"
                )
        plan_sha256 = validated.sha256
        request = {
            "operation": "final_test",
            "plan_sha256": plan_sha256,
            "candidates": [
                {
                    "candidate_id": item.candidate_id,
                    "skill_text": item.skill_text,
                    "skill_sha256": item.skill_sha256,
                }
                for item in validated.candidates
            ],
            "policy": asdict(validated.policy),
        }
        try:
            completed = subprocess.run(
                registration.argv,
                text=True,
                input=canonical_json(request),
                capture_output=True,
                timeout=registration.timeout_seconds,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            raise DataFirewallViolation("final test process could not complete") from error
        if completed.returncode != 0:
            raise DataFirewallViolation(
                f"final test process failed with exit code {completed.returncode}"
            )
        response, _ = parse_signed_response(
            registration=registration,
            request=request,
            stdout=completed.stdout,
        )
        require_exact_keys(
            response,
            {"plan_sha256", "scores"},
            context="final test response",
        )
        if response["plan_sha256"] != plan_sha256:
            raise DataFirewallViolation("final test response plan hash does not match")
        raw_scores = response["scores"]
        if type(raw_scores) is not list or len(raw_scores) != len(validated.candidates):
            raise DataFirewallViolation("final test response has wrong candidate count")
        scores: list[FinalCandidateScore] = []
        for candidate, raw in zip(validated.candidates, raw_scores):
            if type(raw) is not dict:
                raise DataFirewallViolation("final candidate score must be an object")
            require_exact_keys(
                raw,
                {"candidate_id", "skill_sha256", "score"},
                context="final candidate score",
            )
            if (
                raw["candidate_id"] != candidate.candidate_id
                or raw["skill_sha256"] != candidate.skill_sha256
            ):
                raise DataFirewallViolation("final candidate score binding does not match")
            scores.append(
                FinalCandidateScore(
                    candidate_id=candidate.candidate_id,
                    skill_sha256=candidate.skill_sha256,
                    score=require_finite_scalar(
                        raw["score"], context="final test score"
                    ),
                )
            )
        return FinalTestReport(plan_sha256=plan_sha256, scores=tuple(scores))


def _revalidate_plan(plan: FinalEvaluationPlan) -> FinalEvaluationPlan:
    """Rebuild every hash-bound value before granting test process access."""

    try:
        raw_policy = plan.policy
        raw_candidates = plan.candidates
    except AttributeError as error:
        raise DataFirewallViolation("final evaluation plan is incomplete") from error
    if type(raw_policy) is not FinalEvaluationPolicy:
        raise DataFirewallViolation("final evaluation requires exact frozen policy")
    if type(raw_candidates) is not tuple or not raw_candidates:
        raise DataFirewallViolation("final evaluation requires frozen candidates")
    try:
        policy = FinalEvaluationPolicy(
            test_split_id=raw_policy.test_split_id,
            split_manifest_sha256=raw_policy.split_manifest_sha256,
            controller_registry_sha256=raw_policy.controller_registry_sha256,
            runner_sha256=raw_policy.runner_sha256,
            scorer_sha256=raw_policy.scorer_sha256,
            harness_sha256=raw_policy.harness_sha256,
            environment_sha256=raw_policy.environment_sha256,
            profile_sha256=raw_policy.profile_sha256,
            local_code_commit=raw_policy.local_code_commit,
            protocol_id=raw_policy.protocol_id,
        )
    except AttributeError as error:
        raise DataFirewallViolation("final evaluation policy is incomplete") from error
    candidates: list[FrozenCandidate] = []
    for candidate in raw_candidates:
        if type(candidate) is not FrozenCandidate:
            raise DataFirewallViolation("final evaluation requires exact frozen candidates")
        try:
            candidates.append(
                FrozenCandidate(
                    candidate_id=candidate.candidate_id,
                    skill_text=candidate.skill_text,
                    skill_sha256=candidate.skill_sha256,
                )
            )
        except AttributeError as error:
            raise DataFirewallViolation("frozen candidate is incomplete") from error
    return FinalEvaluationPlan(candidates=tuple(candidates), policy=policy)
