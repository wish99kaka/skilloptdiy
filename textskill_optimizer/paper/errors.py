"""Shared fail-closed errors for paper data seams."""

import re


class DataFirewallViolation(ValueError):
    """Raised when non-authorized data crosses a paper controller seam."""


class SkillContractViolation(ValueError):
    """Raised when a learned skill negates an immutable rollout contract."""

    def __init__(self, code: str, message: str) -> None:
        if type(code) is not str or not code.strip():
            raise ValueError("skill contract violation requires a code")
        if type(message) is not str or not message.strip():
            raise ValueError("skill contract violation requires a message")
        super().__init__(message)
        self.code = code


class OptimizerProviderError(RuntimeError):
    """A classified external-optimizer transport or response-contract failure."""

    def __init__(self, code: str, message: str) -> None:
        if (
            type(code) is not str
            or re.fullmatch(r"optimizer_[a-z0-9_]+", code) is None
        ):
            raise ValueError("optimizer provider error requires a stable code")
        if type(message) is not str or not message.strip():
            raise ValueError("optimizer provider error requires a message")
        super().__init__(message)
        self.code = code
