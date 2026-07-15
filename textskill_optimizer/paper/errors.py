"""Shared fail-closed errors for paper data seams."""


class DataFirewallViolation(ValueError):
    """Raised when non-authorized data crosses a paper controller seam."""
