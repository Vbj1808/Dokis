"""Dokis exception types."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dokis.models import ProvenanceResult


class ComplianceViolation(Exception):
    """Raised when a provenance audit fails in ``enforce`` mode.

    Always carries the full :class:`~dokis.models.ProvenanceResult` so callers
    can inspect exactly which claims were unsupported.

    Args:
        result: The failed audit result.
    """

    def __init__(self, result: ProvenanceResult) -> None:
        self.result = result
        super().__init__(
            f"Dokis compliance check failed: {result.compliance_rate:.1%} grounded "
            f"(minimum required: {result.min_citation_rate:.1%}). "
            f"{len(result.violations)} unsupported claim(s). "
            f"Freshness passed: {result.freshness_passed}. "
            f"Trust passed: {result.trust_passed}. "
            f"Enforcement verdict: {result.enforcement_verdict}. "
            f"Policy issues: {', '.join(result.policy_issues) or 'none'}."
        )


class DomainViolation(Exception):
    """Raised by user code when a blocked domain should be treated as an error.

    Dokis internals never raise this automatically. It is exposed so pipeline
    authors can handle blocked sources as exceptions in their own orchestration
    layer.

    Args:
        url: The source URL that was blocked.
        allowed_domains: The domains that were on the allowlist.
    """

    def __init__(self, url: str, allowed_domains: list[str]) -> None:
        self.url = url
        self.allowed_domains = allowed_domains
        super().__init__(
            f"Source URL '{url}' is not in the allowed domains: {allowed_domains}"
        )
