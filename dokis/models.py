"""Pydantic v2 data models for Dokis."""

from __future__ import annotations

from datetime import date
from typing import Any, Literal

from pydantic import BaseModel, Field


class Chunk(BaseModel):
    """A single retrieved document chunk with its source URL.

    Args:
        content: The textual content of the chunk.
        source_url: The URL of the document this chunk came from. Required.
        metadata: Free-form bag for retriever-supplied extras (page numbers,
            chunk IDs, relevance scores, etc.).
    """

    content: str
    source_url: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class Claim(BaseModel):
    """An atomic claim extracted from an LLM response, with its provenance.

    Args:
        text: The sentence-level claim text.
        supported: Whether this claim is grounded in a retrieved chunk above
            the configured similarity threshold.
        confidence: Cosine similarity score against the best-matching chunk.
            Always populated - even when ``supported=False`` - so callers can
            tune thresholds without re-running inference.
        source_chunk: The chunk that best supports this claim, or ``None``
            when ``supported=False``.
        source_url: Shortcut for ``source_chunk.source_url``, or ``None``
            when ``supported=False``.
        freshness_status: Temporal trust status for the supporting source.
        source_date: Parsed publication/update date derived from chunk
            metadata when freshness policy is active.
        source_age_days: Source age in days when derivable.
    """

    text: str
    supported: bool
    confidence: float
    source_chunk: Chunk | None
    source_url: str | None
    freshness_status: Literal["fresh", "stale", "unknown", "not_applicable"] = (
        "not_applicable"
    )
    source_date: date | None = None
    source_age_days: int | None = None
    source_date_metadata_key: str | None = None


class BlockedSource(BaseModel):
    """Structured details for a source blocked by domain enforcement."""

    url: str
    domain: str | None = None
    reason: Literal[
        "domain_not_allowlisted",
        "malformed_source_url",
        "missing_source_url",
    ]


class ClaimVerdict(BaseModel):
    """Report-friendly summary of a claim-level provenance decision."""

    claim_text: str
    verdict: Literal["supported", "unsupported"]
    trust_status: Literal[
        "supported_fresh",
        "supported_stale",
        "supported_unknown_age",
        "unsupported",
    ] = "unsupported"
    freshness_status: Literal["fresh", "stale", "unknown", "not_applicable"] = (
        "not_applicable"
    )
    confidence: float
    supporting_url: str | None = None
    source_date: date | None = None
    source_age_days: int | None = None
    note: str | None = None


class SourceFreshness(BaseModel):
    """Structured freshness status for a unique source URL."""

    url: str
    status: Literal["fresh", "stale", "unknown", "not_applicable"]
    source_date: date | None = None
    age_days: int | None = None
    metadata_key: str | None = None
    note: str | None = None


class ProvenanceResult(BaseModel):
    """The outcome of a full provenance audit.

    Args:
        response: The original LLM response text that was audited.
        claims: Every atomic claim extracted from the response, each annotated
            with its support status and source provenance.
        compliance_rate: Fraction of claims that are grounded in a retrieved
            chunk (``supported_count / total_claims``). Always ``1.0`` when
            there are no claims.
        passed: ``True`` when ``compliance_rate >= min_citation_rate``.
        blocked_sources: URLs that were removed by the domain enforcer before
            the chunks reached the LLM. Preserved for backwards compatibility.
        blocked_source_details: Structured blocked-source records including
            URL, normalised domain, and block reason.
        claim_verdicts: Report-oriented claim-level decisions with explicit
            support verdicts and supporting URLs.
        policy_issues: Compact structured summary of whether blocked sources
            unsupported claims, stale support, and other trust issues were
            present in this audit result.
        has_blocked_sources: ``True`` when any source was blocked by policy.
        has_unsupported_claims: ``True`` when any extracted claim is
            unsupported by the filtered chunk set.
        freshness_enabled: ``True`` when temporal trust policy is active.
        freshness_passed: ``True`` when freshness policy did not fail.
        trust_passed: ``True`` when both support/compliance and freshness
            policy passed.
        domain: Optional domain label from the config (e.g. ``"oncology"``).
        min_citation_rate: The threshold used to compute ``passed``. Stored
            here so callers and exceptions can inspect it without needing the
            originating :class:`~dokis.config.Config` object.
        enforcement_mode: Configured runtime behavior for policy failure.
        enforcement_verdict: Final report-level enforcement outcome.
        raised_on_violation: ``True`` when the audit failed in enforce mode
            and the result was attached to a raised ComplianceViolation.
    """

    response: str
    claims: list[Claim]
    compliance_rate: float
    passed: bool
    blocked_sources: list[str]
    blocked_source_details: list[BlockedSource] = Field(default_factory=list)
    source_freshness_details: list[SourceFreshness] = Field(default_factory=list)
    claim_verdicts: list[ClaimVerdict] = Field(default_factory=list)
    policy_issues: list[
        Literal[
            "blocked_sources",
            "unsupported_claims",
            "stale_sources",
            "stale_supported_claims",
            "unknown_source_ages",
        ]
    ] = Field(default_factory=list)
    has_blocked_sources: bool = False
    has_unsupported_claims: bool = False
    has_stale_sources: bool = False
    has_stale_supported_claims: bool = False
    has_unknown_source_ages: bool = False
    freshness_enabled: bool = False
    freshness_passed: bool = True
    trust_passed: bool = True
    max_source_age_days: int | None = None
    stale_source_action: Literal["warn", "fail"] | None = None
    domain: str | None
    min_citation_rate: float
    enforcement_mode: Literal["audit", "guardrail", "enforce"] = "guardrail"
    enforcement_verdict: Literal[
        "passed",
        "audit_failed",
        "guardrail_failed",
        "enforce_raised",
    ] = "passed"
    raised_on_violation: bool = False

    @property
    def violations(self) -> list[Claim]:
        """Claims that are not grounded in any retrieved chunk.

        Returns:
            List of :class:`Claim` objects where ``supported=False``.
        """
        return [c for c in self.claims if not c.supported]

    @property
    def provenance_map(self) -> dict[str, str]:
        """Mapping of supported claim text to its source URL.

        Returns:
            Dict keyed by claim text, valued by the URL of the chunk that
            supports it. Unsupported claims are excluded.
        """
        return {
            c.text: c.source_url for c in self.claims if c.supported and c.source_url
        }

    @property
    def stale_claims(self) -> list[Claim]:
        """Supported claims whose evidence is stale under freshness policy."""
        return [c for c in self.claims if c.supported and c.freshness_status == "stale"]
