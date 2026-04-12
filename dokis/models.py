"""Pydantic v2 data models for Dokis."""

from __future__ import annotations

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
    """

    text: str
    supported: bool
    confidence: float
    source_chunk: Chunk | None
    source_url: str | None


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
    confidence: float
    supporting_url: str | None = None
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
            and/or unsupported claims were present in this audit result.
        has_blocked_sources: ``True`` when any source was blocked by policy.
        has_unsupported_claims: ``True`` when any extracted claim is
            unsupported by the filtered chunk set.
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
    claim_verdicts: list[ClaimVerdict] = Field(default_factory=list)
    policy_issues: list[Literal["blocked_sources", "unsupported_claims"]] = (
        Field(default_factory=list)
    )
    has_blocked_sources: bool = False
    has_unsupported_claims: bool = False
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
            c.text: c.source_url
            for c in self.claims
            if c.supported and c.source_url
        }
