"""Pydantic v2 data models for Dokis."""

from __future__ import annotations

from typing import Any

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
            the chunks reached the LLM.
        domain: Optional domain label from the config (e.g. ``"oncology"``).
        min_citation_rate: The threshold used to compute ``passed``. Stored
            here so callers and exceptions can inspect it without needing the
            originating :class:`~dokis.config.Config` object.
    """

    response: str
    claims: list[Claim]
    compliance_rate: float
    passed: bool
    blocked_sources: list[str]
    domain: str | None
    min_citation_rate: float

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
