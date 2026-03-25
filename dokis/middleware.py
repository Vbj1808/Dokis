"""Dokis provenance middleware - main pipeline orchestrator."""

from __future__ import annotations

import asyncio
import functools
from typing import cast

from dokis.config import Config
from dokis.core.enforcer import DomainEnforcer
from dokis.core.extractor import ClaimExtractor
from dokis.core.matcher import ClaimMatcher
from dokis.core.scorer import ComplianceScorer
from dokis.exceptions import ComplianceViolation
from dokis.models import Chunk, ProvenanceResult


class ProvenanceMiddleware:
    """Orchestrates the full Dokis provenance pipeline.

    Owns one instance of each core module (enforcer, extractor, matcher,
    scorer) created eagerly at construction time - never lazily.

    Pipeline order for :meth:`audit`: enforcer → extractor → matcher → scorer.
    No LLM calls. No network requests. Deterministic for identical input.

    Args:
        config: Runtime configuration for the pipeline.

    Example::

        from dokis import ProvenanceMiddleware, Config

        pg = ProvenanceMiddleware(Config.from_yaml("provenance.yaml"))
        chunks   = pg.filter(retriever.get_relevant_documents(query))
        response = llm.invoke(build_prompt(query, chunks))
        result   = pg.audit(query, chunks, response)
    """

    def __init__(self, config: Config) -> None:
        self.config = config
        self.enforcer = DomainEnforcer(config)
        self.extractor = ClaimExtractor(config)
        self.matcher = ClaimMatcher(config)
        self.scorer = ComplianceScorer(config)

    def filter(self, chunks: list[Chunk]) -> list[Chunk]:
        """Remove chunks whose source URL is not on the allowlist.

        A convenience wrapper around :class:`~dokis.core.enforcer.DomainEnforcer`
        for the common case where only the clean chunks are needed. To inspect
        which URLs were blocked, call :meth:`audit` instead and read
        :attr:`~dokis.models.ProvenanceResult.blocked_sources`.

        If ``config.allowed_domains`` is empty, all chunks pass through
        unmodified.

        Args:
            chunks: Raw chunks from your retriever.

        Returns:
            Chunks whose source domain is in ``config.allowed_domains``.
        """
        clean, _ = self.enforcer.filter(chunks)
        return clean

    def audit(
        self,
        query: str,
        chunks: list[Chunk],
        response: str,
    ) -> ProvenanceResult:
        """Audit a generated response for source provenance.

        Runs the full pipeline:

        1. **Enforcer** - filter ``chunks`` to only allowed domains and
           record ``blocked_sources``.
        2. **Extractor** - split ``response`` into atomic claim sentences.
        3. **Matcher** - match each claim to its best-supporting chunk via
           cosine similarity.
        4. **Scorer** - compute the overall ``compliance_rate`` and
           ``passed`` verdict.

        No LLM calls are made in the default configuration. Output is
        deterministic for identical input.

        Args:
            query: The original user query (reserved for future use in
                query-aware matching; passed through to
                :class:`~dokis.models.ProvenanceResult` context).
            chunks: Raw retrieved chunks. The enforcer runs on these even if
                :meth:`filter` was already called - the operation is
                idempotent for clean chunks.
            response: The LLM-generated response text to audit.

        Returns:
            A :class:`~dokis.models.ProvenanceResult` with per-claim
            grounding information, compliance rate, and blocked source URLs.

        Raises:
            :class:`~dokis.exceptions.ComplianceViolation`: When
                ``config.fail_on_violation=True`` and the result does not
                pass. The exception always carries the full result.
        """
        # 1. Enforce domain allowlist.
        clean_chunks, blocked_sources = self.enforcer.filter(chunks)

        # 2. Extract atomic claims from the response.
        claim_texts = self.extractor.extract(response)

        # 3. Match claims to supporting chunks.
        claims = self.matcher.match(claim_texts, clean_chunks)

        # 4. Score compliance.
        compliance_rate, passed = self.scorer.score(claims)

        result = ProvenanceResult(
            response=response,
            claims=claims,
            compliance_rate=compliance_rate,
            passed=passed,
            blocked_sources=blocked_sources,
            domain=self.config.domain,
            min_citation_rate=self.config.min_citation_rate,
        )

        if self.config.fail_on_violation and not result.passed:
            raise ComplianceViolation(result)

        return result

    async def afilter(self, chunks: list[Chunk]) -> list[Chunk]:
        """Async version of :meth:`filter`.

        Runs :meth:`filter` in a thread-pool executor so the event loop
        is not blocked by I/O or CPU work inside the enforcer.

        Args:
            chunks: Raw chunks from your retriever.

        Returns:
            Chunks whose source domain is in ``config.allowed_domains``.
        """
        loop = asyncio.get_running_loop()
        return cast(
            list[Chunk],
            await loop.run_in_executor(None, self.filter, chunks),
        )

    async def aaudit(
        self,
        query: str,
        chunks: list[Chunk],
        response: str,
    ) -> ProvenanceResult:
        """Async version of :meth:`audit`.

        Runs the full synchronous pipeline in a thread-pool executor so
        CPU-bound embedding work does not block the event loop.

        The result is identical to what :meth:`audit` would return for
        the same arguments.

        Args:
            query: The original user query.
            chunks: Raw retrieved chunks.
            response: The LLM-generated response text to audit.

        Returns:
            A :class:`~dokis.models.ProvenanceResult` - same as
            :meth:`audit` for identical input.

        Raises:
            :class:`~dokis.exceptions.ComplianceViolation`: When
                ``config.fail_on_violation=True`` and the result does not
                pass.
        """
        loop = asyncio.get_running_loop()
        return cast(
            ProvenanceResult,
            await loop.run_in_executor(
                None,
                functools.partial(self.audit, query, chunks, response),
            ),
        )
