"""Dokis provenance middleware - main pipeline orchestrator."""

from __future__ import annotations

from typing import Literal

from dokis.config import Config
from dokis.core.enforcer import DomainEnforcer
from dokis.core.extractor import ClaimExtractor
from dokis.core.matcher import ClaimMatcher
from dokis.core.scorer import ComplianceScorer
from dokis.exceptions import ComplianceViolation
from dokis.models import (
    BlockedSource,
    Chunk,
    Claim,
    ClaimVerdict,
    ProvenanceResult,
)


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

        pg = ProvenanceMiddleware(Config.from_yaml("provenance.toml"))
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
           the configured matcher.
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
            grounding information, blocked-source reporting, policy issues,
            and enforcement metadata.

        Raises:
            :class:`~dokis.exceptions.ComplianceViolation`: When
                ``config.enforcement_mode="enforce"`` and the result does not
                pass. The exception always carries the full result.
        """
        # 1. Enforce domain allowlist.
        clean_chunks, blocked_source_details = self.enforcer.inspect(chunks)

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
            blocked_sources=[entry.url for entry in blocked_source_details],
            blocked_source_details=blocked_source_details,
            claim_verdicts=self._build_claim_verdicts(claims),
            policy_issues=self._build_policy_issues(
                blocked_source_details=blocked_source_details,
                claims=claims,
            ),
            has_blocked_sources=bool(blocked_source_details),
            has_unsupported_claims=any(not claim.supported for claim in claims),
            domain=self.config.domain,
            min_citation_rate=self.config.min_citation_rate,
            enforcement_mode=self.config.enforcement_mode or "guardrail",
            enforcement_verdict=self._resolve_enforcement_verdict(passed),
        )

        if self.config.enforcement_mode == "enforce" and not result.passed:
            result.raised_on_violation = True
            result.enforcement_verdict = "enforce_raised"
            raise ComplianceViolation(result)

        return result

    def _build_claim_verdicts(self, claims: list[Claim]) -> list[ClaimVerdict]:
        """Return a compact, report-oriented view of claim support status."""
        verdicts: list[ClaimVerdict] = []
        for claim in claims:
            supported = claim.supported and claim.source_url is not None
            verdicts.append(
                ClaimVerdict(
                    claim_text=claim.text,
                    verdict="supported" if supported else "unsupported",
                    confidence=claim.confidence,
                    supporting_url=claim.source_url if supported else None,
                    note=(
                        None
                        if supported
                        else "No supporting source met the configured threshold."
                    ),
                )
            )
        return verdicts

    def _build_policy_issues(
        self,
        *,
        blocked_source_details: list[BlockedSource],
        claims: list[Claim],
    ) -> list[Literal["blocked_sources", "unsupported_claims"]]:
        """Return a compact structured summary of audit-level policy issues."""
        issues: list[Literal["blocked_sources", "unsupported_claims"]] = []
        if blocked_source_details:
            issues.append("blocked_sources")
        if any(not claim.supported for claim in claims):
            issues.append("unsupported_claims")
        return issues

    def _resolve_enforcement_verdict(
        self,
        passed: bool,
    ) -> Literal["passed", "audit_failed", "guardrail_failed"]:
        """Return a single explicit enforcement outcome for the audit."""
        if passed:
            return "passed"
        if self.config.enforcement_mode == "audit":
            return "audit_failed"
        return "guardrail_failed"

    async def afilter(self, chunks: list[Chunk]) -> list[Chunk]:
        """Async version of :meth:`filter`.

        Provides an awaitable wrapper for async call sites while reusing the
        synchronous enforcement path.

        Args:
            chunks: Raw chunks from your retriever.

        Returns:
            Chunks whose source domain is in ``config.allowed_domains``.
        """
        return self.filter(chunks)

    async def aaudit(
        self,
        query: str,
        chunks: list[Chunk],
        response: str,
    ) -> ProvenanceResult:
        """Async version of :meth:`audit`.

        Provides an awaitable wrapper for async call sites. The result is
        identical to what :meth:`audit` would return for the same arguments.

        Args:
            query: The original user query.
            chunks: Raw retrieved chunks.
            response: The LLM-generated response text to audit.

        Returns:
            A :class:`~dokis.models.ProvenanceResult` - same as
            :meth:`audit` for identical input.

        Raises:
            :class:`~dokis.exceptions.ComplianceViolation`: When
                ``config.enforcement_mode="enforce"`` and the result does not
                pass.
        """
        return self.audit(query, chunks, response)
