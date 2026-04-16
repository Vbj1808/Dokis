"""Dokis provenance middleware - main pipeline orchestrator."""

from __future__ import annotations

from typing import Literal

from dokis.config import Config
from dokis.core.enforcer import DomainEnforcer
from dokis.core.extractor import ClaimExtractor
from dokis.core.freshness import FreshnessEvaluator
from dokis.core.matcher import ClaimMatcher
from dokis.core.scorer import ComplianceScorer
from dokis.exceptions import ComplianceViolation
from dokis.models import (
    BlockedSource,
    Chunk,
    Claim,
    ClaimVerdict,
    ProvenanceResult,
    SourceFreshness,
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
        self.freshness = FreshnessEvaluator(config)
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
        source_freshness_details = self._build_source_freshness_details(clean_chunks)
        claims = self._apply_freshness(claims, source_freshness_details)

        # 4. Score compliance.
        compliance_rate, passed = self.scorer.score(claims)
        freshness_passed = self._freshness_passed(claims)
        trust_passed = passed and freshness_passed

        result = ProvenanceResult(
            response=response,
            claims=claims,
            compliance_rate=compliance_rate,
            passed=passed,
            blocked_sources=[entry.url for entry in blocked_source_details],
            blocked_source_details=blocked_source_details,
            source_freshness_details=source_freshness_details,
            claim_verdicts=self._build_claim_verdicts(claims),
            policy_issues=self._build_policy_issues(
                blocked_source_details=blocked_source_details,
                claims=claims,
                source_freshness_details=source_freshness_details,
            ),
            has_blocked_sources=bool(blocked_source_details),
            has_unsupported_claims=any(not claim.supported for claim in claims),
            has_stale_sources=any(
                detail.status == "stale" for detail in source_freshness_details
            ),
            has_stale_supported_claims=any(
                claim.supported and claim.freshness_status == "stale"
                for claim in claims
            ),
            has_unknown_source_ages=any(
                detail.status == "unknown" for detail in source_freshness_details
            ),
            freshness_enabled=self.config.max_source_age_days is not None,
            freshness_passed=freshness_passed,
            trust_passed=trust_passed,
            max_source_age_days=self.config.max_source_age_days,
            stale_source_action=(
                self.config.stale_source_action
                if self.config.max_source_age_days is not None
                else None
            ),
            domain=self.config.domain,
            min_citation_rate=self.config.min_citation_rate,
            enforcement_mode=self.config.enforcement_mode or "guardrail",
            enforcement_verdict=self._resolve_enforcement_verdict(trust_passed),
        )

        if self.config.enforcement_mode == "enforce" and not result.trust_passed:
            result.raised_on_violation = True
            result.enforcement_verdict = "enforce_raised"
            raise ComplianceViolation(result)

        return result

    def _build_claim_verdicts(self, claims: list[Claim]) -> list[ClaimVerdict]:
        """Return a compact, report-oriented view of claim support status."""
        verdicts: list[ClaimVerdict] = []
        for claim in claims:
            supported = claim.supported and claim.source_url is not None
            trust_status = self._claim_trust_status(claim)
            verdicts.append(
                ClaimVerdict(
                    claim_text=claim.text,
                    verdict="supported" if supported else "unsupported",
                    trust_status=trust_status,
                    freshness_status=claim.freshness_status,
                    confidence=claim.confidence,
                    supporting_url=claim.source_url if supported else None,
                    source_date=claim.source_date,
                    source_age_days=claim.source_age_days,
                    note=(self._claim_note(claim, supported)),
                )
            )
        return verdicts

    def _build_policy_issues(
        self,
        *,
        blocked_source_details: list[BlockedSource],
        claims: list[Claim],
        source_freshness_details: list[SourceFreshness],
    ) -> list[
        Literal[
            "blocked_sources",
            "unsupported_claims",
            "stale_sources",
            "stale_supported_claims",
            "unknown_source_ages",
        ]
    ]:
        """Return a compact structured summary of audit-level policy issues."""
        issues: list[
            Literal[
                "blocked_sources",
                "unsupported_claims",
                "stale_sources",
                "stale_supported_claims",
                "unknown_source_ages",
            ]
        ] = []
        if blocked_source_details:
            issues.append("blocked_sources")
        if any(not claim.supported for claim in claims):
            issues.append("unsupported_claims")
        if any(detail.status == "stale" for detail in source_freshness_details):
            issues.append("stale_sources")
        if any(
            claim.supported and claim.freshness_status == "stale" for claim in claims
        ):
            issues.append("stale_supported_claims")
        if any(detail.status == "unknown" for detail in source_freshness_details):
            issues.append("unknown_source_ages")
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

    def _build_source_freshness_details(
        self,
        chunks: list[Chunk],
    ) -> list[SourceFreshness]:
        """Return one freshness record per unique allowed source URL."""
        details: list[SourceFreshness] = []
        seen: set[str] = set()
        for chunk in chunks:
            if chunk.source_url in seen:
                continue
            seen.add(chunk.source_url)
            assessment = self.freshness.assess(chunk)
            details.append(
                SourceFreshness(
                    url=chunk.source_url,
                    status=assessment.status,
                    source_date=assessment.source_date,
                    age_days=assessment.age_days,
                    metadata_key=assessment.metadata_key,
                    note=assessment.note,
                )
            )
        return details

    def _apply_freshness(
        self,
        claims: list[Claim],
        source_freshness_details: list[SourceFreshness],
    ) -> list[Claim]:
        """Annotate supported claims with freshness details from their source."""
        by_url = {detail.url: detail for detail in source_freshness_details}
        updated: list[Claim] = []
        for claim in claims:
            if not claim.supported or claim.source_url is None:
                updated.append(claim)
                continue
            detail = by_url.get(claim.source_url)
            if detail is None:
                updated.append(claim)
                continue
            updated.append(
                claim.model_copy(
                    update={
                        "freshness_status": detail.status,
                        "source_date": detail.source_date,
                        "source_age_days": detail.age_days,
                        "source_date_metadata_key": detail.metadata_key,
                    }
                )
            )
        return updated

    def _freshness_passed(self, claims: list[Claim]) -> bool:
        """Return whether freshness policy failed trust."""
        if self.config.max_source_age_days is None:
            return True
        if self.config.stale_source_action != "fail":
            return True
        return not any(
            claim.supported and claim.freshness_status == "stale" for claim in claims
        )

    def _claim_trust_status(
        self,
        claim: Claim,
    ) -> Literal[
        "supported_fresh",
        "supported_stale",
        "supported_unknown_age",
        "unsupported",
    ]:
        if not claim.supported:
            return "unsupported"
        if claim.freshness_status == "stale":
            return "supported_stale"
        if claim.freshness_status == "unknown":
            return "supported_unknown_age"
        return "supported_fresh"

    def _claim_note(self, claim: Claim, supported: bool) -> str | None:
        """Return a compact claim-level explanation for report output."""
        if not supported:
            return "No supporting source met the configured threshold."
        if claim.freshness_status == "stale":
            assert claim.source_age_days is not None
            assert self.config.max_source_age_days is not None
            return (
                f"Supporting source is stale: {claim.source_age_days} days old "
                f"(max allowed: {self.config.max_source_age_days})."
            )
        if claim.freshness_status == "unknown":
            return "Supporting source age could not be determined from chunk metadata."
        return None

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
