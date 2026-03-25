"""Full test suite for ProvenanceMiddleware."""

import pytest
import pytest_asyncio  # noqa: F401 — ensures pytest-asyncio is available

from dokis.config import Config
from dokis.exceptions import ComplianceViolation
from dokis.middleware import ProvenanceMiddleware
from dokis.models import Chunk, ProvenanceResult

# A response whose claims are clearly ungrounded (topic mismatch) to force
# a failing compliance rate when combined with a high min_citation_rate.
_UNGROUNDED_RESPONSE = (
    "Cryptocurrency prices have been extremely volatile in recent financial markets. "
    "Blockchain technology enables decentralised ledger systems without intermediaries."
)


def test_middleware_audit_returns_provenance_result(
    sample_chunks: list[Chunk],
    permissive_config: Config,
    grounded_response: str,
) -> None:
    middleware = ProvenanceMiddleware(permissive_config)
    result = middleware.audit("test query", sample_chunks, grounded_response)
    assert isinstance(result, ProvenanceResult)


def test_middleware_audit_compliance_rate_between_zero_and_one(
    sample_chunks: list[Chunk],
    semantic_config: Config,
    grounded_response: str,
) -> None:
    middleware = ProvenanceMiddleware(semantic_config)
    result = middleware.audit("test query", sample_chunks, grounded_response)
    assert 0.0 <= result.compliance_rate <= 1.0


def test_middleware_audit_populates_min_citation_rate(
    sample_chunks: list[Chunk],
    strict_config: Config,
    grounded_response: str,
) -> None:
    middleware = ProvenanceMiddleware(strict_config)
    result = middleware.audit("test query", sample_chunks, grounded_response)
    assert result.min_citation_rate == strict_config.min_citation_rate


def test_middleware_audit_populates_blocked_sources(
    sample_chunks: list[Chunk],
    strict_config: Config,
    grounded_response: str,
) -> None:
    middleware = ProvenanceMiddleware(strict_config)
    result = middleware.audit("test query", sample_chunks, grounded_response)
    assert "https://discountpharma.biz/meds" in result.blocked_sources


def test_middleware_audit_populates_provenance_map(
    sample_chunks: list[Chunk],
    semantic_config: Config,
    grounded_response: str,
) -> None:
    middleware = ProvenanceMiddleware(semantic_config)
    result = middleware.audit("test query", sample_chunks, grounded_response)
    # At least some supported claims should appear in the provenance map.
    assert isinstance(result.provenance_map, dict)
    assert len(result.provenance_map) > 0


def test_middleware_filter_removes_blocked_chunks(
    sample_chunks: list[Chunk],
    strict_config: Config,
) -> None:
    middleware = ProvenanceMiddleware(strict_config)
    clean = middleware.filter(sample_chunks)
    urls = {c.source_url for c in clean}
    assert "https://discountpharma.biz/meds" not in urls
    assert len(clean) == 2


def test_middleware_filter_passes_allowlisted_chunks(
    sample_chunks: list[Chunk],
    strict_config: Config,
) -> None:
    middleware = ProvenanceMiddleware(strict_config)
    clean = middleware.filter(sample_chunks)
    urls = {c.source_url for c in clean}
    assert "https://pubmed.ncbi.nlm.nih.gov/12345" in urls
    assert "https://cochrane.org/review/67890" in urls


def test_middleware_raises_compliance_violation_when_configured(
    sample_chunks: list[Chunk],
) -> None:
    # min_citation_rate=1.0 with an ungrounded response guarantees a failure.
    config = Config(
        min_citation_rate=1.0,
        fail_on_violation=True,
    )
    middleware = ProvenanceMiddleware(config)
    with pytest.raises(ComplianceViolation) as exc_info:
        middleware.audit("test query", sample_chunks, _UNGROUNDED_RESPONSE)
    assert isinstance(exc_info.value.result, ProvenanceResult)


def test_middleware_no_raise_when_fail_on_violation_false(
    sample_chunks: list[Chunk],
) -> None:
    config = Config(min_citation_rate=1.0, fail_on_violation=False)
    middleware = ProvenanceMiddleware(config)
    # Must return a result without raising, even when compliance rate is low.
    result = middleware.audit("test query", sample_chunks, _UNGROUNDED_RESPONSE)
    assert isinstance(result, ProvenanceResult)
    assert result.passed is False


@pytest.mark.asyncio
async def test_middleware_aaudit_matches_sync_audit(
    sample_chunks: list[Chunk],
    semantic_config: Config,
    grounded_response: str,
) -> None:
    middleware = ProvenanceMiddleware(semantic_config)
    sync_result = middleware.audit("test query", sample_chunks, grounded_response)
    async_result = await middleware.aaudit("test query", sample_chunks, grounded_response)
    assert async_result.compliance_rate == sync_result.compliance_rate
    assert async_result.passed == sync_result.passed
    assert async_result.blocked_sources == sync_result.blocked_sources
    assert len(async_result.claims) == len(sync_result.claims)


def test_middleware_bm25_audit_returns_provenance_result(
    sample_chunks: list[Chunk],
) -> None:
    """BM25 default path must return a valid ProvenanceResult."""
    config = Config(matcher="bm25", min_citation_rate=0.0)
    middleware = ProvenanceMiddleware(config)
    response = (
        "Aspirin inhibits the COX enzymes reducing fever. "
        "Ibuprofen is a nonsteroidal anti-inflammatory medication."
    )
    result = middleware.audit("test query", sample_chunks, response)
    assert isinstance(result, ProvenanceResult)
    assert 0.0 <= result.compliance_rate <= 1.0


def test_middleware_bm25_ungrounded_response_fails(
    sample_chunks: list[Chunk],
) -> None:
    """BM25 path must mark off-topic claims as unsupported."""
    config = Config(
        matcher="bm25",
        min_citation_rate=1.0,
        fail_on_violation=False,
    )
    middleware = ProvenanceMiddleware(config)
    # Crypto/blockchain has zero token overlap with medical chunks.
    response = (
        "Cryptocurrency blockchain ledgers enable decentralised "
        "trustless transactions without financial intermediaries."
    )
    result = middleware.audit("test query", sample_chunks, response)
    assert result.passed is False
    assert result.compliance_rate == 0.0


def test_middleware_bm25_grounded_response_passes(
    sample_chunks: list[Chunk],
) -> None:
    """BM25 path must mark on-topic claims as supported."""
    config = Config(
        matcher="bm25",
        min_citation_rate=0.5,
        claim_threshold=0.3,
    )
    middleware = ProvenanceMiddleware(config)
    response = (
        "Aspirin reduces fever by inhibiting the COX enzymes directly. "
        "Ibuprofen is a nonsteroidal anti-inflammatory drug for pain."
    )
    result = middleware.audit("test query", sample_chunks, response)
    assert result.compliance_rate > 0.0


@pytest.mark.asyncio
async def test_aaudit_does_not_emit_deprecation_warning(
    sample_chunks: list[Chunk],
    grounded_response: str,
) -> None:
    """aaudit must not emit DeprecationWarning for get_event_loop."""
    import warnings

    config = Config(matcher="semantic", min_citation_rate=0.0)
    middleware = ProvenanceMiddleware(config)
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        result = await middleware.aaudit(
            "test query", sample_chunks, grounded_response
        )
    assert isinstance(result, ProvenanceResult)


def test_middleware_violations_property_returns_unsupported_claims(
    sample_chunks: list[Chunk],
) -> None:
    config = Config(min_citation_rate=0.0)
    middleware = ProvenanceMiddleware(config)
    result = middleware.audit("test query", sample_chunks, _UNGROUNDED_RESPONSE)
    violations = result.violations
    assert all(not c.supported for c in violations)
    assert len(violations) == sum(1 for c in result.claims if not c.supported)
