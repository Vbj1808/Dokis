"""Full test suite for ClaimMatcher."""

import pytest

from dokis.config import Config
from dokis.core.matcher import ClaimMatcher, _MAX_CHUNKS, _MAX_CLAIMS
from dokis.models import Chunk


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def semantic_matcher() -> ClaimMatcher:
    return ClaimMatcher(Config(matcher="semantic", claim_threshold=0.72))


@pytest.fixture
def bm25_matcher() -> ClaimMatcher:
    return ClaimMatcher(Config(matcher="bm25", claim_threshold=0.72))


@pytest.fixture
def aspirin_chunk() -> Chunk:
    return Chunk(
        content="Aspirin reduces fever by inhibiting COX enzymes.",
        source_url="https://pubmed.ncbi.nlm.nih.gov/12345",
    )


@pytest.fixture
def ibuprofen_chunk() -> Chunk:
    return Chunk(
        content="Ibuprofen is a nonsteroidal anti-inflammatory drug.",
        source_url="https://cochrane.org/review/67890",
    )


# ---------------------------------------------------------------------------
# Semantic matcher tests
# ---------------------------------------------------------------------------


def test_matcher_returns_supported_true_above_threshold(
    semantic_matcher: ClaimMatcher, aspirin_chunk: Chunk
) -> None:
    claims = ["Aspirin inhibits the COX enzymes and reduces fever."]
    results = semantic_matcher.match(claims, [aspirin_chunk])
    assert len(results) == 1
    assert results[0].supported is True


def test_matcher_returns_supported_false_below_threshold(
    aspirin_chunk: Chunk, ibuprofen_chunk: Chunk
) -> None:
    # threshold=1.0 means nothing can match.
    config = Config(matcher="semantic", claim_threshold=1.0)
    strict_matcher = ClaimMatcher(config)
    claims = ["Aspirin inhibits the COX enzymes and reduces fever."]
    results = strict_matcher.match(claims, [aspirin_chunk, ibuprofen_chunk])
    assert len(results) == 1
    assert results[0].supported is False


def test_matcher_confidence_always_populated(aspirin_chunk: Chunk) -> None:
    config = Config(matcher="semantic", claim_threshold=1.0)
    strict_matcher = ClaimMatcher(config)
    claims = ["Aspirin inhibits the COX enzymes and reduces fever."]
    results = strict_matcher.match(claims, [aspirin_chunk])
    assert results[0].confidence is not None
    assert isinstance(results[0].confidence, float)
    assert 0.0 <= results[0].confidence <= 1.0


def test_matcher_source_url_populated_on_supported_claim(
    semantic_matcher: ClaimMatcher, aspirin_chunk: Chunk
) -> None:
    claims = ["Aspirin inhibits the COX enzymes and reduces fever."]
    results = semantic_matcher.match(claims, [aspirin_chunk])
    assert results[0].supported is True
    assert results[0].source_url == "https://pubmed.ncbi.nlm.nih.gov/12345"
    assert results[0].source_chunk == aspirin_chunk


def test_matcher_source_url_none_on_unsupported_claim(aspirin_chunk: Chunk) -> None:
    config = Config(matcher="semantic", claim_threshold=1.0)
    strict_matcher = ClaimMatcher(config)
    claims = ["Aspirin inhibits the COX enzymes and reduces fever."]
    results = strict_matcher.match(claims, [aspirin_chunk])
    assert results[0].supported is False
    assert results[0].source_url is None
    assert results[0].source_chunk is None


def test_matcher_empty_claims_returns_empty_list(
    semantic_matcher: ClaimMatcher, aspirin_chunk: Chunk
) -> None:
    results = semantic_matcher.match([], [aspirin_chunk])
    assert results == []


def test_matcher_empty_chunks_returns_empty_list(semantic_matcher: ClaimMatcher) -> None:
    claims = ["Aspirin inhibits the COX enzymes and reduces fever."]
    results = semantic_matcher.match(claims, [])
    assert results == []


# ---------------------------------------------------------------------------
# BM25 matcher tests
# ---------------------------------------------------------------------------


def test_bm25_matcher_returns_supported_true_above_threshold(
    bm25_matcher: ClaimMatcher, aspirin_chunk: Chunk
) -> None:
    claims = ["Aspirin inhibits the COX enzymes and reduces fever."]
    results = bm25_matcher.match(claims, [aspirin_chunk])
    assert len(results) == 1
    assert results[0].supported is True


def test_bm25_matcher_returns_supported_false_below_threshold(
    aspirin_chunk: Chunk, ibuprofen_chunk: Chunk
) -> None:
    """BM25 claims with no content-word overlap are unsupported (confidence=0.0).

    After FIX 1, raw scores below _MIN_BM25_RAW_SCORE are treated as no-match
    regardless of claim_threshold, so vocabulary-disjoint claims are unsupported.
    """
    config = Config(matcher="bm25", claim_threshold=0.5)
    strict_matcher = ClaimMatcher(config)
    # Claim vocabulary is entirely disjoint from both chunks → all BM25 scores 0
    claims = ["Quantum entanglement photon spin correlation measurement apparatus."]
    results = strict_matcher.match(claims, [aspirin_chunk, ibuprofen_chunk])
    assert len(results) == 1
    assert results[0].supported is False


def test_bm25_matcher_confidence_always_populated(aspirin_chunk: Chunk) -> None:
    config = Config(matcher="bm25", claim_threshold=1.0)
    strict_matcher = ClaimMatcher(config)
    claims = ["Aspirin inhibits the COX enzymes and reduces fever."]
    results = strict_matcher.match(claims, [aspirin_chunk])
    assert results[0].confidence is not None
    assert isinstance(results[0].confidence, float)
    assert 0.0 <= results[0].confidence <= 1.0


def test_bm25_matcher_source_url_populated_on_supported_claim(
    bm25_matcher: ClaimMatcher, aspirin_chunk: Chunk
) -> None:
    claims = ["Aspirin inhibits the COX enzymes and reduces fever."]
    results = bm25_matcher.match(claims, [aspirin_chunk])
    assert results[0].supported is True
    assert results[0].source_url == "https://pubmed.ncbi.nlm.nih.gov/12345"
    assert results[0].source_chunk == aspirin_chunk


def test_bm25_matcher_no_token_overlap_yields_zero_confidence(
    bm25_matcher: ClaimMatcher,
) -> None:
    """A claim with zero BM25 token overlap must get confidence=0.0, supported=False."""
    chunk = Chunk(
        content="Ibuprofen is a nonsteroidal anti-inflammatory drug.",
        source_url="https://cochrane.org/review/67890",
    )
    claims = ["Quantum entanglement photon spin correlation measurement."]
    results = bm25_matcher.match(claims, [chunk])
    assert results[0].supported is False
    assert results[0].confidence == pytest.approx(0.0)
    assert results[0].source_url is None


def test_bm25_matcher_selects_correct_chunk_from_multiple(
    bm25_matcher: ClaimMatcher, aspirin_chunk: Chunk, ibuprofen_chunk: Chunk
) -> None:
    """BM25 must pick the aspirin chunk for an aspirin claim, not the ibuprofen chunk."""
    claims = ["Aspirin inhibits the COX enzymes and reduces fever."]
    results = bm25_matcher.match(claims, [aspirin_chunk, ibuprofen_chunk])
    assert results[0].supported is True
    assert results[0].source_chunk == aspirin_chunk


def test_bm25_matcher_empty_claims_returns_empty_list(
    bm25_matcher: ClaimMatcher, aspirin_chunk: Chunk
) -> None:
    results = bm25_matcher.match([], [aspirin_chunk])
    assert results == []


def test_bm25_matcher_empty_chunks_returns_empty_list(bm25_matcher: ClaimMatcher) -> None:
    claims = ["Aspirin inhibits the COX enzymes and reduces fever."]
    results = bm25_matcher.match(claims, [])
    assert results == []


# ---------------------------------------------------------------------------
# Default config and guard tests (strategy-agnostic)
# ---------------------------------------------------------------------------


def test_default_config_uses_bm25() -> None:
    """Config() must default to matcher='bm25' — zero cold start."""
    config = Config()
    assert config.matcher == "bm25"
    matcher = ClaimMatcher(config)
    chunk = Chunk(
        content="Aspirin reduces fever by inhibiting COX enzymes.",
        source_url="https://pubmed.ncbi.nlm.nih.gov/12345",
    )
    results = matcher.match(
        ["Aspirin inhibits the COX enzymes and reduces fever."], [chunk]
    )
    assert len(results) == 1


def test_matcher_raises_on_too_many_chunks() -> None:
    """Passing more than _MAX_CHUNKS chunks must raise ValueError."""
    matcher = ClaimMatcher(Config())
    oversized = [
        Chunk(content="text", source_url="https://example.com")
        for _ in range(_MAX_CHUNKS + 1)
    ]
    with pytest.raises(ValueError, match="chunk count"):
        matcher.match(["Some claim that is long enough to count."], oversized)


def test_matcher_raises_on_too_many_claims() -> None:
    """Passing more than _MAX_CLAIMS claims must raise ValueError."""
    matcher = ClaimMatcher(Config())
    chunk = Chunk(content="text", source_url="https://example.com")
    oversized_claims = [
        "This is a claim that is definitely longer than eight words."
    ] * (_MAX_CLAIMS + 1)
    with pytest.raises(ValueError, match="claim count"):
        matcher.match(oversized_claims, [chunk])


def test_semantic_matcher_raises_import_error_without_package(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ClaimMatcher(matcher='semantic') must raise ImportError when sentence-transformers is absent."""
    import dokis.core.matcher as matcher_module

    monkeypatch.setattr(matcher_module, "_SEMANTIC_AVAILABLE", False)
    with pytest.raises(ImportError, match="sentence-transformers"):
        ClaimMatcher(Config(matcher="semantic"))


# ---------------------------------------------------------------------------
# BM25 floor and warning tests (FIX 1, 3, 5)
# ---------------------------------------------------------------------------


def test_bm25_matcher_rejects_stopword_only_overlap(
    bm25_matcher: ClaimMatcher,
) -> None:
    """A claim that shares only stopwords must not be supported."""
    chunk = Chunk(
        content="The quick brown fox jumps over the lazy dog.",
        source_url="https://example.com/fox",
    )
    # Claim shares only stopwords "the", "is", "a" with chunk.
    claim = ["The sky is a beautiful place to be in the world."]
    results = bm25_matcher.match(claim, [chunk])
    assert results[0].supported is False
    assert results[0].confidence == 0.0


def test_bm25_matcher_strong_overlap_still_supported(
    bm25_matcher: ClaimMatcher, aspirin_chunk: Chunk, ibuprofen_chunk: Chunk
) -> None:
    """A claim with strong content-word overlap must pass the raw floor.

    Two chunks are required so that IDF is non-trivial (terms appearing
    in only one document score higher than in a single-document corpus).
    """
    claims = ["Aspirin inhibits COX enzymes reducing fever in patients."]
    results = bm25_matcher.match(claims, [aspirin_chunk, ibuprofen_chunk])
    assert results[0].supported is True
    assert results[0].confidence > 0.0


def test_bm25_min_raw_score_env_override(
    monkeypatch: pytest.MonkeyPatch, aspirin_chunk: Chunk
) -> None:
    """Setting DOKIS_MIN_BM25_RAW_SCORE=999 must block all matches."""
    monkeypatch.setenv("DOKIS_MIN_BM25_RAW_SCORE", "999")
    import importlib

    import dokis.core.matcher as m

    importlib.reload(m)
    matcher = m.ClaimMatcher(Config(matcher="bm25", claim_threshold=0.0))
    claims = ["Aspirin inhibits COX enzymes reducing fever in patients."]
    results = matcher.match(claims, [aspirin_chunk])
    assert results[0].supported is False
    assert results[0].confidence == 0.0
    # Reload with default to avoid polluting other tests.
    monkeypatch.delenv("DOKIS_MIN_BM25_RAW_SCORE")
    importlib.reload(m)


def test_bm25_high_threshold_logs_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """claim_threshold > 0.5 with bm25 must log a WARNING at init."""
    import logging

    with caplog.at_level(logging.WARNING, logger="dokis.core.matcher"):
        ClaimMatcher(Config(matcher="bm25", claim_threshold=0.72))
    assert any("claim_threshold" in r.message for r in caplog.records)


def test_bm25_matcher_warns_when_model_is_set(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Setting a custom model with bm25 matcher must log a WARNING."""
    import logging

    with caplog.at_level(logging.WARNING, logger="dokis.core.matcher"):
        ClaimMatcher(Config(matcher="bm25", model="paraphrase-MiniLM-L6-v2"))
    assert any(
        "model" in r.message and "ignored" in r.message for r in caplog.records
    )


# ---------------------------------------------------------------------------
# BM25 index cache tests (FIX 3)
# ---------------------------------------------------------------------------


def test_bm25_index_is_cached_on_repeated_calls(
    bm25_matcher: ClaimMatcher, aspirin_chunk: Chunk
) -> None:
    """The BM25 index must be built once and reused on repeated calls
    with the same chunk set."""
    claims = ["Aspirin inhibits COX enzymes reducing fever in patients."]
    bm25_matcher.match(claims, [aspirin_chunk])
    cache_size_after_first = len(bm25_matcher._bm25_cache)
    bm25_matcher.match(claims, [aspirin_chunk])
    cache_size_after_second = len(bm25_matcher._bm25_cache)
    assert cache_size_after_first == 1
    assert cache_size_after_second == 1  # no new entry built


def test_bm25_index_rebuilds_for_different_chunk_sets(
    bm25_matcher: ClaimMatcher,
    aspirin_chunk: Chunk,
    ibuprofen_chunk: Chunk,
) -> None:
    """Different chunk sets must produce separate cache entries."""
    claims = ["Aspirin inhibits COX enzymes reducing fever in patients."]
    bm25_matcher.match(claims, [aspirin_chunk])
    bm25_matcher.match(claims, [ibuprofen_chunk])
    assert len(bm25_matcher._bm25_cache) == 2


def test_bm25_cache_evicts_oldest_when_full(
    aspirin_chunk: Chunk,
) -> None:
    """Cache must evict the oldest entry when _MAX_BM25_CACHE_SIZE is reached."""
    import dokis.core.matcher as m

    original = m._MAX_BM25_CACHE_SIZE
    m._MAX_BM25_CACHE_SIZE = 2
    try:
        matcher = ClaimMatcher(Config(matcher="bm25", claim_threshold=0.3))
        claims = ["Aspirin inhibits COX enzymes."]
        chunks_a = [Chunk(content="alpha content about aspirin", source_url="https://a.com")]
        chunks_b = [Chunk(content="beta content about ibuprofen", source_url="https://b.com")]
        chunks_c = [Chunk(content="gamma content about paracetamol", source_url="https://c.com")]
        matcher.match(claims, chunks_a)
        matcher.match(claims, chunks_b)
        assert len(matcher._bm25_cache) == 2
        matcher.match(claims, chunks_c)
        # Still 2 entries — oldest was evicted to make room.
        assert len(matcher._bm25_cache) == 2
    finally:
        m._MAX_BM25_CACHE_SIZE = original


def test_matcher_cosine_similarity_matches_sklearn_output(
    aspirin_chunk: Chunk,
) -> None:
    """Inline numpy cosine similarity must match sklearn's output within 1e-6."""
    import numpy as np

    from dokis.core.matcher import _cosine_similarity

    rng = np.random.default_rng(42)
    a = rng.random((4, 16)).astype(np.float32)
    b = rng.random((3, 16)).astype(np.float32)
    got = _cosine_similarity(a, b)
    # Reference: manual numpy
    a_n = a / np.linalg.norm(a, axis=1, keepdims=True)
    b_n = b / np.linalg.norm(b, axis=1, keepdims=True)
    expected = (a_n @ b_n.T).astype(np.float64)
    np.testing.assert_allclose(got, expected, atol=1e-6)
