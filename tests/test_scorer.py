    """Full test suite for ComplianceScorer."""

    import pytest

    from dokis.config import Config
    from dokis.core.scorer import ComplianceScorer
    from dokis.models import Chunk, Claim


    def _make_claim(supported: bool, confidence: float = 0.9) -> Claim:
        chunk = Chunk(content="content", source_url="https://example.com")
        return Claim(
            text="Some claim text that is long enough.",
            supported=supported,
            confidence=confidence,
            source_chunk=chunk if supported else None,
            source_url="https://example.com" if supported else None,
        )


    def test_scorer_empty_claims_returns_full_compliance() -> None:
        scorer = ComplianceScorer(Config())
        rate, passed = scorer.score([])
        assert rate == 1.0
        assert passed is True


    def test_scorer_all_supported_returns_one() -> None:
        scorer = ComplianceScorer(Config())
        claims = [_make_claim(supported=True) for _ in range(5)]
        rate, passed = scorer.score(claims)
        assert rate == pytest.approx(1.0)
        assert passed is True


    def test_scorer_none_supported_returns_zero() -> None:
        scorer = ComplianceScorer(Config(min_citation_rate=0.5))
        claims = [_make_claim(supported=False) for _ in range(3)]
        rate, passed = scorer.score(claims)
        assert rate == pytest.approx(0.0)
        assert passed is False


    def test_scorer_partial_support_correct_rate() -> None:
        scorer = ComplianceScorer(Config())
        claims = [
            _make_claim(supported=True),
            _make_claim(supported=True),
            _make_claim(supported=False),
            _make_claim(supported=False),
        ]
        rate, _ = scorer.score(claims)
        assert rate == pytest.approx(0.5)


    def test_scorer_passed_true_when_rate_meets_threshold() -> None:
        scorer = ComplianceScorer(Config(min_citation_rate=0.5))
        claims = [_make_claim(supported=True), _make_claim(supported=False)]
        rate, passed = scorer.score(claims)
        assert rate == pytest.approx(0.5)
        assert passed is True


    def test_scorer_passed_false_when_rate_below_threshold() -> None:
        scorer = ComplianceScorer(Config(min_citation_rate=0.85))
        # 1/2 = 0.5, which is < 0.85
        claims = [_make_claim(supported=True), _make_claim(supported=False)]
        rate, passed = scorer.score(claims)
        assert rate == pytest.approx(0.5)
        assert passed is False
