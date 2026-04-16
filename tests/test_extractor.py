"""Full test suite for ClaimExtractor."""

from unittest.mock import MagicMock

import pytest

from dokis.config import Config
from dokis.core.extractor import ClaimExtractor


@pytest.fixture
def extractor() -> ClaimExtractor:
    return ClaimExtractor(Config())


def test_extractor_splits_multi_sentence_response(extractor: ClaimExtractor) -> None:
    response = (
        "Aspirin inhibits the COX-1 and COX-2 enzymes, reducing prostaglandin synthesis. "
        "Ibuprofen is a nonsteroidal anti-inflammatory drug used to relieve pain and fever. "
        "Both drugs work by blocking the cyclooxygenase pathway in human cells."
    )
    claims = extractor.extract(response)
    assert len(claims) == 3
    assert all(isinstance(c, str) for c in claims)


def test_extractor_filters_short_sentences(extractor: ClaimExtractor) -> None:
    # "In summary." is 2 words — must be filtered out.
    response = (
        "In summary. Aspirin reduces fever by inhibiting COX enzymes in the body."
    )
    claims = extractor.extract(response)
    assert len(claims) == 1
    assert "Aspirin" in claims[0]


def test_extractor_returns_empty_list_for_empty_response(
    extractor: ClaimExtractor,
) -> None:
    assert extractor.extract("") == []


def test_extractor_returns_empty_list_for_whitespace_response(
    extractor: ClaimExtractor,
) -> None:
    assert extractor.extract("   \n\t  ") == []


def test_extractor_llm_path_calls_llm_fn() -> None:
    mock_fn: MagicMock = MagicMock(
        return_value=(
            "Aspirin inhibits COX enzymes and reduces fever in the bloodstream.\n"
            "Ibuprofen is a nonsteroidal anti-inflammatory drug for pain relief."
        )
    )
    config = Config(extractor="llm", llm_fn=mock_fn)
    extractor = ClaimExtractor(config)
    claims = extractor.extract("Some response text that is long enough.")
    mock_fn.assert_called_once()
    assert len(claims) == 2


def test_extractor_returns_empty_list_for_none_response(
    extractor: ClaimExtractor,
) -> None:
    """extract() must not raise when response is None."""
    assert extractor.extract(None) == []


def test_extractor_llm_path_not_called_when_sentence_transformers(
    extractor: ClaimExtractor,
) -> None:
    # The default sentence_transformers path must never call llm_fn.
    # config.llm_fn is None — if it were invoked it would raise AttributeError.
    response = "Aspirin reduces fever by inhibiting COX enzymes in the body."
    result = extractor.extract(response)
    assert len(result) == 1


def test_extractor_regex_path_splits_correctly() -> None:
    """Default regex extractor must split multi-sentence responses."""
    config = Config()  # extractor="regex" is now the default
    extractor = ClaimExtractor(config)
    response = (
        "Aspirin inhibits the COX-1 and COX-2 enzymes, reducing prostaglandin synthesis. "
        "Ibuprofen is a nonsteroidal anti-inflammatory drug used to relieve pain and fever."
    )
    claims = extractor.extract(response)
    assert len(claims) == 2


def test_extractor_regex_path_filters_short_sentences() -> None:
    """Regex path must still apply the 8-word minimum filter."""
    config = Config()
    extractor = ClaimExtractor(config)
    response = (
        "In summary. Aspirin reduces fever by inhibiting COX enzymes in the body."
    )
    claims = extractor.extract(response)
    assert len(claims) == 1
    assert "Aspirin" in claims[0]


def test_extractor_regex_does_not_split_decimal_numbers(
    extractor: ClaimExtractor,
) -> None:
    """Decimal numbers must not trigger a sentence split."""
    response = (
        "The blood pH of 7.4 is considered normal for healthy adults. "
        "Aspirin reduces fever by inhibiting the COX enzymes in tissue."
    )
    claims = extractor.extract(response)
    assert len(claims) == 2
    assert any("7.4" in c for c in claims)


def test_extractor_regex_does_not_split_on_dr_abbreviation(
    extractor: ClaimExtractor,
) -> None:
    """Dr. abbreviation must not trigger a sentence split."""
    response = (
        "Dr. Smith confirmed that aspirin inhibits COX enzyme activity. "
        "Ibuprofen is a nonsteroidal anti-inflammatory drug for pain relief."
    )
    claims = extractor.extract(response)
    assert len(claims) == 2
    assert any("Dr." in c for c in claims)


def test_extractor_regex_does_not_split_on_vs_abbreviation(
    extractor: ClaimExtractor,
) -> None:
    """vs. abbreviation must not produce fewer than 2 claims."""
    response = (
        "Aspirin vs. ibuprofen shows different COX inhibition profiles in patients. "
        "Both drugs reduce fever by blocking prostaglandin production in tissue."
    )
    claims = extractor.extract(response)
    assert len(claims) == 2


def test_extractor_nltk_path_still_works() -> None:
    """extractor='nltk' must still call sent_tokenize correctly."""
    config = Config(extractor="nltk")
    extractor = ClaimExtractor(config)
    response = (
        "Aspirin inhibits the COX-1 and COX-2 enzymes, reducing prostaglandin synthesis. "
        "Ibuprofen is a nonsteroidal anti-inflammatory drug used to relieve pain and fever."
    )
    claims = extractor.extract(response)
    assert len(claims) == 2
