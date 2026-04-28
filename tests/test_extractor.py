"""Full test suite for ClaimExtractor."""

from unittest.mock import MagicMock

import pytest

from dokis.config import Config
from dokis.core.extractor import ClaimExtractor


@pytest.fixture
def extractor() -> ClaimExtractor:
    return ClaimExtractor(Config(extractor="regex"))


@pytest.fixture
def default_extractor() -> ClaimExtractor:
    return ClaimExtractor(Config())


@pytest.fixture
def claimify_extractor() -> ClaimExtractor:
    return ClaimExtractor(Config(extractor="claimify"))


def test_extractor_splits_multi_sentence_response(extractor: ClaimExtractor) -> None:
    response = (
        "Aspirin inhibits the COX-1 and COX-2 enzymes, reducing "
        "prostaglandin synthesis. "
        "Ibuprofen is a nonsteroidal anti-inflammatory drug used to relieve "
        "pain and fever. "
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


def test_extract_still_returns_list_of_strings(extractor: ClaimExtractor) -> None:
    claims = extractor.extract(
        "Aspirin reduces fever by inhibiting COX enzymes in the body."
    )
    assert claims
    assert all(isinstance(claim, str) for claim in claims)


def test_extract_detailed_returns_records_with_metadata() -> None:
    default_extractor = ClaimExtractor(Config())
    detailed = default_extractor.extract_detailed("Dokis does not require an LLM.")
    assert len(detailed) == 1
    assert detailed[0].text == "Dokis does not require an LLM."
    assert detailed[0].source_sentence == "Dokis does not require an LLM."
    assert detailed[0].extraction_method == "claimify"
    assert detailed[0].flags == ()


def test_default_extractor_is_claimify_and_does_not_call_llm_path(
    default_extractor: ClaimExtractor,
) -> None:
    # The default path must remain deterministic and must not call llm_fn.
    response = "Aspirin reduces fever by inhibiting COX enzymes in the body."
    detailed = default_extractor.extract_detailed(response)
    assert len(detailed) == 1
    assert detailed[0].extraction_method == "claimify"


def test_extractor_regex_path_splits_correctly() -> None:
    """Explicit regex extractor must split multi-sentence responses."""
    config = Config(extractor="regex")
    extractor = ClaimExtractor(config)
    response = (
        "Aspirin inhibits the COX-1 and COX-2 enzymes, reducing "
        "prostaglandin synthesis. "
        "Ibuprofen is a nonsteroidal anti-inflammatory drug used to relieve "
        "pain and fever."
    )
    claims = extractor.extract(response)
    assert len(claims) == 2


def test_extractor_regex_path_filters_short_sentences() -> None:
    """Regex path must still apply the 8-word minimum filter."""
    config = Config(extractor="regex")
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
        "Aspirin inhibits the COX-1 and COX-2 enzymes, reducing "
        "prostaglandin synthesis. "
        "Ibuprofen is a nonsteroidal anti-inflammatory drug used to relieve "
        "pain and fever."
    )
    claims = extractor.extract(response)
    assert len(claims) == 2


def test_claimify_keeps_factual_verifiable_sentence(
    claimify_extractor: ClaimExtractor,
) -> None:
    claims = claimify_extractor.extract("Aspirin inhibits platelet aggregation.")
    assert claims == ["Aspirin inhibits platelet aggregation."]


@pytest.mark.parametrize(
    "sentence",
    [
        "Aspirin reduces fever.",
        "BM25 is deterministic.",
        "Dokis uses BM25.",
        "Dokis does not require an LLM.",
        "You can configure Dokis with max_source_age_days.",
        "Users can install Dokis from PyPI.",
    ],
)
def test_claimify_keeps_required_factual_claims(
    claimify_extractor: ClaimExtractor,
    sentence: str,
) -> None:
    assert claimify_extractor.extract(sentence) == [sentence]


@pytest.mark.parametrize(
    "sentence",
    [
        "Vaccines prevent disease.",
        "Autophagy impacts human health.",
        "Waste-to-energy plants reduce the volume of waste by about 87%.",
    ],
)
def test_claimify_keeps_short_factual_science_and_product_claims(
    claimify_extractor: ClaimExtractor,
    sentence: str,
) -> None:
    assert claimify_extractor.extract(sentence) == [sentence]


def test_claimify_keeps_passive_factual_claim(
    claimify_extractor: ClaimExtractor,
) -> None:
    sentence = "Penicillin was discovered in 1928."
    assert claimify_extractor.extract(sentence) == [sentence]


def test_claimify_keeps_modal_effect_claim(
    claimify_extractor: ClaimExtractor,
) -> None:
    sentence = "Aspirin can reduce fever."
    assert claimify_extractor.extract(sentence) == [sentence]


def test_claimify_keeps_copula_definitional_claim(
    claimify_extractor: ClaimExtractor,
) -> None:
    sentence = "Epigenetics is the study of gene expression."
    assert claimify_extractor.extract(sentence) == [sentence]


def test_claimify_keeps_default_readiness_definitional_claim(
    claimify_extractor: ClaimExtractor,
) -> None:
    sentence = "Calligraphy is the art of beautiful handwriting."
    assert claimify_extractor.extract(sentence) == [sentence]


@pytest.mark.parametrize(
    "sentence",
    [
        "People who have synesthesia are called synesthetes.",
        "This is called magnetoreception.",
    ],
)
def test_claimify_keeps_called_classification_claims(
    claimify_extractor: ClaimExtractor,
    sentence: str,
) -> None:
    assert claimify_extractor.extract(sentence) == [sentence]


def test_claimify_keeps_short_bullet_facts(
    claimify_extractor: ClaimExtractor,
) -> None:
    response = "- The brain is 73% water.\n- The brain can't feel pain."
    assert claimify_extractor.extract(response) == [
        "The brain is 73% water.",
        "The brain can't feel pain.",
    ]


def test_claimify_keeps_examples_list_factual_claim(
    claimify_extractor: ClaimExtractor,
) -> None:
    sentence = "Examples are aspirin, ibuprofen or acetaminophen."
    assert claimify_extractor.extract(sentence) == [sentence]


def test_claimify_drops_pure_opinion(
    claimify_extractor: ClaimExtractor,
) -> None:
    assert claimify_extractor.extract("This is amazing.") == []


@pytest.mark.parametrize(
    "sentence",
    [
        "This is amazing.",
        "BM25 works.",
        "It works well.",
        "This is useful.",
        "You should be careful.",
        "Here is the answer.",
    ],
)
def test_claimify_drops_required_low_quality_sentences(
    claimify_extractor: ClaimExtractor,
    sentence: str,
) -> None:
    assert claimify_extractor.extract(sentence) == []


@pytest.mark.parametrize(
    "sentence",
    [
        "Some examples include:",
        "According to the search results, some of the methods are:",
        "Flow can be achieved by finding activities that match the following criteria:",
        "Some of the factors that contributed to this evolution are:",
        "According to sources, some of the most significant ones are:",
    ],
)
def test_claimify_drops_broad_setup_and_list_headers(
    claimify_extractor: ClaimExtractor,
    sentence: str,
) -> None:
    assert claimify_extractor.extract(sentence) == []


@pytest.mark.parametrize(
    "sentence",
    [
        "I hope this helps you learn more about ocean plastic cleanup.",
        "I hope this helps you understand the topic.",
        "Sure, I can explain the psychological phenomenon of flow.",
        "Sure, I can explain the process.",
        "Hello, this is Bing.",
        "If you have any follow-up questions, please let me know.",
    ],
)
def test_claimify_drops_assistant_filler(
    claimify_extractor: ClaimExtractor,
    sentence: str,
) -> None:
    assert claimify_extractor.extract(sentence) == []


def test_claimify_drops_there_are_many_setup_sentence(
    claimify_extractor: ClaimExtractor,
) -> None:
    sentence = "There are many strategies that can help reduce food waste."
    assert claimify_extractor.extract(sentence) == []


def test_claimify_still_drops_generic_opinion_and_weak_predicate(
    claimify_extractor: ClaimExtractor,
) -> None:
    assert claimify_extractor.extract("BM25 works.") == []
    assert claimify_extractor.extract("This is useful.") == []


@pytest.mark.parametrize(
    "sentence",
    [
        "[^2^]",
        "[^4^] [^5^]",
    ],
)
def test_claimify_drops_standalone_citation_fragments(
    claimify_extractor: ClaimExtractor,
    sentence: str,
) -> None:
    assert claimify_extractor.extract(sentence) == []


@pytest.mark.parametrize(
    "sentence",
    [
        "and opting for reusable alternatives instead.",
        "with pilot projects,",
    ],
)
def test_claimify_drops_broken_fragments(
    claimify_extractor: ClaimExtractor,
    sentence: str,
) -> None:
    assert claimify_extractor.extract(sentence) == []


def test_claimify_flags_ambiguous_factual_pronoun_claim(
    claimify_extractor: ClaimExtractor,
) -> None:
    detailed = claimify_extractor.extract_detailed(
        "It improved performance by 12% on MMLU."
    )
    assert [claim.text for claim in detailed] == [
        "It improved performance by 12% on MMLU."
    ]
    assert detailed[0].flags == ("ambiguous_reference",)


def test_claimify_drops_vague_pronoun_claim(
    claimify_extractor: ClaimExtractor,
) -> None:
    assert claimify_extractor.extract("It improved performance.") == []


def test_claimify_drops_meta_filler_text(
    claimify_extractor: ClaimExtractor,
) -> None:
    response = "Here is the answer. Let me explain. Dokis filters blocked sources."
    assert claimify_extractor.extract(response) == ["Dokis filters blocked sources."]


def test_claimify_decomposes_simple_shared_subject_compound_claim(
    claimify_extractor: ClaimExtractor,
) -> None:
    response = (
        "Dokis filters blocked sources, audits generated claims, "
        "and produces compliance logs."
    )
    assert claimify_extractor.extract(response) == [
        "Dokis filters blocked sources.",
        "Dokis audits generated claims.",
        "Dokis produces compliance logs.",
    ]


def test_claimify_decomposes_two_shared_subject_predicates(
    claimify_extractor: ClaimExtractor,
) -> None:
    response = "Dokis filters blocked sources and audits generated claims."
    assert claimify_extractor.extract(response) == [
        "Dokis filters blocked sources.",
        "Dokis audits generated claims.",
    ]


def test_claimify_preserves_negation_during_decomposition(
    claimify_extractor: ClaimExtractor,
) -> None:
    response = "Dokis does not require an LLM and does not make network calls."
    assert claimify_extractor.extract(response) == [
        "Dokis does not require an LLM.",
        "Dokis does not make network calls.",
    ]


def test_claimify_decomposes_safe_support_object_list(
    claimify_extractor: ClaimExtractor,
) -> None:
    response = "Dokis supports BM25 matching, semantic matching, and freshness checks."
    assert claimify_extractor.extract(response) == [
        "Dokis supports BM25 matching.",
        "Dokis supports semantic matching.",
        "Dokis supports freshness checks.",
    ]


def test_claimify_does_not_split_trailing_modifier_medical_object_list(
    claimify_extractor: ClaimExtractor,
) -> None:
    response = (
        "The drug treats migraines, cluster headaches, and tension headaches in adults."
    )
    assert claimify_extractor.extract(response) == [response]


def test_claimify_does_not_split_python_version_list(
    claimify_extractor: ClaimExtractor,
) -> None:
    response = "Dokis supports Python 3.10, 3.11, and 3.12."
    assert claimify_extractor.extract(response) == [response]


def test_claimify_splits_safe_semicolon_claims(
    claimify_extractor: ClaimExtractor,
) -> None:
    response = "Dokis filters blocked sources; ClaimMatcher scores claims."
    assert claimify_extractor.extract(response) == [
        "Dokis filters blocked sources.",
        "ClaimMatcher scores claims.",
    ]


def test_claimify_does_not_emit_pronoun_semicolon_fragment(
    claimify_extractor: ClaimExtractor,
) -> None:
    response = "Dokis filters blocked sources; it also audits generated claims."
    assert claimify_extractor.extract(response) == [response]


def test_claimify_propagates_features_lead_in_to_bullet_list(
    claimify_extractor: ClaimExtractor,
) -> None:
    response = (
        "Dokis features:\n\n"
        "filters blocked sources\n"
        "audits generated claims\n"
        "produces compliance logs"
    )
    assert claimify_extractor.extract(response) == [
        "Dokis filters blocked sources.",
        "Dokis audits generated claims.",
        "Dokis produces compliance logs.",
    ]


def test_claimify_propagates_provides_lead_in_to_numbered_list(
    claimify_extractor: ClaimExtractor,
) -> None:
    response = "Dokis provides:\n\n1. source allowlisting\n2. freshness checks"
    assert claimify_extractor.extract(response) == [
        "Dokis provides source allowlisting.",
        "Dokis provides freshness checks.",
    ]


def test_claimify_deduplicates_after_quality_gate(
    claimify_extractor: ClaimExtractor,
) -> None:
    response = "Dokis uses BM25. dokis uses bm25. Dokis uses BM25!"
    assert claimify_extractor.extract(response) == ["Dokis uses BM25."]


def test_claimify_keeps_risky_compound_sentence_without_bad_fragments(
    claimify_extractor: ClaimExtractor,
) -> None:
    response = (
        "Dokis filters blocked sources, audits generated claims, and compliance logs."
    )
    assert claimify_extractor.extract(response) == [response]


def test_claimify_handles_none_and_empty_input(
    claimify_extractor: ClaimExtractor,
) -> None:
    assert claimify_extractor.extract(None) == []
    assert claimify_extractor.extract("") == []
    assert claimify_extractor.extract(" \n\t ") == []


def test_extract_detailed_handles_none_and_empty_input(
    claimify_extractor: ClaimExtractor,
) -> None:
    assert claimify_extractor.extract_detailed(None) == []
    assert claimify_extractor.extract_detailed("") == []
    assert claimify_extractor.extract_detailed(" \n\t ") == []


def test_claimify_does_not_split_decimal_numbers(
    claimify_extractor: ClaimExtractor,
) -> None:
    response = (
        "The measured support score was 0.72 for the generated claim. "
        "The source was published in 2020."
    )
    claims = claimify_extractor.extract(response)
    assert claims == [
        "The measured support score was 0.72 for the generated claim.",
        "The source was published in 2020.",
    ]


def test_claimify_does_not_split_common_abbreviations(
    claimify_extractor: ClaimExtractor,
) -> None:
    response = (
        "Dr. Smith confirmed that aspirin inhibits platelet aggregation. "
        "Fig. 2 shows the normalized score was 0.72."
    )
    claims = claimify_extractor.extract(response)
    assert claims == [
        "Dr. Smith confirmed that aspirin inhibits platelet aggregation.",
        "Fig. 2 shows the normalized score was 0.72.",
    ]


def test_extract_detailed_regex_nltk_and_llm_metadata() -> None:
    regex = ClaimExtractor(Config(extractor="regex"))
    regex_records = regex.extract_detailed(
        "Aspirin reduces fever by inhibiting COX enzymes in the body."
    )
    assert regex_records[0].extraction_method == "regex"
    assert regex_records[0].flags == ()

    nltk = ClaimExtractor(Config(extractor="nltk"))
    nltk_records = nltk.extract_detailed(
        "Aspirin reduces fever by inhibiting COX enzymes in the body."
    )
    assert nltk_records[0].extraction_method == "nltk"
    assert nltk_records[0].source_sentence == nltk_records[0].text

    llm_fn: MagicMock = MagicMock(
        return_value="Aspirin reduces fever by inhibiting COX enzymes in the body."
    )
    llm = ClaimExtractor(Config(extractor="llm", llm_fn=llm_fn))
    llm_records = llm.extract_detailed("Ignored response text.")
    assert llm_records[0].extraction_method == "llm"
    assert llm_records[0].source_sentence == llm_records[0].text


def test_extract_detailed_llm_preserves_missing_llm_fn_error() -> None:
    extractor = ClaimExtractor(Config(extractor="llm"))
    with pytest.raises(ValueError, match="llm_fn is None"):
        extractor.extract_detailed("Some response text that is long enough.")
