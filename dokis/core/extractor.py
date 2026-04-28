"""Claim extractor - splits an LLM response into atomic claim sentences."""

from __future__ import annotations

import contextlib
import logging
import re as _re
from collections.abc import Sequence
from dataclasses import dataclass

# nltk is an optional dependency (pip install dokis[nltk]).
# The module-level import is guarded so that the package loads cleanly when
# nltk is not installed. _ensure_punkt_tab() raises ImportError with a clear
# message if extractor="nltk" is requested without nltk present.
with contextlib.suppress(ImportError):
    import nltk  # type: ignore[import-untyped]

from dokis.config import Config

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ExtractedClaim:
    """Internal detailed extraction record.

    ``extractor="claimify"`` is deterministic and English-oriented in v2. Its
    rule-based verb detection, meta/filler filtering, and capability-claim
    handling are English-specific. Non-English users should use ``regex``,
    ``nltk``, or a custom ``llm_fn`` for now.
    """

    text: str
    source_sentence: str
    extraction_method: str
    flags: tuple[str, ...] = ()


# Minimum word count for a sentence to be considered an atomic claim.
# Transitions and headings (e.g. "In summary." / "For example.") fall below
# this threshold and are not useful for provenance matching.
_MIN_WORDS = 8

# Prompt template used when extractor="llm". The user-supplied llm_fn must
# return a newline-separated list of atomic claims extracted from the response.
_LLM_PROMPT_TEMPLATE = (
    "Extract every atomic factual claim from the following text as a "
    "newline-separated list. Output one claim per line. Do not include "
    "bullet points, numbering, or any other formatting.\n\nText:\n{response}"
)

# Regex sentence boundary pattern used by the legacy "regex" extractor path
# and as a sentence splitter inside the deterministic "claimify" pipeline.
# Matches sentence boundaries while avoiding:
# - Common abbreviations (Dr., Mr., Ms., etc.): [A-Z][a-z]\. lookbehind
# - Single letter initials (J. Smith): \s[A-Z]\. lookbehind
# - Numbered items (Fig. 3): not followed by a digit
# - Decimal numbers (3.14, pH 7.4): protected naturally - the digit after
#   the decimal point means \s+ cannot match at that position.
# All lookbehinds are fixed-width for Python re compatibility.
_SENTENCE_SPLIT = _re.compile(
    r"(?<![A-Z][a-z]\.)"  # not preceded by Uppercase+lowercase+period (Dr., Mr., etc.)
    r"(?<!\s[A-Z]\.)"  # not preceded by whitespace+Uppercase+period (J. initial)
    r"(?<=[.?!])"  # must follow sentence-ending punctuation
    r"(?!\d)"  # not followed by a digit
    r"\s+"
)
_ABBREVIATION_DOT = "<DOKIS_DOT>"
_COMMON_ABBREVIATIONS = (
    "Dr.",
    "Fig.",
    "Mr.",
    "Mrs.",
    "Ms.",
    "Prof.",
    "Sr.",
    "St.",
    "e.g.",
    "i.e.",
    "vs.",
)

_TERMINAL_PUNCTUATION = ".?!"
_OPINION_WORDS = {
    "amazing",
    "awesome",
    "bad",
    "beautiful",
    "best",
    "boring",
    "excellent",
    "fantastic",
    "good",
    "great",
    "interesting",
    "nice",
    "terrible",
    "useful",
    "wonderful",
}
_META_PREFIXES = (
    "as an ai",
    "hello, this is",
    "here is",
    "here are",
    "i'm happy to help",
    "i hope",
    "i'm glad",
    "i will explain",
    "if you want",
    "in conclusion",
    "in short",
    "in summary",
    "let me explain",
    "overall",
    "sure, i can",
    "that depends",
    "the answer is",
    "the answer depends",
    "the answer may depend",
    "this answer",
    "to summarize",
)
_DIRECTIVE_PREFIXES = (
    "avoid ",
    "be careful",
    "consider ",
    "do not ",
    "please ",
    "repair ",
    "remember to ",
    "you can ",
    "you could ",
    "you may ",
    "you might ",
    "you must ",
    "you need to ",
    "you should ",
)
_AMBIGUOUS_PRONOUNS = {"it", "this", "that", "they", "these", "those"}
_FRAGMENT_PREFIXES = ("but ", "or ", "without ", "to ")
_GENERIC_PREDICATES = {"work", "works", "worked", "working"}
_WEAK_PREDICATE_COMPLEMENTS = {"best", "fine", "great", "ok", "okay", "well"}
_LEAD_IN_VERBS = {"features", "includes", "provides", "supports"}
_SUBJECT_ONLY_LEAD_IN_VERBS = {"features"}
_SAFE_OBJECT_LIST_VERBS = {"includes", "offers", "provides", "supports"}
_TRAILING_MODIFIER_PREPOSITIONS = (
    "during",
    "for",
    "in",
    "under",
    "with",
    "without",
)
_VERB_WORDS = {
    "adds",
    "add",
    "allows",
    "apply",
    "applies",
    "appoints",
    "audits",
    "became",
    "become",
    "belongs",
    "blocks",
    "builds",
    "bury",
    "buried",
    "capture",
    "captures",
    "cause",
    "causes",
    "change",
    "changes",
    "checks",
    "choose",
    "claim",
    "claims",
    "configure",
    "contribute",
    "contributes",
    "cool",
    "cools",
    "computes",
    "contains",
    "cover",
    "covers",
    "come",
    "comes",
    "continue",
    "continues",
    "could",
    "creates",
    "deduct",
    "deducts",
    "derives",
    "detect",
    "detects",
    "developed",
    "develops",
    "depict",
    "depicts",
    "discover",
    "discovered",
    "displayed",
    "diversify",
    "diversifies",
    "enables",
    "emphasize",
    "emphasizes",
    "encourage",
    "encourages",
    "enforces",
    "evaluates",
    "evolved",
    "evolves",
    "explain",
    "explained",
    "explains",
    "explore",
    "explores",
    "expresses",
    "exposes",
    "features",
    "find",
    "finds",
    "filters",
    "fosters",
    "found",
    "gave",
    "generates",
    "gesture",
    "gestures",
    "grow",
    "grows",
    "handles",
    "has",
    "help",
    "helps",
    "include",
    "includes",
    "incorporated",
    "incorporates",
    "impact",
    "impacts",
    "increase",
    "increases",
    "inhibits",
    "install",
    "installs",
    "invest",
    "invests",
    "involve",
    "involves",
    "keeps",
    "lived",
    "lives",
    "loads",
    "maps",
    "marks",
    "matches",
    "nominate",
    "nominates",
    "orbit",
    "orbits",
    "prevent",
    "prevents",
    "produces",
    "promote",
    "promotes",
    "provides",
    "raises",
    "reach",
    "reaches",
    "records",
    "reduce",
    "removes",
    "requires",
    "reduces",
    "refer",
    "refers",
    "reflect",
    "reflects",
    "revived",
    "revives",
    "returns",
    "runs",
    "saw",
    "scores",
    "secrete",
    "secretes",
    "select",
    "selects",
    "splits",
    "stores",
    "strips",
    "studies",
    "supports",
    "take",
    "takes",
    "treats",
    "updated",
    "uses",
    "went",
    "wrote",
}
_AUXILIARY_VERBS = {
    "am",
    "are",
    "be",
    "been",
    "being",
    "can",
    "could",
    "did",
    "do",
    "does",
    "had",
    "has",
    "have",
    "is",
    "may",
    "might",
    "must",
    "should",
    "was",
    "were",
    "will",
}
_TOKEN = _re.compile(r"[A-Za-z][A-Za-z0-9_-]*|\d+(?:\.\d+)?")
_BULLET_PREFIX = _re.compile(r"^\s*(?:[-*+]\s+|\d+[.)]\s+)?(.+?)\s*$")
_CITATION_MARKER = _re.compile(r"\[\^\d+\^\]")
_BARE_URL = _re.compile(r"https?://\S+")


def _regex_split(text: str) -> list[str]:
    """Split text into sentences using a punctuation-boundary regex.

    Handles common scientific and medical abbreviations (Dr., Fig., vs.),
    decimal numbers (7.4, 3.14), and single initials (J. Smith) without
    false splits. Uses only fixed-width lookbehinds for Python re
    compatibility and explicit abbreviation protection for variable-length
    abbreviations.

    Args:
        text: Raw input text.

    Returns:
        List of sentence strings.
    """
    protected = text
    for abbreviation in _COMMON_ABBREVIATIONS:
        protected = protected.replace(
            abbreviation,
            abbreviation.replace(".", _ABBREVIATION_DOT),
        )
    sentences = [
        s.replace(_ABBREVIATION_DOT, ".").strip()
        for s in _SENTENCE_SPLIT.split(protected)
        if s.strip()
    ]
    return sentences


def _normalise_sentence(sentence: str) -> str:
    return " ".join(sentence.strip().split())


def _clean_claim_text(sentence: str) -> str:
    text = _strip_bullet_prefix(sentence)
    text = _CITATION_MARKER.sub("", text)
    text = _BARE_URL.sub("", text)
    text = text.replace("**", "")
    text = _normalise_sentence(text)
    text = text.strip(" \t\"'")
    text = text.rstrip()
    return text


def _ensure_terminal_punctuation(sentence: str) -> str:
    if sentence.endswith(tuple(_TERMINAL_PUNCTUATION)):
        return sentence
    return f"{sentence}."


def _word_count(sentence: str) -> int:
    return len(_TOKEN.findall(sentence))


def _starts_with_any(text: str, prefixes: tuple[str, ...]) -> bool:
    return any(text.startswith(prefix) for prefix in prefixes)


def _has_factual_anchor(sentence: str) -> bool:
    lower = sentence.lower()
    return bool(
        _re.search(r"\b\d{4}\b|\b\d+(?:\.\d+)?\b", sentence)
        or _re.search(r"\b(?:is|are|was|were|has|have|had)\b", lower)
        or _re.search(r"\b[A-Z]{2,}\d*\b", sentence)
    )


def _has_concrete_anchor(sentence: str) -> bool:
    return bool(
        _re.search(r"\b\d{4}\b|\b\d+(?:\.\d+)?%?\b", sentence)
        or _re.search(r"\b[A-Z]{2,}\d*\b", sentence)
        or _re.search(r"\b[a-z]+(?:_[a-z0-9]+)+\b", sentence)
        or _re.search(r"\b[A-Z][A-Za-z]*[A-Z][A-Za-z]*\b", sentence)
        or _re.search(r"\b[A-Z][a-z]+(?:[A-Z][a-z]+)+\b", sentence)
    )


def _first_token(sentence: str) -> str | None:
    match = _TOKEN.search(sentence)
    return match.group(0).lower() if match else None


def _starts_with_ambiguous_pronoun(sentence: str) -> bool:
    token = _first_token(sentence)
    return token in _AMBIGUOUS_PRONOUNS if token is not None else False


def _has_supported_ambiguous_reference(sentence: str) -> bool:
    lower = sentence.lower()
    if _looks_like_demonstrative_noun_claim(sentence):
        return True
    if _looks_like_substantive_pronoun_claim(sentence):
        return True
    if _looks_like_classification_claim(lower):
        return True
    if _looks_like_passive_use_claim(lower):
        return True
    if _looks_like_pronoun_effect_claim(lower):
        return True
    if _has_strong_ambiguous_anchor(sentence):
        return True
    if _first_token(sentence) != "it":
        return False
    words = [match.group(0).lower() for match in _TOKEN.finditer(sentence)]
    verb_index = _find_first_verb_index(words)
    if verb_index is None:
        return False
    verb = words[verb_index]
    if verb in _GENERIC_PREDICATES:
        return False
    if verb in _AUXILIARY_VERBS:
        return _looks_like_definitional_it_claim(words, verb_index)
    return len(words) >= 6


def _looks_like_substantive_pronoun_claim(sentence: str) -> bool:
    words = [match.group(0).lower() for match in _TOKEN.finditer(sentence)]
    if len(words) < 5 or words[0] not in _AMBIGUOUS_PRONOUNS:
        return False
    if words[1] in {"has", "have", "had"}:
        return words[2] not in _OPINION_WORDS
    if words[1] in {"can", "could", "may", "might"} and len(words) > 3:
        return words[2] not in _GENERIC_PREDICATES
    if words[1] in {"is", "are", "was", "were"}:
        complement_index = 2
        if words[complement_index] in {"also", "mostly", "currently", "originally"}:
            complement_index += 1
        if complement_index >= len(words):
            return False
        complement = words[complement_index]
        return complement not in _OPINION_WORDS | _GENERIC_PREDICATES
    return False


def _looks_like_demonstrative_noun_claim(sentence: str) -> bool:
    words = [match.group(0).lower() for match in _TOKEN.finditer(sentence)]
    if len(words) < 5 or words[0] not in {"this", "that", "these", "those"}:
        return False
    if words[1] in _AUXILIARY_VERBS or words[1] in _GENERIC_PREDICATES:
        return False
    return _find_first_verb_index(words) is not None


def _looks_like_classification_claim(lower: str) -> bool:
    return bool(
        _re.search(
            r"\b(?:is|are|was|were)\s+(?:also\s+)?"
            r"(?:called|known as|referred to as)\s+\w+",
            lower,
        )
    )


def _looks_like_passive_use_claim(lower: str) -> bool:
    return bool(
        _re.search(
            r"\b(?:is|are|was|were|has been|have been)\s+(?:originally\s+)?"
            r"(?:used|adopted|introduced|developed|created|designed|displayed|"
            r"found|derived|shown|discovered|thought|believed|regarded|influenced)\b",
            lower,
        )
    )


def _looks_like_pronoun_effect_claim(lower: str) -> bool:
    return bool(
        _re.search(
            r"\b(?:helps?|can help|may help)\s+(?:them|it|people|users|patients)\s+\w+",
            lower,
        )
        or _re.search(
            r"\b(?:encourage|encourages|promote|promotes|prevent|prevents|reduce|reduces)\s+"
            r"\w+",
            lower,
        )
    )


def _looks_like_definitional_it_claim(words: Sequence[str], verb_index: int) -> bool:
    if verb_index + 2 >= len(words):
        return False
    return (
        words[verb_index] in {"is", "was"}
        and words[verb_index + 1] in {"a", "an", "the"}
        and words[verb_index + 2] not in _OPINION_WORDS
    )


def _claim_flags(sentence: str) -> tuple[str, ...]:
    if _starts_with_ambiguous_pronoun(sentence):
        return ("ambiguous_reference",)
    return ()


def _has_strong_ambiguous_anchor(sentence: str) -> bool:
    return bool(
        _re.search(r"\b\d{4}\b|\b\d+(?:\.\d+)?%?\b", sentence)
        or _re.search(r"\b[A-Z]{2,}\d*\b", sentence)
        or _re.search(r"\b[a-z]+(?:_[a-z0-9]+)+\b", sentence)
    )


def _has_explicit_subject(sentence: str) -> bool:
    words = [match.group(0) for match in _TOKEN.finditer(sentence)]
    verb_index = _find_first_verb_index(words)
    if verb_index is None or verb_index == 0:
        return False
    subject = " ".join(words[:verb_index]).lower()
    return subject not in _AMBIGUOUS_PRONOUNS


def _has_nontrivial_predicate(sentence: str) -> bool:
    words = [match.group(0).lower() for match in _TOKEN.finditer(sentence)]
    for index, word in enumerate(words):
        if _is_verb_like(word):
            if word in _GENERIC_PREDICATES:
                return _generic_predicate_has_substantive_complement(words, index)
            if word in _AUXILIARY_VERBS:
                if index + 1 >= len(words):
                    return False
                complement = words[index + 1]
                if complement in _OPINION_WORDS or complement in _GENERIC_PREDICATES:
                    return False
            return index + 1 < len(words)
    return False


def _generic_predicate_has_substantive_complement(
    words: Sequence[str],
    index: int,
) -> bool:
    if index + 1 >= len(words):
        return False
    complement = words[index + 1]
    if complement in _WEAK_PREDICATE_COMPLEMENTS or complement in _OPINION_WORDS:
        return False
    return complement in {"as", "because", "by", "for", "in", "on", "to", "with"}


def _has_only_generic_predicate(sentence: str) -> bool:
    words = [match.group(0).lower() for match in _TOKEN.finditer(sentence)]
    has_generic = False
    for word in words:
        if not _is_verb_like(word):
            continue
        if word in _GENERIC_PREDICATES:
            has_generic = True
            continue
        if word not in _AUXILIARY_VERBS:
            return False
    return has_generic


def _is_non_claim_template(text: str, lower: str) -> bool:
    if _is_standalone_reference_fragment(text, lower):
        return True
    if _starts_with_any(lower, _FRAGMENT_PREFIXES):
        return True
    if _re.match(r"and\s+\w+ing\b", lower):
        return True
    if text.endswith(","):
        return True
    if lower.endswith(":") and _is_list_header(lower):
        return True
    if lower.endswith(":") and _looks_like_setup_header(lower):
        return True
    if _re.search(
        r"\b(?:there are|there is|there have been)\s+(?:so\s+)?many\b", lower
    ):
        return True
    if _re.search(r"\bthere is no (?:simple or )?definitive answer\b", lower):
        return True
    if _re.search(r"\bthere is no conclusive evidence\b", lower):
        return True
    if _re.search(
        r"\bthere is still (?:a lot of )?(?:uncertainty|controversy)\b",
        lower,
    ):
        return True
    if _re.search(r"\bit depends on\b", lower):
        return True
    if lower.startswith("as for how "):
        return True
    if lower.startswith("as individuals, "):
        return True
    if _re.search(r"\bif you have any (?:follow-up |further )?questions\b", lower):
        return True
    if _re.search(r"\bthe key is to\b", lower):
        return True
    if lower.startswith("according to ") and _re.search(r"\bsome\b.+\binclude$", lower):
        return True
    if _re.search(r"\bthese are just some\b", lower):
        return True
    if _re.search(r"\bthese are just (?:two|\d+) examples\b", lower):
        return True
    if _re.search(r"\bsome possible (?:explanations|causes|solutions)\b", lower):
        return True
    if _re.search(r"\b(?:some|many) (?:examples|factors|methods|ways)\b", lower):
        return True
    if _looks_like_broad_summary_claim(lower):
        return True
    if _re.search(r"\b(?:fascinating|interesting|amazing|wonderful)\b", lower):
        return not _has_concrete_anchor(text)
    return False


def _looks_like_broad_summary_claim(lower: str) -> bool:
    return bool(
        _re.search(r"\b(?:has|have|had)?\s*evolved over time to become\b", lower)
        or _re.search(r"\b(?:changed|impacted) our understanding\b", lower)
        or _re.search(r"\bopened (?:a )?new windows? into\b", lower)
        or _re.search(
            r"\b(?:inspiration|inspired curiosity|mysteries to explore)\b",
            lower,
        )
        or _re.search(r"\bmay be (?:a way|due to|anything)\b", lower)
        or _re.search(r"\bthis behavior may help\b", lower)
        or _re.search(r"\b(?:has|have) become an essential part\b", lower)
        or _re.search(
            r"\b(?:has|have) (?:its|their|his|her|own) "
            r"strengths and limitations\b",
            lower,
        )
        or _re.search(r"\bmay work better for some\b", lower)
        or _re.search(r"\bdifferent \w+ have different ways\b", lower)
        or _re.search(r"\bexpected to continue to evolve\b", lower)
        or _re.search(r"\bcan help you get started\b", lower)
        or _re.search(r"\bthis can help\b", lower)
        or _re.search(r"\bit is important for\b", lower)
        or _re.search(r"\bit is important to consider\b", lower)
        or _re.search(
            r"\bis a (?:long and diverse|rich and complex|diverse and vibrant) \w+\b",
            lower,
        )
        or _re.search(r"\b(?:complex|personal) topic\b", lower)
        or _re.search(r"\bcommon problem\b", lower)
        or _re.search(r"\b(?:huge|very challenging) challenge\b", lower)
        or _re.search(r"\bnot an easy feat\b", lower)
        or _re.search(r"\bno single answer\b", lower)
        or _re.search(r"\bno single or simple solution\b", lower)
        or _re.search(r"\bit is clear that\b", lower)
        or _re.search(r"\bmore research is needed\b", lower)
        or _re.search(r"\bthere are also many .+ challenges\b", lower)
        or _re.search(r"\b(?:one|another) possible (?:method|way)\b", lower)
        or _re.search(r"\beach method has its own\b", lower)
        or _re.search(r"\bnot a perfect solution\b", lower)
        or _re.search(r"\bcan be a viable alternative\b", lower)
        or _re.search(r"\bnot conclusive\b", lower)
        or _re.search(r"\bbeyond any doubt\b", lower)
        or _re.search(r"\bdon't know how close\b", lower)
        or _re.search(r"\bcan be anything from\b", lower)
        or _re.search(r"\ba sense of (?:timelessness|wonder|purpose)\b", lower)
        or _re.search(r"\bessential part of our\b", lower)
        or _re.search(r"\b(?:important|benefits) for (?:saving|promoting)\b", lower)
        or _re.search(r"\bbenefits compared to traditional\b", lower)
        or _re.search(r"\bwhat's enjoyable\b", lower)
        or _re.search(r"\bwidely regarded as\b", lower)
        or _re.search(r"\bdevastating consequences\b", lower)
        or _re.search(r"\bit is crucial that\b", lower)
        or _re.search(r"\btogether, we can\b", lower)
        or _looks_like_advice_gerund_bullet(lower)
    )


def _looks_like_advice_gerund_bullet(lower: str) -> bool:
    return bool(
        _re.match(
            r"(?:intercepting|stemming|recycling|supporting|educating|using|"
            r"getting|raising|planning|sharing)\b",
            lower,
        )
    )


def _is_standalone_reference_fragment(text: str, lower: str) -> bool:
    without_citations = _CITATION_MARKER.sub("", text)
    without_markup = without_citations.replace("*", "").strip(" .;:,\"'()[]")
    if not without_markup:
        return True
    if (
        _CITATION_MARKER.match(text.strip())
        and without_citations.lstrip().startswith(":")
    ):
        return True
    if _re.fullmatch(r"\(?\d{4}\)?", without_markup):
        return True
    if _BARE_URL.fullmatch(text.strip()):
        return True
    return lower in {'", or "ok"', "spacewar"}


def _looks_like_setup_header(lower: str) -> bool:
    header = lower.rstrip(":")
    ends_like_setup = bool(
        _re.search(r"\b(?:by|are|include|includes|following|below|criteria)$", header)
        or _re.search(r"\b(?:found that|such as|may include)$", header)
    )
    has_setup_topic = bool(
        _re.search(
            r"\b(?:criteria|methods|ways|steps|effort|lives|activities|points|"
            r"studies|policies|interventions|challenges|limitations)\b",
            header,
        )
    )
    return bool(
        ends_like_setup and has_setup_topic
    )


def _is_list_header(lower: str) -> bool:
    header = lower.rstrip(":")
    return bool(
        _re.search(
            r"\b(?:include|includes|may include|are|include the following|"
            r"are the following)$",
            header,
        )
        and _re.search(
            r"\b(?:some|examples|factors|methods|ways|criteria|changes|"
            r"causes|solutions|milestones|sources|references|below)\b",
            header,
        )
    )


def _is_verifiable_sentence(sentence: str) -> bool:
    """Return whether an English sentence is useful as a factual claim."""
    text = _clean_claim_text(sentence)
    if not text:
        return False
    lower = text.lower().strip(" .?!")
    if _word_count(text) < 3:
        return False
    if text.endswith("?"):
        return False
    if _starts_with_any(lower, _META_PREFIXES):
        return False
    if _is_non_claim_template(text, lower):
        return False
    has_concrete_anchor = _has_concrete_anchor(text)
    if _starts_with_any(lower, _DIRECTIVE_PREFIXES) and not has_concrete_anchor:
        return False
    if _looks_like_equation_claim(text):
        return True
    words = {match.group(0).lower() for match in _TOKEN.finditer(text)}
    if _has_only_generic_predicate(text):
        return False
    if (
        _re.fullmatch(
            r"(?:this|that|it)\s+(?:is|was|seems|sounds|looks)\s+\w+",
            lower,
        )
        and words & _OPINION_WORDS
    ):
        return False
    if words & _OPINION_WORDS and not _has_factual_anchor(text):
        return False
    if _starts_with_ambiguous_pronoun(text) and not _has_supported_ambiguous_reference(
        text
    ):
        return False
    return _has_nontrivial_predicate(text)


def _looks_like_equation_claim(text: str) -> bool:
    return bool("=" in text and _re.search(r"\d", text) and _word_count(text) >= 3)


def _is_verb_like(word: str) -> bool:
    lower = word.lower().strip("-")
    return (
        lower in _VERB_WORDS
        or lower in _AUXILIARY_VERBS
        or lower.endswith("ed")
        or lower.endswith("ing")
    )


def _find_first_verb_index(words: list[str]) -> int | None:
    for index, word in enumerate(words[1:], start=1):
        if _is_verb_like(word):
            return index
    return None


def _decompose_compound_claim(sentence: str) -> list[str]:
    """Split conservative shared-subject compounds when every claim is clear."""
    text = _clean_claim_text(sentence)
    body = text.rstrip(_TERMINAL_PUNCTUATION).strip()

    semicolon_claims = _decompose_semicolon_claim(body)
    if semicolon_claims is not None:
        return semicolon_claims

    pattern_a_claims = _decompose_shared_subject_verb_list(body)
    if pattern_a_claims is not None:
        return pattern_a_claims

    pattern_b_claims = _decompose_shared_subject_two_predicates(body)
    if pattern_b_claims is not None:
        return pattern_b_claims

    pattern_c_claims = _decompose_safe_object_list(body)
    if pattern_c_claims is not None:
        return pattern_c_claims

    return [text]


def _finalize_decomposition(original: str, claims: list[str]) -> list[str] | None:
    finalized = [_ensure_terminal_punctuation(claim) for claim in claims]
    if all(_is_verifiable_sentence(claim) for claim in finalized):
        return finalized
    return None


def _decompose_semicolon_claim(body: str) -> list[str] | None:
    if ";" not in body:
        return None
    parts = [_normalise_sentence(part) for part in body.split(";")]
    if len(parts) < 2 or any(not part for part in parts):
        return None
    if not all(_has_explicit_subject(part) for part in parts):
        return None
    return _finalize_decomposition(body, parts)


def _decompose_shared_subject_verb_list(body: str) -> list[str] | None:
    if ", and " not in body or body.count(",") < 2:
        return None

    parts = [_normalise_sentence(part) for part in _re.split(r",\s*(?:and\s+)?", body)]
    if len(parts) < 3 or any(not part for part in parts):
        return None

    first_words = [match.group(0) for match in _TOKEN.finditer(parts[0])]
    verb_index = _find_first_verb_index(first_words)
    if verb_index is None:
        return None

    subject = " ".join(first_words[:verb_index])
    first_verb = first_words[verb_index].lower()
    if not subject or subject.lower() in _AMBIGUOUS_PRONOUNS:
        return None
    if first_verb in _AUXILIARY_VERBS:
        return None

    claims = [parts[0]]
    for part in parts[1:]:
        part_words = [match.group(0) for match in _TOKEN.finditer(part)]
        if not part_words or not _is_predicate_start(part_words):
            return None
        claims.append(f"{subject} {part}")

    return _finalize_decomposition(body, claims)


def _decompose_shared_subject_two_predicates(body: str) -> list[str] | None:
    if "," in body or body.count(" and ") != 1:
        return None
    left, right = [_normalise_sentence(part) for part in body.split(" and ", 1)]
    if not left or not right:
        return None

    left_words = [match.group(0) for match in _TOKEN.finditer(left)]
    verb_index = _find_first_verb_index(left_words)
    if verb_index is None:
        return None

    subject = " ".join(left_words[:verb_index])
    if not subject or subject.lower() in _AMBIGUOUS_PRONOUNS:
        return None

    right_words = [match.group(0) for match in _TOKEN.finditer(right)]
    if not right_words or not _is_predicate_start(right_words):
        return None

    return _finalize_decomposition(body, [left, f"{subject} {right}"])


def _is_predicate_start(words: list[str]) -> bool:
    first = words[0].lower()
    if first in {"not"} and len(words) > 1:
        return _is_verb_like(words[1])
    if first in _AUXILIARY_VERBS and len(words) > 1:
        return words[1].lower() == "not" or _is_verb_like(words[1])
    return _is_verb_like(first)


def _decompose_safe_object_list(body: str) -> list[str] | None:
    if "," not in body or " and " not in body:
        return None

    words = [match.group(0) for match in _TOKEN.finditer(body)]
    verb_index = _find_first_verb_index(words)
    if verb_index is None:
        return None

    subject = " ".join(words[:verb_index])
    verb = words[verb_index].lower()
    if not subject or verb not in _SAFE_OBJECT_LIST_VERBS:
        return None

    prefix = f"{subject} {words[verb_index]}"
    object_text = body[len(prefix) :].strip()
    if not object_text:
        return None
    if _has_trailing_modifier(object_text):
        return None

    parts = [
        _normalise_sentence(part) for part in _re.split(r",\s*(?:and\s+)?", object_text)
    ]
    if len(parts) < 3 or any(not part for part in parts):
        return None
    if any(_looks_like_version_or_numeric_object(part) for part in parts):
        return None
    if any(
        _is_predicate_start([match.group(0) for match in _TOKEN.finditer(part)])
        for part in parts
    ):
        return None

    return _finalize_decomposition(body, [f"{prefix} {part}" for part in parts])


def _has_trailing_modifier(object_text: str) -> bool:
    tail = object_text.rsplit(",", maxsplit=1)[-1]
    return bool(
        _re.search(
            rf"\b(?:{'|'.join(_TRAILING_MODIFIER_PREPOSITIONS)})\b\s+\w+",
            tail,
            flags=_re.IGNORECASE,
        )
    )


def _looks_like_version_or_numeric_object(part: str) -> bool:
    return bool(_re.fullmatch(r"(?:python\s+)?\d+(?:\.\d+)*", part, _re.IGNORECASE))


def _claimify_units(response: str) -> list[tuple[str, str]]:
    units: list[tuple[str, str]] = []
    lead_subject: str | None = None
    lead_verb: str | None = None

    for raw_line in response.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        item = _clean_claim_text(line)
        lead = _parse_lead_in(item)
        if lead is not None:
            lead_subject, lead_verb = lead
            continue
        if lead_subject is not None and lead_verb is not None and item:
            claim = _apply_lead_in(lead_subject, lead_verb, item)
            if claim is not None:
                units.append((_ensure_terminal_punctuation(claim), item))
                continue
        lead_subject = None
        lead_verb = None
        units.extend((sentence, sentence) for sentence in _regex_split(item))
    return units


def _strip_bullet_prefix(line: str) -> str:
    match = _BULLET_PREFIX.match(line)
    return match.group(1).strip() if match else line.strip()


def _parse_lead_in(line: str) -> tuple[str, str] | None:
    if not line.endswith(":"):
        return None
    body = line[:-1].strip()
    words = [match.group(0) for match in _TOKEN.finditer(body)]
    if len(words) < 2:
        return None
    verb_index = _find_first_verb_index(words)
    if verb_index is None:
        return None
    verb = words[verb_index].lower()
    if verb not in _LEAD_IN_VERBS:
        return None
    subject = " ".join(words[:verb_index])
    if not subject or subject.lower() in _AMBIGUOUS_PRONOUNS:
        return None
    return subject, words[verb_index]


def _apply_lead_in(subject: str, verb: str, item: str) -> str | None:
    if item.endswith(":"):
        return None
    words = [match.group(0) for match in _TOKEN.finditer(item)]
    if not words:
        return None
    if verb.lower() in _SUBJECT_ONLY_LEAD_IN_VERBS:
        if not _is_predicate_start(words):
            return None
        return f"{subject} {item}"
    if _is_predicate_start(words):
        return f"{subject} {item}"
    return f"{subject} {verb} {item}"


def _deduplicate_claims(claims: list[ExtractedClaim]) -> list[ExtractedClaim]:
    seen: set[str] = set()
    deduplicated: list[ExtractedClaim] = []
    for claim in claims:
        key = _normalise_sentence(claim.text).lower().rstrip(_TERMINAL_PUNCTUATION)
        if key in seen:
            continue
        seen.add(key)
        deduplicated.append(claim)
    return deduplicated


def _ensure_punkt_tab() -> None:
    """Download ``punkt_tab`` tokeniser data if not already present.

    NLTK 3.9+ (and Python 3.12+) replaced the pickled ``punkt`` package with
    a pickle-free ``punkt_tab`` package. Calling ``sent_tokenize`` without it
    raises a ``LookupError``.

    This function is called once during :class:`ClaimExtractor` construction
    when ``extractor="nltk"``. All subsequent ``extract()`` calls are fully
    local with no network access.

    Raises:
        ImportError: If nltk is not installed.
    """
    try:
        import nltk as _nltk
    except ImportError as exc:
        raise ImportError(
            "Dokis: extractor='nltk' requires the nltk package. "
            "Install it with: pip install 'dokis[nltk]'"
        ) from exc
    try:
        _nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        logger.info("Dokis: downloading NLTK punkt_tab data (one-time setup).")
        _nltk.download("punkt_tab", quiet=True)


class ClaimExtractor:
    """Splits an LLM response string into atomic claim sentences.

    Default path uses the deterministic ``"claimify"`` extractor - fully
    local, zero extra dependencies, no network calls, and no model downloads.
    The legacy ``"regex"`` path remains available for users who want the old
    sentence splitter. The optional ``"nltk"`` path provides higher accuracy
    sentence tokenization at the cost of requiring
    ``pip install dokis[nltk]`` and a one-time punkt_tab download.
    When ``config.extractor="llm"`` the user-supplied ``config.llm_fn`` is
    called instead; Dokis never hardcodes any LLM client.

    Args:
        config: Runtime configuration. Controls which extraction strategy is
            used and, for the ``"llm"`` path, supplies the callable.
    """

    def __init__(self, config: Config) -> None:
        self._config = config
        if config.extractor == "nltk":
            _ensure_punkt_tab()

    def extract(self, response: str | None) -> list[str]:
        """Extract atomic claim sentences from an LLM response.

        Sentences shorter than 8 words are discarded - they are typically
        transitional phrases that cannot be grounded in a source chunk.

        Args:
            response: The raw LLM-generated response text to split.
                ``None`` is treated identically to an empty string and
                returns ``[]``.

        Returns:
            A list of atomic claim strings. Empty when ``response`` is empty
            or contains no sentences of sufficient length.
        """
        return [claim.text for claim in self.extract_detailed(response)]

    def extract_detailed(self, response: str | None) -> list[ExtractedClaim]:
        """Extract claims with internal provenance metadata."""
        if response is None:
            return []
        if not response.strip():
            return []
        if self._config.extractor == "llm":
            return self._extract_detailed_with_llm(response)
        if self._config.extractor == "nltk":
            return self._extract_detailed_with_nltk(response)
        if self._config.extractor == "claimify":
            return self._extract_detailed_with_claimify(response)
        return self._extract_detailed_with_regex(response)

    def _extract_with_regex(self, response: str) -> list[str]:
        return [claim.text for claim in self._extract_detailed_with_regex(response)]

    def _extract_detailed_with_regex(self, response: str) -> list[ExtractedClaim]:
        sentences = _regex_split(response)
        return [
            ExtractedClaim(text=s, source_sentence=s, extraction_method="regex")
            for s in sentences
            if len(s.split()) >= _MIN_WORDS
        ]

    def _extract_with_nltk(self, response: str) -> list[str]:
        return [claim.text for claim in self._extract_detailed_with_nltk(response)]

    def _extract_detailed_with_nltk(self, response: str) -> list[ExtractedClaim]:
        sentences: list[str] = nltk.sent_tokenize(response)
        return [
            ExtractedClaim(text=s, source_sentence=s, extraction_method="nltk")
            for s in sentences
            if len(s.split()) >= _MIN_WORDS
        ]

    def _extract_with_llm(self, response: str) -> list[str]:
        return [claim.text for claim in self._extract_detailed_with_llm(response)]

    def _extract_detailed_with_llm(self, response: str) -> list[ExtractedClaim]:
        if self._config.llm_fn is None:
            raise ValueError(
                "config.extractor='llm' but config.llm_fn is None. "
                "Provide a llm_fn callable in your Config."
            )
        prompt = _LLM_PROMPT_TEMPLATE.format(response=response)
        raw = self._config.llm_fn(prompt)
        lines = [line.strip() for line in raw.splitlines()]
        return [
            ExtractedClaim(text=line, source_sentence=line, extraction_method="llm")
            for line in lines
            if len(line.split()) >= _MIN_WORDS
        ]

    def _extract_with_claimify(self, response: str) -> list[str]:
        return [claim.text for claim in self._extract_detailed_with_claimify(response)]

    def _extract_detailed_with_claimify(self, response: str) -> list[ExtractedClaim]:
        claims: list[ExtractedClaim] = []
        for sentence, source_sentence in _claimify_units(response):
            if not _is_verifiable_sentence(sentence):
                continue
            for claim_text in _decompose_compound_claim(sentence):
                if not _is_verifiable_sentence(claim_text):
                    continue
                claims.append(
                    ExtractedClaim(
                        text=claim_text,
                        source_sentence=source_sentence,
                        extraction_method="claimify",
                        flags=_claim_flags(claim_text),
                    )
                )
        return _deduplicate_claims(claims)
