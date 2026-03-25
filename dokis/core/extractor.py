"""Claim extractor - splits an LLM response into atomic claim sentences."""

from __future__ import annotations

import contextlib
import logging
import re as _re

# nltk is an optional dependency (pip install dokis[nltk]).
# The module-level import is guarded so that the package loads cleanly when
# nltk is not installed. _ensure_punkt_tab() raises ImportError with a clear
# message if extractor="nltk" is requested without nltk present.
with contextlib.suppress(ImportError):
    import nltk  # type: ignore[import-untyped]

from dokis.config import Config

logger = logging.getLogger(__name__)

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

# Regex sentence boundary pattern for the default "regex" extractor path.
# Matches sentence boundaries while avoiding:
# - Common abbreviations (Dr., Mr., Ms., etc.): [A-Z][a-z]\. lookbehind
# - Single letter initials (J. Smith): \s[A-Z]\. lookbehind
# - Numbered items (Fig. 3): not followed by a digit
# - Decimal numbers (3.14, pH 7.4): protected naturally - the digit after
#   the decimal point means \s+ cannot match at that position.
# All lookbehinds are fixed-width for Python re compatibility.
_SENTENCE_SPLIT = _re.compile(
    r"(?<![A-Z][a-z]\.)" # not preceded by Uppercase+lowercase+period (Dr., Mr., etc.)
    r"(?<!\s[A-Z]\.)"    # not preceded by whitespace+Uppercase+period (J. initial)
    r"(?<=[.?!])"        # must follow sentence-ending punctuation
    r"(?!\d)"            # not followed by a digit
    r"\s+"
)


def _regex_split(text: str) -> list[str]:
    """Split text into sentences using a punctuation-boundary regex.

    Handles common scientific and medical abbreviations (Dr., Mr., Ms.),
    decimal numbers (7.4, 3.14), and single initials (J. Smith) without
    false splits. Uses only fixed-width lookbehinds for Python re
    compatibility. Remaining false splits (e.g. vs.) are caught by the
    8-word minimum filter downstream.

    Args:
        text: Raw input text.

    Returns:
        List of sentence strings.
    """
    return [s.strip() for s in _SENTENCE_SPLIT.split(text) if s.strip()]


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

    Default path uses a fast regex sentence splitter - fully local, zero
    extra dependencies, no network calls. The optional ``"nltk"`` path
    provides higher accuracy at the cost of requiring
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
        if response is None:
            return []
        if not response.strip():
            return []
        if self._config.extractor == "llm":
            return self._extract_with_llm(response)
        if self._config.extractor == "nltk":
            return self._extract_with_nltk(response)
        return self._extract_with_regex(response)

    def _extract_with_regex(self, response: str) -> list[str]:
        sentences = _regex_split(response)
        return [s for s in sentences if len(s.split()) >= _MIN_WORDS]

    def _extract_with_nltk(self, response: str) -> list[str]:
        sentences: list[str] = nltk.sent_tokenize(response)
        return [s for s in sentences if len(s.split()) >= _MIN_WORDS]

    def _extract_with_llm(self, response: str) -> list[str]:
        if self._config.llm_fn is None:
            raise ValueError(
                "config.extractor='llm' but config.llm_fn is None. "
                "Provide a llm_fn callable in your Config."
            )
        prompt = _LLM_PROMPT_TEMPLATE.format(response=response)
        raw = self._config.llm_fn(prompt)
        lines = [line.strip() for line in raw.splitlines()]
        return [line for line in lines if len(line.split()) >= _MIN_WORDS]
