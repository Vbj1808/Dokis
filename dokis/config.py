"""Dokis configuration model."""

from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlparse

if sys.version_info >= (3, 11):
    import tomllib
else:
    try:
        import tomllib
    except ImportError:
        import tomli as tomllib

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from typing_extensions import Self


class Config(BaseModel):
    """Dokis runtime configuration.

    All fields have safe defaults so that ``Config()`` produces a working
    zero-config instance - no filtering, 80% minimum citation rate.

    Args:
        allowed_domains: Allowlist of hostnames. Leading ``www.`` is stripped
            on ingestion. An empty list means no filtering - all chunks pass.
        min_citation_rate: Minimum fraction of claims that must be grounded for
            the audit to pass. Must be in ``[0.0, 1.0]``. Defaults to ``0.80``.
        claim_threshold: Minimum score for a claim to be considered
            supported. Meaning differs by matcher:

            - ``matcher="semantic"``: cosine similarity in ``[0.0, 1.0]``.
              Geometrically grounded - 0.72 means the vectors are well
              aligned. Tune in 0.65–0.85 range.
            - ``matcher="bm25"``: normalised BM25 score in ``[0.0, 1.0]``
              where 1.0 = best-matching chunk for this query. Relative,
              not absolute. Tune in 0.3–0.5 range. Values above 0.5
              trigger a WARNING at construction time.

            Must be in ``[0.0, 1.0]``. Defaults to ``0.72``.
        extractor: Sentence extraction strategy. ``"regex"`` uses a fast
            punctuation-boundary splitter (default, zero extra dependencies).
            ``"nltk"`` uses NLTK for higher accuracy (requires
            ``pip install dokis[nltk]``). ``"llm"`` delegates to a
            user-supplied callable - never hardcodes any LLM client.
        matcher: Claim matching strategy. ``"bm25"`` uses BM25 lexical
            scoring via bm25s (default, zero cold start, zero model download).
            ``"semantic"`` uses SentenceTransformer dense cosine similarity
            (requires ``pip install dokis[semantic]``).
        model: SentenceTransformer model name used when ``matcher="semantic"``.
        enforcement_mode: Runtime behavior when provenance policy fails.
            ``"audit"`` always returns a result and never raises.
            ``"guardrail"`` returns a result and marks policy failure in the
            result. ``"enforce"`` fails closed by raising
            :class:`~dokis.exceptions.ComplianceViolation`.
        max_source_age_days: Optional freshness threshold for supporting
            evidence. When set, Dokis derives source age from chunk metadata
            and marks support as stale once the source age exceeds this many
            days. ``None`` disables freshness checks entirely.
        stale_source_action: Whether stale supporting evidence should only be
            reported (``"warn"``) or should fail trust policy
            (``"fail"``). Ignored when ``max_source_age_days`` is ``None``.
        source_date_metadata_key: Optional metadata key to check first when
            deriving source dates from chunk metadata. Dokis still falls back
            to common local keys like ``year`` and ``published_at``.
        fail_on_violation: Backwards-compatible alias for enforce mode. Kept
            for existing configs and examples. If both settings are present,
            ``enforcement_mode`` takes precedence.
        domain: Optional label for the knowledge domain (e.g. ``"oncology"``).
            Carried through to :class:`~dokis.models.ProvenanceResult` for
            downstream use.
        llm_fn: User-supplied callable used when ``extractor="llm"``. Must
            accept a prompt string and return the extracted claims as a
            newline-separated string. Ignored unless
            ``extractor="llm"``.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    allowed_domains: list[str] = Field(default_factory=list)
    min_citation_rate: float = 0.80
    claim_threshold: float = 0.35
    extractor: Literal["regex", "nltk", "llm"] = "regex"
    matcher: Literal["bm25", "semantic"] = "bm25"
    model: str = "all-MiniLM-L6-v2"
    enforcement_mode: Literal["audit", "guardrail", "enforce"] | None = None
    max_source_age_days: int | None = None
    stale_source_action: Literal["warn", "fail"] = "warn"
    source_date_metadata_key: str | None = None
    fail_on_violation: bool = False
    domain: str | None = None
    llm_fn: Callable[[str], str] | None = Field(default=None, exclude=True)

    @field_validator("min_citation_rate", "claim_threshold", mode="after")
    @classmethod
    def _validate_rate(cls, value: float) -> float:
        if not 0.0 <= value <= 1.0:
            raise ValueError("Value must be between 0.0 and 1.0 inclusive.")
        return value

    @field_validator("max_source_age_days", mode="after")
    @classmethod
    def _validate_max_source_age_days(cls, value: int | None) -> int | None:
        if value is not None and value < 0:
            raise ValueError("max_source_age_days must be non-negative.")
        return value

    @model_validator(mode="after")
    def _strip_www(self) -> Self:
        """Strip leading ``www.`` from every entry in ``allowed_domains``."""
        self.allowed_domains = [_strip_www_prefix(d) for d in self.allowed_domains]
        return self

    @model_validator(mode="after")
    def _resolve_enforcement_mode(self) -> Self:
        """Keep the new enforcement mode and legacy bool in sync."""
        if self.enforcement_mode is None:
            self.enforcement_mode = "enforce" if self.fail_on_violation else "guardrail"
        self.fail_on_violation = self.enforcement_mode == "enforce"
        return self

    @classmethod
    def from_yaml(cls, path: str | Path) -> Config:
        """Load configuration from a TOML file.

        The method is named from_yaml for backwards compatibility but now
        expects a TOML file. Pass a path ending in .toml.

        Args:
            path: Path to the TOML configuration file.

        Returns:
            A validated :class:`Config` instance.

        Raises:
            FileNotFoundError: If path does not exist.
            ValueError: If a .yaml or .yml path is passed - use TOML instead.
            pydantic.ValidationError: If the file contents fail validation.
        """
        path = Path(path)
        if path.suffix in {".yaml", ".yml"}:
            raise ValueError(
                f"Dokis no longer supports YAML config files. "
                f"Convert {path.name} to TOML format (.toml) and pass the "
                f"new path. See README for the equivalent TOML config."
            )
        with open(path, "rb") as fh:
            data: dict[str, Any] = tomllib.load(fh) or {}
        return cls.model_validate(data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Config:
        """Load configuration from a plain Python dictionary.

        Args:
            data: Mapping of config field names to values.

        Returns:
            A validated :class:`Config` instance.

        Raises:
            pydantic.ValidationError: If ``data`` fails validation.
        """
        return cls.model_validate(data)


def _strip_www_prefix(domain: str) -> str:
    """Remove a leading ``www.`` from a domain string.

    Works on bare hostnames and on full URLs alike by delegating to
    :func:`urllib.parse.urlparse` when a scheme is present.

    Args:
        domain: A bare hostname or full URL string.

    Returns:
        The hostname with any ``www.`` prefix removed.
    """
    parsed = urlparse(domain)
    host = parsed.netloc if parsed.netloc else parsed.path
    if host.startswith("www."):
        host = host[4:]
    return host
