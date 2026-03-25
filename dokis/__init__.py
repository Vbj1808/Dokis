"""Dokis — lightweight RAG source provenance governance middleware.

Public API surface. All names a user needs are importable directly from
``dokis``. Nothing from submodules should ever be imported by users.

Usage::

    import dokis

    # Zero-config audit
    result = dokis.audit(query, chunks, response)

    # With config
    config = dokis.Config.from_yaml("provenance.yaml")
    clean_chunks = dokis.filter(raw_chunks, config)
    result = dokis.audit(query, clean_chunks, response, config=config)

    if not result.passed:
        raise dokis.ComplianceViolation(result)
"""

from __future__ import annotations

import threading

from dokis.config import Config
from dokis.exceptions import ComplianceViolation, DomainViolation
from dokis.middleware import ProvenanceMiddleware
from dokis.models import Chunk, Claim, ProvenanceResult

# Module-level default configuration set via configure().
_default_config: Config | None = None
_config_lock: threading.Lock = threading.Lock()

# Cache of ProvenanceMiddleware instances keyed on (id(config), config_json).
# The id alone is not safe - Python reuses memory addresses after GC, so a
# new Config object can share the id of a deleted one. Including the JSON
# serialization makes stale hits impossible.
_middleware_cache: dict[tuple[int, str], ProvenanceMiddleware] = {}


def configure(config: Config) -> None:
    """Set a module-level default configuration.

    Subsequent calls to :func:`filter` and :func:`audit` that do not pass
    an explicit ``config`` argument will use this instance.

    Args:
        config: A :class:`Config` instance to use as the module default.
    """
    global _default_config
    with _config_lock:
        _default_config = config


def _resolve_config(config: Config | None) -> Config:
    """Return the effective Config, falling back to the module default.

    When neither ``config`` nor a module-level default has been set, a
    zero-config :class:`Config` is created once and stored as the module
    default so that repeated calls return the *same* object - enabling the
    ``_get_middleware`` cache to function correctly.

    Args:
        config: Caller-supplied config, or ``None``.

    Returns:
        ``config`` if provided; the module-level default if set; otherwise a
        zero-config :class:`Config` instance (stored for subsequent reuse).
    """
    global _default_config
    if config is not None:
        return config
    with _config_lock:
        if _default_config is None:
            _default_config = Config()
        return _default_config


def _get_middleware(config: Config) -> ProvenanceMiddleware:
    """Return a cached ProvenanceMiddleware for the given Config.

    Keyed on (id(config), config.model_dump_json()) so that:
    - Same Config object → same instance (no model reload).
    - Reused memory address with different config values → new instance
      (stale-ID bug is impossible).

    Note: llm_fn is excluded from model_dump_json() via exclude=True,
    so two configs that differ only in llm_fn share a cache entry.
    This is acceptable - llm_fn affects the extractor, not the matcher
    or middleware construction.

    Args:
        config: The Config instance to look up or create for.

    Returns:
        A ProvenanceMiddleware constructed from config.
    """
    cache_key = (id(config), config.model_dump_json())
    with _config_lock:
        mw = _middleware_cache.get(cache_key)
        if mw is None:
            mw = ProvenanceMiddleware(config)
            _middleware_cache[cache_key] = mw
    return mw


def filter(  # noqa: A001 - intentionally shadows builtin for API ergonomics
    chunks: list[Chunk],
    config: Config | None = None,
) -> list[Chunk]:
    """Remove chunks whose source URL is not on the allowlist.

    If ``config`` is not supplied, falls back to the instance set by
    :func:`configure`, or a zero-config :class:`Config` if neither was set.

    Args:
        chunks: Raw chunks from your retriever.
        config: Optional :class:`Config`. Defaults to the module-level config
            or a zero-config instance.

    Returns:
        Chunks whose source domain is in ``config.allowed_domains``.
    """
    effective = _resolve_config(config)
    return _get_middleware(effective).filter(chunks)


def audit(
    query: str,
    chunks: list[Chunk],
    response: str,
    config: Config | None = None,
) -> ProvenanceResult:
    """Audit a generated response for source provenance.

    If ``config`` is not supplied, falls back to the instance set by
    :func:`configure`, or a zero-config :class:`Config` if neither was set.

    Args:
        query: The original user query.
        chunks: Retrieved chunks that were used to generate ``response``.
        response: The LLM-generated response to audit.
        config: Optional :class:`Config`. Defaults to the module-level config
            or a zero-config instance.

    Returns:
        A :class:`ProvenanceResult` with per-claim grounding information.
    """
    effective = _resolve_config(config)
    return _get_middleware(effective).audit(query, chunks, response)


__all__ = [
    # Module-level functions
    "configure",
    "filter",
    "audit",
    # Classes
    "Config",
    "Chunk",
    "Claim",
    "ProvenanceResult",
    "ProvenanceMiddleware",
    "ComplianceViolation",
    "DomainViolation",
]

# Version
__version__ = "0.1.0"
