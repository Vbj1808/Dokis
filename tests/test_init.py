"""Tests for module-level functions in dokis/__init__.py."""

from __future__ import annotations

import threading
from unittest.mock import patch

import pytest

import dokis
from dokis.config import Config
from dokis.models import Chunk, ProvenanceResult


def test_configure_is_thread_safe() -> None:
    """Concurrent configure() calls must not raise or corrupt state."""
    errors: list[Exception] = []

    def set_config(rate: float) -> None:
        try:
            dokis.configure(Config(min_citation_rate=rate))
        except Exception as exc:
            errors.append(exc)

    threads = [
        threading.Thread(target=set_config, args=(i / 10,)) for i in range(10)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == [], f"Thread errors: {errors}"


def test_configure_never_called_audit_defaults_gracefully(
    sample_chunks: list[Chunk],
    grounded_response: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """audit() must not crash when configure() has never been called."""
    import dokis as _d

    monkeypatch.setattr(_d, "_default_config", None)
    monkeypatch.setattr(_d, "_middleware_cache", {})
    result = _d.audit("query", sample_chunks, grounded_response)
    assert isinstance(result, ProvenanceResult)


def test_cache_is_safe_after_config_gc(
    sample_chunks: list[Chunk],
    grounded_response: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A new Config at the same memory address must not hit a stale cache."""
    import gc

    monkeypatch.setattr(dokis, "_middleware_cache", {})

    config_a = dokis.Config(min_citation_rate=0.0, matcher="semantic")
    dokis.audit("q", sample_chunks, grounded_response, config=config_a)

    # Delete config_a and force GC to potentially reuse its memory.
    del config_a
    gc.collect()

    # Create a new config — may or may not share id with deleted config_a.
    config_b = dokis.Config(min_citation_rate=1.0, matcher="semantic")
    result = dokis.audit("q", sample_chunks, grounded_response, config=config_b)

    # Must use config_b's min_citation_rate=1.0, not config_a's 0.0.
    assert result.min_citation_rate == 1.0


def test_same_config_object_reuses_cached_middleware(
    sample_chunks: list[Chunk],
    grounded_response: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The same Config instance must return the same middleware object."""
    monkeypatch.setattr(dokis, "_middleware_cache", {})

    config = dokis.Config(min_citation_rate=0.0, matcher="semantic")
    dokis.audit("q", sample_chunks, grounded_response, config=config)
    dokis.audit("q", sample_chunks, grounded_response, config=config)

    cache_key = (id(config), config.model_dump_json())
    assert cache_key in dokis._middleware_cache  # type: ignore[attr-defined]
    assert len(dokis._middleware_cache) == 1  # type: ignore[attr-defined]


def test_module_level_audit_reuses_middleware_for_same_config(
    sample_chunks: list[Chunk],
    grounded_response: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two audit() calls with the same Config instance must not reload the model."""
    monkeypatch.setattr(dokis, "_middleware_cache", {})

    config = dokis.Config(min_citation_rate=0.0, matcher="semantic")
    dokis.configure(config)

    original_init = dokis.ProvenanceMiddleware.__init__
    construction_count = 0

    def counting_init(self: dokis.ProvenanceMiddleware, cfg: Config) -> None:  # type: ignore[no-untyped-def]
        nonlocal construction_count
        construction_count += 1
        original_init(self, cfg)

    with patch.object(dokis.ProvenanceMiddleware, "__init__", counting_init):
        monkeypatch.setattr(dokis, "_middleware_cache", {})
        dokis.audit("q", sample_chunks, grounded_response)
        dokis.audit("q", sample_chunks, grounded_response)

    assert construction_count == 1, (
        "ProvenanceMiddleware was constructed more than once for the "
        "same Config"
    )
