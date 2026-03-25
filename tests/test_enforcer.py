"""Full test suite for DomainEnforcer."""

import logging

import pytest

from dokis.config import Config
from dokis.core.enforcer import DomainEnforcer
from dokis.models import Chunk


def test_enforcer_blocks_unlisted_domain(
    sample_chunks: list[Chunk], strict_config: Config
) -> None:
    enforcer = DomainEnforcer(strict_config)
    clean, blocked = enforcer.filter(sample_chunks)
    assert "https://discountpharma.biz/meds" in blocked
    assert all(c.source_url != "https://discountpharma.biz/meds" for c in clean)


def test_enforcer_passes_allowlisted_domain(
    sample_chunks: list[Chunk], strict_config: Config
) -> None:
    enforcer = DomainEnforcer(strict_config)
    clean, _ = enforcer.filter(sample_chunks)
    clean_urls = {c.source_url for c in clean}
    assert "https://pubmed.ncbi.nlm.nih.gov/12345" in clean_urls
    assert "https://cochrane.org/review/67890" in clean_urls


def test_enforcer_passes_all_when_no_allowlist(
    sample_chunks: list[Chunk], permissive_config: Config
) -> None:
    enforcer = DomainEnforcer(permissive_config)
    clean, blocked = enforcer.filter(sample_chunks)
    assert len(clean) == len(sample_chunks)
    assert blocked == []


def test_enforcer_handles_malformed_url_without_raising(
    strict_config: Config,
) -> None:
    chunks = [Chunk(content="text", source_url=":::not::a::url:::")]
    enforcer = DomainEnforcer(strict_config)
    # Must not raise; the malformed URL should be treated as blocked.
    clean, blocked = enforcer.filter(chunks)
    assert clean == []
    assert ":::not::a::url:::" in blocked


def test_enforcer_strips_www_prefix(strict_config: Config) -> None:
    # www.pubmed.ncbi.nlm.nih.gov should match the allowlist entry
    # pubmed.ncbi.nlm.nih.gov (which has www. stripped at Config ingestion).
    chunks = [
        Chunk(
            content="Some medical content about aspirin and fever.",
            source_url="https://www.pubmed.ncbi.nlm.nih.gov/99999",
        )
    ]
    enforcer = DomainEnforcer(strict_config)
    clean, blocked = enforcer.filter(chunks)
    assert len(clean) == 1
    assert blocked == []


def test_enforcer_returns_blocked_urls_list(
    sample_chunks: list[Chunk], strict_config: Config
) -> None:
    enforcer = DomainEnforcer(strict_config)
    _, blocked = enforcer.filter(sample_chunks)
    assert blocked == ["https://discountpharma.biz/meds"]


def test_enforcer_empty_input_returns_empty_lists(strict_config: Config) -> None:
    enforcer = DomainEnforcer(strict_config)
    clean, blocked = enforcer.filter([])
    assert clean == []
    assert blocked == []


def test_enforcer_malformed_url_log_does_not_contain_full_path(
    strict_config: Config,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """WARNING log for malformed URLs must not emit the full URL path."""
    sensitive_url = ":::not::a::url:::/patient/99999/record"
    chunks = [Chunk(content="text", source_url=sensitive_url)]
    enforcer = DomainEnforcer(strict_config)
    with caplog.at_level(logging.WARNING, logger="dokis.core.enforcer"):
        enforcer.filter(chunks)
    for record in caplog.records:
        assert "/patient/99999/record" not in record.message


def test_enforcer_all_blocked_when_none_on_allowlist() -> None:
    config = Config(allowed_domains=["trusted.org"])
    chunks = [
        Chunk(content="Content A.", source_url="https://untrusted-a.com/1"),
        Chunk(content="Content B.", source_url="https://untrusted-b.com/2"),
    ]
    enforcer = DomainEnforcer(config)
    clean, blocked = enforcer.filter(chunks)
    assert clean == []
    assert len(blocked) == 2
