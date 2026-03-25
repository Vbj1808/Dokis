"""Shared pytest fixtures for the Dokis test suite."""

import pytest

from dokis.config import Config
from dokis.models import Chunk


@pytest.fixture
def sample_chunks() -> list[Chunk]:
    """Three chunks: two from allowlisted domains, one from a spam domain."""
    return [
        Chunk(
            content="Aspirin reduces fever by inhibiting COX enzymes.",
            source_url="https://pubmed.ncbi.nlm.nih.gov/12345",
        ),
        Chunk(
            content="Ibuprofen is a nonsteroidal anti-inflammatory drug.",
            source_url="https://cochrane.org/review/67890",
        ),
        Chunk(
            content="Buy cheap meds here.",
            source_url="https://discountpharma.biz/meds",
        ),
    ]


@pytest.fixture
def strict_config() -> Config:
    """Config with a two-domain allowlist and an 85% citation requirement."""
    return Config(
        allowed_domains=["pubmed.ncbi.nlm.nih.gov", "cochrane.org"],
        min_citation_rate=0.85,
        claim_threshold=0.72,
    )


@pytest.fixture
def permissive_config() -> Config:
    """Zero-config instance — no filtering, 80% default citation rate."""
    return Config()


@pytest.fixture
def semantic_config() -> Config:
    """Config using the semantic (SentenceTransformer) matcher with no filtering."""
    return Config(matcher="semantic", min_citation_rate=0.0)


@pytest.fixture
def grounded_response() -> str:
    """A response whose claims are semantically supported by sample_chunks.

    Used in middleware integration tests to produce a ProvenanceResult
    with a high compliance rate without needing to mock the matcher.
    """
    return (
        "Aspirin inhibits the COX enzymes, which reduces fever and inflammation. "
        "Ibuprofen is a widely used nonsteroidal anti-inflammatory medication."
    )
