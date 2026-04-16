"""Tests for Dokis framework adapters.

Both adapters carry optional third-party imports (langchain-core and
llama-index-core) that may not be installed in the test environment.  The
tests below inject minimal mock modules into ``sys.modules`` *before* the
adapter modules are loaded so that the top-level ``try/except ImportError``
guards in each adapter never trigger.

LangChain tests  (5):  test_provenance_retriever_*
LlamaIndex tests (3):  test_provenance_query_engine_*
"""

from __future__ import annotations

import sys
from types import ModuleType
from typing import Any
from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel

from dokis.config import Config
from dokis.models import Chunk, ProvenanceResult


# ---------------------------------------------------------------------------
# Minimal LangChain mock - BaseRetriever must be a Pydantic BaseModel so that
# ProvenanceRetriever (which inherits from it) can declare Pydantic fields.
# ---------------------------------------------------------------------------


class _FakeDocument:
    """Minimal stand-in for langchain_core.documents.Document."""

    def __init__(
        self, page_content: str, metadata: dict[str, Any] | None = None
    ) -> None:
        self.page_content = page_content
        self.metadata: dict[str, Any] = metadata or {}


class _FakeBaseRetriever(BaseModel):
    """Minimal stand-in for langchain_core.retrievers.BaseRetriever.

    Must be a Pydantic BaseModel so that ProvenanceRetriever, which
    subclasses it and declares Pydantic fields, can be constructed.
    """

    model_config = {"arbitrary_types_allowed": True}

    def invoke(
        self, input: str, config: Any = None, **kwargs: Any
    ) -> list[_FakeDocument]:  # noqa: ARG002,A002
        """Stub - overridden by per-test MagicMock."""
        return []

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: Any = None,  # noqa: ARG002
    ) -> list[_FakeDocument]:
        return []


class _FakeCallbackManagerForRetrieverRun:
    """Minimal stand-in for CallbackManagerForRetrieverRun."""


def _install_langchain_mocks() -> None:
    """Inject minimal fake langchain_core modules into sys.modules."""
    lc = ModuleType("langchain_core")
    lc_docs = ModuleType("langchain_core.documents")
    lc_docs.Document = _FakeDocument  # type: ignore[attr-defined]
    lc_retrievers = ModuleType("langchain_core.retrievers")
    lc_retrievers.BaseRetriever = _FakeBaseRetriever  # type: ignore[attr-defined]
    lc_callbacks = ModuleType("langchain_core.callbacks")
    lc_callbacks.CallbackManagerForRetrieverRun = (  # type: ignore[attr-defined]
        _FakeCallbackManagerForRetrieverRun
    )

    sys.modules["langchain_core"] = lc
    sys.modules["langchain_core.documents"] = lc_docs
    sys.modules["langchain_core.retrievers"] = lc_retrievers
    sys.modules["langchain_core.callbacks"] = lc_callbacks


# ---------------------------------------------------------------------------
# Minimal LlamaIndex mock
# ---------------------------------------------------------------------------


class _FakeResponse:
    """Minimal stand-in for a LlamaIndex response object."""

    def __init__(self, text: str) -> None:
        self._text = text
        self.metadata: dict[str, Any] | None = None

    def __str__(self) -> str:
        return self._text


class _FakeBaseQueryEngine:
    """Minimal stand-in for llama_index.core.base.base_query_engine.BaseQueryEngine."""

    def query(self, query: str) -> _FakeResponse:  # noqa: ARG002
        """Stub - overridden by per-test MagicMock."""
        return _FakeResponse("")


def _install_llamaindex_mocks() -> None:
    """Inject minimal fake llama_index modules into sys.modules."""
    li = ModuleType("llama_index")
    li_core = ModuleType("llama_index.core")
    li_core_base = ModuleType("llama_index.core.base")
    li_core_base_qe = ModuleType("llama_index.core.base.base_query_engine")
    li_core_base_qe.BaseQueryEngine = _FakeBaseQueryEngine  # type: ignore[attr-defined]
    li_core_base_resp = ModuleType("llama_index.core.base.response")
    li_core_base_resp_schema = ModuleType("llama_index.core.base.response.schema")
    # RESPONSE_TYPE is a type alias; a plain class is sufficient for isinstance checks.
    li_core_base_resp_schema.RESPONSE_TYPE = _FakeResponse  # type: ignore[attr-defined]

    sys.modules["llama_index"] = li
    sys.modules["llama_index.core"] = li_core
    sys.modules["llama_index.core.base"] = li_core_base
    sys.modules["llama_index.core.base.base_query_engine"] = li_core_base_qe
    sys.modules["llama_index.core.base.response"] = li_core_base_resp
    sys.modules["llama_index.core.base.response.schema"] = li_core_base_resp_schema


# Install mocks before importing the adapter modules so the top-level
# try/except guards in each adapter see the fake packages.
_install_langchain_mocks()
_install_llamaindex_mocks()

# These imports must come AFTER the mock installation above.
from dokis.adapters.langchain import ProvenanceRetriever  # noqa: E402
from dokis.adapters.llamaindex import ProvenanceQueryEngine  # noqa: E402


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def two_domain_config() -> Config:
    """Config that only allows pubmed and cochrane."""
    return Config(
        allowed_domains=["pubmed.ncbi.nlm.nih.gov", "cochrane.org"],
        min_citation_rate=0.0,  # don't fail on low compliance in adapter tests
    )


@pytest.fixture
def medical_chunks() -> list[Chunk]:
    """Two allowlisted chunks and one blocked chunk."""
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


def _make_docs(chunks: list[Chunk], url_key: str = "source") -> list[_FakeDocument]:
    """Convert Chunks to fake LangChain Documents for test setup."""
    return [
        _FakeDocument(
            page_content=c.content,
            metadata={url_key: c.source_url},
        )
        for c in chunks
    ]


# ---------------------------------------------------------------------------
# LangChain adapter tests
# ---------------------------------------------------------------------------


def test_provenance_retriever_filters_blocked_domains(
    two_domain_config: Config,
    medical_chunks: list[Chunk],
) -> None:
    """Domains not on the allowlist must be removed from returned documents."""
    mock_base = MagicMock(spec=_FakeBaseRetriever)
    mock_base.invoke.return_value = _make_docs(medical_chunks)

    retriever = ProvenanceRetriever(
        base_retriever=mock_base,
        config=two_domain_config,
    )
    docs = retriever._get_relevant_documents("query", run_manager=MagicMock())

    returned_urls = {d.metadata["source"] for d in docs}
    assert "https://discountpharma.biz/meds" not in returned_urls
    assert len(docs) == 2


def test_provenance_retriever_passes_allowlisted_domains(
    two_domain_config: Config,
    medical_chunks: list[Chunk],
) -> None:
    """Documents from allowlisted domains must all be present in the result."""
    mock_base = MagicMock(spec=_FakeBaseRetriever)
    mock_base.invoke.return_value = _make_docs(medical_chunks)

    retriever = ProvenanceRetriever(
        base_retriever=mock_base,
        config=two_domain_config,
    )
    docs = retriever._get_relevant_documents("query", run_manager=MagicMock())

    returned_urls = {d.metadata["source"] for d in docs}
    assert "https://pubmed.ncbi.nlm.nih.gov/12345" in returned_urls
    assert "https://cochrane.org/review/67890" in returned_urls


def test_provenance_retriever_passes_all_when_no_allowlist(
    medical_chunks: list[Chunk],
) -> None:
    """With an empty allowed_domains all documents must pass through."""
    mock_base = MagicMock(spec=_FakeBaseRetriever)
    mock_base.invoke.return_value = _make_docs(medical_chunks)

    retriever = ProvenanceRetriever(
        base_retriever=mock_base,
        config=Config(),  # empty allowlist - no filtering
    )
    docs = retriever._get_relevant_documents("query", run_manager=MagicMock())

    assert len(docs) == len(medical_chunks)


def test_provenance_retriever_handles_missing_url_metadata_key(
    two_domain_config: Config,
) -> None:
    """Documents with no source URL key must be treated as blocked domains."""
    doc_without_url = _FakeDocument(
        page_content="Some content.",
        metadata={},  # no "source" key
    )
    mock_base = MagicMock(spec=_FakeBaseRetriever)
    mock_base.invoke.return_value = [doc_without_url]

    retriever = ProvenanceRetriever(
        base_retriever=mock_base,
        config=two_domain_config,
    )
    docs = retriever._get_relevant_documents("query", run_manager=MagicMock())

    # Document with missing URL is blocked when allowed_domains is set.
    assert docs == []


def test_provenance_retriever_returns_documents_not_chunks(
    medical_chunks: list[Chunk],
) -> None:
    """The return type must be a list of Document objects, not Chunk objects."""
    mock_base = MagicMock(spec=_FakeBaseRetriever)
    mock_base.invoke.return_value = _make_docs(medical_chunks)

    retriever = ProvenanceRetriever(
        base_retriever=mock_base,
        config=Config(),
    )
    docs = retriever._get_relevant_documents("query", run_manager=MagicMock())

    assert all(isinstance(d, _FakeDocument) for d in docs)
    assert not any(isinstance(d, Chunk) for d in docs)


# ---------------------------------------------------------------------------
# LlamaIndex adapter tests
# ---------------------------------------------------------------------------


def test_provenance_query_engine_attaches_provenance_to_metadata(
    medical_chunks: list[Chunk],
) -> None:
    """After query(), response.metadata['provenance'] must be populated."""
    fake_response = _FakeResponse(
        "Aspirin inhibits COX enzymes to reduce fever. "
        "Ibuprofen is an anti-inflammatory medication."
    )
    mock_engine = MagicMock(spec=_FakeBaseQueryEngine)
    mock_engine.query.return_value = fake_response

    engine = ProvenanceQueryEngine(
        base_engine=mock_engine,
        chunks=medical_chunks,
        config=Config(min_citation_rate=0.0),
    )
    response = engine.query("What reduces fever?")

    assert response.metadata is not None
    assert "provenance" in response.metadata


def test_provenance_query_engine_delegates_query_to_base_engine(
    medical_chunks: list[Chunk],
) -> None:
    """The base engine's query() must be called exactly once with the query."""
    fake_response = _FakeResponse("Aspirin inhibits COX enzymes.")
    mock_engine = MagicMock(spec=_FakeBaseQueryEngine)
    mock_engine.query.return_value = fake_response

    engine = ProvenanceQueryEngine(
        base_engine=mock_engine,
        chunks=medical_chunks,
        config=Config(min_citation_rate=0.0),
    )
    engine.query("aspirin?")

    mock_engine.query.assert_called_once_with("aspirin?")


def test_provenance_query_engine_result_is_provenance_result_instance(
    medical_chunks: list[Chunk],
) -> None:
    """The value stored in metadata['provenance'] must be a ProvenanceResult."""
    fake_response = _FakeResponse(
        "Aspirin inhibits COX enzymes to reduce fever. "
        "Ibuprofen is an anti-inflammatory medication."
    )
    mock_engine = MagicMock(spec=_FakeBaseQueryEngine)
    mock_engine.query.return_value = fake_response

    engine = ProvenanceQueryEngine(
        base_engine=mock_engine,
        chunks=medical_chunks,
        config=Config(min_citation_rate=0.0),
    )
    response = engine.query("What is ibuprofen?")

    provenance = response.metadata["provenance"]
    assert isinstance(provenance, ProvenanceResult)
