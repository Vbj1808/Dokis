"""LangChain adapter for Dokis provenance middleware."""

from __future__ import annotations

import logging
from typing import Any, cast

try:
    from langchain_core.callbacks import CallbackManagerForRetrieverRun
    from langchain_core.documents import Document
    from langchain_core.retrievers import BaseRetriever
except ImportError as e:
    raise ImportError(
        "Install the LangChain adapter: pip install dokis[langchain]"
    ) from e

from dokis.config import Config
from dokis.core.enforcer import DomainEnforcer
from dokis.models import Chunk

logger = logging.getLogger(__name__)

 
class ProvenanceRetriever(BaseRetriever): 
    """LangChain retriever that enforces source provenance via DomainEnforcer.

    Wraps any existing LangChain ``BaseRetriever`` and filters its results
    through Dokis's :class:`~dokis.core.enforcer.DomainEnforcer` before
    returning them to the chain. Documents whose source URL is not on the
    allowlist are silently removed. When ``allowed_domains`` is empty all
    documents pass through unmodified.

    Args:
        base_retriever: The underlying retriever to delegate to.
        config: Dokis configuration controlling allowed domains.
        url_metadata_key: The metadata key that holds the document's source
            URL. Defaults to ``"source"``. If the key is absent a warning is
            logged and the document is treated as an unknown domain (blocked
            when ``allowed_domains`` is set).

    Example::

        retriever = ProvenanceRetriever(
            base_retriever=your_retriever,
            config=Config(allowed_domains=["example.com"]),
        )
        docs = retriever.invoke("What is aspirin?")
    """

    base_retriever: BaseRetriever
    config: Config
    url_metadata_key: str = "source"

    model_config = {"arbitrary_types_allowed": True}

    def __init__(
        self,
        base_retriever: BaseRetriever,
        config: Config | None = None,
        url_metadata_key: str = "source",
        **kwargs: Any,
    ) -> None:
        super_init = cast(Any, super().__init__)
        super_init(
            base_retriever=base_retriever,
            config=config or Config(),
            url_metadata_key=url_metadata_key,
            **kwargs,
        )
        self._enforcer = DomainEnforcer(self.config)

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> list[Document]:
        """Retrieve and filter documents by allowed domain.

        Delegates to the base retriever via ``invoke()``, converts the
        results to Dokis :class:`~dokis.models.Chunk` objects, passes them
        through :class:`~dokis.core.enforcer.DomainEnforcer`, then converts
        the clean chunks back to LangChain :class:`Document` objects.

        Args:
            query: The user's query string.
            run_manager: LangChain callback manager for this retriever run.

        Returns:
            Documents whose source URL is on the allowlist, or all documents
            if ``allowed_domains`` is empty in config.
        """
        raw_docs: list[Document] = self.base_retriever.invoke(query)
        chunks = self._docs_to_chunks(raw_docs)
        clean_chunks, _ = self._enforcer.filter(chunks)
        return self._chunks_to_docs(clean_chunks, raw_docs)

    def _docs_to_chunks(self, docs: list[Document]) -> list[Chunk]:
        """Convert LangChain Documents to Dokis Chunks.

        Documents whose metadata does not contain ``url_metadata_key`` are
        assigned an empty string as ``source_url``; DomainEnforcer will then
        block them when ``allowed_domains`` is non-empty.

        Args:
            docs: LangChain Document list from the base retriever.

        Returns:
            Corresponding list of Dokis :class:`~dokis.models.Chunk` objects.
        """
        chunks: list[Chunk] = []
        for doc in docs:
            source_url = doc.metadata.get(self.url_metadata_key)
            if source_url is None:
                logger.warning(
                    "Dokis: document missing metadata key %r - treating as "
                    "unknown domain.",
                    self.url_metadata_key,
                )
                source_url = ""
            chunks.append(
                Chunk(
                    content=doc.page_content,
                    source_url=str(source_url),
                    metadata=doc.metadata,
                )
            )
        return chunks

    def _chunks_to_docs(
        self,
        chunks: list[Chunk],
        original_docs: list[Document],
    ) -> list[Document]:
        """Convert Dokis Chunks back to LangChain Documents.

        Preserves the original :class:`Document` objects (including any extra
        metadata not stored on the :class:`~dokis.models.Chunk`) by filtering
        ``original_docs`` to only those whose source URL appears in the clean
        chunk set.

        Args:
            chunks: Filtered :class:`~dokis.models.Chunk` list from
                :class:`~dokis.core.enforcer.DomainEnforcer`.
            original_docs: The original Document list used to preserve any
                extra metadata not stored in the Chunk.

        Returns:
            LangChain Document list corresponding to the clean chunks.
        """
        clean_urls = {c.source_url for c in chunks}
        return [
            doc
            for doc in original_docs
            if doc.metadata.get(self.url_metadata_key, "") in clean_urls
        ]
