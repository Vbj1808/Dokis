"""LlamaIndex adapter for Dokis provenance middleware."""

from __future__ import annotations

import logging
from typing import Any

try:
    from llama_index.core.base.base_query_engine import BaseQueryEngine
    from llama_index.core.base.response.schema import RESPONSE_TYPE
except ImportError as e:
    raise ImportError(
        "Install the LlamaIndex adapter: pip install dokis[llamaindex]"
    ) from e

from dokis.config import Config
from dokis.middleware import ProvenanceMiddleware
from dokis.models import Chunk

logger = logging.getLogger(__name__)


class ProvenanceQueryEngine:
    """LlamaIndex query engine that attaches provenance metadata to responses.

    Wraps any LlamaIndex ``BaseQueryEngine`` and runs the generated response
    through Dokis's :class:`~dokis.middleware.ProvenanceMiddleware` after the
    base engine returns. The resulting
    :class:`~dokis.models.ProvenanceResult` is attached to
    ``response.metadata["provenance"]`` so downstream code can inspect
    grounding without coupling to Dokis types directly.

    No LLM calls are made by Dokis. The middleware runs entirely locally
    using sentence embeddings and cosine similarity.

    Args:
        base_engine: The underlying LlamaIndex query engine to delegate to.
        chunks: The source chunks used to build the index. These are passed
            to :meth:`~dokis.middleware.ProvenanceMiddleware.audit` for
            claim matching.
        config: Dokis configuration controlling allowed domains and the
            minimum citation rate. Defaults to a zero-config instance when
            not supplied.

    Example::

        engine = ProvenanceQueryEngine(
            base_engine=your_engine,
            chunks=source_chunks,
            config=Config(allowed_domains=["example.com"]),
        )
        response = engine.query("What is aspirin?")
        result = response.metadata["provenance"]
        print(result.compliance_rate)
    """

    def __init__(
        self,
        base_engine: BaseQueryEngine,
        chunks: list[Chunk],
        config: Config | None = None,
    ) -> None:
        self._base_engine = base_engine
        self._chunks = chunks
        self._middleware = ProvenanceMiddleware(config or Config())

    def query(self, query: str) -> RESPONSE_TYPE:
        """Run a query and attach provenance metadata to the response.

        Delegates to the base engine, converts the response to a string,
        runs it through
        :meth:`~dokis.middleware.ProvenanceMiddleware.audit`, and stores
        the :class:`~dokis.models.ProvenanceResult` in
        ``response.metadata["provenance"]``.

        Args:
            query: The user's query string.

        Returns:
            The original LlamaIndex response object with
            ``metadata["provenance"]`` populated as a
            :class:`~dokis.models.ProvenanceResult`.
        """
        response: Any = self._base_engine.query(query)
        result = self._middleware.audit(
            query=query,
            chunks=self._chunks,
            response=str(response),
        )
        if response.metadata is None:
            response.metadata = {}
        response.metadata["provenance"] = result
        return response
