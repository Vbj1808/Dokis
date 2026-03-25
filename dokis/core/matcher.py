"""Claim matcher - BM25 and semantic grounding of claims against chunks."""

from __future__ import annotations

import logging
import os
from typing import Any

import numpy as np

try:
    from sentence_transformers import SentenceTransformer
    _SEMANTIC_AVAILABLE: bool = True
except ImportError:
    _SEMANTIC_AVAILABLE = False

try:
    import bm25s  # type: ignore[import-untyped]
    _BM25_AVAILABLE: bool = True
except ImportError:
    _BM25_AVAILABLE = False

from dokis.config import Config
from dokis.models import Chunk, Claim

logger = logging.getLogger(__name__)

# Upper bounds on input sizes to prevent large matrix allocations from
# exhausting memory. Configurable via environment variables for power users.
_MAX_CHUNKS: int = int(os.environ.get("DOKIS_MAX_CHUNKS", 2_000))
_MAX_CLAIMS: int = int(os.environ.get("DOKIS_MAX_CLAIMS", 500))

# Minimum raw BM25 score the best-matching chunk must achieve before
# normalization. Scores below this floor indicate only stopword overlap
# and are treated as no-match regardless of claim_threshold.
# Overridable via env var for domain tuning.
_MIN_BM25_RAW_SCORE: float = float(
    os.environ.get("DOKIS_MIN_BM25_RAW_SCORE", "0.5")
)

# Maximum number of distinct chunk sets to cache BM25 indexes for.
# Oldest entry is evicted when the limit is reached.
# Overridable via env var for servers with many distinct chunk sets.
_MAX_BM25_CACHE_SIZE: int = int(
    os.environ.get("DOKIS_MAX_BM25_CACHE_SIZE", "32")
)


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute cosine similarity matrix between row-vectors in a and b.

    Args:
        a: Array of shape (m, d).
        b: Array of shape (n, d).

    Returns:
        Similarity matrix of shape (m, n) with values in [-1.0, 1.0].
    """
    a_norm = a / np.linalg.norm(a, axis=1, keepdims=True)
    b_norm = b / np.linalg.norm(b, axis=1, keepdims=True)
    return (a_norm @ b_norm.T).astype(np.float64)  # type: ignore[no-any-return]


class ClaimMatcher:
    """Matches atomic claim strings to their most likely source chunk.

    Supports two strategies controlled by ``config.matcher``:

    - ``"bm25"`` (default): BM25 lexical scoring via bm25s. Zero cold start,
      zero model download. Scores are normalised per query to ``[0.0, 1.0]``.
    - ``"semantic"``: SentenceTransformer dense cosine similarity. Loads the
      model once at construction time. Requires ``pip install dokis[semantic]``.

    Args:
        config: Runtime configuration. Controls which strategy is used and,
            for the semantic path, which model to load.

    Raises:
        ImportError: If ``matcher="semantic"`` is requested but
            ``sentence-transformers`` is not installed.
        ImportError: If ``matcher="bm25"`` is requested but ``bm25s`` is not
            installed (unlikely - bm25s is a core dependency).
    """

    def __init__(self, config: Config) -> None:
        self._config = config
        # SentenceTransformer when matcher="semantic", None otherwise.
        self._model: Any = None
        # BM25 index cache: frozenset(chunk contents) → bm25s.BM25 instance.
        # Avoids rebuilding the index on every match() call when the chunk
        # set is stable across multiple audit() calls.
        self._bm25_cache: dict[frozenset[str], Any] = {}

        if config.matcher == "semantic":
            if not _SEMANTIC_AVAILABLE:
                raise ImportError(
                    "matcher='semantic' requires sentence-transformers. "
                    "Install it with: pip install dokis[semantic]"
                )
            self._model = SentenceTransformer(config.model)
        elif config.matcher == "bm25":
            if not _BM25_AVAILABLE:
                raise ImportError(
                    "matcher='bm25' requires bm25s. "
                    "Install it with: pip install dokis  # bm25s is a core dep"
                )
            if config.claim_threshold > 0.5:
                logger.warning(
                    "Dokis: claim_threshold=%.2f with matcher='bm25'. BM25 scores "
                    "are normalised per-query (best chunk = 1.0), not absolute. "
                    "Values above 0.5 may be overly strict. Consider 0.3–0.5 for "
                    "BM25 or switch to matcher='semantic' for geometric thresholds.",
                    config.claim_threshold,
                )
            if config.model != "all-MiniLM-L6-v2":
                logger.warning(
                    "Dokis: config.model=%r is ignored when matcher='bm25'. "
                    "The model field only applies to matcher='semantic'.",
                    config.model,
                )

    def match(self, claims: list[str], chunks: list[Chunk]) -> list[Claim]:
        """Match each claim to its best-scoring chunk.

        Dispatches to the BM25 or semantic (SentenceTransformer) strategy
        based on config.matcher. Both strategies return Claim objects in the
        same order as the input claims list.

        Args:
            claims: Atomic claim strings from ClaimExtractor.
            chunks: Retrieved chunks to match against.

        Returns:
            A Claim for every input claim string, in the same order.
            Returns an empty list if either claims or chunks is empty.
        """
        if not claims or not chunks:
            return []
        if len(chunks) > _MAX_CHUNKS:
            raise ValueError(
                f"Dokis: chunk count {len(chunks)} exceeds _MAX_CHUNKS "
                f"({_MAX_CHUNKS}). Pass a pre-filtered list or raise "
                "DOKIS_MAX_CHUNKS to override."
            )
        if len(claims) > _MAX_CLAIMS:
            raise ValueError(
                f"Dokis: claim count {len(claims)} exceeds _MAX_CLAIMS "
                f"({_MAX_CLAIMS}). Truncate the response or raise "
                "DOKIS_MAX_CLAIMS to override."
            )
        if self._config.matcher == "semantic":
            return self._match_semantic(claims, chunks)
        return self._match_bm25(claims, chunks)

    def _get_bm25_index(self, chunk_texts: list[str]) -> Any:  # returns bm25s.BM25
        """Return a cached BM25 index for chunk_texts, building if needed.

        The cache key is a frozenset of the chunk text strings. Order
        does not matter for BM25 indexing - the same set of chunks
        always produces the same index regardless of input order.

        Args:
            chunk_texts: The content strings of the chunks to index.

        Returns:
            A bm25s.BM25 instance indexed on chunk_texts.
        """
        cache_key: frozenset[str] = frozenset(chunk_texts)
        if cache_key not in self._bm25_cache:
            if len(self._bm25_cache) >= _MAX_BM25_CACHE_SIZE:
                # Evict the oldest entry (insertion-order dict, Python 3.7+).
                oldest = next(iter(self._bm25_cache))
                del self._bm25_cache[oldest]
            corpus_tokens = bm25s.tokenize(chunk_texts, stopwords="en")
            retriever = bm25s.BM25()
            retriever.index(corpus_tokens)
            self._bm25_cache[cache_key] = retriever
        return self._bm25_cache[cache_key]

    def _match_bm25(
        self, claims: list[str], chunks: list[Chunk]
    ) -> list[Claim]:
        """Match claims to chunks using BM25 lexical scoring.

        Builds a BM25 index over chunk contents, queries it once per claim.
        Raw BM25 scores are only normalised to [0.0, 1.0] when the best
        score for a query clears ``_MIN_BM25_RAW_SCORE`` (default 0.5).
        Below this floor, the claim is unsupported with confidence=0.0 -
        stopword-only overlap is not treated as provenance evidence.

        Args:
            claims: Atomic claim strings.
            chunks: Chunks to index and search.

        Returns:
            One Claim per input claim, in the same order.
        """
        chunk_texts = [c.content for c in chunks]
        retriever = self._get_bm25_index(chunk_texts)

        result: list[Claim] = []
        threshold = self._config.claim_threshold

        for claim_text in claims:
            query_tokens = bm25s.tokenize(claim_text, stopwords="en")
            # Retrieve all chunks ranked by score (descending).
            # doc_indices[0]: original corpus indices in ranked order.
            # scores[0]: corresponding BM25 scores.
            doc_indices, scores = retriever.retrieve(
                query_tokens, k=len(chunks)
            )
            scores_1d: np.ndarray = np.asarray(scores[0], dtype=np.float64)
            indices_1d: np.ndarray = np.asarray(doc_indices[0], dtype=np.int64)

            max_score = float(scores_1d.max()) if scores_1d.size > 0 else 0.0

            if max_score < _MIN_BM25_RAW_SCORE:
                # Raw score below floor - stopword-only overlap or empty corpus.
                # Treat as unsupported regardless of claim_threshold.
                result.append(
                    Claim(
                        text=claim_text,
                        supported=False,
                        confidence=0.0,
                        source_chunk=None,
                        source_url=None,
                    )
                )
                continue

            # Raw score clears floor - normalise to [0.0, 1.0] and apply threshold.
            normalised = scores_1d / max_score
            best_rank = int(normalised.argmax())
            best_orig_idx = int(indices_1d[best_rank])  # original chunk index
            best_score = float(normalised[best_rank])
            supported = best_score >= threshold
            best_chunk = chunks[best_orig_idx] if supported else None
            result.append(
                Claim(
                    text=claim_text,
                    supported=supported,
                    confidence=best_score,
                    source_chunk=best_chunk,
                    source_url=best_chunk.source_url if best_chunk else None,
                )
            )

        return result

    def _match_semantic(
        self, claims: list[str], chunks: list[Chunk]
    ) -> list[Claim]:
        """Match claims to chunks using SentenceTransformer cosine similarity.

        Batch-encodes claims and chunks into dense vectors, computes a full
        (n_claims × n_chunks) cosine similarity matrix, and picks the
        best-scoring chunk per claim.

        Args:
            claims: Atomic claim strings.
            chunks: Chunks to encode and match against.

        Returns:
            One Claim per input claim, in the same order.
        """
        chunk_texts = [c.content for c in chunks]
        claim_embeddings: np.ndarray = np.array(
            self._model.encode(claims, convert_to_numpy=True)
        )
        chunk_embeddings: np.ndarray = np.array(
            self._model.encode(chunk_texts, convert_to_numpy=True)
        )
        sim_matrix: np.ndarray = _cosine_similarity(
            claim_embeddings, chunk_embeddings
        )
        result: list[Claim] = []
        threshold = self._config.claim_threshold
        for i, text in enumerate(claims):
            best_idx = int(np.argmax(sim_matrix[i]))
            best_score = float(sim_matrix[i, best_idx])
            supported = best_score >= threshold
            best_chunk = chunks[best_idx] if supported else None
            result.append(
                Claim(
                    text=text,
                    supported=supported,
                    confidence=best_score,
                    source_chunk=best_chunk,
                    source_url=best_chunk.source_url if best_chunk else None,
                )
            )
        return result
