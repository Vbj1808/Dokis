# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.1] - 2026-03-28

### Fixed
- Use absolute asset URLs for PyPI README rendering
- Force Python badge cache refresh

## [0.1.0] - 2026-03-25

### Added

- **`ProvenanceMiddleware`** - orchestrator class that composes all core modules.
  Owns one instance of each core component, created eagerly at `__init__` time.
  Provides `audit()` (sync) and `aaudit()` (async, runs in a thread-pool executor)
  with identical semantics. Raises `ComplianceViolation` when
  `fail_on_violation=True` and the compliance rate falls below threshold.

- **`DomainEnforcer`** (`core/enforcer.py`) - pre-retrieval allowlist filter.
  Parses `source_url` with `urllib.parse.urlparse`, strips leading `www.` and
  matches against `Config.allowed_domains`. Empty allowlist passes all chunks.
  Logs `WARNING` on malformed URLs rather than raising. Returns filtered chunks
  and a list of blocked URLs.

- **`ClaimExtractor`** (`core/extractor.py`) - sentence-level claim splitter
  with three strategies:
  - `"regex"` (default) - fast regex splitter with no network access and no
    extra dependencies. Filters sentences shorter than 8 words.
  - `"nltk"` - higher-accuracy tokenisation via `nltk.sent_tokenize`. Downloads
    `punkt_tab` once on first construction, then fully local. Requires
    `pip install dokis[nltk]`.
  - `"llm"` - delegates to a user-supplied `config.llm_fn` callable. No LLM
    client is bundled inside Dokis.
  Accepts `str | None`; `None` is treated as an empty string and returns `[]`.

- **`ClaimMatcher`** (`core/matcher.py`) - maps each extracted claim to its best
  supporting chunk with two strategies:
  - `"bm25"` (default) - lexical scoring via `bm25s`. Zero model download, zero
    cold start. Scores are normalised to `[0.0, 1.0]` only when the raw max
    score for a query clears a configurable floor (`DOKIS_MIN_BM25_RAW_SCORE`,
    default 0.5); below the floor the claim is marked unsupported with
    `confidence=0.0` so that stopword-only overlap is never treated as
    provenance evidence.
  - `"semantic"` - dense cosine similarity via `SentenceTransformer`. Loads the
    model once at construction, batch-encodes all claims and chunks and computes
    the full similarity matrix. Requires `pip install dokis[semantic]`.
  Guards: raises `ValueError` when `len(chunks) > _MAX_CHUNKS` or
  `len(claims) > _MAX_CLAIMS` (overridable via env vars). Empty claims or empty
  chunks return `[]` immediately.

- **`ComplianceScorer`** (`core/scorer.py`) - computes `compliance_rate` as
  `supported_claims / total_claims` and determines `passed` against
  `Config.min_citation_rate`. Returns `(1.0, True)` with a `WARNING` when the
  claim list is empty.

- **`Config`** (`config.py`) - Pydantic v2 `BaseModel` with alternate
  constructors `Config.from_yaml()` (accepts TOML files) and `Config.from_dict()`,
  both routing through `model_validate`. Fields: `allowed_domains`,
  `min_citation_rate`, `claim_threshold`, `extractor`, `matcher`, `model`,
  `fail_on_violation`, `domain`, `llm_fn` (excluded from serialisation).
  Range validators enforce `0.0 ≤ min_citation_rate, claim_threshold ≤ 1.0`.
  `allowed_domains` strips leading `www.` on ingestion via a `model_validator`.

- **`ProvenanceResult`** (`models.py`) - Pydantic v2 model returned by every
  audit. Stores `response`, `claims`, `compliance_rate`, `passed`,
  `blocked_sources`, `domain` and `min_citation_rate`. Exposes two computed
  properties:
  - `violations` - claims where `supported=False`.
  - `provenance_map` - `{claim_text: source_url}` for all supported claims.

- **`Claim`** (`models.py`) - represents one extracted sentence with `text`,
  `supported`, `confidence` (always populated, even when unsupported),
  `source_chunk` and `source_url`.

- **`Chunk`** (`models.py`) - input unit with `content`, `source_url` and a
  free-form `metadata` dict.

- **`ComplianceViolation`** - exception raised when `fail_on_violation=True` and
  the audit fails. Always carries the full `ProvenanceResult` so callers can
  inspect what failed.

- **`DomainViolation`** - exception exposed for users who want to handle blocked
  domains as errors in their own pipeline code. Never raised automatically by
  Dokis internals.

- **LangChain adapter** (`adapters/langchain.py`) - `ProvenanceRetriever` wraps
  any `BaseRetriever`, overrides `_get_relevant_documents` and passes results
  through `DomainEnforcer` before returning. Converts `Document` ↔ `Chunk`
  transparently. Configurable via `url_metadata_key` (default `"source"`).
  Import-guarded: missing `langchain-core` raises a clear `ImportError` with
  install instructions. Requires `pip install dokis[langchain]`.

- **LlamaIndex adapter** (`adapters/llamaindex.py`) - `ProvenanceQueryEngine`
  wraps any `BaseQueryEngine`, post-processes the response through `audit()`
  and attaches the `ProvenanceResult` to `response.metadata["provenance"]`.
  Import-guarded identically to the LangChain adapter. Requires
  `pip install dokis[llamaindex]`.

- **Module-level API** (`dokis/__init__.py`) - three convenience functions for
  users who do not need a long-lived middleware instance:
  - `dokis.configure(config)` - sets a module-level default `Config`.
  - `dokis.filter(chunks, config=None)` - returns `list[Chunk]`.
  - `dokis.audit(query, chunks, response, config=None)` - returns
    `ProvenanceResult`.
  Backed by a thread-safe `ProvenanceMiddleware` cache keyed on
  `(id(config), config.model_dump_json())` so identical configs reuse the same
  model-loaded instance across calls.

- **86 tests** across seven modules (`test_enforcer.py` ×8, `test_extractor.py`
  ×9, `test_matcher.py` ×16, `test_scorer.py` ×6, `test_middleware.py` ×14,
  `test_adapters.py` ×8, `test_init.py` ×7, `test_config.py` ×2)
  plus shared fixtures in `conftest.py`.

### Changed

- **Dependencies slimmed to ~42 MB core install.** `pydantic`, `numpy`, and
  `bm25s` are the only mandatory runtime dependencies. `sentence-transformers`
  moved to the `[semantic]` optional extra. `nltk` moved to the `[nltk]` optional
  extra. `scikit-learn` removed entirely (cosine similarity is now implemented
  with pure NumPy). `pyyaml` removed; TOML is now the config file format, parsed
  with stdlib `tomllib` (Python 3.11+) or the `tomli` backport (Python 3.10,
  included automatically in `[dev]`).

### Fixed

- **BM25 raw-score floor** - claims where the best raw BM25 score is below
  `_MIN_BM25_RAW_SCORE` (default 0.5) are now marked unsupported with
  `confidence=0.0` instead of receiving an artificially normalised positive
  score from stopword-only overlap.

- **Cache stale-ID bug** - the module-level middleware cache is now keyed on
  `(id(config), config.model_dump_json())`. Keying on `id()` alone was unsafe
  because Python reuses memory addresses after garbage collection, which could
  cause a new `Config` object to hit a stale cache entry from a deleted one.

- **`extract()` accepts `None`** - `ClaimExtractor.extract()` now accepts
  `str | None`; passing `None` returns `[]` instead of raising `TypeError`.

- **`asyncio.get_event_loop()` deprecation** - `aaudit()` now uses
  `asyncio.get_running_loop()` and `loop.run_in_executor()`, eliminating the
  deprecation warning raised in Python 3.10+ by the former `asyncio.run()`
  pattern inside an already-running event loop.

- **BM25 index cached per chunk set** - the BM25 corpus index is rebuilt only
  when the chunk set changes, not on every `match()` call.

- **Regex splitter hardened** - the sentence-boundary regex no longer splits on
  decimal points (e.g. `0.72`) or common abbreviations (e.g. `Dr.`, `Fig.`).

- **`claim_threshold` warning for BM25** - `ClaimMatcher` now emits a `WARNING`
  at construction when `matcher="bm25"` and `claim_threshold > 0.5`, because
  BM25 scores are relative per-query (the best chunk always scores 1.0) and a
  threshold above 0.5 will produce false violations in practice.

- **`model` field warning for BM25** - `ClaimMatcher` emits a `WARNING` when
  `matcher="bm25"` and a non-default `model` is set, since the `model` field
  is only used by the semantic path.

- **Thread-safe `_default_config`** - all reads and writes to the module-level
  `_default_config` variable in `dokis/__init__.py` are now protected by a
  `threading.Lock`, preventing a race condition when `configure()` is called
  concurrently with `audit()` or `filter()`.

- **`monkeypatch` migration in tests** - test fixtures that previously patched
  internals with `unittest.mock` have been migrated to `pytest`'s `monkeypatch`
  fixture for consistency with the rest of the test suite.

[0.1.1]: https://github.com/Vbj1808/dokis/releases/tag/v0.1.1
[0.1.0]: https://github.com/Vbj1808/dokis/releases/tag/v0.1.0
