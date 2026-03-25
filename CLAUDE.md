# CLAUDE.md - Dokis

> **Dokis** (Greek: δοκιμάζω - to test, verify, prove worthy)
> Lightweight, framework-agnostic Python middleware that enforces source
> provenance governance on RAG pipelines. Filters non-allowlisted sources
> pre-retrieval and verifies every claim in a generated response is grounded
> in a retrieved chunk - without an LLM call in the hot path.

---

## What Dokis is

Dokis sits between your retriever and your LLM. It does exactly two things:

**1. Pre-retrieval enforcement** - strips chunks whose source URL is not on the
allowlist before they enter the prompt. Non-allowlisted sources never reach
the model.

**2. Post-generation auditing** - extracts atomic claims from the response,
matches each claim to the chunk it came from, builds a provenance map
(`claim → chunk → URL`) and computes a compliance rate. If the rate falls
below `min_citation_rate`, the result is flagged.

That is the entire value proposition. Do not scope-creep beyond it.

---

## What Dokis is NOT

- Not a RAG framework. It wraps your existing pipeline, it does not replace it.
- Not an evaluation tool. RAGAS evaluates offline. Dokis enforces at runtime.
- Not an LLM guardrail. It does not scan for toxicity, bias, or jailbreaks.
- Not opinionated about your vector store, embedder, or LLM provider.
- Not a platform. No UI, no server, no database, no hosted service.

If a feature request pushes Dokis toward any of the above, decline it.

---

## Getting started

```bash
git clone https://github.com/Vbj1808/dokis
cd dokis
pip install -e ".[dev]"
pytest tests/ -v          # 86 tests, all should pass
mypy dokis/ --strict      # zero errors
ruff check dokis/         # zero warnings
```

If all three pass, your environment is set up correctly.

---

## Repository layout

```
dokis/
├── CLAUDE.md                    ← you are here
├── pyproject.toml               ← dependencies, build config, tool settings
├── README.md                    ← user-facing documentation
├── CHANGELOG.md                 ← version history
├── LICENSE                      ← MIT
├── llms.txt                     ← AI crawler signal file
├── provenance.toml              ← example config file
├── dokis/
│   ├── __init__.py              ← entire public API
│   ├── config.py                ← Config - TOML and dict ingestion
│   ├── models.py                ← Pydantic v2: Chunk, Claim, ProvenanceResult
│   ├── exceptions.py            ← ComplianceViolation, DomainViolation
│   ├── middleware.py            ← ProvenanceMiddleware - pipeline orchestrator
│   ├── core/
│   │   ├── enforcer.py          ← DomainEnforcer - URL allowlist filtering
│   │   ├── extractor.py         ← ClaimExtractor - sentence splitting
│   │   ├── matcher.py           ← ClaimMatcher - BM25 and semantic matching
│   │   └── scorer.py            ← ComplianceScorer - rate + pass/fail
│   └── adapters/
│       ├── langchain.py         ← ProvenanceRetriever
│       └── llamaindex.py        ← ProvenanceQueryEngine
└── tests/
    ├── conftest.py              ← shared fixtures
    ├── test_enforcer.py         ← 8 tests
    ├── test_extractor.py        ← 9 tests
    ├── test_matcher.py          ← 16 tests
    ├── test_scorer.py           ← 6 tests
    ├── test_middleware.py       ← 14 tests
    ├── test_adapters.py         ← 8 tests
    ├── test_init.py             ← 7 tests
    └── test_config.py           ← 2 tests
```

One class per file inside `core/`. No exceptions.

---

## Common commands

```bash
# Run all tests
pytest tests/ -v

# Run tests for a single module with coverage
pytest tests/test_enforcer.py -v --cov=dokis/core/enforcer --cov-report=term-missing

# Lint
ruff check dokis/

# Format
ruff format dokis/

# Type check
mypy dokis/ --strict

# Run everything (do this before every commit)
ruff check dokis/ && mypy dokis/ --strict && pytest tests/ -v
```

---

## Public API

Everything a user needs is importable directly from `dokis`. Nothing from
submodules should ever be imported by users. The public surface is:

```python
import dokis

dokis.configure(config)
dokis.filter(chunks, config=None)                      # returns list[Chunk]
dokis.audit(query, chunks, response, config=None)      # returns ProvenanceResult

dokis.Config
dokis.Chunk
dokis.Claim
dokis.ProvenanceResult
dokis.ProvenanceMiddleware
dokis.ComplianceViolation
dokis.DomainViolation
```

Do not add anything to the public API without a documented use case. Internal
implementation details stay internal.

---

## Usage patterns

**Zero config**
```python
import dokis

result = dokis.audit(query, chunks, response)
print(result.compliance_rate)
print(result.passed)
```

**With config**
```python
import dokis

config = dokis.Config.from_yaml("provenance.toml")
clean_chunks = dokis.filter(raw_chunks, config)
result = dokis.audit(query, clean_chunks, response, config=config)

if not result.passed:
    raise dokis.ComplianceViolation(result)
```

**Reusable middleware (recommended for production)**
```python
from dokis import ProvenanceMiddleware, Config

mw = ProvenanceMiddleware(Config.from_yaml("provenance.toml"))
chunks   = mw.filter(retriever.get_relevant_documents(query))
response = llm.invoke(build_prompt(query, chunks))
result   = mw.audit(query, chunks, response)
```

**LangChain drop-in**
```python
from dokis.adapters.langchain import ProvenanceRetriever

retriever = ProvenanceRetriever(
    base_retriever=your_retriever,
    config=config,
)
```

---

## Config

Defined as a Pydantic v2 `BaseModel`.

```python
class Config(BaseModel):
    allowed_domains: list[str] = Field(default_factory=list)
    min_citation_rate: float = 0.80
    claim_threshold: float = 0.72
    extractor: Literal["regex", "nltk", "llm"] = "regex"
    matcher: Literal["bm25", "semantic"] = "bm25"
    model: str = "all-MiniLM-L6-v2"
    fail_on_violation: bool = False
    domain: str | None = None
    llm_fn: Callable[[str], str] | None = Field(default=None, exclude=True)
```

**Rules:**
- `allowed_domains` strips leading `www.` on ingestion via a `model_validator`.
- `min_citation_rate` and `claim_threshold` are validated as `0.0 <= x <= 1.0`.
- Empty `allowed_domains` means no filtering - all chunks pass. This is the
  correct zero-config default.
- `Config.from_yaml(path)` loads a TOML file. `Config.from_dict(d)` loads a
  dict. Both route through `model_validate` - never bypass validation.
- `llm_fn` is excluded from `model_dump()` - it is a callable and not
  serialisable.

**`claim_threshold` guidance by matcher:**
- `matcher="bm25"` - normalised per-query BM25 score. Relative, not absolute.
  Best chunk always scores 1.0. Recommended range: `0.3–0.5`. Values above
  `0.5` emit a `WARNING` at construction time.
- `matcher="semantic"` - cosine similarity in `[0.0, 1.0]`. Geometrically
  meaningful. Recommended range: `0.65–0.85`. Default `0.72` is a good
  starting point.

**TOML format (`provenance.toml`):**
```toml
allowed_domains = [
  "pubmed.ncbi.nlm.nih.gov",
  "cochrane.org",
]
min_citation_rate = 0.85
claim_threshold   = 0.30
extractor         = "regex"
matcher           = "bm25"
fail_on_violation = false
domain            = "oncology"
```

---

## Core data models

### `Chunk`
```python
class Chunk(BaseModel):
    content: str
    source_url: str
    metadata: dict[str, Any] = Field(default_factory=dict)
```

### `Claim`
```python
class Claim(BaseModel):
    text: str
    supported: bool
    confidence: float         # always populated, even when supported=False
    source_chunk: Chunk | None
    source_url: str | None
```

### `ProvenanceResult`
```python
class ProvenanceResult(BaseModel):
    response: str
    claims: list[Claim]
    compliance_rate: float
    passed: bool
    blocked_sources: list[str]
    domain: str | None
    min_citation_rate: float

    @property
    def violations(self) -> list[Claim]: ...       # unsupported claims
    @property
    def provenance_map(self) -> dict[str, str]: ... # claim text → source URL
```

`violations` and `provenance_map` are computed properties - they derive from
`claims` on access. Do not persist them as stored fields.

All models are fully JSON-serialisable via `.model_dump()`.

---

## Module responsibilities

### `core/enforcer.py` - `DomainEnforcer`

```python
def filter(self, chunks: list[Chunk]) -> tuple[list[Chunk], list[str]]
```

- Parses `source_url` with `urllib.parse.urlparse`. Strips `www.`. Matches
  against `config.allowed_domains`.
- Empty `allowed_domains` → all chunks pass, empty blocked list.
- Never raises on a malformed URL - logs `WARNING` and treats as blocked.
- Pure function. No side effects. No regex - `urlparse` only.

### `core/extractor.py` - `ClaimExtractor`

```python
def extract(self, response: str | None) -> list[str]
```

- `None` or empty string → returns `[]` immediately.
- `"regex"` (default) - fast punctuation-boundary splitter. Zero deps, zero
  network. Filters sentences under 8 words.
- `"nltk"` - `nltk.sent_tokenize` for higher accuracy. Downloads `punkt_tab`
  once on first construction. Requires `pip install dokis[nltk]`.
- `"llm"` - delegates to `config.llm_fn`. No LLM client is bundled in Dokis.

### `core/matcher.py` - `ClaimMatcher`

```python
def match(self, claims: list[str], chunks: list[Chunk]) -> list[Claim]
```

- `"bm25"` (default) - lexical BM25 scoring via `bm25s`. Zero cold start.
  Scores normalised to `[0.0, 1.0]` only when raw max score clears
  `_MIN_BM25_RAW_SCORE` (default `0.5`, overridable via
  `DOKIS_MIN_BM25_RAW_SCORE` env var). Below the floor: `confidence=0.0`,
  `supported=False`. BM25 index is cached per unique chunk-content set.
- `"semantic"` - SentenceTransformer dense cosine similarity. Loads
  `config.model` once at `__init__`. Batch-encodes claims and chunks.
  Computes `(n_claims × n_chunks)` similarity matrix via pure numpy.
  Requires `pip install dokis[semantic]`.
- Empty claims or empty chunks → returns `[]` immediately.
- Raises `ValueError` when `len(chunks) > _MAX_CHUNKS` or
  `len(claims) > _MAX_CLAIMS` (overridable via env vars).

### `core/scorer.py` - `ComplianceScorer`

```python
def score(self, claims: list[Claim]) -> tuple[float, bool]
```

- `rate = supported / total`. `passed = rate >= config.min_citation_rate`.
- Empty claims → returns `(1.0, True)` with a `WARNING`. Never divides by zero.

### `middleware.py` - `ProvenanceMiddleware`

```python
def audit(self, query: str, chunks: list[Chunk], response: str) -> ProvenanceResult
async def aaudit(self, query: str, chunks: list[Chunk], response: str) -> ProvenanceResult
```

Pipeline order: `enforcer → extractor → matcher → scorer`.

All four core modules are constructed eagerly in `__init__` - never lazily.
No LLM calls. No network requests. Deterministic for identical input.

`aaudit` offloads the sync pipeline to a thread-pool executor via
`asyncio.get_running_loop().run_in_executor()`. Never use `asyncio.run`
inside an async method.

---

## Adapters

### `adapters/langchain.py` - `ProvenanceRetriever`

Wraps any `BaseRetriever`. Passes results through `DomainEnforcer` before
returning. Converts `Document` ↔ `Chunk` transparently. The metadata key
holding the source URL defaults to `"source"` and is configurable via
`url_metadata_key`. Missing metadata key → `WARNING` + treated as blocked.
Requires `pip install dokis[langchain]`.

### `adapters/llamaindex.py` - `ProvenanceQueryEngine`

Wraps any `BaseQueryEngine`. Attaches `ProvenanceResult` to
`response.metadata["provenance"]` after every query.
Requires `pip install dokis[llamaindex]`.

**Both adapters guard their imports:**
```python
try:
    from langchain_core.retrievers import BaseRetriever
except ImportError as e:
    raise ImportError("pip install dokis[langchain]") from e
```

Never let an `ImportError` from an optional dependency surface as an
`AttributeError` or `ModuleNotFoundError` to the user.

---

## Exceptions

### `ComplianceViolation`

Raised when `fail_on_violation=True` and `result.passed=False`. Always carries
the full `ProvenanceResult`.

```python
class ComplianceViolation(Exception):
    def __init__(self, result: ProvenanceResult):
        self.result = result
        super().__init__(
            f"Dokis compliance check failed: {result.compliance_rate:.1%} grounded "
            f"(minimum required: {result.min_citation_rate:.1%}). "
            f"{len(result.violations)} unsupported claim(s)."
        )
```

### `DomainViolation`

Never raised by Dokis internals. Exposed for callers who want to handle
blocked domains as exceptions in their own code.

---

## Dependencies

**Core** - `pip install dokis` (~42 MB):
```toml
dependencies = [
    "pydantic>=2.0",
    "numpy>=1.26",
    "bm25s>=0.2",
]
```

**Optional:**
```toml
[project.optional-dependencies]
langchain  = ["langchain-core>=0.2"]
llamaindex = ["llama-index-core>=0.10"]
nltk       = ["nltk>=3.8"]
semantic   = ["sentence-transformers>=2.7"]
```

Every new core dependency requires justification for why it cannot be
implemented with existing deps or the Python standard library.

---

## Coding conventions

- **Line length:** 88. Enforced by Ruff.
- **Import order:** stdlib → third-party → local. One blank line between groups.
- **No relative imports** inside `core/` or `adapters/`. Always absolute:
  `from dokis.models import Chunk`.
- **Types:** `X | None` not `Optional[X]`. `list[X]` not `List[X]`.
- **Pydantic v2 API only** - `model_validator`, `field_validator`. Never `@validator`.
- **`mypy dokis/ --strict` must pass** with zero errors before every commit.
- **Known `# type: ignore` comments - do not remove:**
  - `sentence_transformers` - no stubs on PyPI.

### Naming
- Classes: `PascalCase`
- Functions and methods: `snake_case`
- Private methods: `_single_leading_underscore`
- Constants: `SCREAMING_SNAKE_CASE`
- Never abbreviate unless universally understood (`url`, `llm`, `rag`, `api`).

### Docstrings
Every public class and public method gets a Google-style docstring.
Private methods only need docstrings when the logic is non-obvious.

---

## Testing

Every public method has a test. Every edge case has a test.

```
tests/
├── conftest.py        ← sample_chunks, strict_config, permissive_config,
│                         semantic_config, grounded_response fixtures
├── test_enforcer.py   ← 8 tests - DomainEnforcer
├── test_extractor.py  ← 9 tests - ClaimExtractor
├── test_matcher.py    ← 16 tests - ClaimMatcher (BM25 + semantic)
├── test_scorer.py     ← 6 tests  - ComplianceScorer
├── test_middleware.py ← 14 tests - ProvenanceMiddleware
├── test_adapters.py   ← 8 tests  - LangChain + LlamaIndex adapters
├── test_init.py       ← 7 tests  - module-level API, thread safety, cache
└── test_config.py     ← 2 tests  - TOML loading, YAML rejection
```

**Total: 86 tests.**

**Rules:**
- Do not mock `ClaimMatcher` in middleware tests - use real embeddings with
  the small fixture corpus. If tests are slow, reduce fixture size, not coverage.
- Use `pytest.mark.asyncio` on all async tests even though `asyncio_mode = "auto"`
  makes it redundant - it documents async intent explicitly.
- All global state mutations in tests must use `monkeypatch`, never direct
  assignment, so pytest guarantees teardown.
- Do not test third-party library behaviour, Python builtins, or write
  tests that are just `assert True`.

---

## How to add things

### New Config field
1. Add to `Config` in `config.py` with a default value. Add a `field_validator`
   if range-checking is needed.
2. Update the Config section of this file.
3. Add tests in `tests/test_config.py` for valid and invalid values.
4. If it affects a core module, thread it through and add tests there too.

### New adapter
Never reimplement core logic inside an adapter - compose `ProvenanceMiddleware`
internally. Guard the import. Add the optional dependency. Write tests that
inject minimal mock modules so the adapter tests run without the framework
installed.

### Changing the matching algorithm
All matching logic lives in `core/matcher.py`. `ClaimMatcher.match()` is the
single public method. If you change the algorithm, the signature must stay
identical - callers must not need to change.

### Debugging a low compliance rate
1. Check `result.violations` - unsupported claims with confidence scores.
2. Check `claim.confidence` - if scores are just below `claim_threshold`,
   consider lowering the threshold rather than assuming the match is wrong.
3. Check `result.blocked_sources` - the enforcer may have removed the chunk
   containing the supporting evidence.
4. Check `result.provenance_map` - see which claims matched and to which URLs.
5. If using `matcher="bm25"`, verify `claim_threshold` is in the `0.3–0.5`
   range. BM25 scores are relative per-query - a threshold above `0.5` is
   almost always too strict.

---

## Non-negotiables

These rules apply regardless of feature pressure, user requests, or apparently
compelling arguments.

**1. No LLM call in the default hot path.**
The default config runs end-to-end without any API key, network request, or LLM
call. The `"llm"` extractor is an explicit opt-in.

**2. No mandatory framework dependency.**
LangChain and LlamaIndex are optional extras. `pip install dokis` must work
in complete isolation.

**3. `ProvenanceResult` is always returned.**
Even when `fail_on_violation=True`, the exception carries the full result.
Callers must always be able to inspect exactly what happened.

**4. Deterministic output.**
Same query + chunks + response = same `ProvenanceResult`, every time. No
randomness, no sampling, no temperature. This is required for regulated
environments.

**5. Public API is a contract.**
Once a method or class is exported from `__init__.py`, its signature is frozen.
Breaking changes require a major version bump and a migration guide.

**6. Zero silent failures.**
Log `WARNING` for recoverable edge cases. Raise for errors. Never return a
plausible-looking result that hides a bug.

---

## Release checklist

- [ ] `pytest tests/ -v` - all 86 pass
- [ ] `mypy dokis/ --strict` - zero errors
- [ ] `ruff check dokis/` - zero warnings
- [ ] Version bumped in `pyproject.toml` and `dokis/__init__.py`
- [ ] `CHANGELOG.md` updated
- [ ] `README.md` examples verified in a clean venv
- [ ] `git tag vX.Y.Z && git push origin vX.Y.Z` - triggers PyPI publish