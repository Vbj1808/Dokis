# AGENTS.md - Dokis

> Dokis is a Python middleware library for RAG provenance enforcement.
> It does exactly two things: domain allowlist filtering before chunks
> reach the LLM and per-claim provenance auditing after generation.
> Do not add features outside this scope.

---

## Setup

```bash
git clone https://github.com/Vbj1808/dokis
cd dokis
pip install -e ".[dev]"
```

---

## Before every commit

All three must pass with zero errors or warnings. Never commit if any fail.

```bash
ruff check dokis/
mypy dokis/ --strict
pytest tests/ -v
```

One-liner:
```bash
ruff check dokis/ && mypy dokis/ --strict && pytest tests/ -v
```

---

## Running tests

```bash
# Full suite (86 tests)
pytest tests/ -v

# Single module with coverage
pytest tests/test_matcher.py -v
pytest tests/test_enforcer.py -v --cov=dokis/core/enforcer --cov-report=term-missing

# Formatting
ruff format dokis/
```

---

## Architecture

```
dokis/
├── __init__.py         ← entire public API, nothing else exported here
├── config.py           ← Config Pydantic model, TOML loading
├── models.py           ← Chunk, Claim, ProvenanceResult
├── exceptions.py       ← ComplianceViolation, DomainViolation
├── middleware.py       ← ProvenanceMiddleware, pipeline orchestrator
└── core/
    ├── enforcer.py     ← DomainEnforcer, URL allowlist filtering
    ├── extractor.py    ← ClaimExtractor, sentence splitting
    ├── matcher.py      ← ClaimMatcher, BM25 and semantic matching
    └── scorer.py       ← ComplianceScorer, compliance rate
```

One class per file in `core/`. No exceptions.

---

## Non-negotiables

These rules apply regardless of what seems like a good reason to break them:

- **No LLM call in the default hot path.** `Config()` with no arguments must
  run end-to-end with zero API keys, zero network requests, zero model downloads.
- **No new core dependencies** without justification for why it cannot be
  implemented with `pydantic`, `numpy`, `bm25s`, or the Python standard library.
- **Public API signatures are frozen.** Once exported from `__init__.py`, a
  method signature cannot change without a major version bump.
- **`mypy dokis/ --strict` must pass** with zero errors. No bare `Any` without
  a comment explaining why it is unavoidable.
- **Zero silent failures.** Log `WARNING` for recoverable edge cases. Raise for
  errors. Never return a plausible-looking result that hides a bug.

---

## Public API

Everything a user needs is importable from `dokis` directly. Nothing from
submodules should be imported by users.

```python
import dokis

dokis.configure(config)
dokis.filter(chunks, config=None)               # returns list[Chunk]
dokis.audit(query, chunks, response, config=None)  # returns ProvenanceResult

dokis.Config
dokis.Chunk
dokis.Claim
dokis.ProvenanceResult
dokis.ProvenanceMiddleware
dokis.ComplianceViolation
dokis.DomainViolation
```

Do not add to this surface without a documented use case. Run `/audit-api`
after any edit to `__init__.py`.

---

## Matchers

- Default is `matcher="bm25"` via `bm25s`. Zero cold start, zero model download.
- Semantic path uses `SentenceTransformer` and requires
  `pip install dokis[semantic]`. Cold start ~1,666ms.
- Do not make semantic the default.
- Do not add a new matcher without updating `config.py`, `matcher.py`,
  and adding tests for both the happy path and edge cases.

## claim_threshold

Means different things per matcher:
- `bm25`: normalised per-query score, relative not absolute. Recommended `0.3–0.5`.
  Values above `0.5` emit a WARNING.
- `semantic`: cosine similarity `[0.0, 1.0]`. Recommended `0.65–0.85`.

Always document this distinction when touching threshold logic.

---

## Extractor strategies

- `"regex"` (default) - fast, zero deps, zero network. Filters sentences under
  8 words.
- `"nltk"` - higher accuracy, requires `pip install dokis[nltk]`. Downloads
  `punkt_tab` once on first construction.
- `"llm"` - delegates to `config.llm_fn`. Never bundle any LLM client inside
  Dokis.

---

## Adding an adapter

- Never reimplement core logic inside an adapter. Compose `ProvenanceMiddleware`
  internally.
- Guard the optional import with `try/except ImportError` and raise a clear
  message with install instructions.
- Add the optional dependency to `pyproject.toml`.
- Write tests that inject minimal mock modules so adapter tests run without
  the framework installed.

---

## Config files

Config uses TOML format via stdlib `tomllib` (Python 3.11+) or `tomli` backport
(Python 3.10). The method is named `from_yaml()` for backwards compatibility
but expects `.toml` files. Passing a `.yaml` or `.yml` path raises `ValueError`.

---

## Dependencies

```
Core (pip install dokis):
  pydantic>=2.0
  numpy>=1.26
  bm25s>=0.2

Optional:
  semantic    → sentence-transformers>=2.7
  nltk        → nltk>=3.8
  langchain   → langchain-core>=0.2
  llamaindex  → llama-index-core>=0.10
```

---

## Key files to read before editing

| Editing | Read first |
|---|---|
| `core/matcher.py` | CLAUDE.md matcher section + bm25s docs |
| `core/enforcer.py` | CLAUDE.md enforcer section |
| `core/extractor.py` | CLAUDE.md extractor section |
| `__init__.py` | CLAUDE.md public API contract |
| `config.py` | CLAUDE.md config section + pydantic v2 docs |

---

## What Dokis is NOT

If a proposed change pushes Dokis toward any of the following, decline it:

- A RAG framework (it wraps pipelines, it does not replace them)
- An offline evaluation tool (RAGAS does that)
- An LLM guardrail for toxicity or jailbreaks
- A platform with UI, server, database, or hosted service
- Opinionated about vector stores, embedders, or LLM providers