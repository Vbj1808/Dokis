# Contributing to Dokis

Dokis is a small, focused library - please read this before opening a PR.

## Philosophy

Dokis does exactly two things: pre-retrieval domain enforcement and
post-generation claim provenance auditing. Contributions that expand
this scope will be declined. See "What Dokis is NOT" in CLAUDE.md.

## Setup

```bash
git clone https://github.com/Vbj1808/dokis
cd dokis
pip install -e ".[dev]"
```

## Before every PR

All three must pass with zero errors or warnings:

```bash
ruff check dokis/
mypy dokis/ --strict
pytest tests/ -v
```

## Adding a dependency

Every new core dependency requires justification for why it cannot be
implemented with existing deps or the Python standard library. Optional
dependencies are preferred wherever possible.

## Adding an adapter

See the "Adding an adapter" section in CLAUDE.md for the required pattern.
Never reimplement core logic inside an adapter - compose
`ProvenanceMiddleware` internally. Guard the optional import with
`try/except ImportError` and raise a clear message with install instructions.

## Public API

Once exported from `__init__.py`, a signature is frozen. Breaking changes
require a major version bump and a migration guide.