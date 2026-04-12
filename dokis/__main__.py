"""Module entrypoint for ``python -m dokis``."""

from __future__ import annotations

from dokis.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
