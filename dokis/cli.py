"""Command-line interface for Dokis."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dokis.config import Config
from dokis.exceptions import ComplianceViolation
from dokis.formatter import render_audit_report
from dokis.middleware import ProvenanceMiddleware
from dokis.models import Chunk


def main(argv: list[str] | None = None) -> int:
    """Run the Dokis CLI."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command != "audit":
        parser.print_help(sys.stderr)
        return 2

    try:
        payload = _load_json(args.input)
        config = _resolve_config(args)

        chunks_raw = payload["chunks"]
        if not isinstance(chunks_raw, list):
            raise ValueError("'chunks' must be a list.")

        chunks = [Chunk.model_validate(item) for item in chunks_raw]
        query = str(payload["query"])
        response = str(payload["response"])
        middleware = ProvenanceMiddleware(config)
        try:
            result = middleware.audit(query, chunks, response)
        except ComplianceViolation as exc:
            result = exc.result

        report = render_audit_report(
            query=query,
            raw_chunks=chunks,
            result=result,
            config=config,
            color=not args.no_color,
        )
        stream = sys.stdout
        stream.write(report)
        return 0 if result.passed else 1
    except FileNotFoundError as exc:
        print(f"Dokis CLI error: {exc}", file=sys.stderr)
        return 2
    except (KeyError, TypeError, ValueError) as exc:
        print(f"Dokis CLI error: {exc}", file=sys.stderr)
        return 2


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="dokis")
    subparsers = parser.add_subparsers(dest="command")

    audit_parser = subparsers.add_parser(
        "audit",
        help="Audit a JSON payload with query, chunks, and response.",
    )
    audit_parser.add_argument(
        "input",
        type=Path,
        help="Path to the audit JSON file.",
    )
    audit_parser.add_argument(
        "--config",
        type=Path,
        help=(
            "Path to a Dokis TOML config file. Defaults to provenance.toml "
            "when present."
        ),
    )
    audit_parser.add_argument(
        "--no-color",
        action="store_true",
        help="Disable ANSI colors in terminal output.",
    )
    return parser


def _load_json(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    if not isinstance(payload, dict):
        raise ValueError("Audit input must be a JSON object.")
    for field in ("query", "chunks", "response"):
        if field not in payload:
            raise KeyError(f"Missing required field: {field}")
    return payload


def _resolve_config(args: argparse.Namespace) -> Config:
    if args.config is not None:
        return Config.from_yaml(args.config)

    for candidate in (
        Path.cwd() / "provenance.toml",
        args.input.resolve().parent / "provenance.toml",
    ):
        if candidate.exists():
            return Config.from_yaml(candidate)
    return Config()


if __name__ == "__main__":
    raise SystemExit(main())
