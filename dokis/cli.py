"""Command-line interface for Dokis provenance auditing.

Usage:
    dokis audit input.json
    dokis audit input.json --config provenance.toml
    cat input.json | dokis audit -

The input JSON must contain:
    - query (str): the user query
    - chunks (list): each with "content" and "source_url"
    - response (str): the LLM-generated response
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, cast

from dokis.config import Config
from dokis.middleware import ProvenanceMiddleware
from dokis.models import Chunk


def _load_input(path: str) -> dict[str, Any]:
    """Load and validate the input JSON from a file path or stdin.

    Args:
        path: Path to a JSON file or "-" to read from stdin.

    Returns:
        Parsed JSON as a dictionary.

    Raises:
        SystemExit: If the file is missing, unreadable or invalid JSON.
    """
    try:
        if path == "-":
            raw = sys.stdin.read()
        else:
            raw = Path(path).read_text(encoding="utf-8")
    except FileNotFoundError:
        _exit_error(f"File not found: {path}")
    except OSError as e:
        _exit_error(f"Cannot read file: {e}")

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        _exit_error(f"Invalid JSON: {e}")

    if not isinstance(data, dict):
        _exit_error("Input JSON must be an object, not an array or scalar.")

    missing = [
        key for key in ("query", "chunks", "response") if key not in data
    ]
    if missing:
        _exit_error(f"Missing required key(s): {', '.join(missing)}")

    if not isinstance(data["chunks"], list):
        _exit_error('"chunks" must be a JSON array.')

    for i, chunk in enumerate(data["chunks"]):
        if not isinstance(chunk, dict):
            _exit_error(f"chunks[{i}] must be an object.")
        if "content" not in chunk or "source_url" not in chunk:
            _exit_error(
                f'chunks[{i}] must have "content" and "source_url" keys.'
            )

    return cast(dict[str, Any], data)


def _parse_chunks(raw_chunks: list[dict[str, Any]]) -> list[Chunk]:
    """Convert raw JSON chunk dicts to Chunk models.

    Args:
        raw_chunks: List of dicts with at least "content" and "source_url".

    Returns:
        List of validated Chunk instances.
    """
    return [
        Chunk(
            content=c["content"],
            source_url=c["source_url"],
            metadata=c.get("metadata", {}),
        )
        for c in raw_chunks
    ]


def _format_report(
    query: str,
    result: Any,
    config: Config,
) -> str:
    """Build a human-readable provenance report.

    Args:
        query: The original user query.
        result: A ProvenanceResult instance.
        config: The Config used for this audit.

    Returns:
        Formatted report string for terminal output.
    """
    lines: list[str] = []
    lines.append("")
    lines.append("Dokis Provenance Audit")
    lines.append("=" * 54)
    lines.append("")
    lines.append(f"  Query:     {query}")
    lines.append(f"  Matcher:   {config.matcher}")
    lines.append(f"  Threshold: {config.claim_threshold}")
    lines.append("")

    # Claims table
    if result.claims:
        lines.append("  Claims")
        lines.append("  " + "-" * 50)
        for claim in result.claims:
            icon = "\u2713" if claim.supported else "\u2717"
            conf = f"{claim.confidence:.2f}"
            url = claim.source_url or "\u2014"

            # Truncate long claim text for terminal readability
            text = claim.text
            if len(text) > 52:
                text = text[:49] + "..."

            lines.append(f"  {icon} {text}")
            lines.append(f"    confidence: {conf}  source: {url}")
        lines.append("")
    else:
        lines.append("  No claims extracted.")
        lines.append("")

    # Summary
    rate = result.compliance_rate
    threshold = result.min_citation_rate
    status = "PASSED" if result.passed else "FAILED"
    lines.append(f"  Compliance: {rate:.1%} (threshold: {threshold:.1%})")
    lines.append(f"  Status:     {status}")
    lines.append("")

    # Violations
    if result.violations:
        lines.append("  Violations")
        lines.append("  " + "-" * 50)
        for v in result.violations:
            text = v.text
            if len(text) > 60:
                text = text[:57] + "..."
            lines.append(f'  \u2022 "{text}" (confidence: {v.confidence:.2f})')
        lines.append("")

    # Blocked sources
    if result.blocked_sources:
        lines.append("  Blocked sources")
        lines.append("  " + "-" * 50)
        for src in result.blocked_sources:
            lines.append(f"  \u2022 {src}")
        lines.append("")

    return "\n".join(lines)


def _exit_error(message: str) -> None:
    """Print an error message to stderr and exit with code 1.

    Args:
        message: The error message to display.
    """
    print(f"dokis: error: {message}", file=sys.stderr)
    sys.exit(1)


def _build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser for the dokis CLI.

    Returns:
        Configured ArgumentParser instance.
    """
    parser = argparse.ArgumentParser(
        prog="dokis",
        description=(
            "Runtime provenance enforcement for RAG pipelines. "
            "Every claim. Every source. Zero LLM calls."
        ),
    )

    subparsers = parser.add_subparsers(dest="command")

    audit_parser = subparsers.add_parser(
        "audit",
        help="Audit a RAG response for source provenance.",
    )
    audit_parser.add_argument(
        "input",
        help='Path to input JSON file, or "-" to read from stdin.',
    )
    audit_parser.add_argument(
        "--config",
        default=None,
        help="Path to a provenance.toml config file.",
    )

    return parser


def main(argv: list[str] | None = None) -> None:
    """Entry point for the dokis CLI.

    Args:
        argv: Command-line arguments. Defaults to sys.argv[1:].
    """
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    if args.command == "audit":
        _run_audit(args)


def _run_audit(args: argparse.Namespace) -> None:
    """Execute the audit subcommand.

    Args:
        args: Parsed CLI arguments with 'input' and optional 'config'.
    """
    # Load config
    if args.config:
        config_path = Path(args.config)
        if not config_path.exists():
            _exit_error(f"Config file not found: {args.config}")
        config = Config.from_yaml(str(config_path))
    else:
        config = Config()

    # Load and parse input
    data = _load_input(args.input)
    query: str = data["query"]
    chunks = _parse_chunks(data["chunks"])
    response: str = data["response"]

    # Run audit
    mw = ProvenanceMiddleware(config)
    result = mw.audit(query, chunks, response)

    # Output
    report = _format_report(query, result, config)
    print(report)

    # Exit code reflects compliance
    if not result.passed:
        sys.exit(1)