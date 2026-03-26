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
import os
import sys
from pathlib import Path
from typing import Any, cast

from dokis.config import Config
from dokis.middleware import ProvenanceMiddleware
from dokis.models import Chunk

# ---------------------------------------------------------------------------
# Colour helpers — disabled when stdout is not a TTY or NO_COLOR is set.
# ---------------------------------------------------------------------------

_COLOURS_ENABLED: bool = (
    sys.stdout.isatty() and os.environ.get("NO_COLOR") is None
)

_RESET  = "\033[0m"   if _COLOURS_ENABLED else ""
_BOLD   = "\033[1m"   if _COLOURS_ENABLED else ""
_DIM    = "\033[2m"   if _COLOURS_ENABLED else ""
_GREEN  = "\033[32m"  if _COLOURS_ENABLED else ""
_RED    = "\033[31m"  if _COLOURS_ENABLED else ""
_YELLOW = "\033[33m"  if _COLOURS_ENABLED else ""
_CYAN   = "\033[36m"  if _COLOURS_ENABLED else ""
_WHITE  = "\033[97m"  if _COLOURS_ENABLED else ""


def _green(text: str) -> str:
    return f"{_GREEN}{text}{_RESET}"


def _red(text: str) -> str:
    return f"{_RED}{text}{_RESET}"


def _yellow(text: str) -> str:
    return f"{_YELLOW}{text}{_RESET}"


def _cyan(text: str) -> str:
    return f"{_CYAN}{text}{_RESET}"


def _dim(text: str) -> str:
    return f"{_DIM}{text}{_RESET}"


def _bold(text: str) -> str:
    return f"{_BOLD}{text}{_RESET}"


# ---------------------------------------------------------------------------
# Input loading
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Chunk parsing
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------


def _format_report(
    query: str,
    result: Any,
    config: Config,
) -> str:
    """Build a human-readable provenance report with colour formatting.

    Args:
        query: The original user query.
        result: A ProvenanceResult instance.
        config: The Config used for this audit.

    Returns:
        Formatted report string for terminal output.
    """
    lines: list[str] = []
    lines.append("")
    lines.append(_bold(_white("Dokis Provenance Audit")))
    lines.append(_dim("=" * 54))
    lines.append("")
    lines.append(f"  {_dim('Query:')}     {query}")
    lines.append(f"  {_dim('Matcher:')}   {_cyan(config.matcher)}")
    lines.append(f"  {_dim('Threshold:')} {config.claim_threshold}")
    lines.append("")

    # Claims table
    if result.claims:
        lines.append(f"  {_bold('Claims')}")
        lines.append("  " + _dim("-" * 50))
        for claim in result.claims:
            if claim.supported:
                icon = _green("\u2713")
                text_color = _white
            else:
                icon = _red("\u2717")
                text_color = _dim

            conf = f"{claim.confidence:.2f}"
            url = claim.source_url or "\u2014"

            text = claim.text
            if len(text) > 52:
                text = text[:49] + "..."

            lines.append(f"  {icon} {text_color(text)}")
            lines.append(
                f"    {_dim('confidence:')} {conf}  "
                f"{_dim('source:')} {_cyan(url) if claim.source_url else _dim(url)}"
            )
        lines.append("")
    else:
        lines.append(_dim("  No claims extracted."))
        lines.append("")

    # Summary
    rate = result.compliance_rate
    threshold = result.min_citation_rate
    passed = result.passed

    status = _green("PASSED") if passed else _red("FAILED")
    rate_str = _green(f"{rate:.1%}") if passed else _red(f"{rate:.1%}")

    lines.append(
        f"  {_dim('Compliance:')} {rate_str} "
        f"{_dim(f'(threshold: {threshold:.1%})')}"
    )
    lines.append(f"  {_dim('Status:')}     {_bold(status)}")
    lines.append("")

    # Violations
    if result.violations:
        lines.append(f"  {_bold(_red('Violations'))}")
        lines.append("  " + _dim("-" * 50))
        for v in result.violations:
            text = v.text
            if len(text) > 60:
                text = text[:57] + "..."
            lines.append(
                f"  {_red(chr(8226))} {_dim(repr(text))} "
                f"{_dim(f'(confidence: {v.confidence:.2f})')}"
            )
        lines.append("")

    # Blocked sources
    if result.blocked_sources:
        lines.append(f"  {_bold(_yellow('Blocked sources'))}")
        lines.append("  " + _dim("-" * 50))
        for src in result.blocked_sources:
            lines.append(f"  {_yellow(chr(8226))} {_dim(src)}")
        lines.append("")

    return "\n".join(lines)


def _white(text: str) -> str:
    return f"{_WHITE}{text}{_RESET}"


# ---------------------------------------------------------------------------
# Error helper
# ---------------------------------------------------------------------------


def _exit_error(message: str) -> None:
    """Print an error message to stderr and exit with code 1.

    Args:
        message: The error message to display.
    """
    print(f"{_red('dokis: error:')} {message}", file=sys.stderr)
    sys.exit(1)


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------


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
    if args.config:
        config_path = Path(args.config)
        if not config_path.exists():
            _exit_error(f"Config file not found: {args.config}")
        config = Config.from_yaml(str(config_path))
    else:
        config = Config()

    data = _load_input(args.input)
    query: str = data["query"]
    chunks = _parse_chunks(data["chunks"])
    response: str = data["response"]

    mw = ProvenanceMiddleware(config)
    result = mw.audit(query, chunks, response)

    report = _format_report(query, result, config)
    print(report)

    if not result.passed:
        sys.exit(1)