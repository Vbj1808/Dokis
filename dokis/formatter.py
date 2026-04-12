"""Terminal formatter for Dokis trust reports."""

from __future__ import annotations

import os
import shutil
import sys
import textwrap
from urllib.parse import urlparse

from dokis.config import Config
from dokis.models import BlockedSource, Chunk, ProvenanceResult

_ANSI_RESET = "\033[0m"
_ANSI_BOLD = "\033[1m"
_ANSI_DIM = "\033[2m"
_ANSI_CYAN = "\033[36m"
_ANSI_GREEN = "\033[32m"
_ANSI_RED = "\033[31m"
_ANSI_YELLOW = "\033[33m"

_BLOCK_REASON_LABELS = {
    "domain_not_allowlisted": "Domain not on the allowlist",
    "malformed_source_url": "Source URL is malformed",
    "missing_source_url": "Source URL is missing",
}

_POLICY_ISSUE_LABELS = {
    "blocked_sources": "blocked_sources",
    "unsupported_claims": "unsupported_claims",
}


def render_audit_report(
    *,
    query: str,
    raw_chunks: list[Chunk],
    result: ProvenanceResult,
    config: Config,
    color: bool | None = None,
    width: int | None = None,
) -> str:
    """Render a screenshot-friendly audit report for terminal output."""
    use_color = _supports_color() if color is None else color
    report_width = _resolve_width(width)

    allowed_urls, blocked_entries = _partition_sources(raw_chunks, result)
    sections = [
        _render_title(use_color),
        _render_summary(
            query,
            result,
            config,
            allowed_urls,
            blocked_entries,
            report_width,
            use_color,
        ),
        _render_policy_issues(result, report_width, use_color),
        _render_sources(allowed_urls, blocked_entries, report_width, use_color),
        _render_claims(result, report_width, use_color),
        _render_compliance(result, report_width, use_color),
    ]
    return "\n\n".join(section for section in sections if section).rstrip() + "\n"


def _render_title(use_color: bool) -> str:
    title = _style("Dokis Trust Report", _ANSI_BOLD + _ANSI_CYAN, use_color)
    rule = _style("=" * 18, _ANSI_DIM, use_color)
    return f"{title}\n{rule}"


def _render_summary(
    query: str,
    result: ProvenanceResult,
    config: Config,
    allowed_urls: list[str],
    blocked_entries: list[BlockedSource],
    width: int,
    use_color: bool,
) -> str:
    lines = [
        _section_heading("Audit Summary", use_color),
        _wrapped_kv("Query", query, width),
        f"Matcher: {_safe_text(config.matcher)}",
        f"Claim Threshold: {config.claim_threshold:.2f}",
        f"Enforcement mode: {_safe_text(result.enforcement_mode)}",
        "Enforcement verdict: "
        f"{_format_enforcement_verdict(result.enforcement_verdict, use_color)}",
        f"Sources: {len(allowed_urls)} allowed, {len(blocked_entries)} blocked",
    ]
    return "\n".join(lines)


def _render_policy_issues(
    result: ProvenanceResult,
    width: int,
    use_color: bool,
) -> str:
    lines = [_section_heading("Policy Issues", use_color)]
    if not result.policy_issues:
        lines.append(_style("[PASS]", _ANSI_GREEN, use_color) + " none")
        return "\n".join(lines)

    for issue in result.policy_issues:
        label = _style("[ISSUE]", _ANSI_YELLOW, use_color)
        lines.extend(_wrap_lines(f"{label} {_POLICY_ISSUE_LABELS[issue]}", width))
    return "\n".join(lines)


def _render_sources(
    allowed_urls: list[str],
    blocked_entries: list[BlockedSource],
    width: int,
    use_color: bool,
) -> str:
    lines = [_section_heading("Source Boundary", use_color)]

    if not allowed_urls and not blocked_entries:
        lines.append(
            _style("[PASS]", _ANSI_GREEN, use_color)
            + " no source URLs were provided"
        )
        return "\n".join(lines)

    if allowed_urls:
        for url in allowed_urls:
            lines.extend(
                _wrap_lines(
                    f"{_style('[ALLOWED]', _ANSI_GREEN, use_color)} {_safe_text(url)}",
                    width,
                )
            )
    else:
        lines.append(_style("[ALLOWED]", _ANSI_GREEN, use_color) + " none")

    if blocked_entries:
        for entry in blocked_entries:
            blocked_line = (
                f"{_style('[BLOCKED]', _ANSI_RED, use_color)} "
                f"{_safe_text(entry.url)}"
            )
            lines.extend(
                _wrap_lines(
                    blocked_line,
                    width,
                )
            )
            domain = entry.domain or infer_domain(entry.url) or "-"
            lines.append(
                f"  domain: {_safe_text(domain)}"
            )
            lines.append(f"  reason: {_BLOCK_REASON_LABELS[entry.reason]}")
    else:
        lines.append(_style("[BLOCKED]", _ANSI_RED, use_color) + " none")

    return "\n".join(lines)


def _render_claims(
    result: ProvenanceResult,
    width: int,
    use_color: bool,
) -> str:
    lines = [_section_heading("Claim Verdicts", use_color)]

    if not result.claim_verdicts:
        lines.append(
            _style("[PASS]", _ANSI_GREEN, use_color)
            + " no claims were extracted from the response"
        )
        return "\n".join(lines)

    for verdict in result.claim_verdicts:
        is_supported = verdict.verdict == "supported"
        badge = _style(
            "[SUPPORTED]" if is_supported else "[UNSUPPORTED]",
            _ANSI_GREEN if is_supported else _ANSI_RED,
            use_color,
        )
        lines.extend(_wrap_lines(f"{badge} {verdict.claim_text}", width))
        lines.append(f"  confidence: {verdict.confidence:.2f}")
        lines.append(f"  source: {_safe_text(verdict.supporting_url or '-')}")
        if verdict.note:
            lines.extend(_wrap_lines(f"  note: {verdict.note}", width))
    return "\n".join(lines)


def _render_compliance(
    result: ProvenanceResult,
    width: int,
    use_color: bool,
) -> str:
    status = "PASSED" if result.passed else "FAILED"
    status_style = _ANSI_GREEN if result.passed else _ANSI_RED
    lines = [
        _section_heading("Final Compliance", use_color),
        _wrap_lines(
            f"Compliance: {result.compliance_rate:.1%} "
            f"(required: {result.min_citation_rate:.1%})",
            width,
        )[0],
        f"Status: {_style(status, status_style + _ANSI_BOLD, use_color)}",
    ]
    return "\n".join(lines)


def _partition_sources(
    raw_chunks: list[Chunk],
    result: ProvenanceResult,
) -> tuple[list[str], list[BlockedSource]]:
    blocked_urls = {entry.url for entry in result.blocked_source_details}
    allowed_urls: list[str] = []
    seen: set[str] = set()

    for chunk in raw_chunks:
        url = chunk.source_url
        if url in seen or url in blocked_urls:
            continue
        seen.add(url)
        allowed_urls.append(url)

    return allowed_urls, result.blocked_source_details


def _format_enforcement_verdict(verdict: str, use_color: bool) -> str:
    if verdict == "passed":
        return _style(verdict, _ANSI_GREEN + _ANSI_BOLD, use_color)
    return _style(verdict, _ANSI_RED + _ANSI_BOLD, use_color)


def _resolve_width(width: int | None) -> int:
    if width is not None:
        return max(60, min(width, 100))
    terminal_width = shutil.get_terminal_size(fallback=(88, 24)).columns
    return max(60, min(terminal_width, 100))


def _section_heading(text: str, use_color: bool) -> str:
    return _style(text, _ANSI_BOLD, use_color)


def _wrapped_kv(label: str, value: str, width: int) -> str:
    wrapped = _wrap_lines(f"{label}: {value}", width)
    return "\n".join(wrapped)


def _wrap_lines(text: str, width: int) -> list[str]:
    return textwrap.wrap(
        text,
        width=width,
        replace_whitespace=False,
        drop_whitespace=False,
        subsequent_indent="  ",
    ) or [text]


def _safe_text(value: str) -> str:
    return value.strip() or "-"


def _style(text: str, codes: str, use_color: bool) -> str:
    if not use_color:
        return text
    return f"{codes}{text}{_ANSI_RESET}"


def _supports_color() -> bool:
    return (
        sys.stdout.isatty()
        and os.getenv("NO_COLOR") is None
        and os.getenv("TERM") not in {None, "dumb"}
    )


def infer_domain(url: str) -> str | None:
    """Return a display-friendly hostname for a URL string."""
    host = urlparse(url).hostname
    return host[4:] if host and host.startswith("www.") else host
