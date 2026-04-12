"""CLI and terminal formatter coverage for Dokis trust reports."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from dokis.config import Config
from dokis.formatter import render_audit_report
from dokis.middleware import ProvenanceMiddleware
from dokis.models import Chunk


def test_render_audit_report_surfaces_double_x_sections(
    sample_chunks: list[Chunk],
    strict_config: Config,
) -> None:
    middleware = ProvenanceMiddleware(strict_config)
    response = (
        "Aspirin reduces fever by inhibiting COX enzymes in the body. "
        "Ibuprofen is a nonsteroidal anti-inflammatory drug used for pain relief. "
        "Aspirin has no meaningful drug interactions with other medicines."
    )
    result = middleware.audit("What does aspirin do?", sample_chunks, response)

    report = render_audit_report(
        query="What does aspirin do?",
        raw_chunks=sample_chunks,
        result=result,
        config=strict_config,
        color=False,
        width=88,
    )

    assert "Dokis Trust Report" in report
    assert "Audit Summary" in report
    assert "Enforcement mode: guardrail" in report
    assert "Enforcement verdict: guardrail_failed" in report
    assert "Policy Issues" in report
    assert "[ISSUE] blocked_sources" in report
    assert "[ISSUE] unsupported_claims" in report
    assert "Source Boundary" in report
    assert "[ALLOWED] https://pubmed.ncbi.nlm.nih.gov/12345" in report
    assert "[BLOCKED] https://discountpharma.biz/meds" in report
    assert "reason: Domain not on the allowlist" in report
    assert "Claim Verdicts" in report
    assert "[SUPPORTED]" in report
    assert "[UNSUPPORTED]" in report
    assert "Final Compliance" in report


def test_cli_audit_command_renders_report_for_sample_file() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    sample_path = repo_root / "sample_audit.json"

    completed = subprocess.run(
        [sys.executable, "-m", "dokis", "audit", str(sample_path), "--no-color"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 1
    assert "Dokis Trust Report" in completed.stdout
    assert "Enforcement mode: guardrail" in completed.stdout
    assert "Enforcement verdict: guardrail_failed" in completed.stdout
    assert "[BLOCKED] https://discountpharma.biz/meds" in completed.stdout
    assert (
        "[UNSUPPORTED] Aspirin has no meaningful drug interactions "
        "with other medicines."
    ) in completed.stdout
    assert "Compliance: 66.7% (required: 85.0%)" in completed.stdout
