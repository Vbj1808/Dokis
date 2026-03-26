"""Tests for dokis.cli — the command-line interface."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from dokis.cli import _build_parser, _format_report, _load_input, _parse_chunks, main
from dokis.config import Config
from dokis.models import Chunk, Claim, ProvenanceResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def valid_input_data() -> dict[str, Any]:
    """Minimal valid input for the audit command."""
    return {
        "query": "What are the side effects of aspirin?",
        "chunks": [
            {
                "content": "Aspirin inhibits COX-1 and COX-2 enzymes.",
                "source_url": "https://pubmed.ncbi.nlm.nih.gov/12345",
            },
            {
                "content": "Common side effects include stomach irritation.",
                "source_url": "https://who.int/fact-sheets/aspirin",
            },
        ],
        "response": "Aspirin inhibits COX enzymes. It can cause stomach irritation.",
    }


@pytest.fixture()
def valid_input_file(
    tmp_path: Path, valid_input_data: dict[str, Any]
) -> Path:
    """Write valid input data to a temp JSON file."""
    path = tmp_path / "input.json"
    path.write_text(json.dumps(valid_input_data), encoding="utf-8")
    return path


@pytest.fixture()
def sample_config_file(tmp_path: Path) -> Path:
    """Write a minimal provenance.toml to a temp file."""
    path = tmp_path / "provenance.toml"
    path.write_text(
        'allowed_domains = ["pubmed.ncbi.nlm.nih.gov", "who.int"]\n'
        "min_citation_rate = 0.50\n"
        "claim_threshold = 0.30\n"
        'matcher = "bm25"\n',
        encoding="utf-8",
    )
    return path


@pytest.fixture()
def sample_result() -> ProvenanceResult:
    """A ProvenanceResult with one supported and one unsupported claim."""
    chunk = Chunk(
        content="Aspirin inhibits COX-1 and COX-2 enzymes.",
        source_url="https://pubmed.ncbi.nlm.nih.gov/12345",
    )
    claims = [
        Claim(
            text="Aspirin inhibits COX enzymes.",
            supported=True,
            confidence=0.87,
            source_chunk=chunk,
            source_url="https://pubmed.ncbi.nlm.nih.gov/12345",
        ),
        Claim(
            text="No known drug interactions exist.",
            supported=False,
            confidence=0.12,
            source_chunk=None,
            source_url=None,
        ),
    ]
    return ProvenanceResult(
        response="Aspirin inhibits COX enzymes. No known drug interactions exist.",
        claims=claims,
        compliance_rate=0.5,
        passed=False,
        blocked_sources=[],
        domain=None,
        min_citation_rate=0.80,
    )


# ---------------------------------------------------------------------------
# _load_input
# ---------------------------------------------------------------------------


class TestLoadInput:
    """Tests for JSON loading and validation."""

    def test_loads_valid_file(self, valid_input_file: Path) -> None:
        data = _load_input(str(valid_input_file))
        assert data["query"] == "What are the side effects of aspirin?"
        assert len(data["chunks"]) == 2
        assert isinstance(data["response"], str)

    def test_missing_file_exits(self) -> None:
        with pytest.raises(SystemExit) as exc_info:
            _load_input("/nonexistent/path.json")
        assert exc_info.value.code == 1

    def test_invalid_json_exits(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.json"
        bad.write_text("{not valid json", encoding="utf-8")
        with pytest.raises(SystemExit) as exc_info:
            _load_input(str(bad))
        assert exc_info.value.code == 1

    def test_missing_keys_exits(self, tmp_path: Path) -> None:
        incomplete = tmp_path / "incomplete.json"
        incomplete.write_text(
            json.dumps({"query": "test"}), encoding="utf-8"
        )
        with pytest.raises(SystemExit) as exc_info:
            _load_input(str(incomplete))
        assert exc_info.value.code == 1

    def test_non_object_exits(self, tmp_path: Path) -> None:
        arr = tmp_path / "array.json"
        arr.write_text("[1, 2, 3]", encoding="utf-8")
        with pytest.raises(SystemExit) as exc_info:
            _load_input(str(arr))
        assert exc_info.value.code == 1

    def test_chunks_not_array_exits(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad_chunks.json"
        bad.write_text(
            json.dumps({
                "query": "test",
                "chunks": "not a list",
                "response": "test",
            }),
            encoding="utf-8",
        )
        with pytest.raises(SystemExit) as exc_info:
            _load_input(str(bad))
        assert exc_info.value.code == 1

    def test_chunk_missing_fields_exits(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad_chunk_fields.json"
        bad.write_text(
            json.dumps({
                "query": "test",
                "chunks": [{"content": "only content"}],
                "response": "test",
            }),
            encoding="utf-8",
        )
        with pytest.raises(SystemExit) as exc_info:
            _load_input(str(bad))
        assert exc_info.value.code == 1

    def test_stdin_read(
        self, monkeypatch: pytest.MonkeyPatch, valid_input_data: dict[str, Any]
    ) -> None:
        import io

        monkeypatch.setattr(
            "sys.stdin", io.StringIO(json.dumps(valid_input_data))
        )
        data = _load_input("-")
        assert data["query"] == valid_input_data["query"]


# ---------------------------------------------------------------------------
# _parse_chunks
# ---------------------------------------------------------------------------


class TestParseChunks:
    """Tests for converting raw dicts to Chunk models."""

    def test_parses_valid_chunks(self) -> None:
        raw = [
            {"content": "Hello world.", "source_url": "https://example.com"},
            {
                "content": "Goodbye.",
                "source_url": "https://test.com",
                "metadata": {"page": 3},
            },
        ]
        chunks = _parse_chunks(raw)
        assert len(chunks) == 2
        assert chunks[0].content == "Hello world."
        assert chunks[0].source_url == "https://example.com"
        assert chunks[1].metadata == {"page": 3}

    def test_default_metadata_is_empty_dict(self) -> None:
        raw = [{"content": "No meta.", "source_url": "https://x.com"}]
        chunks = _parse_chunks(raw)
        assert chunks[0].metadata == {}


# ---------------------------------------------------------------------------
# _format_report
# ---------------------------------------------------------------------------


class TestFormatReport:
    """Tests for the human-readable report output."""

    def test_report_contains_query(self, sample_result: ProvenanceResult) -> None:
        report = _format_report("What about aspirin?", sample_result, Config())
        assert "What about aspirin?" in report

    def test_report_shows_compliance_rate(
        self, sample_result: ProvenanceResult
    ) -> None:
        report = _format_report("q", sample_result, Config())
        assert "50.0%" in report

    def test_report_shows_status_failed(
        self, sample_result: ProvenanceResult
    ) -> None:
        report = _format_report("q", sample_result, Config())
        assert "FAILED" in report

    def test_report_shows_violations(
        self, sample_result: ProvenanceResult
    ) -> None:
        report = _format_report("q", sample_result, Config())
        assert "No known drug interactions" in report

    def test_report_shows_passed_when_passing(self) -> None:
        result = ProvenanceResult(
            response="All good.",
            claims=[
                Claim(
                    text="Supported claim.",
                    supported=True,
                    confidence=0.9,
                    source_chunk=None,
                    source_url="https://example.com",
                ),
            ],
            compliance_rate=1.0,
            passed=True,
            blocked_sources=[],
            domain=None,
            min_citation_rate=0.80,
        )
        report = _format_report("q", result, Config())
        assert "PASSED" in report

    def test_report_shows_blocked_sources(self) -> None:
        result = ProvenanceResult(
            response="test",
            claims=[],
            compliance_rate=1.0,
            passed=True,
            blocked_sources=["https://untrusted.io/page"],
            domain=None,
            min_citation_rate=0.80,
        )
        report = _format_report("q", result, Config())
        assert "untrusted.io" in report

    def test_no_claims_message(self) -> None:
        result = ProvenanceResult(
            response="",
            claims=[],
            compliance_rate=1.0,
            passed=True,
            blocked_sources=[],
            domain=None,
            min_citation_rate=0.80,
        )
        report = _format_report("q", result, Config())
        assert "No claims extracted" in report

    def test_long_claim_text_truncated(self) -> None:
        long_text = "A" * 100
        result = ProvenanceResult(
            response=long_text,
            claims=[
                Claim(
                    text=long_text,
                    supported=True,
                    confidence=0.9,
                    source_chunk=None,
                    source_url="https://example.com",
                ),
            ],
            compliance_rate=1.0,
            passed=True,
            blocked_sources=[],
            domain=None,
            min_citation_rate=0.80,
        )
        report = _format_report("q", result, Config())
        assert "..." in report


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


class TestParser:
    """Tests for argument parsing."""

    def test_audit_command_parsed(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["audit", "input.json"])
        assert args.command == "audit"
        assert args.input == "input.json"
        assert args.config is None

    def test_audit_with_config(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(
            ["audit", "input.json", "--config", "provenance.toml"]
        )
        assert args.config == "provenance.toml"

    def test_stdin_dash(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["audit", "-"])
        assert args.input == "-"


# ---------------------------------------------------------------------------
# main() integration
# ---------------------------------------------------------------------------


class TestMainIntegration:
    """Integration tests for the full CLI flow."""

    def test_audit_runs_and_prints(
        self, valid_input_file: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        # May exit 0 or 1 depending on compliance; we just check it runs
        try:
            main(["audit", str(valid_input_file)])
        except SystemExit as e:
            # Exit code 1 is acceptable (compliance failure)
            assert e.code in (0, 1)
        output = capsys.readouterr().out
        assert "Dokis Provenance Audit" in output
        assert "Compliance:" in output

    def test_audit_with_config(
        self,
        valid_input_file: Path,
        sample_config_file: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        try:
            main([
                "audit",
                str(valid_input_file),
                "--config",
                str(sample_config_file),
            ])
        except SystemExit as e:
            assert e.code in (0, 1)
        output = capsys.readouterr().out
        assert "Dokis Provenance Audit" in output

    def test_no_command_shows_help(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        with pytest.raises(SystemExit) as exc_info:
            main([])
        assert exc_info.value.code == 0

    def test_missing_config_exits(self, valid_input_file: Path) -> None:
        with pytest.raises(SystemExit) as exc_info:
            main([
                "audit",
                str(valid_input_file),
                "--config",
                "/nonexistent/config.toml",
            ])
        assert exc_info.value.code == 1

    def test_exit_code_1_on_failure(self, tmp_path: Path) -> None:
        """A response with no grounded claims should exit 1."""
        data = {
            "query": "test",
            "chunks": [
                {
                    "content": "The sky is blue because of Rayleigh scattering.",
                    "source_url": "https://example.com/sky",
                },
            ],
            "response": (
                "Bananas are the largest fruit in the world and "
                "they grow on oak trees in Antarctica."
            ),
        }
        path = tmp_path / "ungrounded.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        with pytest.raises(SystemExit) as exc_info:
            main(["audit", str(path)])
        assert exc_info.value.code == 1