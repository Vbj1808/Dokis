"""Tests for development benchmark harnesses."""

import json
from pathlib import Path

import pytest

from benchmarks.run_claim_extraction import (
    SUMMARY_JSON_FILENAME,
    SUMMARY_MD_FILENAME,
    BenchmarkResult,
    BenchmarkRow,
    ErrorSamples,
    build_summary,
    evaluate_extractor,
    load_rows,
    main,
    read_dataset_csv,
    render_markdown_summary,
    resolve_extractors,
    write_summary_files,
)


def write_claimify_csv(path: Path) -> None:
    path.write_text(
        "answer_id,question,sentence_id,sentence,contains_factual_claim\n"
        'a1,Q,0,"Aspirin reduces fever.",true\n'
        'a1,Q,1,"This is amazing.",false\n',
        encoding="utf-8",
    )


def sample_summary(tmp_path: Path) -> dict[str, object]:
    rows = [
        BenchmarkRow("Aspirin reduces fever.", True),
        BenchmarkRow("This is amazing.", False),
    ]
    result = BenchmarkResult(
        extractor="claimify",
        examples=2,
        positives=1,
        predicted_positive=1,
        true_positive=1,
        false_positive=0,
        false_negative=0,
        true_negative=1,
        accuracy=1.0,
        precision=1.0,
        recall=1.0,
        f1=1.0,
    )
    csv_path = tmp_path / "claimify.csv"
    return build_summary(
        input_csv=csv_path,
        dataset_url="unused",
        limit=2,
        extractors=["claimify"],
        rows=rows,
        results=[result],
        error_samples=[
            ErrorSamples(
                extractor="claimify",
                false_positive_samples=["This is amazing."],
                false_negative_samples=[],
            )
        ],
    )


def test_claim_extraction_benchmark_loads_local_csv(tmp_path: Path) -> None:
    csv_path = tmp_path / "claimify.csv"
    write_claimify_csv(csv_path)

    rows = load_rows(input_csv=csv_path, dataset_url="unused", limit=None)

    assert len(rows) == 2
    assert rows[0].sentence == "Aspirin reduces fever."
    assert rows[0].contains_factual_claim is True
    assert rows[1].contains_factual_claim is False


def test_claim_extraction_benchmark_validates_required_fields(
    tmp_path: Path,
) -> None:
    csv_path = tmp_path / "bad.csv"
    csv_path.write_text("sentence\nAspirin reduces fever.\n", encoding="utf-8")

    with pytest.raises(ValueError, match="missing required field"):
        read_dataset_csv(csv_path)


def test_claim_extraction_benchmark_evaluates_claimify_rows() -> None:
    rows = [
        BenchmarkRow("Aspirin reduces fever.", True),
        BenchmarkRow("This is amazing.", False),
    ]

    result = evaluate_extractor("claimify", rows)

    assert result.extractor == "claimify"
    assert result.examples == 2
    assert result.true_positive == 1
    assert result.true_negative == 1
    assert result.f1 == pytest.approx(1.0)


def test_claim_extraction_benchmark_defaults_to_regex_and_claimify() -> None:
    extractors = resolve_extractors(None)
    assert extractors[0] == "regex"
    assert "claimify" in extractors


def test_claim_extraction_benchmark_writes_default_output_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    csv_path = tmp_path / "claimify.csv"
    write_claimify_csv(csv_path)

    exit_code = main(["--input-csv", str(csv_path), "--extractors", "claimify"])

    output_dir = tmp_path / "benchmarks/results/claim_extraction"
    assert exit_code == 0
    assert (output_dir / SUMMARY_JSON_FILENAME).is_file()
    assert (output_dir / SUMMARY_MD_FILENAME).is_file()


def test_claim_extraction_benchmark_respects_custom_output_dir(
    tmp_path: Path,
) -> None:
    csv_path = tmp_path / "claimify.csv"
    output_dir = tmp_path / "results"
    write_claimify_csv(csv_path)

    exit_code = main(
        [
            "--input-csv",
            str(csv_path),
            "--extractors",
            "claimify",
            "--output-dir",
            str(output_dir),
        ]
    )

    assert exit_code == 0
    assert (output_dir / SUMMARY_JSON_FILENAME).is_file()
    assert (output_dir / SUMMARY_MD_FILENAME).is_file()


def test_claim_extraction_json_summary_writer_works_without_network(
    tmp_path: Path,
) -> None:
    summary = sample_summary(tmp_path)
    output_dir = tmp_path / "summary"

    write_summary_files(output_dir, summary)

    data = json.loads((output_dir / SUMMARY_JSON_FILENAME).read_text())
    assert data["dataset"] == str(tmp_path / "claimify.csv")
    assert data["row_count"] == 2
    assert data["limit"] == 2
    assert data["extractors_evaluated"] == ["claimify"]
    assert data["metrics_per_extractor"]["claimify"]["f1"] == 1.0
    assert data["false_positive_samples"]["claimify"] == ["This is amazing."]
    assert data["false_negative_samples"]["claimify"] == []


def test_claim_extraction_markdown_summary_writer_works_without_network(
    tmp_path: Path,
) -> None:
    markdown = render_markdown_summary(sample_summary(tmp_path))

    assert markdown.startswith("# Claimify Selection Benchmark Summary")
    assert f"Dataset/source: {tmp_path / 'claimify.csv'}" in markdown
    assert "| claimify | 1.000 | 1.000 | 1.000 | 1.000 |" in markdown
    assert (
        "This benchmark measures Selection-stage factual-claim detection only."
        in markdown
    )


def test_claim_extraction_benchmark_no_write_disables_file_creation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    csv_path = tmp_path / "claimify.csv"
    write_claimify_csv(csv_path)

    exit_code = main(
        [
            "--input-csv",
            str(csv_path),
            "--extractors",
            "claimify",
            "--no-write",
        ]
    )

    assert exit_code == 0
    assert not (tmp_path / "benchmarks/results/claim_extraction").exists()


def test_claim_extraction_benchmark_json_stdout_remains_compatible(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    csv_path = tmp_path / "claimify.csv"
    write_claimify_csv(csv_path)

    exit_code = main(
        [
            "--input-csv",
            str(csv_path),
            "--extractors",
            "claimify",
            "--json",
            "--no-write",
        ]
    )

    stdout = capsys.readouterr().out
    data = json.loads(stdout)
    assert exit_code == 0
    assert isinstance(data, list)
    assert data[0]["extractor"] == "claimify"
    assert data[0]["examples"] == 2


def test_claim_extraction_benchmark_sample_size_limits_error_samples(
    tmp_path: Path,
) -> None:
    csv_path = tmp_path / "claimify.csv"
    output_dir = tmp_path / "results"
    csv_path.write_text(
        "answer_id,question,sentence_id,sentence,contains_factual_claim\n"
        'a1,Q,0,"This is amazing.",true\n'
        'a1,Q,1,"Here is the answer.",true\n'
        'a1,Q,2,"This is useful.",true\n',
        encoding="utf-8",
    )

    exit_code = main(
        [
            "--input-csv",
            str(csv_path),
            "--extractors",
            "claimify",
            "--output-dir",
            str(output_dir),
            "--sample-size",
            "2",
        ]
    )

    data = json.loads((output_dir / SUMMARY_JSON_FILENAME).read_text())
    assert exit_code == 0
    assert len(data["false_negative_samples"]["claimify"]) == 2
