"""Benchmark claim selection against microsoft/claimify-dataset.

This is development infrastructure only. It downloads the public CSV at
benchmark runtime by default and does not add Hugging Face dependencies to
Dokis runtime.

Usage from the repo root:

    python benchmarks/run_claim_extraction.py
    python benchmarks/run_claim_extraction.py --limit 500
    python benchmarks/run_claim_extraction.py --input-csv /path/to/data.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from urllib.error import URLError
from urllib.request import urlopen

from dokis.config import Config
from dokis.core.extractor import ClaimExtractor

DATASET_ID = "microsoft/claimify-dataset"
DEFAULT_DATASET_URL = (
    "https://huggingface.co/datasets/microsoft/claimify-dataset/resolve/main/data.csv"
)
REQUIRED_FIELDS = frozenset({"sentence", "contains_factual_claim"})
DEFAULT_EXTRACTORS = ("regex", "claimify")
DEFAULT_OUTPUT_DIR = Path("benchmarks/results/claim_extraction")
SUMMARY_JSON_FILENAME = "claimify_selection_summary.json"
SUMMARY_MD_FILENAME = "claimify_selection_summary.md"
CAVEAT = (
    "This benchmark measures Selection-stage factual-claim detection only. "
    "It does not measure full Claimify reproduction, element-level coverage, "
    "citation faithfulness, or answer correctness."
)


@dataclass(frozen=True)
class BenchmarkRow:
    sentence: str
    contains_factual_claim: bool


@dataclass(frozen=True)
class BenchmarkResult:
    extractor: str
    examples: int
    positives: int
    predicted_positive: int
    true_positive: int
    false_positive: int
    false_negative: int
    true_negative: int
    accuracy: float
    precision: float
    recall: float
    f1: float


@dataclass(frozen=True)
class ErrorSamples:
    extractor: str
    false_positive_samples: list[str]
    false_negative_samples: list[str]


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark Dokis claim extractors on the public "
            "microsoft/claimify-dataset factual-claim selection labels."
        )
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=None,
        help="Local Claimify dataset CSV. Defaults to downloading from Hugging Face.",
    )
    parser.add_argument(
        "--dataset-url",
        default=DEFAULT_DATASET_URL,
        help="CSV URL to download when --input-csv is not provided.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Evaluate at most this many rows, useful for quick smoke runs.",
    )
    parser.add_argument(
        "--extractors",
        nargs="+",
        choices=("regex", "nltk", "claimify"),
        default=None,
        help=(
            "Extractor strategies to evaluate. Defaults to regex and claimify, "
            "plus nltk when its tokenizer data is already installed locally."
        ),
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON instead of a text table.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=25,
        help=(
            "Maximum false positive and false negative examples to include "
            "per extractor in written summaries."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=(
            "Directory for benchmark summary files. Defaults to "
            "benchmarks/results/claim_extraction/."
        ),
    )
    parser.add_argument(
        "--no-write",
        action="store_true",
        help="Disable benchmark summary file output.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    try:
        extractors = resolve_extractors(args.extractors)
        rows = load_rows(
            input_csv=args.input_csv,
            dataset_url=args.dataset_url,
            limit=args.limit,
        )
        results = [evaluate_extractor(extractor, rows) for extractor in extractors]
        error_samples = [
            collect_error_samples(extractor, rows, sample_size=args.sample_size)
            for extractor in extractors
        ]
        if not args.no_write:
            summary = build_summary(
                input_csv=args.input_csv,
                dataset_url=args.dataset_url,
                limit=args.limit,
                extractors=extractors,
                rows=rows,
                results=results,
                error_samples=error_samples,
            )
            write_summary_files(args.output_dir, summary)
    except (OSError, ValueError, URLError) as exc:
        print(f"Benchmark failed: {exc}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps([asdict(result) for result in results], indent=2))
    else:
        print_report(results, rows)
    return 0


def resolve_extractors(requested: Sequence[str] | None) -> list[str]:
    if requested is not None:
        return list(dict.fromkeys(requested))

    extractors = list(DEFAULT_EXTRACTORS)
    if nltk_tokenizer_available():
        extractors.insert(1, "nltk")
    return extractors


def nltk_tokenizer_available() -> bool:
    try:
        import nltk
    except ImportError:
        return False

    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        return False
    return True


def load_rows(
    *,
    input_csv: Path | None,
    dataset_url: str,
    limit: int | None,
) -> list[BenchmarkRow]:
    if limit is not None and limit < 1:
        raise ValueError("--limit must be a positive integer.")

    if input_csv is not None:
        rows = read_dataset_csv(input_csv)
    else:
        rows = download_and_read_dataset_csv(dataset_url)

    if limit is not None:
        return rows[:limit]
    return rows


def download_and_read_dataset_csv(dataset_url: str) -> list[BenchmarkRow]:
    with urlopen(dataset_url, timeout=60) as response:
        csv_bytes = response.read()

    with NamedTemporaryFile("wb", suffix=".csv", delete=False) as tmp:
        tmp.write(csv_bytes)
        tmp_path = Path(tmp.name)

    try:
        return read_dataset_csv(tmp_path)
    finally:
        tmp_path.unlink(missing_ok=True)


def read_dataset_csv(path: Path) -> list[BenchmarkRow]:
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        validate_schema(reader.fieldnames)
        rows = [
            BenchmarkRow(
                sentence=row["sentence"],
                contains_factual_claim=parse_bool(row["contains_factual_claim"]),
            )
            for row in reader
        ]

    if not rows:
        raise ValueError(f"{path} did not contain any benchmark rows.")
    return rows


def validate_schema(fieldnames: Sequence[str] | None) -> None:
    if fieldnames is None:
        raise ValueError("Dataset CSV is missing a header row.")

    missing = sorted(REQUIRED_FIELDS.difference(fieldnames))
    if missing:
        available = ", ".join(fieldnames)
        required = ", ".join(sorted(REQUIRED_FIELDS))
        raise ValueError(
            "Dataset schema mismatch for "
            f"{DATASET_ID}: missing required field(s): {', '.join(missing)}. "
            f"Required fields: {required}. Available fields: {available}."
        )


def parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise ValueError(
        "contains_factual_claim must contain boolean values 'true' or 'false'; "
        f"got {value!r}."
    )


def evaluate_extractor(
    extractor_name: str,
    rows: Sequence[BenchmarkRow],
) -> BenchmarkResult:
    extractor = ClaimExtractor(Config(extractor=extractor_name))
    predicted = [predict_contains_claim(extractor, row.sentence) for row in rows]
    gold = [row.contains_factual_claim for row in rows]

    true_positive = sum(p and g for p, g in zip(predicted, gold, strict=True))
    false_positive = sum(p and not g for p, g in zip(predicted, gold, strict=True))
    false_negative = sum(not p and g for p, g in zip(predicted, gold, strict=True))
    true_negative = sum(not p and not g for p, g in zip(predicted, gold, strict=True))

    examples = len(rows)
    positives = sum(gold)
    predicted_positive = sum(predicted)
    accuracy = safe_divide(true_positive + true_negative, examples)
    precision = safe_divide(true_positive, true_positive + false_positive)
    recall = safe_divide(true_positive, true_positive + false_negative)
    f1 = safe_divide(2 * precision * recall, precision + recall)

    return BenchmarkResult(
        extractor=extractor_name,
        examples=examples,
        positives=positives,
        predicted_positive=predicted_positive,
        true_positive=true_positive,
        false_positive=false_positive,
        false_negative=false_negative,
        true_negative=true_negative,
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
    )


def collect_error_samples(
    extractor_name: str,
    rows: Sequence[BenchmarkRow],
    *,
    sample_size: int,
) -> ErrorSamples:
    if sample_size < 0:
        raise ValueError("--sample-size must be zero or a positive integer.")

    extractor = ClaimExtractor(Config(extractor=extractor_name))
    false_positive_samples: list[str] = []
    false_negative_samples: list[str] = []

    for row in rows:
        predicted = predict_contains_claim(extractor, row.sentence)
        if (
            predicted
            and not row.contains_factual_claim
            and len(false_positive_samples) < sample_size
        ):
            false_positive_samples.append(row.sentence)
        elif (
            not predicted
            and row.contains_factual_claim
            and len(false_negative_samples) < sample_size
        ):
            false_negative_samples.append(row.sentence)

    return ErrorSamples(
        extractor=extractor_name,
        false_positive_samples=false_positive_samples,
        false_negative_samples=false_negative_samples,
    )


def predict_contains_claim(extractor: ClaimExtractor, sentence: str) -> bool:
    return bool(extractor.extract(sentence))


def safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def print_report(
    results: Iterable[BenchmarkResult],
    rows: Sequence[BenchmarkRow],
) -> None:
    print(f"Dataset: {DATASET_ID}")
    print(f"Rows: {len(rows)}")
    print()
    print(
        f"{'extractor':<10} {'accuracy':>8} {'precision':>9} {'recall':>8} "
        f"{'f1':>8} {'pred+':>7} {'gold+':>7} {'tp':>6} {'fp':>6} {'fn':>6}"
    )
    print("-" * 88)
    for result in results:
        print(
            f"{result.extractor:<10} "
            f"{result.accuracy:>8.3f} "
            f"{result.precision:>9.3f} "
            f"{result.recall:>8.3f} "
            f"{result.f1:>8.3f} "
            f"{result.predicted_positive:>7} "
            f"{result.positives:>7} "
            f"{result.true_positive:>6} "
            f"{result.false_positive:>6} "
            f"{result.false_negative:>6}"
        )


def build_summary(
    *,
    input_csv: Path | None,
    dataset_url: str,
    limit: int | None,
    extractors: Sequence[str],
    rows: Sequence[BenchmarkRow],
    results: Sequence[BenchmarkResult],
    error_samples: Sequence[ErrorSamples],
) -> dict[str, object]:
    sample_by_extractor = {samples.extractor: samples for samples in error_samples}
    metrics_by_extractor = {
        result.extractor: asdict(result) for result in results
    }

    false_positive_samples = {
        extractor: sample_by_extractor[extractor].false_positive_samples
        for extractor in extractors
    }
    false_negative_samples = {
        extractor: sample_by_extractor[extractor].false_negative_samples
        for extractor in extractors
    }

    return {
        "dataset": DATASET_ID if input_csv is None else str(input_csv),
        "source": dataset_url if input_csv is None else str(input_csv),
        "row_count": len(rows),
        "limit": limit,
        "extractors_evaluated": list(extractors),
        "metrics_per_extractor": metrics_by_extractor,
        "false_positive_samples": false_positive_samples,
        "false_negative_samples": false_negative_samples,
    }


def write_summary_files(output_dir: Path, summary: dict[str, object]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / SUMMARY_JSON_FILENAME).write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / SUMMARY_MD_FILENAME).write_text(
        render_markdown_summary(summary),
        encoding="utf-8",
    )


def render_markdown_summary(summary: dict[str, object]) -> str:
    metrics = summary["metrics_per_extractor"]
    if not isinstance(metrics, dict):
        raise ValueError("Summary metrics_per_extractor must be a dictionary.")

    lines = [
        "# Claimify Selection Benchmark Summary",
        "",
        f"Dataset/source: {summary['source']}",
        f"Rows: {summary['row_count']}",
        "",
        (
            "| extractor | accuracy | precision | recall | f1 | pred+ | gold+ | "
            "tp | fp | fn |"
        ),
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for extractor in summary["extractors_evaluated"]:
        result = metrics[extractor]
        lines.append(
            "| "
            f"{result['extractor']} | "
            f"{result['accuracy']:.3f} | "
            f"{result['precision']:.3f} | "
            f"{result['recall']:.3f} | "
            f"{result['f1']:.3f} | "
            f"{result['predicted_positive']} | "
            f"{result['positives']} | "
            f"{result['true_positive']} | "
            f"{result['false_positive']} | "
            f"{result['false_negative']} |"
        )

    lines.extend(["", CAVEAT, ""])
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
