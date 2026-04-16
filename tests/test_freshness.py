"""Tests for metadata-backed source freshness evaluation."""

from __future__ import annotations

from dokis.config import Config
from dokis.core.freshness import FreshnessEvaluator
from dokis.models import Chunk


def test_freshness_evaluator_parses_year_metadata() -> None:
    evaluator = FreshnessEvaluator(Config(max_source_age_days=365))
    chunk = Chunk(
        content="Old policy text.",
        source_url="https://example.com/old",
        metadata={"year": 2018},
    )

    assessment = evaluator.assess(chunk)

    assert assessment.status == "stale"
    assert assessment.source_date is not None
    assert assessment.source_date.isoformat() == "2018-01-01"
    assert assessment.metadata_key == "year"


def test_freshness_evaluator_parses_iso_date_from_configured_key() -> None:
    evaluator = FreshnessEvaluator(
        Config(
            max_source_age_days=365,
            source_date_metadata_key="published_on",
        )
    )
    chunk = Chunk(
        content="Recent policy text.",
        source_url="https://example.com/recent",
        metadata={"published_on": "2026-03-01"},
    )

    assessment = evaluator.assess(chunk)

    assert assessment.status == "fresh"
    assert assessment.source_date is not None
    assert assessment.source_date.isoformat() == "2026-03-01"
    assert assessment.metadata_key == "published_on"


def test_freshness_evaluator_reports_unknown_when_metadata_missing() -> None:
    evaluator = FreshnessEvaluator(Config(max_source_age_days=30))
    chunk = Chunk(
        content="Undated text.",
        source_url="https://example.com/unknown",
        metadata={},
    )

    assessment = evaluator.assess(chunk)

    assert assessment.status == "unknown"
    assert assessment.source_date is None
    assert assessment.age_days is None
