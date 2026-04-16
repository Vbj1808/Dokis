"""Freshness evaluation for source metadata-backed temporal trust checks."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Any, Literal

from dokis.config import Config
from dokis.models import Chunk

FreshnessStatus = Literal["fresh", "stale", "unknown", "not_applicable"]

_DEFAULT_DATE_METADATA_KEYS = (
    "published_at",
    "publication_date",
    "published_date",
    "updated_at",
    "last_updated",
    "date",
    "issued_at",
    "year",
)


@dataclass(frozen=True)
class FreshnessAssessment:
    """Compact freshness assessment for a single chunk source."""

    status: FreshnessStatus
    source_date: date | None = None
    age_days: int | None = None
    metadata_key: str | None = None
    note: str | None = None


class FreshnessEvaluator:
    """Derive source age and freshness from local chunk metadata only."""

    def __init__(self, config: Config) -> None:
        self._config = config

    def assess(self, chunk: Chunk) -> FreshnessAssessment:
        """Return freshness details for one chunk."""
        if self._config.max_source_age_days is None:
            return FreshnessAssessment(status="not_applicable")

        source_date, metadata_key, note = self._extract_source_date(chunk.metadata)
        if source_date is None:
            return FreshnessAssessment(
                status="unknown",
                metadata_key=metadata_key,
                note=note or "No parseable source date was found in chunk metadata.",
            )

        today = datetime.now(timezone.utc).date()
        age_days = max((today - source_date).days, 0)
        status: FreshnessStatus = (
            "stale" if age_days > self._config.max_source_age_days else "fresh"
        )
        return FreshnessAssessment(
            status=status,
            source_date=source_date,
            age_days=age_days,
            metadata_key=metadata_key,
            note=note,
        )

    def _extract_source_date(
        self,
        metadata: dict[str, Any],
    ) -> tuple[date | None, str | None, str | None]:
        keys = self._candidate_metadata_keys(metadata)
        if not keys:
            return None, None, "Chunk metadata is empty."

        for key in keys:
            if key not in metadata:
                continue
            parsed, note = self._parse_date_value(metadata[key])
            if parsed is not None:
                return parsed, key, note
            return None, key, note
        return None, None, "No configured freshness metadata key was present."

    def _candidate_metadata_keys(self, metadata: dict[str, Any]) -> list[str]:
        keys: list[str] = []
        if self._config.source_date_metadata_key is not None:
            keys.append(self._config.source_date_metadata_key)
        for key in _DEFAULT_DATE_METADATA_KEYS:
            if key not in keys:
                keys.append(key)
        lower_to_actual = {str(k).lower(): str(k) for k in metadata}
        resolved: list[str] = []
        for key in keys:
            actual = lower_to_actual.get(key.lower())
            if actual is not None and actual not in resolved:
                resolved.append(actual)
        return resolved

    def _parse_date_value(
        self,
        value: Any,
    ) -> tuple[date | None, str | None]:
        if isinstance(value, datetime):
            return value.date(), None
        if isinstance(value, date):
            return value, None
        if isinstance(value, int):
            if 1000 <= value <= 9999:
                return date(value, 1, 1), "Year-only metadata is treated as January 1."
            return None, f"Unsupported numeric date value: {value!r}."
        if isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                return None, "Date metadata value was empty."
            if stripped.isdigit() and len(stripped) == 4:
                return (
                    date(int(stripped), 1, 1),
                    "Year-only metadata is treated as January 1.",
                )
            normalized = stripped.replace("Z", "+00:00")
            for candidate in (
                normalized,
                normalized.split("T", 1)[0],
                normalized.split(" ", 1)[0],
                normalized.replace("/", "-"),
            ):
                try:
                    return date.fromisoformat(candidate), None
                except ValueError:
                    pass
                try:
                    return datetime.fromisoformat(candidate).date(), None
                except ValueError:
                    pass
            return None, f"Unsupported date format: {stripped!r}."
        return None, f"Unsupported date metadata type: {type(value).__name__}."
