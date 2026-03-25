"""Compliance scorer - computes the grounding rate and pass/fail decision."""

from __future__ import annotations

import logging

from dokis.config import Config
from dokis.models import Claim

logger = logging.getLogger(__name__)


class ComplianceScorer:
    """Computes the compliance rate from a list of grounded claims.

    The compliance rate is ``supported_claims / total_claims``. A result
    passes when the rate meets or exceeds ``config.min_citation_rate``.

    An empty claim list is treated as fully compliant (rate ``1.0``,
    ``passed=True``). A WARNING is logged so the caller is aware that
    no claims were evaluated — this is a recoverable edge case, not an
    error.

    Args:
        config: Runtime configuration. Uses ``min_citation_rate`` for the
            pass/fail threshold.
    """

    def __init__(self, config: Config) -> None:
        self._config = config

    def score(self, claims: list[Claim]) -> tuple[float, bool]:
        """Compute the compliance rate and pass/fail verdict.

        Args:
            claims: All :class:`~dokis.models.Claim` objects produced by
                the matcher for a single audit run.

        Returns:
            A two-tuple ``(compliance_rate, passed)`` where
            ``compliance_rate`` is a float in ``[0.0, 1.0]`` and
            ``passed`` is ``True`` when
            ``compliance_rate >= config.min_citation_rate``.
        """
        if not claims:
            logger.warning(
                "Dokis: no claims were extracted from the response. "
                "Scoring as fully compliant (1.0). "
                "Check that the response is non-empty and the extractor is working."
            )
            return 1.0, True

        supported = sum(1 for c in claims if c.supported)
        rate = supported / len(claims)
        passed = rate >= self._config.min_citation_rate
        return rate, passed
