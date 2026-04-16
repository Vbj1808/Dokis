"""Domain enforcer - pre-retrieval URL allowlist filtering."""

from __future__ import annotations

import logging
import re
from typing import Literal
from urllib.parse import urlparse

from dokis.config import Config
from dokis.models import BlockedSource, Chunk

logger = logging.getLogger(__name__)
_HOSTLIKE_PATTERN = re.compile(r"^[A-Za-z0-9.-]+(?::\d+)?$")

BlockedReason = Literal[
    "domain_not_allowlisted",
    "malformed_source_url",
    "missing_source_url",
]


def _safe_log_url(url: str) -> str:
    """Return scheme+host only, stripping path/query/fragment.

    Used to sanitise URLs before writing them to log output, so that
    sensitive path segments (patient IDs, session tokens, etc.) are never
    emitted in HIPAA/GDPR deployments.
    """
    try:
        p = urlparse(url)
        if p.netloc:
            return f"{p.scheme}://{p.netloc}"
        # No netloc parsed — strip path by splitting on '/' to avoid leaking
        # sensitive path segments.
        return url.split("/")[0][:40]
    except Exception:  # noqa: BLE001
        return url.split("/")[0][:40]


class DomainEnforcer:
    """Filters chunks whose source URL is not on the configured allowlist.

    Sits at the front of the Dokis pipeline. Chunks that fail the domain
    check are removed before they can reach the LLM prompt.

    Args:
        config: Runtime configuration. Uses ``allowed_domains`` to decide
            which source URLs are permitted.
    """

    def __init__(self, config: Config) -> None:
        self._config = config

    def inspect(self, chunks: list[Chunk]) -> tuple[list[Chunk], list[BlockedSource]]:
        """Filter chunks and return structured blocked-source details."""
        if not self._config.allowed_domains:
            return list(chunks), []

        clean: list[Chunk] = []
        blocked: list[BlockedSource] = []

        for chunk in chunks:
            host, reason = self._classify_source_url(chunk.source_url)
            if host is None:
                logger.warning(
                    "Dokis: blocked source_url with reason=%s (host: %r).",
                    reason or "malformed_source_url",
                    _safe_log_url(chunk.source_url),
                )
                blocked.append(
                    BlockedSource(
                        url=chunk.source_url,
                        domain=None,
                        reason=reason or "malformed_source_url",
                    )
                )
            elif host in self._config.allowed_domains:
                clean.append(chunk)
            else:
                blocked.append(
                    BlockedSource(
                        url=chunk.source_url,
                        domain=host,
                        reason="domain_not_allowlisted",
                    )
                )

        return clean, blocked

    def filter(self, chunks: list[Chunk]) -> tuple[list[Chunk], list[str]]:
        """Remove chunks whose source URL is not on the allowlist.

        If ``config.allowed_domains`` is empty, all chunks pass through
        unmodified and the blocked list is empty.

        Malformed URLs are logged at WARNING level and treated as blocked
        rather than raising an exception.

        Args:
            chunks: Raw chunks from the retriever.

        Returns:
            A two-tuple of ``(clean_chunks, blocked_urls)`` where
            ``clean_chunks`` is the filtered list and ``blocked_urls``
            contains every source URL that was removed.
        """
        clean, blocked = self.inspect(chunks)
        return clean, [entry.url for entry in blocked]

    @staticmethod
    def _classify_source_url(
        url: str,
    ) -> tuple[str | None, BlockedReason | None]:
        """Parse a source URL and classify why it should be blocked.

        Args:
            url: The raw source URL string.

        Returns:
            A two-tuple of ``(normalised_host, blocked_reason)``.
            ``blocked_reason`` is ``None`` only when a host was extracted.
        """
        try:
            if not url.strip():
                return None, "missing_source_url"
            parsed = urlparse(url)
            host = parsed.hostname
            if host:
                if host.startswith("www."):
                    host = host[4:]
                return host, None

            bare_host = url.split("/", 1)[0]
            if _HOSTLIKE_PATTERN.fullmatch(bare_host):
                if bare_host.startswith("www."):
                    bare_host = bare_host[4:]
                return bare_host, None

            host = parsed.netloc or parsed.path
            if not host:
                return None, "missing_source_url"
            return None, "malformed_source_url"
        except Exception:  # noqa: BLE001 - urlparse rarely raises; catch all to honour "never raise"
            return None, "malformed_source_url"
