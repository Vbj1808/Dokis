"""Domain enforcer - pre-retrieval URL allowlist filtering."""

from __future__ import annotations

import logging
from urllib.parse import urlparse

from dokis.config import Config
from dokis.models import Chunk

logger = logging.getLogger(__name__)


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

    def filter(
        self, chunks: list[Chunk]
    ) -> tuple[list[Chunk], list[str]]:
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
        if not self._config.allowed_domains:
            return list(chunks), []

        clean: list[Chunk] = []
        blocked: list[str] = []

        for chunk in chunks:
            host = self._extract_host(chunk.source_url)
            if host is None:
                # Malformed URL - treat as blocked.
                logger.warning(
                    "Dokis: malformed source_url (host: %r) - treating as blocked.",
                    _safe_log_url(chunk.source_url),
                )
                blocked.append(chunk.source_url)
            elif host in self._config.allowed_domains:
                clean.append(chunk)
            else:
                blocked.append(chunk.source_url)

        return clean, blocked

    @staticmethod
    def _extract_host(url: str) -> str | None:
        """Parse a URL and return its hostname with ``www.`` stripped.

        Args:
            url: The raw source URL string.

        Returns:
            The normalised hostname, or ``None`` if the URL cannot be parsed
            to a non-empty hostname.
        """
        try:
            parsed = urlparse(url)
            host = parsed.netloc or parsed.path
            if not host:
                return None
            # Strip www. prefix to match the normalised allowlist entries.
            if host.startswith("www."):
                host = host[4:]
            return host
        except Exception:  # noqa: BLE001 - urlparse rarely raises; catch all to honour "never raise"
            return None
