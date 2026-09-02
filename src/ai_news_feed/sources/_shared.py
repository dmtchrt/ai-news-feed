"""Internal helpers shared by source adapters."""

from __future__ import annotations

import hashlib
import re
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from pathlib import PurePath
from urllib.parse import urljoin, urlsplit, urlunsplit

from ai_news_feed.domain.models import CollectionCursor, SourceConfig

DEFAULT_USER_AGENT = "AI-News-Feed/0.1 (+personal feed; respectful polling)"


def effective_cursor(
    config: SourceConfig,
    explicit_cursor: CollectionCursor | None,
) -> CollectionCursor | None:
    return explicit_cursor if explicit_cursor is not None else config.cursor


def error_message(exc: BaseException) -> str:
    """Render an exception for CollectionError.message, which requires >=1 char.

    str(exc) alone is not safe: a bare TimeoutError/asyncio.TimeoutError (raised
    with no arguments) and a few httpx/Telethon internals stringify to "". Falling
    back to the exception's class name keeps the message non-empty without hiding
    what actually failed.
    """
    text = str(exc).strip()
    return text if text else type(exc).__name__


def utc_datetime(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def parse_rfc822_datetime(value: str) -> datetime:
    return utc_datetime(parsedate_to_datetime(value))


def rss_cursor_key(published_at: datetime, external_id: str) -> tuple[datetime, str]:
    return (utc_datetime(published_at), external_id)


def is_after_rss_cursor(
    published_at: datetime,
    external_id: str,
    cursor: CollectionCursor | None,
) -> bool:
    if cursor is None or cursor.published_at is None or cursor.external_id is None:
        return True
    return rss_cursor_key(published_at, external_id) > rss_cursor_key(
        cursor.published_at,
        cursor.external_id,
    )


def next_rss_cursor(
    positions: list[tuple[datetime, str]],
    current: CollectionCursor | None,
) -> CollectionCursor | None:
    if not positions:
        return current
    published_at, external_id = max(positions, key=lambda position: rss_cursor_key(*position))
    candidate = CollectionCursor(published_at=published_at, external_id=external_id)
    if current is None or current.published_at is None or current.external_id is None:
        return candidate
    if rss_cursor_key(published_at, external_id) > rss_cursor_key(
        current.published_at,
        current.external_id,
    ):
        return candidate
    return current


def canonical_absolute_url(base_url: str, href: str) -> str:
    absolute = urljoin(base_url, href.strip())
    parts = urlsplit(absolute)
    return urlunsplit(
        (
            parts.scheme.lower(),
            parts.netloc.lower(),
            re.sub(r"/{2,}", "/", parts.path) or "/",
            parts.query,
            "",
        )
    )


def fallback_external_id(url: str, title: str | None = None) -> str:
    identity = f"{url}\n{title or ''}".encode()
    return hashlib.sha256(identity).hexdigest()


def safe_filename(name: str, fallback: str) -> str:
    leaf = PurePath(name).name
    cleaned = re.sub(r"[^0-9A-Za-zА-Яа-я._-]+", "_", leaf).strip("._")
    return cleaned[:180] or fallback


def first_line(text: str | None, *, limit: int = 240) -> str | None:
    if not text:
        return None
    line = next((part.strip() for part in text.splitlines() if part.strip()), "")
    if not line:
        return None
    return line if len(line) <= limit else f"{line[: limit - 1]}…"
