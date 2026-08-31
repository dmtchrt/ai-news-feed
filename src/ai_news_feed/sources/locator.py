"""Canonical source identity shared by bot and persistence adapters."""

from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import urlsplit

from ai_news_feed.domain.models import CollectorKind, SourceKind
from ai_news_feed.normalization import canonicalize_url
from ai_news_feed.sources.telegram import telegram_handle


@dataclass(frozen=True)
class ParsedSourceLocator:
    kind: SourceKind
    locator: str
    normalized_locator: str
    collector: CollectorKind


def parse_source_locator(value: str) -> ParsedSourceLocator:
    """Validate user input and select a conservative default collector."""
    raw = value.strip()
    if not raw:
        raise ValueError("источник не может быть пустым")

    if raw.startswith("@") or _is_telegram_url(raw):
        handle = telegram_handle(raw).lower()
        locator = f"@{handle}"
        return ParsedSourceLocator(
            kind=SourceKind.TELEGRAM,
            locator=locator,
            normalized_locator=locator,
            collector=CollectorKind.WEB_PREVIEW,
        )

    try:
        locator = canonicalize_url(raw)
    except ValueError as exc:
        raise ValueError("пришлите абсолютный http(s)-URL или Telegram @handle") from exc
    path = urlsplit(locator).path.casefold()
    collector = (
        CollectorKind.NATIVE_RSS
        if path.endswith((".rss", ".xml", ".atom")) or "/feed" in path
        else CollectorKind.UNIVERSAL_SCRAPER
    )
    return ParsedSourceLocator(
        kind=SourceKind.WEBSITE,
        locator=locator,
        normalized_locator=locator,
        collector=collector,
    )


def normalize_source_locator(locator: str, kind: SourceKind | None = None) -> str:
    """Return the value stored in the database unique-key column."""
    if kind is SourceKind.TELEGRAM or (kind is None and _looks_like_telegram(locator)):
        return f"@{telegram_handle(locator).lower()}"
    return canonicalize_url(locator)


def _looks_like_telegram(value: str) -> bool:
    return value.strip().startswith("@") or _is_telegram_url(value)


def _is_telegram_url(value: str) -> bool:
    parts = urlsplit(value.strip())
    return parts.scheme.lower() in {"http", "https"} and parts.netloc.lower() in {
        "t.me",
        "www.t.me",
        "telegram.me",
        "www.telegram.me",
    }
