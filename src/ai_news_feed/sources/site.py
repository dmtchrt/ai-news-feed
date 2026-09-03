"""Configurable HTTP/HTML scraper for sites without feeds."""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from typing import Any
from zoneinfo import ZoneInfo

import httpx
from bs4 import BeautifulSoup, Tag
from pydantic import BaseModel, ConfigDict, Field

from ai_news_feed.domain.models import (
    CollectionBatch,
    CollectionCursor,
    CollectionError,
    CollectorKind,
    RawItem,
    SourceConfig,
    SourceKind,
)
from ai_news_feed.sources._shared import (
    DEFAULT_USER_AGENT,
    canonical_absolute_url,
    effective_cursor,
    error_message,
    fallback_external_id,
    is_after_rss_cursor,
    next_rss_cursor,
)


class UniversalSiteSettings(BaseModel):
    """Selectors stay in SourceConfig so a layout change does not change connector code."""

    model_config = ConfigDict(extra="forbid")

    link_selector: str = "a[href]"
    link_attribute: str = "href"
    include_url_pattern: str | None = None
    exclude_url_pattern: str | None = None
    detail_title_selector: str = "h1"
    detail_date_selector: str | None = None
    detail_date_attribute: str | None = "datetime"
    date_formats: tuple[str, ...] = ()
    max_items: int = Field(default=30, ge=1, le=200)
    timeout_seconds: float = Field(default=20.0, gt=0, le=120)
    user_agent: str = Field(default=DEFAULT_USER_AGENT, min_length=1)
    headers: dict[str, str] = Field(default_factory=dict)


class UniversalSiteConnector:
    def __init__(self, client: httpx.AsyncClient | None = None) -> None:
        self._client = client

    async def collect(
        self,
        config: SourceConfig,
        cursor: CollectionCursor | None = None,
    ) -> CollectionBatch:
        if config.kind is not SourceKind.WEBSITE:
            raise ValueError("universal scraper requires kind=website")
        if config.collector is not CollectorKind.UNIVERSAL_SCRAPER:
            raise ValueError("universal scraper requires collector=universal_scraper")
        settings = UniversalSiteSettings.model_validate(config.settings)
        active_cursor = effective_cursor(config, cursor)
        if self._client is not None:
            return await self._collect_with_client(
                self._client,
                config,
                active_cursor,
                settings,
            )
        async with httpx.AsyncClient(follow_redirects=True) as client:
            return await self._collect_with_client(client, config, active_cursor, settings)

    async def _collect_with_client(
        self,
        client: httpx.AsyncClient,
        config: SourceConfig,
        cursor: CollectionCursor | None,
        settings: UniversalSiteSettings,
    ) -> CollectionBatch:
        headers = {"User-Agent": settings.user_agent, **settings.headers}
        try:
            listing_response = await client.get(
                config.locator,
                headers=headers,
                timeout=settings.timeout_seconds,
            )
            listing_response.raise_for_status()
        except httpx.HTTPError as exc:
            return CollectionBatch(
                next_cursor=cursor,
                errors=(
                    CollectionError(
                        source_id=config.id,
                        code="listing_fetch_failed",
                        message=error_message(exc),
                        retryable=True,
                    ),
                ),
            )

        candidates = _discover_links(
            listing_response.text,
            str(listing_response.url),
            settings,
        )[: settings.max_items]
        if not candidates:
            return CollectionBatch(
                next_cursor=cursor,
                errors=(
                    CollectionError(
                        source_id=config.id,
                        code="selector_empty",
                        message=f"selector {settings.link_selector!r} matched no article URLs",
                        retryable=False,
                    ),
                ),
            )

        items: list[RawItem] = []
        positions: list[tuple[datetime, str]] = []
        errors: list[CollectionError] = []
        for article_url, listing_title in candidates:
            external_id = fallback_external_id(article_url)
            try:
                response = await client.get(
                    article_url,
                    headers=headers,
                    timeout=settings.timeout_seconds,
                )
                response.raise_for_status()
            except httpx.HTTPError as exc:
                errors.append(
                    CollectionError(
                        source_id=config.id,
                        code="article_fetch_failed",
                        message=error_message(exc),
                        external_id=external_id,
                        retryable=True,
                    )
                )
                continue

            soup = BeautifulSoup(response.text, "html.parser")
            title = _detail_title(soup, settings) or listing_title
            try:
                published_at = _detail_datetime(soup, settings)
            except ValueError as exc:
                errors.append(
                    CollectionError(
                        source_id=config.id,
                        code="article_date_missing",
                        message=f"{article_url}: {exc}",
                        external_id=external_id,
                        retryable=False,
                    )
                )
                continue

            item = RawItem(
                source_id=config.id,
                external_id=external_id,
                original_url=str(response.url),
                published_at=published_at,
                title=title,
                raw_html=response.text,
                metadata={
                    "collector": config.collector.value,
                    "listing_url": str(listing_response.url),
                },
            )
            positions.append((item.published_at, item.external_id))
            if is_after_rss_cursor(item.published_at, item.external_id, cursor):
                items.append(item)

        items.sort(key=lambda item: (item.published_at, item.external_id))
        return CollectionBatch(
            raw_items=tuple(items),
            next_cursor=next_rss_cursor(positions, cursor),
            errors=tuple(errors),
        )


def _discover_links(
    html: str,
    base_url: str,
    settings: UniversalSiteSettings,
) -> list[tuple[str, str | None]]:
    soup = BeautifulSoup(html, "html.parser")
    by_url: dict[str, str | None] = {}
    include = re.compile(settings.include_url_pattern) if settings.include_url_pattern else None
    exclude = re.compile(settings.exclude_url_pattern) if settings.exclude_url_pattern else None
    for element in soup.select(settings.link_selector):
        if not isinstance(element, Tag):
            continue
        href = element.get(settings.link_attribute)
        if not isinstance(href, str) or not href.strip():
            continue
        url = canonical_absolute_url(base_url, href)
        if include and not include.search(url):
            continue
        if exclude and exclude.search(url):
            continue
        title = element.get_text(" ", strip=True) or None
        existing = by_url.get(url)
        if existing is None or len(title or "") > len(existing):
            by_url[url] = title
    return list(by_url.items())


def _detail_title(soup: BeautifulSoup, settings: UniversalSiteSettings) -> str | None:
    element = soup.select_one(settings.detail_title_selector)
    if element:
        text = element.get_text(" ", strip=True)
        if text:
            return text
    for selector, attribute in (
        ('meta[property="og:title"]', "content"),
        ('meta[name="twitter:title"]', "content"),
    ):
        meta = soup.select_one(selector)
        value = meta.get(attribute) if isinstance(meta, Tag) else None
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _detail_datetime(soup: BeautifulSoup, settings: UniversalSiteSettings) -> datetime:
    candidates: list[str] = []
    if settings.detail_date_selector:
        element = soup.select_one(settings.detail_date_selector)
        if isinstance(element, Tag):
            value = (
                element.get(settings.detail_date_attribute)
                if settings.detail_date_attribute
                else element.get_text(" ", strip=True)
            )
            if isinstance(value, str):
                candidates.append(value)

    for selector, attribute in (
        ('meta[property="article:published_time"]', "content"),
        ('meta[name="date"]', "content"),
        ('meta[itemprop="datePublished"]', "content"),
        ("time[datetime]", "datetime"),
    ):
        element = soup.select_one(selector)
        value = element.get(attribute) if isinstance(element, Tag) else None
        if isinstance(value, str):
            candidates.append(value)

    candidates.extend(_json_ld_dates(soup))
    for value in candidates:
        parsed = _parse_datetime(value, settings.date_formats)
        if parsed is not None:
            return parsed
    raise ValueError("no parseable publication date in configured selector, metadata or JSON-LD")


def _json_ld_dates(soup: BeautifulSoup) -> list[str]:
    dates: list[str] = []
    for script in soup.select('script[type="application/ld+json"]'):
        try:
            payload: Any = json.loads(script.get_text())
        except (TypeError, json.JSONDecodeError):
            continue
        nodes = payload if isinstance(payload, list) else [payload]
        for node in nodes:
            if not isinstance(node, dict):
                continue
            graph = node.get("@graph")
            if isinstance(graph, list):
                nodes.extend(graph)
            value = node.get("datePublished")
            if isinstance(value, str):
                dates.append(value)
    return dates


def _parse_datetime(value: str, formats: tuple[str, ...]) -> datetime | None:
    normalized = value.strip()
    if not normalized:
        return None
    try:
        parsed = datetime.fromisoformat(normalized.replace("Z", "+00:00"))
        return _ensure_utc(parsed)
    except ValueError:
        pass
    try:
        return _ensure_utc(parsedate_to_datetime(normalized))
    except (TypeError, ValueError):
        pass
    for date_format in formats:
        try:
            return _ensure_utc(datetime.strptime(normalized, date_format))
        except ValueError:
            continue
    russian_date = _parse_russian_date(normalized)
    return _ensure_utc(russian_date) if russian_date is not None else None


_RUSSIAN_MONTHS = {
    "января": 1,
    "февраля": 2,
    "марта": 3,
    "апреля": 4,
    "мая": 5,
    "июня": 6,
    "июля": 7,
    "августа": 8,
    "сентября": 9,
    "октября": 10,
    "ноября": 11,
    "декабря": 12,
}


def _parse_russian_date(value: str) -> datetime | None:
    match = re.search(
        r"(?P<day>\d{1,2})\s+(?P<month>[А-Яа-яЁё]+)\s+(?P<year>\d{4})"
        r"(?:\s+(?P<hour>\d{1,2}):(?P<minute>\d{2}))?",
        value,
    )
    if not match:
        return None
    month = _RUSSIAN_MONTHS.get(match.group("month").lower())
    if month is None:
        return None
    return datetime(
        int(match.group("year")),
        month,
        int(match.group("day")),
        int(match.group("hour") or 0),
        int(match.group("minute") or 0),
    )


_SOURCE_LOCAL_TIMEZONE = ZoneInfo("Europe/Moscow")


def _ensure_utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        return value.replace(tzinfo=_SOURCE_LOCAL_TIMEZONE).astimezone(UTC)
    return value.astimezone(UTC)
