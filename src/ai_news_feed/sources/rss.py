"""Native RSS and RSS-Bridge connectors."""

from __future__ import annotations

import calendar
from datetime import UTC, datetime
from time import struct_time
from typing import Any
from urllib.parse import urlencode, urlsplit

import feedparser
import httpx
from pydantic import BaseModel, ConfigDict, Field

from ai_news_feed.domain.models import (
    Attachment,
    AttachmentKind,
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


class NativeRssSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    fetch_full_text: bool = True
    max_items: int = Field(default=50, ge=1, le=500)
    timeout_seconds: float = Field(default=20.0, gt=0, le=120)
    user_agent: str = Field(default=DEFAULT_USER_AGENT, min_length=1)
    headers: dict[str, str] = Field(default_factory=dict)


class RssBridgeSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    bridge_base_url: str = "http://127.0.0.1:3000/"
    feed_url: str | None = None
    bridge: str = "Telegram"
    context: str = "By username"
    parameters: dict[str, str] = Field(default_factory=dict)
    max_items: int = Field(default=50, ge=1, le=500)
    timeout_seconds: float = Field(default=20.0, gt=0, le=120)
    user_agent: str = Field(default=DEFAULT_USER_AGENT, min_length=1)
    headers: dict[str, str] = Field(default_factory=dict)


class _RssCollector:
    def __init__(self, client: httpx.AsyncClient | None = None) -> None:
        self._client = client

    async def _collect_feed(
        self,
        *,
        config: SourceConfig,
        cursor: CollectionCursor | None,
        feed_url: str,
        fetch_full_text: bool,
        max_items: int,
        timeout_seconds: float,
        user_agent: str,
        headers: dict[str, str],
        telegram_cursor: bool,
    ) -> CollectionBatch:
        request_headers = {"User-Agent": user_agent, **headers}
        if self._client is not None:
            return await self._collect_with_client(
                client=self._client,
                config=config,
                cursor=cursor,
                feed_url=feed_url,
                fetch_full_text=fetch_full_text,
                max_items=max_items,
                timeout_seconds=timeout_seconds,
                headers=request_headers,
                telegram_cursor=telegram_cursor,
            )
        async with httpx.AsyncClient(follow_redirects=True) as client:
            return await self._collect_with_client(
                client=client,
                config=config,
                cursor=cursor,
                feed_url=feed_url,
                fetch_full_text=fetch_full_text,
                max_items=max_items,
                timeout_seconds=timeout_seconds,
                headers=request_headers,
                telegram_cursor=telegram_cursor,
            )

    async def _collect_with_client(
        self,
        *,
        client: httpx.AsyncClient,
        config: SourceConfig,
        cursor: CollectionCursor | None,
        feed_url: str,
        fetch_full_text: bool,
        max_items: int,
        timeout_seconds: float,
        headers: dict[str, str],
        telegram_cursor: bool,
    ) -> CollectionBatch:
        try:
            response = await client.get(feed_url, headers=headers, timeout=timeout_seconds)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            return CollectionBatch(
                next_cursor=cursor,
                errors=(
                    CollectionError(
                        source_id=config.id,
                        code="feed_fetch_failed",
                        message=error_message(exc),
                        retryable=True,
                    ),
                ),
            )

        parsed: Any = feedparser.parse(response.content)
        entries: list[Any] = list(parsed.entries[:max_items])
        errors: list[CollectionError] = []
        if not entries and bool(parsed.get("bozo")):
            errors.append(
                CollectionError(
                    source_id=config.id,
                    code="feed_parse_failed",
                    message=str(parsed.get("bozo_exception", "invalid feed")),
                    retryable=False,
                )
            )

        raw_items: list[RawItem] = []
        positions: list[tuple[datetime, str]] = []
        telegram_message_ids: list[int] = []
        for entry in entries:
            try:
                item = await self._entry_to_raw_item(
                    client=client,
                    config=config,
                    entry=entry,
                    feed_url=str(response.url),
                    fetch_full_text=fetch_full_text,
                    timeout_seconds=timeout_seconds,
                    headers=headers,
                    errors=errors,
                )
            except (TypeError, ValueError) as exc:
                errors.append(
                    CollectionError(
                        source_id=config.id,
                        code="entry_invalid",
                        message=error_message(exc),
                        external_id=_optional_string(entry.get("id")),
                        retryable=False,
                    )
                )
                continue
            positions.append((item.published_at, item.external_id))
            if telegram_cursor:
                message_id = int(item.external_id)
                telegram_message_ids.append(message_id)
                is_new = cursor is None or message_id > (cursor.message_id or 0)
            else:
                is_new = is_after_rss_cursor(item.published_at, item.external_id, cursor)
            if is_new:
                raw_items.append(item)

        raw_items.sort(key=lambda item: (item.published_at, item.external_id))
        next_cursor = next_rss_cursor(positions, cursor)
        if telegram_cursor:
            next_cursor = cursor
            if telegram_message_ids:
                next_cursor = CollectionCursor(message_id=max(telegram_message_ids))
        return CollectionBatch(
            raw_items=tuple(raw_items),
            next_cursor=next_cursor,
            errors=tuple(errors),
        )

    async def _entry_to_raw_item(
        self,
        *,
        client: httpx.AsyncClient,
        config: SourceConfig,
        entry: Any,
        feed_url: str,
        fetch_full_text: bool,
        timeout_seconds: float,
        headers: dict[str, str],
        errors: list[CollectionError],
    ) -> RawItem:
        title = _optional_string(entry.get("title"))
        link = _optional_string(entry.get("link"))
        if link is None:
            raise ValueError("RSS entry has no original link")
        original_url = canonical_absolute_url(feed_url, link)
        feed_external_id = (
            _optional_string(entry.get("id"))
            or _optional_string(entry.get("guid"))
            or fallback_external_id(original_url, title)
        )
        external_id = _telegram_external_id(original_url, feed_external_id, config.kind)
        published_at = _entry_datetime(entry)
        raw_html = _entry_html(entry)

        if fetch_full_text:
            try:
                article_response = await client.get(
                    original_url,
                    headers=headers,
                    timeout=timeout_seconds,
                )
                article_response.raise_for_status()
                raw_html = article_response.text
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

        return RawItem(
            source_id=config.id,
            external_id=external_id,
            original_url=original_url,
            published_at=published_at,
            title=title,
            raw_html=raw_html,
            attachments=_entry_attachments(entry),
            metadata={
                "collector": config.collector.value,
                "feed_url": feed_url,
                "author": _optional_string(entry.get("author")),
            },
        )


class NativeRssConnector(_RssCollector):
    async def collect(
        self,
        config: SourceConfig,
        cursor: CollectionCursor | None = None,
    ) -> CollectionBatch:
        if config.kind is not SourceKind.WEBSITE:
            raise ValueError("native RSS connector requires kind=website")
        if config.collector is not CollectorKind.NATIVE_RSS:
            raise ValueError("native RSS connector requires collector=native_rss")
        settings = NativeRssSettings.model_validate(config.settings)
        active_cursor = effective_cursor(config, cursor)
        return await self._collect_feed(
            config=config,
            cursor=active_cursor,
            feed_url=config.locator,
            fetch_full_text=settings.fetch_full_text,
            max_items=settings.max_items,
            timeout_seconds=settings.timeout_seconds,
            user_agent=settings.user_agent,
            headers=settings.headers,
            telegram_cursor=False,
        )


class RssBridgeConnector(_RssCollector):
    async def collect(
        self,
        config: SourceConfig,
        cursor: CollectionCursor | None = None,
    ) -> CollectionBatch:
        if config.collector is not CollectorKind.RSS_BRIDGE:
            raise ValueError("RSS-Bridge connector requires collector=rss_bridge")
        settings = RssBridgeSettings.model_validate(config.settings)
        active_cursor = effective_cursor(config, cursor)
        telegram_cursor = config.kind is SourceKind.TELEGRAM
        if active_cursor is not None:
            if telegram_cursor and active_cursor.message_id is None:
                raise ValueError("Telegram RSS-Bridge cursor must use message_id")
            if not telegram_cursor and active_cursor.message_id is not None:
                raise ValueError("website RSS-Bridge cursor must use published_at+external_id")
        feed_url = settings.feed_url or _rss_bridge_url(config, settings)
        return await self._collect_feed(
            config=config,
            cursor=active_cursor,
            feed_url=feed_url,
            fetch_full_text=False,
            max_items=settings.max_items,
            timeout_seconds=settings.timeout_seconds,
            user_agent=settings.user_agent,
            headers=settings.headers,
            telegram_cursor=telegram_cursor,
        )


def _rss_bridge_url(config: SourceConfig, settings: RssBridgeSettings) -> str:
    parameters = dict(settings.parameters)
    if config.kind is SourceKind.TELEGRAM:
        parameters.setdefault("u", _telegram_handle(config.locator))
    query = urlencode(
        {
            "action": "display",
            "bridge": settings.bridge,
            "context": settings.context,
            **parameters,
            "format": "Atom",
        }
    )
    return f"{settings.bridge_base_url.rstrip('/')}/?{query}"


def _telegram_handle(locator: str) -> str:
    value = locator.strip()
    if value.startswith("@"):
        return value[1:]
    parts = urlsplit(value)
    if parts.netloc.lower() in {"t.me", "www.t.me", "telegram.me", "www.telegram.me"}:
        path = parts.path.removeprefix("/s/").strip("/")
        if path:
            return path.split("/", maxsplit=1)[0]
    raise ValueError(f"invalid Telegram locator: {locator}")


def _telegram_external_id(
    original_url: str,
    feed_external_id: str,
    source_kind: SourceKind,
) -> str:
    if source_kind is not SourceKind.TELEGRAM:
        return feed_external_id
    for candidate in (original_url, feed_external_id):
        message_id = urlsplit(candidate).path.rstrip("/").rsplit("/", maxsplit=1)[-1]
        if message_id.isdigit():
            return message_id
    raise ValueError("Telegram RSS-Bridge entry has no numeric message id")


def _entry_datetime(entry: Any) -> datetime:
    parsed: struct_time | None = entry.get("published_parsed") or entry.get("updated_parsed")
    if parsed is None:
        raise ValueError("RSS entry has no published/updated date")
    return datetime.fromtimestamp(calendar.timegm(parsed), tz=UTC)


def _entry_html(entry: Any) -> str | None:
    content = entry.get("content")
    if content and isinstance(content, list):
        value = _optional_string(content[0].get("value"))
        if value:
            return value
    return _optional_string(entry.get("summary")) or _optional_string(entry.get("description"))


def _entry_attachments(entry: Any) -> tuple[Attachment, ...]:
    attachments: list[Attachment] = []
    for enclosure in entry.get("enclosures", []):
        href = _optional_string(enclosure.get("href"))
        mime_type = _optional_string(enclosure.get("type"))
        name = _optional_string(enclosure.get("title"))
        if not name and href:
            name = urlsplit(href).path.rsplit("/", maxsplit=1)[-1]
        size = _optional_int(enclosure.get("length"))
        attachments.append(
            Attachment(
                kind=_attachment_kind(mime_type),
                name=name or "attachment",
                size=size,
                download_ref=href,
                mime_type=mime_type,
            )
        )
    return tuple(attachments)


def _attachment_kind(mime_type: str | None) -> AttachmentKind:
    if not mime_type:
        return AttachmentKind.OTHER
    family = mime_type.split("/", maxsplit=1)[0].lower()
    return {
        "image": AttachmentKind.IMAGE,
        "video": AttachmentKind.VIDEO,
        "audio": AttachmentKind.AUDIO,
        "application": AttachmentKind.DOCUMENT,
        "text": AttachmentKind.DOCUMENT,
    }.get(family, AttachmentKind.OTHER)


def _optional_string(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _optional_int(value: object) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(str(value))
    except ValueError:
        return None
