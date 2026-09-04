"""Telegram public-preview and Telethon source connectors."""

from __future__ import annotations

import asyncio
import mimetypes
import re
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import httpx
from bs4 import BeautifulSoup, Tag
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
    effective_cursor,
    error_message,
    first_line,
    safe_filename,
    utc_datetime,
)


class TelegramPreviewSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    preview_base_url: str = "https://t.me/s"
    max_items: int = Field(default=50, ge=1, le=100)
    timeout_seconds: float = Field(default=20.0, gt=0, le=120)
    user_agent: str = Field(default=DEFAULT_USER_AGENT, min_length=1)
    headers: dict[str, str] = Field(default_factory=dict)


class TelethonSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_items: int = Field(default=100, ge=1, le=1000)
    download_documents: bool = True
    timeout_seconds: float = Field(default=60.0, gt=0, le=300)


class TelegramWebPreviewConnector:
    def __init__(self, client: httpx.AsyncClient | None = None) -> None:
        self._client = client

    async def collect(
        self,
        config: SourceConfig,
        cursor: CollectionCursor | None = None,
    ) -> CollectionBatch:
        _validate_telegram_config(config, CollectorKind.WEB_PREVIEW)
        settings = TelegramPreviewSettings.model_validate(config.settings)
        active_cursor = effective_cursor(config, cursor)
        if active_cursor is not None and active_cursor.message_id is None:
            raise ValueError("web preview cursor must use message_id")
        handle = telegram_handle(config.locator)
        preview_url = f"{settings.preview_base_url.rstrip('/')}/{handle}"
        headers = {"User-Agent": settings.user_agent, **settings.headers}
        if self._client is not None:
            return await self._collect_with_client(
                self._client,
                config,
                active_cursor,
                settings,
                preview_url,
                headers,
            )
        async with httpx.AsyncClient(follow_redirects=True) as client:
            return await self._collect_with_client(
                client,
                config,
                active_cursor,
                settings,
                preview_url,
                headers,
            )

    async def _collect_with_client(
        self,
        client: httpx.AsyncClient,
        config: SourceConfig,
        cursor: CollectionCursor | None,
        settings: TelegramPreviewSettings,
        preview_url: str,
        headers: dict[str, str],
    ) -> CollectionBatch:
        try:
            response = await client.get(
                preview_url,
                headers=headers,
                timeout=settings.timeout_seconds,
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            return CollectionBatch(
                next_cursor=cursor,
                errors=(
                    CollectionError(
                        source_id=config.id,
                        code="preview_fetch_failed",
                        message=error_message(exc),
                        retryable=True,
                    ),
                ),
            )

        soup = BeautifulSoup(response.text, "html.parser")
        cards = soup.select(".tgme_widget_message")[-settings.max_items :]
        if not cards:
            return CollectionBatch(
                next_cursor=cursor,
                errors=(
                    CollectionError(
                        source_id=config.id,
                        code="preview_unavailable",
                        message=(
                            f"{preview_url} returned no public posts; configure collector=telethon "
                            "after explicit source validation"
                        ),
                        retryable=False,
                    ),
                ),
            )

        items: list[RawItem] = []
        message_ids: list[int] = []
        errors: list[CollectionError] = []
        for card in cards:
            try:
                item = _preview_card_to_item(card, config)
            except ValueError as exc:
                data_post = card.get("data-post") if isinstance(card, Tag) else None
                errors.append(
                    CollectionError(
                        source_id=config.id,
                        code="preview_post_invalid",
                        message=error_message(exc),
                        external_id=str(data_post) if data_post else None,
                        retryable=False,
                    )
                )
                continue
            message_id = int(item.external_id)
            message_ids.append(message_id)
            if cursor is None or message_id > (cursor.message_id or 0):
                items.append(item)

        items.sort(key=lambda item: (item.published_at, int(item.external_id)))
        next_cursor = cursor
        if message_ids:
            next_cursor = CollectionCursor(message_id=max(message_ids))
        return CollectionBatch(
            raw_items=tuple(items),
            next_cursor=next_cursor,
            errors=tuple(errors),
        )


class TelethonConnector:
    """Collect from a caller-owned, authenticated Telethon client."""

    def __init__(self, client: Any, download_dir: Path | None = None) -> None:
        self._client = client
        self._download_dir = download_dir or Path(tempfile.gettempdir()) / "ai-news-feed"

    async def collect(
        self,
        config: SourceConfig,
        cursor: CollectionCursor | None = None,
    ) -> CollectionBatch:
        _validate_telegram_config(config, CollectorKind.TELETHON)
        settings = TelethonSettings.model_validate(config.settings)
        active_cursor = effective_cursor(config, cursor)
        if active_cursor is not None and active_cursor.message_id is None:
            raise ValueError("Telethon cursor must use message_id")
        last_message_id = active_cursor.message_id if active_cursor else 0
        handle = telegram_handle(config.locator)
        items: list[RawItem] = []
        errors: list[CollectionError] = []
        seen_ids: list[int] = []
        try:
            messages = aiter(
                self._client.iter_messages(
                    handle,
                    min_id=last_message_id,
                    reverse=True,
                    limit=settings.max_items,
                )
            )
            while True:
                try:
                    message = await asyncio.wait_for(
                        anext(messages),
                        timeout=settings.timeout_seconds,
                    )
                except StopAsyncIteration:
                    break
                message_id = int(message.id)
                seen_ids.append(message_id)
                attachments = await self._message_attachments(
                    config=config,
                    message=message,
                    settings=settings,
                    errors=errors,
                )
                raw_text = str(message.message).strip() if message.message else None
                item = RawItem(
                    source_id=config.id,
                    external_id=str(message_id),
                    original_url=f"https://t.me/{handle}/{message_id}",
                    published_at=utc_datetime(message.date),
                    title=first_line(raw_text),
                    raw_text=raw_text,
                    attachments=attachments,
                    metadata={
                        "collector": config.collector.value,
                        "telegram_channel": handle,
                        "message_id": message_id,
                        "views": _json_scalar(getattr(message, "views", None)),
                        "forwards": _json_scalar(getattr(message, "forwards", None)),
                        "grouped_id": _json_scalar(getattr(message, "grouped_id", None)),
                    },
                )
                items.append(item)
        except Exception as exc:  # Telethon exposes many RPC/network exception subclasses.
            message = (
                f"Telethon message iteration timed out after {settings.timeout_seconds:g} seconds"
                if isinstance(exc, TimeoutError)
                else f"{type(exc).__name__}: {error_message(exc)}"
            )
            return CollectionBatch(
                raw_items=tuple(items),
                next_cursor=active_cursor,
                errors=(
                    *errors,
                    CollectionError(
                        source_id=config.id,
                        code="telethon_collect_failed",
                        message=message,
                        retryable=True,
                    ),
                ),
            )

        next_cursor = active_cursor
        if seen_ids:
            next_cursor = CollectionCursor(message_id=max(seen_ids))
        return CollectionBatch(
            raw_items=tuple(items),
            next_cursor=next_cursor,
            errors=tuple(errors),
        )

    async def _message_attachments(
        self,
        *,
        config: SourceConfig,
        message: Any,
        settings: TelethonSettings,
        errors: list[CollectionError],
    ) -> tuple[Attachment, ...]:
        document = getattr(message, "document", None)
        if document is None:
            return ()
        file_info = getattr(message, "file", None)
        mime_type = _optional_string(
            getattr(file_info, "mime_type", None) or getattr(document, "mime_type", None)
        )
        size = _optional_int(getattr(file_info, "size", None) or getattr(document, "size", None))
        original_name = _optional_string(getattr(file_info, "name", None))
        suffix = _optional_string(getattr(file_info, "ext", None))
        if not suffix and mime_type:
            suffix = mimetypes.guess_extension(mime_type)
        name = safe_filename(
            original_name or f"document{suffix or ''}",
            f"document-{message.id}",
        )
        download_ref: str | None = None
        if settings.download_documents:
            source_dir = self._download_dir / safe_filename(config.id, "source")
            source_dir.mkdir(parents=True, exist_ok=True)
            target = source_dir / f"{message.id}-{name}"
            try:
                downloaded = await asyncio.wait_for(
                    self._client.download_media(message, file=str(target)),
                    timeout=settings.timeout_seconds,
                )
                if downloaded:
                    download_ref = str(Path(str(downloaded)).resolve())
                elif target.exists():
                    download_ref = str(target.resolve())
                else:
                    raise OSError("Telethon returned no downloaded file")
            except Exception as exc:  # Telethon download errors include RPC subclasses.
                message_text = (
                    f"Telethon media download timed out after {settings.timeout_seconds:g} seconds"
                    if isinstance(exc, TimeoutError)
                    else error_message(exc)
                )
                errors.append(
                    CollectionError(
                        source_id=config.id,
                        code="attachment_download_failed",
                        message=message_text,
                        external_id=str(message.id),
                        retryable=True,
                    )
                )
        return (
            Attachment(
                kind=AttachmentKind.DOCUMENT,
                name=name,
                size=size,
                download_ref=download_ref,
                mime_type=mime_type,
            ),
        )


def telegram_handle(locator: str) -> str:
    value = locator.strip()
    if value.startswith("@"):
        handle = value[1:]
    else:
        parts = urlsplit(value)
        if parts.netloc.lower() not in {
            "t.me",
            "www.t.me",
            "telegram.me",
            "www.telegram.me",
        }:
            raise ValueError(f"invalid Telegram locator: {locator}")
        handle = parts.path.removeprefix("/s/").strip("/").split("/", maxsplit=1)[0]
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{3,31}", handle):
        raise ValueError(f"invalid Telegram handle: {locator}")
    return handle


def _validate_telegram_config(config: SourceConfig, collector: CollectorKind) -> None:
    if config.kind is not SourceKind.TELEGRAM:
        raise ValueError("Telegram connector requires kind=telegram")
    if config.collector is not collector:
        raise ValueError(f"connector requires collector={collector.value}")


def _preview_card_to_item(card: Tag, config: SourceConfig) -> RawItem:
    data_post = card.get("data-post")
    if not isinstance(data_post, str) or "/" not in data_post:
        raise ValueError("Telegram preview post has no data-post identity")
    _, external_id = data_post.rsplit("/", maxsplit=1)
    if not external_id.isdigit():
        raise ValueError(f"invalid Telegram message id: {external_id}")
    date_link = card.select_one("a.tgme_widget_message_date")
    time_element = card.select_one("time[datetime]")
    datetime_value = time_element.get("datetime") if isinstance(time_element, Tag) else None
    if not isinstance(datetime_value, str):
        raise ValueError(f"Telegram preview post {data_post} has no publication date")
    published_at = datetime.fromisoformat(datetime_value.replace("Z", "+00:00")).astimezone(UTC)
    href = date_link.get("href") if isinstance(date_link, Tag) else None
    original_url = str(href) if isinstance(href, str) else f"https://t.me/{data_post}"
    text_element = card.select_one(".tgme_widget_message_text")
    raw_text = text_element.get_text("\n", strip=True) if text_element else None
    raw_html = str(text_element) if text_element else None
    return RawItem(
        source_id=config.id,
        external_id=external_id,
        original_url=original_url,
        published_at=published_at,
        title=first_line(raw_text),
        raw_text=raw_text,
        raw_html=raw_html,
        attachments=_preview_attachments(card),
        metadata={
            "collector": config.collector.value,
            "telegram_post": data_post,
        },
    )


def _preview_attachments(card: Tag) -> tuple[Attachment, ...]:
    attachments: list[Attachment] = []
    for document in card.select(".tgme_widget_message_document_wrap"):
        title_element = document.select_one(".tgme_widget_message_document_title")
        extra_element = document.select_one(".tgme_widget_message_document_extra")
        name = title_element.get_text(" ", strip=True) if title_element else "document"
        extra = extra_element.get_text(" ", strip=True) if extra_element else ""
        attachments.append(
            Attachment(
                kind=AttachmentKind.DOCUMENT,
                name=name,
                size=_human_size_to_bytes(extra),
                download_ref=None,
                mime_type=mimetypes.guess_type(name)[0],
            )
        )
    return tuple(attachments)


def _human_size_to_bytes(value: str) -> int | None:
    match = re.search(r"([\d.,]+)\s*(B|KB|MB|GB|Б|КБ|МБ|ГБ)\b", value, flags=re.I)
    if not match:
        return None
    number = float(match.group(1).replace(",", "."))
    unit = match.group(2).upper()
    multiplier = {
        "B": 1,
        "Б": 1,
        "KB": 1024,
        "КБ": 1024,
        "MB": 1024**2,
        "МБ": 1024**2,
        "GB": 1024**3,
        "ГБ": 1024**3,
    }[unit]
    return int(number * multiplier)


def _optional_string(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return None


def _json_scalar(value: object) -> str | int | float | bool | None:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)
