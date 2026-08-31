"""Strict data contracts at the source-collection boundary."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, JsonValue, field_validator, model_validator


class ContractModel(BaseModel):
    """Base for public contracts: reject misspelled or undocumented fields."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class SourceKind(StrEnum):
    WEBSITE = "website"
    TELEGRAM = "telegram"
    MANUAL = "manual"


class CollectorKind(StrEnum):
    NATIVE_RSS = "native_rss"
    RSS_BRIDGE = "rss_bridge"
    WEB_PREVIEW = "web_preview"
    TELETHON = "telethon"
    UNIVERSAL_SCRAPER = "universal_scraper"


class AttachmentKind(StrEnum):
    DOCUMENT = "document"
    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"
    OTHER = "other"


class CollectionCursor(ContractModel):
    """JSON-serializable high-water mark for RSS/site or Telegram collection."""

    published_at: datetime | None = None
    external_id: str | None = None
    message_id: int | None = Field(default=None, ge=0)

    @field_validator("published_at")
    @classmethod
    def published_at_is_utc(cls, value: datetime | None) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("published_at must be timezone-aware")
        return value.astimezone(UTC)

    @model_validator(mode="after")
    def cursor_has_a_position(self) -> CollectionCursor:
        rss_position = self.published_at is not None and self.external_id is not None
        telegram_position = self.message_id is not None
        rss_fields_are_partial = (self.published_at is None) != (self.external_id is None)
        if rss_fields_are_partial or rss_position == telegram_position:
            raise ValueError("cursor must contain exactly published_at+external_id or message_id")
        return self


class SourceConfig(ContractModel):
    id: str = Field(min_length=1)
    kind: SourceKind
    locator: str = Field(min_length=1)
    collector: CollectorKind
    enabled: bool = True
    settings: dict[str, JsonValue] = Field(default_factory=dict)
    cursor: CollectionCursor | None = None

    @field_validator("id", "locator")
    @classmethod
    def nonempty_string_is_stripped(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("value must not be blank")
        return stripped


class Attachment(ContractModel):
    kind: AttachmentKind
    name: str = Field(min_length=1)
    size: int | None = Field(default=None, ge=0)
    download_ref: str | None = Field(default=None, min_length=1)
    mime_type: str | None = Field(default=None, min_length=1)

    @field_validator("name")
    @classmethod
    def name_is_stripped(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("attachment name must not be blank")
        return stripped


class RawItem(ContractModel):
    source_id: str = Field(min_length=1)
    external_id: str = Field(min_length=1)
    original_url: str = Field(min_length=1)
    published_at: datetime
    title: str | None = None
    raw_text: str | None = None
    raw_html: str | None = None
    attachments: tuple[Attachment, ...] = ()
    metadata: dict[str, JsonValue] = Field(default_factory=dict)

    @field_validator("source_id", "external_id")
    @classmethod
    def identity_is_stripped(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("identity must not be blank")
        return stripped

    @field_validator("original_url")
    @classmethod
    def original_url_is_http(cls, value: str) -> str:
        stripped = value.strip()
        parsed = urlsplit(stripped)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("original_url must be an absolute HTTP(S) URL")
        return stripped

    @field_validator("published_at")
    @classmethod
    def published_at_is_utc(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("published_at must be timezone-aware")
        return value.astimezone(UTC)

    @field_validator("title", "raw_text", "raw_html")
    @classmethod
    def empty_optional_text_is_none(cls, value: str | None) -> str | None:
        if value is None:
            return None
        stripped = value.strip()
        return stripped or None


class CollectionError(ContractModel):
    source_id: str = Field(min_length=1)
    code: str = Field(min_length=1)
    message: str = Field(min_length=1)
    external_id: str | None = None
    retryable: bool = False


class CollectionBatch(ContractModel):
    raw_items: tuple[RawItem, ...] = ()
    next_cursor: CollectionCursor | None = None
    errors: tuple[CollectionError, ...] = ()
