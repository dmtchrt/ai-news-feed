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


class SummaryLength(StrEnum):
    """Digest summary length preset (see ClusterSummarizer)."""

    BRIEF = "brief"
    NORMAL = "normal"
    DETAILED = "detailed"


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


class InterestProfile(ContractModel):
    id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    enabled: bool = True
    version: int = Field(default=1, ge=1)
    freshness_days: int = Field(default=7, ge=1, le=365)
    summary_length: SummaryLength = SummaryLength.NORMAL
    tone_instructions: str | None = None
    created_at: datetime
    updated_at: datetime
    updated_by_telegram_user_id: int | None = Field(default=None, ge=1)

    @field_validator("id", "name", "description")
    @classmethod
    def profile_text_is_stripped(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("profile text must not be blank")
        return stripped

    @field_validator("tone_instructions")
    @classmethod
    def tone_instructions_is_stripped(cls, value: str | None) -> str | None:
        if value is None:
            return None
        stripped = value.strip()
        return stripped or None

    @field_validator("created_at", "updated_at")
    @classmethod
    def profile_time_is_utc(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("profile timestamps must be timezone-aware")
        return value.astimezone(UTC)

    @model_validator(mode="after")
    def update_cannot_predate_creation(self) -> InterestProfile:
        if self.updated_at < self.created_at:
            raise ValueError("updated_at cannot predate created_at")
        return self


class Material(ContractModel):
    id: str = Field(min_length=1)
    source_id: str = Field(min_length=1)
    external_id: str = Field(min_length=1)
    original_url: str = Field(min_length=1)
    published_at: datetime
    fetched_at: datetime
    title: str = Field(min_length=1)
    text: str = Field(min_length=1)
    language: str | None = Field(default=None, min_length=1)
    content_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    metadata: dict[str, JsonValue] = Field(default_factory=dict)

    @field_validator("id", "source_id", "external_id", "title", "text")
    @classmethod
    def material_text_is_stripped(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("material text must not be blank")
        return stripped

    @field_validator("language")
    @classmethod
    def language_is_stripped(cls, value: str | None) -> str | None:
        if value is None:
            return None
        stripped = value.strip()
        return stripped or None

    @field_validator("original_url")
    @classmethod
    def material_url_is_http(cls, value: str) -> str:
        parsed = urlsplit(value)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("original_url must be an absolute HTTP(S) URL")
        return value

    @field_validator("published_at", "fetched_at")
    @classmethod
    def material_time_is_utc(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("material timestamps must be timezone-aware")
        return value.astimezone(UTC)


class DuplicateKind(StrEnum):
    EXACT_CONTENT = "exact_content"
    SEMANTIC = "semantic"


class DuplicateLink(ContractModel):
    material_id: str = Field(min_length=1)
    duplicate_of_id: str = Field(min_length=1)
    kind: DuplicateKind
    similarity: float = Field(ge=0.0, le=1.0)

    @field_validator("material_id", "duplicate_of_id")
    @classmethod
    def duplicate_ids_are_stripped(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("duplicate ids must not be blank")
        return stripped

    @model_validator(mode="after")
    def duplicate_has_a_distinct_target(self) -> DuplicateLink:
        if self.material_id == self.duplicate_of_id:
            raise ValueError("a material cannot duplicate itself")
        return self


class ExactDeduplicationResult(ContractModel):
    unique_materials: tuple[Material, ...] = ()
    duplicate_links: tuple[DuplicateLink, ...] = ()


class NewsCluster(ContractModel):
    id: str = Field(min_length=1)
    material_ids: tuple[str, ...] = Field(min_length=1)
    representative_id: str = Field(min_length=1)
    similarities: dict[str, float]

    @field_validator("id", "representative_id")
    @classmethod
    def cluster_ids_are_stripped(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("cluster ids must not be blank")
        return stripped

    @field_validator("material_ids")
    @classmethod
    def material_ids_are_stripped(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        stripped = tuple(value.strip() for value in values)
        if any(not value for value in stripped):
            raise ValueError("cluster material ids must not be blank")
        return stripped

    @model_validator(mode="after")
    def cluster_is_consistent(self) -> NewsCluster:
        if len(set(self.material_ids)) != len(self.material_ids):
            raise ValueError("material_ids must be unique")
        if self.representative_id not in self.material_ids:
            raise ValueError("representative_id must belong to the cluster")
        if set(self.similarities) != set(self.material_ids):
            raise ValueError("similarities must contain every cluster material")
        if any(score < 0.0 or score > 1.0 for score in self.similarities.values()):
            raise ValueError("similarities must be in range 0..1")
        if self.similarities[self.representative_id] != 1.0:
            raise ValueError("representative similarity must equal 1")
        return self


class ClusterBatch(ContractModel):
    clusters: tuple[NewsCluster, ...] = ()
    duplicate_links: tuple[DuplicateLink, ...] = ()


class ScreeningResult(ContractModel):
    cluster_id: str = Field(min_length=1)
    relevance_score: float = Field(ge=0.0, le=1.0)
    noise_score: float = Field(ge=0.0, le=1.0)
    uncertain: bool = False
    reason: str = Field(min_length=1)
    model: str = Field(min_length=1)
    prompt_version: str = Field(min_length=1)

    @field_validator("cluster_id", "reason", "model", "prompt_version")
    @classmethod
    def screening_text_is_stripped(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("screening text must not be blank")
        return stripped

    def accepted(self, *, relevance_threshold: float, noise_threshold: float) -> bool:
        """Apply adjustable policy in code; uncertain classifications pass through."""
        if not 0.0 <= relevance_threshold <= 1.0:
            raise ValueError("relevance_threshold must be in range 0..1")
        if not 0.0 <= noise_threshold <= 1.0:
            raise ValueError("noise_threshold must be in range 0..1")
        return self.uncertain or (
            self.relevance_score >= relevance_threshold and self.noise_score <= noise_threshold
        )


class DigestItem(ContractModel):
    cluster_id: str = Field(min_length=1)
    summary: str = Field(min_length=1)
    source_links: tuple[str, ...] = Field(min_length=1)
    model: str = Field(min_length=1)
    prompt_version: str = Field(min_length=1)

    @field_validator("cluster_id", "summary", "model", "prompt_version")
    @classmethod
    def digest_text_is_stripped(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("digest text must not be blank")
        return stripped

    @field_validator("source_links")
    @classmethod
    def digest_links_are_http(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        stripped = tuple(value.strip() for value in values)
        for value in stripped:
            parsed = urlsplit(value)
            if parsed.scheme not in {"http", "https"} or not parsed.netloc:
                raise ValueError("source links must be absolute HTTP(S) URLs")
        return stripped


class Digest(ContractModel):
    """A deterministic, Telegram-sized delivery plan."""

    id: str = Field(min_length=1)
    profile_id: str = Field(min_length=1)
    profile_version: int = Field(ge=1)
    items: tuple[DigestItem, ...] = Field(min_length=1)
    posts: tuple[str, ...] = Field(min_length=1)
    created_at: datetime

    @field_validator("id", "profile_id")
    @classmethod
    def digest_ids_are_stripped(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("digest ids must not be blank")
        return stripped

    @field_validator("posts")
    @classmethod
    def digest_posts_are_nonempty(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        stripped = tuple(value.strip() for value in values)
        if any(not value for value in stripped):
            raise ValueError("digest posts must not be blank")
        return stripped

    @field_validator("created_at")
    @classmethod
    def digest_time_is_utc(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("created_at must be timezone-aware")
        return value.astimezone(UTC)


class DeliveryReceipt(ContractModel):
    digest_id: str = Field(min_length=1)
    telegram_message_ids: tuple[int, ...] = Field(min_length=1)
    sent_at: datetime

    @field_validator("digest_id")
    @classmethod
    def receipt_id_is_stripped(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("digest_id must not be blank")
        return stripped

    @field_validator("telegram_message_ids")
    @classmethod
    def message_ids_are_positive(cls, values: tuple[int, ...]) -> tuple[int, ...]:
        if any(value < 1 for value in values):
            raise ValueError("telegram message ids must be positive")
        return values

    @field_validator("sent_at")
    @classmethod
    def sent_time_is_utc(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("sent_at must be timezone-aware")
        return value.astimezone(UTC)
