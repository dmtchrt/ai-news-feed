"""Domain models shared by pipeline modules."""

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

__all__ = [
    "Attachment",
    "AttachmentKind",
    "CollectionBatch",
    "CollectionCursor",
    "CollectionError",
    "CollectorKind",
    "RawItem",
    "SourceConfig",
    "SourceKind",
]
