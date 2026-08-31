"""Persistence protocol shared by pipeline-runner and bot-worker."""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

from ai_news_feed.domain.models import (
    CollectionCursor,
    DeliveryReceipt,
    Digest,
    DuplicateLink,
    InterestProfile,
    Material,
    NewsCluster,
    ScreeningResult,
    SourceConfig,
)


class DuplicateSourceError(ValueError):
    def __init__(self, normalized_locator: str, existing_source_id: str) -> None:
        self.normalized_locator = normalized_locator
        self.existing_source_id = existing_source_id
        super().__init__(f"source already exists: {normalized_locator}")


class ConcurrentUpdateError(RuntimeError):
    """An optimistic write used an obsolete InterestProfile.version."""


@dataclass(frozen=True)
class PendingDigestPost:
    digest_id: str
    position: int
    text: str


class Repository(Protocol):
    async def load_context(
        self,
        profile_id: str,
    ) -> tuple[tuple[SourceConfig, ...], InterestProfile]: ...

    async def list_sources(self, *, include_disabled: bool = False) -> tuple[SourceConfig, ...]: ...

    async def add_source(self, source: SourceConfig) -> SourceConfig: ...

    async def delete_source(self, source_id: str) -> bool: ...

    async def update_source_cursor(
        self,
        source_id: str,
        cursor: CollectionCursor,
    ) -> SourceConfig: ...

    async def get_interest_profile(self, profile_id: str) -> InterestProfile: ...

    async def create_interest_profile(self, profile: InterestProfile) -> InterestProfile: ...

    async def update_interest_profile(
        self,
        profile_id: str,
        *,
        description: str,
        expected_version: int,
        updated_by_telegram_user_id: int | None,
        name: str | None = None,
    ) -> InterestProfile: ...

    async def find_materials_by_content_hashes(
        self,
        content_hashes: Collection[str],
    ) -> tuple[Material, ...]: ...

    async def find_materials_by_ids(
        self, material_ids: Collection[str]
    ) -> tuple[Material, ...]: ...

    async def list_materials_since(self, published_since: datetime) -> tuple[Material, ...]: ...

    async def save_processing_result(
        self,
        *,
        materials: Sequence[Material],
        clusters: Sequence[NewsCluster],
        duplicate_links: Sequence[DuplicateLink],
        screening_results: Sequence[ScreeningResult],
        profile_id: str,
        profile_version: int,
        source_cursors: Mapping[str, CollectionCursor] | None = None,
        digest: Digest | None = None,
        channel_id: str | None = None,
    ) -> None: ...

    async def prepare_digest(self, digest: Digest, *, channel_id: str) -> None: ...

    async def list_pending_digests(self) -> tuple[Digest, ...]: ...

    async def list_pending_digest_posts(self, digest_id: str) -> tuple[PendingDigestPost, ...]: ...

    async def mark_digest_post_sent(
        self,
        digest_id: str,
        position: int,
        *,
        telegram_message_id: int,
        sent_at: datetime,
    ) -> None: ...

    async def get_delivery_receipt(self, digest_id: str) -> DeliveryReceipt | None: ...

    async def close(self) -> None: ...
