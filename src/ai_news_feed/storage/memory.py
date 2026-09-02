"""In-memory implementation of the complete Repository protocol for tests."""

from __future__ import annotations

from collections.abc import Collection, Iterable, Mapping, Sequence
from datetime import UTC, datetime

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
    SummaryLength,
)
from ai_news_feed.sources.locator import normalize_source_locator
from ai_news_feed.storage.base import (
    ConcurrentUpdateError,
    DuplicateSourceError,
    PendingDigestPost,
)


class InMemoryRepository:
    def __init__(
        self,
        *,
        sources: Iterable[SourceConfig] = (),
        interest_profiles: Iterable[InterestProfile] = (),
        materials: Iterable[Material] = (),
    ) -> None:
        self._sources = {source.id: source for source in sources}
        self._profiles = {profile.id: profile for profile in interest_profiles}
        self._materials = {material.id: material for material in materials}
        self._clusters: dict[str, NewsCluster] = {}
        self._duplicate_links: dict[tuple[str, str], DuplicateLink] = {}
        self._screening_results: dict[tuple[object, ...], ScreeningResult] = {}
        self._digests: dict[str, tuple[Digest, str]] = {}
        self._digest_message_ids: dict[tuple[str, int], tuple[int, datetime]] = {}

    async def load_context(
        self,
        profile_id: str,
    ) -> tuple[tuple[SourceConfig, ...], InterestProfile]:
        return await self.list_sources(), await self.get_interest_profile(profile_id)

    async def list_sources(self, *, include_disabled: bool = False) -> tuple[SourceConfig, ...]:
        return tuple(
            sorted(
                (source for source in self._sources.values() if include_disabled or source.enabled),
                key=lambda source: (source.kind.value, source.locator, source.id),
            )
        )

    async def add_source(self, source: SourceConfig) -> SourceConfig:
        normalized = normalize_source_locator(source.locator, source.kind)
        for existing in self._sources.values():
            if normalize_source_locator(existing.locator, existing.kind) != normalized:
                continue
            if existing.enabled:
                raise DuplicateSourceError(normalized, existing.id)
            restored = source.model_copy(update={"id": existing.id, "enabled": True})
            self._sources[existing.id] = restored
            return restored
        if source.id in self._sources:
            raise ValueError(f"source id already exists: {source.id}")
        self._sources[source.id] = source
        return source

    async def delete_source(self, source_id: str) -> bool:
        source = self._sources.get(source_id)
        if source is None or not source.enabled:
            return False
        self._sources[source_id] = source.model_copy(update={"enabled": False})
        return True

    async def update_source_cursor(
        self,
        source_id: str,
        cursor: CollectionCursor,
    ) -> SourceConfig:
        try:
            source = self._sources[source_id]
        except KeyError as exc:
            raise LookupError(f"source not found: {source_id}") from exc
        updated = source.model_copy(update={"cursor": cursor})
        self._sources[source_id] = updated
        return updated

    async def get_interest_profile(self, profile_id: str) -> InterestProfile:
        try:
            profile = self._profiles[profile_id]
        except KeyError as exc:
            raise LookupError(f"interest profile not found: {profile_id}") from exc
        if not profile.enabled:
            raise LookupError(f"interest profile is disabled: {profile_id}")
        return profile

    async def create_interest_profile(self, profile: InterestProfile) -> InterestProfile:
        if profile.id in self._profiles:
            raise ConcurrentUpdateError(f"interest profile already exists: {profile.id}")
        self._profiles[profile.id] = profile
        return profile

    async def update_interest_profile(
        self,
        profile_id: str,
        *,
        description: str,
        expected_version: int,
        updated_by_telegram_user_id: int | None,
        name: str | None = None,
    ) -> InterestProfile:
        try:
            profile = self._profiles[profile_id]
        except KeyError as exc:
            raise LookupError(f"interest profile not found: {profile_id}") from exc
        if profile.version != expected_version:
            raise ConcurrentUpdateError(
                f"interest profile {profile_id} changed from version {expected_version} "
                f"to {profile.version}"
            )
        updated = profile.model_copy(
            update={
                "name": name or profile.name,
                "description": description,
                "version": profile.version + 1,
                "updated_at": datetime.now(UTC),
                "updated_by_telegram_user_id": updated_by_telegram_user_id,
            }
        )
        updated = InterestProfile.model_validate(updated)
        self._profiles[profile_id] = updated
        return updated

    async def update_digest_freshness(
        self,
        profile_id: str,
        *,
        freshness_days: int,
        expected_version: int,
        updated_by_telegram_user_id: int | None,
    ) -> InterestProfile:
        return self._cas_update_profile(
            profile_id,
            expected_version=expected_version,
            updated_by_telegram_user_id=updated_by_telegram_user_id,
            update={"freshness_days": freshness_days},
        )

    async def update_digest_length(
        self,
        profile_id: str,
        *,
        summary_length: SummaryLength,
        expected_version: int,
        updated_by_telegram_user_id: int | None,
    ) -> InterestProfile:
        return self._cas_update_profile(
            profile_id,
            expected_version=expected_version,
            updated_by_telegram_user_id=updated_by_telegram_user_id,
            update={"summary_length": summary_length},
        )

    async def update_digest_tone(
        self,
        profile_id: str,
        *,
        tone_instructions: str | None,
        expected_version: int,
        updated_by_telegram_user_id: int | None,
    ) -> InterestProfile:
        return self._cas_update_profile(
            profile_id,
            expected_version=expected_version,
            updated_by_telegram_user_id=updated_by_telegram_user_id,
            update={"tone_instructions": tone_instructions.strip() if tone_instructions else None},
        )

    def _cas_update_profile(
        self,
        profile_id: str,
        *,
        expected_version: int,
        updated_by_telegram_user_id: int | None,
        update: dict[str, object],
    ) -> InterestProfile:
        try:
            profile = self._profiles[profile_id]
        except KeyError as exc:
            raise LookupError(f"interest profile not found: {profile_id}") from exc
        if profile.version != expected_version:
            raise ConcurrentUpdateError(
                f"interest profile {profile_id} changed from version {expected_version} "
                f"to {profile.version}"
            )
        updated = profile.model_copy(
            update={
                **update,
                "version": profile.version + 1,
                "updated_at": datetime.now(UTC),
                "updated_by_telegram_user_id": updated_by_telegram_user_id,
            }
        )
        updated = InterestProfile.model_validate(updated)
        self._profiles[profile_id] = updated
        return updated

    async def find_materials_by_content_hashes(
        self,
        content_hashes: Collection[str],
    ) -> tuple[Material, ...]:
        requested = set(content_hashes)
        return tuple(
            sorted(
                (
                    material
                    for material in self._materials.values()
                    if material.content_hash in requested
                ),
                key=_material_order,
            )
        )

    async def find_materials_by_ids(self, material_ids: Collection[str]) -> tuple[Material, ...]:
        requested = set(material_ids)
        return tuple(
            sorted(
                (material for material in self._materials.values() if material.id in requested),
                key=_material_order,
            )
        )

    async def list_materials_since(self, published_since: datetime) -> tuple[Material, ...]:
        if published_since.tzinfo is None or published_since.utcoffset() is None:
            raise ValueError("published_since must be timezone-aware")
        return tuple(
            sorted(
                (
                    material
                    for material in self._materials.values()
                    if material.published_at >= published_since
                ),
                key=_material_order,
            )
        )

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
    ) -> None:
        for material in materials:
            self._materials.setdefault(material.id, material)
        for cluster in clusters:
            self._clusters.setdefault(cluster.id, cluster)
        for link in duplicate_links:
            self._duplicate_links.setdefault((link.material_id, link.duplicate_of_id), link)
        for result in screening_results:
            key = (
                result.cluster_id,
                profile_id,
                profile_version,
                result.model,
                result.prompt_version,
            )
            self._screening_results.setdefault(key, result)
        for source_id, cursor in (source_cursors or {}).items():
            await self.update_source_cursor(source_id, cursor)
        if digest is not None:
            if channel_id is None:
                raise ValueError("channel_id is required with digest")
            await self.prepare_digest(digest, channel_id=channel_id)
        elif channel_id is not None:
            raise ValueError("digest is required with channel_id")

    async def prepare_digest(self, digest: Digest, *, channel_id: str) -> None:
        existing = self._digests.get(digest.id)
        value = (digest, channel_id)
        if existing is not None and existing != value:
            raise ValueError(f"digest id collision: {digest.id}")
        self._digests.setdefault(digest.id, value)

    async def list_pending_digests(self) -> tuple[Digest, ...]:
        pending: list[Digest] = []
        for digest, _ in self._digests.values():
            if await self.get_delivery_receipt(digest.id) is None:
                pending.append(digest)
        return tuple(sorted(pending, key=lambda digest: (digest.created_at, digest.id)))

    async def list_pending_digest_posts(self, digest_id: str) -> tuple[PendingDigestPost, ...]:
        try:
            digest, _ = self._digests[digest_id]
        except KeyError as exc:
            raise LookupError(f"digest not found: {digest_id}") from exc
        return tuple(
            PendingDigestPost(digest_id=digest_id, position=position, text=text)
            for position, text in enumerate(digest.posts)
            if (digest_id, position) not in self._digest_message_ids
        )

    async def mark_digest_post_sent(
        self,
        digest_id: str,
        position: int,
        *,
        telegram_message_id: int,
        sent_at: datetime,
    ) -> None:
        if sent_at.tzinfo is None or sent_at.utcoffset() is None:
            raise ValueError("sent_at must be timezone-aware")
        try:
            digest, _ = self._digests[digest_id]
            digest.posts[position]
        except (KeyError, IndexError) as exc:
            raise LookupError(f"digest post not found: {digest_id}/{position}") from exc
        key = (digest_id, position)
        existing = self._digest_message_ids.get(key)
        receipt = (telegram_message_id, sent_at.astimezone(UTC))
        if existing is not None and existing != receipt:
            raise ConcurrentUpdateError(f"digest post already sent: {digest_id}/{position}")
        self._digest_message_ids.setdefault(key, receipt)

    async def get_delivery_receipt(self, digest_id: str) -> DeliveryReceipt | None:
        try:
            digest, _ = self._digests[digest_id]
        except KeyError as exc:
            raise LookupError(f"digest not found: {digest_id}") from exc
        receipts = [
            self._digest_message_ids.get((digest_id, position))
            for position in range(len(digest.posts))
        ]
        if any(receipt is None for receipt in receipts):
            return None
        complete = [receipt for receipt in receipts if receipt is not None]
        return DeliveryReceipt(
            digest_id=digest_id,
            telegram_message_ids=tuple(receipt[0] for receipt in complete),
            sent_at=max(receipt[1] for receipt in complete),
        )

    async def close(self) -> None:
        return None


def _material_order(material: Material) -> tuple[object, ...]:
    return (material.published_at, material.id)
