"""Async PostgreSQL Repository implemented with SQLAlchemy Core and psycopg 3."""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import and_, func, select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.engine import RowMapping
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine, create_async_engine
from sqlalchemy.pool import NullPool

from ai_news_feed.domain.models import (
    CollectionCursor,
    CollectorKind,
    DeliveryReceipt,
    Digest,
    DigestItem,
    DigestSendTime,
    DuplicateKind,
    DuplicateLink,
    InterestProfile,
    Material,
    NewsCluster,
    ScreeningResult,
    SourceConfig,
    SourceKind,
    SummaryLength,
)
from ai_news_feed.sources.locator import normalize_source_locator
from ai_news_feed.storage.base import (
    ConcurrentUpdateError,
    DuplicateSourceError,
    PendingDigestPost,
    ScreeningReview,
)
from ai_news_feed.storage.tables import (
    cluster_materials,
    digest_items,
    digest_posts,
    digests,
    interest_profiles,
    news_clusters,
    sources,
)
from ai_news_feed.storage.tables import (
    duplicate_links as duplicate_links_table,
)
from ai_news_feed.storage.tables import (
    materials as materials_table,
)
from ai_news_feed.storage.tables import (
    screening_results as screening_results_table,
)


class PostgresRepository:
    def __init__(
        self,
        database_url: str | None = None,
        *,
        engine: AsyncEngine | None = None,
        pooled: bool = True,
    ) -> None:
        if engine is None and not database_url:
            raise ValueError("database_url or engine is required")
        self._owns_engine = engine is None
        if engine is not None:
            self._engine = engine
        elif pooled:
            self._engine = create_async_engine(
                _async_database_url(str(database_url)),
                pool_size=2,
                max_overflow=1,
                pool_pre_ping=True,
            )
        else:
            self._engine = create_async_engine(
                _async_database_url(str(database_url)),
                poolclass=NullPool,
            )

    @property
    def engine(self) -> AsyncEngine:
        return self._engine

    async def load_context(
        self,
        profile_id: str,
    ) -> tuple[tuple[SourceConfig, ...], InterestProfile]:
        async with self._engine.connect() as raw_connection:
            connection = await raw_connection.execution_options(isolation_level="REPEATABLE READ")
            async with connection.begin():
                source_rows = (
                    (
                        await connection.execute(
                            select(sources)
                            .where(sources.c.enabled.is_(True))
                            .order_by(sources.c.kind, sources.c.locator, sources.c.id)
                        )
                    )
                    .mappings()
                    .all()
                )
                profile_row = (
                    (
                        await connection.execute(
                            select(interest_profiles).where(interest_profiles.c.id == profile_id)
                        )
                    )
                    .mappings()
                    .one_or_none()
                )
        if profile_row is None:
            raise LookupError(f"interest profile not found: {profile_id}")
        profile = _profile_from_row(profile_row)
        if not profile.enabled:
            raise LookupError(f"interest profile is disabled: {profile_id}")
        return tuple(_source_from_row(row) for row in source_rows), profile

    async def list_sources(self, *, include_disabled: bool = False) -> tuple[SourceConfig, ...]:
        statement = select(sources)
        if not include_disabled:
            statement = statement.where(sources.c.enabled.is_(True))
        statement = statement.order_by(sources.c.kind, sources.c.locator, sources.c.id)
        async with self._engine.connect() as connection:
            rows = (await connection.execute(statement)).mappings().all()
        return tuple(_source_from_row(row) for row in rows)

    async def add_source(self, source: SourceConfig) -> SourceConfig:
        normalized = normalize_source_locator(source.locator, source.kind)
        async with self._engine.begin() as connection:
            existing = (
                (
                    await connection.execute(
                        select(sources)
                        .where(sources.c.normalized_locator == normalized)
                        .with_for_update()
                    )
                )
                .mappings()
                .one_or_none()
            )
            if existing is not None:
                if bool(existing["enabled"]):
                    raise DuplicateSourceError(normalized, str(existing["id"]))
                restored = source.model_copy(update={"id": str(existing["id"]), "enabled": True})
                row = (
                    (
                        await connection.execute(
                            update(sources)
                            .where(sources.c.id == existing["id"])
                            .values(**_source_values(restored, normalized), updated_at=func.now())
                            .returning(sources)
                        )
                    )
                    .mappings()
                    .one()
                )
                return _source_from_row(row)
            try:
                row = (
                    (
                        await connection.execute(
                            pg_insert(sources)
                            .values(**_source_values(source, normalized))
                            .returning(sources)
                        )
                    )
                    .mappings()
                    .one()
                )
            except IntegrityError as exc:
                raise DuplicateSourceError(normalized, source.id) from exc
        return _source_from_row(row)

    async def delete_source(self, source_id: str) -> bool:
        async with self._engine.begin() as connection:
            result = await connection.execute(
                update(sources)
                .where(and_(sources.c.id == source_id, sources.c.enabled.is_(True)))
                .values(enabled=False, updated_at=func.now())
            )
        return bool(result.rowcount)

    async def update_source_cursor(
        self,
        source_id: str,
        cursor: CollectionCursor,
    ) -> SourceConfig:
        async with self._engine.begin() as connection:
            row = (
                (
                    await connection.execute(
                        update(sources)
                        .where(sources.c.id == source_id)
                        .values(cursor=cursor.model_dump(mode="json"), updated_at=func.now())
                        .returning(sources)
                    )
                )
                .mappings()
                .one_or_none()
            )
        if row is None:
            raise LookupError(f"source not found: {source_id}")
        return _source_from_row(row)

    async def get_interest_profile(self, profile_id: str) -> InterestProfile:
        async with self._engine.connect() as connection:
            row = (
                (
                    await connection.execute(
                        select(interest_profiles).where(interest_profiles.c.id == profile_id)
                    )
                )
                .mappings()
                .one_or_none()
            )
        if row is None:
            raise LookupError(f"interest profile not found: {profile_id}")
        profile = _profile_from_row(row)
        if not profile.enabled:
            raise LookupError(f"interest profile is disabled: {profile_id}")
        return profile

    async def create_interest_profile(self, profile: InterestProfile) -> InterestProfile:
        try:
            async with self._engine.begin() as connection:
                row = (
                    (
                        await connection.execute(
                            pg_insert(interest_profiles)
                            .values(**_profile_values(profile))
                            .returning(interest_profiles)
                        )
                    )
                    .mappings()
                    .one()
                )
        except IntegrityError as exc:
            raise ConcurrentUpdateError(f"interest profile already exists: {profile.id}") from exc
        return _profile_from_row(row)

    async def update_interest_profile(
        self,
        profile_id: str,
        *,
        description: str,
        expected_version: int,
        updated_by_telegram_user_id: int | None,
        name: str | None = None,
    ) -> InterestProfile:
        values: dict[str, Any] = {
            "description": description.strip(),
            "version": interest_profiles.c.version + 1,
            "updated_at": func.now(),
            "updated_by_telegram_user_id": updated_by_telegram_user_id,
        }
        if not values["description"]:
            raise ValueError("interest description must not be blank")
        if name is not None:
            values["name"] = name.strip()
            if not values["name"]:
                raise ValueError("interest name must not be blank")
        async with self._engine.begin() as connection:
            row = (
                (
                    await connection.execute(
                        update(interest_profiles)
                        .where(
                            and_(
                                interest_profiles.c.id == profile_id,
                                interest_profiles.c.version == expected_version,
                            )
                        )
                        .values(**values)
                        .returning(interest_profiles)
                    )
                )
                .mappings()
                .one_or_none()
            )
            if row is None:
                actual = await connection.scalar(
                    select(interest_profiles.c.version).where(interest_profiles.c.id == profile_id)
                )
                if actual is None:
                    raise LookupError(f"interest profile not found: {profile_id}")
                raise ConcurrentUpdateError(
                    f"interest profile {profile_id} changed from version {expected_version} "
                    f"to {actual}"
                )
        return _profile_from_row(row)

    async def update_digest_freshness(
        self,
        profile_id: str,
        *,
        freshness_days: int,
        expected_version: int,
        updated_by_telegram_user_id: int | None,
    ) -> InterestProfile:
        return await self._cas_update_profile(
            profile_id,
            values={"freshness_days": freshness_days},
            expected_version=expected_version,
            updated_by_telegram_user_id=updated_by_telegram_user_id,
        )

    async def update_digest_length(
        self,
        profile_id: str,
        *,
        summary_length: SummaryLength,
        expected_version: int,
        updated_by_telegram_user_id: int | None,
    ) -> InterestProfile:
        return await self._cas_update_profile(
            profile_id,
            values={"summary_length": summary_length.value},
            expected_version=expected_version,
            updated_by_telegram_user_id=updated_by_telegram_user_id,
        )

    async def update_digest_tone(
        self,
        profile_id: str,
        *,
        tone_instructions: str | None,
        expected_version: int,
        updated_by_telegram_user_id: int | None,
    ) -> InterestProfile:
        normalized_tone = (
            (tone_instructions.strip() or None) if tone_instructions is not None else None
        )
        return await self._cas_update_profile(
            profile_id,
            values={"tone_instructions": normalized_tone},
            expected_version=expected_version,
            updated_by_telegram_user_id=updated_by_telegram_user_id,
        )

    async def update_digest_send_times(
        self,
        profile_id: str,
        *,
        digest_send_times: tuple[DigestSendTime, ...],
        expected_version: int,
        updated_by_telegram_user_id: int | None,
    ) -> InterestProfile:
        return await self._cas_update_profile(
            profile_id,
            values={
                "digest_send_times": [
                    send_time.model_dump(mode="json") for send_time in digest_send_times
                ]
            },
            expected_version=expected_version,
            updated_by_telegram_user_id=updated_by_telegram_user_id,
        )

    async def _cas_update_profile(
        self,
        profile_id: str,
        *,
        values: dict[str, Any],
        expected_version: int,
        updated_by_telegram_user_id: int | None,
    ) -> InterestProfile:
        async with self._engine.begin() as connection:
            row = (
                (
                    await connection.execute(
                        update(interest_profiles)
                        .where(
                            and_(
                                interest_profiles.c.id == profile_id,
                                interest_profiles.c.version == expected_version,
                            )
                        )
                        .values(
                            **values,
                            version=interest_profiles.c.version + 1,
                            updated_at=func.now(),
                            updated_by_telegram_user_id=updated_by_telegram_user_id,
                        )
                        .returning(interest_profiles)
                    )
                )
                .mappings()
                .one_or_none()
            )
            if row is None:
                actual = await connection.scalar(
                    select(interest_profiles.c.version).where(interest_profiles.c.id == profile_id)
                )
                if actual is None:
                    raise LookupError(f"interest profile not found: {profile_id}")
                raise ConcurrentUpdateError(
                    f"interest profile {profile_id} changed from version {expected_version} "
                    f"to {actual}"
                )
        return _profile_from_row(row)

    async def find_materials_by_content_hashes(
        self,
        content_hashes: Collection[str],
    ) -> tuple[Material, ...]:
        requested = tuple(set(content_hashes))
        if not requested:
            return ()
        statement = (
            select(materials_table)
            .where(materials_table.c.content_hash.in_(requested))
            .order_by(materials_table.c.published_at, materials_table.c.id)
        )
        async with self._engine.connect() as connection:
            rows = (await connection.execute(statement)).mappings().all()
        return tuple(_material_from_row(row) for row in rows)

    async def find_materials_by_ids(self, material_ids: Collection[str]) -> tuple[Material, ...]:
        requested = tuple(set(material_ids))
        if not requested:
            return ()
        statement = (
            select(materials_table)
            .where(materials_table.c.id.in_(requested))
            .order_by(materials_table.c.published_at, materials_table.c.id)
        )
        async with self._engine.connect() as connection:
            rows = (await connection.execute(statement)).mappings().all()
        return tuple(_material_from_row(row) for row in rows)

    async def list_materials_since(self, published_since: datetime) -> tuple[Material, ...]:
        if published_since.tzinfo is None or published_since.utcoffset() is None:
            raise ValueError("published_since must be timezone-aware")
        statement = (
            select(materials_table)
            .where(materials_table.c.published_at >= published_since)
            .order_by(materials_table.c.published_at, materials_table.c.id)
        )
        async with self._engine.connect() as connection:
            rows = (await connection.execute(statement)).mappings().all()
        return tuple(_material_from_row(row) for row in rows)

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
        if (digest is None) != (channel_id is None):
            raise ValueError("digest and channel_id must be provided together")
        async with self._engine.begin() as connection:
            if materials:
                await connection.execute(
                    pg_insert(materials_table)
                    .values([_material_values(material) for material in materials])
                    .on_conflict_do_nothing()
                )
            if clusters:
                await connection.execute(
                    pg_insert(news_clusters)
                    .values(
                        [
                            {
                                "id": cluster.id,
                                "representative_id": cluster.representative_id,
                                "similarities": cluster.similarities,
                            }
                            for cluster in clusters
                        ]
                    )
                    .on_conflict_do_nothing()
                )
                await connection.execute(
                    pg_insert(cluster_materials)
                    .values(
                        [
                            {
                                "cluster_id": cluster.id,
                                "material_id": material_id,
                                "position": position,
                                "similarity": cluster.similarities[material_id],
                            }
                            for cluster in clusters
                            for position, material_id in enumerate(cluster.material_ids)
                        ]
                    )
                    .on_conflict_do_nothing()
                )
            if duplicate_links:
                await connection.execute(
                    pg_insert(duplicate_links_table)
                    .values([_duplicate_values(link) for link in duplicate_links])
                    .on_conflict_do_nothing()
                )
            if screening_results:
                await connection.execute(
                    pg_insert(screening_results_table)
                    .values(
                        [
                            _screening_values(
                                result,
                                profile_id=profile_id,
                                profile_version=profile_version,
                            )
                            for result in screening_results
                        ]
                    )
                    .on_conflict_do_nothing()
                )
            for source_id, cursor in (source_cursors or {}).items():
                result = await connection.execute(
                    update(sources)
                    .where(sources.c.id == source_id)
                    .values(cursor=cursor.model_dump(mode="json"), updated_at=func.now())
                )
                if not result.rowcount:
                    raise LookupError(f"source not found: {source_id}")
            if digest is not None and channel_id is not None:
                await _prepare_digest_on_connection(connection, digest, channel_id)

    async def prepare_digest(self, digest: Digest, *, channel_id: str) -> None:
        async with self._engine.begin() as connection:
            await _prepare_digest_on_connection(connection, digest, channel_id)

    async def list_pending_digests(self) -> tuple[Digest, ...]:
        async with self._engine.connect() as connection:
            digest_rows = (
                (
                    await connection.execute(
                        select(digests)
                        .where(digests.c.sent_at.is_(None))
                        .order_by(digests.c.created_at, digests.c.id)
                    )
                )
                .mappings()
                .all()
            )
            result: list[Digest] = []
            for digest_row in digest_rows:
                item_rows = (
                    (
                        await connection.execute(
                            select(digest_items)
                            .where(digest_items.c.digest_id == digest_row["id"])
                            .order_by(digest_items.c.position)
                        )
                    )
                    .mappings()
                    .all()
                )
                post_rows = (
                    await connection.execute(
                        select(digest_posts.c.text)
                        .where(digest_posts.c.digest_id == digest_row["id"])
                        .order_by(digest_posts.c.position)
                    )
                ).all()
                result.append(
                    Digest(
                        id=str(digest_row["id"]),
                        profile_id=str(digest_row["profile_id"]),
                        profile_version=int(digest_row["profile_version"]),
                        items=tuple(
                            DigestItem(
                                cluster_id=str(row["cluster_id"]),
                                summary=str(row["summary"]),
                                source_links=tuple(row["source_links"]),
                                model=str(row["model"]),
                                prompt_version=str(row["prompt_version"]),
                            )
                            for row in item_rows
                        ),
                        posts=tuple(str(row.text) for row in post_rows),
                        created_at=digest_row["created_at"],
                    )
                )
        return tuple(result)

    async def list_pending_digest_posts(self, digest_id: str) -> tuple[PendingDigestPost, ...]:
        statement = (
            select(digest_posts)
            .where(
                and_(
                    digest_posts.c.digest_id == digest_id,
                    digest_posts.c.telegram_message_id.is_(None),
                )
            )
            .order_by(digest_posts.c.position)
        )
        async with self._engine.connect() as connection:
            rows = (await connection.execute(statement)).mappings().all()
            exists = await connection.scalar(select(digests.c.id).where(digests.c.id == digest_id))
        if exists is None:
            raise LookupError(f"digest not found: {digest_id}")
        return tuple(
            PendingDigestPost(
                digest_id=digest_id,
                position=int(row["position"]),
                text=str(row["text"]),
            )
            for row in rows
        )

    async def mark_digest_post_sent(
        self,
        digest_id: str,
        position: int,
        *,
        telegram_message_id: int,
        sent_at: datetime,
    ) -> None:
        if telegram_message_id < 1:
            raise ValueError("telegram_message_id must be positive")
        if sent_at.tzinfo is None or sent_at.utcoffset() is None:
            raise ValueError("sent_at must be timezone-aware")
        sent_at = sent_at.astimezone(UTC)
        async with self._engine.begin() as connection:
            row = (
                await connection.execute(
                    update(digest_posts)
                    .where(
                        and_(
                            digest_posts.c.digest_id == digest_id,
                            digest_posts.c.position == position,
                            digest_posts.c.telegram_message_id.is_(None),
                        )
                    )
                    .values(telegram_message_id=telegram_message_id, sent_at=sent_at)
                    .returning(digest_posts.c.telegram_message_id)
                )
            ).one_or_none()
            if row is None:
                existing = (
                    await connection.execute(
                        select(digest_posts.c.telegram_message_id).where(
                            and_(
                                digest_posts.c.digest_id == digest_id,
                                digest_posts.c.position == position,
                            )
                        )
                    )
                ).scalar_one_or_none()
                if existing is None:
                    raise LookupError(f"digest post not found: {digest_id}/{position}")
                if int(existing) != telegram_message_id:
                    raise ConcurrentUpdateError(
                        f"digest post already sent with another id: {digest_id}/{position}"
                    )
            pending = await connection.scalar(
                select(func.count())
                .select_from(digest_posts)
                .where(
                    and_(
                        digest_posts.c.digest_id == digest_id,
                        digest_posts.c.telegram_message_id.is_(None),
                    )
                )
            )
            if pending == 0:
                await connection.execute(
                    update(digests).where(digests.c.id == digest_id).values(sent_at=sent_at)
                )

    async def get_delivery_receipt(self, digest_id: str) -> DeliveryReceipt | None:
        statement = (
            select(digest_posts.c.telegram_message_id, digest_posts.c.sent_at)
            .where(digest_posts.c.digest_id == digest_id)
            .order_by(digest_posts.c.position)
        )
        async with self._engine.connect() as connection:
            rows = (await connection.execute(statement)).all()
            exists = await connection.scalar(select(digests.c.id).where(digests.c.id == digest_id))
        if exists is None:
            raise LookupError(f"digest not found: {digest_id}")
        if not rows or any(row.telegram_message_id is None or row.sent_at is None for row in rows):
            return None
        return DeliveryReceipt(
            digest_id=digest_id,
            telegram_message_ids=tuple(int(row.telegram_message_id) for row in rows),
            sent_at=max(row.sent_at for row in rows if row.sent_at is not None),
        )

    async def list_recent_screenings(
        self, profile_id: str, *, limit: int = 10
    ) -> tuple[ScreeningReview, ...]:
        if limit < 1:
            raise ValueError("limit must be at least 1")
        # A cluster can have more than one screening_results row if it was re-screened
        # under a different model/prompt_version (see the composite PK in tables.py) --
        # DISTINCT ON keeps only the latest row per cluster before the outer LIMIT, so a
        # re-screened cluster never occupies two of the N slots shown to the operator.
        latest_per_cluster = (
            select(
                screening_results_table.c.cluster_id,
                screening_results_table.c.relevance_score,
                screening_results_table.c.noise_score,
                screening_results_table.c.uncertain,
                screening_results_table.c.reason,
                screening_results_table.c.model,
                screening_results_table.c.prompt_version,
                screening_results_table.c.created_at,
                materials_table.c.title,
                materials_table.c.original_url,
                materials_table.c.published_at,
            )
            .select_from(
                screening_results_table.join(
                    news_clusters,
                    news_clusters.c.id == screening_results_table.c.cluster_id,
                ).join(
                    materials_table,
                    materials_table.c.id == news_clusters.c.representative_id,
                )
            )
            .where(screening_results_table.c.profile_id == profile_id)
            .distinct(screening_results_table.c.cluster_id)
            .order_by(
                screening_results_table.c.cluster_id,
                screening_results_table.c.created_at.desc(),
            )
            .subquery()
        )
        statement = (
            select(latest_per_cluster).order_by(latest_per_cluster.c.created_at.desc()).limit(limit)
        )
        async with self._engine.connect() as connection:
            rows = (await connection.execute(statement)).mappings().all()
        return tuple(_screening_review_from_row(row) for row in rows)

    async def close(self) -> None:
        if self._owns_engine:
            await self._engine.dispose()


async def _prepare_digest_on_connection(
    connection: AsyncConnection,
    digest: Digest,
    channel_id: str,
) -> None:
    await connection.execute(
        pg_insert(digests)
        .values(
            id=digest.id,
            profile_id=digest.profile_id,
            profile_version=digest.profile_version,
            channel_id=channel_id,
            created_at=digest.created_at,
        )
        .on_conflict_do_nothing()
    )
    stored = (
        (await connection.execute(select(digests).where(digests.c.id == digest.id)))
        .mappings()
        .one()
    )
    if (
        str(stored["profile_id"]) != digest.profile_id
        or int(stored["profile_version"]) != digest.profile_version
        or str(stored["channel_id"]) != str(channel_id)
    ):
        raise ValueError(f"digest id collision: {digest.id}")
    await connection.execute(
        pg_insert(digest_items)
        .values(
            [
                {
                    "digest_id": digest.id,
                    "position": position,
                    "cluster_id": item.cluster_id,
                    "summary": item.summary,
                    "source_links": list(item.source_links),
                    "model": item.model,
                    "prompt_version": item.prompt_version,
                }
                for position, item in enumerate(digest.items)
            ]
        )
        .on_conflict_do_nothing()
    )
    await connection.execute(
        pg_insert(digest_posts)
        .values(
            [
                {"digest_id": digest.id, "position": position, "text": text}
                for position, text in enumerate(digest.posts)
            ]
        )
        .on_conflict_do_nothing()
    )


def _async_database_url(database_url: str) -> str:
    if database_url.startswith("postgres://"):
        return f"postgresql+psycopg://{database_url.removeprefix('postgres://')}"
    if database_url.startswith("postgresql://"):
        return f"postgresql+psycopg://{database_url.removeprefix('postgresql://')}"
    if database_url.startswith("postgresql+psycopg://"):
        return database_url
    raise ValueError("DATABASE_URL must use postgres:// or postgresql://")


def _source_values(source: SourceConfig, normalized_locator: str) -> dict[str, Any]:
    return {
        "id": source.id,
        "kind": source.kind,
        "locator": source.locator,
        "normalized_locator": normalized_locator,
        "collector": source.collector,
        "enabled": source.enabled,
        "settings": source.settings,
        "cursor": source.cursor.model_dump(mode="json") if source.cursor else None,
    }


def _source_from_row(row: RowMapping) -> SourceConfig:
    cursor = row["cursor"]
    return SourceConfig(
        id=str(row["id"]),
        kind=SourceKind(row["kind"]),
        locator=str(row["locator"]),
        collector=CollectorKind(row["collector"]),
        enabled=bool(row["enabled"]),
        settings=dict(row["settings"]),
        cursor=CollectionCursor.model_validate(cursor) if cursor is not None else None,
    )


def _profile_values(profile: InterestProfile) -> dict[str, Any]:
    values = profile.model_dump(mode="python")
    values["digest_send_times"] = [
        send_time.model_dump(mode="json") for send_time in profile.digest_send_times
    ]
    return values


def _profile_from_row(row: RowMapping) -> InterestProfile:
    return InterestProfile(
        id=str(row["id"]),
        name=str(row["name"]),
        description=str(row["description"]),
        enabled=bool(row["enabled"]),
        version=int(row["version"]),
        freshness_days=int(row["freshness_days"]),
        summary_length=SummaryLength(row["summary_length"]),
        tone_instructions=row["tone_instructions"],
        digest_send_times=tuple(
            DigestSendTime.model_validate(value) for value in row["digest_send_times"]
        ),
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        updated_by_telegram_user_id=row["updated_by_telegram_user_id"],
    )


def _material_values(material: Material) -> dict[str, Any]:
    return material.model_dump(mode="python")


def _material_from_row(row: RowMapping) -> Material:
    return Material(
        id=str(row["id"]),
        source_id=str(row["source_id"]),
        external_id=str(row["external_id"]),
        original_url=str(row["original_url"]),
        published_at=row["published_at"],
        fetched_at=row["fetched_at"],
        title=str(row["title"]),
        text=str(row["text"]),
        language=str(row["language"]) if row["language"] is not None else None,
        content_hash=str(row["content_hash"]),
        metadata=dict(row["metadata"]),
    )


def _duplicate_values(link: DuplicateLink) -> dict[str, Any]:
    return {
        "material_id": link.material_id,
        "duplicate_of_id": link.duplicate_of_id,
        "kind": DuplicateKind(link.kind),
        "similarity": link.similarity,
    }


def _screening_values(
    result: ScreeningResult,
    *,
    profile_id: str,
    profile_version: int,
) -> dict[str, Any]:
    return {
        "cluster_id": result.cluster_id,
        "profile_id": profile_id,
        "profile_version": profile_version,
        "model": result.model,
        "prompt_version": result.prompt_version,
        "relevance_score": result.relevance_score,
        "noise_score": result.noise_score,
        "uncertain": result.uncertain,
        "reason": result.reason,
    }


def _screening_review_from_row(row: RowMapping) -> ScreeningReview:
    return ScreeningReview(
        result=ScreeningResult(
            cluster_id=str(row["cluster_id"]),
            relevance_score=float(row["relevance_score"]),
            noise_score=float(row["noise_score"]),
            uncertain=bool(row["uncertain"]),
            reason=str(row["reason"]),
            model=str(row["model"]),
            prompt_version=str(row["prompt_version"]),
        ),
        material_title=str(row["title"]),
        material_url=str(row["original_url"]),
        material_published_at=row["published_at"],
    )
