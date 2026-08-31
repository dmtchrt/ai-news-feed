import os
from datetime import UTC, datetime

import pytest
import pytest_asyncio
from sqlalchemy import text

from ai_news_feed.digest import DigestComposer
from ai_news_feed.domain.models import (
    CollectorKind,
    DuplicateKind,
    DuplicateLink,
    InterestProfile,
    Material,
    NewsCluster,
    ScreeningResult,
    SourceConfig,
    SourceKind,
)
from ai_news_feed.storage.base import ConcurrentUpdateError, DuplicateSourceError
from ai_news_feed.storage.postgres import PostgresRepository

pytestmark = pytest.mark.skipif(
    not os.environ.get("TEST_DATABASE_URL"),
    reason="TEST_DATABASE_URL is required for PostgreSQL integration tests",
)


@pytest_asyncio.fixture
async def repository() -> PostgresRepository:
    value = os.environ.get("TEST_DATABASE_URL")
    assert value is not None
    repo = PostgresRepository(value, pooled=False)
    async with repo.engine.begin() as connection:
        await connection.execute(
            text(
                "TRUNCATE digest_posts, digest_items, digests, screening_results, "
                "duplicate_links, cluster_materials, news_clusters, materials, "
                "interest_profiles, sources CASCADE"
            )
        )
    try:
        yield repo
    finally:
        await repo.close()


@pytest.mark.asyncio
async def test_source_identity_soft_delete_restore_and_profile_concurrency(
    repository: PostgresRepository,
) -> None:
    source = SourceConfig(
        id="source-one",
        kind=SourceKind.TELEGRAM,
        locator="@AiLev_Blog",
        collector=CollectorKind.WEB_PREVIEW,
    )
    added = await repository.add_source(source)
    assert added.locator == "@AiLev_Blog"

    with pytest.raises(DuplicateSourceError):
        await repository.add_source(
            source.model_copy(update={"id": "another", "locator": "https://t.me/ailev_blog"})
        )

    assert await repository.delete_source(source.id)
    restored = await repository.add_source(
        source.model_copy(update={"id": "replacement", "locator": "@ailev_blog"})
    )
    assert restored.id == source.id
    assert restored.enabled

    now = datetime(2026, 8, 31, tzinfo=UTC)
    await repository.create_interest_profile(
        InterestProfile(
            id="default",
            name="Основные интересы",
            description="ИИ-агенты",
            created_at=now,
            updated_at=now,
        )
    )
    updated = await repository.update_interest_profile(
        "default",
        description="ИИ-агенты и рынок ИИ",
        expected_version=1,
        updated_by_telegram_user_id=42,
    )
    assert updated.version == 2
    with pytest.raises(ConcurrentUpdateError):
        await repository.update_interest_profile(
            "default",
            description="устаревшая запись",
            expected_version=1,
            updated_by_telegram_user_id=42,
        )


@pytest.mark.asyncio
async def test_processing_storage_and_digest_checkpoints(repository: PostgresRepository) -> None:
    now = datetime(2026, 8, 31, 8, tzinfo=UTC)
    await repository.add_source(
        SourceConfig(
            id="source",
            kind=SourceKind.WEBSITE,
            locator="https://example.test/feed.xml",
            collector=CollectorKind.NATIVE_RSS,
        )
    )
    profile = await repository.create_interest_profile(
        InterestProfile(
            id="default",
            name="Основные интересы",
            description="ИИ-агенты",
            created_at=now,
            updated_at=now,
        )
    )
    first = _material("first", "1", now)
    second = _material("second", "2", now)
    cluster = NewsCluster(
        id="cluster",
        material_ids=(first.id, second.id),
        representative_id=first.id,
        similarities={first.id: 1.0, second.id: 0.95},
    )
    link = DuplicateLink(
        material_id=second.id,
        duplicate_of_id=first.id,
        kind=DuplicateKind.SEMANTIC,
        similarity=0.95,
    )
    screening = ScreeningResult(
        cluster_id=cluster.id,
        relevance_score=0.9,
        noise_score=0.1,
        reason="По теме.",
        model="fake-screen",
        prompt_version="screen-v1",
    )

    await repository.save_processing_result(
        materials=(first, second),
        clusters=(cluster,),
        duplicate_links=(link,),
        screening_results=(screening,),
        profile_id=profile.id,
        profile_version=profile.version,
    )
    assert await repository.find_materials_by_content_hashes([first.content_hash]) == (first,)
    assert await repository.list_materials_since(now) == (first, second)

    digest = DigestComposer().compose(
        [
            _digest_item(
                cluster_id=cluster.id,
                links=(first.original_url, second.original_url),
            )
        ],
        profile_id=profile.id,
        profile_version=profile.version,
        created_at=now,
    )
    assert digest is not None
    await repository.prepare_digest(digest, channel_id="-100123")
    pending = await repository.list_pending_digest_posts(digest.id)
    assert len(pending) == 1
    await repository.mark_digest_post_sent(
        digest.id,
        pending[0].position,
        telegram_message_id=777,
        sent_at=now,
    )
    receipt = await repository.get_delivery_receipt(digest.id)
    assert receipt is not None
    assert receipt.telegram_message_ids == (777,)


def _material(material_id: str, external_id: str, timestamp: datetime) -> Material:
    return Material(
        id=material_id,
        source_id="source",
        external_id=external_id,
        original_url=f"https://example.test/{external_id}",
        published_at=timestamp,
        fetched_at=timestamp,
        title=f"Новость {external_id}",
        text=f"Текст новости {external_id}",
        content_hash=(external_id * 64)[:64],
    )


def _digest_item(*, cluster_id: str, links: tuple[str, ...]):
    from ai_news_feed.domain.models import DigestItem

    return DigestItem(
        cluster_id=cluster_id,
        summary="Краткое содержание.",
        source_links=links,
        model="fake-summary",
        prompt_version="summary-v1",
    )
