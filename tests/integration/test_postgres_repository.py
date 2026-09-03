import os
from datetime import UTC, datetime

import pytest
import pytest_asyncio
from sqlalchemy import text

from ai_news_feed.digest import DigestComposer
from ai_news_feed.domain.models import (
    CollectorKind,
    DigestSendTime,
    DuplicateKind,
    DuplicateLink,
    InterestProfile,
    Material,
    NewsCluster,
    ScreeningResult,
    SourceConfig,
    SourceKind,
)
from ai_news_feed.screening import ScreeningThresholds
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
async def test_digest_send_times_round_trip_and_optimistic_update(
    repository: PostgresRepository,
) -> None:
    now = datetime(2026, 9, 3, tzinfo=UTC)
    created = await repository.create_interest_profile(
        InterestProfile(
            id="default",
            name="Основные интересы",
            description="ИИ-агенты",
            digest_send_times=(
                DigestSendTime(hour=9),
                DigestSendTime(weekday=1, hour=18),
            ),
            created_at=now,
            updated_at=now,
        )
    )
    assert created.digest_send_times == (
        DigestSendTime(hour=9),
        DigestSendTime(weekday=1, hour=18),
    )

    updated = await repository.update_digest_send_times(
        "default",
        digest_send_times=(DigestSendTime(weekday=4, hour=12),),
        expected_version=created.version,
        updated_by_telegram_user_id=42,
    )
    assert updated.digest_send_times == (DigestSendTime(weekday=4, hour=12),)
    assert updated.version == 2
    assert (await repository.get_interest_profile("default")).digest_send_times == (
        DigestSendTime(weekday=4, hour=12),
    )

    with pytest.raises(ConcurrentUpdateError):
        await repository.update_digest_send_times(
            "default",
            digest_send_times=(),
            expected_version=created.version,
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


@pytest.mark.asyncio
async def test_list_recent_screenings_dedupes_by_cluster_and_filters_by_profile(
    repository: PostgresRepository,
) -> None:
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
    other_profile = await repository.create_interest_profile(
        InterestProfile(
            id="other",
            name="Другой профиль",
            description="Не про ИИ",
            created_at=now,
            updated_at=now,
        )
    )
    accepted_material = _material("accepted", "1", now)
    rejected_material = _material("rejected", "2", now)
    other_profile_material = _material("other-profile", "3", now)
    accepted_cluster = NewsCluster(
        id="cluster-accepted",
        material_ids=(accepted_material.id,),
        representative_id=accepted_material.id,
        similarities={accepted_material.id: 1.0},
    )
    rejected_cluster = NewsCluster(
        id="cluster-rejected",
        material_ids=(rejected_material.id,),
        representative_id=rejected_material.id,
        similarities={rejected_material.id: 1.0},
    )
    other_profile_cluster = NewsCluster(
        id="cluster-other-profile",
        material_ids=(other_profile_material.id,),
        representative_id=other_profile_material.id,
        similarities={other_profile_material.id: 1.0},
    )
    accepted_screening = ScreeningResult(
        cluster_id=accepted_cluster.id,
        relevance_score=0.9,
        noise_score=0.1,
        reason="Точно по теме ИИ-агентов.",
        model="fake-screen",
        prompt_version="screen-v1",
    )
    rejected_screening = ScreeningResult(
        cluster_id=rejected_cluster.id,
        relevance_score=0.2,
        noise_score=0.1,
        reason="Не по теме.",
        model="fake-screen",
        prompt_version="screen-v1",
    )
    other_profile_screening = ScreeningResult(
        cluster_id=other_profile_cluster.id,
        relevance_score=0.9,
        noise_score=0.1,
        reason="Для другого профиля.",
        model="fake-screen",
        prompt_version="screen-v1",
    )

    await repository.save_processing_result(
        materials=(accepted_material, rejected_material, other_profile_material),
        clusters=(accepted_cluster, rejected_cluster, other_profile_cluster),
        duplicate_links=(),
        screening_results=(accepted_screening, rejected_screening),
        profile_id=profile.id,
        profile_version=profile.version,
    )
    await repository.save_processing_result(
        materials=(),
        clusters=(),
        duplicate_links=(),
        screening_results=(other_profile_screening,),
        profile_id=other_profile.id,
        profile_version=other_profile.version,
    )
    # Re-screen the same cluster under a different model: a second row under the same
    # composite PK's cluster_id (see screening_results in tables.py) -- must still
    # surface only once, not twice, in the deduped read.
    rescreened = rejected_screening.model_copy(
        update={"model": "fake-screen-v2", "reason": "Пересмотрено: всё ещё не по теме."}
    )
    await repository.save_processing_result(
        materials=(),
        clusters=(),
        duplicate_links=(),
        screening_results=(rescreened,),
        profile_id=profile.id,
        profile_version=profile.version,
    )

    reviews = await repository.list_recent_screenings(profile.id, limit=10)

    assert len(reviews) == 2
    assert {review.result.cluster_id for review in reviews} == {
        accepted_cluster.id,
        rejected_cluster.id,
    }
    thresholds = ScreeningThresholds()
    verdicts = {review.result.cluster_id: thresholds.accepts(review.result) for review in reviews}
    assert verdicts[accepted_cluster.id] is True
    assert verdicts[rejected_cluster.id] is False
    accepted_review = next(r for r in reviews if r.result.cluster_id == accepted_cluster.id)
    assert accepted_review.material_title == accepted_material.title
    assert accepted_review.material_url == accepted_material.original_url
    assert accepted_review.material_published_at == now

    limited = await repository.list_recent_screenings(profile.id, limit=1)
    assert len(limited) == 1


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
