from dataclasses import dataclass
from datetime import UTC, datetime

import pytest

from ai_news_feed.delivery.telegram import TelegramDelivery
from ai_news_feed.domain.models import (
    ClusterBatch,
    CollectionBatch,
    CollectionCursor,
    CollectorKind,
    DigestItem,
    InterestProfile,
    Material,
    NewsCluster,
    RawItem,
    ScreeningResult,
    SourceConfig,
    SourceKind,
)
from ai_news_feed.orchestration.pipeline import PipelineRunner
from ai_news_feed.processing import AIProcessingResult, ExactDeduplicationResult
from ai_news_feed.storage.memory import InMemoryRepository


class _Connector:
    async def collect(
        self,
        config: SourceConfig,
        cursor: CollectionCursor | None = None,
    ) -> CollectionBatch:
        assert cursor is None
        # Catches a real regression: passing cursor=None alone does NOT disable the
        # "newer than" filter if config.cursor is still set (effective_cursor() falls back
        # to it). run_backfill must pass a config whose own cursor is cleared too.
        assert config.cursor is None
        return CollectionBatch(
            raw_items=(
                RawItem(
                    source_id=config.id,
                    external_id="1",
                    original_url="https://example.test/1",
                    published_at=datetime(2026, 8, 31, 8, tzinfo=UTC),
                    raw_text="Подробный текст новости про автономных агентов.",
                ),
            ),
            next_cursor=CollectionCursor(
                published_at=datetime(2026, 8, 31, 8, tzinfo=UTC),
                external_id="1",
            ),
        )


class _Processor:
    def __init__(self, result: AIProcessingResult) -> None:
        self.result = result
        self.min_published_at_calls: list[datetime | None] = []

    async def process(
        self,
        items,
        *,
        interest_profile_id,
        interest_profile=None,
        min_published_at=None,
    ):
        assert len(items) == 1
        assert interest_profile_id == "default"
        assert interest_profile is not None and interest_profile.version == 1
        self.min_published_at_calls.append(min_published_at)
        return self.result


@dataclass(frozen=True)
class _Message:
    message_id: int


class _Bot:
    def __init__(self) -> None:
        self.posts: list[str] = []

    async def send_message(self, *, chat_id, text, parse_mode, disable_web_page_preview):
        assert chat_id == -100123
        assert parse_mode == "HTML"
        assert disable_web_page_preview
        self.posts.append(text)
        return _Message(len(self.posts))


@pytest.mark.asyncio
async def test_pipeline_runner_persists_delivery_plan_before_advancing_cursor() -> None:
    now = datetime(2026, 8, 31, 9, tzinfo=UTC)
    source = SourceConfig(
        id="source",
        kind=SourceKind.WEBSITE,
        locator="https://example.test/feed.xml",
        collector=CollectorKind.NATIVE_RSS,
    )
    profile = InterestProfile(
        id="default",
        name="Основные интересы",
        description="ИИ-агенты",
        created_at=now,
        updated_at=now,
    )
    material = Material(
        id="material",
        source_id=source.id,
        external_id="1",
        original_url="https://example.test/1",
        published_at=now,
        fetched_at=now,
        title="Автономные агенты",
        text="Подробный текст новости про автономных агентов.",
        content_hash="a" * 64,
    )
    cluster = NewsCluster(
        id="cluster",
        material_ids=(material.id,),
        representative_id=material.id,
        similarities={material.id: 1.0},
    )
    screening = ScreeningResult(
        cluster_id=cluster.id,
        relevance_score=0.9,
        noise_score=0.1,
        reason="По теме.",
        model="screen",
        prompt_version="screen-v1",
    )
    item = DigestItem(
        cluster_id=cluster.id,
        summary="Вышло обновление автономных агентов.",
        source_links=(material.original_url,),
        model="summary",
        prompt_version="summary-v1",
    )
    processing = AIProcessingResult(
        materials=(material,),
        exact_deduplication=ExactDeduplicationResult(unique_materials=(material,)),
        cluster_batch=ClusterBatch(clusters=(cluster,)),
        screening_results=(screening,),
        digest_items=(item,),
    )
    repository = InMemoryRepository(sources=(source,), interest_profiles=(profile,))
    bot = _Bot()
    delivery = TelegramDelivery(bot=bot, repository=repository, channel_id=-100123)
    runner = PipelineRunner(
        repository=repository,
        connectors={CollectorKind.NATIVE_RSS: _Connector()},
        processor=_Processor(processing),
        delivery=delivery,
        channel_id=-100123,
    )

    report = await runner.run()

    assert report.digest_posts == 1
    assert len(bot.posts) == 1
    stored_source = (await repository.list_sources())[0]
    assert stored_source.cursor is not None
    assert stored_source.cursor.external_id == "1"


@pytest.mark.asyncio
async def test_pipeline_runner_backfill_ignores_cursor_and_forwards_explicit_cutoff() -> None:
    now = datetime(2026, 8, 31, 9, tzinfo=UTC)
    # A source that has already been collected before (has its own stored cursor) is the
    # realistic case: this is what an owner's "Собрать за период" tap actually hits, and
    # it's exactly the case where a naive cursor=None would silently do nothing extra.
    source = SourceConfig(
        id="source",
        kind=SourceKind.WEBSITE,
        locator="https://example.test/feed.xml",
        collector=CollectorKind.NATIVE_RSS,
        cursor=CollectionCursor(
            published_at=datetime(2026, 8, 30, 8, tzinfo=UTC),
            external_id="0",
        ),
    )
    profile = InterestProfile(
        id="default",
        name="Основные интересы",
        description="ИИ-агенты",
        created_at=now,
        updated_at=now,
    )
    material = Material(
        id="material",
        source_id=source.id,
        external_id="1",
        original_url="https://example.test/1",
        published_at=now,
        fetched_at=now,
        title="Автономные агенты",
        text="Подробный текст новости про автономных агентов.",
        content_hash="a" * 64,
    )
    cluster = NewsCluster(
        id="cluster",
        material_ids=(material.id,),
        representative_id=material.id,
        similarities={material.id: 1.0},
    )
    screening = ScreeningResult(
        cluster_id=cluster.id,
        relevance_score=0.9,
        noise_score=0.1,
        reason="По теме.",
        model="screen",
        prompt_version="screen-v1",
    )
    item = DigestItem(
        cluster_id=cluster.id,
        summary="Вышло обновление автономных агентов.",
        source_links=(material.original_url,),
        model="summary",
        prompt_version="summary-v1",
    )
    processing = AIProcessingResult(
        materials=(material,),
        exact_deduplication=ExactDeduplicationResult(unique_materials=(material,)),
        cluster_batch=ClusterBatch(clusters=(cluster,)),
        screening_results=(screening,),
        digest_items=(item,),
    )
    repository = InMemoryRepository(sources=(source,), interest_profiles=(profile,))
    bot = _Bot()
    delivery = TelegramDelivery(bot=bot, repository=repository, channel_id=-100123)
    processor = _Processor(processing)
    runner = PipelineRunner(
        repository=repository,
        connectors={CollectorKind.NATIVE_RSS: _Connector()},
        processor=processor,
        delivery=delivery,
        channel_id=-100123,
    )

    cutoff = datetime(2020, 1, 1, tzinfo=UTC)
    report = await runner.run_backfill(min_published_at=cutoff)

    # _Connector.collect asserts cursor is None regardless of the source's own cursor --
    # run_backfill always requests a fresh wide snapshot, never the incremental one.
    assert report.digest_posts == 1
    assert report.resumed_digests == 0
    assert len(bot.posts) == 1
    assert processor.min_published_at_calls == [cutoff]
    # The cursor still advances from what was collected, same as a scheduled run, so the
    # next scheduled run does not redundantly re-collect the same items.
    stored_source = (await repository.list_sources())[0]
    assert stored_source.cursor is not None
    assert stored_source.cursor.external_id == "1"
