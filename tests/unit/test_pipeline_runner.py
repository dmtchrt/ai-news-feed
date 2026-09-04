from dataclasses import dataclass
from datetime import UTC, datetime
from zoneinfo import ZoneInfo

import pytest

from ai_news_feed.delivery.telegram import TelegramDelivery
from ai_news_feed.domain.models import (
    ClusterBatch,
    CollectionBatch,
    CollectionCursor,
    CollectorKind,
    DigestItem,
    DigestSendTime,
    InterestProfile,
    Material,
    NewsCluster,
    RawItem,
    ScreeningResult,
    SourceConfig,
    SourceKind,
)
from ai_news_feed.orchestration.pipeline import PipelineRunner, digest_delivery_is_due
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


class _EmptyConnector:
    async def collect(
        self,
        config: SourceConfig,
        cursor: CollectionCursor | None = None,
    ) -> CollectionBatch:
        del config, cursor
        return CollectionBatch()


class _EmptyProcessor:
    async def process(
        self,
        items,
        *,
        interest_profile_id,
        interest_profile=None,
        min_published_at=None,
    ) -> AIProcessingResult:
        del interest_profile_id, interest_profile, min_published_at
        assert not items
        return AIProcessingResult(
            materials=(),
            exact_deduplication=ExactDeduplicationResult(),
            cluster_batch=ClusterBatch(),
            screening_results=(),
            digest_items=(),
        )


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


def _build_scheduled_runner(
    *,
    send_times: tuple[DigestSendTime, ...],
    run_at: datetime,
) -> tuple[PipelineRunner, InMemoryRepository, _Bot]:
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
        digest_send_times=send_times,
        created_at=run_at,
        updated_at=run_at,
    )
    material = Material(
        id="material",
        source_id=source.id,
        external_id="1",
        original_url="https://example.test/1",
        published_at=run_at,
        fetched_at=run_at,
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
    processing = AIProcessingResult(
        materials=(material,),
        exact_deduplication=ExactDeduplicationResult(unique_materials=(material,)),
        cluster_batch=ClusterBatch(clusters=(cluster,)),
        screening_results=(
            ScreeningResult(
                cluster_id=cluster.id,
                relevance_score=0.9,
                noise_score=0.1,
                reason="По теме.",
                model="screen",
                prompt_version="screen-v1",
            ),
        ),
        digest_items=(
            DigestItem(
                cluster_id=cluster.id,
                summary="Вышло обновление автономных агентов.",
                source_links=(material.original_url,),
                model="summary",
                prompt_version="summary-v1",
            ),
        ),
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
        now=lambda: run_at,
    )
    return runner, repository, bot


def test_digest_delivery_schedule_snaps_to_first_six_hour_run() -> None:
    moscow = ZoneInfo("Europe/Moscow")
    daily = (DigestSendTime(hour=9),)
    assert digest_delivery_is_due(
        daily,
        now=datetime(2026, 9, 1, 10, 17, tzinfo=moscow),
    )
    assert not digest_delivery_is_due(
        daily,
        now=datetime(2026, 9, 1, 16, 17, tzinfo=moscow),
    )

    weekly = (DigestSendTime(weekday=1, hour=9),)
    assert digest_delivery_is_due(
        weekly,
        now=datetime(2026, 9, 1, 10, 17, tzinfo=moscow),
    )
    assert not digest_delivery_is_due(
        weekly,
        now=datetime(2026, 9, 2, 10, 17, tzinfo=moscow),
    )
    assert digest_delivery_is_due(
        (DigestSendTime(weekday=6, hour=23),),
        now=datetime(2026, 9, 7, 4, 17, tzinfo=moscow),
    )
    assert digest_delivery_is_due((), now=datetime(2026, 9, 1, tzinfo=UTC))


@pytest.mark.asyncio
async def test_pipeline_schedule_gates_only_delivery_and_keeps_pending_digest() -> None:
    moscow = ZoneInfo("Europe/Moscow")
    run_at = datetime(2026, 9, 1, 16, 17, tzinfo=moscow)
    runner, repository, bot = _build_scheduled_runner(
        send_times=(DigestSendTime(hour=17),),
        run_at=run_at,
    )

    report = await runner.run()

    assert report.digest_posts == 0
    assert bot.posts == []
    assert len(await repository.list_pending_digests()) == 1
    assert len(await repository.find_materials_by_ids(("material",))) == 1
    stored_source = (await repository.list_sources())[0]
    assert stored_source.cursor is not None
    assert stored_source.cursor.external_id == "1"

    next_runner = PipelineRunner(
        repository=repository,
        connectors={CollectorKind.NATIVE_RSS: _EmptyConnector()},
        processor=_EmptyProcessor(),
        delivery=TelegramDelivery(bot=bot, repository=repository, channel_id=-100123),
        channel_id=-100123,
        now=lambda: datetime(2026, 9, 1, 22, 17, tzinfo=moscow),
    )
    next_report = await next_runner.run()

    assert next_report.resumed_digests == 1
    assert next_report.digest_posts == 1
    assert len(bot.posts) == 1
    assert await repository.list_pending_digests() == ()


@pytest.mark.asyncio
async def test_pipeline_ignore_schedule_sends_immediately() -> None:
    moscow = ZoneInfo("Europe/Moscow")
    run_at = datetime(2026, 9, 1, 16, 17, tzinfo=moscow)
    runner, _repository, bot = _build_scheduled_runner(
        send_times=(DigestSendTime(hour=17),),
        run_at=run_at,
    )

    report = await runner.run(ignore_schedule=True)

    assert report.digest_posts == 1
    assert len(bot.posts) == 1


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


class _CrashingConnector:
    async def collect(
        self,
        config: SourceConfig,
        cursor: CollectionCursor | None = None,
    ) -> CollectionBatch:
        del config, cursor
        raise TimeoutError


@pytest.mark.asyncio
async def test_pipeline_runner_survives_one_connector_crashing() -> None:
    """Regression: a source's connector raising instead of returning a CollectionBatch
    (e.g. site.py handing httpx an unfetchable URL) used to take down run() entirely --
    no digest at all, not even for the other, healthy sources."""
    now = datetime(2026, 8, 31, 9, tzinfo=UTC)
    good_source = SourceConfig(
        id="good",
        kind=SourceKind.WEBSITE,
        locator="https://example.test/feed.xml",
        collector=CollectorKind.NATIVE_RSS,
    )
    broken_source = SourceConfig(
        id="broken",
        kind=SourceKind.WEBSITE,
        locator="https://a-ai.ru/",
        collector=CollectorKind.UNIVERSAL_SCRAPER,
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
        source_id=good_source.id,
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
    repository = InMemoryRepository(
        sources=(good_source, broken_source),
        interest_profiles=(profile,),
    )
    bot = _Bot()
    delivery = TelegramDelivery(bot=bot, repository=repository, channel_id=-100123)
    runner = PipelineRunner(
        repository=repository,
        connectors={
            CollectorKind.NATIVE_RSS: _Connector(),
            CollectorKind.UNIVERSAL_SCRAPER: _CrashingConnector(),
        },
        processor=_Processor(processing),
        delivery=delivery,
        channel_id=-100123,
    )

    report = await runner.run()

    assert report.sources == 2
    assert report.collected_items == 1
    assert report.digest_posts == 1
    assert len(bot.posts) == 1
    sources_by_id = {source.id: source for source in await repository.list_sources()}
    assert sources_by_id["good"].cursor is not None
    assert sources_by_id["broken"].cursor is None


@pytest.mark.asyncio
async def test_pipeline_runner_backfill_survives_connector_crashing() -> None:
    """Same guard as run(): the on-demand catch-up must not let one connector's
    exception kill the whole backfill either."""
    now = datetime(2026, 8, 31, 9, tzinfo=UTC)
    source = SourceConfig(
        id="broken",
        kind=SourceKind.WEBSITE,
        locator="https://a-ai.ru/",
        collector=CollectorKind.UNIVERSAL_SCRAPER,
    )
    profile = InterestProfile(
        id="default",
        name="Основные интересы",
        description="ИИ-агенты",
        created_at=now,
        updated_at=now,
    )
    repository = InMemoryRepository(sources=(source,), interest_profiles=(profile,))
    bot = _Bot()
    delivery = TelegramDelivery(bot=bot, repository=repository, channel_id=-100123)
    runner = PipelineRunner(
        repository=repository,
        connectors={CollectorKind.UNIVERSAL_SCRAPER: _CrashingConnector()},
        processor=_EmptyProcessor(),
        delivery=delivery,
        channel_id=-100123,
    )

    report = await runner.run_backfill(min_published_at=datetime(2020, 1, 1, tzinfo=UTC))

    assert report.collected_items == 0
    assert report.digest_posts == 0
    assert bot.posts == []
    stored_source = (await repository.list_sources())[0]
    assert stored_source.cursor is None
