"""One finite, idempotent pipeline-runner invocation."""

from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Protocol, cast
from zoneinfo import ZoneInfo

import httpx
from telegram import Bot
from telethon import TelegramClient
from telethon.sessions import StringSession

from ai_news_feed.dedup.semantic import DEFAULT_MODEL, SemanticClusterer
from ai_news_feed.delivery.telegram import TelegramBotAPI, TelegramDelivery
from ai_news_feed.digest import DigestComposer
from ai_news_feed.domain.models import (
    CollectionBatch,
    CollectionError,
    CollectorKind,
    DeliveryReceipt,
    Digest,
    DigestSendTime,
    InterestProfile,
)
from ai_news_feed.extraction import ContentExtractor, ExtractedItem
from ai_news_feed.llm.openai import OpenAIResponsesClient
from ai_news_feed.processing import AIProcessingResult, AIProcessor, ExtractedRawItem
from ai_news_feed.screening import ClusterScreener
from ai_news_feed.sources._shared import error_message
from ai_news_feed.sources.base import SourceConnector
from ai_news_feed.sources.rss import NativeRssConnector, RssBridgeConnector
from ai_news_feed.sources.site import UniversalSiteConnector
from ai_news_feed.sources.telegram import TelegramWebPreviewConnector, TelethonConnector
from ai_news_feed.storage.base import Repository
from ai_news_feed.storage.postgres import PostgresRepository
from ai_news_feed.summarization import ClusterSummarizer

logger = logging.getLogger(__name__)

_DIGEST_TIMEZONE = ZoneInfo("Europe/Moscow")
_PIPELINE_INTERVAL = timedelta(hours=6)


class Processor(Protocol):
    async def process(
        self,
        items: Sequence[ExtractedRawItem],
        *,
        interest_profile_id: str,
        interest_profile: InterestProfile | None = None,
        min_published_at: datetime | None = None,
    ) -> AIProcessingResult: ...


class Delivery(Protocol):
    async def send(self, digest: Digest) -> DeliveryReceipt: ...


@dataclass(frozen=True)
class PipelineRunReport:
    sources: int
    collected_items: int
    extraction_failures: int
    stored_materials: int
    clusters: int
    digest_posts: int
    resumed_digests: int


class PipelineRunner:
    def __init__(
        self,
        *,
        repository: Repository,
        connectors: Mapping[CollectorKind, SourceConnector],
        processor: Processor,
        delivery: Delivery,
        channel_id: str | int,
        interest_profile_id: str = "default",
        extractor: ContentExtractor | None = None,
        composer: DigestComposer | None = None,
        now: Callable[[], datetime] | None = None,
    ) -> None:
        self._repository = repository
        self._connectors = connectors
        self._processor = processor
        self._delivery = delivery
        self._channel_id = channel_id
        self._profile_id = interest_profile_id
        self._extractor = extractor or ContentExtractor()
        self._composer = composer or DigestComposer()
        self._now = now or (lambda: datetime.now(UTC))

    async def run(self, *, ignore_schedule: bool = False) -> PipelineRunReport:
        sources, profile = await self._repository.load_context(self._profile_id)
        should_deliver = ignore_schedule or digest_delivery_is_due(
            profile.digest_send_times,
            now=self._now(),
        )
        pending = await self._repository.list_pending_digests()
        delivered_posts = 0
        resumed_digests = 0
        if should_deliver:
            for pending_digest in pending:
                pending_posts = await self._repository.list_pending_digest_posts(pending_digest.id)
                logger.info(
                    "sending pending digest: digest_id=%s posts=%d",
                    pending_digest.id,
                    len(pending_posts),
                )
                await self._delivery.send(pending_digest)
                delivered_posts += len(pending_posts)
                resumed_digests += 1

        extracted: list[ExtractedRawItem] = []
        cursor_updates = {}
        collected_items = 0
        extraction_failures = 0
        for source in sources:
            try:
                connector = self._connectors[source.collector]
            except KeyError as exc:
                raise RuntimeError(
                    f"collector {source.collector.value} is not configured for source {source.id}"
                ) from exc
            try:
                batch = await connector.collect(source, source.cursor)
            except Exception as exc:
                logger.exception("source=%s collector crashed", source.id)
                batch = CollectionBatch(
                    errors=(
                        CollectionError(
                            source_id=source.id,
                            code="collector_crashed",
                            message=error_message(exc),
                            retryable=True,
                        ),
                    ),
                )
            collected_items += len(batch.raw_items)
            for error in batch.errors:
                logger.warning(
                    "source=%s code=%s retryable=%s: %s",
                    error.source_id,
                    error.code,
                    error.retryable,
                    error.message,
                )
            if batch.next_cursor is not None and not any(error.retryable for error in batch.errors):
                cursor_updates[source.id] = batch.next_cursor
            for raw_item in batch.raw_items:
                result = self._extractor.extract(raw_item)
                if isinstance(result, ExtractedItem):
                    extracted.append(ExtractedRawItem(raw_item, result))
                else:
                    extraction_failures += 1
                    logger.warning(
                        "source=%s external_id=%s extraction=%s: %s",
                        raw_item.source_id,
                        raw_item.external_id,
                        result.code,
                        result.message,
                    )

        logger.info(
            "collection completed: sources=%d collected_items=%d extracted_items=%d "
            "extraction_failures=%d",
            len(sources),
            collected_items,
            len(extracted),
            extraction_failures,
        )
        processing = await self._processor.process(
            extracted,
            interest_profile_id=profile.id,
            interest_profile=profile,
        )
        logger.info(
            "processing completed: materials=%d clusters=%d digest_items=%d",
            len(processing.materials),
            len(processing.cluster_batch.clusters),
            len(processing.digest_items),
        )
        digest = self._composer.compose(
            processing.digest_items,
            profile_id=profile.id,
            profile_version=profile.version,
        )
        await self._repository.save_processing_result(
            materials=processing.materials,
            clusters=processing.cluster_batch.clusters,
            duplicate_links=processing.cluster_batch.duplicate_links,
            screening_results=processing.screening_results,
            profile_id=profile.id,
            profile_version=profile.version,
            digest=digest,
            channel_id=str(self._channel_id) if digest is not None else None,
        )
        if digest is not None and should_deliver:
            logger.info(
                "sending digest: digest_id=%s posts=%d",
                digest.id,
                len(digest.posts),
            )
            await self._delivery.send(digest)
            delivered_posts += len(digest.posts)

        await self._repository.save_processing_result(
            materials=(),
            clusters=(),
            duplicate_links=(),
            screening_results=(),
            profile_id=profile.id,
            profile_version=profile.version,
            source_cursors=cursor_updates,
        )
        return PipelineRunReport(
            sources=len(sources),
            collected_items=collected_items,
            extraction_failures=extraction_failures,
            stored_materials=len(processing.materials),
            clusters=len(processing.cluster_batch.clusters),
            digest_posts=delivered_posts,
            resumed_digests=resumed_digests,
        )

    async def run_backfill(self, *, min_published_at: datetime) -> PipelineRunReport:
        """On-demand wide catch-up across all sources, for the bot's "collect for period" action.

        Unlike ``run()`` this does not resume previously pending digests -- that stays the
        scheduled run's job -- and it asks every connector for one fresh snapshot, with the
        source's own stored cursor cleared for that one call (see the comment at the
        ``model_copy`` call below for why clearing the cursor field, not just passing
        ``cursor=None``, is what actually disables the "newer than" filter). Each connector's
        own per-call item cap still applies, so this is a single wide fetch, not deep
        historical pagination: how far back it actually reaches depends on what the source
        itself exposes. Only items at or after ``min_published_at`` survive into the digest
        via ``Processor.process``; the source cursors are still advanced from what was
        actually collected, exactly as ``run()`` would, so a later scheduled run does not
        redundantly re-collect the same items.
        """
        sources, profile = await self._repository.load_context(self._profile_id)
        extracted: list[ExtractedRawItem] = []
        cursor_updates = {}
        collected_items = 0
        extraction_failures = 0
        for source in sources:
            try:
                connector = self._connectors[source.collector]
            except KeyError as exc:
                raise RuntimeError(
                    f"collector {source.collector.value} is not configured for source {source.id}"
                ) from exc
            # Passing cursor=None here is NOT enough to bypass the source's own stored
            # cursor: every connector resolves it via effective_cursor(config, explicit),
            # which falls back to config.cursor whenever the explicit argument is None (see
            # sources/_shared.py). A source config with its own cursor cleared is what
            # actually makes the connector treat this as a fresh, unfiltered snapshot.
            wide_source = source.model_copy(update={"cursor": None})
            try:
                batch = await connector.collect(wide_source, cursor=None)
            except Exception as exc:
                logger.exception("backfill source=%s collector crashed", source.id)
                batch = CollectionBatch(
                    errors=(
                        CollectionError(
                            source_id=source.id,
                            code="collector_crashed",
                            message=error_message(exc),
                            retryable=True,
                        ),
                    ),
                )
            collected_items += len(batch.raw_items)
            for error in batch.errors:
                logger.warning(
                    "backfill source=%s code=%s retryable=%s: %s",
                    error.source_id,
                    error.code,
                    error.retryable,
                    error.message,
                )
            if batch.next_cursor is not None and not any(error.retryable for error in batch.errors):
                cursor_updates[source.id] = batch.next_cursor
            for raw_item in batch.raw_items:
                result = self._extractor.extract(raw_item)
                if isinstance(result, ExtractedItem):
                    extracted.append(ExtractedRawItem(raw_item, result))
                else:
                    extraction_failures += 1
                    logger.warning(
                        "backfill source=%s external_id=%s extraction=%s: %s",
                        raw_item.source_id,
                        raw_item.external_id,
                        result.code,
                        result.message,
                    )

        logger.info(
            "backfill collection completed: sources=%d collected_items=%d extracted_items=%d "
            "extraction_failures=%d",
            len(sources),
            collected_items,
            len(extracted),
            extraction_failures,
        )
        processing = await self._processor.process(
            extracted,
            interest_profile_id=profile.id,
            interest_profile=profile,
            min_published_at=min_published_at,
        )
        logger.info(
            "backfill processing completed: materials=%d clusters=%d digest_items=%d",
            len(processing.materials),
            len(processing.cluster_batch.clusters),
            len(processing.digest_items),
        )
        digest = self._composer.compose(
            processing.digest_items,
            profile_id=profile.id,
            profile_version=profile.version,
        )
        await self._repository.save_processing_result(
            materials=processing.materials,
            clusters=processing.cluster_batch.clusters,
            duplicate_links=processing.cluster_batch.duplicate_links,
            screening_results=processing.screening_results,
            profile_id=profile.id,
            profile_version=profile.version,
            digest=digest,
            channel_id=str(self._channel_id) if digest is not None else None,
        )
        if digest is not None:
            logger.info(
                "sending backfill digest: digest_id=%s posts=%d",
                digest.id,
                len(digest.posts),
            )
            await self._delivery.send(digest)

        await self._repository.save_processing_result(
            materials=(),
            clusters=(),
            duplicate_links=(),
            screening_results=(),
            profile_id=profile.id,
            profile_version=profile.version,
            source_cursors=cursor_updates,
        )
        return PipelineRunReport(
            sources=len(sources),
            collected_items=collected_items,
            extraction_failures=extraction_failures,
            stored_materials=len(processing.materials),
            clusters=len(processing.cluster_batch.clusters),
            digest_posts=len(digest.posts) if digest is not None else 0,
            resumed_digests=0,
        )


@dataclass(frozen=True)
class PipelineSettings:
    database_url: str
    telegram_bot_token: str
    telegram_channel_id: str | int
    openai_api_key: str
    openai_screening_model: str
    openai_summary_model: str
    openai_base_url: str
    interest_profile_id: str
    semhash_model: str
    telegram_api_id: int | None
    telegram_api_hash: str | None
    telegram_session: str | None

    @classmethod
    def from_env(cls) -> PipelineSettings:
        channel = _required_env("TELEGRAM_CHANNEL_ID")
        channel_id: str | int = int(channel) if channel.lstrip("-").isdigit() else channel
        api_id = os.environ.get("TELEGRAM_API_ID", "").strip()
        return cls(
            database_url=_required_env("DATABASE_URL"),
            telegram_bot_token=_required_env("TELEGRAM_BOT_TOKEN"),
            telegram_channel_id=channel_id,
            openai_api_key=_required_env("OPENAI_API_KEY"),
            openai_screening_model=_required_env("OPENAI_SCREENING_MODEL"),
            openai_summary_model=_required_env("OPENAI_SUMMARY_MODEL"),
            openai_base_url=os.environ.get(
                "OPENAI_BASE_URL",
                "https://api.openai.com/v1",
            ).strip(),
            interest_profile_id=os.environ.get("INTEREST_PROFILE_ID", "default").strip(),
            semhash_model=os.environ.get("SEMHASH_MODEL", DEFAULT_MODEL).strip(),
            telegram_api_id=int(api_id) if api_id else None,
            telegram_api_hash=os.environ.get("TELEGRAM_API_HASH") or None,
            telegram_session=os.environ.get("TELEGRAM_SESSION") or None,
        )


async def run_from_env() -> PipelineRunReport:
    settings = PipelineSettings.from_env()
    repository = PostgresRepository(settings.database_url, pooled=False)
    telethon_client: TelegramClient | None = None
    try:
        async with httpx.AsyncClient(follow_redirects=True) as http_client:
            connectors: dict[CollectorKind, SourceConnector] = {
                CollectorKind.NATIVE_RSS: NativeRssConnector(http_client),
                CollectorKind.RSS_BRIDGE: RssBridgeConnector(http_client),
                CollectorKind.UNIVERSAL_SCRAPER: UniversalSiteConnector(http_client),
                CollectorKind.WEB_PREVIEW: TelegramWebPreviewConnector(http_client),
            }
            sources = await repository.list_sources()
            if any(source.collector is CollectorKind.TELETHON for source in sources):
                if (
                    settings.telegram_api_id is None
                    or not settings.telegram_api_hash
                    or not settings.telegram_session
                ):
                    raise RuntimeError(
                        "TELEGRAM_API_ID, TELEGRAM_API_HASH and TELEGRAM_SESSION are required "
                        "for Telethon sources"
                    )
                telethon_client = TelegramClient(
                    StringSession(settings.telegram_session),
                    settings.telegram_api_id,
                    settings.telegram_api_hash,
                )
                await telethon_client.connect()
                if not await telethon_client.is_user_authorized():
                    raise RuntimeError("TELEGRAM_SESSION is not authorized")
                connectors[CollectorKind.TELETHON] = TelethonConnector(telethon_client)

            screening_client = OpenAIResponsesClient(
                api_key=settings.openai_api_key,
                model=settings.openai_screening_model,
                base_url=settings.openai_base_url,
                max_output_tokens=1_000,
                client=http_client,
            )
            summary_client = OpenAIResponsesClient(
                api_key=settings.openai_api_key,
                model=settings.openai_summary_model,
                base_url=settings.openai_base_url,
                max_output_tokens=2_000,
                client=http_client,
            )
            processor = AIProcessor(
                repository=repository,
                semantic_clusterer=SemanticClusterer(model_name=settings.semhash_model),
                screener=ClusterScreener(screening_client),
                summarizer=ClusterSummarizer(summary_client),
            )
            async with Bot(settings.telegram_bot_token) as bot:
                delivery = TelegramDelivery(
                    bot=cast(TelegramBotAPI, bot),
                    repository=repository,
                    channel_id=settings.telegram_channel_id,
                )
                runner = PipelineRunner(
                    repository=repository,
                    connectors=connectors,
                    processor=processor,
                    delivery=delivery,
                    channel_id=settings.telegram_channel_id,
                    interest_profile_id=settings.interest_profile_id,
                )
                return await runner.run()
    finally:
        if telethon_client is not None:
            await telethon_client.disconnect()
        await repository.close()


def main() -> None:
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    # httpx includes the complete Telegram Bot API URL (and its token) in INFO logs.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    report = asyncio.run(run_from_env())
    logger.info("pipeline completed: %s", report)


def _required_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"{name} is required")
    return value


def digest_delivery_is_due(
    send_times: Sequence[DigestSendTime],
    *,
    now: datetime,
) -> bool:
    """Return whether this is the first six-hour run after a configured local slot."""
    if not send_times:
        return True
    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("now must be timezone-aware")

    local_now = now.astimezone(_DIGEST_TIMEZONE)
    previous_run = local_now - _PIPELINE_INTERVAL
    for send_time in send_times:
        days_since_slot = (
            0 if send_time.weekday is None else (local_now.weekday() - send_time.weekday) % 7
        )
        occurrence = local_now.replace(
            hour=send_time.hour,
            minute=0,
            second=0,
            microsecond=0,
        ) - timedelta(days=days_since_slot)
        cycle = timedelta(days=1 if send_time.weekday is None else 7)
        if occurrence > local_now:
            occurrence -= cycle
        if previous_run < occurrence <= local_now:
            return True
    return False
