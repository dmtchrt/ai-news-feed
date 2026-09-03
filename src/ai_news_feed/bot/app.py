"""python-telegram-bot long-polling adapter for BotWorkerHandlers."""

from __future__ import annotations

import logging
import os
from typing import Any, cast

import httpx
from telegram import (
    Bot,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    MessageOriginChannel,
    Update,
)
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)
from telethon import TelegramClient
from telethon.sessions import StringSession

from ai_news_feed.bot.handlers import BotWorkerHandlers, Keyboard
from ai_news_feed.dedup.semantic import SemanticClusterer
from ai_news_feed.delivery.telegram import TelegramBotAPI, TelegramDelivery
from ai_news_feed.domain.models import CollectorKind
from ai_news_feed.llm.openai import OpenAIResponsesClient
from ai_news_feed.orchestration.pipeline import PipelineRunner, PipelineSettings
from ai_news_feed.processing import AIProcessor
from ai_news_feed.screening import ClusterScreener
from ai_news_feed.sources.base import SourceConnector
from ai_news_feed.sources.rss import NativeRssConnector, RssBridgeConnector
from ai_news_feed.sources.site import UniversalSiteConnector
from ai_news_feed.sources.telegram import TelegramWebPreviewConnector, TelethonConnector
from ai_news_feed.storage.postgres import PostgresRepository
from ai_news_feed.summarization import ClusterSummarizer

logger = logging.getLogger(__name__)


class _BackfillResources:
    """Owns the long-lived clients the bot's in-chat backfill action needs.

    Mirrors ``orchestration.pipeline.run_from_env``'s wiring, but kept alive for the
    process lifetime (built once from ``post_init``, torn down from ``post_shutdown``)
    instead of scoped to one pipeline invocation. Reuses the running ``Application``'s
    own already-initialized ``bot`` for delivery rather than opening a second bot
    connection.

    Deliberately optional: if ``PipelineSettings.from_env()`` can't find the pipeline's
    OpenAI/Telegram-channel settings, ``start()`` returns ``None`` and the bot's core
    source/interest management keeps working without the backfill button. This is a
    conscious relaxation of "pipeline secrets stay in GitHub Actions, never on the VPS"
    (see deploy/README.md) made specifically so the button can answer synchronously in
    chat, as chosen over an async GitHub Actions dispatch or a DB-queued alternative --
    it stays opt-in by leaving these variables unset in deploy/.env.example.
    """

    def __init__(self) -> None:
        self._http_client: httpx.AsyncClient | None = None
        self._telethon_client: TelegramClient | None = None

    async def start(self, *, repository: PostgresRepository, bot: Bot) -> PipelineRunner | None:
        """Build the runner, or leave backfill disabled -- but never let bot startup crash.

        python-telegram-bot's post_init hook is not guarded against arbitrary exceptions
        (only KeyboardInterrupt/SystemExit pass through untouched); anything else raised
        here would otherwise take the whole bot process down with it, contradicting the
        promise that backfill is a pure opt-in add-on. So every failure path below --
        missing settings, a DB error listing sources, a Telethon connection error, an
        unexpected error building any client -- funnels into returning None here, with
        whatever was partially opened cleaned up before returning.
        """
        try:
            settings = PipelineSettings.from_env()
        except RuntimeError as exc:
            logger.warning("in-chat backfill disabled: %s", exc)
            return None
        try:
            runner = await self._build(settings, repository=repository, bot=bot)
        except Exception:
            logger.exception("in-chat backfill disabled: unexpected error during setup")
            runner = None
        if runner is None:
            await self.stop()
            self._http_client = None
            self._telethon_client = None
            return None
        logger.info("in-chat backfill enabled")
        return runner

    async def _build(
        self, settings: PipelineSettings, *, repository: PostgresRepository, bot: Bot
    ) -> PipelineRunner | None:
        # Assigned to self immediately (before any await that could fail) so start()'s
        # cleanup always has a reference to close, no matter where below this raises.
        http_client = httpx.AsyncClient(follow_redirects=True)
        self._http_client = http_client
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
                logger.warning(
                    "in-chat backfill disabled: TELEGRAM_API_ID/TELEGRAM_API_HASH/"
                    "TELEGRAM_SESSION are required because a Telethon source is configured"
                )
                return None
            telethon_client = TelegramClient(
                StringSession(settings.telegram_session),
                settings.telegram_api_id,
                settings.telegram_api_hash,
            )
            self._telethon_client = telethon_client
            await telethon_client.connect()
            if not await telethon_client.is_user_authorized():
                logger.warning("in-chat backfill disabled: TELEGRAM_SESSION is not authorized")
                return None
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
        delivery = TelegramDelivery(
            bot=cast(TelegramBotAPI, bot),
            repository=repository,
            channel_id=settings.telegram_channel_id,
        )
        return PipelineRunner(
            repository=repository,
            connectors=connectors,
            processor=processor,
            delivery=delivery,
            channel_id=settings.telegram_channel_id,
            interest_profile_id=settings.interest_profile_id,
        )

    async def stop(self) -> None:
        if self._telethon_client is not None:
            await self._telethon_client.disconnect()
        if self._http_client is not None:
            await self._http_client.aclose()


class PythonTelegramBotAPI:
    def __init__(self, bot: Bot) -> None:
        self._bot = bot

    async def send_message(
        self,
        chat_id: int,
        text: str,
        *,
        buttons: Keyboard = (),
    ) -> None:
        markup = None
        if buttons:
            markup = InlineKeyboardMarkup(
                [
                    [
                        InlineKeyboardButton(button.text, callback_data=button.callback_data)
                        for button in row
                    ]
                    for row in buttons
                ]
            )
        await self._bot.send_message(
            chat_id=chat_id,
            text=text,
            reply_markup=markup,
            disable_web_page_preview=True,
        )


def build_application(
    *,
    token: str,
    repository: PostgresRepository,
    owner_user_id: int,
    interest_profile_id: str = "default",
) -> Application[Any, Any, Any, Any, Any, Any]:
    resources = _BackfillResources()

    async def post_init(application: Application[Any, Any, Any, Any, Any, Any]) -> None:
        handlers.set_backfill_runner(
            await resources.start(repository=repository, bot=application.bot)
        )

    async def post_shutdown(
        _application: Application[Any, Any, Any, Any, Any, Any],
    ) -> None:
        await resources.stop()
        await repository.close()

    application = (
        Application.builder().token(token).post_init(post_init).post_shutdown(post_shutdown).build()
    )
    handlers = BotWorkerHandlers(
        repository=repository,
        api=PythonTelegramBotAPI(application.bot),
        owner_user_id=owner_user_id,
        interest_profile_id=interest_profile_id,
    )

    async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        del context
        if update.effective_chat is not None and update.effective_user is not None:
            await handlers.handle_start(
                chat_id=update.effective_chat.id,
                user_id=update.effective_user.id,
            )

    async def callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        del context
        query = update.callback_query
        if query is None or update.effective_chat is None or update.effective_user is None:
            return
        await query.answer()
        await handlers.handle_callback(
            chat_id=update.effective_chat.id,
            user_id=update.effective_user.id,
            data=query.data or "",
        )

    async def text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        del context
        message = update.effective_message
        if update.effective_chat is None or update.effective_user is None or message is None:
            return
        # A forwarded channel post lets "Добавить источник" add the channel directly,
        # without the user having to go find and paste its link or @handle themselves.
        forwarded_channel_username: str | None = None
        forward_missing_username = False
        origin = message.forward_origin
        if isinstance(origin, MessageOriginChannel):
            if origin.chat.username:
                forwarded_channel_username = origin.chat.username
            else:
                forward_missing_username = True
        is_channel_forward = forwarded_channel_username is not None or forward_missing_username
        if message.text is None and not is_channel_forward:
            return
        await handlers.handle_text(
            chat_id=update.effective_chat.id,
            user_id=update.effective_user.id,
            text=message.text or "",
            forwarded_channel_username=forwarded_channel_username,
            forward_missing_username=forward_missing_username,
        )

    application.add_handler(CommandHandler("start", start, filters=filters.ChatType.PRIVATE))
    application.add_handler(CallbackQueryHandler(callback))
    application.add_handler(
        MessageHandler(
            filters.ChatType.PRIVATE & (filters.TEXT | filters.FORWARDED) & ~filters.COMMAND,
            text,
        )
    )
    return application


def main() -> None:
    token = _required_env("TELEGRAM_BOT_TOKEN")
    database_url = _required_env("DATABASE_URL")
    owner_user_id = int(_required_env("TELEGRAM_OWNER_USER_ID"))
    repository = PostgresRepository(database_url, pooled=True)
    application = build_application(
        token=token,
        repository=repository,
        owner_user_id=owner_user_id,
        interest_profile_id=os.environ.get("INTEREST_PROFILE_ID", "default"),
    )
    application.run_polling(drop_pending_updates=False, allowed_updates=Update.ALL_TYPES)


def _required_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"{name} is required")
    return value
