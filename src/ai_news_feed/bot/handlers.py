"""Framework-independent bot command handlers, testable with a fake Bot API."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import Protocol
from uuid import uuid4

from ai_news_feed.domain.models import DigestSendTime, InterestProfile, SourceConfig, SummaryLength
from ai_news_feed.screening import ScreeningThresholds
from ai_news_feed.sources.locator import parse_source_locator
from ai_news_feed.storage.base import (
    ConcurrentUpdateError,
    DuplicateSourceError,
    Repository,
    ScreeningReview,
)

ADD_SOURCE = "menu:add-source"
LIST_SOURCES = "menu:list-sources"
DELETE_SOURCE = "menu:delete-source"
VIEW_INTERESTS = "interests:view"
EDIT_INTERESTS = "interests:edit"
DELETE_SOURCE_PREFIX = "source:delete:"
DIGEST_SETTINGS = "menu:digest-settings"
EDIT_FRESHNESS = "digest:edit-freshness"
EDIT_LENGTH = "digest:edit-length"
SET_LENGTH_PREFIX = "digest:set-length:"
EDIT_TONE = "digest:edit-tone"
EDIT_SEND_TIMES = "digest:edit-send-times"
BACKFILL_MENU = "backfill:menu"
BACKFILL_PREFIX = "backfill:run:"
RUN_NOW = "menu:run-now"
CHECK_SCREENING = "menu:check-screening"
SETTINGS_MENU = "menu:settings"
ABOUT = "menu:about"

logger = logging.getLogger(__name__)

_LENGTH_LABELS: dict[SummaryLength, str] = {
    SummaryLength.BRIEF: "Кратко",
    SummaryLength.NORMAL: "Нормально",
    SummaryLength.DETAILED: "Подробно",
}
_TONE_RESET = "-"
_SEND_TIMES_RESET = "-"
_NO_PROFILE_YET = "Сначала задайте интересы — без них не с чем связывать настройки дайджеста."
_ABOUT_TEXT = (
    "🤖 AI News Feed\n\n"
    "Личный бот-дайджест: собирает новости из твоих источников (сайты и "
    "Telegram-каналы), убирает дубли, отсеивает нерелевантное через LLM-фильтр и "
    "присылает саммари с рабочими ссылками в канал по настроенному расписанию.\n\n"
    "Источники, интересы и настройки дайджеста — в «Настройки»."
)
_BACKFILL_NOT_CONFIGURED = (
    "Сбор за период недоступен: бот запущен без настроек пайплайна "
    "(OpenAI/канал/Telethon). Обратитесь к администратору бота."
)
_RUN_NOW_NOT_CONFIGURED = (
    "Сбор прямо сейчас недоступен: бот запущен без настроек пайплайна "
    "(OpenAI/канал/Telethon). Обратитесь к администратору бота."
)
_WEEKDAY_NAMES = (
    "понедельник",
    "вторник",
    "среда",
    "четверг",
    "пятница",
    "суббота",
    "воскресенье",
)
_WEEKDAY_NUMBERS = {name: weekday for weekday, name in enumerate(_WEEKDAY_NAMES)}
_EPOCH_FLOOR = datetime(2000, 1, 1, tzinfo=UTC)
_BACKFILL_PERIODS: dict[str, tuple[str, int | None]] = {
    "week": ("неделю", 7),
    "month": ("месяц", 30),
    "year": ("год", 365),
    "all": ("всё время", None),
}
_SCREENING_REVIEW_LIMIT = 10
_SCREENING_REVIEW_REASON_CHARS = 160
_SCREENING_REVIEW_TITLE_CHARS = 200
_SCREENING_REVIEW_MAX_MESSAGE_CHARS = 3800


@dataclass(frozen=True)
class Button:
    text: str
    callback_data: str


Keyboard = tuple[tuple[Button, ...], ...]


class BotAPI(Protocol):
    async def send_message(
        self,
        chat_id: int,
        text: str,
        *,
        buttons: Keyboard = (),
    ) -> None: ...


class BackfillReport(Protocol):
    """Structurally matches orchestration.pipeline.PipelineRunReport, without importing it
    (that module pulls in OpenAI/Telethon/python-telegram-bot at module scope; handlers.py
    stays limited to storage + domain models so its own tests stay dependency-light).

    Declared as read-only properties, not plain attributes: PipelineRunReport is a frozen
    dataclass, and mypy treats a Protocol's plain `name: int` attribute as read-write, which
    a frozen dataclass's field can never satisfy (assigning to it raises at runtime). A
    read-only property only requires the attribute to be gettable, which frozen dataclass
    fields already are -- and handlers.py only ever reads these fields, never assigns them.
    """

    @property
    def collected_items(self) -> int: ...
    @property
    def extraction_failures(self) -> int: ...
    @property
    def stored_materials(self) -> int: ...
    @property
    def clusters(self) -> int: ...
    @property
    def digest_posts(self) -> int: ...


class PipelineReport(BackfillReport, Protocol):
    @property
    def sources(self) -> int: ...


class BackfillRunner(Protocol):
    async def run_backfill(self, *, min_published_at: datetime) -> BackfillReport: ...


class RunNowRunner(Protocol):
    async def run(self, *, ignore_schedule: bool = False) -> PipelineReport: ...


class PendingAction(StrEnum):
    ADD_SOURCE = "add_source"
    EDIT_INTERESTS = "edit_interests"
    EDIT_FRESHNESS = "edit_freshness"
    EDIT_TONE = "edit_tone"
    EDIT_SEND_TIMES = "edit_send_times"


@dataclass(frozen=True)
class _PendingState:
    action: PendingAction
    expected_profile_version: int | None = None


class BotWorkerHandlers:
    def __init__(
        self,
        *,
        repository: Repository,
        api: BotAPI,
        owner_user_id: int,
        interest_profile_id: str = "default",
        backfill: BackfillRunner | None = None,
        run_now: RunNowRunner | None = None,
    ) -> None:
        if owner_user_id < 1:
            raise ValueError("owner_user_id must be positive")
        self._repository = repository
        self._api = api
        self._owner_user_id = owner_user_id
        self._profile_id = interest_profile_id
        self._backfill = backfill
        self._run_now_runner = run_now
        self._pipeline_running = False
        self._pending: dict[tuple[int, int], _PendingState] = {}

    async def handle_start(self, *, chat_id: int, user_id: int) -> None:
        if not await self._authorize(chat_id, user_id):
            return
        await self._api.send_message(
            chat_id,
            "Управление AI News Feed",
            buttons=_main_menu(),
        )

    async def handle_callback(self, *, chat_id: int, user_id: int, data: str) -> None:
        if not await self._authorize(chat_id, user_id):
            return
        key = (chat_id, user_id)
        if data == ADD_SOURCE:
            self._pending[key] = _PendingState(PendingAction.ADD_SOURCE)
            await self._api.send_message(
                chat_id,
                "Пришлите URL сайта/RSS, Telegram-канал в формате @handle, "
                "либо перешлите сюда сообщение из канала.",
            )
        elif data == LIST_SOURCES:
            await self._send_sources(chat_id)
        elif data == DELETE_SOURCE:
            await self._send_delete_menu(chat_id)
        elif data.startswith(DELETE_SOURCE_PREFIX):
            source_id = data.removeprefix(DELETE_SOURCE_PREFIX)
            deleted = await self._repository.delete_source(source_id)
            text = "Источник удалён." if deleted else "Источник уже удалён или не найден."
            await self._api.send_message(chat_id, text, buttons=_main_menu())
        elif data == VIEW_INTERESTS:
            await self._send_interests(chat_id)
        elif data == EDIT_INTERESTS:
            expected_version: int | None
            try:
                profile = await self._repository.get_interest_profile(self._profile_id)
                expected_version = profile.version
            except LookupError:
                expected_version = None
            self._pending[key] = _PendingState(
                PendingAction.EDIT_INTERESTS,
                expected_profile_version=expected_version,
            )
            await self._api.send_message(
                chat_id,
                "Пришлите новый текст интересов одним сообщением.",
            )
        elif data == DIGEST_SETTINGS:
            await self._send_digest_settings(chat_id)
        elif data == EDIT_FRESHNESS:
            freshness_version = await self._current_profile_version()
            if freshness_version is None:
                await self._api.send_message(chat_id, _NO_PROFILE_YET, buttons=_main_menu())
            else:
                self._pending[key] = _PendingState(
                    PendingAction.EDIT_FRESHNESS,
                    expected_profile_version=freshness_version,
                )
                await self._api.send_message(
                    chat_id,
                    "Пришлите число дней (1-365): не показывать в дайджесте новости старше этого.",
                )
        elif data == EDIT_LENGTH:
            await self._api.send_message(
                chat_id,
                "Выберите длину саммари:",
                buttons=tuple(
                    (Button(label, f"{SET_LENGTH_PREFIX}{value.value}"),)
                    for value, label in _LENGTH_LABELS.items()
                ),
            )
        elif data.startswith(SET_LENGTH_PREFIX):
            await self._set_length(chat_id, user_id, data.removeprefix(SET_LENGTH_PREFIX))
        elif data == EDIT_TONE:
            tone_version = await self._current_profile_version()
            if tone_version is None:
                await self._api.send_message(chat_id, _NO_PROFILE_YET, buttons=_main_menu())
            else:
                self._pending[key] = _PendingState(
                    PendingAction.EDIT_TONE,
                    expected_profile_version=tone_version,
                )
                await self._api.send_message(
                    chat_id,
                    "Пришлите описание стиля/тона (промт или примеры текста) одним "
                    f'сообщением, или "{_TONE_RESET}", чтобы вернуть стиль по умолчанию.',
                )
        elif data == EDIT_SEND_TIMES:
            send_times_version = await self._current_profile_version()
            if send_times_version is None:
                await self._api.send_message(chat_id, _NO_PROFILE_YET, buttons=_main_menu())
            else:
                self._pending[key] = _PendingState(
                    PendingAction.EDIT_SEND_TIMES,
                    expected_profile_version=send_times_version,
                )
                await self._api.send_message(
                    chat_id,
                    "Пришлите одно или несколько времён через запятую или с новой строки. "
                    "Например: «09:00» — каждый день, «вторник 09:00» — раз в неделю. "
                    f"Или «{_SEND_TIMES_RESET}», чтобы отправлять при каждом запуске.",
                )
        elif data == BACKFILL_MENU:
            await self._send_backfill_menu(chat_id)
        elif data.startswith(BACKFILL_PREFIX):
            await self._run_backfill(chat_id, data.removeprefix(BACKFILL_PREFIX))
        elif data == RUN_NOW:
            await self._run_pipeline_now(chat_id)
        elif data == CHECK_SCREENING:
            await self._send_screening_review(chat_id)
        elif data == SETTINGS_MENU:
            await self._api.send_message(chat_id, "Настройки:", buttons=_settings_menu())
        elif data == ABOUT:
            await self._api.send_message(chat_id, _ABOUT_TEXT, buttons=_main_menu())
        else:
            await self._api.send_message(chat_id, "Неизвестная команда.", buttons=_main_menu())

    async def handle_text(
        self,
        *,
        chat_id: int,
        user_id: int,
        text: str,
        forwarded_channel_username: str | None = None,
        forward_missing_username: bool = False,
    ) -> None:
        if not await self._authorize(chat_id, user_id):
            return
        key = (chat_id, user_id)
        state = self._pending.get(key)
        if state is None:
            await self._api.send_message(
                chat_id,
                "Выберите действие кнопкой.",
                buttons=_main_menu(),
            )
            return
        if state.action is PendingAction.ADD_SOURCE:
            done = await self._add_source(
                chat_id,
                text,
                forwarded_channel_username=forwarded_channel_username,
                forward_missing_username=forward_missing_username,
            )
        elif state.action is PendingAction.EDIT_INTERESTS:
            done = await self._edit_interests(
                chat_id=chat_id,
                user_id=user_id,
                text=text,
                expected_version=state.expected_profile_version,
            )
        elif state.action is PendingAction.EDIT_FRESHNESS:
            done = await self._edit_freshness(
                chat_id=chat_id,
                user_id=user_id,
                text=text,
                expected_version=state.expected_profile_version,
            )
        elif state.action is PendingAction.EDIT_TONE:
            done = await self._edit_tone(
                chat_id=chat_id,
                user_id=user_id,
                text=text,
                expected_version=state.expected_profile_version,
            )
        else:
            done = await self._edit_send_times(
                chat_id=chat_id,
                user_id=user_id,
                text=text,
                expected_version=state.expected_profile_version,
            )
        # Only clear the pending action once it actually resolves (success, or a
        # terminal error that sends the user back to the main menu). Invalid input
        # that expects a retry (e.g. "not a number" for freshness) must leave the
        # pending state in place -- otherwise the next message silently falls
        # through to "Выберите действие кнопкой" instead of being read as the retry.
        if done:
            self._pending.pop(key, None)

    async def _add_source(
        self,
        chat_id: int,
        text: str,
        *,
        forwarded_channel_username: str | None = None,
        forward_missing_username: bool = False,
    ) -> bool:
        if forward_missing_username:
            await self._api.send_message(
                chat_id,
                "У этого канала нет публичного @username, так его нельзя добавить "
                "пересылкой. Пришлите ссылку или @handle вручную.",
            )
            return False
        locator_input = f"@{forwarded_channel_username}" if forwarded_channel_username else text
        try:
            parsed = parse_source_locator(locator_input)
            source = SourceConfig(
                id=str(uuid4()),
                kind=parsed.kind,
                locator=parsed.locator,
                collector=parsed.collector,
            )
            added = await self._repository.add_source(source)
        except DuplicateSourceError as exc:
            await self._api.send_message(
                chat_id,
                f"Такой источник уже есть (id: {exc.existing_source_id}).",
                buttons=_main_menu(),
            )
            return True
        except ValueError as exc:
            await self._api.send_message(chat_id, f"Не удалось добавить источник: {exc}")
            return False
        await self._api.send_message(
            chat_id,
            f"Источник добавлен: {added.locator}\nid: {added.id}",
            buttons=_main_menu(),
        )
        return True

    async def _send_sources(self, chat_id: int) -> None:
        sources = await self._repository.list_sources()
        if not sources:
            text = "Активных источников пока нет."
        else:
            lines = ["Активные источники:"]
            lines.extend(
                (
                    f"• {source.locator} — "
                    f"{source.kind.value}/{source.collector.value}\n  id: {source.id}"
                )
                for source in sources
            )
            text = "\n".join(lines)
        await self._api.send_message(chat_id, text, buttons=_main_menu())

    async def _send_delete_menu(self, chat_id: int) -> None:
        sources = await self._repository.list_sources()
        if not sources:
            await self._api.send_message(
                chat_id,
                "Удалять нечего: активных источников нет.",
                buttons=_main_menu(),
            )
            return
        buttons = tuple(
            (Button(source.locator[:40], f"{DELETE_SOURCE_PREFIX}{source.id}"),)
            for source in sources
        )
        await self._api.send_message(chat_id, "Какой источник удалить?", buttons=buttons)

    async def _send_interests(self, chat_id: int) -> None:
        try:
            profile = await self._repository.get_interest_profile(self._profile_id)
            text = f"{profile.name} (версия {profile.version}):\n\n{profile.description}"
        except LookupError:
            text = "Профиль интересов ещё не задан."
        await self._api.send_message(
            chat_id,
            text,
            buttons=((Button("Редактировать интересы", EDIT_INTERESTS),),),
        )

    async def _edit_interests(
        self,
        *,
        chat_id: int,
        user_id: int,
        text: str,
        expected_version: int | None,
    ) -> bool:
        description = text.strip()
        if not description:
            await self._api.send_message(chat_id, "Текст интересов не может быть пустым.")
            return False
        try:
            if expected_version is None:
                now = datetime.now(UTC)
                profile = await self._repository.create_interest_profile(
                    InterestProfile(
                        id=self._profile_id,
                        name="Основные интересы",
                        description=description,
                        created_at=now,
                        updated_at=now,
                        updated_by_telegram_user_id=user_id,
                    )
                )
            else:
                profile = await self._repository.update_interest_profile(
                    self._profile_id,
                    description=description,
                    expected_version=expected_version,
                    updated_by_telegram_user_id=user_id,
                )
        except ConcurrentUpdateError:
            await self._api.send_message(
                chat_id,
                "Профиль уже изменился в другом запросе. Откройте его снова и повторите правку.",
                buttons=_main_menu(),
            )
            return True
        await self._api.send_message(
            chat_id,
            f"Интересы сохранены, версия {profile.version}.",
            buttons=_main_menu(),
        )
        return True

    async def _current_profile_version(self) -> int | None:
        try:
            profile = await self._repository.get_interest_profile(self._profile_id)
        except LookupError:
            return None
        return profile.version

    async def _send_digest_settings(self, chat_id: int) -> None:
        try:
            profile = await self._repository.get_interest_profile(self._profile_id)
        except LookupError:
            await self._api.send_message(chat_id, _NO_PROFILE_YET, buttons=_main_menu())
            return
        tone = profile.tone_instructions or "по умолчанию (нейтральный редакторский стиль)"
        send_times = _format_digest_send_times(profile.digest_send_times)
        text = (
            f"Свежесть: не показывать новости старше {profile.freshness_days} дн.\n"
            f"Длина саммари: {_LENGTH_LABELS[profile.summary_length]}\n"
            f"Стиль: {tone}\n"
            f"Время отправки: {send_times}"
        )
        await self._api.send_message(
            chat_id,
            text,
            buttons=(
                (Button("Изменить свежесть", EDIT_FRESHNESS),),
                (Button("Изменить длину", EDIT_LENGTH),),
                (Button("Изменить стиль", EDIT_TONE),),
                (Button("Изменить время отправки", EDIT_SEND_TIMES),),
            ),
        )

    async def _set_length(self, chat_id: int, user_id: int, raw_value: str) -> None:
        try:
            length = SummaryLength(raw_value)
        except ValueError:
            await self._api.send_message(chat_id, "Неизвестная длина.", buttons=_main_menu())
            return
        try:
            profile = await self._repository.get_interest_profile(self._profile_id)
        except LookupError:
            await self._api.send_message(chat_id, _NO_PROFILE_YET, buttons=_main_menu())
            return
        try:
            await self._repository.update_digest_length(
                self._profile_id,
                summary_length=length,
                expected_version=profile.version,
                updated_by_telegram_user_id=user_id,
            )
        except ConcurrentUpdateError:
            await self._api.send_message(
                chat_id,
                "Настройки уже изменились в другом запросе. Откройте их снова и повторите.",
                buttons=_main_menu(),
            )
            return
        await self._api.send_message(
            chat_id,
            f"Длина саммари: {_LENGTH_LABELS[length]}.",
            buttons=_main_menu(),
        )

    async def _edit_freshness(
        self,
        *,
        chat_id: int,
        user_id: int,
        text: str,
        expected_version: int | None,
    ) -> bool:
        raw = text.strip()
        if not raw.isdigit() or not (1 <= int(raw) <= 365):
            await self._api.send_message(chat_id, "Нужно целое число дней от 1 до 365.")
            return False
        if expected_version is None:
            await self._api.send_message(chat_id, _NO_PROFILE_YET, buttons=_main_menu())
            return True
        try:
            profile = await self._repository.update_digest_freshness(
                self._profile_id,
                freshness_days=int(raw),
                expected_version=expected_version,
                updated_by_telegram_user_id=user_id,
            )
        except ConcurrentUpdateError:
            await self._api.send_message(
                chat_id,
                "Настройки уже изменились в другом запросе. Откройте их снова и повторите.",
                buttons=_main_menu(),
            )
            return True
        await self._api.send_message(
            chat_id,
            f"Свежесть обновлена: не старше {profile.freshness_days} дн.",
            buttons=_main_menu(),
        )
        return True

    async def _edit_tone(
        self,
        *,
        chat_id: int,
        user_id: int,
        text: str,
        expected_version: int | None,
    ) -> bool:
        raw = text.strip()
        tone: str | None = None if raw == _TONE_RESET else raw
        if tone is not None and not tone:
            await self._api.send_message(chat_id, f'Пришлите текст стиля или "{_TONE_RESET}".')
            return False
        if expected_version is None:
            await self._api.send_message(chat_id, _NO_PROFILE_YET, buttons=_main_menu())
            return True
        try:
            await self._repository.update_digest_tone(
                self._profile_id,
                tone_instructions=tone,
                expected_version=expected_version,
                updated_by_telegram_user_id=user_id,
            )
        except ConcurrentUpdateError:
            await self._api.send_message(
                chat_id,
                "Настройки уже изменились в другом запросе. Откройте их снова и повторите.",
                buttons=_main_menu(),
            )
            return True
        confirmation = (
            "Стиль сброшен на значение по умолчанию." if tone is None else "Стиль сохранён."
        )
        await self._api.send_message(chat_id, confirmation, buttons=_main_menu())
        return True

    async def _edit_send_times(
        self,
        *,
        chat_id: int,
        user_id: int,
        text: str,
        expected_version: int | None,
    ) -> bool:
        raw = text.strip()
        try:
            send_times = () if raw == _SEND_TIMES_RESET else _parse_digest_send_times(raw)
        except ValueError as exc:
            await self._api.send_message(chat_id, str(exc))
            return False
        if expected_version is None:
            await self._api.send_message(chat_id, _NO_PROFILE_YET, buttons=_main_menu())
            return True
        try:
            await self._repository.update_digest_send_times(
                self._profile_id,
                digest_send_times=send_times,
                expected_version=expected_version,
                updated_by_telegram_user_id=user_id,
            )
        except ConcurrentUpdateError:
            await self._api.send_message(
                chat_id,
                "Настройки уже изменились в другом запросе. Откройте их снова и повторите.",
                buttons=_main_menu(),
            )
            return True
        confirmation = (
            "Время отправки сброшено: дайджест будет отправляться при каждом запуске."
            if not send_times
            else f"Время отправки сохранено: {_format_digest_send_times(send_times)}."
        )
        await self._api.send_message(chat_id, confirmation, buttons=_main_menu())
        return True

    def set_backfill_runner(self, runner: BackfillRunner | None) -> None:
        """Wire (or disable) the in-chat "collect for period" backfill action.

        Called once from bot-worker startup after the heavier pipeline dependencies
        (OpenAI clients, connectors, delivery, ...) are built from the same settings
        pipeline-runner uses -- or left unset if those aren't configured, in which case
        the backfill menu tells the owner it isn't available rather than failing.
        """
        self._backfill = runner

    def set_run_now_runner(self, runner: RunNowRunner | None) -> None:
        self._run_now_runner = runner

    async def _send_backfill_menu(self, chat_id: int) -> None:
        if self._backfill is None:
            await self._api.send_message(chat_id, _BACKFILL_NOT_CONFIGURED, buttons=_main_menu())
            return
        await self._api.send_message(
            chat_id,
            "За какой период собрать новости по всем источникам? Учтите: некоторые "
            "источники (в первую очередь сайты и RSS) отдают только недавние материалы "
            "независимо от периода — это ограничение самого источника, не бота.",
            buttons=tuple(
                (Button(f"За {label}", f"{BACKFILL_PREFIX}{key}"),)
                for key, (label, _) in _BACKFILL_PERIODS.items()
            ),
        )

    async def _run_backfill(self, chat_id: int, period_key: str) -> None:
        if self._backfill is None:
            await self._api.send_message(chat_id, _BACKFILL_NOT_CONFIGURED, buttons=_main_menu())
            return
        period = _BACKFILL_PERIODS.get(period_key)
        if period is None:
            await self._api.send_message(chat_id, "Неизвестный период.", buttons=_main_menu())
            return
        if self._pipeline_running:
            await self._api.send_message(chat_id, "Уже выполняется.")
            return
        label, days = period
        min_published_at = (
            _EPOCH_FLOOR if days is None else datetime.now(UTC) - timedelta(days=days)
        )
        self._pipeline_running = True
        try:
            await self._api.send_message(
                chat_id,
                f"⏳ Собираю новости за {label} по всем источникам. "
                "Это может занять несколько минут…",
            )
            report = await self._backfill.run_backfill(min_published_at=min_published_at)
        except Exception as exc:
            logger.exception("in-bot backfill failed")
            await self._api.send_message(
                chat_id,
                f"❌ Не удалось собрать новости: {exc}",
                buttons=_main_menu(),
            )
            return
        finally:
            self._pipeline_running = False
        if report.digest_posts:
            text = (
                f"Готово: собрано {report.collected_items}, материалов "
                f"{report.stored_materials}, кластеров {report.clusters}, в дайджест "
                f"попало {report.digest_posts} (отправлено в канал)."
            )
        else:
            text = (
                f"Готово: собрано {report.collected_items}, материалов "
                f"{report.stored_materials}, кластеров {report.clusters}, но по теме "
                "интересов ничего не прошло отбор — постов не отправлено."
            )
        if report.extraction_failures:
            text += f"\nОшибок извлечения: {report.extraction_failures}."
        await self._api.send_message(chat_id, text, buttons=_main_menu())

    async def _run_pipeline_now(self, chat_id: int) -> None:
        if self._run_now_runner is None:
            await self._api.send_message(chat_id, _RUN_NOW_NOT_CONFIGURED, buttons=_main_menu())
            return
        if self._pipeline_running:
            await self._api.send_message(chat_id, "Уже выполняется.")
            return

        self._pipeline_running = True
        try:
            await self._api.send_message(chat_id, "Запускаю сбор прямо сейчас…")
            report = await self._run_now_runner.run(ignore_schedule=True)
        except Exception as exc:
            logger.exception("in-bot pipeline run failed")
            await self._api.send_message(
                chat_id,
                f"❌ Не удалось собрать новости: {exc}",
                buttons=_main_menu(),
            )
            return
        finally:
            self._pipeline_running = False

        text = (
            f"Готово: проверено источников {report.sources}, новых материалов "
            f"{report.stored_materials}, отправлено постов дайджеста {report.digest_posts}."
        )
        if report.digest_posts == 0:
            if report.stored_materials == 0:
                text += " Ничего нового не найдено."
            else:
                text += " Среди новых материалов ничего релевантного не найдено."
        elif report.stored_materials == 0:
            text += " Отправлены ранее накопленные материалы."
        if report.extraction_failures:
            text += f"\nОшибок извлечения: {report.extraction_failures}."
        await self._api.send_message(chat_id, text, buttons=_main_menu())

    async def _send_screening_review(self, chat_id: int) -> None:
        reviews = await self._repository.list_recent_screenings(
            self._profile_id, limit=_SCREENING_REVIEW_LIMIT
        )
        if not reviews:
            await self._api.send_message(
                chat_id,
                "Пока нет ни одной проверенной новости — сначала запустите сбор.",
                buttons=_main_menu(),
            )
            return
        thresholds = ScreeningThresholds()
        blocks = [f"Последние {len(reviews)} проверенных новостей:"]
        blocks.extend(_format_screening_review(review, thresholds) for review in reviews)
        text = "\n\n".join(blocks)
        if len(text) > _SCREENING_REVIEW_MAX_MESSAGE_CHARS:
            text = text[:_SCREENING_REVIEW_MAX_MESSAGE_CHARS].rstrip() + "\n\n(сообщение обрезано)"
        await self._api.send_message(chat_id, text, buttons=_main_menu())

    async def _authorize(self, chat_id: int, user_id: int) -> bool:
        if user_id == self._owner_user_id:
            return True
        await self._api.send_message(chat_id, "У этого пользователя нет доступа к настройкам.")
        return False


def _main_menu() -> Keyboard:
    return (
        (Button("⚙️ Настройки", SETTINGS_MENU),),
        (Button("Прислать всё актуальное", RUN_NOW),),
        (Button("Собрать за период", BACKFILL_MENU),),
        (Button("О боте", ABOUT),),
    )


def _settings_menu() -> Keyboard:
    return (
        (Button("Добавить источник", ADD_SOURCE),),
        (Button("Посмотреть все источники", LIST_SOURCES),),
        (Button("Удалить источник", DELETE_SOURCE),),
        (
            Button("Посмотреть интересы", VIEW_INTERESTS),
            Button("Редактировать интересы", EDIT_INTERESTS),
        ),
        (Button("Настройки дайджеста", DIGEST_SETTINGS),),
        (Button("Проверить фильтр", CHECK_SCREENING),),
    )


def _truncate(text: str, limit: int) -> str:
    stripped = text.strip()
    if len(stripped) <= limit:
        return stripped
    return stripped[: limit - 1].rstrip() + "…"


def _format_screening_review(review: ScreeningReview, thresholds: ScreeningThresholds) -> str:
    result = review.result
    verdict = "✅ прошла" if thresholds.accepts(result) else "❌ отклонена"
    if result.uncertain:
        verdict += " (не уверен)"
    date = review.material_published_at.strftime("%d.%m.%Y")
    title = _truncate(review.material_title, _SCREENING_REVIEW_TITLE_CHARS)
    reason = _truncate(result.reason, _SCREENING_REVIEW_REASON_CHARS)
    return (
        f"{verdict} — релевантность {result.relevance_score:.2f}, шум {result.noise_score:.2f}\n"
        f"{title} ({date})\n"
        f"{review.material_url}\n"
        f"Причина: {reason}"
    )


def _parse_digest_send_times(text: str) -> tuple[DigestSendTime, ...]:
    entries = [entry.strip() for entry in re.split(r"[,\n]", text) if entry.strip()]
    if not entries:
        raise ValueError("Укажите время в формате «09:00» или «вторник 09:00».")

    result: list[DigestSendTime] = []
    for entry in entries:
        parts = entry.lower().split()
        if len(parts) == 1:
            weekday = None
            time_text = parts[0]
        elif len(parts) == 2:
            weekday_name, time_text = parts
            try:
                weekday = _WEEKDAY_NUMBERS[weekday_name]
            except KeyError as exc:
                raise ValueError(
                    f"Неизвестный день недели «{weekday_name}». Используйте: "
                    f"{', '.join(_WEEKDAY_NAMES)}."
                ) from exc
        else:
            raise ValueError(
                f"Не удалось разобрать «{entry}»: ожидается «09:00» или «вторник 09:00»."
            )

        hour_text, separator, minute_text = time_text.partition(":")
        if separator != ":" or not hour_text.isdigit() or not 1 <= len(hour_text) <= 2:
            raise ValueError(f"Не удалось разобрать время «{time_text}»: используйте формат ЧЧ:00.")
        hour = int(hour_text)
        if not 0 <= hour <= 23:
            raise ValueError(f"Недопустимый час в «{time_text}»: укажите значение от 0 до 23.")
        if minute_text != "00":
            raise ValueError(
                f"Недопустимые минуты в «{time_text}»: можно указать только полный час ЧЧ:00."
            )
        send_time = DigestSendTime(weekday=weekday, hour=hour)
        if send_time not in result:
            result.append(send_time)
    return tuple(result)


def _format_digest_send_times(send_times: tuple[DigestSendTime, ...]) -> str:
    if not send_times:
        return "при каждом запуске"
    labels = []
    for send_time in send_times:
        hour = f"{send_time.hour:02d}:00"
        labels.append(
            f"каждый день в {hour}"
            if send_time.weekday is None
            else f"{_WEEKDAY_NAMES[send_time.weekday]} в {hour}"
        )
    return "; ".join(labels)
