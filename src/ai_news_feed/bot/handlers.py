"""Framework-independent bot command handlers, testable with a fake Bot API."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from typing import Protocol
from uuid import uuid4

from ai_news_feed.domain.models import InterestProfile, SourceConfig
from ai_news_feed.sources.locator import parse_source_locator
from ai_news_feed.storage.base import ConcurrentUpdateError, DuplicateSourceError, Repository

ADD_SOURCE = "menu:add-source"
LIST_SOURCES = "menu:list-sources"
DELETE_SOURCE = "menu:delete-source"
VIEW_INTERESTS = "interests:view"
EDIT_INTERESTS = "interests:edit"
DELETE_SOURCE_PREFIX = "source:delete:"


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


class PendingAction(StrEnum):
    ADD_SOURCE = "add_source"
    EDIT_INTERESTS = "edit_interests"


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
    ) -> None:
        if owner_user_id < 1:
            raise ValueError("owner_user_id must be positive")
        self._repository = repository
        self._api = api
        self._owner_user_id = owner_user_id
        self._profile_id = interest_profile_id
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
                "Пришлите URL сайта/RSS или Telegram-канал в формате @handle.",
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
        else:
            await self._api.send_message(chat_id, "Неизвестная команда.", buttons=_main_menu())

    async def handle_text(self, *, chat_id: int, user_id: int, text: str) -> None:
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
            await self._add_source(chat_id, text)
        else:
            await self._edit_interests(
                chat_id=chat_id,
                user_id=user_id,
                text=text,
                expected_version=state.expected_profile_version,
            )
        self._pending.pop(key, None)

    async def _add_source(self, chat_id: int, text: str) -> None:
        try:
            parsed = parse_source_locator(text)
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
            return
        except ValueError as exc:
            await self._api.send_message(chat_id, f"Не удалось добавить источник: {exc}")
            return
        await self._api.send_message(
            chat_id,
            f"Источник добавлен: {added.locator}\nid: {added.id}",
            buttons=_main_menu(),
        )

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
    ) -> None:
        description = text.strip()
        if not description:
            await self._api.send_message(chat_id, "Текст интересов не может быть пустым.")
            return
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
            return
        await self._api.send_message(
            chat_id,
            f"Интересы сохранены, версия {profile.version}.",
            buttons=_main_menu(),
        )

    async def _authorize(self, chat_id: int, user_id: int) -> bool:
        if user_id == self._owner_user_id:
            return True
        await self._api.send_message(chat_id, "У этого пользователя нет доступа к настройкам.")
        return False


def _main_menu() -> Keyboard:
    return (
        (Button("Добавить источник", ADD_SOURCE),),
        (Button("Посмотреть все источники", LIST_SOURCES),),
        (Button("Удалить источник", DELETE_SOURCE),),
        (
            Button("Посмотреть интересы", VIEW_INTERESTS),
            Button("Редактировать интересы", EDIT_INTERESTS),
        ),
    )
