from dataclasses import dataclass

import pytest

from ai_news_feed.bot.handlers import (
    ADD_SOURCE,
    DELETE_SOURCE,
    EDIT_INTERESTS,
    LIST_SOURCES,
    VIEW_INTERESTS,
    BotWorkerHandlers,
    Keyboard,
)
from ai_news_feed.storage.memory import InMemoryRepository


@dataclass(frozen=True)
class _Message:
    chat_id: int
    text: str
    buttons: Keyboard


class _FakeBotAPI:
    def __init__(self) -> None:
        self.messages: list[_Message] = []

    async def send_message(
        self,
        chat_id: int,
        text: str,
        *,
        buttons: Keyboard = (),
    ) -> None:
        self.messages.append(_Message(chat_id, text, buttons))


@pytest.mark.asyncio
async def test_bot_adds_lists_rejects_duplicate_and_soft_deletes_source() -> None:
    repository = InMemoryRepository()
    api = _FakeBotAPI()
    handlers = BotWorkerHandlers(repository=repository, api=api, owner_user_id=42)

    await handlers.handle_start(chat_id=100, user_id=42)
    assert [button.text for row in api.messages[-1].buttons for button in row][:3] == [
        "Добавить источник",
        "Посмотреть все источники",
        "Удалить источник",
    ]

    await handlers.handle_callback(chat_id=100, user_id=42, data=ADD_SOURCE)
    await handlers.handle_text(chat_id=100, user_id=42, text="@AiLev_Blog")
    source = (await repository.list_sources())[0]
    assert source.locator == "@ailev_blog"

    await handlers.handle_callback(chat_id=100, user_id=42, data=ADD_SOURCE)
    await handlers.handle_text(chat_id=100, user_id=42, text="https://t.me/ailev_blog")
    assert "уже есть" in api.messages[-1].text
    assert len(await repository.list_sources()) == 1

    await handlers.handle_callback(chat_id=100, user_id=42, data=LIST_SOURCES)
    assert source.id in api.messages[-1].text
    await handlers.handle_callback(chat_id=100, user_id=42, data=DELETE_SOURCE)
    delete_callback = api.messages[-1].buttons[0][0].callback_data
    await handlers.handle_callback(chat_id=100, user_id=42, data=delete_callback)
    assert not await repository.list_sources()
    assert len(await repository.list_sources(include_disabled=True)) == 1


@pytest.mark.asyncio
async def test_bot_creates_views_and_updates_interest_profile() -> None:
    repository = InMemoryRepository()
    api = _FakeBotAPI()
    handlers = BotWorkerHandlers(repository=repository, api=api, owner_user_id=42)

    await handlers.handle_callback(chat_id=100, user_id=42, data=EDIT_INTERESTS)
    await handlers.handle_text(chat_id=100, user_id=42, text="ИИ-агенты и рынок ИИ")
    profile = await repository.get_interest_profile("default")
    assert profile.version == 1
    assert profile.updated_by_telegram_user_id == 42

    await handlers.handle_callback(chat_id=100, user_id=42, data=VIEW_INTERESTS)
    assert "ИИ-агенты" in api.messages[-1].text

    await handlers.handle_callback(chat_id=100, user_id=42, data=EDIT_INTERESTS)
    await handlers.handle_text(chat_id=100, user_id=42, text="Автономные агенты")
    assert (await repository.get_interest_profile("default")).version == 2


@pytest.mark.asyncio
async def test_bot_rejects_non_owner_without_mutating_repository() -> None:
    repository = InMemoryRepository()
    api = _FakeBotAPI()
    handlers = BotWorkerHandlers(repository=repository, api=api, owner_user_id=42)

    await handlers.handle_callback(chat_id=200, user_id=7, data=ADD_SOURCE)
    await handlers.handle_text(chat_id=200, user_id=7, text="@ailev_blog")

    assert not await repository.list_sources()
    assert all("нет доступа" in message.text for message in api.messages)
