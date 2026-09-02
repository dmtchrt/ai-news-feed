from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import pytest

from ai_news_feed.bot.handlers import (
    ADD_SOURCE,
    BACKFILL_MENU,
    BACKFILL_PREFIX,
    DELETE_SOURCE,
    DIGEST_SETTINGS,
    EDIT_FRESHNESS,
    EDIT_INTERESTS,
    EDIT_LENGTH,
    EDIT_TONE,
    LIST_SOURCES,
    SET_LENGTH_PREFIX,
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


@dataclass(frozen=True)
class _FakeBackfillReport:
    collected_items: int
    extraction_failures: int
    stored_materials: int
    clusters: int
    digest_posts: int


class _FakeBackfillRunner:
    def __init__(
        self,
        report: _FakeBackfillReport | None = None,
        *,
        error: Exception | None = None,
    ) -> None:
        self.report = report
        self.error = error
        self.calls: list[datetime] = []

    async def run_backfill(self, *, min_published_at: datetime) -> _FakeBackfillReport:
        self.calls.append(min_published_at)
        if self.error is not None:
            raise self.error
        assert self.report is not None
        return self.report


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


@pytest.mark.asyncio
async def test_digest_settings_require_interests_first() -> None:
    repository = InMemoryRepository()
    api = _FakeBotAPI()
    handlers = BotWorkerHandlers(repository=repository, api=api, owner_user_id=42)

    await handlers.handle_callback(chat_id=100, user_id=42, data=DIGEST_SETTINGS)
    assert "интерес" in api.messages[-1].text.lower()

    await handlers.handle_callback(chat_id=100, user_id=42, data=EDIT_FRESHNESS)
    assert "интерес" in api.messages[-1].text.lower()


@pytest.mark.asyncio
async def test_bot_manages_digest_settings() -> None:
    repository = InMemoryRepository()
    api = _FakeBotAPI()
    handlers = BotWorkerHandlers(repository=repository, api=api, owner_user_id=42)
    await handlers.handle_callback(chat_id=100, user_id=42, data=EDIT_INTERESTS)
    await handlers.handle_text(chat_id=100, user_id=42, text="ИИ-агенты и рынок ИИ")

    await handlers.handle_callback(chat_id=100, user_id=42, data=DIGEST_SETTINGS)
    assert "7 дн." in api.messages[-1].text
    assert "Нормально" in api.messages[-1].text
    assert "по умолчанию" in api.messages[-1].text

    await handlers.handle_callback(chat_id=100, user_id=42, data=EDIT_FRESHNESS)
    await handlers.handle_text(chat_id=100, user_id=42, text="not a number")
    assert "целое число" in api.messages[-1].text
    await handlers.handle_text(chat_id=100, user_id=42, text="3")
    profile = await repository.get_interest_profile("default")
    assert profile.freshness_days == 3
    assert profile.version == 2

    await handlers.handle_callback(chat_id=100, user_id=42, data=EDIT_LENGTH)
    length_callback = api.messages[-1].buttons[2][0].callback_data
    assert length_callback == f"{SET_LENGTH_PREFIX}detailed"
    await handlers.handle_callback(chat_id=100, user_id=42, data=length_callback)
    profile = await repository.get_interest_profile("default")
    assert profile.summary_length.value == "detailed"
    assert profile.version == 3

    await handlers.handle_callback(chat_id=100, user_id=42, data=EDIT_TONE)
    await handlers.handle_text(chat_id=100, user_id=42, text="Сухо, по-деловому, без эмодзи.")
    profile = await repository.get_interest_profile("default")
    assert profile.tone_instructions == "Сухо, по-деловому, без эмодзи."
    assert profile.version == 4

    await handlers.handle_callback(chat_id=100, user_id=42, data=EDIT_TONE)
    await handlers.handle_text(chat_id=100, user_id=42, text="-")
    profile = await repository.get_interest_profile("default")
    assert profile.tone_instructions is None
    assert profile.version == 5


@pytest.mark.asyncio
async def test_backfill_menu_reports_when_not_configured() -> None:
    repository = InMemoryRepository()
    api = _FakeBotAPI()
    handlers = BotWorkerHandlers(repository=repository, api=api, owner_user_id=42)

    await handlers.handle_callback(chat_id=100, user_id=42, data=BACKFILL_MENU)
    assert "недоступен" in api.messages[-1].text.lower()

    await handlers.handle_callback(chat_id=100, user_id=42, data=f"{BACKFILL_PREFIX}week")
    assert "недоступен" in api.messages[-1].text.lower()


@pytest.mark.asyncio
async def test_backfill_runs_for_selected_period_and_reports_counts() -> None:
    repository = InMemoryRepository()
    api = _FakeBotAPI()
    runner = _FakeBackfillRunner(
        _FakeBackfillReport(
            collected_items=42,
            extraction_failures=0,
            stored_materials=10,
            clusters=4,
            digest_posts=3,
        )
    )
    handlers = BotWorkerHandlers(repository=repository, api=api, owner_user_id=42)
    handlers.set_backfill_runner(runner)

    await handlers.handle_callback(chat_id=100, user_id=42, data=BACKFILL_MENU)
    period_callback = next(
        button.callback_data
        for row in api.messages[-1].buttons
        for button in row
        if button.callback_data == f"{BACKFILL_PREFIX}week"
    )

    await handlers.handle_callback(chat_id=100, user_id=42, data=period_callback)

    assert len(runner.calls) == 1
    assert runner.calls[0].tzinfo is not None
    delta = datetime.now(UTC) - runner.calls[0]
    assert timedelta(days=7) <= delta < timedelta(days=7, minutes=1)
    assert "42" in api.messages[-1].text
    assert "3" in api.messages[-1].text


@pytest.mark.asyncio
async def test_backfill_all_time_uses_epoch_floor() -> None:
    repository = InMemoryRepository()
    api = _FakeBotAPI()
    runner = _FakeBackfillRunner(
        _FakeBackfillReport(
            collected_items=5,
            extraction_failures=0,
            stored_materials=5,
            clusters=2,
            digest_posts=0,
        )
    )
    handlers = BotWorkerHandlers(repository=repository, api=api, owner_user_id=42, backfill=runner)

    await handlers.handle_callback(chat_id=100, user_id=42, data=f"{BACKFILL_PREFIX}all")

    assert runner.calls == [datetime(2000, 1, 1, tzinfo=UTC)]
    assert "не прошло отбор" in api.messages[-1].text


@pytest.mark.asyncio
async def test_backfill_reports_failure_without_crashing() -> None:
    repository = InMemoryRepository()
    api = _FakeBotAPI()
    runner = _FakeBackfillRunner(error=RuntimeError("boom"))
    handlers = BotWorkerHandlers(repository=repository, api=api, owner_user_id=42, backfill=runner)

    await handlers.handle_callback(chat_id=100, user_id=42, data=f"{BACKFILL_PREFIX}month")

    assert "не удалось" in api.messages[-1].text.lower()
    assert "boom" in api.messages[-1].text


@pytest.mark.asyncio
async def test_backfill_rejects_unknown_period() -> None:
    repository = InMemoryRepository()
    api = _FakeBotAPI()
    runner = _FakeBackfillRunner()
    handlers = BotWorkerHandlers(repository=repository, api=api, owner_user_id=42, backfill=runner)

    await handlers.handle_callback(chat_id=100, user_id=42, data=f"{BACKFILL_PREFIX}decade")

    assert not runner.calls
    assert "неизвестный период" in api.messages[-1].text.lower()


@pytest.mark.asyncio
async def test_add_source_keeps_pending_state_after_invalid_input() -> None:
    """Regression: handle_text used to clear the pending action unconditionally,
    even when the handler rejected the input and expected a retry (invalid source
    text, or "not a number" for freshness -- see test_bot_manages_digest_settings).
    That silently dropped the next, valid message instead of treating it as the
    retry."""
    repository = InMemoryRepository()
    api = _FakeBotAPI()
    handlers = BotWorkerHandlers(repository=repository, api=api, owner_user_id=42)

    await handlers.handle_callback(chat_id=100, user_id=42, data=ADD_SOURCE)
    await handlers.handle_text(chat_id=100, user_id=42, text="not a url")
    assert "http(s)-url" in api.messages[-1].text.lower()
    assert not await repository.list_sources()

    await handlers.handle_text(chat_id=100, user_id=42, text="@ailev_blog")
    sources = await repository.list_sources()
    assert len(sources) == 1
    assert sources[0].locator == "@ailev_blog"
