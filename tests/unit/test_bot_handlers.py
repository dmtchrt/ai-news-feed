from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import pytest

from ai_news_feed.bot.handlers import (
    ABOUT,
    ADD_SOURCE,
    BACKFILL_MENU,
    BACKFILL_PREFIX,
    CHECK_SCREENING,
    DELETE_SOURCE,
    DIGEST_SETTINGS,
    EDIT_FRESHNESS,
    EDIT_INTERESTS,
    EDIT_LENGTH,
    EDIT_TONE,
    LIST_SOURCES,
    SET_LENGTH_PREFIX,
    SETTINGS_MENU,
    VIEW_INTERESTS,
    BotWorkerHandlers,
    Keyboard,
)
from ai_news_feed.domain.models import Material, NewsCluster, ScreeningResult
from ai_news_feed.normalization import content_hash
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


def _material(material_id: str, published_at: datetime) -> Material:
    text = f"Текст новости {material_id}"
    return Material(
        id=material_id,
        source_id="source",
        external_id=material_id,
        original_url=f"https://example.test/{material_id}",
        published_at=published_at,
        fetched_at=published_at,
        title=f"Новость {material_id}",
        text=text,
        content_hash=content_hash(text),
    )


@pytest.mark.asyncio
async def test_bot_adds_lists_rejects_duplicate_and_soft_deletes_source() -> None:
    repository = InMemoryRepository()
    api = _FakeBotAPI()
    handlers = BotWorkerHandlers(repository=repository, api=api, owner_user_id=42)

    await handlers.handle_start(chat_id=100, user_id=42)
    assert [button.text for row in api.messages[-1].buttons for button in row] == [
        "⚙️ Настройки",
        "Собрать за период",
        "О боте",
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


@pytest.mark.asyncio
async def test_bot_settings_menu_and_about_screen() -> None:
    repository = InMemoryRepository()
    api = _FakeBotAPI()
    handlers = BotWorkerHandlers(repository=repository, api=api, owner_user_id=42)

    await handlers.handle_start(chat_id=100, user_id=42)
    assert [button.text for row in api.messages[-1].buttons for button in row] == [
        "⚙️ Настройки",
        "Собрать за период",
        "О боте",
    ]

    await handlers.handle_callback(chat_id=100, user_id=42, data=SETTINGS_MENU)
    assert [button.text for row in api.messages[-1].buttons for button in row] == [
        "Добавить источник",
        "Посмотреть все источники",
        "Удалить источник",
        "Посмотреть интересы",
        "Редактировать интересы",
        "Настройки дайджеста",
        "Проверить фильтр",
    ]

    await handlers.handle_callback(chat_id=100, user_id=42, data=ABOUT)
    assert "AI News Feed" in api.messages[-1].text
    assert [button.text for row in api.messages[-1].buttons for button in row] == [
        "⚙️ Настройки",
        "Собрать за период",
        "О боте",
    ]


@pytest.mark.asyncio
async def test_bot_add_source_from_forwarded_channel() -> None:
    """Forwarding a channel post is an alternative to typing a link/@handle --
    app.py resolves the forward to a public username before calling handle_text."""
    repository = InMemoryRepository()
    api = _FakeBotAPI()
    handlers = BotWorkerHandlers(repository=repository, api=api, owner_user_id=42)

    await handlers.handle_callback(chat_id=100, user_id=42, data=ADD_SOURCE)
    await handlers.handle_text(
        chat_id=100,
        user_id=42,
        text="",
        forwarded_channel_username="AiLev_Blog",
    )
    sources = await repository.list_sources()
    assert len(sources) == 1
    assert sources[0].locator == "@ailev_blog"
    assert "добавлен" in api.messages[-1].text.lower()


@pytest.mark.asyncio
async def test_bot_add_source_from_forward_without_username_keeps_pending_state() -> None:
    repository = InMemoryRepository()
    api = _FakeBotAPI()
    handlers = BotWorkerHandlers(repository=repository, api=api, owner_user_id=42)

    await handlers.handle_callback(chat_id=100, user_id=42, data=ADD_SOURCE)
    await handlers.handle_text(
        chat_id=100,
        user_id=42,
        text="",
        forward_missing_username=True,
    )
    assert "нет публичного" in api.messages[-1].text.lower()
    assert not await repository.list_sources()

    # Pending state must survive the rejection, exactly like any other invalid
    # add-source attempt -- otherwise the next, valid message silently falls
    # through to "Выберите действие кнопкой" instead of being read as the retry.
    await handlers.handle_text(chat_id=100, user_id=42, text="@ailev_blog")
    sources = await repository.list_sources()
    assert len(sources) == 1
    assert sources[0].locator == "@ailev_blog"


@pytest.mark.asyncio
async def test_bot_screening_review_reports_when_nothing_screened_yet() -> None:
    repository = InMemoryRepository()
    api = _FakeBotAPI()
    handlers = BotWorkerHandlers(repository=repository, api=api, owner_user_id=42)

    await handlers.handle_callback(chat_id=100, user_id=42, data=CHECK_SCREENING)

    assert "сначала запустите сбор" in api.messages[-1].text.lower()
    assert api.messages[-1].buttons != ()


@pytest.mark.asyncio
async def test_bot_shows_recent_screening_verdicts_with_reason_and_date() -> None:
    repository = InMemoryRepository()
    api = _FakeBotAPI()
    handlers = BotWorkerHandlers(repository=repository, api=api, owner_user_id=42)
    published_at = datetime(2026, 8, 3, tzinfo=UTC)
    accepted_material = _material("accepted", published_at)
    rejected_material = _material("rejected", published_at)
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
    await repository.save_processing_result(
        materials=(accepted_material, rejected_material),
        clusters=(accepted_cluster, rejected_cluster),
        duplicate_links=(),
        screening_results=(accepted_screening, rejected_screening),
        profile_id="default",
        profile_version=1,
    )

    await handlers.handle_callback(chat_id=100, user_id=42, data=CHECK_SCREENING)

    text = api.messages[-1].text
    assert "✅" in text
    assert "❌" in text
    assert accepted_material.title in text
    assert rejected_material.title in text
    assert accepted_material.original_url in text
    assert rejected_material.original_url in text
    assert "Точно по теме ИИ-агентов." in text
    assert "Не по теме." in text
    assert text.count("03.08.2026") == 2


@pytest.mark.asyncio
async def test_bot_screening_review_message_stays_under_telegram_limit() -> None:
    """Regression: BotAPI.send_message is a single, unchunked message (nothing like
    digest.py's _split_text/_pack is reused here) -- ten items with realistically long
    titles/URLs/reasons must still fit, or Telegram would reject the whole message."""
    repository = InMemoryRepository()
    api = _FakeBotAPI()
    handlers = BotWorkerHandlers(repository=repository, api=api, owner_user_id=42)
    published_at = datetime(2026, 8, 3, tzinfo=UTC)
    long_title_suffix = "очень длинный заголовок с массой подробностей " * 5
    long_query = "utm_source=test&utm_campaign=" + "x" * 250
    materials = [
        Material(
            id=f"m{i}",
            source_id="source",
            external_id=f"m{i}",
            original_url=f"https://example.test/article-{i}?{long_query}",
            published_at=published_at,
            fetched_at=published_at,
            title=f"Новость {i}: {long_title_suffix}",
            text=f"Текст новости {i}",
            content_hash=content_hash(f"Текст новости {i}"),
        )
        for i in range(10)
    ]
    clusters = [
        NewsCluster(
            id=f"cluster-{i}",
            material_ids=(material.id,),
            representative_id=material.id,
            similarities={material.id: 1.0},
        )
        for i, material in enumerate(materials)
    ]
    screenings = [
        ScreeningResult(
            cluster_id=cluster.id,
            relevance_score=0.9,
            noise_score=0.1,
            reason="Подробное обоснование релевантности этой новости. " * 10,
            model="fake-screen",
            prompt_version="screen-v1",
        )
        for cluster in clusters
    ]
    await repository.save_processing_result(
        materials=materials,
        clusters=clusters,
        duplicate_links=(),
        screening_results=screenings,
        profile_id="default",
        profile_version=1,
    )

    await handlers.handle_callback(chat_id=100, user_id=42, data=CHECK_SCREENING)

    text = api.messages[-1].text
    assert len(text) <= 4096
    assert "(сообщение обрезано)" in text
