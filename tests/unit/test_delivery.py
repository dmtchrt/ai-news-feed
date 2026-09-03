from dataclasses import dataclass
from datetime import UTC, datetime

import pytest

from ai_news_feed.delivery.telegram import TelegramDelivery, TelegramDeliveryError
from ai_news_feed.digest import DigestComposer
from ai_news_feed.domain.models import DigestItem
from ai_news_feed.storage.memory import InMemoryRepository


def _item(
    index: int,
    *,
    summary: str | None = None,
    source_links: tuple[str, ...] | None = None,
    source_published_ats: tuple[datetime, ...] | None = None,
) -> DigestItem:
    return DigestItem(
        cluster_id=f"cluster-{index}",
        summary=summary or f"Новость номер {index} о развитии автономных ИИ-агентов.",
        source_links=source_links or (f"https://example.test/{index}",),
        source_published_ats=source_published_ats,
        model="fake-summary",
        prompt_version="summary-v1",
    )


@dataclass(frozen=True)
class _SentMessage:
    message_id: int


class _FakeTelegramBot:
    def __init__(self, *, fail_on_call: int | None = None) -> None:
        self.calls: list[tuple[str | int, str, str | None, bool]] = []
        self.fail_on_call = fail_on_call

    async def send_message(
        self,
        *,
        chat_id: str | int,
        text: str,
        parse_mode: str | None,
        disable_web_page_preview: bool,
    ) -> _SentMessage:
        self.calls.append((chat_id, text, parse_mode, disable_web_page_preview))
        if self.fail_on_call == len(self.calls):
            raise RuntimeError("fake Telegram outage")
        return _SentMessage(message_id=1000 + len(self.calls))


def test_digest_composer_returns_none_for_expected_empty_result_and_respects_limit() -> None:
    composer = DigestComposer(max_post_chars=160)

    assert composer.compose([], profile_id="default", profile_version=1) is None
    digest = composer.compose(
        [_item(1), _item(2)],
        profile_id="default",
        profile_version=1,
        created_at=datetime(2026, 8, 31, tzinfo=UTC),
    )

    assert digest is not None
    assert all(len(post) <= 160 for post in digest.posts)
    assert len(digest.posts) >= 2


def test_digest_composer_never_merges_two_items_into_one_post() -> None:
    # Default 4096-char limit: both items would easily fit in a single post together,
    # but "one news item -> one post" must hold regardless of how much room is left.
    composer = DigestComposer()

    digest = composer.compose(
        [_item(1), _item(2)],
        profile_id="default",
        profile_version=1,
        created_at=datetime(2026, 8, 31, tzinfo=UTC),
    )

    assert digest is not None
    assert len(digest.posts) == 2
    assert "Новость номер 1" in digest.posts[0]
    assert "Новость номер 2" not in digest.posts[0]
    assert "Новость номер 2" in digest.posts[1]
    assert "Новость номер 1" not in digest.posts[1]


def test_digest_item_link_is_hidden_html_anchor_not_raw_url() -> None:
    composer = DigestComposer()

    digest = composer.compose(
        [_item(1)],
        profile_id="default",
        profile_version=1,
        created_at=datetime(2026, 8, 31, tzinfo=UTC),
    )

    assert digest is not None
    sources_section = digest.posts[0].split("Источники:\n", 1)[1]
    assert sources_section == '1. <a href="https://example.test/1">example.test</a>'


def test_digest_item_shows_date_under_link_when_available() -> None:
    composer = DigestComposer()
    item = _item(1, source_published_ats=(datetime(2026, 8, 3, tzinfo=UTC),))

    digest = composer.compose(
        [item],
        profile_id="default",
        profile_version=1,
        created_at=datetime(2026, 8, 31, tzinfo=UTC),
    )

    assert digest is not None
    sources_section = digest.posts[0].split("Источники:\n", 1)[1]
    assert sources_section == ('1. <a href="https://example.test/1">example.test</a>\n03.08.2026')


def test_digest_item_pairs_each_link_with_its_own_date_in_order() -> None:
    composer = DigestComposer()
    item = _item(
        1,
        source_links=("https://a.example.test/x", "https://b.example.test/y"),
        source_published_ats=(
            datetime(2026, 8, 3, tzinfo=UTC),
            datetime(2026, 8, 10, tzinfo=UTC),
        ),
    )

    digest = composer.compose(
        [item],
        profile_id="default",
        profile_version=1,
        created_at=datetime(2026, 8, 31, tzinfo=UTC),
    )

    assert digest is not None
    sources_section = digest.posts[0].split("Источники:\n", 1)[1]
    assert sources_section == (
        '1. <a href="https://a.example.test/x">a.example.test</a>\n'
        "03.08.2026\n"
        '2. <a href="https://b.example.test/y">b.example.test</a>\n'
        "10.08.2026"
    )


def test_digest_item_escapes_html_special_characters_in_summary_and_link_label() -> None:
    composer = DigestComposer()
    item = _item(
        1,
        summary="Компании A&B обсуждают <AGI> risk > reward",
        source_links=("https://example.test/a&b",),
    )

    digest = composer.compose(
        [item],
        profile_id="default",
        profile_version=1,
        created_at=datetime(2026, 8, 31, tzinfo=UTC),
    )

    assert digest is not None
    post = digest.posts[0]
    assert "A&amp;B" in post
    assert "&lt;AGI&gt;" in post
    assert "risk &gt; reward" in post
    assert "<AGI>" not in post
    assert 'href="https://example.test/a&amp;b"' in post


@pytest.mark.asyncio
async def test_delivery_is_checkpointed_and_second_send_is_noop() -> None:
    digest = DigestComposer().compose(
        [_item(1)],
        profile_id="default",
        profile_version=1,
        created_at=datetime(2026, 8, 31, tzinfo=UTC),
    )
    assert digest is not None
    repository = InMemoryRepository()
    bot = _FakeTelegramBot()
    delivery = TelegramDelivery(
        bot=bot,
        repository=repository,
        channel_id="@private_channel",
        clock=lambda: datetime(2026, 8, 31, 12, tzinfo=UTC),
    )

    first = await delivery.send(digest)
    second = await delivery.send(digest)

    assert first == second
    assert first.telegram_message_ids == (1001,)
    assert len(bot.calls) == 1
    assert bot.calls[0][2] == "HTML"
    assert bot.calls[0][3] is True


@pytest.mark.asyncio
async def test_delivery_retry_skips_already_confirmed_posts() -> None:
    digest = DigestComposer(max_post_chars=160).compose(
        [_item(1), _item(2)],
        profile_id="default",
        profile_version=1,
        created_at=datetime(2026, 8, 31, tzinfo=UTC),
    )
    assert digest is not None
    assert len(digest.posts) == 2
    repository = InMemoryRepository()
    failing_bot = _FakeTelegramBot(fail_on_call=2)
    delivery = TelegramDelivery(
        bot=failing_bot,
        repository=repository,
        channel_id=-100123,
        clock=lambda: datetime(2026, 8, 31, 12, tzinfo=UTC),
    )

    with pytest.raises(TelegramDeliveryError, match="post 1"):
        await delivery.send(digest)

    retry_bot = _FakeTelegramBot()
    receipt = await TelegramDelivery(
        bot=retry_bot,
        repository=repository,
        channel_id=-100123,
        clock=lambda: datetime(2026, 8, 31, 12, 1, tzinfo=UTC),
    ).send(digest)

    assert len(retry_bot.calls) == 1
    assert receipt.telegram_message_ids == (1001, 1001)
