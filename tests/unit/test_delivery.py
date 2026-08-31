from dataclasses import dataclass
from datetime import UTC, datetime

import pytest

from ai_news_feed.delivery.telegram import TelegramDelivery, TelegramDeliveryError
from ai_news_feed.digest import DigestComposer
from ai_news_feed.domain.models import DigestItem
from ai_news_feed.storage.memory import InMemoryRepository


def _item(index: int, *, summary: str | None = None) -> DigestItem:
    return DigestItem(
        cluster_id=f"cluster-{index}",
        summary=summary or f"Новость номер {index} о развитии автономных ИИ-агентов.",
        source_links=(f"https://example.test/{index}",),
        model="fake-summary",
        prompt_version="summary-v1",
    )


@dataclass(frozen=True)
class _SentMessage:
    message_id: int


class _FakeTelegramBot:
    def __init__(self, *, fail_on_call: int | None = None) -> None:
        self.calls: list[tuple[str | int, str, bool]] = []
        self.fail_on_call = fail_on_call

    async def send_message(
        self,
        *,
        chat_id: str | int,
        text: str,
        disable_web_page_preview: bool,
    ) -> _SentMessage:
        self.calls.append((chat_id, text, disable_web_page_preview))
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
    assert bot.calls[0][2]


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
