"""Checkpointed Telegram Bot API delivery."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from typing import Protocol

from ai_news_feed.domain.models import DeliveryReceipt, Digest
from ai_news_feed.storage.base import Repository


class SentTelegramMessage(Protocol):
    message_id: int


class TelegramBotAPI(Protocol):
    async def send_message(
        self,
        *,
        chat_id: str | int,
        text: str,
        parse_mode: str | None,
        disable_web_page_preview: bool,
    ) -> SentTelegramMessage: ...


class TelegramDeliveryError(RuntimeError):
    pass


class TelegramDelivery:
    def __init__(
        self,
        *,
        bot: TelegramBotAPI,
        repository: Repository,
        channel_id: str | int,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._bot = bot
        self._repository = repository
        self._channel_id = channel_id
        self._clock = clock or (lambda: datetime.now(UTC))

    async def send(self, digest: Digest) -> DeliveryReceipt:
        await self._repository.prepare_digest(digest, channel_id=str(self._channel_id))
        pending = await self._repository.list_pending_digest_posts(digest.id)
        for post in pending:
            try:
                message = await self._bot.send_message(
                    chat_id=self._channel_id,
                    text=post.text,
                    parse_mode="HTML",
                    disable_web_page_preview=True,
                )
            except Exception as exc:
                raise TelegramDeliveryError(
                    f"Telegram rejected digest {digest.id} post {post.position}: {exc}"
                ) from exc
            sent_at = self._clock()
            await self._repository.mark_digest_post_sent(
                digest.id,
                post.position,
                telegram_message_id=message.message_id,
                sent_at=sent_at,
            )
        receipt = await self._repository.get_delivery_receipt(digest.id)
        if receipt is None:
            raise TelegramDeliveryError(f"digest has unconfirmed posts: {digest.id}")
        return receipt
