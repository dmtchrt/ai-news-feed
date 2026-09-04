"""Checkpointed Telegram Bot API delivery."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import Protocol

from telegram.error import RetryAfter

from ai_news_feed.domain.models import DeliveryReceipt, Digest
from ai_news_feed.storage.base import PendingDigestPost, Repository

logger = logging.getLogger(__name__)

_DEFAULT_MAX_RETRIES = 3
_RETRY_AFTER_BUFFER_SECONDS = 0.1


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
        max_retries: int = _DEFAULT_MAX_RETRIES,
        retry_after_buffer_seconds: float = _RETRY_AFTER_BUFFER_SECONDS,
        sleep: Callable[[float], Awaitable[None]] | None = None,
    ) -> None:
        if max_retries < 0:
            raise ValueError("max_retries must not be negative")
        if retry_after_buffer_seconds < 0:
            raise ValueError("retry_after_buffer_seconds must not be negative")
        self._bot = bot
        self._repository = repository
        self._channel_id = channel_id
        self._clock = clock or (lambda: datetime.now(UTC))
        self._max_retries = max_retries
        self._retry_after_buffer_seconds = retry_after_buffer_seconds
        self._sleep: Callable[[float], Awaitable[None]] = sleep or asyncio.sleep

    async def send(self, digest: Digest) -> DeliveryReceipt:
        await self._repository.prepare_digest(digest, channel_id=str(self._channel_id))
        pending = await self._repository.list_pending_digest_posts(digest.id)
        for post in pending:
            try:
                message = await self._send_post(digest.id, post)
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

    async def _send_post(
        self,
        digest_id: str,
        post: PendingDigestPost,
    ) -> SentTelegramMessage:
        for retry_number in range(self._max_retries + 1):
            try:
                return await self._bot.send_message(
                    chat_id=self._channel_id,
                    text=post.text,
                    parse_mode="HTML",
                    disable_web_page_preview=True,
                )
            except RetryAfter as exc:
                if retry_number >= self._max_retries:
                    raise
                delay_seconds = _retry_after_seconds(exc) + self._retry_after_buffer_seconds
                logger.warning(
                    "Telegram flood control: digest_id=%s post=%d retry=%d/%d delay=%.1fs",
                    digest_id,
                    post.position,
                    retry_number + 1,
                    self._max_retries,
                    delay_seconds,
                )
                await self._sleep(delay_seconds)
        raise AssertionError("unreachable")


def _retry_after_seconds(exc: RetryAfter) -> float:
    # PTB 22.8's public retry_after property warns while transitioning from int to
    # timedelta. Its own AIORateLimiter reads the normalized timedelta the same way.
    return exc._retry_after.total_seconds()
