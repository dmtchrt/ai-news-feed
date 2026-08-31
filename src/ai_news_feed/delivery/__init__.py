"""Digest delivery adapters."""

from ai_news_feed.delivery.telegram import TelegramDelivery, TelegramDeliveryError

__all__ = ["TelegramDelivery", "TelegramDeliveryError"]
