"""Lightweight Telegram bot-worker; this package does not import pipeline/ML modules."""

from ai_news_feed.bot.handlers import BotWorkerHandlers

__all__ = ["BotWorkerHandlers"]
