"""Persistence boundary and test-friendly implementations."""

from ai_news_feed.storage.base import Repository
from ai_news_feed.storage.memory import InMemoryRepository

__all__ = ["InMemoryRepository", "Repository"]
