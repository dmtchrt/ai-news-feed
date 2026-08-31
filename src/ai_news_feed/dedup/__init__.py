"""Exact and semantic news deduplication."""

from ai_news_feed.dedup.exact import ExactDeduper
from ai_news_feed.dedup.semantic import SemanticClusterer

__all__ = ["ExactDeduper", "SemanticClusterer"]
