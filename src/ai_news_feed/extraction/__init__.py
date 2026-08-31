"""Content extraction for HTML and downloaded documents."""

from ai_news_feed.extraction.content import ContentExtractor
from ai_news_feed.extraction.models import ExtractedItem, ExtractionFailure

__all__ = ["ContentExtractor", "ExtractedItem", "ExtractionFailure"]
