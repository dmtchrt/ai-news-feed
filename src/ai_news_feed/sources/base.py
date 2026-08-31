"""Protocol implemented by every external source adapter."""

from __future__ import annotations

from typing import Protocol

from ai_news_feed.domain.models import CollectionBatch, CollectionCursor, SourceConfig


class SourceConnector(Protocol):
    async def collect(
        self,
        config: SourceConfig,
        cursor: CollectionCursor | None = None,
    ) -> CollectionBatch:
        """Collect items newer than cursor without persisting the returned cursor."""
        ...
