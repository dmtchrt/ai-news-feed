"""Repository subset required by the AI/ML pipeline."""

from __future__ import annotations

from collections.abc import Collection
from datetime import datetime
from typing import Protocol

from ai_news_feed.domain.models import InterestProfile, Material


class Repository(Protocol):
    async def get_interest_profile(self, profile_id: str) -> InterestProfile: ...

    async def find_materials_by_content_hashes(
        self,
        content_hashes: Collection[str],
    ) -> tuple[Material, ...]: ...

    async def list_materials_since(self, published_since: datetime) -> tuple[Material, ...]: ...
