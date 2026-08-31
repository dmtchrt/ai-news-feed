"""In-memory repository for unit/integration tests; no Postgres dependency."""

from __future__ import annotations

from collections.abc import Collection, Iterable
from datetime import datetime

from ai_news_feed.domain.models import InterestProfile, Material


class InMemoryRepository:
    def __init__(
        self,
        *,
        interest_profiles: Iterable[InterestProfile] = (),
        materials: Iterable[Material] = (),
    ) -> None:
        self._profiles = {profile.id: profile for profile in interest_profiles}
        self._materials = {material.id: material for material in materials}

    async def get_interest_profile(self, profile_id: str) -> InterestProfile:
        try:
            profile = self._profiles[profile_id]
        except KeyError as exc:
            raise LookupError(f"interest profile not found: {profile_id}") from exc
        if not profile.enabled:
            raise LookupError(f"interest profile is disabled: {profile_id}")
        return profile

    async def find_materials_by_content_hashes(
        self,
        content_hashes: Collection[str],
    ) -> tuple[Material, ...]:
        requested = set(content_hashes)
        return tuple(
            sorted(
                (
                    material
                    for material in self._materials.values()
                    if material.content_hash in requested
                ),
                key=_material_order,
            )
        )

    async def list_materials_since(self, published_since: datetime) -> tuple[Material, ...]:
        if published_since.tzinfo is None or published_since.utcoffset() is None:
            raise ValueError("published_since must be timezone-aware")
        return tuple(
            sorted(
                (
                    material
                    for material in self._materials.values()
                    if material.published_at >= published_since
                ),
                key=_material_order,
            )
        )

    def add_materials(self, materials: Iterable[Material]) -> None:
        for material in materials:
            self._materials[material.id] = material

    def put_interest_profile(self, profile: InterestProfile) -> None:
        self._profiles[profile.id] = profile


def _material_order(material: Material) -> tuple[object, ...]:
    return (material.published_at, material.id)
