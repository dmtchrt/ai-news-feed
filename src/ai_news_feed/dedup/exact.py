"""Cheap content-hash deduplication before embeddings."""

from __future__ import annotations

from collections.abc import Sequence

from ai_news_feed.domain.models import (
    DuplicateKind,
    DuplicateLink,
    ExactDeduplicationResult,
    Material,
)


class ExactDeduper:
    def deduplicate(
        self,
        materials: Sequence[Material],
        existing_materials: Sequence[Material] = (),
    ) -> ExactDeduplicationResult:
        existing_by_hash: dict[str, Material] = {}
        for material in sorted(existing_materials, key=_material_order):
            existing_by_hash.setdefault(material.content_hash, material)

        unique: list[Material] = []
        duplicate_links: list[DuplicateLink] = []
        selected_by_hash = dict(existing_by_hash)
        seen_ids: set[str] = {material.id for material in existing_materials}

        for material in materials:
            if material.id in seen_ids:
                raise ValueError(f"material id occurs more than once: {material.id}")
            seen_ids.add(material.id)
            representative = selected_by_hash.get(material.content_hash)
            if representative is None:
                selected_by_hash[material.content_hash] = material
                unique.append(material)
                continue
            duplicate_links.append(
                DuplicateLink(
                    material_id=material.id,
                    duplicate_of_id=representative.id,
                    kind=DuplicateKind.EXACT_CONTENT,
                    similarity=1.0,
                )
            )

        return ExactDeduplicationResult(
            unique_materials=tuple(unique),
            duplicate_links=tuple(duplicate_links),
        )


def _material_order(material: Material) -> tuple[object, ...]:
    return (material.published_at, material.id)
