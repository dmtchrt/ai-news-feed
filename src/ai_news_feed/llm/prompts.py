"""Deterministic cluster serialization with a bounded prompt budget."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence

from ai_news_feed.domain.models import Material, NewsCluster


def cluster_documents(
    cluster: NewsCluster,
    materials: Sequence[Material],
    *,
    max_total_chars: int,
) -> str:
    if max_total_chars < 1_000:
        raise ValueError("max_total_chars must be at least 1000")
    material_by_id = _material_map(materials)
    missing_ids = set(cluster.material_ids).difference(material_by_id)
    if missing_ids:
        raise ValueError(f"cluster materials are missing: {sorted(missing_ids)}")

    documents: list[dict[str, str]] = []
    for material_id in cluster.material_ids:
        material = material_by_id[material_id]
        documents.append(
            {
                "material_id": material.id,
                "title": material.title[:300],
                "text": "",
            }
        )
    base = _serialize(documents)
    available = max_total_chars - len(base)
    if available < len(documents):
        raise ValueError("cluster metadata exceeds the prompt budget")

    per_material = available // len(documents)
    for document, material_id in zip(documents, cluster.material_ids, strict=True):
        document["text"] = material_by_id[material_id].text[:per_material]
    serialized = _serialize(documents)
    while len(serialized) > max_total_chars:
        overflow_per_material = max(1, (len(serialized) - max_total_chars) // len(documents) + 1)
        for document in documents:
            document["text"] = document["text"][:-overflow_per_material]
        serialized = _serialize(documents)
    return serialized


def ordered_cluster_materials(
    cluster: NewsCluster,
    materials: Sequence[Material],
) -> tuple[Material, ...]:
    material_by_id = _material_map(materials)
    missing_ids = set(cluster.material_ids).difference(material_by_id)
    if missing_ids:
        raise ValueError(f"cluster materials are missing: {sorted(missing_ids)}")
    return tuple(material_by_id[material_id] for material_id in cluster.material_ids)


def _material_map(materials: Sequence[Material]) -> Mapping[str, Material]:
    result: dict[str, Material] = {}
    for material in materials:
        previous = result.get(material.id)
        if previous is not None and previous != material:
            raise ValueError(f"different materials share id: {material.id}")
        result[material.id] = material
    return result


def _serialize(documents: Sequence[Mapping[str, str]]) -> str:
    return json.dumps(documents, ensure_ascii=False, separators=(",", ":"))
