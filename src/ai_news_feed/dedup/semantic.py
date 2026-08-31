"""Turn SemHash duplicate groups into explicit, attributable news clusters."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from typing import Protocol, cast

import numpy as np
from model2vec import StaticModel
from semhash import SemHash
from vicinity import Backend

from ai_news_feed.domain.models import (
    ClusterBatch,
    DuplicateKind,
    DuplicateLink,
    Material,
    NewsCluster,
)

DEFAULT_MODEL = "minishlab/potion-multilingual-128M"


class Encoder(Protocol):
    def encode(self, sentences: Sequence[str], **kwargs: object) -> np.ndarray: ...


class SemanticClusterer:
    """Cluster current materials together with a persisted lookback window."""

    def __init__(
        self,
        *,
        threshold: float = 0.9,
        model_name: str = DEFAULT_MODEL,
        encoder: Encoder | None = None,
        ann_backend: Backend | str = Backend.BASIC,
        max_text_chars: int = 4_000,
    ) -> None:
        if not 0.0 < threshold <= 1.0:
            raise ValueError("threshold must be in range (0, 1]")
        if max_text_chars < 100:
            raise ValueError("max_text_chars must be at least 100")
        self.threshold = threshold
        self.model_name = model_name
        self._encoder = encoder
        self.ann_backend = ann_backend
        self.max_text_chars = max_text_chars

    def cluster(
        self,
        materials: Sequence[Material],
        lookback_materials: Sequence[Material] = (),
    ) -> ClusterBatch:
        new_ids = {material.id for material in materials}
        if len(new_ids) != len(materials):
            raise ValueError("new material ids must be unique")

        combined = _distinct_materials((*lookback_materials, *materials))
        if not combined or not new_ids:
            return ClusterBatch()

        records = [
            {"material_id": material.id, "dedup_text": self._dedup_text(material)}
            for material in combined
        ]
        semhash = SemHash.from_records(
            records=records,
            columns=["dedup_text"],
            model=self._get_encoder(),
            ann_backend=self.ann_backend,
        )
        result = semhash.self_deduplicate(threshold=self.threshold)

        clusters: list[NewsCluster] = []
        links: list[DuplicateLink] = []
        for selected in result.selected_with_duplicates:
            representative_id = str(selected.record["material_id"])
            member_scores = {representative_id: 1.0}
            for duplicate, score in selected.duplicates:
                duplicate_id = str(duplicate["material_id"])
                member_scores[duplicate_id] = max(
                    float(score),
                    member_scores.get(duplicate_id, 0.0),
                )
            if not new_ids.intersection(member_scores):
                continue

            material_ids = (
                representative_id,
                *(
                    sorted(
                        member_id for member_id in member_scores if member_id != representative_id
                    )
                ),
            )
            clusters.append(
                NewsCluster(
                    id=cluster_id(material_ids),
                    material_ids=material_ids,
                    representative_id=representative_id,
                    similarities={
                        member_id: member_scores[member_id] for member_id in material_ids
                    },
                )
            )
            links.extend(
                DuplicateLink(
                    material_id=member_id,
                    duplicate_of_id=representative_id,
                    kind=DuplicateKind.SEMANTIC,
                    similarity=member_scores[member_id],
                )
                for member_id in material_ids
                if member_id != representative_id
            )

        return ClusterBatch(clusters=tuple(clusters), duplicate_links=tuple(links))

    def _get_encoder(self) -> Encoder:
        if self._encoder is None:
            self._encoder = cast(Encoder, StaticModel.from_pretrained(self.model_name))
        return self._encoder

    def _dedup_text(self, material: Material) -> str:
        title = material.title.strip()
        text = material.text[: self.max_text_chars].strip()
        return f"{title}\n{text}"


def _distinct_materials(materials: Sequence[Material]) -> tuple[Material, ...]:
    by_id: dict[str, Material] = {}
    for material in materials:
        previous = by_id.get(material.id)
        if previous is not None and previous != material:
            raise ValueError(f"different materials share id: {material.id}")
        by_id.setdefault(material.id, material)
    return tuple(by_id.values())


def cluster_id(material_ids: Sequence[str]) -> str:
    payload = "\x1f".join(sorted(material_ids)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
