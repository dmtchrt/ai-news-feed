from collections.abc import Sequence
from datetime import UTC, datetime

import numpy as np

from ai_news_feed.dedup.semantic import SemanticClusterer
from ai_news_feed.domain.models import DuplicateKind, Material
from ai_news_feed.normalization import content_hash


class _KeywordEncoder:
    def encode(self, sentences: Sequence[str], **kwargs: object) -> np.ndarray:
        vectors = []
        for sentence in sentences:
            if "агент" in sentence.casefold():
                vectors.append([1.0, 0.0])
            else:
                vectors.append([0.0, 1.0])
        return np.asarray(vectors, dtype=np.float32)


def _material(material_id: str, title: str, text: str, *, hour: int) -> Material:
    return Material(
        id=material_id,
        source_id=f"source-{material_id}",
        external_id=material_id,
        original_url=f"https://example.test/{material_id}",
        published_at=datetime(2026, 8, 31, hour, tzinfo=UTC),
        fetched_at=datetime(2026, 8, 31, 10, tzinfo=UTC),
        title=title,
        text=text,
        content_hash=content_hash(text),
    )


def test_semhash_clusters_new_materials_with_lookback() -> None:
    historical = _material("old", "Релиз AI-агента", "Вчера вышел новый продукт", hour=1)
    paraphrase = _material(
        "new",
        "Компания представила автономного агента",
        "Опубликован новый инструмент",
        hour=2,
    )
    unrelated = _material("other", "Отчёт о чипах", "Продажи ускорителей выросли", hour=3)
    clusterer = SemanticClusterer(threshold=0.9, encoder=_KeywordEncoder())

    batch = clusterer.cluster([paraphrase, unrelated], [historical])

    story_cluster = next(cluster for cluster in batch.clusters if "new" in cluster.material_ids)
    assert story_cluster.representative_id == "old"
    assert set(story_cluster.material_ids) == {"old", "new"}
    assert story_cluster.similarities["new"] >= 0.9
    assert any(cluster.material_ids == ("other",) for cluster in batch.clusters)
    assert batch.duplicate_links[0].kind is DuplicateKind.SEMANTIC
