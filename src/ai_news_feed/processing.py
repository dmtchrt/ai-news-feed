"""AI/ML slice orchestration for pipeline steps 4 through 8."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from ai_news_feed.dedup.exact import ExactDeduper
from ai_news_feed.dedup.semantic import SemanticClusterer, cluster_id
from ai_news_feed.domain.models import (
    ClusterBatch,
    DigestItem,
    DuplicateLink,
    ExactDeduplicationResult,
    Material,
    NewsCluster,
    RawItem,
    ScreeningResult,
)
from ai_news_feed.extraction.models import ExtractedItem
from ai_news_feed.normalization import Normalizer
from ai_news_feed.screening import ClusterScreener, ScreeningThresholds
from ai_news_feed.storage.base import Repository
from ai_news_feed.summarization import ClusterSummarizer


@dataclass(frozen=True)
class ExtractedRawItem:
    raw_item: RawItem
    extracted_item: ExtractedItem


@dataclass(frozen=True)
class AIProcessingResult:
    materials: tuple[Material, ...]
    exact_deduplication: ExactDeduplicationResult
    cluster_batch: ClusterBatch
    screening_results: tuple[ScreeningResult, ...]
    digest_items: tuple[DigestItem, ...]


class AIProcessor:
    """Compose pure modules while leaving persistence/transactions to step 11."""

    def __init__(
        self,
        *,
        repository: Repository,
        semantic_clusterer: SemanticClusterer,
        screener: ClusterScreener,
        summarizer: ClusterSummarizer,
        normalizer: Normalizer | None = None,
        exact_deduper: ExactDeduper | None = None,
        screening_thresholds: ScreeningThresholds | None = None,
        lookback: timedelta = timedelta(days=7),
    ) -> None:
        if lookback <= timedelta(0):
            raise ValueError("lookback must be positive")
        self.repository = repository
        self.normalizer = normalizer or Normalizer()
        self.exact_deduper = exact_deduper or ExactDeduper()
        self.semantic_clusterer = semantic_clusterer
        self.screener = screener
        self.summarizer = summarizer
        self.screening_thresholds = screening_thresholds or ScreeningThresholds()
        self.lookback = lookback

    async def process(
        self,
        items: Sequence[ExtractedRawItem],
        *,
        interest_profile_id: str,
        now: datetime | None = None,
    ) -> AIProcessingResult:
        run_time = now or datetime.now(UTC)
        if run_time.tzinfo is None or run_time.utcoffset() is None:
            raise ValueError("now must be timezone-aware")
        run_time = run_time.astimezone(UTC)
        profile = await self.repository.get_interest_profile(interest_profile_id)

        materials = tuple(
            self.normalizer.normalize(
                item.raw_item,
                item.extracted_item,
                fetched_at=run_time,
            )
            for item in items
        )
        existing_exact = await self.repository.find_materials_by_content_hashes(
            {material.content_hash for material in materials}
        )
        exact_result = self.exact_deduper.deduplicate(materials, existing_exact)
        lookback_materials = await self.repository.list_materials_since(run_time - self.lookback)
        semantic_batch = self.semantic_clusterer.cluster(
            exact_result.unique_materials,
            lookback_materials,
        )
        cluster_batch = _attach_exact_duplicates(
            semantic_batch,
            exact_result.duplicate_links,
        )

        material_by_id = {material.id: material for material in lookback_materials}
        material_by_id.update((material.id, material) for material in materials)
        screenings: list[ScreeningResult] = []
        digest_items: list[DigestItem] = []
        for cluster in cluster_batch.clusters:
            cluster_materials = tuple(material_by_id[item_id] for item_id in cluster.material_ids)
            screening = await self.screener.screen(
                cluster,
                cluster_materials,
                profile.description,
            )
            screenings.append(screening)
            if self.screening_thresholds.accepts(screening):
                digest_items.append(await self.summarizer.summarize(cluster, cluster_materials))

        return AIProcessingResult(
            materials=materials,
            exact_deduplication=exact_result,
            cluster_batch=cluster_batch,
            screening_results=tuple(screenings),
            digest_items=tuple(digest_items),
        )


def _attach_exact_duplicates(
    semantic_batch: ClusterBatch,
    exact_links: Sequence[DuplicateLink],
) -> ClusterBatch:
    links_by_target: dict[str, list[DuplicateLink]] = {}
    for link in exact_links:
        links_by_target.setdefault(link.duplicate_of_id, []).append(link)

    clusters: list[NewsCluster] = []
    for cluster in semantic_batch.clusters:
        material_ids = list(cluster.material_ids)
        similarities = dict(cluster.similarities)
        for target_id in cluster.material_ids:
            for link in links_by_target.get(target_id, ()):
                if link.material_id not in similarities:
                    material_ids.append(link.material_id)
                    similarities[link.material_id] = 1.0
        clusters.append(
            NewsCluster(
                id=cluster_id(material_ids),
                material_ids=tuple(material_ids),
                representative_id=cluster.representative_id,
                similarities=similarities,
            )
        )
    return ClusterBatch(
        clusters=tuple(clusters),
        duplicate_links=tuple((*exact_links, *semantic_batch.duplicate_links)),
    )
