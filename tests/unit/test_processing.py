import json
from collections.abc import Sequence
from datetime import UTC, datetime

import numpy as np
import pytest

from ai_news_feed.dedup.semantic import SemanticClusterer
from ai_news_feed.domain.models import InterestProfile, RawItem
from ai_news_feed.extraction.models import ExtractedItem
from ai_news_feed.llm.base import LLMRequest, LLMResponse
from ai_news_feed.processing import AIProcessor, ExtractedRawItem
from ai_news_feed.screening import ClusterScreener
from ai_news_feed.storage.memory import InMemoryRepository
from ai_news_feed.summarization import ClusterSummarizer


class _ConstantEncoder:
    def encode(self, sentences: Sequence[str], **kwargs: object) -> np.ndarray:
        return np.ones((len(sentences), 2), dtype=np.float32)


class _QueuedLLM:
    def __init__(self, *, accepted: bool = True) -> None:
        self.requests: list[LLMRequest] = []
        self.accepted = accepted

    async def complete(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        if len(self.requests) == 1:
            payload = {
                "relevance_score": 0.95 if self.accepted else 0.1,
                "noise_score": 0.05 if self.accepted else 0.95,
                "uncertain": False,
                "reason": "Результат классификации.",
            }
            return LLMResponse(text=json.dumps(payload), model="fake-screen")
        return LLMResponse(
            text=json.dumps({"summary": "Вышел новый фреймворк автономных агентов."}),
            model="fake-summary",
        )


def _raw(source: str, external_id: str) -> RawItem:
    return RawItem(
        source_id=source,
        external_id=external_id,
        original_url=f"https://{source}.test/{external_id}",
        published_at=datetime(2026, 8, 31, 6, tzinfo=UTC),
        title="Новый агентный фреймворк",
    )


@pytest.mark.asyncio
async def test_ai_processor_uses_fake_repository_and_preserves_exact_duplicate_links() -> None:
    now = datetime(2026, 8, 31, 8, tzinfo=UTC)
    profile = InterestProfile(
        id="default",
        name="Интересы v1",
        description="ИИ-агенты и автономные агенты; рынок ИИ в России.",
        created_at=now,
        updated_at=now,
    )
    repository = InMemoryRepository(interest_profiles=[profile])
    llm = _QueuedLLM()
    processor = AIProcessor(
        repository=repository,
        semantic_clusterer=SemanticClusterer(encoder=_ConstantEncoder()),
        screener=ClusterScreener(llm),
        summarizer=ClusterSummarizer(llm),
    )
    extracted_text = "Компания выпустила новый фреймворк автономных ИИ-агентов."
    items = [
        ExtractedRawItem(_raw("first", "1"), ExtractedItem(text=extracted_text)),
        ExtractedRawItem(_raw("second", "2"), ExtractedItem(text=extracted_text)),
    ]

    result = await processor.process(items, interest_profile_id="default", now=now)

    assert len(result.exact_deduplication.unique_materials) == 1
    assert len(result.cluster_batch.clusters) == 1
    assert len(result.screening_results) == 1
    assert len(result.digest_items) == 1
    assert result.digest_items[0].source_links == (
        "https://first.test/1",
        "https://second.test/2",
    )
    assert len(llm.requests) == 2


@pytest.mark.asyncio
async def test_ai_processor_does_not_summarize_rejected_cluster() -> None:
    now = datetime(2026, 8, 31, 8, tzinfo=UTC)
    profile = InterestProfile(
        id="default",
        name="Интересы v1",
        description="ИИ-агенты",
        created_at=now,
        updated_at=now,
    )
    llm = _QueuedLLM(accepted=False)
    processor = AIProcessor(
        repository=InMemoryRepository(interest_profiles=[profile]),
        semantic_clusterer=SemanticClusterer(encoder=_ConstantEncoder()),
        screener=ClusterScreener(llm),
        summarizer=ClusterSummarizer(llm),
    )
    item = ExtractedRawItem(
        _raw("source", "1"),
        ExtractedItem(text="Нерелевантная рекламная публикация."),
    )

    result = await processor.process([item], interest_profile_id="default", now=now)

    assert not result.digest_items
    assert len(llm.requests) == 1
