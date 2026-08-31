import json
from datetime import UTC, datetime

import pytest

from ai_news_feed.domain.models import Material, NewsCluster
from ai_news_feed.llm.base import LLMRequest, LLMResponse
from ai_news_feed.llm.parsing import LLMOutputError
from ai_news_feed.normalization import content_hash
from ai_news_feed.summarization import ClusterSummarizer


class _FakeLLM:
    def __init__(self, summary: str) -> None:
        self.summary = summary
        self.calls = 0

    async def complete(self, request: LLMRequest) -> LLMResponse:
        self.calls += 1
        return LLMResponse(text=json.dumps({"summary": self.summary}), model="fake-summary")


def _material(material_id: str) -> Material:
    text = f"Текст новости {material_id}"
    return Material(
        id=material_id,
        source_id=f"source-{material_id}",
        external_id=material_id,
        original_url=f"https://example.test/{material_id}",
        published_at=datetime(2026, 8, 31, tzinfo=UTC),
        fetched_at=datetime(2026, 8, 31, tzinfo=UTC),
        title=f"Новость {material_id}",
        text=text,
        content_hash=content_hash(text),
    )


@pytest.mark.asyncio
async def test_summary_has_every_source_link_added_by_code() -> None:
    materials = [_material("one"), _material("two")]
    cluster = NewsCluster(
        id="cluster",
        material_ids=("one", "two"),
        representative_id="one",
        similarities={"one": 1.0, "two": 0.94},
    )
    client = _FakeLLM("Два источника сообщили о запуске нового AI-инструмента.")

    digest_item = await ClusterSummarizer(client).summarize(cluster, materials)

    assert client.calls == 1
    assert digest_item.source_links == (
        "https://example.test/one",
        "https://example.test/two",
    )


@pytest.mark.asyncio
async def test_summary_rejects_model_generated_link() -> None:
    material = _material("one")
    cluster = NewsCluster(
        id="cluster",
        material_ids=("one",),
        representative_id="one",
        similarities={"one": 1.0},
    )

    with pytest.raises(LLMOutputError):
        await ClusterSummarizer(_FakeLLM("Читайте https://hallucinated.test")).summarize(
            cluster,
            [material],
        )
