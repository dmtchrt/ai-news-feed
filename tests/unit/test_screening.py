import json
from datetime import UTC, datetime

import pytest

from ai_news_feed.domain.models import Material, NewsCluster
from ai_news_feed.llm.base import LLMRequest, LLMResponse
from ai_news_feed.normalization import content_hash
from ai_news_feed.screening import ClusterScreener, ScreeningThresholds


class _FakeLLM:
    def __init__(self, text: str) -> None:
        self.text = text
        self.requests: list[LLMRequest] = []

    async def complete(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        return LLMResponse(text=self.text, model="fake-cheap-model")


def _material() -> Material:
    text = "Новый фреймворк для автономных агентов."
    return Material(
        id="one",
        source_id="source",
        external_id="1",
        original_url="https://example.test/1",
        published_at=datetime(2026, 8, 31, tzinfo=UTC),
        fetched_at=datetime(2026, 8, 31, tzinfo=UTC),
        title="Фреймворк агентов",
        text=text,
        content_hash=content_hash(text),
    )


@pytest.mark.asyncio
async def test_screening_is_one_call_and_uncertainty_passes() -> None:
    response = json.dumps(
        {
            "relevance_score": 0.3,
            "noise_score": 0.9,
            "uncertain": True,
            "reason": "Недостаточно контекста для уверенного исключения.",
        }
    )
    client = _FakeLLM(response)
    material = _material()
    cluster = NewsCluster(
        id="cluster",
        material_ids=(material.id,),
        representative_id=material.id,
        similarities={material.id: 1.0},
    )

    result = await ClusterScreener(client).screen(
        cluster,
        [material],
        "ИИ-агенты и кейсы внедрения",
    )

    assert len(client.requests) == 1
    assert "ИИ-агенты и кейсы внедрения" in client.requests[0].user_prompt
    assert ScreeningThresholds().accepts(result)


@pytest.mark.asyncio
async def test_screening_prompt_respects_content_budget() -> None:
    response = json.dumps(
        {
            "relevance_score": 0.9,
            "noise_score": 0.1,
            "uncertain": False,
            "reason": "Релевантно.",
        }
    )
    client = _FakeLLM(response)
    material = _material().model_copy(update={"text": "агент " * 10_000})
    cluster = NewsCluster(
        id="cluster",
        material_ids=(material.id,),
        representative_id=material.id,
        similarities={material.id: 1.0},
    )

    await ClusterScreener(client, max_prompt_chars=1_000).screen(
        cluster,
        [material],
        "ИИ-агенты",
    )

    documents = client.requests[0].user_prompt.split("МАТЕРИАЛЫ КЛАСТЕРА:\n", 1)[1]
    assert len(documents) <= 1_000
