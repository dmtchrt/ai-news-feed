import json

import httpx
import pytest

from ai_news_feed.llm.base import LLMRequest
from ai_news_feed.llm.openai import OpenAIResponsesClient


@pytest.mark.asyncio
async def test_responses_client_requests_strict_structured_output() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        assert request.headers["Authorization"] == "Bearer test-key"
        return httpx.Response(
            200,
            json={"model": "test-model-2026", "output_text": '{"answer":"ok"}'},
            request=request,
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        client = OpenAIResponsesClient(
            api_key="test-key",
            model="test-model",
            base_url="https://api.test/v1",
            client=http_client,
        )
        response = await client.complete(
            LLMRequest(
                system_prompt="Return JSON.",
                user_prompt="Classify this.",
                response_schema={
                    "type": "object",
                    "properties": {"answer": {"type": "string"}},
                    "required": ["answer"],
                    "additionalProperties": False,
                },
            )
        )

    assert response.text == '{"answer":"ok"}'
    assert captured["store"] is False
    text_config = captured["text"]
    assert isinstance(text_config, dict)
    assert text_config["format"]["type"] == "json_schema"
    assert text_config["format"]["strict"] is True
