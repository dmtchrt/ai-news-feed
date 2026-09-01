import json

import httpx
import pytest

from ai_news_feed.llm.base import LLMRequest
from ai_news_feed.llm.openai import OpenAIResponseError, OpenAIResponsesClient


def _request() -> LLMRequest:
    return LLMRequest(
        system_prompt="Return JSON.",
        user_prompt="Classify this.",
        response_schema={
            "type": "object",
            "properties": {"answer": {"type": "string"}},
            "required": ["answer"],
            "additionalProperties": False,
        },
    )


@pytest.mark.asyncio
async def test_responses_client_requests_strict_structured_output() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        assert request.headers["Authorization"] == "Bearer test-key"
        return httpx.Response(
            200,
            json={
                "model": "test-model-2026",
                "output": [
                    {"id": "rs_1", "type": "reasoning", "summary": []},
                    {
                        "id": "msg_1",
                        "type": "message",
                        "role": "assistant",
                        "status": "completed",
                        "content": [
                            {
                                "type": "output_text",
                                "text": '{"answer":',
                                "annotations": [],
                            },
                            {
                                "type": "output_text",
                                "text": '"ok"}',
                                "annotations": [],
                            },
                        ],
                    },
                ],
            },
            request=request,
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        client = OpenAIResponsesClient(
            api_key="test-key",
            model="test-model",
            base_url="https://api.test/v1",
            client=http_client,
        )
        response = await client.complete(_request())

    assert response.text == '{"answer":"ok"}'
    assert captured["store"] is False
    text_config = captured["text"]
    assert isinstance(text_config, dict)
    assert text_config["format"]["type"] == "json_schema"
    assert text_config["format"]["strict"] is True


@pytest.mark.asyncio
async def test_responses_client_raises_on_refusal() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "model": "test-model-2026",
                "output": [
                    {
                        "id": "msg_1",
                        "type": "message",
                        "role": "assistant",
                        "status": "completed",
                        "content": [
                            {
                                "type": "refusal",
                                "refusal": "Unable to comply with this request.",
                            }
                        ],
                    }
                ],
            },
            request=request,
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        client = OpenAIResponsesClient(
            api_key="test-key",
            model="test-model",
            base_url="https://api.test/v1",
            client=http_client,
        )
        with pytest.raises(OpenAIResponseError, match=r"model refusal.*Unable to comply"):
            await client.complete(_request())


@pytest.mark.asyncio
async def test_responses_client_raises_when_output_has_no_message_text() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "model": "test-model-2026",
                "output": [
                    {"id": "rs_1", "type": "reasoning", "summary": []},
                    {
                        "id": "fc_1",
                        "type": "function_call",
                        "name": "unexpected_tool",
                        "arguments": "{}",
                    },
                ],
            },
            request=request,
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        client = OpenAIResponsesClient(
            api_key="test-key",
            model="test-model",
            base_url="https://api.test/v1",
            client=http_client,
        )
        with pytest.raises(OpenAIResponseError, match="no output_text content"):
            await client.complete(_request())
