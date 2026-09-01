"""Thin OpenAI Responses API adapter for the provider-neutral LLMClient protocol."""

from __future__ import annotations

from typing import Any

import httpx
from pydantic import BaseModel, ConfigDict, Field

from ai_news_feed.llm.base import LLMRequest, LLMResponse


class OpenAIResponseError(RuntimeError):
    pass


class _ResponseContentPart(BaseModel):
    model_config = ConfigDict(extra="ignore")

    type: str = Field(min_length=1)
    text: str | None = None
    refusal: str | None = None


class _ResponseOutputItem(BaseModel):
    model_config = ConfigDict(extra="ignore")

    type: str = Field(min_length=1)
    content: tuple[_ResponseContentPart, ...] = ()


class _ResponsesPayload(BaseModel):
    model_config = ConfigDict(extra="ignore")

    model: str = Field(min_length=1)
    output: tuple[_ResponseOutputItem, ...]


def _extract_output_text(payload: _ResponsesPayload) -> str:
    text_parts: list[str] = []
    refusals: list[str] = []

    for item in payload.output:
        if item.type != "message":
            continue
        for part in item.content:
            if part.type == "output_text" and part.text:
                text_parts.append(part.text)
            elif part.type == "refusal":
                refusals.append(part.refusal or "no refusal reason provided")

    if refusals:
        raise OpenAIResponseError(f"Responses API model refusal: {'; '.join(refusals)}")
    if not text_parts:
        raise OpenAIResponseError("Responses API returned no output_text content in a message")
    return "".join(text_parts)


class OpenAIResponsesClient:
    """Use Structured Outputs without exposing provider types to pipeline modules."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        base_url: str = "https://api.openai.com/v1",
        max_output_tokens: int = 2_000,
        timeout_seconds: float = 60.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._api_key = api_key.strip()
        self._model = model.strip()
        self._base_url = base_url.rstrip("/")
        self._max_output_tokens = max_output_tokens
        self._timeout_seconds = timeout_seconds
        self._client = client
        if not self._api_key:
            raise ValueError("api_key must not be blank")
        if not self._model:
            raise ValueError("model must not be blank")
        if max_output_tokens < 1:
            raise ValueError("max_output_tokens must be positive")

    async def complete(self, request: LLMRequest) -> LLMResponse:
        payload: dict[str, Any] = {
            "model": self._model,
            "instructions": request.system_prompt,
            "input": request.user_prompt,
            "max_output_tokens": self._max_output_tokens,
            "store": False,
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "ai_news_feed_response",
                    "strict": True,
                    "schema": request.response_schema,
                }
            },
        }
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        if self._client is not None:
            return await self._complete_with_client(self._client, payload, headers)
        async with httpx.AsyncClient() as client:
            return await self._complete_with_client(client, payload, headers)

    async def _complete_with_client(
        self,
        client: httpx.AsyncClient,
        payload: dict[str, Any],
        headers: dict[str, str],
    ) -> LLMResponse:
        try:
            response = await client.post(
                f"{self._base_url}/responses",
                json=payload,
                headers=headers,
                timeout=self._timeout_seconds,
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise OpenAIResponseError(f"Responses API request failed: {exc}") from exc
        try:
            parsed = _ResponsesPayload.model_validate(response.json())
        except (ValueError, TypeError) as exc:
            raise OpenAIResponseError("Responses API returned an invalid payload") from exc
        return LLMResponse(text=_extract_output_text(parsed), model=parsed.model)
