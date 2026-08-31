"""Minimal LLM protocol; provider adapters belong outside domain logic."""

from __future__ import annotations

from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field, JsonValue


class LLMModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class LLMRequest(LLMModel):
    system_prompt: str = Field(min_length=1)
    user_prompt: str = Field(min_length=1)
    response_schema: dict[str, JsonValue]


class LLMResponse(LLMModel):
    text: str = Field(min_length=1)
    model: str = Field(min_length=1)


class LLMClient(Protocol):
    async def complete(self, request: LLMRequest) -> LLMResponse: ...
