"""Strict JSON parsing shared by screening and summarization."""

from __future__ import annotations

import json
import re

from pydantic import BaseModel, ValidationError

_JSON_FENCE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL | re.IGNORECASE)


class LLMOutputError(ValueError):
    """The provider returned an answer that violates the requested schema."""


def parse_json_response[Payload: BaseModel](
    text: str,
    payload_type: type[Payload],
) -> Payload:
    candidate = text.strip()
    fence = _JSON_FENCE.fullmatch(candidate)
    if fence:
        candidate = fence.group(1).strip()
    try:
        value = json.loads(candidate)
        return payload_type.model_validate(value)
    except (json.JSONDecodeError, ValidationError) as exc:
        raise LLMOutputError(f"invalid LLM JSON response: {exc}") from exc
