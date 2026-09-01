"""Manual smoke test for the OpenAI Responses API adapter -- NOT part of pytest/CI.

Makes exactly one real, billed call to the real OpenAI API using the project's own
OpenAIResponsesClient, to confirm response parsing (fixed in commit 4b8a927) actually
works against the live API, not just against hand-written mocks in tests. Costs a
fraction of a cent with a small model and max_output_tokens=200.

Usage:
    OPENAI_API_KEY=sk-... uv run python scripts/smoke_test_openai.py <model-id>

<model-id> must be a real model id your account currently has access to (check your
OpenAI dashboard) -- this script does not guess or default one, for the same reason
OPENAI_SCREENING_MODEL/OPENAI_SUMMARY_MODEL have no default in the app itself.
"""

from __future__ import annotations

import asyncio
import os
import sys

from ai_news_feed.llm.base import LLMRequest
from ai_news_feed.llm.openai import OpenAIResponseError, OpenAIResponsesClient

SCHEMA = {
    "type": "object",
    "properties": {
        "ok": {"type": "boolean"},
        "note": {"type": "string"},
    },
    "required": ["ok", "note"],
    "additionalProperties": False,
}


async def main() -> int:
    if len(sys.argv) != 2:
        print("usage: OPENAI_API_KEY=... uv run python scripts/smoke_test_openai.py <model-id>")
        return 2
    model = sys.argv[1]
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        print("OPENAI_API_KEY is not set.")
        return 2

    client = OpenAIResponsesClient(api_key=api_key, model=model, max_output_tokens=200)
    request = LLMRequest(
        system_prompt="Reply with strict JSON matching the schema. No extra commentary.",
        user_prompt="Set ok to true and note to a short greeting in Russian.",
        response_schema=SCHEMA,
    )
    try:
        response = await client.complete(request)
    except OpenAIResponseError as exc:
        print(f"FAIL: {exc}")
        return 1

    print(f"PASS: model={response.model!r}")
    print(f"      text={response.text!r}")
    print("Responses API parsing works against the real API.")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
