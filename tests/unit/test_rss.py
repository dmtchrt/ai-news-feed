from datetime import UTC, datetime
from pathlib import Path

import httpx
import pytest

from ai_news_feed.domain.models import CollectionCursor
from ai_news_feed.extraction import ContentExtractor, ExtractedItem
from ai_news_feed.sources.presets import (
    issek_hse_source,
    tadviser_source,
    telegram_rss_bridge_source,
)
from ai_news_feed.sources.rss import NativeRssConnector, RssBridgeConnector


def _text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


@pytest.mark.asyncio
async def test_tadviser_native_rss_fetches_articles_and_honours_cursor(
    fixture_dir: Path,
) -> None:
    feed = _text(fixture_dir / "rss" / "tadviser.xml")

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/xml/tadviser.xml":
            return httpx.Response(200, text=feed, request=request)
        article_id = request.url.path.rsplit("/", maxsplit=1)[-1]
        return httpx.Response(
            200,
            text=(
                "<html><body><article><h1>Материал</h1>"
                f"<p>Полный текст статьи TAdviser {article_id} об искусственном интеллекте.</p>"
                "</article></body></html>"
            ),
            request=request,
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        connector = NativeRssConnector(client)
        first = await connector.collect(tadviser_source())
        second = await connector.collect(tadviser_source(), first.next_cursor)

    assert [item.external_id for item in first.raw_items] == ["tadviser-100", "tadviser-101"]
    assert first.next_cursor == CollectionCursor(
        published_at=datetime(2026, 8, 31, 5, 15, tzinfo=UTC),
        external_id="tadviser-101",
    )
    assert not second.raw_items
    extracted = ContentExtractor().extract(first.raw_items[-1])
    assert isinstance(extracted, ExtractedItem)
    assert "Полный текст статьи TAdviser 101" in extracted.text


@pytest.mark.asyncio
async def test_hse_verified_feed_url_produces_contract_raw_item(fixture_dir: Path) -> None:
    feed = _text(fixture_dir / "rss" / "hse.xml")

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("news_and_announcements.rss"):
            return httpx.Response(200, text=feed, request=request)
        return httpx.Response(
            200,
            text=(
                "<article><h1>Исследование</h1>"
                "<p>Подробный текст исследования НИУ ВШЭ.</p></article>"
            ),
            request=request,
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        batch = await NativeRssConnector(client).collect(issek_hse_source())

    assert not batch.errors
    assert len(batch.raw_items) == 1
    assert batch.raw_items[0].source_id == "issek-hse"
    assert batch.raw_items[0].original_url == "https://issek.hse.ru/news/100000042.html"
    assert batch.raw_items[0].published_at.tzinfo is UTC


@pytest.mark.asyncio
async def test_rss_bridge_builds_telegram_feed_request(fixture_dir: Path) -> None:
    atom = _text(fixture_dir / "rss" / "telegram_bridge.atom")
    requested_query = ""

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal requested_query
        requested_query = request.url.query.decode()
        return httpx.Response(200, text=atom, request=request)

    config = telegram_rss_bridge_source("@ailev_blog")
    config = config.model_copy(
        update={"settings": {**config.settings, "bridge_base_url": "https://bridge.test/"}}
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        batch = await RssBridgeConnector(client).collect(config)

    assert "bridge=Telegram" in requested_query
    assert "u=ailev_blog" in requested_query
    assert batch.raw_items[0].external_id == "9001"
    assert batch.next_cursor == CollectionCursor(message_id=9001)
    assert "Полный текст поста" in (batch.raw_items[0].raw_html or "")


@pytest.mark.asyncio
async def test_feed_fetch_error_with_empty_str_still_yields_valid_error() -> None:
    """Regression: some httpx errors (e.g. a bare timeout) stringify to "" -- the
    connector must not let that reach CollectionError.message, which requires
    at least 1 character (see sources/_shared.py:error_message)."""

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        connector = NativeRssConnector(client)
        batch = await connector.collect(tadviser_source())

    assert batch.raw_items == ()
    assert len(batch.errors) == 1
    assert batch.errors[0].code == "feed_fetch_failed"
    assert batch.errors[0].message == "ConnectError"
