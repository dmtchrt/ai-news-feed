from datetime import UTC, datetime
from pathlib import Path

import httpx
import pytest

from ai_news_feed.extraction import ContentExtractor, ExtractedItem
from ai_news_feed.sources.presets import ict_moscow_source
from ai_news_feed.sources.site import UniversalSiteConnector, _ensure_utc


@pytest.mark.asyncio
async def test_ict_moscow_preset_discovers_detail_pages_and_extracts_text(
    fixture_dir: Path,
) -> None:
    listing = (fixture_dir / "sites" / "ict_listing.html").read_text(encoding="utf-8")
    adoption = (fixture_dir / "sites" / "ict_ai_adoption.html").read_text(encoding="utf-8")
    open_source = (fixture_dir / "sites" / "ict_open_source.html").read_text(encoding="utf-8")

    def handler(request: httpx.Request) -> httpx.Response:
        pages = {
            "/analytics/": listing,
            "/analytics/ai-adoption-2026": adoption,
            "/analytics/open-source-russia": open_source,
        }
        return httpx.Response(200, text=pages[request.url.path], request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        batch = await UniversalSiteConnector(client).collect(ict_moscow_source())

    assert not batch.errors
    assert len(batch.raw_items) == 2
    assert [item.title for item in batch.raw_items] == [
        "Обзор Open Source в России",
        "59,6% российских компаний используют различные формы ИИ",
    ]
    extracted = ContentExtractor().extract(batch.raw_items[-1])
    assert isinstance(extracted, ExtractedItem)
    assert "3% организаций" in extracted.text


@pytest.mark.asyncio
async def test_universal_scraper_returns_typed_error_when_layout_changes() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="<html><body>No cards</body></html>", request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        batch = await UniversalSiteConnector(client).collect(ict_moscow_source())

    assert batch.raw_items == ()
    assert batch.errors[0].code == "selector_empty"


@pytest.mark.asyncio
async def test_listing_fetch_error_with_empty_str_still_yields_valid_error() -> None:
    """Regression: same empty-message hazard as rss.py, for the listing-page fetch."""

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        batch = await UniversalSiteConnector(client).collect(ict_moscow_source())

    assert batch.raw_items == ()
    assert len(batch.errors) == 1
    assert batch.errors[0].code == "listing_fetch_failed"
    assert batch.errors[0].message == "ConnectError"


def test_ensure_utc_treats_naive_datetime_as_moscow_local_time() -> None:
    """Regression: a scraped date with no explicit UTC offset -- the normal case on
    RU sites, which print local time with no timezone suffix at all -- used to be
    labeled UTC directly instead of converted from Moscow time. For anything
    published 00:00-02:59 MSK that silently rolled the stored date back a day."""
    naive = datetime(2026, 9, 2, 1, 0)
    assert _ensure_utc(naive) == datetime(2026, 9, 1, 22, 0, tzinfo=UTC)


def test_ensure_utc_keeps_an_explicit_offset_unchanged_in_value() -> None:
    aware = datetime(2026, 9, 2, 1, 0, tzinfo=UTC)
    assert _ensure_utc(aware) == aware
