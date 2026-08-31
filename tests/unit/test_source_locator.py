import pytest

from ai_news_feed.domain.models import CollectorKind, SourceKind
from ai_news_feed.sources.locator import normalize_source_locator, parse_source_locator


def test_telegram_locator_has_one_case_insensitive_identity() -> None:
    parsed = parse_source_locator("https://t.me/s/AiLev_Blog/")

    assert parsed.kind is SourceKind.TELEGRAM
    assert parsed.collector is CollectorKind.WEB_PREVIEW
    assert parsed.locator == "@ailev_blog"
    assert normalize_source_locator("@AILEV_BLOG", SourceKind.TELEGRAM) == "@ailev_blog"


def test_website_locator_removes_tracking_and_detects_rss() -> None:
    parsed = parse_source_locator("HTTPS://Example.Test/feed.xml/?utm_source=bot")

    assert parsed.kind is SourceKind.WEBSITE
    assert parsed.collector is CollectorKind.NATIVE_RSS
    assert parsed.normalized_locator == "https://example.test/feed.xml"


def test_source_locator_rejects_non_http_input() -> None:
    with pytest.raises(ValueError, match="http"):
        parse_source_locator("example.test/feed")
