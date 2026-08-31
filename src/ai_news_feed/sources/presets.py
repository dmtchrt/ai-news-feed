"""Verified Phase 0/1 source presets; persistence may assign different ids."""

from __future__ import annotations

from ai_news_feed.domain.models import CollectorKind, SourceConfig, SourceKind

TADVISER_RSS_URL = "https://www.tadviser.ru/xml/tadviser.xml"
ISSEK_HSE_RSS_URL = "https://www.hse.ru/rss/orgs/70333/news_and_announcements.rss"
ICT_MOSCOW_AI_ANALYTICS_URL = "https://ict.moscow/analytics/?tags=искусственный_интеллект&year=2026"


def tadviser_source(source_id: str = "tadviser") -> SourceConfig:
    return SourceConfig(
        id=source_id,
        kind=SourceKind.WEBSITE,
        locator=TADVISER_RSS_URL,
        collector=CollectorKind.NATIVE_RSS,
        settings={"fetch_full_text": True, "max_items": 50},
    )


def issek_hse_source(source_id: str = "issek-hse") -> SourceConfig:
    return SourceConfig(
        id=source_id,
        kind=SourceKind.WEBSITE,
        locator=ISSEK_HSE_RSS_URL,
        collector=CollectorKind.NATIVE_RSS,
        settings={"fetch_full_text": True, "max_items": 50},
    )


def ict_moscow_source(source_id: str = "ict-moscow-analytics") -> SourceConfig:
    return SourceConfig(
        id=source_id,
        kind=SourceKind.WEBSITE,
        locator=ICT_MOSCOW_AI_ANALYTICS_URL,
        collector=CollectorKind.UNIVERSAL_SCRAPER,
        settings={
            "link_selector": 'a[href^="/analytics/"]',
            "include_url_pattern": r"^https://ict\.moscow/analytics/(?!\?|$).+",
            "exclude_url_pattern": r"/(?:authors|companies|tags|themes)/",
            "detail_title_selector": "h1",
            "max_items": 30,
        },
    )


def telegram_preview_source(handle: str, source_id: str | None = None) -> SourceConfig:
    normalized = handle.removeprefix("@").lower()
    return SourceConfig(
        id=source_id or f"telegram-{normalized}",
        kind=SourceKind.TELEGRAM,
        locator=f"@{normalized}",
        collector=CollectorKind.WEB_PREVIEW,
        settings={"max_items": 50},
    )


def telegram_rss_bridge_source(handle: str, source_id: str | None = None) -> SourceConfig:
    normalized = handle.removeprefix("@").lower()
    return SourceConfig(
        id=source_id or f"telegram-{normalized}",
        kind=SourceKind.TELEGRAM,
        locator=f"@{normalized}",
        collector=CollectorKind.RSS_BRIDGE,
        settings={"max_items": 50},
    )


def expertosphere_source(source_id: str = "telegram-expertosphere") -> SourceConfig:
    return SourceConfig(
        id=source_id,
        kind=SourceKind.TELEGRAM,
        locator="@expertosphere",
        collector=CollectorKind.TELETHON,
        settings={"max_items": 100, "download_documents": True},
    )
