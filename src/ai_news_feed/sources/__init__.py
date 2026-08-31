"""Source connectors implementing the common collection protocol."""

from ai_news_feed.sources.base import SourceConnector
from ai_news_feed.sources.rss import NativeRssConnector, RssBridgeConnector
from ai_news_feed.sources.site import UniversalSiteConnector
from ai_news_feed.sources.telegram import TelegramWebPreviewConnector, TelethonConnector

__all__ = [
    "NativeRssConnector",
    "RssBridgeConnector",
    "SourceConnector",
    "TelegramWebPreviewConnector",
    "TelethonConnector",
    "UniversalSiteConnector",
]
