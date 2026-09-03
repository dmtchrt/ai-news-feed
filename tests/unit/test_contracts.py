from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from ai_news_feed.domain.models import (
    CollectionBatch,
    CollectionCursor,
    CollectorKind,
    DigestSendTime,
    RawItem,
    SourceConfig,
    SourceKind,
)
from ai_news_feed.sources.base import SourceConnector


def test_raw_item_rejects_naive_datetime_and_extra_fields() -> None:
    with pytest.raises(ValidationError):
        RawItem.model_validate(
            {
                "source_id": "source",
                "external_id": "42",
                "original_url": "https://example.test/42",
                "published_at": datetime(2026, 8, 31),
                "unexpected": True,
            }
        )


def test_cursor_requires_complete_rss_position() -> None:
    with pytest.raises(ValidationError):
        CollectionCursor(published_at=datetime(2026, 8, 31, tzinfo=UTC))

    with pytest.raises(ValidationError):
        CollectionCursor(
            published_at=datetime(2026, 8, 31, tzinfo=UTC),
            external_id="42",
            message_id=42,
        )


def test_raw_item_requires_absolute_http_original_url() -> None:
    with pytest.raises(ValidationError):
        RawItem(
            source_id="source",
            external_id="42",
            original_url="/relative/42",
            published_at=datetime(2026, 8, 31, tzinfo=UTC),
        )


def test_explicit_cursor_is_json_serializable() -> None:
    cursor = CollectionCursor(message_id=42)
    assert cursor.model_dump(mode="json") == {
        "published_at": None,
        "external_id": None,
        "message_id": 42,
    }


def test_digest_send_time_validates_weekday_and_hour() -> None:
    assert DigestSendTime(hour=9).model_dump(mode="json") == {
        "weekday": None,
        "hour": 9,
    }
    assert DigestSendTime(weekday=1, hour=23).weekday == 1

    with pytest.raises(ValidationError):
        DigestSendTime(weekday=7, hour=9)
    with pytest.raises(ValidationError):
        DigestSendTime(weekday=1, hour=24)


class _ExampleConnector:
    async def collect(
        self,
        config: SourceConfig,
        cursor: CollectionCursor | None = None,
    ) -> CollectionBatch:
        return CollectionBatch(next_cursor=cursor)


def _accepts_source_connector(connector: SourceConnector) -> SourceConnector:
    return connector


def test_protocol_is_structural() -> None:
    connector = _accepts_source_connector(_ExampleConnector())
    assert connector is not None
    assert SourceConfig(
        id="source",
        kind=SourceKind.WEBSITE,
        locator="https://example.test/feed.xml",
        collector=CollectorKind.NATIVE_RSS,
    ).enabled
