from datetime import UTC, datetime

from ai_news_feed.domain.models import RawItem
from ai_news_feed.extraction.models import ExtractedItem
from ai_news_feed.normalization import Normalizer, canonicalize_url


def _raw_item(
    url: str = "https://Example.TEST:443/news//42/?utm_source=rss&b=2&a=1#top",
) -> RawItem:
    return RawItem(
        source_id="source",
        external_id="42",
        original_url=url,
        published_at=datetime(2026, 8, 31, 7, tzinfo=UTC),
        title="  Заголовок   новости ",
        raw_html="<p>HTML is source-only</p>",
        metadata={"transport": "rss"},
    )


def test_normalizer_builds_stable_material_and_canonical_url() -> None:
    normalizer = Normalizer()
    fetched_at = datetime(2026, 8, 31, 8, tzinfo=UTC)

    first = normalizer.normalize(
        _raw_item(),
        ExtractedItem(text="ИИ-агенты\n\n меняют рынок", metadata={"documents": 0}),
        fetched_at=fetched_at,
    )
    second = normalizer.normalize(
        _raw_item("https://example.test/news/42?a=1&b=2"),
        ExtractedItem(text="  ии-АГЕНТЫ  меняют   рынок  "),
        fetched_at=fetched_at,
    )

    assert first.id == second.id
    assert first.content_hash == second.content_hash
    assert first.original_url == "https://example.test/news/42?a=1&b=2"
    assert first.title == "Заголовок новости"
    assert first.metadata["extraction"] == {"documents": 0}
    assert "raw_html" not in type(first).model_fields


def test_canonicalize_url_keeps_non_tracking_parameters() -> None:
    assert canonicalize_url("http://EXAMPLE.test:80/a/../b/?z=&gclid=x") == (
        "http://example.test/b?z="
    )
