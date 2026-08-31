from datetime import UTC, datetime

from ai_news_feed.dedup.exact import ExactDeduper
from ai_news_feed.domain.models import DuplicateKind, Material
from ai_news_feed.normalization import content_hash


def _material(material_id: str, text: str, *, hour: int = 1) -> Material:
    return Material(
        id=material_id,
        source_id=f"source-{material_id}",
        external_id=material_id,
        original_url=f"https://example.test/{material_id}",
        published_at=datetime(2026, 8, 31, hour, tzinfo=UTC),
        fetched_at=datetime(2026, 8, 31, 10, tzinfo=UTC),
        title=f"Title {material_id}",
        text=text,
        content_hash=content_hash(text),
    )


def test_exact_dedup_prefers_saved_then_first_batch_material() -> None:
    saved = _material("saved", "Already persisted")
    first = _material("first", "New story")
    first_copy = _material("copy", "  new STORY ")
    saved_copy = _material("saved-copy", "ALREADY persisted")

    result = ExactDeduper().deduplicate([first, first_copy, saved_copy], [saved])

    assert result.unique_materials == (first,)
    assert [(link.material_id, link.duplicate_of_id) for link in result.duplicate_links] == [
        ("copy", "first"),
        ("saved-copy", "saved"),
    ]
    assert all(link.kind is DuplicateKind.EXACT_CONTENT for link in result.duplicate_links)
