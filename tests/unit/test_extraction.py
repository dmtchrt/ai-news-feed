from datetime import UTC, datetime
from pathlib import Path

from reportlab.pdfgen.canvas import Canvas

from ai_news_feed.domain.models import Attachment, AttachmentKind, RawItem
from ai_news_feed.extraction import ContentExtractor, ExtractedItem, ExtractionFailure


def test_pdf_attachment_is_extracted(tmp_path: Path) -> None:
    pdf_path = tmp_path / "ai-market.pdf"
    canvas = Canvas(str(pdf_path))
    canvas.drawString(72, 760, "AI market grew by 20 percent in 2026.")
    canvas.save()
    item = RawItem(
        source_id="telegram-expertosphere",
        external_id="502",
        original_url="https://t.me/expertosphere/502",
        published_at=datetime(2026, 8, 31, tzinfo=UTC),
        attachments=(
            Attachment(
                kind=AttachmentKind.DOCUMENT,
                name="ai-market.pdf",
                mime_type="application/pdf",
                download_ref=str(pdf_path),
            ),
        ),
    )

    extracted = ContentExtractor().extract(item)

    assert isinstance(extracted, ExtractedItem)
    assert "AI market grew by 20 percent" in extracted.text


def test_unavailable_preview_document_is_a_typed_failure() -> None:
    item = RawItem(
        source_id="telegram-reports",
        external_id="77",
        original_url="https://t.me/reports/77",
        published_at=datetime(2026, 8, 30, tzinfo=UTC),
        attachments=(
            Attachment(
                kind=AttachmentKind.DOCUMENT,
                name="report.pdf",
                mime_type="application/pdf",
            ),
        ),
    )

    extracted = ContentExtractor().extract(item)

    assert isinstance(extracted, ExtractionFailure)
    assert extracted.code == "attachment_unavailable"
    assert "requires Telethon" in extracted.message
