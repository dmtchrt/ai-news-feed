"""Turn a RawItem into plain text without leaking source-specific types downstream."""

from __future__ import annotations

from pathlib import Path
from urllib.parse import unquote, urlsplit

import trafilatura
from bs4 import BeautifulSoup
from pydantic import JsonValue

from ai_news_feed.domain.models import AttachmentKind, RawItem
from ai_news_feed.extraction.documents import DocumentExtractionError, extract_document
from ai_news_feed.extraction.models import ExtractedItem, ExtractionFailure


class ContentExtractor:
    def extract(self, item: RawItem) -> ExtractedItem | ExtractionFailure:
        parts: list[str] = []
        attachment_errors: list[str] = []

        if item.raw_text:
            parts.append(item.raw_text)
        elif item.raw_html:
            html_text = _extract_html(item.raw_html)
            if html_text:
                parts.append(html_text)

        extracted_documents = 0
        for attachment in item.attachments:
            if attachment.kind is not AttachmentKind.DOCUMENT:
                continue
            if not attachment.download_ref:
                attachment_errors.append(
                    f"{attachment.name}: document has no download_ref; source requires Telethon"
                )
                continue
            try:
                path = _local_path(attachment.download_ref)
                parts.append(extract_document(path, attachment.mime_type))
                extracted_documents += 1
            except (DocumentExtractionError, OSError, ValueError) as exc:
                attachment_errors.append(f"{attachment.name}: {exc}")

        text = _join_distinct(parts)
        metadata_errors: list[JsonValue] = []
        metadata_errors.extend(attachment_errors)
        metadata: dict[str, JsonValue] = {
            "extracted_documents": extracted_documents,
            "attachment_errors": metadata_errors,
        }
        if not text:
            code = "attachment_unavailable" if attachment_errors else "empty_content"
            message = "; ".join(attachment_errors) or "item contains no extractable text"
            return ExtractionFailure(code=code, message=message, metadata=metadata)
        return ExtractedItem(text=text, title=item.title, metadata=metadata)


def _extract_html(html: str) -> str | None:
    extracted = trafilatura.extract(
        html,
        include_comments=False,
        include_tables=True,
        no_fallback=False,
    )
    if extracted and extracted.strip():
        return extracted.strip()
    soup = BeautifulSoup(html, "html.parser")
    for element in soup.select("script, style, nav, footer, header, aside"):
        element.decompose()
    fallback = soup.get_text("\n", strip=True)
    return fallback or None


def _local_path(download_ref: str) -> Path:
    parsed = urlsplit(download_ref)
    if parsed.scheme == "file":
        return Path(unquote(parsed.path))
    if parsed.scheme:
        raise ValueError(f"remote download_ref is not materialized: {parsed.scheme}")
    path = Path(download_ref)
    if not path.is_absolute():
        raise ValueError("download_ref must be an absolute local path or file:// URI")
    return path


def _join_distinct(parts: list[str]) -> str:
    result: list[str] = []
    seen: set[str] = set()
    for part in parts:
        cleaned = "\n".join(line.rstrip() for line in part.strip().splitlines()).strip()
        fingerprint = " ".join(cleaned.split()).casefold()
        if cleaned and fingerprint not in seen:
            result.append(cleaned)
            seen.add(fingerprint)
    return "\n\n".join(result)
