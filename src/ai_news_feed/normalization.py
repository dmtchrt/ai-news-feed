"""Normalize extracted source content into the Material contract."""

from __future__ import annotations

import hashlib
import posixpath
import re
import unicodedata
from datetime import UTC, datetime
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from pydantic import JsonValue

from ai_news_feed.domain.models import Material, RawItem
from ai_news_feed.extraction.models import ExtractedItem

_TRACKING_QUERY_KEYS = {
    "fbclid",
    "gclid",
    "mc_cid",
    "mc_eid",
    "yclid",
}
_TRACKING_QUERY_PREFIXES = ("utm_",)


class Normalizer:
    """Build stable identities and hashes without leaking HTML downstream."""

    def normalize(
        self,
        raw_item: RawItem,
        extracted_item: ExtractedItem,
        *,
        fetched_at: datetime | None = None,
    ) -> Material:
        fetched_at = fetched_at or datetime.now(UTC)
        if fetched_at.tzinfo is None or fetched_at.utcoffset() is None:
            raise ValueError("fetched_at must be timezone-aware")

        text = _normalize_display_text(extracted_item.text)
        title = _choose_title(raw_item, extracted_item, text)
        metadata: dict[str, JsonValue] = dict(raw_item.metadata)
        if extracted_item.metadata:
            metadata["extraction"] = dict(extracted_item.metadata)

        return Material(
            id=_stable_hash(raw_item.source_id, raw_item.external_id),
            source_id=raw_item.source_id,
            external_id=raw_item.external_id,
            original_url=canonicalize_url(raw_item.original_url),
            published_at=raw_item.published_at,
            fetched_at=fetched_at,
            title=title,
            text=text,
            content_hash=content_hash(text),
            metadata=metadata,
        )


def canonicalize_url(url: str) -> str:
    """Canonicalize identity-safe URL parts and remove common tracking parameters."""
    parts = urlsplit(url.strip())
    scheme = parts.scheme.lower()
    hostname = (parts.hostname or "").lower()
    if not hostname or scheme not in {"http", "https"}:
        raise ValueError("url must be an absolute HTTP(S) URL")

    port = parts.port
    default_port = (scheme == "http" and port == 80) or (scheme == "https" and port == 443)
    host_literal = f"[{hostname}]" if ":" in hostname else hostname
    host = host_literal if port is None or default_port else f"{host_literal}:{port}"
    if parts.username or parts.password:
        raise ValueError("userinfo is not allowed in material URLs")

    raw_path = re.sub(r"/{2,}", "/", parts.path or "/")
    normalized_path = posixpath.normpath(raw_path)
    if not normalized_path.startswith("/"):
        normalized_path = f"/{normalized_path}"
    if normalized_path != "/":
        normalized_path = normalized_path.rstrip("/")

    query = [
        (key, value)
        for key, value in parse_qsl(parts.query, keep_blank_values=True)
        if key.casefold() not in _TRACKING_QUERY_KEYS
        and not key.casefold().startswith(_TRACKING_QUERY_PREFIXES)
    ]
    return urlunsplit((scheme, host, normalized_path, urlencode(sorted(query)), ""))


def content_hash(text: str) -> str:
    canonical_text = " ".join(unicodedata.normalize("NFKC", text).casefold().split())
    return hashlib.sha256(canonical_text.encode("utf-8")).hexdigest()


def _stable_hash(*parts: str) -> str:
    payload = "\x1f".join(parts).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _normalize_display_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text)
    lines = [" ".join(line.split()) for line in normalized.splitlines()]
    paragraphs = [line for line in lines if line]
    result = "\n\n".join(paragraphs).strip()
    if not result:
        raise ValueError("extracted text must not be blank")
    return result


def _choose_title(raw_item: RawItem, extracted_item: ExtractedItem, text: str) -> str:
    candidate = extracted_item.title or raw_item.title
    if candidate:
        normalized = " ".join(unicodedata.normalize("NFKC", candidate).split())
        if normalized:
            return normalized[:500]
    first_line = text.splitlines()[0]
    return first_line[:240]
