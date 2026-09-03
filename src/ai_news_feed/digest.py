"""Compose deterministic Telegram-sized digest posts."""

from __future__ import annotations

import hashlib
import html
import json
from collections.abc import Sequence
from datetime import UTC, datetime
from urllib.parse import urlsplit

from ai_news_feed.domain.models import Digest, DigestItem

TELEGRAM_TEXT_LIMIT = 4096
DEFAULT_HEADER = "AI News Feed"


class DigestComposer:
    def __init__(
        self,
        *,
        max_post_chars: int = TELEGRAM_TEXT_LIMIT,
        header: str = DEFAULT_HEADER,
    ) -> None:
        if max_post_chars < 128 or max_post_chars > TELEGRAM_TEXT_LIMIT:
            raise ValueError(f"max_post_chars must be in range 128..{TELEGRAM_TEXT_LIMIT}")
        self.max_post_chars = max_post_chars
        self.header = header.strip()
        if not self.header:
            raise ValueError("header must not be blank")
        if len(self.header) + 3 >= max_post_chars:
            raise ValueError("header leaves no room for digest content")

    def compose(
        self,
        items: Sequence[DigestItem],
        *,
        profile_id: str,
        profile_version: int,
        created_at: datetime | None = None,
    ) -> Digest | None:
        """Return None for the expected no-relevant-news case."""
        if not items:
            return None
        now = created_at or datetime.now(UTC)
        if now.tzinfo is None or now.utcoffset() is None:
            raise ValueError("created_at must be timezone-aware")
        item_tuple = tuple(items)
        blocks = tuple(_format_item(item) for item in item_tuple)
        posts = self._pack(blocks)
        return Digest(
            id=_digest_id(profile_id, profile_version, item_tuple),
            profile_id=profile_id,
            profile_version=profile_version,
            items=item_tuple,
            posts=posts,
            created_at=now,
        )

    def _pack(self, blocks: Sequence[str]) -> tuple[str, ...]:
        """One news item -> one post: never merge two items' text into the same post.

        A single item's own text still gets split across multiple posts if it alone
        exceeds the limit -- only cross-item merging is disallowed.
        """
        separator = "\n\n"
        prefix = f"{self.header}\n\n"
        content_limit = self.max_post_chars - len(prefix)
        posts: list[str] = []
        for block in blocks:
            current = ""
            for chunk in _split_text(block, content_limit):
                candidate = chunk if not current else f"{current}{separator}{chunk}"
                if len(candidate) <= content_limit:
                    current = candidate
                    continue
                posts.append(f"{prefix}{current}")
                current = chunk
            if current:
                posts.append(f"{prefix}{current}")
        return tuple(posts)


def _format_item(item: DigestItem) -> str:
    dates = item.source_published_ats
    lines: list[str] = []
    for index, url in enumerate(item.source_links, 1):
        anchor = f'<a href="{html.escape(url, quote=True)}">{html.escape(_link_label(url))}</a>'
        lines.append(f"{index}. {anchor}")
        if dates is not None:
            lines.append(dates[index - 1].strftime("%d.%m.%Y"))
    links = "\n".join(lines)
    return f"• {html.escape(item.summary)}\nИсточники:\n{links}"


def _link_label(url: str) -> str:
    netloc = urlsplit(url).netloc.removeprefix("www.")
    return netloc or url


def _split_text(text: str, limit: int) -> tuple[str, ...]:
    if len(text) <= limit:
        return (text,)
    remaining = text
    parts: list[str] = []
    while len(remaining) > limit:
        split_at = max(
            remaining.rfind("\n", 0, limit + 1),
            remaining.rfind(" ", 0, limit + 1),
        )
        if split_at < limit // 2:
            split_at = limit
        parts.append(remaining[:split_at].rstrip())
        remaining = remaining[split_at:].lstrip()
    if remaining:
        parts.append(remaining)
    return tuple(parts)


def _digest_id(
    profile_id: str,
    profile_version: int,
    items: Sequence[DigestItem],
) -> str:
    payload = {
        "profile_id": profile_id,
        "profile_version": profile_version,
        "items": [item.model_dump(mode="json") for item in items],
    }
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
