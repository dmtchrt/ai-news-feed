"""SQLAlchemy Core schema; Alembic migrations are the production source of truth."""

from __future__ import annotations

from sqlalchemy import (
    BigInteger,
    Boolean,
    CheckConstraint,
    Column,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    ForeignKeyConstraint,
    Index,
    Integer,
    MetaData,
    Table,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB

from ai_news_feed.domain.models import CollectorKind, DuplicateKind, SourceKind, SummaryLength

NAMING_CONVENTION = {
    "ix": "ix_%(table_name)s_%(column_0_name)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}
metadata = MetaData(naming_convention=NAMING_CONVENTION)


def _enum(
    enum_class: type[SourceKind] | type[CollectorKind] | type[DuplicateKind] | type[SummaryLength],
    name: str,
) -> Enum:
    return Enum(
        enum_class,
        name=name,
        values_callable=lambda values: [value.value for value in values],
        validate_strings=True,
    )


sources = Table(
    "sources",
    metadata,
    Column("id", Text, primary_key=True),
    Column("kind", _enum(SourceKind, "source_kind"), nullable=False),
    Column("locator", Text, nullable=False),
    Column("normalized_locator", Text, nullable=False),
    Column("collector", _enum(CollectorKind, "collector_kind"), nullable=False),
    Column("enabled", Boolean, nullable=False, server_default="true"),
    Column("settings", JSONB, nullable=False, server_default="{}"),
    Column("cursor", JSONB),
    Column("created_at", DateTime(timezone=True), nullable=False, server_default=func.now()),
    Column("updated_at", DateTime(timezone=True), nullable=False, server_default=func.now()),
    UniqueConstraint("normalized_locator", name="uq_sources_normalized_locator"),
    CheckConstraint("btrim(locator) <> ''", name="source_locator_not_blank"),
    CheckConstraint("btrim(normalized_locator) <> ''", name="source_normalized_locator_not_blank"),
)

interest_profiles = Table(
    "interest_profiles",
    metadata,
    Column("id", Text, primary_key=True),
    Column("name", Text, nullable=False),
    Column("description", Text, nullable=False),
    Column("enabled", Boolean, nullable=False, server_default="true"),
    Column("version", Integer, nullable=False, server_default="1"),
    Column("freshness_days", Integer, nullable=False, server_default="7"),
    Column(
        "summary_length",
        _enum(SummaryLength, "summary_length"),
        nullable=False,
        server_default=SummaryLength.NORMAL.value,
    ),
    Column("tone_instructions", Text),
    Column("created_at", DateTime(timezone=True), nullable=False, server_default=func.now()),
    Column("updated_at", DateTime(timezone=True), nullable=False, server_default=func.now()),
    Column("updated_by_telegram_user_id", BigInteger),
    CheckConstraint("btrim(name) <> ''", name="interest_name_not_blank"),
    CheckConstraint("btrim(description) <> ''", name="interest_description_not_blank"),
    CheckConstraint("version >= 1", name="interest_version_positive"),
    CheckConstraint("freshness_days BETWEEN 1 AND 365", name="interest_freshness_days_range"),
    CheckConstraint(
        "tone_instructions IS NULL OR btrim(tone_instructions) <> ''",
        name="interest_tone_instructions_not_blank",
    ),
    CheckConstraint(
        "updated_by_telegram_user_id IS NULL OR updated_by_telegram_user_id > 0",
        name="interest_telegram_user_positive",
    ),
)

materials = Table(
    "materials",
    metadata,
    Column("id", Text, primary_key=True),
    Column("source_id", Text, ForeignKey("sources.id", ondelete="RESTRICT"), nullable=False),
    Column("external_id", Text, nullable=False),
    Column("original_url", Text, nullable=False),
    Column("published_at", DateTime(timezone=True), nullable=False),
    Column("fetched_at", DateTime(timezone=True), nullable=False),
    Column("title", Text, nullable=False),
    Column("text", Text, nullable=False),
    Column("language", Text),
    Column("content_hash", Text, nullable=False),
    Column("metadata", JSONB, nullable=False, server_default="{}"),
    UniqueConstraint("source_id", "external_id", name="uq_materials_source_external"),
    CheckConstraint("btrim(external_id) <> ''", name="material_external_id_not_blank"),
    CheckConstraint("btrim(title) <> ''", name="material_title_not_blank"),
    CheckConstraint("btrim(text) <> ''", name="material_text_not_blank"),
    CheckConstraint("content_hash ~ '^[0-9a-f]{64}$'", name="material_content_hash_format"),
)
Index("ix_materials_content_hash", materials.c.content_hash)
Index("ix_materials_published_at", materials.c.published_at)

news_clusters = Table(
    "news_clusters",
    metadata,
    Column("id", Text, primary_key=True),
    Column(
        "representative_id",
        Text,
        ForeignKey("materials.id", ondelete="RESTRICT"),
        nullable=False,
    ),
    Column("similarities", JSONB, nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False, server_default=func.now()),
)

cluster_materials = Table(
    "cluster_materials",
    metadata,
    Column(
        "cluster_id", Text, ForeignKey("news_clusters.id", ondelete="CASCADE"), primary_key=True
    ),
    Column("material_id", Text, ForeignKey("materials.id", ondelete="RESTRICT"), primary_key=True),
    Column("position", Integer, nullable=False),
    Column("similarity", Float, nullable=False),
    UniqueConstraint("cluster_id", "position", name="uq_cluster_materials_position"),
    CheckConstraint("position >= 0", name="cluster_material_position_nonnegative"),
    CheckConstraint("similarity >= 0 AND similarity <= 1", name="cluster_similarity_range"),
)

duplicate_links = Table(
    "duplicate_links",
    metadata,
    Column("material_id", Text, ForeignKey("materials.id", ondelete="CASCADE"), primary_key=True),
    Column(
        "duplicate_of_id",
        Text,
        ForeignKey("materials.id", ondelete="CASCADE"),
        primary_key=True,
    ),
    Column("kind", _enum(DuplicateKind, "duplicate_kind"), nullable=False),
    Column("similarity", Float, nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False, server_default=func.now()),
    CheckConstraint("material_id <> duplicate_of_id", name="duplicate_distinct_materials"),
    CheckConstraint("similarity >= 0 AND similarity <= 1", name="duplicate_similarity_range"),
)

screening_results = Table(
    "screening_results",
    metadata,
    Column(
        "cluster_id", Text, ForeignKey("news_clusters.id", ondelete="CASCADE"), primary_key=True
    ),
    Column("profile_id", Text, primary_key=True),
    Column("profile_version", Integer, primary_key=True),
    Column("model", Text, primary_key=True),
    Column("prompt_version", Text, primary_key=True),
    Column("relevance_score", Float, nullable=False),
    Column("noise_score", Float, nullable=False),
    Column("uncertain", Boolean, nullable=False),
    Column("reason", Text, nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False, server_default=func.now()),
    ForeignKeyConstraint(
        ["profile_id"],
        ["interest_profiles.id"],
        ondelete="RESTRICT",
        name="fk_screening_results_profile_id_interest_profiles",
    ),
    CheckConstraint("profile_version >= 1", name="screening_profile_version_positive"),
    CheckConstraint(
        "relevance_score >= 0 AND relevance_score <= 1",
        name="screening_relevance_range",
    ),
    CheckConstraint("noise_score >= 0 AND noise_score <= 1", name="screening_noise_range"),
    CheckConstraint("btrim(reason) <> ''", name="screening_reason_not_blank"),
)

digests = Table(
    "digests",
    metadata,
    Column("id", Text, primary_key=True),
    Column(
        "profile_id", Text, ForeignKey("interest_profiles.id", ondelete="RESTRICT"), nullable=False
    ),
    Column("profile_version", Integer, nullable=False),
    Column("channel_id", Text, nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("sent_at", DateTime(timezone=True)),
    CheckConstraint("profile_version >= 1", name="digest_profile_version_positive"),
    CheckConstraint("btrim(channel_id) <> ''", name="digest_channel_id_not_blank"),
)

digest_items = Table(
    "digest_items",
    metadata,
    Column("digest_id", Text, ForeignKey("digests.id", ondelete="CASCADE"), primary_key=True),
    Column("position", Integer, primary_key=True),
    Column("cluster_id", Text, ForeignKey("news_clusters.id", ondelete="RESTRICT"), nullable=False),
    Column("summary", Text, nullable=False),
    Column("source_links", JSONB, nullable=False),
    Column("model", Text, nullable=False),
    Column("prompt_version", Text, nullable=False),
    UniqueConstraint("digest_id", "cluster_id", name="uq_digest_items_cluster"),
    CheckConstraint("position >= 0", name="digest_item_position_nonnegative"),
    CheckConstraint("btrim(summary) <> ''", name="digest_item_summary_not_blank"),
)

digest_posts = Table(
    "digest_posts",
    metadata,
    Column("digest_id", Text, ForeignKey("digests.id", ondelete="CASCADE"), primary_key=True),
    Column("position", Integer, primary_key=True),
    Column("text", Text, nullable=False),
    Column("telegram_message_id", BigInteger),
    Column("sent_at", DateTime(timezone=True)),
    CheckConstraint("position >= 0", name="digest_post_position_nonnegative"),
    CheckConstraint("btrim(text) <> ''", name="digest_post_text_not_blank"),
    CheckConstraint(
        "telegram_message_id IS NULL OR telegram_message_id > 0",
        name="digest_post_message_id_positive",
    ),
    CheckConstraint(
        "(telegram_message_id IS NULL) = (sent_at IS NULL)",
        name="digest_post_receipt_complete",
    ),
)
