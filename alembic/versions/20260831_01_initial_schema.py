"""Create the AI News Feed persistence schema.

Revision ID: 20260831_01
Revises:
Create Date: 2026-08-31
"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision: str = "20260831_01"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

source_kind = sa.Enum("website", "telegram", "manual", name="source_kind")
collector_kind = sa.Enum(
    "native_rss",
    "rss_bridge",
    "web_preview",
    "telethon",
    "universal_scraper",
    name="collector_kind",
)
duplicate_kind = sa.Enum("exact_content", "semantic", name="duplicate_kind")


def _check(expression: str, *, name: str) -> sa.CheckConstraint:
    return sa.CheckConstraint(expression, name=op.f(name))


def upgrade() -> None:
    op.create_table(
        "sources",
        sa.Column("id", sa.Text(), nullable=False),
        sa.Column("kind", source_kind, nullable=False),
        sa.Column("locator", sa.Text(), nullable=False),
        sa.Column("normalized_locator", sa.Text(), nullable=False),
        sa.Column("collector", collector_kind, nullable=False),
        sa.Column("enabled", sa.Boolean(), server_default="true", nullable=False),
        sa.Column("settings", postgresql.JSONB(), server_default="{}", nullable=False),
        sa.Column("cursor", postgresql.JSONB(), nullable=True),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        _check("btrim(locator) <> ''", name="ck_sources_source_locator_not_blank"),
        _check(
            "btrim(normalized_locator) <> ''",
            name="ck_sources_source_normalized_locator_not_blank",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_sources"),
        sa.UniqueConstraint("normalized_locator", name="uq_sources_normalized_locator"),
    )
    op.create_table(
        "interest_profiles",
        sa.Column("id", sa.Text(), nullable=False),
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column("description", sa.Text(), nullable=False),
        sa.Column("enabled", sa.Boolean(), server_default="true", nullable=False),
        sa.Column("version", sa.Integer(), server_default="1", nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column("updated_by_telegram_user_id", sa.BigInteger(), nullable=True),
        _check("btrim(name) <> ''", name="ck_interest_profiles_interest_name_not_blank"),
        _check(
            "btrim(description) <> ''",
            name="ck_interest_profiles_interest_description_not_blank",
        ),
        _check("version >= 1", name="ck_interest_profiles_interest_version_positive"),
        _check(
            "updated_by_telegram_user_id IS NULL OR updated_by_telegram_user_id > 0",
            name="ck_interest_profiles_interest_telegram_user_positive",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_interest_profiles"),
    )
    op.create_table(
        "materials",
        sa.Column("id", sa.Text(), nullable=False),
        sa.Column("source_id", sa.Text(), nullable=False),
        sa.Column("external_id", sa.Text(), nullable=False),
        sa.Column("original_url", sa.Text(), nullable=False),
        sa.Column("published_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("fetched_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("title", sa.Text(), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("language", sa.Text(), nullable=True),
        sa.Column("content_hash", sa.Text(), nullable=False),
        sa.Column("metadata", postgresql.JSONB(), server_default="{}", nullable=False),
        _check(
            "content_hash ~ '^[0-9a-f]{64}$'",
            name="ck_materials_material_content_hash_format",
        ),
        _check(
            "btrim(external_id) <> ''",
            name="ck_materials_material_external_id_not_blank",
        ),
        _check("btrim(text) <> ''", name="ck_materials_material_text_not_blank"),
        _check("btrim(title) <> ''", name="ck_materials_material_title_not_blank"),
        sa.ForeignKeyConstraint(
            ["source_id"],
            ["sources.id"],
            name="fk_materials_source_id_sources",
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_materials"),
        sa.UniqueConstraint("source_id", "external_id", name="uq_materials_source_external"),
    )
    op.create_index("ix_materials_content_hash", "materials", ["content_hash"])
    op.create_index("ix_materials_published_at", "materials", ["published_at"])
    op.create_table(
        "news_clusters",
        sa.Column("id", sa.Text(), nullable=False),
        sa.Column("representative_id", sa.Text(), nullable=False),
        sa.Column("similarities", postgresql.JSONB(), nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.ForeignKeyConstraint(
            ["representative_id"],
            ["materials.id"],
            name="fk_news_clusters_representative_id_materials",
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_news_clusters"),
    )
    op.create_table(
        "cluster_materials",
        sa.Column("cluster_id", sa.Text(), nullable=False),
        sa.Column("material_id", sa.Text(), nullable=False),
        sa.Column("position", sa.Integer(), nullable=False),
        sa.Column("similarity", sa.Float(), nullable=False),
        _check("position >= 0", name="ck_cluster_materials_cluster_material_position_nonnegative"),
        _check(
            "similarity >= 0 AND similarity <= 1",
            name="ck_cluster_materials_cluster_similarity_range",
        ),
        sa.ForeignKeyConstraint(
            ["cluster_id"],
            ["news_clusters.id"],
            name="fk_cluster_materials_cluster_id_news_clusters",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["material_id"],
            ["materials.id"],
            name="fk_cluster_materials_material_id_materials",
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("cluster_id", "material_id", name="pk_cluster_materials"),
        sa.UniqueConstraint("cluster_id", "position", name="uq_cluster_materials_position"),
    )
    op.create_table(
        "duplicate_links",
        sa.Column("material_id", sa.Text(), nullable=False),
        sa.Column("duplicate_of_id", sa.Text(), nullable=False),
        sa.Column("kind", duplicate_kind, nullable=False),
        sa.Column("similarity", sa.Float(), nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        _check(
            "material_id <> duplicate_of_id",
            name="ck_duplicate_links_duplicate_distinct_materials",
        ),
        _check(
            "similarity >= 0 AND similarity <= 1",
            name="ck_duplicate_links_duplicate_similarity_range",
        ),
        sa.ForeignKeyConstraint(
            ["duplicate_of_id"],
            ["materials.id"],
            name="fk_duplicate_links_duplicate_of_id_materials",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["material_id"],
            ["materials.id"],
            name="fk_duplicate_links_material_id_materials",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("material_id", "duplicate_of_id", name="pk_duplicate_links"),
    )
    op.create_table(
        "screening_results",
        sa.Column("cluster_id", sa.Text(), nullable=False),
        sa.Column("profile_id", sa.Text(), nullable=False),
        sa.Column("profile_version", sa.Integer(), nullable=False),
        sa.Column("model", sa.Text(), nullable=False),
        sa.Column("prompt_version", sa.Text(), nullable=False),
        sa.Column("relevance_score", sa.Float(), nullable=False),
        sa.Column("noise_score", sa.Float(), nullable=False),
        sa.Column("uncertain", sa.Boolean(), nullable=False),
        sa.Column("reason", sa.Text(), nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        _check(
            "noise_score >= 0 AND noise_score <= 1",
            name="ck_screening_results_screening_noise_range",
        ),
        _check(
            "profile_version >= 1",
            name="ck_screening_results_screening_profile_version_positive",
        ),
        _check("btrim(reason) <> ''", name="ck_screening_results_screening_reason_not_blank"),
        _check(
            "relevance_score >= 0 AND relevance_score <= 1",
            name="ck_screening_results_screening_relevance_range",
        ),
        sa.ForeignKeyConstraint(
            ["cluster_id"],
            ["news_clusters.id"],
            name="fk_screening_results_cluster_id_news_clusters",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["profile_id"],
            ["interest_profiles.id"],
            name="fk_screening_results_profile_id_interest_profiles",
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint(
            "cluster_id",
            "profile_id",
            "profile_version",
            "model",
            "prompt_version",
            name="pk_screening_results",
        ),
    )
    op.create_table(
        "digests",
        sa.Column("id", sa.Text(), nullable=False),
        sa.Column("profile_id", sa.Text(), nullable=False),
        sa.Column("profile_version", sa.Integer(), nullable=False),
        sa.Column("channel_id", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("sent_at", sa.DateTime(timezone=True), nullable=True),
        _check("btrim(channel_id) <> ''", name="ck_digests_digest_channel_id_not_blank"),
        _check("profile_version >= 1", name="ck_digests_digest_profile_version_positive"),
        sa.ForeignKeyConstraint(
            ["profile_id"],
            ["interest_profiles.id"],
            name="fk_digests_profile_id_interest_profiles",
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_digests"),
    )
    op.create_table(
        "digest_items",
        sa.Column("digest_id", sa.Text(), nullable=False),
        sa.Column("position", sa.Integer(), nullable=False),
        sa.Column("cluster_id", sa.Text(), nullable=False),
        sa.Column("summary", sa.Text(), nullable=False),
        sa.Column("source_links", postgresql.JSONB(), nullable=False),
        sa.Column("model", sa.Text(), nullable=False),
        sa.Column("prompt_version", sa.Text(), nullable=False),
        _check("position >= 0", name="ck_digest_items_digest_item_position_nonnegative"),
        _check("btrim(summary) <> ''", name="ck_digest_items_digest_item_summary_not_blank"),
        sa.ForeignKeyConstraint(
            ["cluster_id"],
            ["news_clusters.id"],
            name="fk_digest_items_cluster_id_news_clusters",
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["digest_id"],
            ["digests.id"],
            name="fk_digest_items_digest_id_digests",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("digest_id", "position", name="pk_digest_items"),
        sa.UniqueConstraint("digest_id", "cluster_id", name="uq_digest_items_cluster"),
    )
    op.create_table(
        "digest_posts",
        sa.Column("digest_id", sa.Text(), nullable=False),
        sa.Column("position", sa.Integer(), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("telegram_message_id", sa.BigInteger(), nullable=True),
        sa.Column("sent_at", sa.DateTime(timezone=True), nullable=True),
        _check("position >= 0", name="ck_digest_posts_digest_post_position_nonnegative"),
        _check(
            "telegram_message_id IS NULL OR telegram_message_id > 0",
            name="ck_digest_posts_digest_post_message_id_positive",
        ),
        _check(
            "(telegram_message_id IS NULL) = (sent_at IS NULL)",
            name="ck_digest_posts_digest_post_receipt_complete",
        ),
        _check("btrim(text) <> ''", name="ck_digest_posts_digest_post_text_not_blank"),
        sa.ForeignKeyConstraint(
            ["digest_id"],
            ["digests.id"],
            name="fk_digest_posts_digest_id_digests",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("digest_id", "position", name="pk_digest_posts"),
    )


def downgrade() -> None:
    op.drop_table("digest_posts")
    op.drop_table("digest_items")
    op.drop_table("digests")
    op.drop_table("screening_results")
    op.drop_table("duplicate_links")
    op.drop_table("cluster_materials")
    op.drop_table("news_clusters")
    op.drop_index("ix_materials_published_at", table_name="materials")
    op.drop_index("ix_materials_content_hash", table_name="materials")
    op.drop_table("materials")
    op.drop_table("interest_profiles")
    op.drop_table("sources")
    bind = op.get_bind()
    duplicate_kind.drop(bind, checkfirst=True)
    collector_kind.drop(bind, checkfirst=True)
    source_kind.drop(bind, checkfirst=True)
