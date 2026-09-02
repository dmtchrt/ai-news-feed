"""Add digest settings (freshness window, summary length, tone) to interest_profiles.

Revision ID: 20260902_01
Revises: 20260831_01
Create Date: 2026-09-02
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "20260902_01"
down_revision: str | None = "20260831_01"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

summary_length = sa.Enum("brief", "normal", "detailed", name="summary_length")


def upgrade() -> None:
    summary_length.create(op.get_bind(), checkfirst=True)
    op.add_column(
        "interest_profiles",
        sa.Column("freshness_days", sa.Integer(), server_default="7", nullable=False),
    )
    op.add_column(
        "interest_profiles",
        sa.Column("summary_length", summary_length, server_default="normal", nullable=False),
    )
    op.add_column(
        "interest_profiles",
        sa.Column("tone_instructions", sa.Text(), nullable=True),
    )
    op.create_check_constraint(
        op.f("ck_interest_profiles_interest_freshness_days_range"),
        "interest_profiles",
        "freshness_days BETWEEN 1 AND 365",
    )
    op.create_check_constraint(
        op.f("ck_interest_profiles_interest_tone_instructions_not_blank"),
        "interest_profiles",
        "tone_instructions IS NULL OR btrim(tone_instructions) <> ''",
    )


def downgrade() -> None:
    op.drop_constraint(
        op.f("ck_interest_profiles_interest_tone_instructions_not_blank"),
        "interest_profiles",
        type_="check",
    )
    op.drop_constraint(
        op.f("ck_interest_profiles_interest_freshness_days_range"),
        "interest_profiles",
        type_="check",
    )
    op.drop_column("interest_profiles", "tone_instructions")
    op.drop_column("interest_profiles", "summary_length")
    op.drop_column("interest_profiles", "freshness_days")
    summary_length.drop(op.get_bind(), checkfirst=True)
