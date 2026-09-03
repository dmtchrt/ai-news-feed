"""Add configurable digest delivery times to interest_profiles.

Revision ID: 20260903_01
Revises: 20260902_01
Create Date: 2026-09-03
"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision: str = "20260903_01"
down_revision: str | None = "20260902_01"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "interest_profiles",
        sa.Column(
            "digest_send_times",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'[]'::jsonb"),
            nullable=False,
        ),
    )
    op.create_check_constraint(
        op.f("ck_interest_profiles_interest_digest_send_times_array"),
        "interest_profiles",
        "jsonb_typeof(digest_send_times) = 'array'",
    )


def downgrade() -> None:
    op.drop_constraint(
        op.f("ck_interest_profiles_interest_digest_send_times_array"),
        "interest_profiles",
        type_="check",
    )
    op.drop_column("interest_profiles", "digest_send_times")
