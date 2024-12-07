"""create episodes table

Revision ID: 9c41fc3805b0
Revises:
Create Date: 2024-11-24 13:50:38.229001

"""

from typing import Sequence, Union

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "9c41fc3805b0"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "episodes",
        sa.Column(
            "episode_id",
            sa.UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column("podcast_name", sa.String(), nullable=False),
        sa.Column("episode_title", sa.String(), nullable=False),
        sa.Column("link", sa.String(), nullable=True),
        sa.Column("summary", sa.Text(), nullable=True),
        sa.Column("enc_len", sa.Integer(), nullable=True),
        sa.Column("transcript", sa.Text(), nullable=True),
        sa.Column("published_date", sa.DateTime(timezone=True), nullable=True),
        sa.Column("clean_title", sa.String(), nullable=False),
    )


def downgrade() -> None:
    op.drop_table("episodes")
