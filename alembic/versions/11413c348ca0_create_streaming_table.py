"""create streaming table

Revision ID: 11413c348ca0
Revises: 9c41fc3805b0
Create Date: 2024-11-24 13:51:26.644631

"""

from typing import Sequence, Union

import sqlalchemy as sa

from alembic import op
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID, INET

# revision identifiers, used by Alembic.
revision: str = "11413c348ca0"
down_revision: Union[str, None] = "9c41fc3805b0"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade():
    op.create_table(
        "streaming_events",
        sa.Column(
            "streaming_events_id",
            UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column("ts", sa.DateTime(timezone=True), nullable=False),
        sa.Column("platform", sa.VARCHAR(255), nullable=False),
        sa.Column("ms_played", sa.Integer, nullable=False),
        sa.Column("conn_country", sa.VARCHAR(2), nullable=False),
        sa.Column("ip_addr", INET, nullable=False),
        sa.Column("master_metadata_track_name", sa.VARCHAR(255), nullable=True),
        sa.Column("master_metadata_album_artist_name", sa.VARCHAR(255), nullable=True),
        sa.Column("master_metadata_album_album_name", sa.VARCHAR(255), nullable=True),
        sa.Column("spotify_track_uri", sa.VARCHAR(255), nullable=True),
        sa.Column("episode_name", sa.VARCHAR(255), nullable=True),
        sa.Column("episode_show_name", sa.VARCHAR(255), nullable=True),
        sa.Column("spotify_episode_uri", sa.VARCHAR(255), nullable=True),
        sa.Column("reason_start", sa.VARCHAR(255), nullable=True),
        sa.Column("reason_end", sa.VARCHAR(255), nullable=True),
        sa.Column(
            "shuffle", sa.Boolean, server_default=sa.text("false"), nullable=False
        ),
        sa.Column(
            "skipped", sa.Boolean, server_default=sa.text("false"), nullable=False
        ),
        sa.Column(
            "offline", sa.Boolean, server_default=sa.text("false"), nullable=False
        ),
        sa.Column("offline_timestamp", sa.Float, nullable=True),
        sa.Column(
            "incognito_mode",
            sa.Boolean,
            server_default=sa.text("false"),
            nullable=False,
        ),
        sa.Column("episode_id", UUID(as_uuid=True), nullable=True),
        sa.ForeignKeyConstraint(
            ["episode_id"],
            ["episodes.episode_id"],
            ondelete="CASCADE",
        ),
        sa.Index("ix_streaming_events_episode_id", "episode_id"),
    )


def downgrade():
    op.drop_table("streaming_events")
