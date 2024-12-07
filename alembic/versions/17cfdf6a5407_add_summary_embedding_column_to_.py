"""add summary_embedding column to episodes table

Revision ID: 17cfdf6a5407
Revises: 11413c348ca0
Create Date: 2024-11-24 13:51:47.851559

"""

from typing import Sequence, Union

import sqlalchemy as sa
from pgvector.sqlalchemy import Vector

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "17cfdf6a5407"
down_revision: Union[str, None] = "11413c348ca0"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("episodes", sa.Column("summary_embedding", Vector(1024)))


def downgrade() -> None:
    op.drop_column("episodes", "summary_embedding")
