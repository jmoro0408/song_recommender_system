import datetime
import uuid

from pgvector.sqlalchemy import Vector
from sqlalchemy import DateTime, ForeignKey, text
from sqlalchemy.dialects.postgresql import INET, UUID, VARCHAR
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class Episodes(Base):
    __tablename__ = "episodes"

    episode_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()")
    )
    podcast_name: Mapped[str] = mapped_column(nullable=False)
    episode_title: Mapped[str] = mapped_column(nullable=False)
    link: Mapped[str] = mapped_column(nullable=True)
    summary: Mapped[str] = mapped_column(nullable=True)
    enc_len: Mapped[int] = mapped_column(nullable=True)
    transcript: Mapped[str] = mapped_column(nullable=True)
    published_date: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    streaming_events: Mapped[list["StreamingEvents"]] = relationship(
        "StreamingEvents", back_populates="episode"
    )
    clean_title: Mapped[str] = mapped_column(nullable=False)
    summary_embedding: Mapped[Vector] = mapped_column(Vector(1024))


class StreamingEvents(Base):
    __tablename__ = "streaming_events"
    streaming_events_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()")
    )
    ts: Mapped[DateTime] = mapped_column(DateTime(timezone=True),primary_key=False)
    platform: Mapped[str] = mapped_column(VARCHAR(255), nullable=False)
    ms_played: Mapped[int] = mapped_column(nullable=False)
    conn_country: Mapped[str] = mapped_column(
        VARCHAR(2), nullable=False
    )  # ISO country code
    ip_addr: Mapped[str] = mapped_column(INET, nullable=False)
    master_metadata_track_name: Mapped[str | None] = mapped_column(
        VARCHAR(255), nullable=True
    )
    master_metadata_album_artist_name: Mapped[str | None] = mapped_column(
        VARCHAR(255), nullable=True
    )
    master_metadata_album_album_name: Mapped[str | None] = mapped_column(
        VARCHAR(255), nullable=True
    )
    spotify_track_uri: Mapped[str | None] = mapped_column(VARCHAR(255), nullable=True)
    episode_name: Mapped[str | None] = mapped_column(VARCHAR(255), nullable=True)
    episode_show_name: Mapped[str | None] = mapped_column(VARCHAR(255), nullable=True)
    spotify_episode_uri: Mapped[str | None] = mapped_column(VARCHAR(255), nullable=True)
    reason_start: Mapped[str | None] = mapped_column(VARCHAR(255), nullable=True)
    reason_end: Mapped[str | None] = mapped_column(VARCHAR(255), nullable=True)
    shuffle: Mapped[bool] = mapped_column(default=False, nullable=False)
    skipped: Mapped[bool] = mapped_column(default=False, nullable=False)
    offline: Mapped[bool] = mapped_column(default=False, nullable=False)
    offline_timestamp: Mapped[float | None] = mapped_column(nullable=True)
    incognito_mode: Mapped[bool] = mapped_column(default=False, nullable=False)
    episode_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("episodes.episode_id", ondelete="CASCADE"),
        nullable=True,
        index=True,
    )
    episode: Mapped["Episodes"] = relationship(
        "Episodes", back_populates="streaming_events"
    )
