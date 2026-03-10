from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Iterable, Optional

from sqlalchemy import (
    JSON,
    TIMESTAMP,
    Column,
    Float,
    MetaData,
    String,
    Table,
    Text,
    create_engine,
    text,
)
from sqlalchemy.engine import Engine


def _default_sqlite_path() -> Path:
    # Keep data inside the repo by default (good for local dev).
    data_dir = Path(os.getenv("ROADDAMAGE_DATA_DIR") or Path(__file__).resolve().parent / "data")
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir / "potholes.db"


def get_database_url() -> str:
    """
    If DATABASE_URL is set (e.g. Supabase Postgres), use it.
    Otherwise fallback to a local SQLite file DB.
    """
    url = os.getenv("DATABASE_URL")
    if url and url.strip():
        return url.strip()
    return f"sqlite:///{_default_sqlite_path().as_posix()}"


def create_db_engine(url: Optional[str] = None) -> Engine:
    url = url or get_database_url()

    # SQLite needs this for FastAPI threaded workers.
    if url.startswith("sqlite:///"):
        return create_engine(url, connect_args={"check_same_thread": False})

    # Supabase Postgres requires SSL.
    connect_args: dict[str, Any] = {}
    if url.startswith("postgres://"):
        # Normalize legacy scheme
        url = "postgresql://" + url[len("postgres://") :]
    if url.startswith("postgresql://"):
        connect_args["sslmode"] = os.getenv("PGSSLMODE") or "require"

    return create_engine(url, pool_pre_ping=True, connect_args=connect_args)


metadata = MetaData()


pothole_reports = Table(
    "pothole_reports",
    metadata,
    Column("id", String, primary_key=True),
    Column("latitude", Float, nullable=False),
    Column("longitude", Float, nullable=False),
    Column("road_name", Text, nullable=True),
    Column("damage_type", Text, nullable=True),
    Column("severity_score", Float, nullable=True),
    Column("confidence", Float, nullable=True),
    Column("image_path", Text, nullable=True),
    Column("detected_at", TIMESTAMP, nullable=True),
    # store as JSON in Postgres; SQLite will accept it as TEXT-like.
    Column("ocr_metadata", JSON, nullable=True),
    Column("status", Text, nullable=True),
)


road_segments = Table(
    "road_segments",
    metadata,
    Column("id", String, primary_key=True),
    Column("name", Text, nullable=True),
    Column("coordinates", Text, nullable=True),
    Column("length", Float, nullable=True),
    Column("highway_type", Text, nullable=True),
    Column("last_checked", TIMESTAMP, nullable=True),
)


def init_db(engine: Engine) -> None:
    metadata.create_all(engine)

    # Create indexes (SQLAlchemy doesn't create SQLite indexes reliably with JSON etc; do it explicitly)
    with engine.begin() as conn:
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_pothole_location ON pothole_reports(latitude, longitude)"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_pothole_severity ON pothole_reports(severity_score)"))


def fetch_all(engine: Engine, sql: str, params: Optional[dict[str, Any]] = None) -> list[dict[str, Any]]:
    with engine.begin() as conn:
        rows = conn.execute(text(sql), params or {}).mappings().all()
        return [dict(r) for r in rows]


def fetch_one(engine: Engine, sql: str, params: Optional[dict[str, Any]] = None) -> Optional[dict[str, Any]]:
    with engine.begin() as conn:
        row = conn.execute(text(sql), params or {}).mappings().first()
        return dict(row) if row else None


def execute(engine: Engine, sql: str, params: Optional[dict[str, Any]] = None) -> None:
    with engine.begin() as conn:
        conn.execute(text(sql), params or {})


def executemany(engine: Engine, sql: str, param_list: Iterable[dict[str, Any]]) -> None:
    with engine.begin() as conn:
        conn.execute(text(sql), list(param_list))

