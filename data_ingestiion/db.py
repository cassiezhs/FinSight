"""Database helpers shared by ingestion, modeling, and the dashboard."""

from __future__ import annotations

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine

try:
    from .config import settings
except ImportError:
    from config import settings


def get_engine() -> Engine:
    return create_engine(settings.database_url, pool_pre_ping=True)

