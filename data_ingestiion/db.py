"""Database helpers shared by ingestion, modeling, and the dashboard."""

from __future__ import annotations

from sqlalchemy import create_engine
from sqlalchemy import event
from sqlalchemy.engine import Engine

try:
    from .config import settings
except ImportError:
    from config import settings


def get_engine() -> Engine:
    engine = create_engine(settings.database_url, pool_pre_ping=True)

    @event.listens_for(engine, "checkout")
    def set_search_path(dbapi_connection, connection_record, connection_proxy):
        with dbapi_connection.cursor() as cursor:
            cursor.execute(f"SET search_path TO {settings.db_schema}")

    return engine
