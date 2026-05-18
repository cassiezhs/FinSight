"""Shared configuration for FinSight scripts."""

from __future__ import annotations

import os
from datetime import date, datetime, timedelta
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()

DEFAULT_TICKERS: tuple[str, ...] = (
    "AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA", "TSLA", "BRK-B", "UNH", "JNJ",
    "V", "XOM", "PG", "JPM", "MA", "HD", "LLY", "CVX", "MRK", "PEP",
    "KO", "ABBV", "BAC", "COST", "AVGO", "TMO", "DIS", "WMT", "ADBE", "CRM",
    "NFLX", "PFE", "MCD", "TXN", "ABT", "DHR", "INTC", "NKE", "VZ", "QCOM",
    "MDT", "NEE", "ACN", "AMGN", "LOW", "MS", "SBUX", "UPS", "RTX", "LIN",
    "HON", "UNP", "INTU", "BA", "LMT", "CAT", "T", "ISRG", "PLD", "NOW",
    "GILD", "SPGI", "BLK", "ELV", "BKNG", "ZTS", "MO", "DE", "CI", "C",
    "SCHW", "MDLZ", "SO", "ADP", "SYK", "MMC", "PNC", "AXP", "ETN", "TJX",
    "FDX", "APD", "REGN", "CL", "ADSK", "BSX", "EMR", "WBA", "HUM", "BIIB",
    "ORCL", "GD", "CMCSA", "CSCO", "GM", "PYPL", "TGT", "EBAY", "BK", "COF",
)


def resolve_date(value: str) -> str:
    """Resolve YYYY-MM-DD config values plus dynamic tokens like today."""
    normalized = value.strip().lower()
    if normalized in {"today", "now", "auto"}:
        return date.today().isoformat()
    datetime.strptime(value, "%Y-%m-%d")
    return value


def next_date(value: str) -> str:
    return (datetime.strptime(resolve_date(value), "%Y-%m-%d").date() + timedelta(days=1)).isoformat()


@dataclass(frozen=True)
class Settings:
    db_user: str | None = os.getenv("DB_USER")
    db_password: str | None = os.getenv("DB_PASSWORD")
    db_host: str | None = os.getenv("DB_HOST")
    db_port: str = os.getenv("DB_PORT", "5432")
    db_name: str | None = os.getenv("DB_NAME")
    db_schema: str = os.getenv("DB_SCHEMA", "public")
    sec_user_agent: str = os.getenv("SEC_USER_AGENT", "FinSight/0.1 contact@example.com")
    start_date: str = os.getenv("FINSIGHT_START_DATE", "2018-01-01")
    end_date: str = os.getenv("FINSIGHT_END_DATE", "today")
    start_year: int = int(os.getenv("FINSIGHT_START_YEAR", "2018"))
    end_year: int = int(os.getenv("FINSIGHT_END_YEAR", "2025"))

    @property
    def resolved_end_date(self) -> str:
        return resolve_date(self.end_date)

    @property
    def yfinance_end_date(self) -> str:
        return next_date(self.end_date)

    @property
    def database_url(self) -> str:
        missing = [
            name for name, value in {
                "DB_USER": self.db_user,
                "DB_PASSWORD": self.db_password,
                "DB_HOST": self.db_host,
                "DB_NAME": self.db_name,
            }.items()
            if not value
        ]
        if missing:
            raise RuntimeError(f"Missing DB env vars: {', '.join(missing)}")
        return (
            f"postgresql+psycopg2://{self.db_user}:{self.db_password}"
            f"@{self.db_host}:{self.db_port}/{self.db_name}"
        )


def parse_tickers(value: str | None = None) -> tuple[str, ...]:
    raw = value if value is not None else os.getenv("FINSIGHT_TICKERS")
    if not raw:
        return DEFAULT_TICKERS
    return tuple(t.strip().upper() for t in raw.split(",") if t.strip())


settings = Settings()
TICKERS: tuple[str, ...] = parse_tickers()
