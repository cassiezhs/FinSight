"""Shared configuration for FinSight scripts."""

from __future__ import annotations

import html
import os
import re
from datetime import date, datetime, timedelta
from dataclasses import dataclass
from urllib.parse import urlsplit, urlunsplit

import requests
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

SP500_WIKIPEDIA_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"


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
    database_url_env: str | None = os.getenv("DATABASE_URL")
    db_user: str | None = os.getenv("DB_USER")
    db_password: str | None = os.getenv("DB_PASSWORD")
    db_host: str | None = os.getenv("DB_HOST")
    db_port: str = os.getenv("DB_PORT", "5432")
    db_name: str | None = os.getenv("DB_NAME")
    db_schema: str = os.getenv("DB_SCHEMA", "public")
    db_sslmode: str | None = os.getenv("DB_SSLMODE")
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
        if self.database_url_env:
            return normalize_sqlalchemy_postgres_url(self.database_url_env)

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
        url = (
            f"postgresql+psycopg2://{self.db_user}:{self.db_password}"
            f"@{self.db_host}:{self.db_port}/{self.db_name}"
        )
        if self.db_sslmode:
            url = f"{url}?sslmode={self.db_sslmode}"
        return url


def normalize_sqlalchemy_postgres_url(url: str) -> str:
    """Accept plain Postgres URLs from providers and make them SQLAlchemy-ready."""
    parts = urlsplit(url)
    if parts.scheme in {"postgres", "postgresql", "ostgresql"}:
        return urlunsplit(("postgresql+psycopg2", parts.netloc, parts.path, parts.query, parts.fragment))
    return url


def normalize_ticker(ticker: str) -> str:
    return ticker.strip().upper().replace(".", "-")


def fetch_sp500_tickers() -> tuple[str, ...]:
    try:
        response = requests.get(
            SP500_WIKIPEDIA_URL,
            headers={"User-Agent": settings.sec_user_agent if "settings" in globals() else "FinSight/0.1"},
            timeout=30,
        )
        response.raise_for_status()
        table_match = re.search(r'<table[^>]+id="constituents"[^>]*>(.*?)</table>', response.text, re.DOTALL)
        if not table_match:
            raise RuntimeError("Could not find S&P 500 constituents table.")

        tickers: list[str] = []
        for row in re.findall(r"<tr[^>]*>(.*?)</tr>", table_match.group(1), re.DOTALL):
            cells = re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", row, re.DOTALL)
            if not cells:
                continue
            symbol = re.sub(r"<[^>]+>", "", cells[0])
            symbol = html.unescape(symbol).strip()
            if symbol and symbol.lower() != "symbol":
                tickers.append(normalize_ticker(symbol))
        if tickers:
            return tuple(dict.fromkeys(tickers))
    except Exception as exc:
        print(f"⚠️ Could not fetch S&P 500 ticker list; using default ticker universe. {exc}")
    return DEFAULT_TICKERS


def parse_tickers(value: str | None = None) -> tuple[str, ...]:
    raw = value if value is not None else os.getenv("FINSIGHT_TICKERS")
    if not raw:
        return DEFAULT_TICKERS
    normalized = raw.strip().lower()
    if normalized in {"sp500", "s&p500", "s&p 500", "all_sp500", "all-sp500"}:
        return fetch_sp500_tickers()
    return tuple(normalize_ticker(t) for t in raw.split(",") if t.strip())


settings = Settings()
TICKERS: tuple[str, ...] = parse_tickers()
