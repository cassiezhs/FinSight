#!/usr/bin/env python3
"""Daily FinSight data refresh entrypoint for schedulers such as GitHub Actions."""

from __future__ import annotations

import argparse
import os
from datetime import datetime

try:
    from .config import TICKERS, resolve_date, settings
    from .db import get_engine
    from .load_to_db import (
        collect_8k_filings,
        collect_sec_sections,
        get_daily_start_date,
        load_market_data,
        upsert_8k_filings,
        upsert_sections,
    )
except ImportError:
    from config import TICKERS, resolve_date, settings
    from db import get_engine
    from load_to_db import (
        collect_8k_filings,
        collect_sec_sections,
        get_daily_start_date,
        load_market_data,
        upsert_8k_filings,
        upsert_sections,
    )


def env_enabled(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Refresh FinSight market data and optional SEC filing data for the latest period."
    )
    parser.add_argument(
        "--end-date",
        default=settings.end_date,
        help='Inclusive refresh end date, YYYY-MM-DD or "today".',
    )
    parser.add_argument(
        "--refresh-days",
        type=int,
        default=int(os.getenv("FINSIGHT_DAILY_REFRESH_DAYS", "10")),
        help="Re-fetch this many days before the latest stored market date.",
    )
    parser.add_argument(
        "--with-8k",
        action=argparse.BooleanOptionalAction,
        default=env_enabled("FINSIGHT_DAILY_LOAD_8K", default=False),
        help="Refresh current-year 8-K metadata. Enable detail previews with FINSIGHT_LOAD_8K_DETAILS=1.",
    )
    parser.add_argument(
        "--with-10k-sections",
        action=argparse.BooleanOptionalAction,
        default=env_enabled("FINSIGHT_DAILY_LOAD_10K_SECTIONS", default=False),
        help="Refresh current-year 10-K MD&A and Risk sections.",
    )
    return parser.parse_args()


def refresh_daily_data() -> None:
    args = parse_args()
    engine = get_engine()
    end_date = resolve_date(args.end_date)
    end_year = datetime.strptime(end_date, "%Y-%m-%d").year
    market_start_date = get_daily_start_date(engine, settings.start_date, args.refresh_days)

    print(f"Daily refresh tickers: {len(TICKERS)}")
    print(f"Market window: {market_start_date} through {end_date}")
    market_rows, sp500_rows = load_market_data(engine, market_start_date, end_date)

    if args.with_10k_sections:
        print(f"Refreshing 10-K sections filed in {end_year}")
        risk_rows, mdna_rows = collect_sec_sections(TICKERS, end_year, end_year)
        upsert_sections(risk_rows, engine, "risk_sections")
        upsert_sections(mdna_rows, engine, "mdna_sections")
    else:
        print("Skipping 10-K sections. Pass --with-10k-sections to refresh them.")

    if args.with_8k:
        print(f"Refreshing 8-K filings filed in {end_year}")
        eight_k_rows = collect_8k_filings(TICKERS, end_year, end_year)
        upsert_8k_filings(eight_k_rows, engine)
    else:
        print("Skipping 8-K filings. Pass --with-8k to refresh them.")

    print("Daily refresh complete.")
    print(f"Fetched stock rows: {len(market_rows)}")
    print(f"Fetched S&P 500 rows: {len(sp500_rows)}")


if __name__ == "__main__":
    refresh_daily_data()
