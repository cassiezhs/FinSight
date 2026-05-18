"""Utilities for aligning SEC filing dates to market trading dates."""

from __future__ import annotations

import pandas as pd


def map_to_first_trading_day(
    filing_date: pd.Timestamp,
    ticker: str,
    prices: pd.DataFrame,
    date_col: str = "date",
    ticker_col: str = "ticker",
) -> pd.Timestamp:
    """Return the first trading date on or after a filing date for a ticker."""
    if pd.isna(filing_date) or not ticker:
        return pd.NaT

    px = prices.loc[prices[ticker_col].astype(str).str.upper() == str(ticker).upper()].copy()
    if px.empty:
        return pd.NaT

    dates = pd.to_datetime(px[date_col], errors="coerce").dropna().sort_values()
    valid_dates = dates.loc[dates >= pd.Timestamp(filing_date)]
    if valid_dates.empty:
        return pd.NaT
    return valid_dates.iloc[0]


def add_event_dates(
    filings: pd.DataFrame,
    prices: pd.DataFrame,
    filing_date_col: str = "filing_date",
    ticker_col: str = "ticker",
    output_col: str = "event_date",
) -> pd.DataFrame:
    """Add first-trading-day event dates to a filings DataFrame."""
    out = filings.copy()
    out[output_col] = out.apply(
        lambda row: map_to_first_trading_day(row[filing_date_col], row[ticker_col], prices),
        axis=1,
    )
    return out

