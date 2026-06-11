#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Full Data Ingestion Script:
- Fetches SEC 10-Q/10-K Risk & MD&A sections (via fetch_sec.py)
- Fetches stock prices & S&P 500 index (via yfinance)
- Writes everything to PostgreSQL with de-duplication

Tables created/updated:
    - stock_prices
    - sp500_index
    - risk_section
    - mdna_section
"""

import os
import time
import uuid
import argparse
from datetime import datetime, timedelta
import pandas as pd
import yfinance as yf
from sqlalchemy import inspect, text
from sqlalchemy.exc import IntegrityError

try:
    from .config import TICKERS, next_date, resolve_date, settings
    from .db import get_engine
    from .fetch_sec import (
        get_company_ticker_ciks, get_10k_meta_for_year, get_filing_meta_for_year, get_10k_html_url,
        extract_risk_from_main_html, extract_mdna_from_main_html,
        get_8k_meta, extract_8k_detail_preview,
    )
except ImportError:
    from config import TICKERS, next_date, resolve_date, settings
    from db import get_engine
    from fetch_sec import (
        get_company_ticker_ciks, get_10k_meta_for_year, get_filing_meta_for_year, get_10k_html_url,
        extract_risk_from_main_html, extract_mdna_from_main_html,
        get_8k_meta, extract_8k_detail_preview,
    )

# --------------- DB Helpers ---------------
DB_SCHEMA = settings.db_schema


def _quote_identifier(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _qualified(table_name: str) -> str:
    return f'{_quote_identifier(DB_SCHEMA)}.{_quote_identifier(table_name)}'


def _ensure_unique_index(engine, table_name: str, key_columns: list[str]):
    if not key_columns:
        return
    index_name = f"uq_{table_name}_{'_'.join(c.lower().replace(' ', '_') for c in key_columns)}"
    cols = ", ".join(_quote_identifier(c) for c in key_columns)
    with engine.begin() as conn:
        conn.execute(text(f"CREATE UNIQUE INDEX IF NOT EXISTS {_quote_identifier(index_name)} ON {_qualified(table_name)} ({cols})"))


def append_on_conflict_do_nothing(df: pd.DataFrame, table_name: str, engine, key_columns: list[str]):
    """Append rows using PostgreSQL uniqueness instead of loading existing keys into pandas."""
    if df.empty:
        print(f"⚠️ No data to save for {table_name}")
        return

    inspector = inspect(engine)
    if not inspector.has_table(table_name, schema=DB_SCHEMA):
        df.head(0).to_sql(table_name, engine, schema=DB_SCHEMA, if_exists="fail", index=False)

    tmp_name = f"_tmp_{table_name}_{uuid.uuid4().hex[:8]}"
    df.to_sql(tmp_name, engine, schema=DB_SCHEMA, if_exists="fail", index=False)

    cols = list(df.columns)
    col_sql = ", ".join(_quote_identifier(c) for c in cols)
    conflict_sql = ", ".join(_quote_identifier(c) for c in key_columns)
    match_sql = " AND ".join(
        f"target.{_quote_identifier(c)} = source.{_quote_identifier(c)}"
        for c in key_columns
    )
    try:
        try:
            _ensure_unique_index(engine, table_name, key_columns)
            insert_sql = (
                f"INSERT INTO {_qualified(table_name)} ({col_sql}) "
                f"SELECT {col_sql} FROM {_qualified(tmp_name)} "
                f"ON CONFLICT ({conflict_sql}) DO NOTHING"
            )
        except IntegrityError:
            print(f"⚠️ Existing duplicates in {DB_SCHEMA}.{table_name}; inserting only missing keys.")
            insert_sql = (
                f"INSERT INTO {_qualified(table_name)} ({col_sql}) "
                f"SELECT {col_sql} FROM {_qualified(tmp_name)} source "
                f"WHERE NOT EXISTS ("
                f"SELECT 1 FROM {_qualified(table_name)} target WHERE {match_sql}"
                f")"
            )

        with engine.begin() as conn:
            result = conn.execute(text(insert_sql))
        print(f"✅ Inserted {result.rowcount} new rows into {DB_SCHEMA}.{table_name}")
    finally:
        with engine.begin() as conn:
            conn.execute(text(f"DROP TABLE IF EXISTS {_qualified(tmp_name)}"))


def save_to_db(df: pd.DataFrame, table_name: str, engine, replace=False):
    if df.empty:
        print(f"⚠️ No data to save for {table_name}")
        return

    if replace:
        df.to_sql(table_name, engine, schema=DB_SCHEMA, if_exists="replace", index=False)
        print(f"✅ Replaced all data in {DB_SCHEMA}.{table_name}")
    else:
        keys = ["Date"] if table_name == "sp500_index" else ["Date", "ticker"]
        append_on_conflict_do_nothing(df, table_name, engine, keys)


def get_max_market_date(engine) -> str | None:
    inspector = inspect(engine)
    if not inspector.has_table("stock_prices", schema=DB_SCHEMA):
        return None

    with engine.begin() as conn:
        result = conn.execute(text(f'SELECT MAX("Date") FROM {_qualified("stock_prices")}')).scalar()
    if result is None:
        return None
    return pd.to_datetime(result).date().isoformat()


def get_ticker_latest_market_dates(engine) -> dict[str, str]:
    inspector = inspect(engine)
    if not inspector.has_table("stock_prices", schema=DB_SCHEMA):
        return {}

    query = text(f'SELECT ticker, MAX("Date") AS latest_date FROM {_qualified("stock_prices")} GROUP BY ticker')
    with engine.begin() as conn:
        rows = conn.execute(query).mappings().all()

    return {
        str(row["ticker"]).upper(): pd.to_datetime(row["latest_date"]).date().isoformat()
        for row in rows
        if row["ticker"] and row["latest_date"] is not None
    }


def get_daily_start_date(
    engine,
    fallback_start_date: str,
    refresh_days: int,
    tickers: tuple[str, ...] | None = None,
) -> str:
    latest_dates = get_ticker_latest_market_dates(engine)
    if not latest_dates:
        return fallback_start_date

    expected_tickers = {ticker.upper() for ticker in (tickers or latest_dates.keys())}
    covered_dates = [latest_dates[ticker] for ticker in expected_tickers if ticker in latest_dates]
    if not covered_dates:
        return fallback_start_date

    earliest_latest_date = min(covered_dates)
    start = datetime.strptime(earliest_latest_date, "%Y-%m-%d").date() - timedelta(days=refresh_days)
    return start.isoformat()


# --------------- Market Data ---------------
def fetch_stock_data(ticker_symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch daily OHLCV data for a ticker and de-dupe."""
    try:
        tkr = yf.Ticker(ticker_symbol)
        df = tkr.history(start=start_date, end=end_date)
        if df.empty:
            print(f"⚠️ No stock data for {ticker_symbol}")
            return pd.DataFrame()
        df = df[['Open', 'Close', 'Volume']].reset_index()
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce').dt.tz_localize(None)
        df['ticker'] = ticker_symbol.upper()
        df = df.drop_duplicates(subset=['Date', 'ticker']).sort_values(['ticker', 'Date'])
        return df
    except Exception as e:
        print(f"❌ Failed {ticker_symbol}: {e}")
        return pd.DataFrame()

def fetch_sp500_data(start_date, end_date):
    """Fetch S&P 500 benchmark data."""
    try:
        sp = yf.Ticker("^GSPC").history(start=start_date, end=end_date)
        if sp.empty:
            print("⚠️ No S&P 500 data found.")
            return pd.DataFrame()
        sp = sp[['Close']].reset_index()
        sp.rename(columns={'Close': 'close'}, inplace=True)
        sp['Date'] = pd.to_datetime(sp['Date'], errors='coerce').dt.tz_localize(None)
        sp = sp.drop_duplicates(subset=['Date']).sort_values('Date')
        print("✅ S&P 500 fetched successfully.")
        return sp
    except Exception as e:
        print(f"❌ Failed to fetch S&P 500: {e}")
        return pd.DataFrame()

# --------------- SEC Data ---------------
def collect_sec_sections(tickers, start_year, end_year, form_types: tuple[str, ...] = ("10-K", "10-Q")):
    """Collect Risk and MD&A sections for all tickers and selected periodic filing forms."""
    rows_risk, rows_mdna = [], []
    cik_by_ticker = get_company_ticker_ciks()
    if not cik_by_ticker:
        print("⚠️ Skipping 10-Q/10-K sections because the SEC ticker directory is unavailable.")
        return pd.DataFrame(), pd.DataFrame()

    for tkr in tickers:
        cik = cik_by_ticker.get(tkr.upper())
        if not cik:
            print(f"⚠️ No CIK for {tkr}")
            continue
        for yr in range(start_year, end_year + 1):
            for form_type in form_types:
                try:
                    idx_url, filing_date, found_form = get_filing_meta_for_year(cik, yr, (form_type,))
                    if not idx_url:
                        continue
                    html_url = get_10k_html_url(idx_url)
                    if not html_url:
                        continue
                    time.sleep(0.25)  # polite to SEC servers

                    risk = extract_risk_from_main_html(html_url)
                    mdna = extract_mdna_from_main_html(html_url, found_form or form_type)
                    company_name = None
                    if "Alphabet" in risk or "Alphabet" in mdna:
                        company_name = "Alphabet Inc."
                    elif "Apple" in risk or "Apple" in mdna:
                        company_name = "Apple Inc."
                    else:
                        company_name = f"{tkr} Corp."

                    rows_risk.append({
                        "cik": cik,
                        "company_name": company_name,
                        "filing_date": filing_date,
                        "form_type": found_form or form_type,
                        "content": risk,
                        "chunk_index": 0,
                        "ticker": tkr,
                    })
                    rows_mdna.append({
                        "cik": cik,
                        "company_name": company_name,
                        "filing_date": filing_date,
                        "form_type": found_form or form_type,
                        "content": mdna,
                        "chunk_index": 0,
                        "ticker": tkr,
                    })
                except Exception as e:
                    print(f"❌ Error {tkr} {yr} {form_type}: {e}")
                    continue

    df_risk = pd.DataFrame(rows_risk)
    df_mdna = pd.DataFrame(rows_mdna)
    for df in [df_risk, df_mdna]:
        if not df.empty:
            df["filing_date"] = pd.to_datetime(df["filing_date"], errors="coerce")
    return df_risk, df_mdna

def upsert_sections(df, engine, table):
    """Upsert section data by (cik, filing_date, form_type, chunk_index)."""
    if df.empty:
        print(f"⚠️ No rows to insert for {table}")
        return
    ensure_section_table_columns(engine, table)
    key_columns = ["cik", "filing_date", "form_type", "chunk_index"]
    _ensure_unique_index(engine, table, key_columns)

    tmp_name = f"_tmp_{table}_{uuid.uuid4().hex[:8]}"
    df.to_sql(tmp_name, engine, schema=DB_SCHEMA, if_exists="fail", index=False)

    cols = list(df.columns)
    col_sql = ", ".join(_quote_identifier(c) for c in cols)
    conflict_sql = ", ".join(_quote_identifier(c) for c in key_columns)
    update_cols = [c for c in cols if c not in key_columns]
    update_sql = ", ".join(
        f"{_quote_identifier(c)} = EXCLUDED.{_quote_identifier(c)}"
        for c in update_cols
    )
    try:
        with engine.begin() as conn:
            result = conn.execute(text(
                f"INSERT INTO {_qualified(table)} ({col_sql}) "
                f"SELECT {col_sql} FROM {_qualified(tmp_name)} "
                f"ON CONFLICT ({conflict_sql}) DO UPDATE SET {update_sql}"
            ))
        print(f"✅ Upserted {result.rowcount} rows into {DB_SCHEMA}.{table}")
    finally:
        with engine.begin() as conn:
            conn.execute(text(f"DROP TABLE IF EXISTS {_qualified(tmp_name)}"))


def ensure_section_table_columns(engine, table_name: str):
    if not inspect(engine).has_table(table_name, schema=DB_SCHEMA):
        return
    with engine.begin() as conn:
        conn.execute(text(
            f"ALTER TABLE {_qualified(table_name)} "
            "ADD COLUMN IF NOT EXISTS form_type TEXT DEFAULT '10-K'"
        ))


def collect_8k_filings(tickers, start_year, end_year):
    rows = []
    include_details = os.getenv("FINSIGHT_LOAD_8K_DETAILS", "1") == "1"
    cik_by_ticker = get_company_ticker_ciks()
    if not cik_by_ticker:
        print("⚠️ Skipping 8-K filings because the SEC ticker directory is unavailable.")
        return pd.DataFrame()

    for tkr in tickers:
        try:
            cik = cik_by_ticker.get(tkr.upper())
            if not cik:
                print(f"⚠️ No CIK for {tkr}")
                continue
            filing_rows = get_8k_meta(cik, tkr, start_year, end_year)
            if include_details:
                for row in filing_rows:
                    preview, sources = extract_8k_detail_preview(
                        row.get("filing_url", ""),
                        row.get("filing_index_url", ""),
                        row.get("primary_document", ""),
                        row.get("items", ""),
                    )
                    row["detail_preview"] = preview
                    row["detail_sources"] = sources
                    time.sleep(0.12)
            rows.extend(filing_rows)
            time.sleep(0.25)
        except Exception as e:
            print(f"❌ Error loading 8-K metadata for {tkr}: {e}")

    df = pd.DataFrame(rows)
    if not df.empty:
        df["filing_date"] = pd.to_datetime(df["filing_date"], errors="coerce")
        df = df.drop_duplicates(subset=["cik", "accession_number"]).sort_values(["ticker", "filing_date"])
    return df


def ensure_8k_table_columns(engine):
    columns = {
        "item_descriptions": "TEXT",
        "filing_index_url": "TEXT",
        "detail_preview": "TEXT",
        "detail_sources": "TEXT",
    }
    with engine.begin() as conn:
        for column, sql_type in columns.items():
            conn.execute(text(
                f"ALTER TABLE {_qualified('sec_8k_filings')} "
                f"ADD COLUMN IF NOT EXISTS {_quote_identifier(column)} {sql_type}"
            ))


def upsert_8k_filings(df, engine):
    if df.empty:
        print("⚠️ No rows to insert for sec_8k_filings")
        return

    inspector = inspect(engine)
    if not inspector.has_table("sec_8k_filings", schema=DB_SCHEMA):
        df.head(0).to_sql("sec_8k_filings", engine, schema=DB_SCHEMA, if_exists="fail", index=False)
    ensure_8k_table_columns(engine)
    _ensure_unique_index(engine, "sec_8k_filings", ["cik", "accession_number"])

    tmp_name = f"_tmp_sec_8k_filings_{uuid.uuid4().hex[:8]}"
    df.to_sql(tmp_name, engine, schema=DB_SCHEMA, if_exists="fail", index=False)

    cols = list(df.columns)
    col_sql = ", ".join(_quote_identifier(c) for c in cols)
    update_cols = [
        c for c in cols
        if c not in {"cik", "accession_number"} and c in {
            "ticker", "filing_date", "form_type", "items", "item_descriptions",
            "primary_document", "primary_doc_description", "filing_url",
            "filing_index_url", "detail_preview", "detail_sources",
        }
    ]
    update_sql = ", ".join(
        f"{_quote_identifier(c)} = EXCLUDED.{_quote_identifier(c)}"
        for c in update_cols
    )

    try:
        with engine.begin() as conn:
            result = conn.execute(text(
                f"INSERT INTO {_qualified('sec_8k_filings')} ({col_sql}) "
                f"SELECT {col_sql} FROM {_qualified(tmp_name)} "
                f"ON CONFLICT ({_quote_identifier('cik')}, {_quote_identifier('accession_number')}) "
                f"DO UPDATE SET {update_sql}"
            ))
        print(f"✅ Upserted {result.rowcount} rows into {DB_SCHEMA}.sec_8k_filings")
    finally:
        with engine.begin() as conn:
            conn.execute(text(f"DROP TABLE IF EXISTS {_qualified(tmp_name)}"))


def parse_args():
    parser = argparse.ArgumentParser(description="Load FinSight market and filing data into PostgreSQL.")
    parser.add_argument("--start-date", default=settings.start_date, help="Inclusive start date, YYYY-MM-DD.")
    parser.add_argument(
        "--end-date",
        default=settings.end_date,
        help='Inclusive end date, YYYY-MM-DD or "today". Defaults to FINSIGHT_END_DATE.',
    )
    parser.add_argument(
        "--daily",
        action="store_true",
        help="Fetch a small rolling window ending today instead of the full configured history.",
    )
    parser.add_argument(
        "--refresh-days",
        type=int,
        default=int(os.getenv("FINSIGHT_DAILY_REFRESH_DAYS", "10")),
        help="Days to re-fetch before the latest stored date when --daily is used.",
    )
    return parser.parse_args()


def load_market_data(engine, start_date: str, end_date: str):
    yfinance_end_date = next_date(end_date)
    print(f"Loading market data from {start_date} through {resolve_date(end_date)}")

    all_data = []
    for ticker in TICKERS:
        df = fetch_stock_data(ticker, start_date, yfinance_end_date)
        if not df.empty:
            all_data.append(df)
    combined_df = pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()

    save_to_db(combined_df, "stock_prices", engine, replace=False)
    sp500_df = fetch_sp500_data(start_date, yfinance_end_date)
    save_to_db(sp500_df, "sp500_index", engine, replace=False)
    return combined_df, sp500_df


# --------------- MAIN ---------------
if __name__ == "__main__":
    args = parse_args()
    engine = get_engine()

    end_date = resolve_date(args.end_date)
    start_date = args.start_date
    if args.daily:
        start_date = get_daily_start_date(engine, settings.start_date, args.refresh_days, TICKERS)

    # --- 1. Stock & Benchmark ---
    combined_df, sp500_df = load_market_data(engine, start_date, end_date)

    # --- 2. SEC Filings (Risk + MD&A) ---
    if os.getenv("FINSIGHT_LOAD_SEC", "0") == "1":
        df_risk, df_mdna = collect_sec_sections(TICKERS, settings.start_year, settings.end_year)
        upsert_sections(df_risk, engine, "risk_sections")
        upsert_sections(df_mdna, engine, "mdna_sections")
    else:
        print("ℹ️ Skipping SEC section load. Set FINSIGHT_LOAD_SEC=1 to enable it.")

    if os.getenv("FINSIGHT_LOAD_8K", "0") == "1":
        df_8k = collect_8k_filings(TICKERS, settings.start_year, settings.end_year)
        upsert_8k_filings(df_8k, engine)
    else:
        print("ℹ️ Skipping 8-K metadata load. Set FINSIGHT_LOAD_8K=1 to enable it.")

    # --- 3. Summary ---
    print("\n📊 Data Load Complete!")
    if not combined_df.empty:
        print(f"Stock prices: {len(combined_df)} rows")
    if not sp500_df.empty:
        print(f"S&P 500: {len(sp500_df)} rows")
    # if not df_risk.empty:
    #     print(f"Risk sections: {len(df_risk)} rows")
    # if not df_mdna.empty:
    #     print(f"MD&A sections: {len(df_mdna)} rows")
