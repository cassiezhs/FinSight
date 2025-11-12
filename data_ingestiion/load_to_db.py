#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Full Data Ingestion Script:
- Fetches SEC 10-K Risk & MD&A sections (via fetch_sec.py)
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
import pandas as pd
import yfinance as yf
from sqlalchemy import create_engine
from dotenv import load_dotenv

# Import SEC fetch utilities
from fetch_sec import (
    get_cik, get_10k_meta_for_year, get_10k_html_url,
    extract_risk_from_main_html, extract_mdna_from_main_html
)

# ---------------- CONFIG ----------------
load_dotenv()

DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME")
DB_SCHEMA = os.getenv("DB_SCHEMA", "public")

START_DATE = "2024-01-01"
END_DATE = "2025-09-30"
START_YEAR = 2018
END_YEAR = 2025

# ----------------------------------------
TICKERS = [
    "AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA", "TSLA", "BRK-B", "UNH", "JNJ",
    "V", "XOM", "PG", "JPM", "MA", "HD", "LLY", "CVX", "MRK", "PEP",
    "KO", "ABBV", "BAC", "COST", "AVGO", "TMO", "DIS", "WMT", "ADBE", "CRM",
    "NFLX", "PFE", "MCD", "TXN", "ABT", "DHR", "INTC", "NKE", "VZ", "QCOM",
    "MDT", "NEE", "ACN", "AMGN", "LOW", "MS", "SBUX", "UPS", "RTX", "LIN",
    "HON", "UNP", "INTU", "BA", "LMT", "CAT", "T", "ISRG", "PLD", "NOW",
    "GILD", "SPGI", "BLK", "ELV", "BKNG", "ZTS", "MO", "DE", "CI", "C",
    "SCHW", "MDLZ", "SO", "ADP", "SYK", "MMC", "PNC", "AXP", "ETN", "TJX",
    "FDX", "APD", "REGN", "CL", "ADSK", "BSX", "EMR", "WBA", "HUM", "BIIB",
    "ORCL", "GD", "CMCSA", "CSCO", "GM", "PYPL", "TGT", "EBAY", "BK", "COF"
]

# --------------- DB Helpers ---------------
def get_engine():
    if not all([DB_USER, DB_PASSWORD, DB_HOST, DB_NAME]):
        raise RuntimeError("❌ Missing DB env vars: DB_USER, DB_PASSWORD, DB_HOST, DB_NAME")
    url = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    return create_engine(url)

def save_to_db(df: pd.DataFrame, table_name: str, engine, replace=False):
    if df.empty:
        print(f"⚠️ No data to save for {table_name}")
        return

    if replace:
        df.to_sql(table_name, engine, schema=DB_SCHEMA, if_exists="replace", index=False)
        print(f"✅ Replaced all data in {DB_SCHEMA}.{table_name}")
    else:
        try:
            existing = pd.read_sql(f'SELECT DISTINCT "Date", "ticker" FROM "{DB_SCHEMA}"."{table_name}"', engine)
            existing_keys = set(map(tuple, existing.values))
            mask_new = ~df.apply(lambda r: (r.get("Date"), r.get("ticker")) in existing_keys, axis=1)
            df_new = df.loc[mask_new]
        except Exception:
            df_new = df

        if not df_new.empty:
            df_new.to_sql(table_name, engine, schema=DB_SCHEMA, if_exists="append", index=False)
            print(f"✅ Appended {len(df_new)} new rows to {DB_SCHEMA}.{table_name}")
        else:
            print(f"ℹ️ No new rows to add for {table_name}")

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
def collect_sec_sections(tickers, start_year, end_year):
    """Collect Risk and MD&A sections for all tickers."""
    rows_risk, rows_mdna = [], []
    for tkr in tickers:
        cik = get_cik(tkr)
        if not cik:
            print(f"⚠️ No CIK for {tkr}")
            continue
        for yr in range(start_year, end_year + 1):
            try:
                idx_url, filing_date = get_10k_meta_for_year(cik, yr)
                if not idx_url:
                    continue
                html_url = get_10k_html_url(idx_url)
                if not html_url:
                    continue
                time.sleep(0.25)  # polite to SEC servers

                risk = extract_risk_from_main_html(html_url)
                mdna = extract_mdna_from_main_html(html_url)
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
                    "content": risk,
                    "chunk_index": 0,
                    "ticker": tkr,
                })
                rows_mdna.append({
                    "cik": cik,
                    "company_name": company_name,
                    "filing_date": filing_date,
                    "content": mdna,
                    "chunk_index": 0,
                    "ticker": tkr,
                })
            except Exception as e:
                print(f"❌ Error {tkr} {yr}: {e}")
                continue

    df_risk = pd.DataFrame(rows_risk)
    df_mdna = pd.DataFrame(rows_mdna)
    for df in [df_risk, df_mdna]:
        if not df.empty:
            df["filing_date"] = pd.to_datetime(df["filing_date"], errors="coerce")
    return df_risk, df_mdna

def upsert_sections(df, engine, table):
    """Upsert section data by (cik, filing_date, chunk_index)."""
    if df.empty:
        print(f"⚠️ No rows to insert for {table}")
        return
    try:
        existing = pd.read_sql(f'SELECT cik, filing_date, chunk_index FROM "{DB_SCHEMA}"."{table}"', engine)
        existing["filing_date"] = pd.to_datetime(existing["filing_date"])
        existing_keys = set(map(tuple, existing[["cik", "filing_date", "chunk_index"]].values))
        mask_new = ~df.apply(lambda r: (r["cik"], pd.to_datetime(r["filing_date"]), r["chunk_index"]) in existing_keys, axis=1)
        df_new = df.loc[mask_new]
    except Exception:
        df_new = df

    if df_new.empty:
        print(f"ℹ️ No new rows for {table}")
        return

    df_new.to_sql(table, engine, schema=DB_SCHEMA, if_exists="append", index=False)
    print(f"✅ Inserted {len(df_new)} new rows into {DB_SCHEMA}.{table}")

# --------------- MAIN ---------------
if __name__ == "__main__":
    engine = get_engine()

    # --- 1. Stock & Benchmark ---
    all_data = []
    for ticker in TICKERS:
        df = fetch_stock_data(ticker, START_DATE, END_DATE)
        if not df.empty:
            all_data.append(df)
    combined_df = pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()

    save_to_db(combined_df, "stock_prices", engine, replace=False)
    sp500_df = fetch_sp500_data(START_DATE, END_DATE)
    save_to_db(sp500_df, "sp500_index", engine, replace=False)

    # --- 2. SEC Filings (Risk + MD&A) ---
    df_risk, df_mdna = collect_sec_sections(TICKERS, START_YEAR, END_YEAR)
    upsert_sections(df_risk, engine, "risk_sections")
    upsert_sections(df_mdna, engine, "mdna_sections")

    # --- 3. Summary ---
    print("\n📊 Data Load Complete!")
    if not combined_df.empty:
        print(f"Stock prices: {len(combined_df)} rows")
    if not sp500_df.empty:
        print(f"S&P 500: {len(sp500_df)} rows")
    if not df_risk.empty:
        print(f"Risk sections: {len(df_risk)} rows")
    if not df_mdna.empty:
        print(f"MD&A sections: {len(df_mdna)} rows")
