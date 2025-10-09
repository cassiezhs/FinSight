import os
import pandas as pd
from sqlalchemy import create_engine, text
from datetime import datetime
from dotenv import load_dotenv

# -------------------------------------------------
#  Imports: use fetch_sec for all SEC-related logic
# -------------------------------------------------
from fetch_sec import (
    get_cik,
    get_10k_url_for_year,
    get_10k_html_url,
    extract_mdna_from_main_html
)
from fetch_stock import fetch_stock_data

import requests

# -------------------- CONFIG --------------------
load_dotenv()
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")

def get_engine():
    url = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    return create_engine(url)


# -------------------- SAVE HELPERS --------------------
def save_stock_data(df, ticker, engine):
    df = df.copy()
    df["ticker"] = ticker
    df.reset_index(inplace=True)
    df = df[["Date", "Open", "Close", "Volume", "ticker"]]
    df.columns = ["date", "open", "close", "volume", "ticker"]

    min_date = df["date"].min().date()
    max_date = df["date"].max().date()

    with engine.connect() as conn:
        q = text("""
            SELECT COUNT(*) FROM stock_prices
            WHERE ticker = :ticker AND date BETWEEN :start_date AND :end_date
        """)
        result = conn.execute(q, {
            "ticker": ticker,
            "start_date": min_date,
            "end_date": max_date
        }).scalar()

    if result and result > 0:
        print(f"⏭️ Skipping {ticker}: data already exists between {min_date} and {max_date}.")
        return

    df.to_sql("stock_prices", engine, if_exists="append", index=False)
    print(f"✅ Saved {len(df)} rows of stock data for {ticker} ({min_date} → {max_date})")


def mdna_exists(engine, cik, ticker, filing_date):
    with engine.connect() as conn:
        q = text("""
            SELECT 1 FROM mdna_sections
            WHERE cik = :cik AND ticker = :ticker AND filing_date = :filing_date
            LIMIT 1
        """)
        res = conn.execute(q, {"cik": cik, "ticker": ticker, "filing_date": filing_date}).fetchone()
        return res is not None


def save_mdna_text(text, cik, company_name, ticker, filing_date, engine, chunk_size=3000):
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    if not chunks:
        print(f"⚠️ No text to save for {ticker} {filing_date}")
        return

    rows = [{
        "cik": cik,
        "company_name": company_name,
        "filing_date": filing_date,
        "chunk_index": idx,
        "content": chunk,
        "inserted_at": datetime.now(),
        "ticker": ticker
    } for idx, chunk in enumerate(chunks)]
    pd.DataFrame(rows).to_sql("mdna_sections", engine, if_exists="append", index=False)
    print(f"✅ Saved {len(rows)} chunks for {ticker} ({filing_date})")


# -------------------- MAIN LOGIC --------------------
if __name__ == "__main__":
    engine = get_engine()

    tickers = {
        "AAPL": "Apple Inc.",
        "GOOGL": "Alphabet Inc.",
        "MSFT": "Microsoft Corp.",
        "NVDA": "NVIDIA Corp."
    }

    # ---- 1️⃣ Save stock data ----
    for ticker, company_name in tickers.items():
        stock_df = fetch_stock_data(ticker, "2024-01-01", "2025-09-30")
        if not stock_df.empty:
            save_stock_data(stock_df, ticker, engine)

    # ---- 2️⃣ Save Apple MD&A (2023–2025) ----
    ticker = "GOOGL"
    company_name = "Alphabet Inc."
    cik = get_cik(ticker)

    for year in [2023, 2024, 2025]:
        filing_date = f"{year}-09-30"

        # skip if already saved
        if mdna_exists(engine, cik, ticker, filing_date):
            print(f"⏭️ Skipping {ticker} {year} ({filing_date}): already in mdna_sections")
            continue

        idx_url = get_10k_url_for_year(cik, year)
        if not idx_url:
            print(f"⚠️ No 10-K index.json found for {ticker} in {year}")
            continue

        html_url = get_10k_html_url(idx_url)
        if not html_url:
            print(f"⚠️ No HTML found for {ticker} {year}")
            continue

        mdna_text = extract_mdna_from_main_html(html_url)
        save_mdna_text(mdna_text, cik, company_name, ticker, filing_date, engine)
