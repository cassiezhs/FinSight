import os
import yfinance as yf
import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv

# -------------------- CONFIG --------------------
load_dotenv()

DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")
DB_SCHEMA = os.getenv("DB_SCHEMA", "public")

# -------------------------------------------------
def get_engine():
    """Create a SQLAlchemy engine for PostgreSQL."""
    url = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    return create_engine(url)


def fetch_stock_data(ticker_symbol, start_date, end_date):
    """Fetch daily OHLCV data for a ticker."""
    try:
        ticker = yf.Ticker(ticker_symbol)
        df = ticker.history(start=start_date, end=end_date)
        if df.empty:
            print(f"⚠️ No stock data found for {ticker_symbol}")
            return pd.DataFrame()

        df = df[['Open', 'Close', 'Volume']].reset_index()
        df['ticker'] = ticker_symbol
        return df
    except Exception as e:
        print(f"❌ Failed to fetch stock data for {ticker_symbol}: {e}")
        return pd.DataFrame()


def fetch_sp500_data(start_date, end_date):
    """Fetch S&P 500 (^GSPC) benchmark data."""
    try:
        sp500 = yf.Ticker("^GSPC")
        df = sp500.history(start=start_date, end=end_date)
        if df.empty:
            print("⚠️ No S&P 500 data found.")
            return pd.DataFrame()

        df = df[['Close']].reset_index()
        df.rename(columns={'Close': 'close'}, inplace=True)
        print("✅ Fetched S&P 500 benchmark successfully.")
        return df
    except Exception as e:
        print(f"❌ Failed to fetch S&P 500: {e}")
        return pd.DataFrame()


def save_to_db(df, table_name, engine):
    """Write dataframe to PostgreSQL."""
    if df.empty:
        print(f"⚠️ No data to save for {table_name}")
        return
    try:
        df.to_sql(table_name, engine, schema=DB_SCHEMA, if_exists="replace", index=False)
        print(f"✅ Saved {len(df)} rows to {DB_SCHEMA}.{table_name}")
    except Exception as e:
        print(f"❌ Failed to write {table_name} to DB: {e}")


if __name__ == "__main__":
    tickers = ["GOOGL", "MSFT", "NVDA", "AAPL"]
    start_date = "2024-01-01"
    end_date = "2025-09-30"

    all_data = []
    for ticker in tickers:
        df = fetch_stock_data(ticker, start_date, end_date)
        if not df.empty:
            all_data.append(df)

    # Combine all stock data
    combined_df = pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()

    # Fetch S&P 500
    sp500_df = fetch_sp500_data(start_date, end_date)

    # Save to database
    engine = get_engine()
    save_to_db(combined_df, "stock_prices", engine)
    save_to_db(sp500_df, "sp500_index", engine)

    # Quick preview
    if not combined_df.empty:
        print("\n📈 Stock sample:")
        print(combined_df.head())
    if not sp500_df.empty:
        print("\n💹 S&P 500 sample:")
        print(sp500_df.head())
