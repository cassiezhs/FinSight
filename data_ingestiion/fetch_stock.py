import yfinance as yf
import pandas as pd

def fetch_stock_data(ticker_symbol, start_date, end_date):
    try:
        ticker = yf.Ticker(ticker_symbol)
        df = ticker.history(start=start_date, end=end_date)

        if df.empty:
            print(f"⚠️ No stock data found for {ticker_symbol} between {start_date} and {end_date}.")
            return pd.DataFrame()

        df = df[['Open', 'Close', 'Volume']].reset_index()
        df['ticker'] = ticker_symbol  # Add ticker column for later identification
        return df

    except Exception as e:
        print(f"❌ Failed to fetch stock data for {ticker_symbol}: {e}")
        return pd.DataFrame()

if __name__ == "__main__":
    tickers = ["GOOGL", "MSFT", "NVDA","AAPL"]
    start_date = "2024-01-01"
    end_date = "2025-09-30"

    all_data = []

    for ticker in tickers:
        df = fetch_stock_data(ticker, start_date, end_date)
        if not df.empty:
            all_data.append(df)

    combined_df = pd.concat(all_data, ignore_index=True)
    print(combined_df.head())
