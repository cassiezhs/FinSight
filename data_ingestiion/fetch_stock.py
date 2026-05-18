import yfinance as yf
import pandas as pd
import time

try:
    from .config import TICKERS, settings
except ImportError:
    from config import TICKERS, settings

def fetch_stock_data(ticker_symbol, start_date, end_date):
    try:
        ticker = yf.Ticker(ticker_symbol)
        df = ticker.history(start=start_date, end=end_date)

        if df.empty:
            print(f"⚠️ No stock data for {ticker_symbol}")
            return pd.DataFrame()

        df = df[['Open', 'Close', 'Volume']].reset_index()
        df['ticker'] = ticker_symbol
        return df

    except Exception as e:
        print(f"❌ Failed for {ticker_symbol}: {e}")
        return pd.DataFrame()

if __name__ == "__main__":
    all_data = []

    for i, ticker in enumerate(TICKERS, 1):
        print(f"⏳ Fetching {i}/{len(TICKERS)}: {ticker}")
        df = fetch_stock_data(ticker, settings.start_date, settings.end_date)
        if not df.empty:
            all_data.append(df)
        time.sleep(1.2)  # prevent hitting rate limits

    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        print(f"\n✅ Finished. Combined shape: {combined_df.shape}")
        print(combined_df.head())
    else:
        print("❌ No stock data collected.")
