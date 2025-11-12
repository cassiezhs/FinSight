import yfinance as yf
import pandas as pd
import time

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
    tickers = [
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

    start_date = "2024-01-01"
    end_date = "2025-09-30"

    all_data = []

    for i, ticker in enumerate(tickers, 1):
        print(f"⏳ Fetching {i}/{len(tickers)}: {ticker}")
        df = fetch_stock_data(ticker, start_date, end_date)
        if not df.empty:
            all_data.append(df)
        time.sleep(1.2)  # prevent hitting rate limits

    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        print(f"\n✅ Finished. Combined shape: {combined_df.shape}")
        print(combined_df.head())
    else:
        print("❌ No stock data collected.")

