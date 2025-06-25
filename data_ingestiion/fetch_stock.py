import yfinance as yf

def fetch_stock_data(ticker_symbol, start_date, end_date):
    ticker = yf.Ticker(ticker_symbol)
    df = ticker.history(start=start_date, end=end_date)
    return df[['Open', 'Close', 'Volume']]

if __name__ == "__main__":
    df = fetch_stock_data("AAPL", "2023-01-01", "2024-01-01")
    print(df.head())