import os
import pandas as pd
from sqlalchemy import create_engine
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")

def get_engine():
    url = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    return create_engine(url)

# Save stock price dataframe
def save_stock_data(df, ticker, engine):
    df = df.copy()
    df['ticker'] = ticker
    df.reset_index(inplace=True)  

    # Optional: Select only necessary columns to match your DB schema
    df = df[['Date', 'Open', 'Close', 'Volume', 'ticker']]
    df.columns = ['date', 'open', 'close', 'volume', 'ticker']  # Rename for DB schema

    df.to_sql("stock_prices", engine, if_exists='append', index=False)
    print(f"✅ Saved {len(df)} rows of stock data for {ticker}")

# Save MD&A text
def save_mdna_text(text, cik, company_name, ticker, filing_date, engine, chunk_size=3000):
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    rows = [{
        "cik": cik,
        "company_name": company_name,
        "filing_date": filing_date,
        "chunk_index": idx,
        "content": chunk,
        "inserted_at": datetime.now(),
        "ticker":ticker
    } for idx, chunk in enumerate(chunks)]

    df = pd.DataFrame(rows)
    df.to_sql("mdna_sections", engine, if_exists='append', index=False)
    print(f"✅ Saved {len(df)} chunks for {company_name} ({cik}) on {filing_date}")


if __name__ == "__main__":
    from fetch_stock import fetch_stock_data
    from fetch_sec import extract_mdna_from_main_html  

    tickers = {
        "AAPL": "Apple Inc.",
        "GOOGL": "Alphabet Inc.",
        "MSFT": "Microsoft Corp.",
        "NVDA": "NVIDIA Corp."
    }
    engine = get_engine()
    for ticker, company_name in tickers.items():
        # Save stock data
        stock_df = fetch_stock_data(ticker, "2023-01-01", "2024-01-01")
        #if not stock_df.empty:
        #    save_stock_data(stock_df, ticker, engine)


    # Save MD&A text
    mdna_url = "https://www.sec.gov/Archives/edgar/data/320193/000032019324000123/aapl-20240928.htm"
    mdna_text = extract_mdna_from_main_html(mdna_url)
    print(mdna_text)
    cik = "0000320193"  # or use get_cik(ticker) again if you want to make it dynamic
    company_name = "Apple Inc."
    ticker = 'AAPL'

    save_mdna_text(mdna_text, cik, company_name, ticker, "2024-09-28", engine)

