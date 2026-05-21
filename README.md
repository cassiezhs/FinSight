# FinSight

# Financial Data Pipeline and Insight Dashboard

![Dashboard Mockup](title_img.png)

This dashboard combines **stock market performance** with **company financial narrative** extracted from 10-K filings to help analysts, investors, and researchers compare management commentary with stock trends.

---

## 🚀 Features

- 📉 Interactive stock price visualization (historical data from Yahoo Finance)
- 🧾 Extracted MD&A section from the latest 10-K filings (via SEC EDGAR)
- 🔍 Compare narrative and market movement side-by-side
- 🗃️ Data stored in PostgreSQL for scalability
- 🌐 Web app built with Python Dash

---

## 📦 Tech Stack

- **Backend**: Python, PostgreSQL, SQLAlchemy
- **Frontend**: Dash, Plotly
- **Data Sources**:
  - Yahoo Finance API (via `yfinance`)
  - SEC EDGAR API for 10-K filings
- **Other Tools**: `psycopg2`, `BeautifulSoup`, `dotenv`, `pandas`

---

## 📂 Project Structure

```text
app/
  dashoboard.py              # Dash application
  assets/style.css           # Dashboard styling
data_ingestiion/
  config.py                  # Shared env/config defaults
  db.py                      # SQLAlchemy engine helper
  fetch_sec.py               # SEC filing metadata and section extraction
  fetch_stock.py             # Yahoo Finance stock fetcher
  load_to_db.py              # Price/filing ingestion into PostgreSQL
  event_mapping.py           # Filing date to trading date utilities
  modeling_pipeline_2.py     # Main modeling workflow
tests/
  test_event_mapping.py
  test_fetch_sec.py
```

The source folder currently keeps the original `data_ingestiion` and `dashoboard.py` spellings for compatibility with existing imports and run commands.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
cp .env.example .env
```

Edit `.env` with your PostgreSQL connection details. OpenAI is disabled by default with `OPENAI_ENABLED=0`; set `OPENAI_ENABLED=1` and provide `OPENAI_API_KEY` only when you want live AI summaries.

## Run

Load stock prices and S&P 500 data:

```bash
python3 data_ingestiion/load_to_db.py
```

Backfill prices through today, for example through May 18, 2026 when run on that date:

```bash
FINSIGHT_END_DATE=today python3 data_ingestiion/load_to_db.py
```

Run the lightweight daily updater. It re-fetches the last 10 days by default and inserts only rows that are not already in PostgreSQL:

```bash
python3 data_ingestiion/load_to_db.py --daily
```

For schedulers such as GitHub Actions, use the dedicated daily entrypoint:

```bash
python3 -m data_ingestiion.daily_refresh
```

That command refreshes stock prices and the S&P 500 rolling window. Add SEC filing refreshes explicitly when the scheduled job should include them:

```bash
python3 -m data_ingestiion.daily_refresh --with-8k --with-10k-sections
```

The daily job reads `DATABASE_URL`, `SEC_USER_AGENT`, `FINSIGHT_TICKERS`, `FINSIGHT_DAILY_REFRESH_DAYS`, and optional SEC flags from the environment. Store secrets such as `DATABASE_URL` in GitHub Actions secrets, not in the repo.

Schedule it with cron, usually after the US market close:

```cron
30 18 * * 1-5 cd /path/to/FinSight && mkdir -p logs && .venv/bin/python data_ingestiion/load_to_db.py --daily >> logs/daily_ingestion.log 2>&1
```

Load SEC Risk Factors and MD&A sections as part of ingestion:

```bash
FINSIGHT_LOAD_SEC=1 python3 data_ingestiion/load_to_db.py
```

Run the modeling pipeline:

```bash
python3 data_ingestiion/modeling_pipeline_2.py
```

Run the dashboard:

```bash
python3 app/dashoboard.py
```

Run tests:

```bash
pytest
```
