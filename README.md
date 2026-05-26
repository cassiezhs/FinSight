# FinSight

## Financial Data Pipeline and Dash Insight Dashboard

![FinSight dashboard](title_img.png)

FinSight combines stock market performance with company financial narrative extracted from SEC filings. The current `main` branch runs a Python Dash dashboard backed by PostgreSQL data ingestion jobs.

The dashboard is centered on a practical question:

> Did management's narrative change before, with, or against the market reaction?

## What The Dashboard Does

- Visualizes historical stock prices and trading volume with Plotly charts.
- Marks 10-K and 8-K filing events on the price timeline.
- Compares selected 10-K MD&A and Risk Factors sections with prior filings.
- Shows readability, word-count, sentiment, market reaction, and post-filing return signals.
- Calculates 1-day, 5-day, and 30-day post-10-K returns and excess returns versus the S&P 500.
- Supports optional OpenAI summaries for filing-language changes.
- Stores stock prices, benchmark prices, and filing data in PostgreSQL.

## Data Sources

- Daily stock prices and S&P 500 benchmark data from Yahoo Finance through `yfinance`.
- SEC EDGAR metadata and filing documents for:
  - 10-K MD&A sections
  - 10-K Risk Factors sections
  - 8-K filing events and detail previews

## Tech Stack

- Python Dash and Dash Bootstrap Components
- Plotly
- PostgreSQL or Neon Postgres
- SQLAlchemy and `psycopg2`
- Yahoo Finance via `yfinance`
- SEC parsing with Requests and BeautifulSoup
- Optional OpenAI API summaries
- Optional FinBERT research pipeline for filing sentiment/modeling experiments

## Project Layout

```text
app/
  dashoboard.py              # Dash application
  assets/style.css           # Dashboard styling
data_ingestiion/
  config.py                  # Environment/config handling
  db.py                      # SQLAlchemy engine setup
  daily_refresh.py           # Daily scheduler entrypoint
  fetch_sec.py               # SEC metadata and filing extraction
  fetch_stock.py             # Yahoo Finance fetch helper
  load_to_db.py              # Historical and optional SEC ingestion
  event_mapping.py           # Filing-date to trading-date utilities
  modeling_pipeline_2.py     # FinBERT/modeling research workflow
tests/
  test_event_mapping.py
  test_fetch_sec.py
wsgi.py                      # WSGI entrypoint for Dash deployment
```

The source tree keeps the existing `data_ingestiion` and `dashoboard.py` spellings to avoid breaking imports and run commands.

## Setup

Create a virtual environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
cp .env.example .env
```

Configure `.env`.

For Neon or another hosted Postgres provider, prefer a single URL:

```env
DATABASE_URL=postgresql://USER:PASSWORD@HOST/DATABASE?sslmode=require
DB_SCHEMA=public
```

For local Postgres, the split variables also work:

```env
DB_USER=postgres
DB_PASSWORD=postgres
DB_HOST=localhost
DB_PORT=5432
DB_NAME=finsight
DB_SCHEMA=public
```

Set a real SEC contact user agent before filing ingestion:

```env
SEC_USER_AGENT=FinSight/0.1 your-email@example.com
```

OpenAI is optional:

```env
OPENAI_ENABLED=0
OPENAI_API_KEY=
OPENAI_MODEL=gpt-4o-mini
```

Enable it only when live AI summaries are needed:

```env
OPENAI_ENABLED=1
```

## Ticker Coverage

Ingestion jobs read `FINSIGHT_TICKERS`.

```env
FINSIGHT_TICKERS=AAPL,MSFT,NVDA
```

If the variable is omitted, the repo falls back to the default ticker universe in `data_ingestiion/config.py`.

The dashboard dropdown is database-driven. It shows tickers already present in `stock_prices`, even if a scheduled refresh job only updates a smaller ticker list.

## Ingest Data

Load historical stock prices and S&P 500 data:

```bash
python3 data_ingestiion/load_to_db.py
```

Use `today` as the configured end date when a historical backfill should run through the current date:

```bash
FINSIGHT_END_DATE=today python3 data_ingestiion/load_to_db.py
```

Include 10-K MD&A and Risk Factors extraction:

```bash
FINSIGHT_LOAD_SEC=1 python3 data_ingestiion/load_to_db.py
```

Include 8-K metadata and optional detail previews:

```bash
FINSIGHT_LOAD_8K=1 FINSIGHT_LOAD_8K_DETAILS=1 python3 data_ingestiion/load_to_db.py
```

## Daily Refresh

Use the dedicated scheduler entrypoint for daily market refreshes:

```bash
python3 -m data_ingestiion.daily_refresh
```

By default it:

- refreshes a rolling price window
- updates stock prices and the S&P 500 benchmark
- inserts missing rows without replacing existing history
- uses `FINSIGHT_DAILY_REFRESH_DAYS`, defaulting to 10 days

Refresh current-year SEC data explicitly when needed:

```bash
python3 -m data_ingestiion.daily_refresh --with-8k --with-10k-sections
```

Equivalent environment switches:

```env
FINSIGHT_DAILY_LOAD_8K=1
FINSIGHT_DAILY_LOAD_10K_SECTIONS=1
```

For GitHub Actions, store credentials in repository secrets. A daily job step can look like:

```yaml
- name: Refresh FinSight data
  env:
    DATABASE_URL: ${{ secrets.DATABASE_URL }}
    SEC_USER_AGENT: ${{ vars.SEC_USER_AGENT }}
    FINSIGHT_END_DATE: today
    FINSIGHT_DAILY_REFRESH_DAYS: 10
    FINSIGHT_DAILY_LOAD_8K: "1"
    FINSIGHT_DAILY_LOAD_10K_SECTIONS: "0"
  run: python data_ingestiion/daily_refresh.py
```

## Run The Dashboard

Start the Dash app locally:

```bash
python3 app/dashoboard.py
```

Open the local Dash URL, normally:

```text
http://127.0.0.1:8050
```

For a production WSGI server, use:

```bash
gunicorn wsgi:server
```

## Deployment

Deploy the Dash app with a Python environment that installs `requirements.txt`, provides the database and SEC environment variables, and starts:

```bash
gunicorn wsgi:server
```

Store `DATABASE_URL`, `SEC_USER_AGENT`, and optional OpenAI variables in the hosting provider's environment settings.

## Tests

```bash
pytest
```

## Research Pipeline

The dashboard does not display raw regression outputs. The modeling scripts are kept as a research workflow for FinBERT-derived filing tone analysis and statistical experiments around narrative polarity and forward returns:

```bash
python3 data_ingestiion/modeling_pipeline_2.py
```
