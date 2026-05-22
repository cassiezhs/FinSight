# FinSight

## Narrative vs. Market

![FinSight dashboard](title_img.png)

FinSight is a React and FastAPI application for analyzing whether company filing language aligns with market behavior. It combines SEC 10-K narrative changes, 8-K events, daily stock prices, and S&P 500 benchmarks so users can move from raw disclosures to an investor-facing readout.

The dashboard is centered on a practical question:

> Did management's narrative change before, with, or against the market reaction?

## What The Dashboard Does

- Compares selected 10-K MD&A and Risk sections with the previous available filing.
- Surfaces new and removed filing language, readability shifts, word-count changes, and narrative tone changes.
- Produces a `Market Readout` that combines filing tone, event activity, price behavior, and post-filing reaction.
- Computes a `Narrative vs Market Alignment` signal from:
  - filing tone change
  - new risk language count
  - post-10-K excess return versus the S&P 500
  - sentiment shift
  - market reaction label
- Labels alignment as `Aligned`, `Narrative ahead of market`, `Market skeptical`, or `Risk confirmed`.
- Marks 10-K and 8-K filing events on price charts.
- Shows compact expandable 8-K cards with filing date, event type, impact label, short summary, and SEC filing details.
- Calculates 1-day, 5-day, and 30-day post-10-K returns and excess returns versus the S&P 500.
- Supports optional OpenAI change summaries and filing narrative labels.

## Data Sources

- Daily stock prices and S&P 500 benchmark data from Yahoo Finance through `yfinance`.
- SEC EDGAR metadata and filing documents for:
  - 10-K MD&A sections
  - 10-K Risk Factors sections
  - 8-K filing events and detail previews

## Tech Stack

- React, Vite, and Plotly.js
- FastAPI and Python
- PostgreSQL or Neon Postgres
- SQLAlchemy and `psycopg2`
- SEC parsing with Requests and BeautifulSoup
- Optional OpenAI API summaries
- Optional FinBERT research pipeline for filing sentiment/modeling experiments

## Project Layout

```text
backend/
  main.py                    # FastAPI routes and built-frontend serving
  dashboard_service.py       # JSON dashboard service layer
frontend/
  src/App.jsx                # React dashboard surface
  src/api.js                 # FastAPI client
  src/dashboard.css          # Dashboard design system and visual assets
  src/styles.css             # React-specific layout overrides
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
render.yaml                   # Staging React/FastAPI Render blueprint
```

The data pipeline keeps the existing `data_ingestiion` spelling to avoid breaking imports.

## Setup

Create a virtual environment and install backend dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
cp .env.example .env
```

Install the frontend dependencies:

```bash
npm install --prefix frontend
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

If the variable is omitted, the repo falls back to the default 100-ticker universe in `data_ingestiion/config.py`.

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

Include 10-K MD&A and Risk section extraction:

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
    SEC_USER_AGENT: ${{ secrets.SEC_USER_AGENT }}
    FINSIGHT_END_DATE: today
    FINSIGHT_DAILY_REFRESH_DAYS: 10
  run: python3 -m data_ingestiion.daily_refresh --with-8k
```

## Run The App

Start the FastAPI backend:

```bash
uvicorn backend.main:app --reload --port 8000
```

Start the React dev server in a second terminal:

```bash
npm run dev --prefix frontend
```

Open the Vite URL, normally:

```bash
http://127.0.0.1:5173
```

Vite proxies `/api` requests to FastAPI locally. Build the frontend for one-service deployment:

```bash
npm run build --prefix frontend
uvicorn backend.main:app --port 8000
```

When `frontend/dist` exists, FastAPI serves the React build at `http://127.0.0.1:8000`.

## API

The UI uses two FastAPI endpoints:

```text
GET /api/bootstrap
GET /api/dashboard?ticker=AAPL&start=2023-01-01&end=2026-05-21
```

FastAPI interactive API docs are available locally at `/docs`.

## Deployment

The migration branch includes `render.yaml` for a new staging Render service. It installs Python dependencies, installs and builds the React frontend, then starts:

```bash
uvicorn backend.main:app --host 0.0.0.0 --port $PORT
```

Keep the current Dash Render service linked to `main` until the React/FastAPI staging service is verified. Store `DATABASE_URL`, `SEC_USER_AGENT`, and optional OpenAI variables in Render environment settings.

## Tests

```bash
pytest
```

## Research Pipeline

The dashboard does not display raw regression outputs. The modeling scripts are kept as a research workflow for FinBERT-derived filing tone analysis and statistical experiments around narrative polarity and forward returns:

```bash
python3 data_ingestiion/modeling_pipeline_2.py
```
