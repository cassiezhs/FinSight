import os
import re
import sys
from functools import lru_cache
from urllib.parse import urlencode

import pandas as pd
from dash import Dash, dcc, html, Input, Output
import plotly.express as px
import plotly.graph_objects as go
from dotenv import load_dotenv
import dash_bootstrap_components as dbc
from openai import OpenAI
from sqlalchemy import inspect

try:
    import textstat
except ImportError:
    textstat = None

try:
    from textblob import TextBlob
except ImportError:
    TextBlob = None

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from data_ingestiion.config import settings
from data_ingestiion.db import get_engine

# Load DB credentials
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_ENABLED = os.getenv("OPENAI_ENABLED", "0").strip().lower() in {"1", "true", "yes", "on"}
client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_ENABLED and OPENAI_API_KEY else None

def sec_company_filing_search_url(cik, form_type, filing_date):
    filing_date = pd.to_datetime(filing_date).strftime("%Y%m%d")
    query = urlencode({
        "action": "getcompany",
        "CIK": str(cik).zfill(10),
        "type": form_type,
        "dateb": filing_date,
        "owner": "exclude",
        "count": "10",
    })
    return f"https://www.sec.gov/cgi-bin/browse-edgar?{query}"

def get_available_tickers(engine):
    query = "SELECT DISTINCT ticker FROM stock_prices ORDER BY ticker"
    return pd.read_sql(query, engine)['ticker'].tolist()

def get_stock_date_bounds(engine):
    query = 'SELECT MIN("Date") AS start_date, MAX("Date") AS end_date FROM stock_prices'
    df = pd.read_sql(query, engine)
    if df.empty or pd.isna(df.loc[0, 'start_date']) or pd.isna(df.loc[0, 'end_date']):
        return settings.start_date, settings.resolved_end_date
    return (
        pd.to_datetime(df.loc[0, 'start_date']).date().isoformat(),
        pd.to_datetime(df.loc[0, 'end_date']).date().isoformat(),
    )

def get_stock_data(engine, ticker, start_date, end_date):
    """Load time-series stock data."""
    query = """
        SELECT "Date", "Open", "Close", "Volume"
        FROM stock_prices
        WHERE ticker = %s 
        AND "Date" >= %s::date 
        AND "Date" <= %s::date
        ORDER BY "Date";
    """
    return pd.read_sql(query, engine, params=(ticker, start_date, end_date))

def get_filing_events(engine, ticker, start_date, end_date):
    query = """
        SELECT filing_date, MIN(cik) AS cik, STRING_AGG(section_type, ' + ' ORDER BY section_type) AS section_type
        FROM (
            SELECT DISTINCT filing_date, cik, 'MD&A' AS section_type
            FROM mdna_sections
            WHERE ticker = %s
            AND filing_date >= %s::date
            AND filing_date <= %s::date
            UNION
            SELECT DISTINCT filing_date, cik, 'Risk' AS section_type
            FROM risk_sections
            WHERE ticker = %s
            AND filing_date >= %s::date
            AND filing_date <= %s::date
        ) events
        GROUP BY filing_date
        ORDER BY filing_date;
    """
    ten_k = pd.read_sql(query, engine, params=(ticker, start_date, end_date, ticker, start_date, end_date))
    ten_k["event_type"] = "10-K"
    ten_k["details"] = ten_k["section_type"]
    if not ten_k.empty:
        ten_k["filing_date"] = pd.to_datetime(ten_k["filing_date"])
        ten_k["filing_url"] = ten_k.apply(
            lambda row: sec_company_filing_search_url(row["cik"], "10-K", row["filing_date"]),
            axis=1,
        )
    else:
        ten_k["filing_url"] = ""

    inspector = inspect(engine)
    if not inspector.has_table("sec_8k_filings", schema=settings.db_schema):
        return ten_k

    eight_k_query = """
        SELECT
            filing_date,
            '8-K' AS event_type,
            COALESCE(NULLIF(item_descriptions, ''), NULLIF(items, ''), primary_doc_description, form_type) AS details,
            filing_url
        FROM sec_8k_filings
        WHERE ticker = %s
        AND filing_date >= %s::date
        AND filing_date <= %s::date
        ORDER BY filing_date;
    """
    eight_k = pd.read_sql(eight_k_query, engine, params=(ticker, start_date, end_date))
    if eight_k.empty:
        return ten_k
    eight_k["filing_date"] = pd.to_datetime(eight_k["filing_date"])
    eight_k["section_type"] = eight_k["details"]
    return pd.concat([ten_k, eight_k], ignore_index=True).sort_values(["filing_date", "event_type"])

def load_8k_events(engine, ticker, start_date, end_date):
    inspector = inspect(engine)
    if not inspector.has_table("sec_8k_filings", schema=settings.db_schema):
        return pd.DataFrame()

    columns = {col["name"] for col in inspector.get_columns("sec_8k_filings", schema=settings.db_schema)}
    detail_preview_expr = "detail_preview" if "detail_preview" in columns else "''"
    detail_sources_expr = "detail_sources" if "detail_sources" in columns else "''"
    item_descriptions_expr = "item_descriptions" if "item_descriptions" in columns else "items"

    query = f"""
        SELECT
            filing_date,
            form_type,
            items,
            {item_descriptions_expr} AS item_descriptions,
            primary_doc_description,
            filing_url,
            {detail_preview_expr} AS detail_preview,
            {detail_sources_expr} AS detail_sources
        FROM sec_8k_filings
        WHERE ticker = %s
        AND filing_date >= %s::date
        AND filing_date <= %s::date
        ORDER BY filing_date DESC;
    """
    return pd.read_sql(query, engine, params=(ticker, start_date, end_date))

def load_10k_filings(engine, ticker, start_date, end_date):
    query = """
        SELECT filing_date, MIN(cik) AS cik, STRING_AGG(section_type, ' + ' ORDER BY section_type) AS sections
        FROM (
            SELECT DISTINCT filing_date, cik, 'MD&A' AS section_type
            FROM mdna_sections
            WHERE ticker = %s
            AND filing_date >= %s::date
            AND filing_date <= %s::date
            UNION
            SELECT DISTINCT filing_date, cik, 'Risk' AS section_type
            FROM risk_sections
            WHERE ticker = %s
            AND filing_date >= %s::date
            AND filing_date <= %s::date
        ) filings
        GROUP BY filing_date
        ORDER BY filing_date DESC;
    """
    df = pd.read_sql(query, engine, params=(ticker, start_date, end_date, ticker, start_date, end_date))
    if not df.empty:
        df["filing_date"] = pd.to_datetime(df["filing_date"])
        df["filing_url"] = df.apply(
            lambda row: sec_company_filing_search_url(row["cik"], "10-K", row["filing_date"]),
            axis=1,
        )
    return df

def load_price_window(engine, ticker, start_date, end_date):
    query = """
        SELECT "Date", "Close"
        FROM stock_prices
        WHERE ticker = %s
        AND "Date" >= %s::date
        AND "Date" <= (%s::date + INTERVAL '90 days')
        ORDER BY "Date";
    """
    df = pd.read_sql(query, engine, params=(ticker, start_date, end_date))
    if not df.empty:
        df["Date"] = pd.to_datetime(df["Date"])
    return df

def load_sp500_window(engine, start_date, end_date):
    query = """
        SELECT "Date", close
        FROM sp500_index
        WHERE "Date" >= %s::date
        AND "Date" <= (%s::date + INTERVAL '90 days')
        ORDER BY "Date";
    """
    df = pd.read_sql(query, engine, params=(start_date, end_date))
    if not df.empty:
        df["Date"] = pd.to_datetime(df["Date"])
    return df

def trading_return(df, anchor_date, horizon, price_col):
    if df.empty:
        return None
    df = df.sort_values("Date").reset_index(drop=True)
    anchor_rows = df[df["Date"] >= anchor_date]
    if anchor_rows.empty:
        return None

    base_index = anchor_rows.index[0]
    target_index = base_index + horizon
    if target_index >= len(df):
        return None

    base_price = float(df.loc[base_index, price_col])
    target_price = float(df.loc[target_index, price_col])
    if base_price == 0:
        return None

    return {
        "anchor_date": df.loc[base_index, "Date"],
        "target_date": df.loc[target_index, "Date"],
        "return": (target_price / base_price) - 1,
    }

def format_pct(value):
    if value is None or pd.isna(value):
        return "N/A"
    return f"{value * 100:+.2f}%"

def reaction_label(excess_return):
    if excess_return is None:
        return "Insufficient data"
    if excess_return > 0.01:
        return "Positive vs S&P 500"
    if excess_return < -0.01:
        return "Negative vs S&P 500"
    return "Neutral vs S&P 500"

def build_price_reactions(ticker, start_date, end_date):
    if not ticker:
        return html.Div("No ticker selected.", className="empty-state")
    if not start_date or not end_date:
        return html.Div("No date range selected.", className="empty-state")

    filings = load_10k_filings(engine, ticker, start_date, end_date)
    if filings.empty:
        return html.Div("No 10-K filings found for this ticker and date range.", className="empty-state")

    prices = load_price_window(engine, ticker, start_date, end_date)
    sp500 = load_sp500_window(engine, start_date, end_date)
    cards = []

    for row in filings.itertuples(index=False):
        filing_date = pd.to_datetime(row.filing_date)
        horizons = {}
        for horizon in [1, 5, 30]:
            stock_result = trading_return(prices, filing_date, horizon, "Close")
            if stock_result is None:
                horizons[horizon] = {"stock": None, "sp500": None, "excess": None, "target_date": None}
                continue

            sp500_result = trading_return(sp500, stock_result["anchor_date"], horizon, "close")
            sp500_return = sp500_result["return"] if sp500_result else None
            excess = stock_result["return"] - sp500_return if sp500_return is not None else None
            horizons[horizon] = {
                "stock": stock_result["return"],
                "sp500": sp500_return,
                "excess": excess,
                "target_date": stock_result["target_date"],
            }

        signal_excess = horizons[5]["excess"] if horizons[5]["excess"] is not None else horizons[1]["excess"]
        signal = reaction_label(signal_excess)
        signal_class = "positive" if "Positive" in signal else "negative" if "Negative" in signal else "neutral"

        cards.append(html.Div(className="reaction-item", children=[
            html.Div(className="reaction-head", children=[
                html.Div([
                    html.Span("10-K", className="event-type"),
                    html.Strong(pd.to_datetime(row.filing_date).date().isoformat()),
                    html.P(f"Sections: {row.sections}", className="reaction-subtitle"),
                ]),
                html.A("SEC filing", href=row.filing_url, target="_blank", className="event-link"),
            ]),
            html.Div(signal, className=f"reaction-signal {signal_class}"),
            html.Div(className="reaction-grid", children=[
                reaction_metric("1D", horizons[1]),
                reaction_metric("5D", horizons[5]),
                reaction_metric("30D", horizons[30]),
            ]),
        ]))

    return html.Div(cards, className="reaction-list")

def reaction_metric(label, result):
    return html.Div(className="reaction-metric", children=[
        html.Span(label),
        html.Strong(format_pct(result["stock"])),
        html.Em(f"vs S&P {format_pct(result['excess'])}"),
    ])

def format_number(value):
    if value is None or pd.isna(value):
        return "N/A"
    value = float(value)
    if abs(value) >= 1_000_000_000:
        return f"{value / 1_000_000_000:.2f}B"
    if abs(value) >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    if abs(value) >= 1_000:
        return f"{value / 1_000:.1f}K"
    return f"{value:,.0f}"

def format_price(value):
    if value is None or pd.isna(value):
        return "N/A"
    return f"${float(value):,.2f}"

def kpi_tone(value):
    if value is None or pd.isna(value):
        return "neutral"
    if value > 0:
        return "positive"
    if value < 0:
        return "negative"
    return "neutral"

def sentiment_badge(text):
    score = local_sentiment_score(text)
    label = sentiment_label(score)
    detail = f"Local score {score:+.3f}" if score is not None else "No local score"
    tone = "positive" if label == "Positive" else "negative" if label == "Negative" else "neutral"
    return label, detail, tone

def kpi_item(label, value, detail="", tone="neutral"):
    return html.Div(className=f"kpi-item {tone}", children=[
        html.Span(label),
        html.Strong(value),
        html.Em(detail),
    ])

def build_dashboard_kpis(ticker, start_date, end_date):
    if not ticker:
        return html.Div("No ticker selected.", className="empty-state")
    if not start_date or not end_date:
        return html.Div("No date range selected.", className="empty-state")

    prices = get_stock_data(engine, ticker, start_date, end_date)
    latest_close = None
    range_return = None
    volatility = None
    average_volume = None
    latest_price_date = "No prices"

    if not prices.empty:
        prices["Date"] = pd.to_datetime(prices["Date"])
        prices = prices.sort_values("Date")
        first_close = float(prices.iloc[0]["Close"])
        latest_close = float(prices.iloc[-1]["Close"])
        latest_price_date = prices.iloc[-1]["Date"].date().isoformat()
        if first_close:
            range_return = (latest_close / first_close) - 1
        returns = prices["Close"].pct_change().dropna()
        if not returns.empty:
            volatility = float(returns.std() * (252 ** 0.5))
        average_volume = float(prices["Volume"].mean())

    events = get_filing_events(engine, ticker, start_date, end_date)
    latest_filing = "N/A"
    latest_filing_detail = "No filing in range"
    if events is not None and not events.empty:
        events["filing_date"] = pd.to_datetime(events["filing_date"])
        latest_event = events.sort_values("filing_date").iloc[-1]
        latest_filing = latest_event["filing_date"].date().isoformat()
        latest_filing_detail = f"{latest_event.get('event_type', 'Filing')} filing"

    start_year, end_year = get_year_bounds(start_date, end_date)
    mdna_section = load_latest_section_for_years(engine, "mdna_sections", ticker, start_year, end_year)
    risk_section = load_latest_section_for_years(engine, "risk_sections", ticker, start_year, end_year)
    mdna_label, mdna_detail, mdna_tone = sentiment_badge(mdna_section["full_content"] if mdna_section is not None else "")
    risk_label, risk_detail, risk_tone = sentiment_badge(risk_section["full_content"] if risk_section is not None else "")

    return html.Div(className="kpi-grid", children=[
        kpi_item("Return", format_pct(range_return), f"{start_date} to {latest_price_date}", kpi_tone(range_return)),
        kpi_item("Volatility", format_pct(volatility), "Annualized daily close"),
        kpi_item("Avg Volume", format_number(average_volume), "Shares per trading day"),
        kpi_item("Latest Close", format_price(latest_close), latest_price_date),
        kpi_item("Latest Filing", latest_filing, latest_filing_detail),
        kpi_item("MD&A Sentiment", mdna_label, mdna_detail, mdna_tone),
        kpi_item("Risk Sentiment", risk_label, risk_detail, risk_tone),
    ])

def get_year_bounds(start_date, end_date):
    return pd.to_datetime(start_date).year, pd.to_datetime(end_date).year

def load_latest_section_for_years(engine, table_name, ticker, start_year, end_year):
    if table_name not in {"mdna_sections", "risk_sections"}:
        raise ValueError(f"Unsupported section table: {table_name}")

    query = f"""
        SELECT filing_date, MIN(cik) AS cik, STRING_AGG(content, ' ' ORDER BY chunk_index) AS full_content
        FROM {table_name}
        WHERE ticker = %s
        AND EXTRACT(YEAR FROM filing_date) >= %s
        AND EXTRACT(YEAR FROM filing_date) <= %s
        GROUP BY filing_date
        ORDER BY filing_date DESC
        LIMIT 1;
    """
    df = pd.read_sql(query, engine, params=(ticker, start_year, end_year))
    return None if df.empty else df.iloc[0]

def load_previous_section(engine, table_name, ticker, current_filing_date):
    if table_name not in {"mdna_sections", "risk_sections"}:
        raise ValueError(f"Unsupported section table: {table_name}")

    current_filing_date = pd.to_datetime(current_filing_date).date().isoformat()
    query = f"""
        SELECT filing_date, MIN(cik) AS cik, STRING_AGG(content, ' ' ORDER BY chunk_index) AS full_content
        FROM {table_name}
        WHERE ticker = %s
        AND filing_date < %s::date
        GROUP BY filing_date
        ORDER BY filing_date DESC
        LIMIT 1;
    """
    df = pd.read_sql(query, engine, params=(ticker, current_filing_date))
    return None if df.empty else df.iloc[0]

def word_count(text):
    return len(re.findall(r"\b[\w'-]+\b", text or ""))

def readability_score(text):
    if not textstat or not text:
        return None
    try:
        return round(float(textstat.flesch_reading_ease(text)), 1)
    except Exception:
        return None

def local_sentiment_score(text):
    if not TextBlob or not text:
        return None
    try:
        return round(float(TextBlob(text[:8000]).sentiment.polarity), 3)
    except Exception:
        return None

def sentiment_label(score):
    if score is None:
        return "N/A"
    if score >= 0.05:
        return "Positive"
    if score <= -0.05:
        return "Negative"
    return "Neutral"

def sentence_split(text):
    sentences = re.split(r"(?<=[.!?])\s+", re.sub(r"\s+", " ", text or "").strip())
    return [s.strip() for s in sentences if 70 <= len(s.strip()) <= 320]

def normalize_sentence(sentence):
    return re.sub(r"[^a-z0-9]+", " ", sentence.lower()).strip()

def sentence_changes(current_text, previous_text, limit=4):
    current_sentences = sentence_split(current_text)
    previous_sentences = sentence_split(previous_text)
    previous_norm = {normalize_sentence(sentence) for sentence in previous_sentences}
    current_norm = {normalize_sentence(sentence) for sentence in current_sentences}

    added = [sentence for sentence in current_sentences if normalize_sentence(sentence) not in previous_norm]
    removed = [sentence for sentence in previous_sentences if normalize_sentence(sentence) not in current_norm]
    return added[:limit], removed[:limit]

def section_metrics(current_text, previous_text):
    current_words = word_count(current_text)
    previous_words = word_count(previous_text)
    current_readability = readability_score(current_text)
    previous_readability = readability_score(previous_text)
    current_sentiment = local_sentiment_score(current_text)
    previous_sentiment = local_sentiment_score(previous_text)

    return {
        "word_count": current_words,
        "word_delta": current_words - previous_words,
        "readability": current_readability,
        "readability_delta": (
            round(current_readability - previous_readability, 1)
            if current_readability is not None and previous_readability is not None
            else None
        ),
        "sentiment": current_sentiment,
        "sentiment_delta": (
            round(current_sentiment - previous_sentiment, 3)
            if current_sentiment is not None and previous_sentiment is not None
            else None
        ),
    }

def format_delta(value, suffix=""):
    if value is None:
        return "N/A"
    sign = "+" if value > 0 else ""
    return f"{sign}{value}{suffix}"

def load_mdna(ticker, engine, start_year, end_year):
    query = """
        SELECT filing_date, cik, chunk_index, content
        FROM mdna_sections
        WHERE ticker = %s
        AND EXTRACT(YEAR FROM filing_date) >= %s
        AND EXTRACT(YEAR FROM filing_date) <= %s
        ORDER BY filing_date DESC, chunk_index ASC
    """
    df = pd.read_sql(query, engine, params=(ticker, start_year, end_year))
    
    if df.empty:
        return pd.DataFrame()

    df['full_content'] = df.groupby('filing_date')['content'].transform(lambda x: ' '.join(x))
    df = df.drop_duplicates(subset=['filing_date'])
    return df[['filing_date', 'cik', 'full_content']]

def format_section_display(filing_date, content, filing_url=None):
    filing_date = pd.to_datetime(filing_date).date().isoformat()
    return html.Div(children=[
        html.Div(className="filing-actions", children=[
            html.P(f"Filing date: {filing_date}", className="filing-date"),
            html.A("SEC filing", href=filing_url, target="_blank", className="event-link") if filing_url else None,
        ]),
        html.Div(content)
    ])

@lru_cache(maxsize=256)
def detect_sentiment(text):
    prompt = f"Analyze this financial filing section and respond with one word only: Positive, Neutral, or Negative.\n\n{text[:3000]}"
    if not OPENAI_ENABLED:
        return "Disabled"
    if client is None:
        return "Unavailable"
    try:
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": "You are a financial analyst."},
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error: {str(e)}"

@lru_cache(maxsize=256)
def summarize_mdna(text):
    if not OPENAI_ENABLED:
        return "OpenAI summaries are disabled."
    if client is None:
        return "OpenAI API key is not configured."
    try:
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": "You are a financial analyst. Summarize the following MD&A in 3-4 bullet points."},
                {"role": "user", "content": text[:3000]}
            ],
            temperature=0.4
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error generating summary: {str(e)}"

def load_risk_sections(ticker, engine, start_year, end_year):
    query = """
        SELECT filing_date, cik, chunk_index, content
        FROM risk_sections
        WHERE ticker = %s
        AND EXTRACT(YEAR FROM filing_date) >= %s
        AND EXTRACT(YEAR FROM filing_date) <= %s
        ORDER BY filing_date DESC, chunk_index ASC
    """
    df = pd.read_sql(query, engine, params=(ticker, start_year, end_year))
    
    if df.empty:
        return pd.DataFrame()

    df['full_content'] = df.groupby('filing_date')['content'].transform(lambda x: ' '.join(x))
    df = df.drop_duplicates(subset=['filing_date'])
    return df[['filing_date', 'cik', 'full_content']]

@lru_cache(maxsize=256)
def summarize_risk(text):
    if not OPENAI_ENABLED:
        return "OpenAI summaries are disabled."
    if client is None:
        return "OpenAI API key is not configured."
    try:
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": "You are a financial analyst. Summarize the following risk factors in 3-4 bullet points."},
                {"role": "user", "content": text[:3000]}
            ],
            temperature=0.4
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error generating risk summary: {str(e)}"

@lru_cache(maxsize=128)
def summarize_filing_changes(section_name, current_date, previous_date, current_text, previous_text):
    if not OPENAI_ENABLED:
        return "AI change summary is disabled. Set OPENAI_ENABLED=1 to generate this comparison."
    if client is None:
        return "OpenAI API key is not configured."

    prompt = f"""
Compare the current {section_name} section with the previous year's {section_name}.

Previous filing date: {previous_date}
Current filing date: {current_date}

Focus on what changed, not a generic summary. Return:
- 3-5 bullet points on substantive changes
- any newly emphasized risks, strategy shifts, performance drivers, or uncertainty
- whether the tone became more positive, negative, or cautious

Previous {section_name} excerpt:
{previous_text[:6000]}

Current {section_name} excerpt:
{current_text[:6000]}
"""
    try:
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": "You are a financial analyst comparing SEC filing language year over year."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error generating change summary: {str(e)}"

def style_market_figure(fig, show_legend=True):
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Urbanist, Inter, Arial, sans-serif", color="#111111", size=13),
        margin=dict(l=24, r=18, t=10, b=28),
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(255,255,255,0)",
            font=dict(size=12),
        ),
        showlegend=show_legend,
    )
    fig.update_xaxes(
        showgrid=False,
        zeroline=False,
        linecolor="rgba(17,17,17,0.12)",
        tickfont=dict(color="#77727f"),
    )
    fig.update_yaxes(
        gridcolor="rgba(17,17,17,0.07)",
        zeroline=False,
        linecolor="rgba(17,17,17,0.08)",
        tickfont=dict(color="#77727f"),
    )
    fig.update_traces(line=dict(width=3), selector=dict(type="scatter"))
    fig.update_traces(marker_line_width=0, opacity=0.72, selector=dict(type="bar"))
    return fig

def style_empty_figure(title):
    fig = go.Figure()
    fig.add_annotation(
        text=title,
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
        showarrow=False,
        font=dict(size=18, color="#77727f"),
    )
    return style_market_figure(fig, show_legend=False)

def add_filing_markers(fig, price_df, events_df):
    if events_df is None or events_df.empty:
        return fig

    price_df = price_df.sort_values('Date').reset_index(drop=True)
    marker_data = {
        "10-K": {"x": [], "y": [], "text": [], "customdata": []},
        "8-K": {"x": [], "y": [], "text": [], "customdata": []},
    }

    for event in events_df.itertuples(index=False):
        filing_date = pd.to_datetime(event.filing_date)
        event_type = event.event_type
        details = event.details
        filing_url = getattr(event, "filing_url", "")

        next_trading_rows = price_df[price_df['Date'] >= filing_date]
        if next_trading_rows.empty:
            marker_row = price_df.iloc[-1]
        else:
            marker_row = next_trading_rows.iloc[0]

        if event_type == "10-K":
            fig.add_vline(
                x=filing_date,
                line_width=1.6,
                line_dash="dot",
                line_color="rgba(5, 5, 5, 0.42)",
            )

        target = marker_data["10-K" if event_type == "10-K" else "8-K"]
        target["x"].append(marker_row['Date'])
        target["y"].append(marker_row['Close'])
        target["text"].append(event_type)
        target["customdata"].append([filing_date.date().isoformat(), details, filing_url])

    if marker_data["10-K"]["x"]:
        fig.add_trace(go.Scatter(
            x=marker_data["10-K"]["x"],
            y=marker_data["10-K"]["y"],
            mode="markers+text",
            name="10-K filing",
            text=marker_data["10-K"]["text"],
            textposition="top center",
            customdata=marker_data["10-K"]["customdata"],
            marker=dict(
                symbol="diamond",
                size=12,
                color="#000000",
                line=dict(color="#8FFE01", width=3),
            ),
            textfont=dict(color="#000000", size=12, family="Urbanist, Inter, Arial, sans-serif"),
            hovertemplate=(
                "<b>10-K filing</b><br>"
                "Filing date: %{customdata[0]}<br>"
                "Sections: %{customdata[1]}<br>"
                "Link available in filing panels<br>"
                "Chart date: %{x|%Y-%m-%d}<br>"
                "Close: %{y:.2f}<extra></extra>"
            ),
        ))

    if marker_data["8-K"]["x"]:
        fig.add_trace(go.Scatter(
            x=marker_data["8-K"]["x"],
            y=marker_data["8-K"]["y"],
            mode="markers",
            name="8-K filing",
            customdata=marker_data["8-K"]["customdata"],
            marker=dict(
                symbol="circle",
                size=9,
                color="#7201FF",
                line=dict(color="#FFFFFF", width=2),
            ),
            hovertemplate=(
                "<b>8-K filing</b><br>"
                "Filing date: %{customdata[0]}<br>"
                "Items: %{customdata[1]}<br>"
                "Link available in 8-K Events panel<br>"
                "Chart date: %{x|%Y-%m-%d}<br>"
                "Close: %{y:.2f}<extra></extra>"
            ),
        ))
    return fig

def format_8k_events(events_df):
    if events_df is None or events_df.empty:
        return html.Div("No 8-K events found for this ticker and date range.", className="empty-state")

    cards = []
    for row in events_df.itertuples(index=False):
        filing_date = pd.to_datetime(row.filing_date).date().isoformat()
        item_descriptions = getattr(row, "item_descriptions", "") or getattr(row, "items", "") or "No item description available."
        detail_preview = getattr(row, "detail_preview", "") or "No extracted detail preview is stored yet. Re-run ingestion with FINSIGHT_LOAD_8K=1 and FINSIGHT_LOAD_8K_DETAILS=1 to populate this event."
        detail_sources = getattr(row, "detail_sources", "") or getattr(row, "primary_doc_description", "") or "SEC filing"
        filing_url = getattr(row, "filing_url", "")

        cards.append(html.Div(className="event-item", children=[
            html.Div(className="event-item-head", children=[
                html.Div([
                    html.Span("8-K", className="event-type"),
                    html.Strong(filing_date),
                ]),
                html.A("SEC filing", href=filing_url, target="_blank", className="event-link") if filing_url else None,
            ]),
            html.Div(item_descriptions, className="event-items"),
            html.P(detail_preview, className="event-preview"),
            html.Div(f"Sources: {detail_sources}", className="event-source"),
        ]))

    return html.Div(cards, className="event-list")

def metric_tile(label, value, delta=None):
    return html.Div(className="comparison-metric", children=[
        html.Span(label),
        html.Strong(value),
        html.Em(delta if delta is not None else "")
    ])

def sentence_list(title, sentences):
    if not sentences:
        return html.Div(className="language-list", children=[
            html.H4(title),
            html.P("No major sentence-level changes detected.", className="muted-copy")
        ])

    return html.Div(className="language-list", children=[
        html.H4(title),
        html.Ul([html.Li(sentence) for sentence in sentences])
    ])

def ai_change_summary_block(section_name, current_date, previous_date, current_text, previous_text):
    summary = summarize_filing_changes(
        section_name,
        current_date,
        previous_date,
        current_text,
        previous_text,
    )
    return html.Div(className="ai-change-summary", children=[
        html.Div(className="ai-change-head", children=[
            html.Span("AI what changed"),
            html.Strong(section_name),
        ]),
        html.Div(summary, className="ai-change-body"),
    ])

def build_section_comparison(name, current_row, previous_row, include_language=True):
    if current_row is None:
        return html.Div(f"No {name} filing found for the selected filing year.", className="empty-state")
    if previous_row is None:
        current_date = pd.to_datetime(current_row["filing_date"]).date().isoformat()
        return html.Div(f"{name} filing found for {current_date}, but no prior filing is available for comparison.", className="empty-state")

    current_text = current_row["full_content"] or ""
    previous_text = previous_row["full_content"] or ""
    metrics = section_metrics(current_text, previous_text)
    current_date = pd.to_datetime(current_row["filing_date"]).date().isoformat()
    previous_date = pd.to_datetime(previous_row["filing_date"]).date().isoformat()
    current_url = sec_company_filing_search_url(current_row["cik"], "10-K", current_row["filing_date"])
    previous_url = sec_company_filing_search_url(previous_row["cik"], "10-K", previous_row["filing_date"])
    added, removed = sentence_changes(current_text, previous_text)

    metric_row = html.Div(className="comparison-metrics", children=[
        metric_tile("Words", f"{metrics['word_count']:,}", format_delta(metrics["word_delta"])),
        metric_tile("Readability", metrics["readability"] if metrics["readability"] is not None else "N/A", format_delta(metrics["readability_delta"])),
        metric_tile(
            "Local sentiment",
            sentiment_label(metrics["sentiment"]),
            format_delta(metrics["sentiment_delta"]),
        ),
    ])

    language = []
    if include_language:
        language = [
            ai_change_summary_block(name, current_date, previous_date, current_text, previous_text),
            sentence_list(f"New {name} language", added),
            sentence_list(f"Removed {name} language", removed),
        ]

    return html.Div(className="comparison-section", children=[
        html.Div(className="comparison-section-head", children=[
            html.Div([
                html.Span(name, className="event-type"),
                html.H3(f"{current_date} vs {previous_date}"),
                html.Div(className="comparison-links", children=[
                    html.A("Current filing", href=current_url, target="_blank", className="event-link"),
                    html.A("Previous filing", href=previous_url, target="_blank", className="event-link secondary"),
                ])
            ])
        ]),
        metric_row,
        *language,
    ])

def build_filing_comparison(ticker, start_date, end_date):
    if not ticker:
        return html.Div("No ticker selected.", className="empty-state")
    if not start_date or not end_date:
        return html.Div("No date range selected.", className="empty-state")

    start_year, end_year = get_year_bounds(start_date, end_date)
    current_mdna = load_latest_section_for_years(engine, "mdna_sections", ticker, start_year, end_year)
    current_risk = load_latest_section_for_years(engine, "risk_sections", ticker, start_year, end_year)
    previous_mdna = load_previous_section(engine, "mdna_sections", ticker, current_mdna["filing_date"]) if current_mdna is not None else None
    previous_risk = load_previous_section(engine, "risk_sections", ticker, current_risk["filing_date"]) if current_risk is not None else None

    return html.Div(className="comparison-grid", children=[
        build_section_comparison("MD&A", current_mdna, previous_mdna, include_language=True),
        build_section_comparison("Risk", current_risk, previous_risk, include_language=True),
    ])

# Initialize
engine = get_engine()
tickers = get_available_tickers(engine)
default_ticker = tickers[0] if tickers else None
default_start_date, default_end_date = get_stock_date_bounds(engine)

assets_path = os.path.join(os.path.dirname(__file__), "assets")
app = Dash(__name__, external_stylesheets=[dbc.themes.ZEPHYR], assets_folder=assets_path)
app.title = "FinSight Dashboard"

app.layout = html.Div(className="page-shell", children=[
    html.Div(className="topbar", children=[
        html.Div(className="brand", children=[
            html.Div("F", className="brand-mark"),
            html.Span("FinSight")
        ]),
        html.Div(className="nav-links", children=[
            html.Span("Overview", className="active"),
            html.Span("Filings"),
            html.Span("Signals"),
            html.Span("Reports")
        ]),
        html.Div(className="topbar-action", children="Market intelligence")
    ]),

    html.Div(className="hero-card", children=[
        html.Div(className="hero-meta", children=[
            html.Span("Market pulse", className="eyebrow"),
            html.H1("Stock Price Viewer"),
            html.P("Price action, filing language, risk signals, and AI summaries aligned to the selected period.")
        ]),
        html.Div(className="hero-highlight", children=[
            dcc.Loading(
                type="default",
                parent_className="loading-shell kpi-loading hero-kpis",
                children=html.Div(id="kpi-row", className="kpi-shell")
            )
        ])
    ]),

    html.Div(className="controls-card card", children=[
        html.Div(className="field", children=[
            html.Label("Select Ticker"),
            dcc.Dropdown(tickers, default_ticker, id='ticker-dropdown', className="control")
        ]),
        html.Div(className="field", children=[
            html.Label("Select Date Range"),
            dcc.DatePickerRange(
                id='date-range',
                min_date_allowed=default_start_date,
                max_date_allowed=default_end_date,
                start_date=default_start_date,
                end_date=default_end_date,
                with_portal=True,
                className="control"
            )
        ])
    ]),

    html.Div(className="grid charts", children=[
        html.Div(className="card chart-card", children=[
            html.Div(className="card-head", children=[
                html.Div([
                    html.Span("Trend"),
                    html.H3("Open vs Close Prices"),
                    html.P("Black diamonds mark 10-K filings. Purple dots mark 8-K filings.", className="card-note")
                ])
            ]),
            dcc.Graph(id='price-graph')
        ]),
        html.Div(className="card chart-card", children=[
            html.Div(className="card-head", children=[
                html.Div([
                    html.Span("Liquidity"),
                    html.H3("Trading Volume")
                ])
            ]),
            dcc.Graph(id='volume-graph')
        ])
    ]),

    html.Div(className="card events-card", children=[
        html.Div(className="card-head", children=[
            html.Div([
                html.Span("Event detail"),
                html.H3("8-K Events")
            ])
        ]),
        dcc.Loading(
            type="circle",
            color="#7201FF",
            className="loading-shell",
            children=html.Div(id='eight-k-events')
        )
    ]),

    html.Div(className="card reaction-card", children=[
        html.Div(className="card-head", children=[
            html.Div([
                html.Span("Market reaction"),
                html.H3("Price Reaction After 10-K"),
                html.P("Returns use the next trading day after filing as the anchor and compare against S&P 500 over the same window.", className="card-note")
            ])
        ]),
        dcc.Loading(
            type="circle",
            color="#7201FF",
            className="loading-shell",
            children=html.Div(id='price-reaction')
        )
    ]),

    html.Div(className="card comparison-card", children=[
        html.Div(className="card-head", children=[
            html.Div([
                html.Span("Year over year"),
                html.H3("Filing Comparison"),
                html.P("Compares the selected filing against the previous available filing using local text analytics.", className="card-note")
            ])
        ]),
        dcc.Loading(
            type="circle",
            color="#7201FF",
            className="loading-shell",
            children=html.Div(id='filing-comparison')
        )
    ]),

    html.Div(className="grid info", children=[
        html.Div(className="stack", children=[
            html.Div(className="card info-card", children=[
                html.Div(className="card-head", children=[
                    html.Span("Filing stream"),
                    html.H3("MD&A - Management Discussion")
                ]),
                dcc.Loading(
                    type="circle",
                    color="#7201FF",
                    className="loading-shell",
                    children=html.Div(id='mdna-text-box')
                )
            ]),
            html.Div(className="card info-card", children=[
                html.Div(className="card-head", children=[
                    html.Span("Risk factors"),
                    html.H3("Risk Sections")
                ]),
                dcc.Loading(
                    type="circle",
                    color="#7201FF",
                    className="loading-shell",
                    children=html.Div(id='risk-text-box')
                )
            ])
        ]),
        html.Div(className="stack", children=[
            dcc.Loading(
                type="dot",
                color="#8FFE01",
                className="loading-shell compact",
                children=html.Div(id='sentiment-tag', className="sentiment-tag")
            ),
            html.Div(className="card summary-card", children=[
                html.Div(className="card-head", children=[
                    html.Span("AI digest"),
                    html.H3("Summary of MD&A")
                ]),
                dcc.Loading(
                    type="circle",
                    color="#7201FF",
                    className="loading-shell",
                    children=html.Div(id='summary-box')
                )
            ]),
            html.Div(className="card summary-card", children=[
                html.Div(className="card-head", children=[
                    html.Span("AI digest"),
                    html.H3("Summary of Risk Sections")
                ]),
                dcc.Loading(
                    type="circle",
                    color="#7201FF",
                    className="loading-shell",
                    children=html.Div(id='risk-summary-box')
                )
            ])
        ])
    ])
])

@app.callback(
    Output('kpi-row', 'children'),
    Input('ticker-dropdown', 'value'),
    Input('date-range', 'start_date'),
    Input('date-range', 'end_date')
)
def update_dashboard_kpis(ticker, start_date, end_date):
    return build_dashboard_kpis(ticker, start_date, end_date)


@app.callback(
    Output('price-graph', 'figure'),
    Output('volume-graph', 'figure'),
    Input('ticker-dropdown', 'value'),
    Input('date-range', 'start_date'),
    Input('date-range', 'end_date')
)
def update_graphs(ticker, start_date, end_date):
    if not ticker:
        empty = style_empty_figure("No ticker data available")
        return empty, empty

    df = get_stock_data(engine, ticker, start_date, end_date)
    if df.empty:
        empty = style_empty_figure(f"No stock data found for {ticker}")
        return empty, empty

    df['Date'] = pd.to_datetime(df['Date'])
    events_df = get_filing_events(engine, ticker, start_date, end_date)
    if events_df is not None and not events_df.empty:
        events_df['filing_date'] = pd.to_datetime(events_df['filing_date'])

    fig_price = px.line(
        df, x='Date', y=['Open', 'Close'],
        color_discrete_map={
            'Open': '#8FFE01',
            'Close': '#7201FF'
        }
    )
    style_market_figure(fig_price)
    add_filing_markers(fig_price, df, events_df)

    fig_volume = px.bar(
        df, x='Date', y='Volume',
        color_discrete_sequence=['#7201FF']
    )
    style_market_figure(fig_volume, show_legend=False)


    return fig_price, fig_volume


@app.callback(
    Output('eight-k-events', 'children'),
    Input('ticker-dropdown', 'value'),
    Input('date-range', 'start_date'),
    Input('date-range', 'end_date')
)
def update_8k_events(ticker, start_date, end_date):
    if not ticker:
        return html.Div("No ticker selected.", className="empty-state")
    if not start_date or not end_date:
        return html.Div("No date range selected.", className="empty-state")

    events_df = load_8k_events(engine, ticker, start_date, end_date)
    return format_8k_events(events_df)


@app.callback(
    Output('price-reaction', 'children'),
    Input('ticker-dropdown', 'value'),
    Input('date-range', 'start_date'),
    Input('date-range', 'end_date')
)
def update_price_reaction(ticker, start_date, end_date):
    return build_price_reactions(ticker, start_date, end_date)


@app.callback(
    Output('filing-comparison', 'children'),
    Input('ticker-dropdown', 'value'),
    Input('date-range', 'start_date'),
    Input('date-range', 'end_date')
)
def update_filing_comparison(ticker, start_date, end_date):
    return build_filing_comparison(ticker, start_date, end_date)


@app.callback(
    Output('mdna-text-box', 'children'),
    Output('summary-box', 'children'),
    Output('sentiment-tag', 'children'),
    Output('risk-text-box', 'children'),
    Output('risk-summary-box', 'children'),
    Input('ticker-dropdown', 'value'),
    Input('date-range', 'start_date'),
    Input('date-range', 'end_date')
)
def update_mdna(ticker, start_date, end_date):
    if not ticker:
        return "No ticker selected.", "", "Sentiment: N/A", "No ticker selected.", ""
    if not start_date or not end_date:
        return "No date range selected.", "", "Sentiment: N/A", "No date range selected.", ""

    start_year, end_year = get_year_bounds(start_date, end_date)
    year_label = str(start_year) if start_year == end_year else f"{start_year}-{end_year}"

    mdna_df = load_mdna(ticker, engine, start_year, end_year)
    risk_df = load_risk_sections(ticker, engine, start_year, end_year)

    mdna_text = f"No MD&A filing found for {ticker} in filing year {year_label}."
    mdna_summary = ""
    mdna_sentiment = "N/A"
    mdna_display = mdna_text

    if mdna_df is not None and not mdna_df.empty:
        mdna_row = mdna_df.iloc[0]
        mdna_text = mdna_row['full_content'][:5000]
        mdna_url = sec_company_filing_search_url(mdna_row['cik'], "10-K", mdna_row['filing_date'])
        mdna_display = format_section_display(mdna_row['filing_date'], mdna_text, mdna_url)
        mdna_summary = summarize_mdna(mdna_text)
        mdna_sentiment = detect_sentiment(mdna_text)

    risk_text = f"No risk filing found for {ticker} in filing year {year_label}."
    risk_summary = ""
    risk_sentiment = "N/A"
    risk_display = risk_text

    if risk_df is not None and not risk_df.empty:
        risk_row = risk_df.iloc[0]
        risk_text = risk_row['full_content'][:5000]
        risk_url = sec_company_filing_search_url(risk_row['cik'], "10-K", risk_row['filing_date'])
        risk_display = format_section_display(risk_row['filing_date'], risk_text, risk_url)
        risk_summary = summarize_risk(risk_text)
        risk_sentiment = detect_sentiment(risk_text)

    sentiment_badge = f"Sentiment - MD&A: {mdna_sentiment} | Risk: {risk_sentiment}"
    return mdna_display, mdna_summary, sentiment_badge, risk_display, risk_summary


if __name__ == "__main__":
    app.run(debug=True)
