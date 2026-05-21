import os
import re
import sys
from functools import lru_cache
from urllib.parse import urlencode

import pandas as pd
from dash import Dash, dcc, html, Input, Output, ctx
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
OPENAI_ENABLED = os.getenv("OPENAI_ENABLED", "1").strip().lower() in {"1", "true", "yes", "on"}
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
    label = narrative_sentiment_label(text)
    detail = f"Local score {score:+.3f}" if score is not None else "No local score"
    tone = narrative_tone(label)
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

def first_valid(*values):
    for value in values:
        if value is not None and not pd.isna(value):
            return value
    return None

def latest_10k_excess_return(ticker, start_date, end_date):
    filings = load_10k_filings(engine, ticker, start_date, end_date)
    if filings.empty:
        return None

    prices = load_price_window(engine, ticker, start_date, end_date)
    sp500 = load_sp500_window(engine, start_date, end_date)
    if prices.empty or sp500.empty:
        return None

    filing_date = pd.to_datetime(filings.iloc[0]["filing_date"])
    horizons = []
    for horizon in [5, 1, 30]:
        stock_result = trading_return(prices, filing_date, horizon, "Close")
        if stock_result is None:
            horizons.append(None)
            continue
        sp500_result = trading_return(sp500, stock_result["anchor_date"], horizon, "close")
        if sp500_result is None:
            horizons.append(None)
            continue
        horizons.append(stock_result["return"] - sp500_result["return"])
    return first_valid(*horizons)

def stance_class(stance):
    return {
        "Constructive": "positive",
        "Mixed": "neutral",
        "Cautious": "cautious",
        "Risk-Off": "negative",
    }.get(stance, "neutral")

def score_tone(label):
    return {
        "Bullish": 1.25,
        "Expansion-Oriented": 1.0,
        "Aggressive": 0.5,
        "Defensive": -0.75,
        "Cautious": -1.0,
        "Risk-Elevated": -1.75,
    }.get(label, 0)

def tone_direction(label):
    if label in {"Bullish", "Expansion-Oriented", "Aggressive"}:
        return 1
    if label in {"Defensive", "Cautious", "Risk-Elevated"}:
        return -1
    return 0

def tone_change_label(current_label, previous_label):
    current_score = score_tone(current_label)
    previous_score = score_tone(previous_label)
    if current_label == "N/A" or previous_label == "N/A":
        return "N/A"
    delta = current_score - previous_score
    if delta >= 0.75:
        return "More constructive"
    if delta <= -0.75:
        return "More cautious"
    return "Stable"

def alignment_class(label):
    return {
        "Aligned": "positive",
        "Narrative ahead of market": "ahead",
        "Market skeptical": "skeptical",
        "Risk confirmed": "negative",
    }.get(label, "neutral")

def build_alignment_signal(mdna_section, risk_section, ticker, start_date, end_date, ten_k_excess):
    if mdna_section is None or risk_section is None:
        return None

    previous_mdna = load_previous_section(engine, "mdna_sections", ticker, mdna_section["filing_date"])
    previous_risk = load_previous_section(engine, "risk_sections", ticker, risk_section["filing_date"])
    if previous_mdna is None or previous_risk is None:
        return None

    mdna_metrics = section_metrics(mdna_section["full_content"], previous_mdna["full_content"])
    risk_added, _ = sentence_changes(risk_section["full_content"], previous_risk["full_content"], limit=20)
    current_mdna_tone = narrative_sentiment_label(mdna_section["full_content"])
    previous_mdna_tone = narrative_sentiment_label(previous_mdna["full_content"])
    current_risk_tone = narrative_sentiment_label(risk_section["full_content"])
    reaction = reaction_label(ten_k_excess)
    sentiment_shift = mdna_metrics["sentiment_delta"]
    tone_shift = tone_change_label(current_mdna_tone, previous_mdna_tone)
    risk_pressure = len(risk_added) >= 3 or current_risk_tone == "Risk-Elevated"
    narrative_direction = tone_direction(current_mdna_tone)
    market_direction = 1 if ten_k_excess is not None and ten_k_excess >= 0.01 else -1 if ten_k_excess is not None and ten_k_excess <= -0.01 else 0

    if risk_pressure and market_direction < 0:
        output = "Risk confirmed"
        why = "New risk language and a negative post-10-K excess return point in the same direction."
    elif narrative_direction > 0 and market_direction < 0:
        output = "Market skeptical"
        why = "The filing narrative reads constructive, but the market reaction lagged the S&P 500."
    elif narrative_direction < 0 and market_direction > 0:
        output = "Narrative ahead of market"
        why = "The filing language is cautious while the market reaction has not confirmed that caution."
    elif market_direction and narrative_direction == market_direction:
        output = "Aligned"
        why = "Filing tone and post-10-K market reaction move in the same direction."
    elif risk_pressure:
        output = "Narrative ahead of market"
        why = "Risk language changed before a decisive market reaction appeared."
    else:
        output = "Aligned"
        why = "No strong narrative-market divergence is visible in the selected filing window."

    return {
        "output": output,
        "why": why,
        "class": alignment_class(output),
        "tone_change": tone_shift,
        "new_risk_count": len(risk_added),
        "excess_return": ten_k_excess,
        "sentiment_shift": sentiment_shift,
        "reaction": reaction,
    }

def build_alignment_panel(alignment, range_return, risk_tone, high_impact_8k, medium_impact_8k):
    if alignment is None:
        return html.Div(className="readout-facts", children=[
            html.Div([html.Span("Return"), html.Strong(format_pct(range_return))]),
            html.Div([html.Span("Risk tone"), html.Strong(risk_tone)]),
            html.Div([html.Span("8-K impact"), html.Strong(f"{high_impact_8k} high / {medium_impact_8k} medium")]),
            html.Div([html.Span("Alignment"), html.Strong("Needs prior 10-K")]),
        ])

    return html.Div(className="alignment-panel", children=[
        html.Div(className=f"alignment-head {alignment['class']}", children=[
            html.Span("Narrative vs Market Alignment"),
            html.Strong(alignment["output"]),
            html.P(alignment["why"]),
        ]),
        html.Div(className="alignment-factors", children=[
            html.Div([html.Span("Filing tone change"), html.Strong(alignment["tone_change"])]),
            html.Div([html.Span("New risk language"), html.Strong(str(alignment["new_risk_count"]))]),
            html.Div([html.Span("10-K excess return"), html.Strong(format_pct(alignment["excess_return"]))]),
            html.Div([html.Span("Sentiment shift"), html.Strong(format_delta(alignment["sentiment_shift"]))]),
            html.Div([html.Span("Market reaction"), html.Strong(alignment["reaction"])]),
        ]),
    ])

def build_market_readout(ticker, start_date, end_date):
    if not ticker:
        return html.Div("No ticker selected.", className="empty-state")
    if not start_date or not end_date:
        return html.Div("No date range selected.", className="empty-state")

    prices = get_stock_data(engine, ticker, start_date, end_date)
    range_return = None
    volatility = None
    if not prices.empty:
        prices = prices.sort_values("Date")
        first_close = float(prices.iloc[0]["Close"])
        last_close = float(prices.iloc[-1]["Close"])
        if first_close:
            range_return = (last_close / first_close) - 1
        returns = prices["Close"].pct_change().dropna()
        if not returns.empty:
            volatility = float(returns.std() * (252 ** 0.5))

    start_year, end_year = get_year_bounds(start_date, end_date)
    mdna_section = load_latest_section_for_years(engine, "mdna_sections", ticker, start_year, end_year)
    risk_section = load_latest_section_for_years(engine, "risk_sections", ticker, start_year, end_year)
    mdna_tone = narrative_sentiment_label(mdna_section["full_content"] if mdna_section is not None else "")
    risk_tone = narrative_sentiment_label(risk_section["full_content"] if risk_section is not None else "")
    ten_k_excess = latest_10k_excess_return(ticker, start_date, end_date)
    alignment = build_alignment_signal(mdna_section, risk_section, ticker, start_date, end_date, ten_k_excess)

    eight_k_events = load_8k_events(engine, ticker, start_date, end_date)
    high_impact_8k = 0
    medium_impact_8k = 0
    if eight_k_events is not None and not eight_k_events.empty:
        for row in eight_k_events.itertuples(index=False):
            item_descriptions = getattr(row, "item_descriptions", "") or getattr(row, "items", "")
            detail_preview = getattr(row, "detail_preview", "")
            _, impact_tone = eight_k_impact_label(item_descriptions, detail_preview)
            high_impact_8k += 1 if impact_tone == "high" else 0
            medium_impact_8k += 1 if impact_tone == "medium" else 0

    score = 0
    if range_return is not None:
        score += 1.25 if range_return >= 0.08 else -1.25 if range_return <= -0.08 else 0.4 if range_return > 0 else -0.4
    if ten_k_excess is not None:
        score += 1.2 if ten_k_excess >= 0.02 else -1.2 if ten_k_excess <= -0.02 else 0
    if volatility is not None:
        score -= 0.8 if volatility >= 0.45 else 0.4 if volatility >= 0.32 else 0
    score += score_tone(mdna_tone)
    score += score_tone(risk_tone)
    score -= min(high_impact_8k * 0.55, 1.1)

    if score >= 1.75:
        stance = "Constructive"
    elif score >= -0.75:
        stance = "Mixed"
    elif score >= -2.25:
        stance = "Cautious"
    else:
        stance = "Risk-Off"

    driver = "Filing tone is the main signal."
    if risk_tone == "Risk-Elevated":
        driver = "Risk language is elevated in the selected filing period."
    elif ten_k_excess is not None and ten_k_excess <= -0.02:
        driver = "The latest 10-K reaction underperformed the S&P 500."
    elif ten_k_excess is not None and ten_k_excess >= 0.02:
        driver = "The latest 10-K reaction outperformed the S&P 500."
    elif range_return is not None and abs(range_return) >= 0.08:
        direction = "strong positive" if range_return > 0 else "weak"
        driver = f"Stock performance was {direction} over the selected range."
    elif high_impact_8k:
        driver = f"{high_impact_8k} high-impact 8-K event{'s' if high_impact_8k != 1 else ''} appeared in the range."
    elif mdna_tone in {"Bullish", "Expansion-Oriented"}:
        driver = f"MD&A tone reads {mdna_tone.lower()}."

    why = {
        "Constructive": "Price action and filing language lean supportive, with no dominant risk signal overwhelming the readout.",
        "Mixed": "Signals are not pointing in one clean direction, so the selected period needs confirmation from the next filing or market reaction.",
        "Cautious": "Filing language or market reaction suggests investors should treat the period with more caution than the headline price alone implies.",
        "Risk-Off": "Risk tone, market reaction, and/or event activity point to a materially defensive readout.",
    }[stance]

    if risk_tone in {"Risk-Elevated", "Cautious"}:
        watch_next = "Watch whether the next 10-Q or earnings 8-K repeats the same risk language."
    elif ten_k_excess is not None and abs(ten_k_excess) >= 0.02:
        watch_next = "Watch whether the stock keeps confirming the filing reaction over the next trading window."
    elif high_impact_8k or medium_impact_8k:
        watch_next = "Open the latest 8-K details and check whether the event changes guidance, liquidity, or management tone."
    else:
        watch_next = "Watch the next filing for a clearer change in MD&A drivers or risk factors."

    return html.Div(className="readout-grid", children=[
        html.Div(className=f"readout-stance {stance_class(stance)}", children=[
            html.Span("Overall stance"),
            html.Strong(stance),
            html.Em(f"Score {score:+.1f}")
        ]),
        html.Div(className="readout-copy", children=[
            html.Div(className="readout-line", children=[
                html.Span("Why it matters"),
                html.P(why),
            ]),
            html.Div(className="readout-line", children=[
                html.Span("Key driver"),
                html.P(driver),
            ]),
            html.Div(className="readout-line", children=[
                html.Span("Watch next"),
                html.P(watch_next),
            ]),
        ]),
        build_alignment_panel(alignment, range_return, risk_tone, high_impact_8k, medium_impact_8k),
    ])

def get_year_bounds(start_date, end_date):
    return pd.to_datetime(start_date).year, pd.to_datetime(end_date).year

def preset_date_range(preset_id):
    coverage_start = pd.Timestamp(default_start_date)
    coverage_end = pd.Timestamp(default_end_date)

    if preset_id == "range-all":
        start_date = coverage_start
    elif preset_id == "range-ytd":
        start_date = coverage_end.replace(month=1, day=1)
    else:
        years = {
            "range-1y": 1,
            "range-3y": 3,
            "range-5y": 5,
        }.get(preset_id)
        if years is None:
            return default_start_date, default_end_date
        start_date = coverage_end - pd.DateOffset(years=years)

    return max(start_date, coverage_start).date().isoformat(), coverage_end.date().isoformat()

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

def keyword_hits(text, terms):
    text = (text or "").lower()
    return sum(1 for term in terms if term in text)

def narrative_sentiment_label(text):
    score = local_sentiment_score(text)
    if score is None:
        return "N/A"

    risk_terms = [
        "decline", "decrease", "adverse", "uncertain", "uncertainty", "volatility",
        "litigation", "impairment", "material weakness", "cybersecurity", "regulatory",
        "inflation", "interest rates", "supply chain", "competition", "risk",
    ]
    defensive_terms = [
        "cost reduction", "restructuring", "efficiency", "liquidity", "preserve cash",
        "mitigate", "offset", "headcount", "expenses", "controls",
    ]
    expansion_terms = [
        "growth", "increase", "expanded", "expansion", "new products", "demand",
        "revenue grew", "market share", "capacity", "international", "acquisition",
    ]
    aggressive_terms = [
        "investment", "investments", "capital expenditures", "acquire", "acquisition",
        "launch", "accelerate", "strategic initiative", "repurchase", "buyback",
    ]
    cautious_terms = [
        "may", "could", "expect", "anticipate", "uncertain", "pressure", "headwinds",
        "challenging", "slowdown", "macroeconomic",
    ]

    risk_count = keyword_hits(text, risk_terms)
    defensive_count = keyword_hits(text, defensive_terms)
    expansion_count = keyword_hits(text, expansion_terms)
    aggressive_count = keyword_hits(text, aggressive_terms)
    cautious_count = keyword_hits(text, cautious_terms)

    if risk_count >= 4 or score <= -0.08:
        return "Risk-Elevated"
    if defensive_count >= 2:
        return "Defensive"
    if expansion_count >= 3 and score >= 0.04:
        return "Expansion-Oriented"
    if aggressive_count >= 3:
        return "Aggressive"
    if cautious_count >= 4 or score < -0.02:
        return "Cautious"
    if score >= 0.08:
        return "Bullish"
    return "Cautious"

def narrative_tone(label):
    if label in {"Bullish", "Expansion-Oriented", "Aggressive"}:
        return "positive"
    if label in {"Risk-Elevated", "Cautious"}:
        return "negative"
    return "neutral"

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
    prompt = f"""
Classify this financial filing section using exactly one narrative label from this list:
Bullish, Defensive, Cautious, Aggressive, Risk-Elevated, Expansion-Oriented.

Use the label that best captures the management narrative, not generic text polarity.
Respond with the label only.

Filing excerpt:
{text[:4000]}
"""
    if not OPENAI_ENABLED:
        return narrative_sentiment_label(text)
    if client is None:
        return narrative_sentiment_label(text)
    try:
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a financial analyst classifying filing narrative tone. "
                        "Allowed labels only: Bullish, Defensive, Cautious, Aggressive, Risk-Elevated, Expansion-Oriented."
                    )
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )
        label = response.choices[0].message.content.strip()
        allowed = {"Bullish", "Defensive", "Cautious", "Aggressive", "Risk-Elevated", "Expansion-Oriented"}
        return label if label in allowed else narrative_sentiment_label(text)
    except Exception as e:
        return f"Error: {str(e)}"

@lru_cache(maxsize=256)
def summarize_mdna(text, previous_text=""):
    if not OPENAI_ENABLED:
        return "OpenAI summaries are disabled."
    if client is None:
        return "OpenAI API key is not configured."
    if previous_text:
        user_prompt = f"""
Current MD&A excerpt:
{text[:6000]}

Previous MD&A excerpt:
{previous_text[:6000]}
"""
    else:
        user_prompt = f"Current MD&A excerpt:\n{text[:6000]}"
    try:
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an equity analyst writing a concise filing-change brief. "
                        "Do not restate generic forward-looking statement language or boilerplate. "
                        "Focus only on what is NEW, DIFFERENT, and IMPORTANT for investors. "
                        "Return exactly 3 bullets. Each bullet must start with one of: New:, Different:, Important:. "
                        "If prior-year text is provided, compare against it. If not, identify the most decision-relevant points in the current text. "
                        "Be specific, plain-spoken, and avoid generic GPT-style summaries."
                    )
                },
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.2
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
def summarize_risk(text, previous_text=""):
    if not OPENAI_ENABLED:
        return "OpenAI summaries are disabled."
    if client is None:
        return "OpenAI API key is not configured."
    if previous_text:
        user_prompt = f"""
Current Risk Factors excerpt:
{text[:6000]}

Previous Risk Factors excerpt:
{previous_text[:6000]}
"""
    else:
        user_prompt = f"Current Risk Factors excerpt:\n{text[:6000]}"
    try:
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an equity analyst writing a concise risk-change brief. "
                        "Do not summarize standard legal boilerplate. "
                        "Focus only on what is NEW, DIFFERENT, and IMPORTANT for investors. "
                        "Return exactly 3 bullets. Each bullet must start with one of: New:, Different:, Important:. "
                        "If prior-year text is provided, compare against it and call out newly emphasized, removed, or intensified risks. "
                        "Be specific, plain-spoken, and avoid generic GPT-style summaries."
                    )
                },
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.2
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
- exactly 2 bullets
- each bullet must be one short sentence under 18 words
- only the most decision-relevant change and the most important risk/driver shift
- no filing boilerplate, no section recap, no preamble, no conclusion

Previous {section_name} excerpt:
{previous_text[:6000]}

Current {section_name} excerpt:
{current_text[:6000]}
"""
    try:
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a financial analyst comparing SEC filing language year over year. "
                        "Write terse scan-friendly bullets for a dashboard."
                    ),
                },
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

def compact_text(text, limit=150):
    text = re.sub(r"\s+", " ", text or "").strip()
    if not text:
        return ""
    if len(text) <= limit:
        return text
    return text[:limit].rsplit(" ", 1)[0].rstrip(".,;:") + "..."

def eight_k_one_line_summary(item_descriptions, detail_preview, primary_doc_description):
    preview = re.sub(r"\s+", " ", detail_preview or "").strip()
    if preview and not preview.startswith("No extracted detail preview"):
        sentence = re.split(r"(?<=[.!?])\s+", preview)[0]
        return compact_text(sentence, 150)

    item_text = item_descriptions or primary_doc_description or "8-K filing event"
    return compact_text(f"Company filed an 8-K covering {item_text}.", 150)

def eight_k_impact_label(item_descriptions, detail_preview):
    text = f"{item_descriptions or ''} {detail_preview or ''}".lower()
    high_terms = [
        "bankruptcy",
        "delisting",
        "departure of directors",
        "appointment of certain officers",
        "material definitive agreement",
        "termination of a material definitive agreement",
        "cybersecurity incident",
        "change in control",
        "non-reliance",
    ]
    medium_terms = [
        "results of operations",
        "financial condition",
        "regulation fd",
        "other events",
        "financial statements",
        "exhibits",
        "acquisition",
        "disposition",
    ]

    if any(term in text for term in high_terms):
        return "High impact", "high"
    if any(term in text for term in medium_terms):
        return "Medium impact", "medium"
    return "Low impact", "low"

def format_8k_events(events_df):
    if events_df is None or events_df.empty:
        return html.Div("No 8-K events found for this ticker and date range.", className="empty-state")

    cards = []
    for row in events_df.itertuples(index=False):
        filing_date = pd.to_datetime(row.filing_date).date().isoformat()
        item_descriptions = getattr(row, "item_descriptions", "") or getattr(row, "items", "") or "No item description available."
        detail_preview = getattr(row, "detail_preview", "") or "No extracted detail preview is stored yet. Re-run ingestion with FINSIGHT_LOAD_8K=1 and FINSIGHT_LOAD_8K_DETAILS=1 to populate this event."
        detail_sources = getattr(row, "detail_sources", "") or getattr(row, "primary_doc_description", "") or "SEC filing"
        primary_doc_description = getattr(row, "primary_doc_description", "") or ""
        filing_url = getattr(row, "filing_url", "")
        one_line_summary = eight_k_one_line_summary(item_descriptions, detail_preview, primary_doc_description)
        impact_label, impact_tone = eight_k_impact_label(item_descriptions, detail_preview)

        cards.append(html.Div(className="event-item", children=[
            html.Div(className="event-item-head", children=[
                html.Div([
                    html.Span("8-K", className="event-type"),
                    html.Strong(filing_date),
                ]),
                html.Span(impact_label, className=f"impact-label {impact_tone}"),
            ]),
            html.P(one_line_summary, className="event-summary"),
            html.Details(className="event-details", children=[
                html.Summary("Details"),
                html.Div(item_descriptions, className="event-items"),
                html.P(detail_preview, className="event-preview"),
                html.Div(f"Sources: {detail_sources}", className="event-source"),
                html.A("SEC filing", href=filing_url, target="_blank", className="event-link") if filing_url else None,
            ]),
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
            "Narrative tone",
            narrative_sentiment_label(current_text),
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
server = app.server

app.layout = html.Div(className="page-shell", children=[
    html.Div(className="topbar", children=[
        html.Div(className="brand", children=[
            html.Div("F", className="brand-mark"),
            html.Span("FinSight")
        ]),
        html.Div(className="topbar-action", children="Market intelligence")
    ]),

    html.Div(className="hero-card", children=[
        html.Div(className="hero-meta", children=[
            html.Span("Market pulse", className="eyebrow"),
            html.H1("Narrative vs. Market"),
            html.P("Analyzing how 10-K filing language aligns with stock price movement, market reaction, and risk signals.")
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
                initial_visible_month=default_end_date,
                number_of_months_shown=2,
                display_format="YYYY-MM-DD",
                month_format="MMM YYYY",
                with_portal=True,
                className="control"
            )
        ]),
        html.Div(className="field range-preset-field", children=[
            html.Label("Quick Ranges"),
            html.Div(className="range-presets", children=[
                html.Button("YTD", id="range-ytd", n_clicks=0, title="Year to date"),
                html.Button("1Y", id="range-1y", n_clicks=0, title="Latest year"),
                html.Button("3Y", id="range-3y", n_clicks=0, title="Latest three years"),
                html.Button("5Y", id="range-5y", n_clicks=0, title="Latest five years"),
                html.Button("All", id="range-all", n_clicks=0, title="All available data"),
            ])
        ])
    ]),

    html.Div(className="card readout-card", children=[
        html.Div(className="card-head", children=[
            html.Div([
                html.Span("Final insight"),
                html.H3("Market Readout"),
                html.P("Combines price action, filing tone, 10-K reaction, and 8-K event impact into one investor-facing view.", className="card-note")
            ])
        ]),
        dcc.Loading(
            type="circle",
            color="#7201FF",
            className="loading-shell",
            children=html.Div(id="market-readout")
        )
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

    dcc.Loading(
        type="dot",
        color="#8FFE01",
        className="loading-shell compact sentiment-row",
        children=html.Div(id='sentiment-tag', className="sentiment-tag")
    ),

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
            html.Div(className="card summary-card", children=[
                html.Div(className="card-head", children=[
                    html.Span("AI digest"),
                    html.H3("MD&A: What Changed")
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
                    html.H3("Risk: What Changed")
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
    Output('date-range', 'start_date'),
    Output('date-range', 'end_date'),
    Input('range-ytd', 'n_clicks'),
    Input('range-1y', 'n_clicks'),
    Input('range-3y', 'n_clicks'),
    Input('range-5y', 'n_clicks'),
    Input('range-all', 'n_clicks'),
    prevent_initial_call=True,
)
def apply_date_preset(ytd_clicks, one_year_clicks, three_year_clicks, five_year_clicks, all_clicks):
    return preset_date_range(ctx.triggered_id)


@app.callback(
    Output('kpi-row', 'children'),
    Input('ticker-dropdown', 'value'),
    Input('date-range', 'start_date'),
    Input('date-range', 'end_date')
)
def update_dashboard_kpis(ticker, start_date, end_date):
    return build_dashboard_kpis(ticker, start_date, end_date)


@app.callback(
    Output('market-readout', 'children'),
    Input('ticker-dropdown', 'value'),
    Input('date-range', 'start_date'),
    Input('date-range', 'end_date')
)
def update_market_readout(ticker, start_date, end_date):
    return build_market_readout(ticker, start_date, end_date)


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
        previous_mdna = load_previous_section(engine, "mdna_sections", ticker, mdna_row["filing_date"])
        previous_mdna_text = previous_mdna["full_content"] if previous_mdna is not None else ""
        mdna_url = sec_company_filing_search_url(mdna_row['cik'], "10-K", mdna_row['filing_date'])
        mdna_display = format_section_display(mdna_row['filing_date'], mdna_text, mdna_url)
        mdna_summary = summarize_mdna(mdna_row["full_content"], previous_mdna_text)
        mdna_sentiment = detect_sentiment(mdna_text)

    risk_text = f"No risk filing found for {ticker} in filing year {year_label}."
    risk_summary = ""
    risk_sentiment = "N/A"
    risk_display = risk_text

    if risk_df is not None and not risk_df.empty:
        risk_row = risk_df.iloc[0]
        risk_text = risk_row['full_content'][:5000]
        previous_risk = load_previous_section(engine, "risk_sections", ticker, risk_row["filing_date"])
        previous_risk_text = previous_risk["full_content"] if previous_risk is not None else ""
        risk_url = sec_company_filing_search_url(risk_row['cik'], "10-K", risk_row['filing_date'])
        risk_display = format_section_display(risk_row['filing_date'], risk_text, risk_url)
        risk_summary = summarize_risk(risk_row["full_content"], previous_risk_text)
        risk_sentiment = detect_sentiment(risk_text)

    sentiment_badge = f"Sentiment - MD&A: {mdna_sentiment} | Risk: {risk_sentiment}"
    return mdna_display, mdna_summary, sentiment_badge, risk_display, risk_summary


if __name__ == "__main__":
    app.run(debug=True)
