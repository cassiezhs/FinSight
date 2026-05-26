"""JSON service layer for the React/FastAPI FinSight dashboard."""

from __future__ import annotations

import os
import re
from functools import lru_cache
from typing import Any
from urllib.parse import urlencode

import pandas as pd
from openai import OpenAI
from sqlalchemy import inspect

from data_ingestiion.config import settings
from data_ingestiion.db import get_engine

OPENAI_ENABLED = os.getenv("OPENAI_ENABLED", "1").strip().lower() in {"1", "true", "yes", "on"}
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_CLIENT = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_ENABLED and OPENAI_API_KEY else None


@lru_cache(maxsize=1)
def engine():
    return get_engine()


def sec_filing_search_url(cik: Any, form_type: str, filing_date: Any) -> str:
    query = urlencode(
        {
            "action": "getcompany",
            "CIK": str(cik).zfill(10),
            "type": form_type,
            "dateb": pd.to_datetime(filing_date).strftime("%Y%m%d"),
            "owner": "exclude",
            "count": "10",
        }
    )
    return f"https://www.sec.gov/cgi-bin/browse-edgar?{query}"


def date_text(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    return pd.to_datetime(value).date().isoformat()


def number(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(value)


def get_bootstrap() -> dict[str, Any]:
    tickers = pd.read_sql("SELECT DISTINCT ticker FROM stock_prices ORDER BY ticker", engine())["ticker"].tolist()
    bounds = pd.read_sql('SELECT MIN("Date") AS start_date, MAX("Date") AS end_date FROM stock_prices', engine())
    if bounds.empty or pd.isna(bounds.loc[0, "start_date"]) or pd.isna(bounds.loc[0, "end_date"]):
        start_date, end_date = settings.start_date, settings.resolved_end_date
    else:
        start_date, end_date = date_text(bounds.loc[0, "start_date"]), date_text(bounds.loc[0, "end_date"])
    return {
        "tickers": tickers,
        "default_ticker": tickers[0] if tickers else None,
        "start_date": start_date,
        "end_date": end_date,
        "openai_enabled": bool(OPENAI_ENABLED and OPENAI_CLIENT),
    }


def stock_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    query = """
        SELECT "Date", "Open", "Close", "Volume"
        FROM stock_prices
        WHERE ticker = %s AND "Date" >= %s::date AND "Date" <= %s::date
        ORDER BY "Date";
    """
    df = pd.read_sql(query, engine(), params=(ticker, start_date, end_date))
    if not df.empty:
        df["Date"] = pd.to_datetime(df["Date"])
    return df


def filing_events(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    query = """
        SELECT filing_date, MIN(cik) AS cik, STRING_AGG(section_type, ' + ' ORDER BY section_type) AS details
        FROM (
            SELECT DISTINCT filing_date, cik, 'MD&A' AS section_type
            FROM mdna_sections
            WHERE ticker = %s AND filing_date >= %s::date AND filing_date <= %s::date
            UNION
            SELECT DISTINCT filing_date, cik, 'Risk' AS section_type
            FROM risk_sections
            WHERE ticker = %s AND filing_date >= %s::date AND filing_date <= %s::date
        ) events
        GROUP BY filing_date
        ORDER BY filing_date;
    """
    ten_k = pd.read_sql(query, engine(), params=(ticker, start_date, end_date, ticker, start_date, end_date))
    if not ten_k.empty:
        ten_k["filing_date"] = pd.to_datetime(ten_k["filing_date"])
        ten_k["event_type"] = "10-K"
        ten_k["filing_url"] = ten_k.apply(
            lambda row: sec_filing_search_url(row["cik"], "10-K", row["filing_date"]), axis=1
        )
    if not inspect(engine()).has_table("sec_8k_filings", schema=settings.db_schema):
        return ten_k
    query_8k = """
        SELECT filing_date, '8-K' AS event_type,
            COALESCE(NULLIF(item_descriptions, ''), NULLIF(items, ''), primary_doc_description, form_type) AS details,
            filing_url
        FROM sec_8k_filings
        WHERE ticker = %s AND filing_date >= %s::date AND filing_date <= %s::date
        ORDER BY filing_date;
    """
    eight_k = pd.read_sql(query_8k, engine(), params=(ticker, start_date, end_date))
    if eight_k.empty:
        return ten_k
    eight_k["filing_date"] = pd.to_datetime(eight_k["filing_date"])
    return pd.concat([ten_k, eight_k], ignore_index=True).sort_values(["filing_date", "event_type"])


def load_8k_events(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    inspector = inspect(engine())
    if not inspector.has_table("sec_8k_filings", schema=settings.db_schema):
        return pd.DataFrame()
    columns = {column["name"] for column in inspector.get_columns("sec_8k_filings", schema=settings.db_schema)}
    detail_preview = "detail_preview" if "detail_preview" in columns else "''"
    detail_sources = "detail_sources" if "detail_sources" in columns else "''"
    item_descriptions = "item_descriptions" if "item_descriptions" in columns else "items"
    query = f"""
        SELECT filing_date, form_type, items, {item_descriptions} AS item_descriptions,
            primary_doc_description, filing_url, {detail_preview} AS detail_preview,
            {detail_sources} AS detail_sources
        FROM sec_8k_filings
        WHERE ticker = %s AND filing_date >= %s::date AND filing_date <= %s::date
        ORDER BY filing_date DESC;
    """
    return pd.read_sql(query, engine(), params=(ticker, start_date, end_date))


def section_for_years(table: str, ticker: str, start_year: int, end_year: int) -> pd.Series | None:
    if table not in {"mdna_sections", "risk_sections"}:
        raise ValueError(f"Unsupported filing section table: {table}")
    query = f"""
        SELECT filing_date, MIN(cik) AS cik, STRING_AGG(content, ' ' ORDER BY chunk_index) AS content
        FROM {table}
        WHERE ticker = %s AND EXTRACT(YEAR FROM filing_date) >= %s AND EXTRACT(YEAR FROM filing_date) <= %s
        GROUP BY filing_date
        ORDER BY filing_date DESC
        LIMIT 1;
    """
    df = pd.read_sql(query, engine(), params=(ticker, start_year, end_year))
    return None if df.empty else df.iloc[0]


def previous_section(table: str, ticker: str, current_filing_date: Any) -> pd.Series | None:
    if table not in {"mdna_sections", "risk_sections"}:
        raise ValueError(f"Unsupported filing section table: {table}")
    query = f"""
        SELECT filing_date, MIN(cik) AS cik, STRING_AGG(content, ' ' ORDER BY chunk_index) AS content
        FROM {table}
        WHERE ticker = %s AND filing_date < %s::date
        GROUP BY filing_date
        ORDER BY filing_date DESC
        LIMIT 1;
    """
    df = pd.read_sql(query, engine(), params=(ticker, date_text(current_filing_date)))
    return None if df.empty else df.iloc[0]


def format_pct(value: float | None) -> str:
    return "N/A" if value is None else f"{value * 100:+.2f}%"


def format_compact_number(value: float | None) -> str:
    if value is None:
        return "N/A"
    if abs(value) >= 1_000_000_000:
        return f"{value / 1_000_000_000:.2f}B"
    if abs(value) >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    if abs(value) >= 1_000:
        return f"{value / 1_000:.1f}K"
    return f"{value:,.0f}"


def format_price(value: float | None) -> str:
    return "N/A" if value is None else f"${value:,.2f}"


def keyword_hits(text: str, terms: list[str]) -> int:
    text = (text or "").lower()
    return sum(term in text for term in terms)


@lru_cache(maxsize=1)
def textblob_cls():
    try:
        from textblob import TextBlob

        return TextBlob
    except Exception:
        return None


@lru_cache(maxsize=1)
def textstat_module():
    try:
        import textstat

        return textstat
    except Exception:
        return None


def local_sentiment_score(text: str) -> float | None:
    TextBlob = textblob_cls()
    if not TextBlob or not text:
        return None
    try:
        return round(float(TextBlob(text[:8000]).sentiment.polarity), 3)
    except Exception:
        return None


def narrative_label(text: str) -> str:
    score = local_sentiment_score(text)
    if score is None:
        return "N/A"
    risk = keyword_hits(text, ["decline", "adverse", "uncertain", "volatility", "litigation", "impairment", "cybersecurity", "regulatory", "inflation", "interest rates", "competition", "risk"])
    defensive = keyword_hits(text, ["cost reduction", "restructuring", "efficiency", "liquidity", "preserve cash", "mitigate", "expenses", "controls"])
    expansion = keyword_hits(text, ["growth", "increase", "expanded", "expansion", "new products", "demand", "revenue grew", "market share", "capacity", "acquisition"])
    aggressive = keyword_hits(text, ["investment", "investments", "capital expenditures", "acquire", "acquisition", "launch", "accelerate", "repurchase", "buyback"])
    cautious = keyword_hits(text, ["may", "could", "expect", "anticipate", "uncertain", "pressure", "headwinds", "challenging", "slowdown", "macroeconomic"])
    if risk >= 4 or score <= -0.08:
        return "Risk-Elevated"
    if defensive >= 2:
        return "Defensive"
    if expansion >= 3 and score >= 0.04:
        return "Expansion-Oriented"
    if aggressive >= 3:
        return "Aggressive"
    if cautious >= 4 or score < -0.02:
        return "Cautious"
    return "Bullish" if score >= 0.08 else "Cautious"


def tone(label: str) -> str:
    if label in {"Bullish", "Expansion-Oriented", "Aggressive"}:
        return "positive"
    if label in {"Risk-Elevated", "Cautious"}:
        return "negative"
    return "neutral"


def ai_disabled_message(kind: str) -> str:
    return f"OpenAI {kind} are disabled." if not OPENAI_ENABLED else "OpenAI API key is not configured."


@lru_cache(maxsize=256)
def ai_narrative_label(text: str) -> str:
    if OPENAI_CLIENT is None:
        return narrative_label(text)
    try:
        response = OPENAI_CLIENT.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": "Classify filing narrative tone with one allowed label only: Bullish, Defensive, Cautious, Aggressive, Risk-Elevated, Expansion-Oriented."},
                {"role": "user", "content": text[:4000]},
            ],
            temperature=0,
        )
        label = response.choices[0].message.content.strip()
        return label if label in {"Bullish", "Defensive", "Cautious", "Aggressive", "Risk-Elevated", "Expansion-Oriented"} else narrative_label(text)
    except Exception:
        return narrative_label(text)


@lru_cache(maxsize=256)
def summarize_section(section_name: str, text: str, previous_text: str) -> str:
    if OPENAI_CLIENT is None:
        return ai_disabled_message("summaries")
    prompt = f"Current {section_name} excerpt:\n{text[:6000]}"
    if previous_text:
        prompt += f"\n\nPrevious {section_name} excerpt:\n{previous_text[:6000]}"
    try:
        response = OPENAI_CLIENT.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": "Write exactly 2 short bullets for investors. Each bullet must be 18 words or fewer. Summarize the current filing section, not year-over-year changes. Avoid boilerplate."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
        )
        return response.choices[0].message.content.strip()
    except Exception as exc:
        return f"Error generating summary: {exc}"


@lru_cache(maxsize=128)
def summarize_changes(section_name: str, current_date: str, previous_date: str, current: str, previous: str) -> str:
    if OPENAI_CLIENT is None:
        return ai_disabled_message("change summaries")
    try:
        response = OPENAI_CLIENT.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": "Compare SEC filing language year over year. Return exactly two short scan-friendly bullets."},
                {"role": "user", "content": f"{section_name}: {current_date} vs {previous_date}\nPrevious:\n{previous[:6000]}\nCurrent:\n{current[:6000]}\nFocus on the most decision-relevant change and risk/driver shift."},
            ],
            temperature=0.2,
        )
        return response.choices[0].message.content.strip()
    except Exception as exc:
        return f"Error generating change summary: {exc}"


def sentence_split(text: str) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+", re.sub(r"\s+", " ", text or "").strip())
    return [sentence.strip() for sentence in sentences if 70 <= len(sentence.strip()) <= 320]


def sentence_changes(current: str, previous: str, limit: int = 4) -> tuple[list[str], list[str]]:
    normalize = lambda sentence: re.sub(r"[^a-z0-9]+", " ", sentence.lower()).strip()
    current_sentences, previous_sentences = sentence_split(current), sentence_split(previous)
    current_norm, previous_norm = {normalize(item) for item in current_sentences}, {normalize(item) for item in previous_sentences}
    return (
        [item for item in current_sentences if normalize(item) not in previous_norm][:limit],
        [item for item in previous_sentences if normalize(item) not in current_norm][:limit],
    )


def section_metrics(current: str, previous: str) -> dict[str, Any]:
    words = lambda text: len(re.findall(r"\b[\w'-]+\b", text or ""))
    def readability(text: str) -> float | None:
        textstat = textstat_module()
        if not textstat or not text:
            return None
        try:
            return round(float(textstat.flesch_reading_ease(text)), 1)
        except Exception:
            return None
    current_readability, previous_readability = readability(current), readability(previous)
    current_sentiment, previous_sentiment = local_sentiment_score(current), local_sentiment_score(previous)
    return {
        "word_count": words(current),
        "word_delta": words(current) - words(previous),
        "readability": current_readability,
        "readability_delta": round(current_readability - previous_readability, 1) if current_readability is not None and previous_readability is not None else None,
        "sentiment": current_sentiment,
        "sentiment_delta": round(current_sentiment - previous_sentiment, 3) if current_sentiment is not None and previous_sentiment is not None else None,
        "tone": narrative_label(current),
    }


def compact_text(text: str, limit: int = 150) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    return text if len(text) <= limit else text[:limit].rsplit(" ", 1)[0].rstrip(".,;:") + "..."


def eight_k_impact(items: str, preview: str) -> tuple[str, str]:
    text = f"{items or ''} {preview or ''}".lower()
    high = ["bankruptcy", "delisting", "departure of directors", "appointment of certain officers", "material definitive agreement", "cybersecurity incident", "change in control", "non-reliance"]
    medium = ["results of operations", "financial condition", "regulation fd", "other events", "financial statements", "exhibits", "acquisition", "disposition"]
    if any(term in text for term in high):
        return "High impact", "high"
    if any(term in text for term in medium):
        return "Medium impact", "medium"
    return "Low impact", "low"


def serialize_8k(ticker: str, start_date: str, end_date: str) -> list[dict[str, Any]]:
    events = load_8k_events(ticker, start_date, end_date)
    output = []
    for row in events.itertuples(index=False):
        items = getattr(row, "item_descriptions", "") or getattr(row, "items", "") or "No item description available."
        preview = getattr(row, "detail_preview", "") or "No extracted detail preview is stored yet."
        summary_source = preview if preview and not preview.startswith("No extracted") else f"Company filed an 8-K covering {items}."
        summary = compact_text(re.split(r"(?<=[.!?])\s+", summary_source)[0])
        impact, impact_tone = eight_k_impact(items, preview)
        output.append(
            {
                "date": date_text(row.filing_date),
                "form_type": getattr(row, "form_type", "8-K"),
                "items": items,
                "preview": preview,
                "sources": getattr(row, "detail_sources", "") or getattr(row, "primary_doc_description", "") or "SEC filing",
                "summary": summary,
                "impact": impact,
                "impact_tone": impact_tone,
                "url": getattr(row, "filing_url", "") or None,
            }
        )
    return output


def load_10k_filings(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    query = """
        SELECT filing_date, MIN(cik) AS cik, STRING_AGG(section_type, ' + ' ORDER BY section_type) AS sections
        FROM (
            SELECT DISTINCT filing_date, cik, 'MD&A' AS section_type
            FROM mdna_sections
            WHERE ticker = %s AND filing_date >= %s::date AND filing_date <= %s::date
            UNION
            SELECT DISTINCT filing_date, cik, 'Risk' AS section_type
            FROM risk_sections
            WHERE ticker = %s AND filing_date >= %s::date AND filing_date <= %s::date
        ) filings
        GROUP BY filing_date
        ORDER BY filing_date DESC;
    """
    df = pd.read_sql(query, engine(), params=(ticker, start_date, end_date, ticker, start_date, end_date))
    if not df.empty:
        df["filing_date"] = pd.to_datetime(df["filing_date"])
        df["url"] = df.apply(lambda row: sec_filing_search_url(row["cik"], "10-K", row["filing_date"]), axis=1)
    return df


def price_window(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    query = """
        SELECT "Date", "Close"
        FROM stock_prices
        WHERE ticker = %s AND "Date" >= %s::date AND "Date" <= (%s::date + INTERVAL '90 days')
        ORDER BY "Date";
    """
    df = pd.read_sql(query, engine(), params=(ticker, start_date, end_date))
    if not df.empty:
        df["Date"] = pd.to_datetime(df["Date"])
    return df


def sp500_window(start_date: str, end_date: str) -> pd.DataFrame:
    query = """
        SELECT "Date", close
        FROM sp500_index
        WHERE "Date" >= %s::date AND "Date" <= (%s::date + INTERVAL '90 days')
        ORDER BY "Date";
    """
    df = pd.read_sql(query, engine(), params=(start_date, end_date))
    if not df.empty:
        df["Date"] = pd.to_datetime(df["Date"])
    return df


def trading_return(df: pd.DataFrame, anchor_date: Any, horizon: int, column: str) -> dict[str, Any] | None:
    if df.empty:
        return None
    df = df.sort_values("Date").reset_index(drop=True)
    future = df[df["Date"] >= pd.to_datetime(anchor_date)]
    if future.empty:
        return None
    base_index, target_index = future.index[0], future.index[0] + horizon
    if target_index >= len(df):
        return None
    base, target = float(df.loc[base_index, column]), float(df.loc[target_index, column])
    if not base:
        return None
    return {
        "anchor_date": df.loc[base_index, "Date"],
        "target_date": df.loc[target_index, "Date"],
        "return": (target / base) - 1,
    }


def reaction_label(excess: float | None) -> str:
    if excess is None:
        return "Insufficient data"
    if excess > 0.01:
        return "Positive vs S&P 500"
    if excess < -0.01:
        return "Negative vs S&P 500"
    return "Neutral vs S&P 500"


def horizons_for_filing(prices: pd.DataFrame, sp500: pd.DataFrame, filing_date: Any) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for horizon in (1, 5, 30):
        stock = trading_return(prices, filing_date, horizon, "Close")
        benchmark = trading_return(sp500, stock["anchor_date"], horizon, "close") if stock else None
        result[str(horizon)] = {
            "stock": number(stock["return"]) if stock else None,
            "sp500": number(benchmark["return"]) if benchmark else None,
            "excess": number(stock["return"] - benchmark["return"]) if stock and benchmark else None,
            "target_date": date_text(stock["target_date"]) if stock else None,
        }
    return result


def serialize_reactions(ticker: str, start_date: str, end_date: str) -> list[dict[str, Any]]:
    filings = load_10k_filings(ticker, start_date, end_date)
    prices, sp500 = price_window(ticker, start_date, end_date), sp500_window(start_date, end_date)
    output = []
    for row in filings.itertuples(index=False):
        horizons = horizons_for_filing(prices, sp500, row.filing_date)
        excess = horizons["5"]["excess"] if horizons["5"]["excess"] is not None else horizons["1"]["excess"]
        output.append(
            {
                "date": date_text(row.filing_date),
                "sections": row.sections,
                "url": row.url,
                "label": reaction_label(excess),
                "tone": "positive" if excess is not None and excess > 0.01 else "negative" if excess is not None and excess < -0.01 else "neutral",
                "horizons": horizons,
            }
        )
    return output


def latest_10k_excess(reactions: list[dict[str, Any]]) -> float | None:
    if not reactions:
        return None
    horizons = reactions[0]["horizons"]
    return next((horizons[key]["excess"] for key in ("5", "1", "30") if horizons[key]["excess"] is not None), None)


def tone_score(label: str) -> float:
    return {
        "Bullish": 1.25,
        "Expansion-Oriented": 1.0,
        "Aggressive": 0.5,
        "Defensive": -0.75,
        "Cautious": -1.0,
        "Risk-Elevated": -1.75,
    }.get(label, 0)


def tone_direction(label: str) -> int:
    if label in {"Bullish", "Expansion-Oriented", "Aggressive"}:
        return 1
    if label in {"Defensive", "Cautious", "Risk-Elevated"}:
        return -1
    return 0


def alignment(mdna: pd.Series | None, risk: pd.Series | None, ticker: str, excess: float | None) -> dict[str, Any] | None:
    if mdna is None or risk is None:
        return None
    prior_mdna, prior_risk = previous_section("mdna_sections", ticker, mdna["filing_date"]), previous_section("risk_sections", ticker, risk["filing_date"])
    if prior_mdna is None or prior_risk is None:
        return None
    metrics = section_metrics(mdna["content"], prior_mdna["content"])
    risk_added, _ = sentence_changes(risk["content"], prior_risk["content"], limit=20)
    current_mdna_tone, previous_mdna_tone = narrative_label(mdna["content"]), narrative_label(prior_mdna["content"])
    risk_tone = narrative_label(risk["content"])
    tone_delta = tone_score(current_mdna_tone) - tone_score(previous_mdna_tone)
    tone_change = "More constructive" if tone_delta >= 0.75 else "More cautious" if tone_delta <= -0.75 else "Stable"
    market = 1 if excess is not None and excess >= 0.01 else -1 if excess is not None and excess <= -0.01 else 0
    narrative = tone_direction(current_mdna_tone)
    risk_pressure = len(risk_added) >= 3 or risk_tone == "Risk-Elevated"
    if risk_pressure and market < 0:
        output, why = "Risk confirmed", "New risk language and a negative post-10-K excess return point in the same direction."
    elif narrative > 0 and market < 0:
        output, why = "Market skeptical", "The filing narrative reads constructive, but the market reaction lagged the S&P 500."
    elif narrative < 0 and market > 0:
        output, why = "Narrative ahead of market", "The filing language is cautious while the market reaction has not confirmed that caution."
    elif market and narrative == market:
        output, why = "Aligned", "Filing tone and post-10-K market reaction move in the same direction."
    elif risk_pressure:
        output, why = "Narrative ahead of market", "Risk language changed before a decisive market reaction appeared."
    else:
        output, why = "Aligned", "No strong narrative-market divergence is visible in the selected filing window."
    return {
        "output": output,
        "why": why,
        "tone": {"Aligned": "positive", "Narrative ahead of market": "ahead", "Market skeptical": "skeptical", "Risk confirmed": "negative"}[output],
        "tone_change": tone_change,
        "new_risk_count": len(risk_added),
        "excess_return": excess,
        "sentiment_shift": metrics["sentiment_delta"],
        "reaction": reaction_label(excess),
    }


def build_kpis(ticker: str, start_date: str, end_date: str, prices: pd.DataFrame, events: pd.DataFrame, mdna: pd.Series | None, risk: pd.Series | None) -> list[dict[str, str]]:
    return_value = volatility = average_volume = latest_close = None
    latest_price_date = "No prices"
    if not prices.empty:
        first_close, latest_close = float(prices.iloc[0]["Close"]), float(prices.iloc[-1]["Close"])
        latest_price_date = date_text(prices.iloc[-1]["Date"]) or latest_price_date
        return_value = (latest_close / first_close) - 1 if first_close else None
        daily_returns = prices["Close"].pct_change().dropna()
        volatility = float(daily_returns.std() * (252**0.5)) if not daily_returns.empty else None
        average_volume = float(prices["Volume"].mean())
    latest_filing = "N/A"
    latest_detail = "No filing in range"
    if not events.empty:
        last_event = events.sort_values("filing_date").iloc[-1]
        latest_filing, latest_detail = date_text(last_event["filing_date"]) or "N/A", f"{last_event['event_type']} filing"
    def badge(section: pd.Series | None) -> tuple[str, str, str]:
        label = narrative_label(section["content"] if section is not None else "")
        score = local_sentiment_score(section["content"] if section is not None else "")
        return label, f"Local score {score:+.3f}" if score is not None else "No local score", tone(label)
    mdna_badge, risk_badge = badge(mdna), badge(risk)
    return [
        {"label": "Return", "value": format_pct(return_value), "detail": f"{start_date} to {latest_price_date}", "tone": "positive" if return_value and return_value > 0 else "negative" if return_value and return_value < 0 else "neutral"},
        {"label": "Volatility", "value": format_pct(volatility), "detail": "Annualized daily close", "tone": "neutral"},
        {"label": "Avg Volume", "value": format_compact_number(average_volume), "detail": "Shares per trading day", "tone": "neutral"},
        {"label": "Latest Close", "value": format_price(latest_close), "detail": latest_price_date, "tone": "neutral"},
        {"label": "Latest Filing", "value": latest_filing, "detail": latest_detail, "tone": "neutral"},
        {"label": "MD&A Sentiment", "value": mdna_badge[0], "detail": mdna_badge[1], "tone": mdna_badge[2]},
        {"label": "Risk Sentiment", "value": risk_badge[0], "detail": risk_badge[1], "tone": risk_badge[2]},
    ]


def serialize_comparison(name: str, table: str, ticker: str, current: pd.Series | None) -> dict[str, Any]:
    if current is None:
        return {"name": name, "status": "empty", "message": f"No {name} filing found for the selected filing year."}
    previous = previous_section(table, ticker, current["filing_date"])
    current_date = date_text(current["filing_date"])
    if previous is None:
        return {"name": name, "status": "empty", "message": f"{name} filing found for {current_date}, but no prior filing is available for comparison."}
    previous_date = date_text(previous["filing_date"])
    added, removed = sentence_changes(current["content"], previous["content"])
    return {
        "name": name,
        "status": "ready",
        "current_date": current_date,
        "previous_date": previous_date,
        "current_url": sec_filing_search_url(current["cik"], "10-K", current["filing_date"]),
        "previous_url": sec_filing_search_url(previous["cik"], "10-K", previous["filing_date"]),
        "metrics": section_metrics(current["content"], previous["content"]),
        "ai_change_summary": summarize_changes(name, current_date or "", previous_date or "", current["content"], previous["content"]),
        "added": added,
        "removed": removed,
    }


def serialize_section(name: str, table: str, ticker: str, section: pd.Series | None, year_label: str) -> dict[str, Any]:
    if section is None:
        return {
            "name": name,
            "date": None,
            "url": None,
            "text": f"No {name} filing found for {ticker} in filing year {year_label}.",
            "summary": "",
            "sentiment": "N/A",
        }
    previous = previous_section(table, ticker, section["filing_date"])
    return {
        "name": name,
        "date": date_text(section["filing_date"]),
        "url": sec_filing_search_url(section["cik"], "10-K", section["filing_date"]),
        "text": (section["content"] or "")[:5000],
        "summary": summarize_section(name, section["content"] or "", previous["content"] if previous is not None else ""),
        "sentiment": narrative_label(section["content"] or ""),
    }


def serialize_charts(prices: pd.DataFrame, events: pd.DataFrame) -> dict[str, Any]:
    price_rows = [
        {
            "date": date_text(row.Date),
            "open": number(row.Open),
            "close": number(row.Close),
            "volume": number(row.Volume),
        }
        for row in prices.itertuples(index=False)
    ]
    event_rows = []
    for row in events.itertuples(index=False):
        if prices.empty:
            chart_date = None
            chart_close = None
        else:
            match = prices[prices["Date"] >= pd.to_datetime(row.filing_date)]
            point = match.iloc[0] if not match.empty else prices.iloc[-1]
            chart_date, chart_close = date_text(point["Date"]), number(point["Close"])
        event_rows.append(
            {
                "date": date_text(row.filing_date),
                "chart_date": chart_date,
                "close": chart_close,
                "type": row.event_type,
                "details": row.details,
                "url": getattr(row, "filing_url", "") or None,
            }
        )
    return {"prices": price_rows, "events": event_rows}


def market_readout(prices: pd.DataFrame, mdna: pd.Series | None, risk: pd.Series | None, events_8k: list[dict[str, Any]], comparison_alignment: dict[str, Any] | None, excess: float | None) -> dict[str, Any]:
    range_return = volatility = None
    if not prices.empty:
        first, latest = float(prices.iloc[0]["Close"]), float(prices.iloc[-1]["Close"])
        range_return = (latest / first) - 1 if first else None
        returns = prices["Close"].pct_change().dropna()
        volatility = float(returns.std() * (252**0.5)) if not returns.empty else None
    mdna_tone, risk_tone = narrative_label(mdna["content"] if mdna is not None else ""), narrative_label(risk["content"] if risk is not None else "")
    high = sum(event["impact_tone"] == "high" for event in events_8k)
    medium = sum(event["impact_tone"] == "medium" for event in events_8k)
    score = 0.0
    if range_return is not None:
        score += 1.25 if range_return >= 0.08 else -1.25 if range_return <= -0.08 else 0.4 if range_return > 0 else -0.4
    if excess is not None:
        score += 1.2 if excess >= 0.02 else -1.2 if excess <= -0.02 else 0
    if volatility is not None:
        score -= 0.8 if volatility >= 0.45 else 0.4 if volatility >= 0.32 else 0
    score += tone_score(mdna_tone) + tone_score(risk_tone) - min(high * 0.55, 1.1)
    stance = "Constructive" if score >= 1.75 else "Mixed" if score >= -0.75 else "Cautious" if score >= -2.25 else "Risk-Off"
    why = {
        "Constructive": "Price action and filing language lean supportive, with no dominant risk signal overwhelming the readout.",
        "Mixed": "Signals are not pointing in one clean direction, so the selected period needs confirmation.",
        "Cautious": "Filing language or market reaction suggests more caution than the headline price alone implies.",
        "Risk-Off": "Risk tone, market reaction, and event activity point to a materially defensive readout.",
    }[stance]
    if risk_tone == "Risk-Elevated":
        driver = "Risk language is elevated in the selected filing period."
    elif excess is not None and excess <= -0.02:
        driver = "The latest 10-K reaction underperformed the S&P 500."
    elif excess is not None and excess >= 0.02:
        driver = "The latest 10-K reaction outperformed the S&P 500."
    elif range_return is not None and abs(range_return) >= 0.08:
        driver = f"Stock performance was {'strong positive' if range_return > 0 else 'weak'} over the selected range."
    else:
        driver = "Filing tone is the main signal."
    watch = "Watch whether the next filing repeats the same risk language." if risk_tone in {"Risk-Elevated", "Cautious"} else "Watch the next 10-K or 8-K for a clearer narrative change."
    return {
        "stance": stance,
        "stance_tone": {"Constructive": "positive", "Mixed": "neutral", "Cautious": "cautious", "Risk-Off": "negative"}[stance],
        "score": round(score, 1),
        "why": why,
        "driver": driver,
        "watch": watch,
        "facts": {"return": format_pct(range_return), "risk_tone": risk_tone, "eight_k_impact": f"{high} high / {medium} medium"},
        "alignment": comparison_alignment,
    }


def build_dashboard(ticker: str, start_date: str, end_date: str) -> dict[str, Any]:
    start_year, end_year = pd.to_datetime(start_date).year, pd.to_datetime(end_date).year
    year_label = str(start_year) if start_year == end_year else f"{start_year}-{end_year}"
    prices, events = stock_data(ticker, start_date, end_date), filing_events(ticker, start_date, end_date)
    mdna, risk = section_for_years("mdna_sections", ticker, start_year, end_year), section_for_years("risk_sections", ticker, start_year, end_year)
    eight_k, reactions = serialize_8k(ticker, start_date, end_date), serialize_reactions(ticker, start_date, end_date)
    excess = latest_10k_excess(reactions)
    alignment_signal = alignment(mdna, risk, ticker, excess)
    return {
        "ticker": ticker,
        "range": {"start": start_date, "end": end_date, "filing_years": year_label},
        "kpis": build_kpis(ticker, start_date, end_date, prices, events, mdna, risk),
        "charts": serialize_charts(prices, events),
        "market_readout": market_readout(prices, mdna, risk, eight_k, alignment_signal, excess),
        "comparison": [
            serialize_comparison("MD&A", "mdna_sections", ticker, mdna),
            serialize_comparison("Risk", "risk_sections", ticker, risk),
        ],
        "eight_k_events": eight_k,
        "reactions": reactions,
        "sections": {
            "mdna": serialize_section("MD&A", "mdna_sections", ticker, mdna, year_label),
            "risk": serialize_section("Risk", "risk_sections", ticker, risk, year_label),
        },
    }
