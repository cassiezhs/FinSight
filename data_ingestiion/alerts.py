"""Email filing alerts for FinSight."""

from __future__ import annotations

import os
import re
import smtplib
from email.message import EmailMessage
from typing import Any

import pandas as pd
from sqlalchemy import text

try:
    from .config import settings
except ImportError:
    from config import settings


DB_SCHEMA = settings.db_schema
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def _quote_identifier(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _qualified(table_name: str) -> str:
    return f'{_quote_identifier(DB_SCHEMA)}.{_quote_identifier(table_name)}'


def normalize_email(email: str) -> str:
    email = (email or "").strip().lower()
    if not EMAIL_RE.match(email):
        raise ValueError("Enter a valid email address.")
    return email


def normalize_ticker(ticker: str) -> str:
    ticker = (ticker or "").strip().upper()
    if not ticker or not re.fullmatch(r"[A-Z0-9.\-*]{1,16}", ticker):
        raise ValueError("Enter a valid ticker or * for all tickers.")
    return ticker.replace(".", "-")


def ensure_alert_tables(engine) -> None:
    with engine.begin() as conn:
        conn.execute(text(f"""
            CREATE TABLE IF NOT EXISTS {_qualified("alert_subscriptions")} (
                id BIGSERIAL PRIMARY KEY,
                email TEXT NOT NULL,
                ticker TEXT NOT NULL,
                active BOOLEAN NOT NULL DEFAULT TRUE,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                UNIQUE (email, ticker)
            )
        """))
        conn.execute(text(f"""
            CREATE TABLE IF NOT EXISTS {_qualified("alert_sent")} (
                id BIGSERIAL PRIMARY KEY,
                subscription_id BIGINT NOT NULL REFERENCES {_qualified("alert_subscriptions")}(id) ON DELETE CASCADE,
                filing_key TEXT NOT NULL,
                sent_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                UNIQUE (subscription_id, filing_key)
            )
        """))


def subscribe_alert(engine, email: str, ticker: str) -> dict[str, Any]:
    ensure_alert_tables(engine)
    email, ticker = normalize_email(email), normalize_ticker(ticker)
    with engine.begin() as conn:
        row = conn.execute(
            text(f"""
                INSERT INTO {_qualified("alert_subscriptions")} (email, ticker, active, updated_at)
                VALUES (:email, :ticker, TRUE, NOW())
                ON CONFLICT (email, ticker)
                DO UPDATE SET active = TRUE, updated_at = NOW()
                RETURNING id, email, ticker, active
            """),
            {"email": email, "ticker": ticker},
        ).mappings().one()
    return dict(row)


def active_subscriptions(engine) -> pd.DataFrame:
    ensure_alert_tables(engine)
    return pd.read_sql(
        f"SELECT id, email, ticker FROM {_qualified('alert_subscriptions')} WHERE active = TRUE",
        engine,
    )


def _existing_periodic_keys(engine) -> set[str]:
    query = f"""
        SELECT DISTINCT ticker, filing_date, COALESCE(form_type, '10-K') AS form_type
        FROM {_qualified("risk_sections")}
        UNION
        SELECT DISTINCT ticker, filing_date, COALESCE(form_type, '10-K') AS form_type
        FROM {_qualified("mdna_sections")}
    """
    try:
        df = pd.read_sql(query, engine)
    except Exception:
        return set()
    return {f"{row.ticker}|{pd.to_datetime(row.filing_date).date().isoformat()}|{row.form_type}" for row in df.itertuples(index=False)}


def new_periodic_filings(risk_rows: pd.DataFrame, mdna_rows: pd.DataFrame, engine) -> list[dict[str, Any]]:
    combined = pd.concat([risk_rows, mdna_rows], ignore_index=True)
    if combined.empty:
        return []
    existing = _existing_periodic_keys(engine)
    combined = combined.drop_duplicates(subset=["ticker", "filing_date", "form_type"])
    filings = []
    for row in combined.itertuples(index=False):
        filing_date = pd.to_datetime(row.filing_date).date().isoformat()
        key = f"{row.ticker}|{filing_date}|{row.form_type}"
        if key in existing:
            continue
        filings.append({
            "ticker": row.ticker,
            "form_type": row.form_type,
            "filing_date": filing_date,
            "filing_key": f"periodic|{key}",
            "summary": f"{row.ticker} filed a new {row.form_type} on {filing_date}.",
        })
    return filings


def new_8k_filings(eight_k_rows: pd.DataFrame, engine) -> list[dict[str, Any]]:
    if eight_k_rows.empty:
        return []
    try:
        existing = set(pd.read_sql(
            f"SELECT accession_number FROM {_qualified('sec_8k_filings')}",
            engine,
        )["accession_number"].astype(str))
    except Exception:
        existing = set()
    filings = []
    for row in eight_k_rows.drop_duplicates(subset=["cik", "accession_number"]).itertuples(index=False):
        accession = str(row.accession_number)
        if accession in existing:
            continue
        filing_date = pd.to_datetime(row.filing_date).date().isoformat()
        form_type = getattr(row, "form_type", "8-K")
        items = getattr(row, "item_descriptions", "") or getattr(row, "items", "")
        filings.append({
            "ticker": row.ticker,
            "form_type": form_type,
            "filing_date": filing_date,
            "filing_key": f"8k|{row.cik}|{accession}",
            "summary": f"{row.ticker} filed a new {form_type} on {filing_date}. {items}".strip(),
        })
    return filings


def smtp_ready() -> bool:
    return bool(os.getenv("SMTP_HOST") and os.getenv("SMTP_FROM"))


def send_email(to_email: str, subject: str, body: str) -> None:
    host = os.getenv("SMTP_HOST")
    from_email = os.getenv("SMTP_FROM")
    if not host or not from_email:
        raise RuntimeError("SMTP_HOST and SMTP_FROM must be configured.")
    port = int(os.getenv("SMTP_PORT", "587"))
    username = os.getenv("SMTP_USERNAME")
    password = os.getenv("SMTP_PASSWORD")
    use_tls = os.getenv("SMTP_TLS", "1").strip().lower() in {"1", "true", "yes", "on"}

    message = EmailMessage()
    message["From"] = from_email
    message["To"] = to_email
    message["Subject"] = subject
    message.set_content(body)

    with smtplib.SMTP(host, port, timeout=30) as smtp:
        if use_tls:
            smtp.starttls()
        if username and password:
            smtp.login(username, password)
        smtp.send_message(message)


def send_filing_alerts(engine, filings: list[dict[str, Any]]) -> int:
    if not filings:
        print("No new filing alerts to send.")
        return 0
    subscriptions = active_subscriptions(engine)
    if subscriptions.empty:
        print("No active alert subscriptions.")
        return 0
    if not smtp_ready():
        print("SMTP is not configured; skipping email delivery.")
        return 0

    sent = 0
    with engine.begin() as conn:
        for filing in filings:
            matches = subscriptions[
                (subscriptions["ticker"] == filing["ticker"])
                | (subscriptions["ticker"] == "*")
            ]
            for sub in matches.itertuples(index=False):
                already_sent = conn.execute(
                    text(f"""
                        SELECT 1 FROM {_qualified("alert_sent")}
                        WHERE subscription_id = :subscription_id AND filing_key = :filing_key
                    """),
                    {"subscription_id": int(sub.id), "filing_key": filing["filing_key"]},
                ).first()
                if already_sent:
                    continue
                subject = f"FinSight alert: {filing['ticker']} {filing['form_type']} filed"
                body = (
                    f"{filing['summary']}\n\n"
                    "Open your FinSight dashboard to review the filing, market reaction, financial statements, and decision readiness.\n\n"
                    "Research aid only, not investment advice."
                )
                send_email(sub.email, subject, body)
                conn.execute(
                    text(f"""
                        INSERT INTO {_qualified("alert_sent")} (subscription_id, filing_key)
                        VALUES (:subscription_id, :filing_key)
                        ON CONFLICT (subscription_id, filing_key) DO NOTHING
                    """),
                    {"subscription_id": int(sub.id), "filing_key": filing["filing_key"]},
                )
                sent += 1
    print(f"Sent {sent} filing alert emails.")
    return sent
