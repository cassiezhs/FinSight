import pandas as pd

from data_ingestiion import load_to_db


def test_sec_collectors_skip_when_ticker_directory_is_unavailable(monkeypatch):
    monkeypatch.setattr(load_to_db, "get_company_ticker_ciks", lambda: {})

    risk_rows, mdna_rows = load_to_db.collect_sec_sections(["AAPL"], 2026, 2026)
    eight_k_rows = load_to_db.collect_8k_filings(["AAPL"], 2026, 2026)

    assert risk_rows.empty
    assert mdna_rows.empty
    assert eight_k_rows.empty


def test_upsert_sections_deduplicates_conflict_keys(monkeypatch):
    rows = pd.DataFrame([
        {
            "cik": "0000320193",
            "company_name": "Old name",
            "filing_date": pd.Timestamp("2026-05-01"),
            "form_type": "10-Q",
            "content": "old content",
            "chunk_index": 0,
            "ticker": "AAPL",
        },
        {
            "cik": "0000320193",
            "company_name": "Apple Inc.",
            "filing_date": pd.Timestamp("2026-05-01"),
            "form_type": "10-Q",
            "content": "latest content",
            "chunk_index": 0,
            "ticker": "AAPL",
        },
    ])
    written = {}

    monkeypatch.setattr(load_to_db, "ensure_section_table_columns", lambda engine, table: None)
    monkeypatch.setattr(load_to_db, "_ensure_unique_index", lambda engine, table, keys: None)
    monkeypatch.setattr(
        pd.DataFrame,
        "to_sql",
        lambda self, *args, **kwargs: written.setdefault("rows", self.copy()),
    )

    class Result:
        rowcount = 1

    class Connection:
        def execute(self, statement):
            return Result()

    class Transaction:
        def __enter__(self):
            return Connection()

        def __exit__(self, exc_type, exc, traceback):
            return False

    class Engine:
        def begin(self):
            return Transaction()

    load_to_db.upsert_sections(rows, Engine(), "risk_sections")

    assert len(written["rows"]) == 1
    assert written["rows"].iloc[0]["content"] == "latest content"
