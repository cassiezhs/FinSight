from data_ingestiion import load_to_db


def test_sec_collectors_skip_when_ticker_directory_is_unavailable(monkeypatch):
    monkeypatch.setattr(load_to_db, "get_company_ticker_ciks", lambda: {})

    risk_rows, mdna_rows = load_to_db.collect_sec_sections(["AAPL"], 2026, 2026)
    eight_k_rows = load_to_db.collect_8k_filings(["AAPL"], 2026, 2026)

    assert risk_rows.empty
    assert mdna_rows.empty
    assert eight_k_rows.empty
