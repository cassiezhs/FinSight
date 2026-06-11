from unittest.mock import Mock

import requests

from data_ingestiion import fetch_sec
from data_ingestiion.fetch_sec import _find_best_section, _normalize_text


def test_find_best_section_skips_short_table_of_contents_candidate():
    html = """
    <html><body>
      <p>Item 1A. Risk Factors Item 1B.</p>
      <h1>Part I</h1>
      <h2>Item 1A. Risk Factors</h2>
      <p>Actual risk disclosure starts here and contains enough text to be selected.
      The company faces operational, market, credit, regulatory, cybersecurity,
      liquidity, supplier, concentration, and competitive risks across regions.</p>
      <h2>Item 1B. Unresolved Staff Comments</h2>
    </body></html>
    """
    orig, lower = _normalize_text(html)

    section = _find_best_section(
        orig,
        lower,
        start_label="item 1a",
        end_labels=["item 1b"],
        prefer_after_label="part i",
        min_chars=80,
    )

    assert "Actual risk disclosure starts here" in section
    assert "Unresolved Staff Comments" not in section


def test_find_best_section_returns_empty_when_label_missing():
    orig, lower = _normalize_text("<p>Part I. Business overview only.</p>")

    assert _find_best_section(orig, lower, "item 7", ["item 8"]) == ""


def test_get_cik_fetches_ticker_directory_only_once(monkeypatch):
    response = Mock()
    response.json.return_value = {
        "0": {"ticker": "AAPL", "cik_str": 320193},
        "1": {"ticker": "MSFT", "cik_str": 789019},
    }
    get = Mock(return_value=response)
    monkeypatch.setattr(fetch_sec.requests, "get", get)
    fetch_sec.get_company_ticker_ciks.cache_clear()

    assert fetch_sec.get_cik("aapl") == "0000320193"
    assert fetch_sec.get_cik("MSFT") == "0000789019"
    assert get.call_count == 1

    fetch_sec.get_company_ticker_ciks.cache_clear()


def test_get_cik_caches_sec_directory_failure(monkeypatch):
    response = Mock()
    response.raise_for_status.side_effect = requests.HTTPError("403 Forbidden")
    get = Mock(return_value=response)
    monkeypatch.setattr(fetch_sec.requests, "get", get)
    monkeypatch.setattr(fetch_sec.time, "sleep", Mock())
    fetch_sec.get_company_ticker_ciks.cache_clear()

    assert fetch_sec.get_cik("AAPL") is None
    assert fetch_sec.get_cik("MSFT") is None
    assert get.call_count == 3

    fetch_sec.get_company_ticker_ciks.cache_clear()
