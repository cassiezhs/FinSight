import pandas as pd

from data_ingestiion.event_mapping import add_event_dates, map_to_first_trading_day


def test_map_to_first_trading_day_uses_next_available_date():
    prices = pd.DataFrame({
        "ticker": ["AAPL", "AAPL", "AAPL"],
        "date": pd.to_datetime(["2024-01-05", "2024-01-08", "2024-01-09"]),
    })

    event_date = map_to_first_trading_day(pd.Timestamp("2024-01-06"), "aapl", prices)

    assert event_date == pd.Timestamp("2024-01-08")


def test_map_to_first_trading_day_returns_nat_when_no_future_price():
    prices = pd.DataFrame({
        "ticker": ["MSFT"],
        "date": pd.to_datetime(["2024-01-05"]),
    })

    assert pd.isna(map_to_first_trading_day(pd.Timestamp("2024-01-06"), "MSFT", prices))


def test_add_event_dates_preserves_input_columns():
    filings = pd.DataFrame({
        "ticker": ["NVDA"],
        "filing_date": pd.to_datetime(["2024-03-02"]),
        "content": ["text"],
    })
    prices = pd.DataFrame({
        "ticker": ["NVDA"],
        "date": pd.to_datetime(["2024-03-04"]),
    })

    out = add_event_dates(filings, prices)

    assert list(out.columns) == ["ticker", "filing_date", "content", "event_date"]
    assert out.loc[0, "event_date"] == pd.Timestamp("2024-03-04")

