"""The backtest must never use information from after the simulated date. All offline."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import backtester as B
from fundamentals import analyze_dad_rules, fundamentals_as_of


def fake_fundamentals(future_revenue=1e12):
    """Annual reports for FY ending 31 Mar 2022, 2023, 2024 and 2025."""
    ends = ["2022-03-31", "2023-03-31", "2024-03-31", "2025-03-31"]
    rev = [100e9, 120e9, 150e9, future_revenue]
    return {
        "ticker": "TEST.NS", "sector": "Technology", "market_cap": 5e12,
        "trailing_pe": 99.0, "forward_pe": 50.0, "dividend_yield": 0.05,
        "financial_series": {"revenues": rev, "net_incomes": [10e9, 12e9, 15e9, 99e9]},
        "financial_series_dated": {
            "revenues": list(map(list, zip(ends, rev))),
            "net_incomes": list(map(list, zip(ends, [10e9, 12e9, 15e9, 99e9]))),
            "stockholders_equity": list(map(list, zip(ends, [50e9, 55e9, 60e9, 1e9]))),
            "eps": list(map(list, zip(ends, [10.0, 12.0, 15.0, 100.0]))),
        },
        "dividends_dated": [["2023-07-01", 5.0], ["2024-07-01", 6.0], ["2025-07-01", 50.0]],
    }


def test_only_published_statements_are_visible():
    # On 15 Apr 2024, FY2024 (ended 31 Mar 2024) isn't published yet: 60-day reporting lag.
    pit = fundamentals_as_of(fake_fundamentals(), "2024-04-15", price=300.0)
    assert pit["financial_series"]["revenues"] == [100e9, 120e9]
    # By 1 Jun 2024 it is.
    pit = fundamentals_as_of(fake_fundamentals(), "2024-06-01", price=300.0)
    assert pit["financial_series"]["revenues"] == [100e9, 120e9, 150e9]


def test_pe_and_dividends_use_that_days_price_and_past_payouts():
    pit = fundamentals_as_of(fake_fundamentals(), "2024-06-01", price=300.0)
    assert pit["trailing_pe"] == pytest.approx(300.0 / 15.0)        # last reported EPS, not today's 99x
    assert pit["dividend_yield"] == pytest.approx(5.0 / 300.0)        # only the July 2023 payout is in the past 12 months
    assert pit["forward_pe"] is None and pit["market_cap"] is None    # today's numbers are dropped


def test_a_future_report_cannot_change_a_past_decision():
    a = analyze_dad_rules(fundamentals_as_of(fake_fundamentals(future_revenue=1e12), "2024-06-01", price=300.0))
    b = analyze_dad_rules(fundamentals_as_of(fake_fundamentals(future_revenue=1.0), "2024-06-01", price=300.0))
    assert a == b


def test_old_cache_without_dates_reports_missing_data_instead_of_peeking():
    old = {k: v for k, v in fake_fundamentals().items() if k not in ("financial_series_dated", "dividends_dated")}
    pit = fundamentals_as_of(old, "2024-06-01", price=300.0)
    assert pit["financial_series"]["revenues"] == []


def make_prices(n=6):
    dates = pd.bdate_range("2024-06-03", periods=n)
    close = np.linspace(100, 110, n)
    return pd.DataFrame({"Open": close - 1, "Close": close, "Volume": 1_000}, index=dates)


def test_orders_fill_at_next_open_with_costs(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "logs").mkdir()
    prices = make_prices()
    monkeypatch.setattr(B, "load_ticker_history", lambda t, **kw: prices if t == "TEST.NS" else pd.DataFrame())
    monkeypatch.setattr(B, "fetch_fundamentals_data", lambda t: {})
    calls = []

    def decide(**kw):
        calls.append(kw["date_str"])
        return {"action": "buy" if len(calls) == 1 else "hold", "confidence": 1.0, "reasoning": "test"}

    monkeypatch.setattr(B, "get_trading_decision", decide)
    bt = B.Backtester(tickers=["TEST.NS"], personas=["momentum_trader"], initial_cash=10_000,
                      max_position_pct=1.0, start="2024-06-01", end="2024-06-30", cost_bps=10)
    res = bt.run()
    buys = [t for t in res["portfolios"]["momentum_trader"].trade_log if t["action"] == "buy"]
    assert len(buys) == 1
    day1 = prices.index[1]
    assert buys[0]["date"] == f"{day1:%Y-%m-%d}"                                  # decided day 0, filled day 1
    assert buys[0]["price"] == pytest.approx(prices.loc[day1, "Open"] * 1.001, abs=0.01)  # at the open, plus 10 bps
    assert calls[0] == f"{prices.index[0]:%Y-%m-%d}"
    assert f"{prices.index[-1]:%Y-%m-%d}" not in calls                             # no decision that could never fill
