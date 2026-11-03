"""Tests for metrics.py. Every expected number here can be worked out by hand."""
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import metrics as M


def test_daily_returns():
    assert np.allclose(M.daily_returns([100, 110, 99]), [0.10, -0.10])


def test_cagr_doubling_in_two_years():
    # 100 -> 200 over two years is sqrt(2) - 1 per year, about 41.4%
    assert M.cagr([100, 150, 200], n_days=2 * 365.25) == pytest.approx(math.sqrt(2) - 1)


def test_sharpe_known_value():
    r = np.array([0.01, -0.01, 0.02, 0.0])
    expected = r.mean() / r.std(ddof=1) * math.sqrt(252)
    assert M.sharpe(r) == pytest.approx(expected)


def test_sharpe_subtracts_risk_free_rate():
    r = np.array([0.01, -0.01, 0.02, 0.0])
    ex = r - 0.07 / 252
    assert M.sharpe(r, rf_annual=0.07) == pytest.approx(ex.mean() / ex.std(ddof=1) * math.sqrt(252))


def test_sharpe_flat_line_is_zero():
    assert M.sharpe(np.zeros(10)) == 0.0


def test_sortino_only_penalises_losses():
    up_only = np.array([0.01, 0.02, 0.0, 0.03])
    assert M.sortino(up_only) == 0.0          # no downside -> defined as 0 here
    r = np.array([0.02, -0.01, 0.03, -0.02])
    downside_dev = math.sqrt(((np.minimum(r, 0)) ** 2).mean())
    assert M.sortino(r) == pytest.approx(r.mean() / downside_dev * math.sqrt(252))


def test_max_drawdown():
    # peak 120, trough 90 -> 25% drawdown, even though it recovers later
    assert M.max_drawdown([100, 120, 90, 130, 125]) == pytest.approx(0.25)
    assert M.max_drawdown([100, 101, 102]) == 0.0


def test_turnover():
    trades = [{"shares": 10, "price": 100}, {"shares": -10, "price": 110}]   # buy 1000, sell 1100
    # 2100 traded on an average 10,000 portfolio over one year
    assert M.turnover(trades, avg_portfolio_value=10_000, n_days=365.25) == pytest.approx(0.21)


def test_hit_rate():
    assert M.hit_rate([50, -20, 10, -5]) == pytest.approx(0.5)
    assert math.isnan(M.hit_rate([]))


def test_bootstrap_ci_contains_the_point_estimate_and_is_reproducible():
    rng = np.random.default_rng(1)
    r = rng.normal(0.0005, 0.01, size=500)
    lo, hi = M.bootstrap_ci(r, np.mean, n_boot=500, seed=0)
    assert lo < r.mean() < hi
    assert (lo, hi) == M.bootstrap_ci(r, np.mean, n_boot=500, seed=0)
    lo2, hi2 = M.bootstrap_ci(r, np.mean, n_boot=500, seed=0, alpha=0.5)
    assert hi2 - lo2 < hi - lo                 # a 50% interval is narrower than a 95% one
