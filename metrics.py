"""
metrics.py

Performance metrics used to compare Dad, the LLM personas and Nifty 50 buy-and-hold.
These are the numbers in the Results section, so you should be able to derive
each one on a whiteboard. Write them yourself: tests are in tests/test_metrics.py.

    pytest -q tests/test_metrics.py

Conventions:
  * `values`  = portfolio value per trading day (a list or array of floats)
  * `returns` = daily simple returns, value[t] / value[t-1] - 1
  * 252 trading days per year (for annualising daily numbers)
  * 365.25 calendar days per year (for CAGR, which uses dates)
"""

import numpy as np

TRADING_DAYS = 252


def daily_returns(values):
    """[100, 110, 99] -> [0.10, -0.10].

    TODO(kavya): one line with NumPy slicing. No loop.
    """
    raise NotImplementedError("TODO(kavya): daily_returns")


def cagr(values, n_days):
    """Compound annual growth rate: the constant yearly return that turns
    the first value into the last value over `n_days` calendar days.

    TODO(kavya): (end / start) ** (1 / years) - 1
    """
    raise NotImplementedError("TODO(kavya): cagr")


def sharpe(returns, rf_annual=0.0):
    """Annualised Sharpe ratio: mean excess return / its standard deviation, times sqrt(252).

    TODO(kavya):
      1. Excess return = daily return minus the DAILY risk-free rate (rf_annual / 252).
      2. Use the sample standard deviation (ddof=1).
      3. If the standard deviation is 0, return 0.0 (a flat line has no Sharpe).
    Interview question to prepare: why sqrt(252) and not 252?
    """
    raise NotImplementedError("TODO(kavya): sharpe")


def sortino(returns, rf_annual=0.0):
    """Like Sharpe, but only losses count as risk.

    TODO(kavya): same as sharpe, but the denominator is the downside deviation:
      sqrt(mean(min(excess, 0) ** 2))   (mean over ALL days, not just the losing ones)
    Return 0.0 if there is no downside.
    """
    raise NotImplementedError("TODO(kavya): sortino")


def max_drawdown(values):
    """Largest fall from a running peak, as a positive fraction. [100, 120, 90] -> 0.25.

    TODO(kavya): np.maximum.accumulate gives the running peak. No loop needed.
    """
    raise NotImplementedError("TODO(kavya): max_drawdown")


def turnover(trades, avg_portfolio_value, n_days):
    """Annual turnover: total rupees traded (buys and sells) / average portfolio value, per year.

    trades: list of dicts with "shares" (negative for sells) and "price".
    TODO(kavya): sum |shares * price|, divide by avg value, divide by years.
    Turnover tells you whether a strategy's returns would survive brokerage and taxes.
    """
    raise NotImplementedError("TODO(kavya): turnover")


def hit_rate(round_trips):
    """Share of closed trades (buy then sell) that made money.

    round_trips: list of profit/loss numbers, one per closed trade.
    TODO(kavya): return float("nan") for an empty list.
    """
    raise NotImplementedError("TODO(kavya): hit_rate")


def bootstrap_ci(returns, stat_fn, n_boot=2000, block=20, alpha=0.05, seed=0):
    """95% confidence interval for any statistic of a return series (e.g. sharpe).

    Daily returns are not independent (volatile days cluster), so resample
    BLOCKS of `block` consecutive days, not single days ("block bootstrap").

    TODO(kavya):
      rng = np.random.default_rng(seed)
      repeat n_boot times:
        pick ceil(n / block) random block start positions in [0, n - block]
        glue those blocks together, cut to length n, apply stat_fn
      return the alpha/2 and 1 - alpha/2 quantiles of the results, as floats
    """
    raise NotImplementedError("TODO(kavya): bootstrap_ci")
