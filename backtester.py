"""
backtester.py

Simulates paper-trading strategies across historical NSE stock data.
Supports multi-frequency evaluation:
- Dad's persona evaluates quarterly (or monthly/yearly) to reflect long-term fundamental investing
- Momentum / Sentiment traders evaluate daily
- Value investor evaluates monthly
- Risk-averse investor evaluates weekly

Logs every trade with full LLM / rule reasoning and compares performance against
an Equal-Weight Buy & Hold benchmark and Nifty 50 buy-and-hold.

No look-ahead:
- Fundamentals are point-in-time: on each date a persona only sees annual statements
  published by then (period end + 60 days), and P/E from that day's price.
- Decisions use day t's close and are filled at day t+1's open, with trading costs.
"""

import os
from datetime import datetime
import numpy as np
import pandas as pd

import yfinance as yf

from fetch_data import fetch_and_save
from fundamentals import analyze_dad_rules, fetch_fundamentals_data, format_fundamentals_context, fundamentals_as_of
from llm_decision import get_trading_decision
from portfolio import Portfolio
from price_context import build_price_context

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


DEFAULT_FREQUENCIES = {
    "dad_hybrid": "quarterly",      # Dad holds for years, checks statements quarterly/yearly
    "value_investor": "monthly",    # Value investor checks monthly
    "risk_averse": "weekly",        # Risk averse rebalances weekly
    "momentum_trader": "daily",     # Momentum trades on daily price action
    "news_sentiment": "daily",      # Reacts daily to news/price shocks
}

FREQUENCY_DAYS = {
    "daily": 1,
    "weekly": 5,
    "monthly": 21,
    "quarterly": 63,
    "yearly": 252,
}


NIFTY50 = "^NSEI"
DEFAULT_COST_BPS = 15.0   # per side: brokerage + STT + exchange charges + slippage, roughly, for NSE delivery


def load_ticker_history(ticker: str, period: str = "1y", start: str = None, end: str = None) -> pd.DataFrame:
    """Load historical prices from a CSV cache, or fetch them.

    With start/end, the cache file is named after the date range, so a different range
    never silently reuses old data, and the same range always gives the same numbers.
    With only a relative period, the cache is refreshed daily ("1y" means something new every day).
    """
    clean_name = ticker.replace(".", "_").replace("^", "")
    if start or end:
        csv_path = os.path.join(DATA_DIR, f"{clean_name}_{start or 'begin'}_{end or 'today'}.csv")
    else:
        csv_path = os.path.join(DATA_DIR, f"{clean_name}_{period}_{datetime.now():%Y-%m-%d}.csv")

    if not os.path.exists(csv_path):
        print(f"Fetching historical data for {ticker} ({start or period} to {end or 'today'})...")
        if start or end:
            df = yf.Ticker(ticker).history(start=start, end=end, auto_adjust=False)
            if not df.empty:
                os.makedirs(DATA_DIR, exist_ok=True)
                df.to_csv(csv_path)
        else:
            df = fetch_and_save(ticker, period=period)
            if not df.empty:
                df.to_csv(csv_path)
        df = df.reset_index()
        if df.empty:
            return pd.DataFrame()
    else:
        df = pd.read_csv(csv_path)

    # Ensure Date column or index is standard
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], utc=True).dt.tz_localize(None)
        df = df.set_index("Date")
    else:
        df.index = pd.to_datetime(df.index, utc=True).dt.tz_localize(None)

    df = df.sort_index()
    # Strip any extra time info, keep clean dates
    df.index = df.index.normalize()
    return df


class Backtester:
    def __init__(
        self,
        tickers: list = None,
        period: str = "1y",
        initial_cash: float = 100000.0,
        personas: list = None,
        eval_frequencies: dict = None,
        max_position_pct: float = 0.35,
        llm_provider: str = "mock",
        api_key: str = None,
        start: str = None,
        end: str = None,
        cost_bps: float = DEFAULT_COST_BPS,
    ):
        self.tickers = tickers or ["RELIANCE.NS", "TCS.NS", "INFY.NS"]
        self.period = period
        self.start, self.end = start, end
        self.cost = cost_bps / 10_000
        self.initial_cash = initial_cash
        self.personas = personas or ["dad_hybrid", "momentum_trader", "value_investor", "risk_averse"]
        self.eval_frequencies = {**DEFAULT_FREQUENCIES, **(eval_frequencies or {})}
        self.max_position_pct = max_position_pct
        self.llm_provider = llm_provider
        self.api_key = api_key

        self.price_data = {}
        self.fundamentals_data = {}
        self.fundamentals_text = {}
        self.dad_analysis = {}
        self.nifty = None
        self.common_dates = []

    def prepare_data(self):
        """Fetch and align all historical prices and fundamentals."""
        span = f"{self.start} to {self.end or 'today'}" if self.start else self.period
        print(f"Loading market data for {len(self.tickers)} tickers ({span})...")
        for ticker in self.tickers:
            df = load_ticker_history(ticker, period=self.period, start=self.start, end=self.end)
            if not df.empty:
                self.price_data[ticker] = df

            # Fundamentals are downloaded once (with dates), then cut to what was known on each
            # simulated day in _fundamentals_on(). Never use the raw snapshot in decisions.
            try:
                self.fundamentals_data[ticker] = fetch_fundamentals_data(ticker)
            except Exception as e:
                print(f"  Warning: could not fetch fundamentals for {ticker}: {e}")
                self.fundamentals_data[ticker] = {}

        if not self.price_data:
            raise ValueError("No price data available for any ticker.")

        # Trade only on days every ticker has a price (sorted union as a fallback).
        date_sets = [set(df.index) for df in self.price_data.values()]
        common = sorted(set.intersection(*date_sets))
        self.common_dates = common or sorted(set.union(*date_sets))

        try:
            nifty = load_ticker_history(NIFTY50, period=self.period, start=self.start, end=self.end)
            if not nifty.empty:
                self.nifty = nifty["Close"].reindex(self.common_dates).ffill().bfill()
        except Exception as e:
            print(f"  Warning: no Nifty 50 benchmark ({e})")

        print(f"Data aligned. Found {len(self.common_dates)} trading days from {self.common_dates[0]:%Y-%m-%d} to {self.common_dates[-1]:%Y-%m-%d}.")

    def _fundamentals_on(self, ticker, date, price):
        """(point-in-time fundamentals, Dad-rule analysis, LLM context text) for `ticker` on `date`."""
        raw = self.fundamentals_data.get(ticker) or {}
        if not raw:
            return {}, {}, ""
        pit = fundamentals_as_of(raw, date, price=price)
        analysis = analyze_dad_rules(pit)
        return pit, analysis, format_fundamentals_context(pit, analysis)

    def _price(self, ticker, dt, field="Close"):
        df = self.price_data.get(ticker)
        if df is None or dt not in df.index:
            return None
        col = field if field in df.columns and pd.notna(df.loc[dt, field]) else "Close"
        v = float(df.loc[dt, col])
        return v if v > 0 else None

    def _execute(self, p, order, fill_date, open_prices):
        """Fill an order decided on the previous day at today's open, including costs."""
        ticker, action, conf, reason = order["ticker"], order["action"], order["confidence"], order["reasoning"]
        px = open_prices.get(ticker)
        if not px:
            return
        note = f"[decided {order['decided_on']}, filled at next open] {reason}"
        if action == "buy":
            fill = px * (1 + self.cost)
            total_val = p.get_value(open_prices)
            target_alloc = total_val * self.max_position_pct
            alloc_room = max(0.0, target_alloc - p.holdings.get(ticker, 0) * px)
            spendable = min(p.cash, alloc_room if alloc_room > 0 else p.cash * 0.4)
            shares = int(spendable // fill)
            if shares > 0:
                p.buy(ticker, shares, fill, trade_date=fill_date, confidence=conf, reasoning=note)
            else:
                p.hold(ticker, trade_date=fill_date, confidence=conf, reasoning=f"Cash or allocation cap reached. {note}")
        elif action == "sell":
            held = p.holdings.get(ticker, 0)
            if held > 0:
                p.sell(ticker, held, px * (1 - self.cost), trade_date=fill_date, confidence=conf, reasoning=note)
            else:
                p.hold(ticker, trade_date=fill_date, confidence=conf, reasoning=f"No active position to sell. {note}")

    def run(self, progress_callback=None) -> dict:
        """Execute simulation across all personas and benchmarks."""
        if not self.common_dates:
            self.prepare_data()

        portfolios = {p: Portfolio(starting_cash=self.initial_cash, name=p) for p in self.personas}
        pending = {p: [] for p in self.personas}       # orders decided today, filled tomorrow

        # Equal-weight buy & hold of the chosen stocks, bought at the first day's close (no costs: it's the yardstick).
        benchmark = Portfolio(starting_cash=self.initial_cash, name="buy_and_hold_benchmark")
        d0 = self.common_dates[0]
        per_ticker_cash = self.initial_cash / len(self.tickers)
        for ticker in self.tickers:
            p0 = self._price(ticker, d0)
            if p0 and int(per_ticker_cash // p0) > 0:
                benchmark.buy(ticker, int(per_ticker_cash // p0), p0, trade_date=f"{d0:%Y-%m-%d}", reasoning="Initial Buy & Hold allocation.")

        # Nifty 50 buy & hold (fractional units of the index, so it tracks the index exactly).
        nifty_pf = None
        if self.nifty is not None:
            nifty_pf = Portfolio(starting_cash=self.initial_cash, name="nifty50_buy_and_hold")
            nifty_pf.buy(NIFTY50, self.initial_cash / float(self.nifty.iloc[0]), float(self.nifty.iloc[0]), trade_date=f"{d0:%Y-%m-%d}", reasoning="Nifty 50 buy & hold.")

        step_counters = {p: 0 for p in self.personas}
        total_days = len(self.common_dates)

        for idx, current_dt in enumerate(self.common_dates):
            date_str = current_dt.strftime("%Y-%m-%d")
            close = {t: px for t in self.tickers if (px := self._price(t, current_dt))}
            opens = {t: px for t in self.tickers if (px := self._price(t, current_dt, "Open"))}
            if not close:
                continue

            # 1. Yesterday's decisions fill at today's open.
            for persona in self.personas:
                for order in pending[persona]:
                    self._execute(portfolios[persona], order, date_str, opens)
                pending[persona] = []

            benchmark.record_daily(date_str, close)
            if nifty_pf is not None:
                nifty_pf.record_daily(date_str, {NIFTY50: float(self.nifty.loc[current_dt])})

            # 2. Today's decisions, from information available at today's close.
            is_last_day = idx == total_days - 1
            for persona in self.personas:
                interval = FREQUENCY_DAYS.get(self.eval_frequencies.get(persona, "daily"), 1)
                is_eval_day = (step_counters[persona] % interval == 0) and not is_last_day
                step_counters[persona] += 1

                if is_eval_day:
                    for ticker, price in close.items():
                        price_ctx = build_price_context(ticker, date_str, self.price_data[ticker])
                        fund, analysis, fund_text = self._fundamentals_on(ticker, current_dt, price)
                        decision = get_trading_decision(
                            persona_name=persona,
                            ticker=ticker,
                            date_str=date_str,
                            price_ctx=price_ctx,
                            fundamentals_ctx_text=fund_text,
                            fundamentals_data=fund,
                            dad_analysis=analysis,
                            provider=self.llm_provider,
                            api_key=self.api_key,
                        )
                        action = decision.get("action", "hold")
                        if action in ("buy", "sell"):
                            pending[persona].append({
                                "ticker": ticker, "action": action, "decided_on": date_str,
                                "confidence": decision.get("confidence", 0.5),
                                "reasoning": decision.get("reasoning", ""),
                            })
                        else:
                            portfolios[persona].hold(ticker, trade_date=date_str,
                                                     confidence=decision.get("confidence", 0.5),
                                                     reasoning=decision.get("reasoning", ""))

                portfolios[persona].record_daily(date_str, close)

            if progress_callback and (idx % 10 == 0 or is_last_day):
                progress_callback(idx + 1, total_days)

        # Dad's scorecard as of the last day, for the dashboard.
        last = self.common_dates[-1]
        for ticker in self.tickers:
            _, self.dad_analysis[ticker], _ = self._fundamentals_on(ticker, last, self._price(ticker, last))

        metrics_rows = []
        for persona in self.personas:
            m = portfolios[persona].get_metrics()
            m["persona"] = persona
            m["frequency"] = self.eval_frequencies.get(persona, "daily")
            metrics_rows.append(m)
        for pf, label in [(benchmark, "Buy & Hold (Benchmark)"), (nifty_pf, "Nifty 50 Buy & Hold")]:
            if pf is not None:
                m = pf.get_metrics()
                m["persona"], m["frequency"] = label, "passive"
                metrics_rows.append(m)

        metrics_df = pd.DataFrame(metrics_rows).sort_values("total_return_pct", ascending=False).reset_index(drop=True)

        equity_dict = {"date": [d.strftime("%Y-%m-%d") for d in self.common_dates]}
        for persona in self.personas:
            equity_dict[persona] = [h["total_value"] for h in portfolios[persona].daily_history]
        equity_dict["Buy & Hold Benchmark"] = [h["total_value"] for h in benchmark.daily_history]
        if nifty_pf is not None:
            equity_dict["Nifty 50"] = [h["total_value"] for h in nifty_pf.daily_history]
        equity_df = pd.DataFrame(equity_dict)

        for p in portfolios.values():
            p.save_log()
        benchmark.save_log()

        return {
            "metrics_df": metrics_df,
            "equity_df": equity_df,
            "portfolios": portfolios,
            "benchmark": benchmark,
            "nifty": nifty_pf,
            "dad_analysis": self.dad_analysis,
            "tickers": self.tickers,
            "common_dates": [d.strftime("%Y-%m-%d") for d in self.common_dates],
        }


if __name__ == "__main__":
    print("Testing Backtester with default tickers...")
    bt = Backtester(tickers=["RELIANCE.NS", "TCS.NS", "INFY.NS"], period="6mo", personas=["dad_hybrid", "momentum_trader", "value_investor"])
    results = bt.run()
    print("\n=== BACKTEST RESULTS LEADERBOARD ===")
    print(results["metrics_df"][["persona", "frequency", "total_return_pct", "sharpe_ratio", "max_drawdown_pct", "total_trades"]].to_string())
