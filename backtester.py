"""
backtester.py

Simulates paper-trading strategies across historical NSE stock data.
Supports multi-frequency evaluation:
- Dad's persona evaluates quarterly (or monthly/yearly) to reflect long-term fundamental investing
- Momentum / Sentiment traders evaluate daily
- Value investor evaluates monthly
- Risk-averse investor evaluates weekly

Logs every trade with full LLM / rule reasoning and compares performance against
an Equal-Weight Buy & Hold benchmark.
"""

import os
from datetime import datetime
import numpy as np
import pandas as pd

from fetch_data import fetch_and_save
from fundamentals import analyze_dad_rules, build_fundamentals_context, fetch_fundamentals_data
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


def load_ticker_history(ticker: str, period: str = "1y") -> pd.DataFrame:
    """Load historical price DataFrame from CSV, or fetch if missing."""
    clean_name = ticker.replace(".", "_")
    csv_path = os.path.join(DATA_DIR, f"{clean_name}.csv")

    if not os.path.exists(csv_path):
        print(f"Fetching historical data for {ticker} ({period})...")
        df = fetch_and_save(ticker, period=period)
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
    ):
        self.tickers = tickers or ["RELIANCE.NS", "TCS.NS", "INFY.NS"]
        self.period = period
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
        self.common_dates = []

    def prepare_data(self):
        """Fetch and align all historical prices and fundamentals."""
        print(f"Loading market data for {len(self.tickers)} tickers over {self.period}...")
        for ticker in self.tickers:
            df = load_ticker_history(ticker, period=self.period)
            if not df.empty:
                self.price_data[ticker] = df

            # Fundamentals
            try:
                fund = fetch_fundamentals_data(ticker)
                self.fundamentals_data[ticker] = fund
                self.dad_analysis[ticker] = analyze_dad_rules(fund)
                self.fundamentals_text[ticker] = build_fundamentals_context(ticker)
            except Exception as e:
                print(f"  Warning: could not fetch fundamentals for {ticker}: {e}")
                self.fundamentals_data[ticker] = {}
                self.dad_analysis[ticker] = {}
                self.fundamentals_text[ticker] = ""

        if not self.price_data:
            raise ValueError("No price data available for any ticker.")

        # Find overlapping dates across all tickers
        date_sets = [set(df.index) for df in self.price_data.values()]
        common = sorted(list(set.intersection(*date_sets)))
        if not common:
            # Fallback to sorted union if dates differ slightly
            all_dates = sorted(list(set.union(*date_sets)))
            self.common_dates = all_dates
        else:
            self.common_dates = common

        print(f"Data aligned. Found {len(self.common_dates)} trading days from {self.common_dates[0].strftime('%Y-%m-%d')} to {self.common_dates[-1].strftime('%Y-%m-%d')}.")

    def run(self, progress_callback=None) -> dict:
        """Execute simulation across all personas and benchmarks."""
        if not self.common_dates:
            self.prepare_data()

        # Initialize portfolios
        portfolios = {p: Portfolio(starting_cash=self.initial_cash, name=p) for p in self.personas}

        # Equal-Weight Buy & Hold benchmark
        benchmark = Portfolio(starting_cash=self.initial_cash, name="buy_and_hold_benchmark")

        # Set up benchmark on Day 0: divide initial cash equally among all tickers
        day_0_date = self.common_dates[0].strftime("%Y-%m-%d")
        per_ticker_cash = self.initial_cash / len(self.tickers)
        for ticker in self.tickers:
            if ticker in self.price_data and self.common_dates[0] in self.price_data[ticker].index:
                p0 = float(self.price_data[ticker].loc[self.common_dates[0], "Close"])
                shares0 = int(per_ticker_cash // p0)
                if shares0 > 0:
                    benchmark.buy(ticker, shares0, p0, trade_date=day_0_date, reasoning="Initial Buy & Hold allocation.")

        # Tracking step count per persona to control evaluation frequencies
        step_counters = {p: 0 for p in self.personas}
        total_days = len(self.common_dates)

        for idx, current_dt in enumerate(self.common_dates):
            date_str = current_dt.strftime("%Y-%m-%d")

            # Collect current closing prices for this day
            current_prices = {}
            for ticker in self.tickers:
                if ticker in self.price_data and current_dt in self.price_data[ticker].index:
                    current_prices[ticker] = float(self.price_data[ticker].loc[current_dt, "Close"])

            if not current_prices:
                continue

            # Benchmark daily valuation
            benchmark.record_daily(date_str, current_prices)

            # Evaluate each persona
            for persona in self.personas:
                p = portfolios[persona]
                freq = self.eval_frequencies.get(persona, "daily")
                interval = FREQUENCY_DAYS.get(freq, 1)

                is_eval_day = (step_counters[persona] % interval == 0) or (idx == 0)
                step_counters[persona] += 1

                if is_eval_day:
                    for ticker in self.tickers:
                        price = current_prices.get(ticker)
                        if not price or price <= 0:
                            continue

                        # Build price context up to today
                        df_sub = self.price_data[ticker]
                        price_ctx = build_price_context(ticker, date_str, df_sub)

                        # Decision
                        decision = get_trading_decision(
                            persona_name=persona,
                            ticker=ticker,
                            date_str=date_str,
                            price_ctx=price_ctx,
                            fundamentals_ctx_text=self.fundamentals_text.get(ticker, ""),
                            fundamentals_data=self.fundamentals_data.get(ticker, {}),
                            dad_analysis=self.dad_analysis.get(ticker),
                            provider=self.llm_provider,
                            api_key=self.api_key,
                        )

                        action = decision.get("action", "hold")
                        conf = decision.get("confidence", 0.5)
                        reason = decision.get("reasoning", "")

                        if action == "buy":
                            # Position sizing: up to max_position_pct of total portfolio value
                            total_val = p.get_value(current_prices)
                            target_alloc = total_val * self.max_position_pct
                            current_held_val = p.holdings.get(ticker, 0) * price
                            alloc_room = max(0.0, target_alloc - current_held_val)
                            spendable_cash = min(p.cash, alloc_room if alloc_room > 0 else p.cash * 0.4)

                            shares_to_buy = int(spendable_cash // price)
                            if shares_to_buy > 0:
                                p.buy(ticker, shares_to_buy, price, trade_date=date_str, confidence=conf, reasoning=reason)
                            else:
                                p.hold(ticker, trade_date=date_str, confidence=conf, reasoning=f"Cash or allocation cap reached. {reason}")

                        elif action == "sell":
                            held_shares = p.holdings.get(ticker, 0)
                            if held_shares > 0:
                                p.sell(ticker, held_shares, price, trade_date=date_str, confidence=conf, reasoning=reason)
                            else:
                                p.hold(ticker, trade_date=date_str, confidence=conf, reasoning=f"No active position to sell. {reason}")

                        else:  # hold
                            p.hold(ticker, trade_date=date_str, confidence=conf, reasoning=reason)

                # Record daily total valuation
                p.record_daily(date_str, current_prices)

            if progress_callback and (idx % 10 == 0 or idx == total_days - 1):
                progress_callback(idx + 1, total_days)

        # Assemble summary metrics
        metrics_rows = []
        for persona in self.personas:
            p = portfolios[persona]
            m = p.get_metrics()
            m["persona"] = persona
            m["frequency"] = self.eval_frequencies.get(persona, "daily")
            metrics_rows.append(m)

        b_metrics = benchmark.get_metrics()
        b_metrics["persona"] = "Buy & Hold (Benchmark)"
        b_metrics["frequency"] = "passive"
        metrics_rows.append(b_metrics)

        metrics_df = pd.DataFrame(metrics_rows)
        metrics_df = metrics_df.sort_values("total_return_pct", ascending=False).reset_index(drop=True)

        # Assemble daily equity curves DataFrame
        equity_dict = {"date": [d.strftime("%Y-%m-%d") for d in self.common_dates]}
        for persona in self.personas:
            p = portfolios[persona]
            equity_dict[persona] = [h["total_value"] for h in p.daily_history]
        equity_dict["Buy & Hold Benchmark"] = [h["total_value"] for h in benchmark.daily_history]

        equity_df = pd.DataFrame(equity_dict)

        # Save all trade logs to logs/
        for p in portfolios.values():
            p.save_log()
        benchmark.save_log()

        return {
            "metrics_df": metrics_df,
            "equity_df": equity_df,
            "portfolios": portfolios,
            "benchmark": benchmark,
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
