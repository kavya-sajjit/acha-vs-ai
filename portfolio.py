"""
portfolio.py

A fake ("paper") trading portfolio. Tracks cash, holdings, and a log of
every trade. No real money, no network calls - pure bookkeeping, which is
exactly why this should be the first thing you get right before any LLM
touches the project.

Basic usage:
    p = Portfolio(starting_cash=1000, name="value_investor")
    p.buy("RELIANCE.NS", shares=2, price=2900)
    p.sell("RELIANCE.NS", shares=1, price=2950)
    print(p.get_value({"RELIANCE.NS": 2960}))
    p.save_log()
"""

import csv
import os
from datetime import date


class Portfolio:
    def __init__(self, starting_cash: float, name: str = "portfolio"):
        self.cash = starting_cash
        self.starting_cash = starting_cash
        self.name = name
        # holdings: {ticker: number_of_shares}
        self.holdings = {}
        # every action taken, for later analysis
        self.trade_log = []

    def buy(self, ticker: str, shares: float, price: float, trade_date=None):
        """Buy `shares` of `ticker` at `price` each, if there's enough cash."""
        cost = shares * price
        if cost > self.cash:
            print(f"  ! Not enough cash to buy {shares} of {ticker} "
                  f"(need {cost:.2f}, have {self.cash:.2f}) - skipping.")
            return False

        self.cash -= cost
        self.holdings[ticker] = self.holdings.get(ticker, 0) + shares
        self._log("buy", ticker, shares, price, trade_date)
        return True

    def sell(self, ticker: str, shares: float, price: float, trade_date=None):
        """Sell `shares` of `ticker` at `price` each, if you hold enough."""
        held = self.holdings.get(ticker, 0)
        if shares > held:
            print(f"  ! Can't sell {shares} of {ticker}, only holding {held} - skipping.")
            return False

        self.cash += shares * price
        self.holdings[ticker] = held - shares
        self._log("sell", ticker, shares, price, trade_date)
        return True

    def hold(self, ticker: str, trade_date=None):
        """Log a deliberate 'do nothing' decision - useful for the AI's reasoning trail."""
        self._log("hold", ticker, 0, 0, trade_date)

    def get_value(self, current_prices: dict) -> float:
        """
        Total portfolio value = cash + (shares held * current price) for
        every ticker. `current_prices` is {ticker: price}.
        """
        holdings_value = sum(
            shares * current_prices.get(ticker, 0)
            for ticker, shares in self.holdings.items()
        )
        return self.cash + holdings_value

    def _log(self, action, ticker, shares, price, trade_date):
        self.trade_log.append({
            "date": trade_date or date.today().isoformat(),
            "action": action,
            "ticker": ticker,
            "shares": shares,
            "price": price,
            "cash_after": round(self.cash, 2),
        })

    def save_log(self, logs_dir=None):
        """Write the full trade log to a CSV in logs/, named after this portfolio."""
        logs_dir = logs_dir or os.path.join(os.path.dirname(__file__), "logs")
        os.makedirs(logs_dir, exist_ok=True)
        out_path = os.path.join(logs_dir, f"{self.name}_trades.csv")

        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=["date", "action", "ticker", "shares", "price", "cash_after"]
            )
            writer.writeheader()
            writer.writerows(self.trade_log)

        print(f"  saved {len(self.trade_log)} trade rows -> {out_path}")
        return out_path
