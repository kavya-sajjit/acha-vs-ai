"""
portfolio.py

A fake ("paper") trading portfolio. Tracks cash, holdings, daily valuation history,
and a log of every trade with decision rationale.
"""

import csv
import os
from datetime import date
import numpy as np
import pandas as pd


class Portfolio:
    def __init__(self, starting_cash: float, name: str = "portfolio"):
        self.cash = float(starting_cash)
        self.starting_cash = float(starting_cash)
        self.name = name
        # holdings: {ticker: number_of_shares}
        self.holdings = {}
        # every action taken, for later analysis
        self.trade_log = []
        # daily equity tracking: list of {"date": str, "cash": float, "holdings_value": float, "total_value": float}
        self.daily_history = []

    def buy(self, ticker: str, shares: float, price: float, trade_date=None, confidence=1.0, reasoning=""):
        """Buy `shares` of `ticker` at `price` each, if there's enough cash."""
        cost = shares * price
        if cost > self.cash:
            print(f"  ! Not enough cash to buy {shares} of {ticker} "
                  f"(need {cost:.2f}, have {self.cash:.2f}) - skipping.")
            return False

        self.cash -= cost
        self.holdings[ticker] = self.holdings.get(ticker, 0) + shares
        self._log("buy", ticker, shares, price, trade_date, confidence, reasoning)
        return True

    def sell(self, ticker: str, shares: float, price: float, trade_date=None, confidence=1.0, reasoning=""):
        """Sell `shares` of `ticker` at `price` each, if you hold enough."""
        held = self.holdings.get(ticker, 0)
        if shares > held:
            print(f"  ! Can't sell {shares} of {ticker}, only holding {held} - skipping.")
            return False

        self.cash += shares * price
        self.holdings[ticker] = held - shares
        if self.holdings[ticker] == 0:
            del self.holdings[ticker]
        self._log("sell", ticker, shares, price, trade_date, confidence, reasoning)
        return True

    def hold(self, ticker: str, trade_date=None, confidence=1.0, reasoning=""):
        """Log a deliberate 'do nothing' decision - useful for the AI's reasoning trail."""
        self._log("hold", ticker, 0, 0, trade_date, confidence, reasoning)

    def get_value(self, current_prices: dict) -> float:
        """
        Total portfolio value = cash + (shares held * current price) for
        every ticker. `current_prices` is {ticker: price}.
        """
        holdings_value = sum(
            shares * current_prices.get(ticker, 0.0)
            for ticker, shares in self.holdings.items()
        )
        return self.cash + holdings_value

    def record_daily(self, trade_date: str, current_prices: dict) -> float:
        """Record the total portfolio valuation for an active trading day."""
        holdings_value = sum(
            shares * current_prices.get(ticker, 0.0)
            for ticker, shares in self.holdings.items()
        )
        total_val = self.cash + holdings_value
        self.daily_history.append({
            "date": trade_date,
            "cash": round(self.cash, 2),
            "holdings_value": round(holdings_value, 2),
            "total_value": round(total_val, 2),
        })
        return total_val

    def _log(self, action, ticker, shares, price, trade_date, confidence=1.0, reasoning=""):
        self.trade_log.append({
            "date": trade_date or date.today().isoformat(),
            "action": action,
            "ticker": ticker,
            "shares": round(shares, 4) if isinstance(shares, (int, float)) else shares,
            "price": round(price, 2) if isinstance(price, (int, float)) else price,
            "cash_after": round(self.cash, 2),
            "confidence": round(confidence, 2) if isinstance(confidence, (int, float)) else confidence,
            "reasoning": reasoning or "",
        })

    def get_metrics(self, risk_free_rate: float = 0.07) -> dict:
        """Calculate quantitative performance metrics."""
        if not self.daily_history:
            return {
                "initial_cash": self.starting_cash,
                "final_value": self.cash,
                "total_return_pct": 0.0,
                "cagr_pct": 0.0,
                "sharpe_ratio": 0.0,
                "max_drawdown_pct": 0.0,
                "total_trades": len(self.trade_log),
                "buys": sum(1 for t in self.trade_log if t["action"] == "buy"),
                "sells": sum(1 for t in self.trade_log if t["action"] == "sell"),
            }

        df_hist = pd.DataFrame(self.daily_history)
        df_hist["date"] = pd.to_datetime(df_hist["date"])
        df_hist = df_hist.sort_values("date")

        final_val = df_hist["total_value"].iloc[-1]
        total_ret = ((final_val - self.starting_cash) / self.starting_cash) * 100

        # Duration & CAGR
        days = max((df_hist["date"].iloc[-1] - df_hist["date"].iloc[0]).days, 1)
        years = days / 365.25
        cagr = (((final_val / self.starting_cash) ** (1 / max(years, 0.05))) - 1) * 100 if years > 0 else total_ret

        # Daily returns & Sharpe
        daily_ret = df_hist["total_value"].pct_change().dropna()
        if len(daily_ret) > 1 and daily_ret.std() > 0:
            excess_return = daily_ret.mean() - (risk_free_rate / 252)
            sharpe = (excess_return / daily_ret.std()) * np.sqrt(252)
        else:
            sharpe = 0.0

        # Max Drawdown
        cum_max = df_hist["total_value"].cummax()
        drawdowns = (df_hist["total_value"] - cum_max) / cum_max * 100
        max_dd = abs(float(drawdowns.min())) if not drawdowns.empty else 0.0

        buys = sum(1 for t in self.trade_log if t["action"] == "buy")
        sells = sum(1 for t in self.trade_log if t["action"] == "sell")
        holds = sum(1 for t in self.trade_log if t["action"] == "hold")

        return {
            "initial_cash": self.starting_cash,
            "final_value": round(final_val, 2),
            "net_profit": round(final_val - self.starting_cash, 2),
            "total_return_pct": round(total_ret, 2),
            "cagr_pct": round(cagr, 2),
            "sharpe_ratio": round(float(sharpe), 2),
            "max_drawdown_pct": round(max_dd, 2),
            "total_trades": buys + sells,
            "buys": buys,
            "sells": sells,
            "holds": holds,
            "days_traded": len(df_hist),
        }

    def save_log(self, logs_dir=None):
        """Write the full trade log to a CSV in logs/, named after this portfolio."""
        logs_dir = logs_dir or os.path.join(os.path.dirname(__file__), "logs")
        os.makedirs(logs_dir, exist_ok=True)
        out_path = os.path.join(logs_dir, f"{self.name}_trades.csv")

        with open(out_path, "w", newline="", encoding="utf-8") as f:
            fieldnames = ["date", "action", "ticker", "shares", "price", "cash_after", "confidence", "reasoning"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.trade_log)

        print(f"  saved {len(self.trade_log)} trade rows -> {out_path}")
        return out_path
