"""
fetch_data.py

Pulls historical daily price data for NSE-listed stocks and saves each
one to a CSV in data/. NSE tickers need the ".NS" suffix for yfinance
(e.g. RELIANCE.NS, TCS.NS, INFY.NS) - that's the NSE-specific bit.

Run directly to test:
    python fetch_data.py
"""

import os
import yfinance as yf

# Edit this list to whichever NSE stocks you and your dad want to test with.
# Keep it small (3-5 stocks) while you're still building - easier to debug.
TICKERS = ["RELIANCE.NS", "TCS.NS", "INFY.NS"]

# How far back to pull data. "3mo" is enough to start backtesting with.
# Other valid periods: "1mo", "6mo", "1y", "2y", "5y", "max"
PERIOD = "3mo"

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def fetch_and_save(ticker: str, period: str = PERIOD) -> "pandas.DataFrame":
    """Fetch daily price history for one ticker and save it as a CSV."""
    stock = yf.Ticker(ticker)
    history = stock.history(period=period)

    if history.empty:
        print(f"  ! No data returned for {ticker} - check the ticker symbol.")
        return history

    os.makedirs(DATA_DIR, exist_ok=True)
    out_path = os.path.join(DATA_DIR, f"{ticker.replace('.', '_')}.csv")
    history.to_csv(out_path)
    print(f"  saved {len(history)} rows for {ticker} -> {out_path}")
    return history


def fetch_all(tickers=None, period: str = PERIOD):
    """Fetch and save price history for every ticker in the list."""
    tickers = tickers or TICKERS
    results = {}
    for ticker in tickers:
        print(f"Fetching {ticker}...")
        results[ticker] = fetch_and_save(ticker, period)
    return results


if __name__ == "__main__":
    print(f"Fetching {PERIOD} of daily price history for: {', '.join(TICKERS)}")
    fetch_all()
    print("\nDone. Check the data/ folder for CSVs.")
    print("If a ticker came back empty, double check it exists on NSE and")
    print("has the .NS suffix.")
