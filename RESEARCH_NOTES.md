# Research notes: look-ahead leakage

A backtest is only meaningful if, on each simulated date, the strategy sees **only what was knowable on that date**. Reading the code, I found three places where the future leaks in. Fixing them, and measuring how much each one inflates returns, is the research contribution of this project.

## Leak 1: fundamentals came from today (fixed)

`fundamentals.fetch_fundamentals_data()` downloads the **current** balance sheet, income statement, cash flow and `info` (including today's P/E) from yfinance, once. `Backtester.prepare_data()` then uses that same snapshot for **every** simulated date.

So on a simulated date in 2024, Dad's rules already see the 2025 annual report. Rules 4, 9, 10 and 13 (debt trend, 5-year doubling, reserves growth) are exactly the rules that benefit most from knowing the next report.

**Fix (done, `fundamentals.fundamentals_as_of`):** each statement value is cached with its period-end date. On simulated date `d`, only statements with period end + 60 days ≤ `d` are visible (SEBI LODR gives listed companies up to 60 days after year end for audited annual results). P/E comes from the price on `d` and the last *reported* EPS; the dividend yield from dividends paid in the 12 months before `d`; market cap and forward P/E (today-only numbers) are dropped. `tests/test_no_lookahead.py` checks that a fake future report can't change a past decision.

Run the backtest once after pulling this: old caches in `data/fundamentals/` have no dates, so they're refetched automatically.

Note: yfinance only returns about the last 4 annual statements, so Rule 3 ("study 15–20 years") cannot be checked honestly with yfinance alone. Either state that limit in the README or switch to a source with longer history (company annual reports, Screener.in exports).

## Leak 2: trading at the close that produced the signal (fixed)

Decisions used day `t`'s close and filled at that same close, but you only see the close after the market shuts. Now orders fill at day `t+1`'s open, with costs of 15 bps per side by default (`--cost-bps`; brokerage + STT + exchange charges + slippage for NSE delivery is roughly 0.1–0.2%). No decisions are made on the last day, since they could never fill. Tested in `tests/test_no_lookahead.py`.

## Leak 3: the LLM may remember the future (the headline experiment)

An LLM asked "should I buy TCS on 2024-03-15?" may already know from its training data what TCS did after that date. Two tests, then compare:

| Condition | How |
|---|---|
| **A. Normal** | Real ticker, real date, real prices |
| **B. Post-cutoff** | Only dates after each model's training cutoff. The model can't remember those |
| **C. Anonymised** | Ticker → "Company 7", dates → "Day 1, Day 2...", prices rescaled to start at 100. Same numbers, nothing to recognise |

If the LLM's edge in A disappears in B and C, the "AI beats Dad" result was memory, not reasoning. That finding is publishable-workshop material either way.

Requirements for this to be trustworthy:
- Every call cached in `logs/llm_calls.jsonl` (done: `llm_cache.py`), so all three conditions are re-runnable for free and auditable.
- The same prompts and the same dates across conditions, so the only difference is what the model can recognise.
- Metrics with confidence intervals (`metrics.bootstrap_ci`), so a difference isn't noise.
- Benchmark: Nifty 50 buy-and-hold (`^NSEI`), not only equal-weight buy-and-hold of the chosen stocks.

## Order of work

1. ~~Fixed `--start/--end` dates~~ (done; `make results` uses 2024-04-01 to 2026-04-01).
2. ~~Leak 1~~ and ~~Leak 2~~ (done, with tests).
3. ~~Nifty 50 benchmark~~ (done: `^NSEI` buy-and-hold in the leaderboard and chart).
4. `metrics.py`: fill in the TODOs (tests in `tests/test_metrics.py`), then use them for the leaderboard instead of `Portfolio.get_metrics`, and add bootstrap CIs.
5. Walk-forward windows: run the same strategies on several consecutive 1-year windows and report the spread.
6. Leak 3 experiment, A vs B vs C, then a Results section with plots.

A known limit to state in the write-up: yfinance returns only about the last 4 annual reports, so for a 2024 start date Dad's rules see 1–2 years of history. Rule 3 ("study 15–20 years") can't be tested honestly without a longer source (annual reports or Screener.in exports).
