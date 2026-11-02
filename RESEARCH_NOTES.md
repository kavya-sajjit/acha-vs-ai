# Research notes: look-ahead leakage

A backtest is only meaningful if, on each simulated date, the strategy sees **only what was knowable on that date**. Reading the code, I found three places where the future leaks in. Fixing them, and measuring how much each one inflates returns, is the research contribution of this project.

## Leak 1: fundamentals come from today (found in the code, needs fixing first)

`fundamentals.fetch_fundamentals_data()` downloads the **current** balance sheet, income statement, cash flow and `info` (including today's P/E) from yfinance, once. `Backtester.prepare_data()` then uses that same snapshot for **every** simulated date.

So on a simulated date in 2024, Dad's rules already see the 2025 annual report. Rules 4, 9, 10 and 13 (debt trend, 5-year doubling, reserves growth) are exactly the rules that benefit most from knowing the next report.

**Fix (point-in-time fundamentals):**
1. Keep each statement column with its period-end date.
2. On simulated date `d`, use only statements whose period end + a reporting lag (≈60 days for annual results in India; SEBI LODR allows up to 60 days after year end) is ≤ `d`.
3. Compute P/E from the price on `d` and the last *reported* EPS, not `info["trailingPE"]`.
4. Write a test: a fake statement dated after `d` must never change the decision on `d`.

Note: yfinance only returns about the last 4 annual statements, so Rule 3 ("study 15–20 years") cannot be checked honestly with yfinance alone. Either state that limit in the README or switch to a source with longer history (company annual reports, Screener.in exports).

## Leak 2: trading at the close that produced the signal

Decisions use day `t`'s close, and trades fill at that same close. In reality you see the close only after the market shuts. Fill at day `t+1`'s open (or close), and add brokerage + STT + slippage (≈0.1–0.2% per side for Indian equity delivery).

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

1. `metrics.py`: fill in the TODOs (tests in `tests/test_metrics.py`).
2. Fixed `--start/--end` dates instead of `--period`, so results don't move every day.
3. Leak 1 (point-in-time fundamentals) + its test.
4. Leak 2 (next-day fills + costs).
5. Nifty 50 benchmark + walk-forward windows.
6. Leak 3 experiment, A vs B vs C, then a Results section with plots.
