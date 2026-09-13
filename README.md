# Dad vs AI Trader

Paper-trading experiment: give an LLM a fake ₹1,00,000 (or whatever amount you pick)
and a "great investor" persona prompt, let it trade NSE stocks on historical data,
and compare how it does against your dad's actual trading rules.

No real money is ever used. This only simulates trades against historical price data.

## Project structure

```
dad-vs-ai-trader/
├── fetch_data.py       # pulls NSE stock price history via yfinance
├── portfolio.py        # the fake portfolio tracker (buy/sell/value math)
├── test_portfolio.py   # quick sanity test for portfolio.py, no network needed
├── prompts/            # one persona prompt per file
│   ├── value_investor.txt
│   ├── momentum_trader.txt
│   ├── news_sentiment.txt
│   ├── risk_averse.txt
│   └── dad_hybrid.txt      # <- fill this in after talking to dad
├── data/                # saved price history CSVs land here
├── logs/                # trade logs (one CSV per persona run) land here
├── requirements.txt
└── .gitignore
```

## Setup

```bash
pip install -r requirements.txt
```

## Step 1 — check the data pipeline works

```bash
python fetch_data.py
```

This pulls the last 3 months of daily prices for a few sample NSE stocks
(RELIANCE.NS, TCS.NS, INFY.NS) and saves them to `data/`. Edit the `TICKERS`
list at the top of the file to change which stocks you're testing with.

Note: NSE tickers need the `.NS` suffix for yfinance, e.g. `RELIANCE.NS` not `RELIANCE`.

## Step 2 — check the portfolio tracker works

```bash
python test_portfolio.py
```

This runs a few hardcoded fake trades through `portfolio.py` and prints the
running portfolio value, with no AI and no network call — just checking the
math (buy/sell/cash tracking) is correct before anything else touches it.

## Step 3 — next (not built yet)

Once steps 1-2 both run cleanly:
- Wire up an LLM call (OpenAI or Gemini API) that reads a persona prompt from
  `prompts/`, gets fed a slice of price history from `data/`, and returns a
  structured `{"action": "buy/sell/hold", "confidence": 0-1, "reasoning": "..."}`
  decision.
- Feed that decision into `portfolio.py`'s buy/sell functions, one simulated
  day at a time, over the historical window.
- Log every decision + resulting portfolio value to `logs/`.
- Repeat for each persona, then plot portfolio value over time for all of
  them on one chart.

## Talking to dad

Fill in `prompts/dad_hybrid.txt` with his actual thumb rules once you've
talked to him — write them as direct instructions, not vague vibes, e.g.:

> "Never buy on the day of an earnings announcement. Prefer stocks trading
> below their 50-day moving average. Sell if a position drops more than 8%
> from purchase price."

The more specific and rule-like, the better the LLM can actually follow it.
