# ⚔️ Acha vs AI Trader (Dad vs AI)

> **Paper-trading experiment:** Give an LLM a paper-trading account (e.g. ₹1,00,000) and pit an experienced Indian father's **14 fundamental investment thumb rules** against algorithmic AI personas (Momentum, Value, Sentiment, Risk-Averse) trading real National Stock Exchange (NSE) historical data.

*No real money is ever used. This simulates trades against historical price and financial statement data.*

---

## 🌟 Key Features

- **Dad's 14 Fundamental Rules Engine:** Audits balance sheets, income statements, and cash flows over 5–20 years using `yfinance` (`.balance_sheet`, `.income_stmt`, `.cashflow`, `.dividends`).
- **Frequency-Aware Backtesting:** Respects the fundamental difference in investment horizons — Dad evaluates quarterly (or monthly/yearly) to hold long-term compounders, while momentum traders trade on daily price action.
- **Dual Execution Engines:**
  - **Instant Rule Engine (Default):** Runs 100% offline, free, deterministic, and instant with zero API keys required.
  - **Live Frontier LLMs:** Plug in Google Gemini (`GEMINI_API_KEY`) or OpenAI (`OPENAI_API_KEY`) to let LLMs reason over prompts and financial context.
- **Institutional-Grade Performance Metrics:** Total Return %, CAGR %, Sharpe Ratio, Maximum Drawdown %, Win Rate, and Trade Counts compared directly against an Equal-Weight Buy & Hold benchmark.
- **Interactive Web Terminal & Dashboard:** Built with Streamlit and Plotly for visual comparison, deep-dive rule scorecards, and searchable trade logs with full reasoning trails.

---

## 📜 Dad's 14 Investment Thumb Rules

Dad's rules focus on disciplined, multi-year compounding rather than short-term price momentum:

| # | Category | Rule & Rationale |
|---|---|---|
| **1 & 2** | **P/E Screen & Moat** | Prefer low P/E stocks; exception granted for clear industry leaders (e.g., Oracle/TCS/Reliance). |
| **3** | **Statement Depth** | Study 15–20 years of financial history, not just 1–2 recent quarters. |
| **4** | **Debt Deleveraging** | Long-term borrowings must trend downward over time (acceptable if tied to productive capex). |
| **5 & 6** | **Short-Term Debt** | Short-term borrowings must remain **less than working capital** and grow in line with sales growth. |
| **7** | **Margins** | Operating and net margins must rank among top-tier peers for the industry. |
| **8 & 12**| **Capex Coherence** | Expansion capex is positive *only* if it visibly translates into sales growth in subsequent years. |
| **9 & 10**| **5-Yr Doubling** | Revenues and net profits must at minimum double (or 3x–4x) every 5 years; share price should track. |
| **11**| **Other Income** | Treasury & investments must generate productive returns (reflecting ~3–4%+ return on investment value). |
| **13**| **Reserves Growth** | Reserves & retained earnings must consistently increase year-over-year. |
| **14**| **Dividend Track Record** | Consistent, uninterrupted dividend history with healthy payouts; low yield is a red flag. |

---

## 🏗️ Project Architecture

```
dad-vs-ai-trader/
├── app.py                  # Streamlit web application & interactive dashboard
├── backtester.py           # Multi-frequency simulation engine & benchmark comparison
├── fundamentals.py         # Balance sheet/P&L parser & Dad's 14-rule scoring algorithm
├── price_context.py        # Technical indicator generator (SMA20/50, RSI, Volatility)
├── llm_decision.py         # Decision dispatcher (Mock Rule Engine, Gemini, OpenAI)
├── portfolio.py            # Bookkeeping portfolio tracker (cash, holdings, daily history)
├── fetch_data.py           # NSE historical daily price puller via yfinance
├── run_backtest.py         # CLI entry point with configurable arguments
├── test_portfolio.py       # Deterministic sanity tests for portfolio math
├── prompts/                # Persona system prompts
│   ├── dad_hybrid.txt      # Dad's 14 exact investment rules
│   ├── momentum_trader.txt # Trend following & RSI breakout strategy
│   ├── value_investor.txt  # Benjamin Graham / Warren Buffett value strategy
│   ├── risk_averse.txt     # Low volatility & capital preservation strategy
│   └── news_sentiment.txt  # Headline and price shock momentum strategy
├── data/                   # Cached price histories & fundamental statement JSONs
├── logs/                   # Exported trade logs & decision rationale CSVs
└── requirements.txt
```

---

## 🚀 Quickstart

### 1. Installation
Clone the repository and install dependencies:
```bash
git clone https://github.com/kavya-sajjit/acha-vs-ai.git
cd acha-vs-ai

# Activate your virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Sanity Checks
Verify the portfolio bookkeeping logic:
```bash
python test_portfolio.py
```

### 3. Launch the Web Application
```bash
streamlit run app.py
```
Open `http://localhost:8501` in your browser to inspect the interactive arena!

### 4. Run CLI Backtests
```bash
# Basic 6-month simulation on default bluechips
python run_backtest.py --tickers RELIANCE.NS,TCS.NS,INFY.NS --period 6mo

# 1-year backtest with monthly Dad evaluation
python run_backtest.py --period 1y --dad-frequency monthly

# Using Gemini API
python run_backtest.py --provider gemini --api-key YOUR_GEMINI_API_KEY
```

---

## 🌐 Deploying & Hosting Online (Free in 2 Minutes)

You can easily host this dashboard for free so anyone can use it online:

### Streamlit Community Cloud (Recommended)
1. Go to **[share.streamlit.io](https://share.streamlit.io)** and sign in with GitHub.
2. Click **"New app"**.
3. Select repository: `kavya-sajjit/acha-vs-ai`.
4. Set Main file path: `app.py`.
5. Click **"Deploy"**!
6. *(Optional)* If using Gemini/OpenAI, add `GEMINI_API_KEY` under **App Settings → Secrets**.

---

## ⚖️ Disclaimer
*This project is built purely for academic, research, and entertainment paper-trading purposes. None of the outputs constitute financial or investment advice.*
