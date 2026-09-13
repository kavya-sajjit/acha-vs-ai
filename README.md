# Dad vs AI Trader

Paper-trading experiment: give an LLM a fake ₹1,00,000 (or whatever amount you pick)
and a "great investor" persona prompt, let it trade NSE stocks on historical data,
and compare how it does against your dad's actual trading rules.




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
