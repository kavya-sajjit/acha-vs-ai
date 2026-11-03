.PHONY: install test app results

install:
	pip install -r requirements.txt

test:
	pytest -q tests/

app:
	streamlit run app.py

# Regenerates the leaderboard and trade logs with the offline rule engine.
# Fixed dates, so the numbers are the same every time (prices are cached in data/).
TICKERS = RELIANCE.NS,TCS.NS,INFY.NS,HDFCBANK.NS,ITC.NS
results:
	python run_backtest.py --tickers $(TICKERS) --start 2024-04-01 --end 2026-04-01 --dad-frequency quarterly
