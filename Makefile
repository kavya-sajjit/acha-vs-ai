.PHONY: install test app results

install:
	pip install -r requirements.txt

test:
	pytest -q tests/

app:
	streamlit run app.py

# Regenerates the leaderboard and trade logs with the offline rule engine.
# TODO(kavya): once the backtest takes fixed --start/--end dates instead of a
# relative --period, pin them here so `make results` gives the same numbers every time.
results:
	python run_backtest.py --tickers RELIANCE.NS,TCS.NS,INFY.NS,HDFCBANK.NS,ITC.NS --period 2y --dad-frequency quarterly
