"""
run_backtest.py

Command-line interface to run the Dad vs AI Trader simulation experiment.

Examples:
    python run_backtest.py
    python run_backtest.py --tickers RELIANCE.NS,TCS.NS,INFY.NS,HDFCBANK.NS --period 1y
    python run_backtest.py --personas dad_hybrid,momentum_trader --dad-frequency monthly
    python run_backtest.py --provider gemini --api-key YOUR_GEMINI_KEY
"""

import argparse
import sys
from backtester import Backtester


def parse_args():
    parser = argparse.ArgumentParser(description="Dad vs AI Trader - Historical Paper-Trading Backtester")
    parser.add_argument(
        "--tickers",
        type=str,
        default="RELIANCE.NS,TCS.NS,INFY.NS",
        help="Comma-separated list of NSE tickers (e.g. RELIANCE.NS,TCS.NS,INFY.NS)",
    )
    parser.add_argument(
        "--period",
        type=str,
        default="1y",
        choices=["3mo", "6mo", "1y", "2y", "3y", "5y"],
        help="Historical period to backtest over",
    )
    parser.add_argument(
        "--cash",
        type=float,
        default=100000.0,
        help="Starting cash in INR (default: 100,000)",
    )
    parser.add_argument(
        "--personas",
        type=str,
        default="dad_hybrid,momentum_trader,value_investor,risk_averse",
        help="Comma-separated list of personas to evaluate",
    )
    parser.add_argument(
        "--dad-frequency",
        type=str,
        default="quarterly",
        choices=["daily", "weekly", "monthly", "quarterly", "yearly"],
        help="Evaluation frequency for Dad's hybrid persona (default: quarterly)",
    )
    parser.add_argument(
        "--provider",
        type=str,
        default="mock",
        choices=["mock", "gemini", "openai"],
        help="LLM execution engine: 'mock' (free/instant rule engine), 'gemini', or 'openai'",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="API key for Gemini or OpenAI (optional)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    personas = [p.strip() for p in args.personas.split(",") if p.strip()]

    print("=" * 65)
    print("           ⚔️  DAD VS AI TRADER EXPERIMENT  ⚔️")
    print("=" * 65)
    print(f"Tickers:            {', '.join(tickers)}")
    print(f"Period:             {args.period}")
    print(f"Starting Cash:      ₹{args.cash:,.2f}")
    print(f"Personas:           {', '.join(personas)}")
    print(f"Dad's Frequency:    {args.dad_frequency}")
    print(f"Execution Engine:   {args.provider.upper()}")
    print("=" * 65)

    freqs = {"dad_hybrid": args.dad_frequency}
    bt = Backtester(
        tickers=tickers,
        period=args.period,
        initial_cash=args.cash,
        personas=personas,
        eval_frequencies=freqs,
        llm_provider=args.provider,
        api_key=args.api_key,
    )

    try:
        results = bt.run()
    except Exception as e:
        print(f"\n❌ Error during backtest: {e}")
        sys.exit(1)

    print("\n" + "=" * 65)
    print("                     🏆 FINAL LEADERBOARD 🏆")
    print("=" * 65)
    df_metrics = results["metrics_df"]
    display_cols = ["persona", "frequency", "total_return_pct", "cagr_pct", "sharpe_ratio", "max_drawdown_pct", "total_trades"]
    print(df_metrics[display_cols].to_string(index=False))
    print("=" * 65)

    print("\nTrade logs and decision trails have been written to logs/:")
    for persona in personas:
        print(f"  • logs/{persona}_trades.csv")
    print("  • logs/buy_and_hold_benchmark_trades.csv")
    print("\nRun the interactive web dashboard with: streamlit run app.py\n")


if __name__ == "__main__":
    main()
