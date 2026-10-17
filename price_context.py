"""
price_context.py

Calculates technical indicators and momentum/volatility context from historical price data.
Produces structured price context and formatted text prompts for technical personas
(Momentum Trader, News/Sentiment Trader, Risk-Averse, and Value Investor).
"""

import numpy as np
import pandas as pd


def compute_rsi(series: pd.Series, period: int = 14) -> float:
    """Calculate Relative Strength Index (RSI)."""
    if len(series) < period + 1:
        return 50.0
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi_series = 100 - (100 / (1 + rs))
    latest = rsi_series.iloc[-1]
    return float(latest) if pd.notna(latest) else 50.0


def build_price_context(ticker: str, current_date: str, history_df: pd.DataFrame, window_days: int = 60) -> dict:
    """
    Extract technical metrics up to `current_date` without lookahead bias.
    history_df must have a DatetimeIndex or 'Date' column and 'Close', 'Volume' columns.
    """
    df = history_df.copy()
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.set_index("Date")
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    # Filter strictly up to current_date
    current_dt = pd.to_datetime(current_date)
    sub = df[df.index <= current_dt]

    if sub.empty or len(sub) < 5:
        # Fallback if too few rows
        return {
            "ticker": ticker,
            "date": current_date,
            "current_price": float(df["Close"].iloc[-1]) if not df.empty else 0.0,
            "context_text": f"Limited price history available for {ticker} up to {current_date}.",
            "rsi_14": 50.0,
            "return_5d_pct": 0.0,
            "return_20d_pct": 0.0,
            "sma_20": 0.0,
            "sma_50": 0.0,
            "volatility_20d": 0.0,
        }

    close = sub["Close"]
    volume = sub["Volume"] if "Volume" in sub.columns else pd.Series(1, index=sub.index)

    curr_price = float(close.iloc[-1])
    ret_1d = float((close.pct_change(1).iloc[-1]) * 100) if len(close) >= 2 else 0.0
    ret_5d = float((close.pct_change(5).iloc[-1]) * 100) if len(close) >= 6 else 0.0
    ret_20d = float((close.pct_change(20).iloc[-1]) * 100) if len(close) >= 21 else 0.0

    sma_20 = float(close.rolling(20).mean().iloc[-1]) if len(close) >= 20 else curr_price
    sma_50 = float(close.rolling(50).mean().iloc[-1]) if len(close) >= 50 else sma_20

    rsi_14 = compute_rsi(close, 14)

    # Volatility (annualized 20-day rolling return std)
    daily_returns = close.pct_change().dropna()
    vol_20d = float(daily_returns.tail(20).std() * np.sqrt(252) * 100) if len(daily_returns) >= 10 else 15.0

    # Volume comparison
    avg_vol_20 = float(volume.tail(20).mean()) if len(volume) >= 5 else 1.0
    curr_vol = float(volume.iloc[-1])
    vol_ratio = (curr_vol / max(avg_vol_20, 1.0)) if avg_vol_20 > 0 else 1.0

    # 52w or available window high/low
    high_window = float(close.tail(252).max())
    low_window = float(close.tail(252).min())
    pct_from_high = float((curr_price - high_window) / high_window * 100)
    range_pos = float((curr_price - low_window) / max((high_window - low_window), 1e-6) * 100)

    # Trend categorization
    if curr_price > sma_20 > sma_50:
        trend = "Strong Bullish Uptrend (Price > SMA20 > SMA50)"
    elif curr_price > sma_20:
        trend = "Mild Uptrend (Price > SMA20)"
    elif curr_price < sma_20 < sma_50:
        trend = "Strong Bearish Downtrend (Price < SMA20 < SMA50)"
    else:
        trend = "Consolidation / Mixed"

    # Context text for prompts
    context_text = f"""=== TECHNICAL PRICE CONTEXT FOR {ticker} (as of {current_date}) ===
Current Price: ₹{curr_price:,.2f}
Price Trend: {trend}
• 1-Day Return: {ret_1d:+.2f}% | 5-Day Return: {ret_5d:+.2f}% | 20-Day Return: {ret_20d:+.2f}%
• Moving Averages: SMA-20 = ₹{sma_20:,.2f} | SMA-50 = ₹{sma_50:,.2f}
• Relative Strength Index (RSI 14): {rsi_14:.1f} ({'Overbought >70' if rsi_14 > 70 else 'Oversold <30' if rsi_14 < 30 else 'Neutral/Momentum range'})
• Annualized 20-Day Volatility: {vol_20d:.1f}%
• Volume: {curr_vol:,.0f} ({vol_ratio:.2f}x 20-day average)
• Window Range: Low = ₹{low_window:,.2f} | High = ₹{high_window:,.2f} (Current is {pct_from_high:+.1f}% from peak)"""

    return {
        "ticker": ticker,
        "date": current_date,
        "current_price": curr_price,
        "ret_1d": ret_1d,
        "ret_5d": ret_5d,
        "ret_20d": ret_20d,
        "sma_20": sma_20,
        "sma_50": sma_50,
        "rsi_14": rsi_14,
        "volatility_20d": vol_20d,
        "volume_ratio": vol_ratio,
        "high_window": high_window,
        "low_window": low_window,
        "pct_from_high": pct_from_high,
        "range_pos": range_pos,
        "trend": trend,
        "context_text": context_text,
    }
