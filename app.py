"""
app.py

Acha vs AI Trader — Interactive Streamlit Web Application
Compare your Dad's 14 fundamental investment thumb rules against AI algorithmic trading personas
on historical NSE stock data.
"""

import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from backtester import Backtester, DEFAULT_FREQUENCIES
from fundamentals import analyze_dad_rules, fetch_fundamentals_data, format_inr

# Page Configuration
st.set_page_config(
    page_title="Acha vs AI Trader | Dad's 14 Rules vs Algorithmic Personas",
    page_icon="⚔️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom High-End Financial Dashboard CSS
st.markdown(
    """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    .main {
        background-color: #0b0f19;
    }

    /* Metric Cards */
    .metric-card {
        background: linear-gradient(135deg, rgba(26, 34, 52, 0.75) 0%, rgba(17, 24, 39, 0.95) 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 1.25rem 1.4rem;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
        backdrop-filter: blur(10px);
        margin-bottom: 1rem;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        border-color: rgba(56, 189, 248, 0.4);
    }
    .metric-label {
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #94a3b8;
        font-weight: 600;
        margin-bottom: 0.4rem;
    }
    .metric-value {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.75rem;
        font-weight: 700;
        color: #f8fafc;
    }
    .metric-delta-positive {
        color: #34d399;
        font-size: 0.85rem;
        font-weight: 600;
    }
    .metric-delta-negative {
        color: #f87171;
        font-size: 0.85rem;
        font-weight: 600;
    }

    /* Status Badges */
    .badge-pass {
        background-color: rgba(16, 185, 129, 0.15);
        color: #34d399;
        border: 1px solid rgba(52, 211, 153, 0.3);
        padding: 0.25rem 0.6rem;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.75rem;
        display: inline-block;
    }
    .badge-caution {
        background-color: rgba(245, 158, 11, 0.15);
        color: #fbbf24;
        border: 1px solid rgba(251, 191, 36, 0.3);
        padding: 0.25rem 0.6rem;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.75rem;
        display: inline-block;
    }
    .badge-fail {
        background-color: rgba(239, 68, 68, 0.15);
        color: #f87171;
        border: 1px solid rgba(248, 113, 113, 0.3);
        padding: 0.25rem 0.6rem;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.75rem;
        display: inline-block;
    }

    /* Rule row card */
    .rule-card {
        background: rgba(15, 23, 42, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 8px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.6rem;
    }
</style>
""",
    unsafe_allow_html=True,
)

# Header Section
st.markdown(
    """
<div style="padding: 1.5rem 0 1rem 0;">
    <div style="display: flex; align-items: center; gap: 0.75rem; margin-bottom: 0.5rem;">
        <span style="font-size: 2.2rem;">⚔️</span>
        <h1 style="margin: 0; font-size: 2.3rem; font-weight: 800; background: linear-gradient(90deg, #38bdf8, #818cf8, #c084fc); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
            Acha vs AI Trader
        </h1>
        <span style="background: rgba(56, 189, 248, 0.15); color: #38bdf8; border: 1px solid rgba(56, 189, 248, 0.3); padding: 0.25rem 0.6rem; border-radius: 6px; font-size: 0.75rem; font-weight: 700;">
            NSE PAPER TRADING
        </span>
    </div>
    <p style="color: #94a3b8; font-size: 1.05rem; margin: 0; max-width: 900px;">
        Can an experienced Indian father's <strong>14 thumb rules</strong> for long-term fundamental stock picking beat algorithmic AI momentum, value, and sentiment traders on real National Stock Exchange (NSE) historical data?
    </p>
</div>
""",
    unsafe_allow_html=True,
)

# Sidebar Configuration
st.sidebar.markdown("### ⚙️ Simulation Controls")

# Ticker Selection
DEFAULT_NSE_TICKERS = ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS", "TATAMOTORS.NS"]
selected_tickers = st.sidebar.multiselect(
    "Select NSE Tickers",
    options=DEFAULT_NSE_TICKERS,
    default=["RELIANCE.NS", "TCS.NS", "INFY.NS"],
    help="Tickers traded on National Stock Exchange (NSE) requiring the .NS suffix",
)

custom_ticker = st.sidebar.text_input("Add Custom NSE Ticker", placeholder="e.g. LT.NS, ITC.NS, BHARTIARTL.NS")
if custom_ticker:
    clean_t = custom_ticker.strip().upper()
    if not clean_t.endswith(".NS"):
        clean_t += ".NS"
    if clean_t not in selected_tickers:
        selected_tickers.append(clean_t)

period = st.sidebar.selectbox(
    "Historical Backtest Period",
    options=["3mo", "6mo", "1y", "2y", "3y", "5y"],
    index=2,
    help="How far back to simulate trading",
)

initial_cash = st.sidebar.number_input(
    "Starting Capital (₹ INR)",
    min_value=10000,
    max_value=10000000,
    value=100000,
    step=25000,
    format="%d",
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🎭 Personas in the Battle")

include_dad = st.sidebar.checkbox("👑 Dad's Hybrid Rules (Acha)", value=True)
include_momentum = st.sidebar.checkbox("⚡ Momentum Trader (Technical)", value=True)
include_value = st.sidebar.checkbox("💎 Disciplined Value Investor", value=True)
include_risk = st.sidebar.checkbox("🛡️ Risk-Averse Capital Preserver", value=True)
include_news = st.sidebar.checkbox("📰 News & Sentiment Trader", value=False)

personas_to_run = []
if include_dad:
    personas_to_run.append("dad_hybrid")
if include_momentum:
    personas_to_run.append("momentum_trader")
if include_value:
    personas_to_run.append("value_investor")
if include_risk:
    personas_to_run.append("risk_averse")
if include_news:
    personas_to_run.append("news_sentiment")

st.sidebar.markdown("---")
st.sidebar.markdown("### ⏱️ Dad's Evaluation Cadence")
dad_freq = st.sidebar.selectbox(
    "Dad's Trading Horizon",
    options=["quarterly", "monthly", "yearly", "daily"],
    index=0,
    help="Dad's rules evaluate multi-year financials to pick stocks to hold for years. Quarterly matches his natural fundamental cadence.",
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🧠 LLM Execution Engine")
engine_choice = st.sidebar.radio(
    "Choose Brain Provider",
    options=["Mock Rule Engine (Free, Instant & Offline)", "Google Gemini API", "OpenAI API"],
    index=0,
)

provider_key = "mock"
api_key = None
if engine_choice == "Google Gemini API":
    provider_key = "gemini"
    api_key = st.sidebar.text_input("Gemini API Key", type="password", value=os.environ.get("GEMINI_API_KEY", ""))
elif engine_choice == "OpenAI API":
    provider_key = "openai"
    api_key = st.sidebar.text_input("OpenAI API Key", type="password", value=os.environ.get("OPENAI_API_KEY", ""))

run_btn = st.sidebar.button("🚀 Run Battle Simulation", type="primary", use_container_width=True)

# Session State for storing backtest results
if "backtest_results" not in st.session_state:
    st.session_state["backtest_results"] = None


# Trigger Simulation
if run_btn or st.session_state["backtest_results"] is None:
    if not selected_tickers:
        st.error("Please select at least one NSE ticker.")
    elif not personas_to_run:
        st.error("Please select at least one trading persona.")
    else:
        with st.spinner("⚔️ Running battle simulation across NSE price history & balance sheets..."):
            bt = Backtester(
                tickers=selected_tickers,
                period=period,
                initial_cash=float(initial_cash),
                personas=personas_to_run,
                eval_frequencies={"dad_hybrid": dad_freq},
                llm_provider=provider_key,
                api_key=api_key,
            )
            try:
                results = bt.run()
                st.session_state["backtest_results"] = results
            except Exception as e:
                st.error(f"Simulation failed: {e}")

results = st.session_state.get("backtest_results")

if results:
    metrics_df = results["metrics_df"]
    equity_df = results["equity_df"]
    dad_analysis = results["dad_analysis"]
    portfolios = results["portfolios"]
    benchmark = results["benchmark"]

    # Dashboard Tabs
    tab_arena, tab_dad_rules, tab_logs, tab_personas, tab_host = st.tabs([
        "🏆 Arena Leaderboard",
        "📜 Dad's 14 Rules Scorecard",
        "🔍 Trade Logs & LLM Reasoning",
        "📖 Persona Strategies & Cadence",
        "🚀 How to Host This Online",
    ])

    with tab_arena:
        # Leaderboard Cards
        col1, col2, col3, col4 = st.columns(4)

        winner = metrics_df.iloc[0]
        dad_row = metrics_df[metrics_df["persona"] == "dad_hybrid"]
        bench_row = metrics_df[metrics_df["persona"].str.contains("Benchmark", na=False)]

        dad_ret = dad_row["total_return_pct"].values[0] if not dad_row.empty else 0.0
        bench_ret = bench_row["total_return_pct"].values[0] if not bench_row.empty else 0.0
        dad_alpha = dad_ret - bench_ret

        with col1:
            st.markdown(
                f"""
            <div class="metric-card">
                <div class="metric-label">🥇 #1 Strategy Rank</div>
                <div class="metric-value">{winner['persona'].replace('_', ' ').title()}</div>
                <div class="{'metric-delta-positive' if winner['total_return_pct'] >= 0 else 'metric-delta-negative'}">
                    +{winner['total_return_pct']:.2f}% Return
                </div>
            </div>
            """,
                unsafe_allow_html=True,
            )

        with col2:
            st.markdown(
                f"""
            <div class="metric-card">
                <div class="metric-label">👑 Dad's Hybrid Return</div>
                <div class="metric-value">{dad_ret:+.2f}%</div>
                <div class="{'metric-delta-positive' if dad_alpha >= 0 else 'metric-delta-negative'}">
                    {dad_alpha:+.2f}% Alpha vs Buy & Hold
                </div>
            </div>
            """,
                unsafe_allow_html=True,
            )

        with col3:
            best_sharpe = metrics_df.sort_values("sharpe_ratio", ascending=False).iloc[0]
            st.markdown(
                f"""
            <div class="metric-card">
                <div class="metric-label">📊 Top Sharpe Ratio</div>
                <div class="metric-value">{best_sharpe['sharpe_ratio']:.2f}</div>
                <div style="color: #94a3b8; font-size: 0.85rem;">
                    {best_sharpe['persona'].replace('_', ' ').title()}
                </div>
            </div>
            """,
                unsafe_allow_html=True,
            )

        with col4:
            lowest_dd = metrics_df.sort_values("max_drawdown_pct", ascending=True).iloc[0]
            st.markdown(
                f"""
            <div class="metric-card">
                <div class="metric-label">🛡️ Lowest Max Drawdown</div>
                <div class="metric-value">{lowest_dd['max_drawdown_pct']:.2f}%</div>
                <div style="color: #94a3b8; font-size: 0.85rem;">
                    {lowest_dd['persona'].replace('_', ' ').title()}
                </div>
            </div>
            """,
                unsafe_allow_html=True,
            )

        # Plotly Interactive Equity Curve
        st.markdown("#### 📈 Portfolio Valuation Over Time (₹ INR)")

        fig = go.Figure()

        color_palette = {
            "dad_hybrid": "#f59e0b",           # Warm Gold for Dad
            "momentum_trader": "#38bdf8",      # Bright Cyan for Momentum
            "value_investor": "#10b981",       # Emerald Green for Value
            "risk_averse": "#a855f7",          # Purple for Risk-Averse
            "news_sentiment": "#ec4899",       # Pink for Sentiment
            "Buy & Hold Benchmark": "#94a3b8", # Slate for Benchmark
        }

        for col in equity_df.columns:
            if col == "date":
                continue
            line_color = color_palette.get(col, "#ffffff")
            is_dad = col == "dad_hybrid"
            is_bench = "Benchmark" in col

            fig.add_trace(
                go.Scatter(
                    x=equity_df["date"],
                    y=equity_df[col],
                    mode="lines",
                    name=col.replace("_", " ").title(),
                    line=dict(
                        color=line_color,
                        width=3.5 if is_dad else (2.0 if not is_bench else 1.8),
                        dash="dash" if is_bench else "solid",
                    ),
                    hovertemplate="<b>%{fullData.name}</b><br>Date: %{x}<br>Portfolio: ₹%{y:,.2f}<extra></extra>",
                )
            )

        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(15, 23, 42, 0.4)",
            plot_bgcolor="rgba(15, 23, 42, 0.4)",
            font=dict(family="Inter, sans-serif", color="#e2e8f0"),
            hovermode="x unified",
            margin=dict(l=20, r=20, t=20, b=20),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
            ),
            xaxis=dict(showgrid=True, gridcolor="rgba(255, 255, 255, 0.05)"),
            yaxis=dict(showgrid=True, gridcolor="rgba(255, 255, 255, 0.05)", tickprefix="₹", separatethousands=True),
            height=460,
        )

        st.plotly_chart(fig, use_container_width=True)

        # Performance Comparison Table
        st.markdown("#### 📊 Quantitative Metrics Breakdown")

        display_metrics = metrics_df.copy()
        display_metrics["persona"] = display_metrics["persona"].apply(lambda x: x.replace("_", " ").title())
        display_metrics["initial_cash"] = display_metrics["initial_cash"].apply(lambda x: f"₹{x:,.0f}")
        display_metrics["final_value"] = display_metrics["final_value"].apply(lambda x: f"₹{x:,.2f}")
        display_metrics["net_profit"] = display_metrics["net_profit"].apply(lambda x: f"₹{x:+,.2f}")
        display_metrics["total_return_pct"] = display_metrics["total_return_pct"].apply(lambda x: f"{x:+.2f}%")
        display_metrics["cagr_pct"] = display_metrics["cagr_pct"].apply(lambda x: f"{x:+.2f}%")
        display_metrics["max_drawdown_pct"] = display_metrics["max_drawdown_pct"].apply(lambda x: f"{x:.2f}%")
        display_metrics["sharpe_ratio"] = display_metrics["sharpe_ratio"].apply(lambda x: f"{x:.2f}")

        st.dataframe(
            display_metrics[[
                "persona", "frequency", "initial_cash", "final_value", "net_profit",
                "total_return_pct", "cagr_pct", "sharpe_ratio", "max_drawdown_pct", "total_trades"
            ]],
            use_container_width=True,
            hide_index=True,
        )

    with tab_dad_rules:
        st.markdown("### 📜 Dad's 14 Thumb Rules Scorecard")
        st.markdown("Select an active stock to inspect how it scores against Dad's exact thumb rules using yfinance balance sheets, income statements, and cash flows.")

        selected_analysis_ticker = st.selectbox(
            "Select Ticker for Fundamental Audit",
            options=selected_tickers,
            index=0,
        )

        if selected_analysis_ticker in dad_analysis:
            analysis = dad_analysis[selected_analysis_ticker]
            fund_data = fetch_fundamentals_data(selected_analysis_ticker)

            score_col1, score_col2, score_col3 = st.columns([1, 1, 2])
            with score_col1:
                st.markdown(
                    f"""
                <div class="metric-card" style="text-align: center;">
                    <div class="metric-label">Dad's Quality Score</div>
                    <div class="metric-value" style="color: #38bdf8;">{analysis['overall_score']}/100</div>
                    <div style="font-weight: 700; color: #f8fafc; font-size: 0.9rem;">
                        Verdict: {analysis['verdict']}
                    </div>
                </div>
                """,
                    unsafe_allow_html=True,
                )

            with score_col2:
                st.markdown(
                    f"""
                <div class="metric-card" style="text-align: center;">
                    <div class="metric-label">Rules Breakdown</div>
                    <div style="margin-top: 0.5rem;">
                        <span class="badge-pass">{analysis['pass_count']} PASS</span>
                        <span class="badge-caution">{analysis['caution_count']} CAUTION</span>
                        <span class="badge-fail">{analysis['fail_count']} FAIL</span>
                    </div>
                </div>
                """,
                    unsafe_allow_html=True,
                )

            with score_col3:
                st.markdown(
                    f"""
                <div class="metric-card">
                    <div class="metric-label">Company Snapshot</div>
                    <div style="font-size: 1.1rem; font-weight: 700; color: #f8fafc;">{fund_data.get('name', selected_analysis_ticker)}</div>
                    <div style="color: #94a3b8; font-size: 0.85rem; margin-top: 0.2rem;">
                        Sector: <strong>{fund_data.get('sector', 'N/A')}</strong> | Industry: <strong>{fund_data.get('industry', 'N/A')}</strong><br>
                        Market Cap: <strong>{format_inr(fund_data.get('market_cap'))}</strong> | Trailing P/E: <strong>{fund_data.get('trailing_pe', 'N/A')}</strong>
                    </div>
                </div>
                """,
                    unsafe_allow_html=True,
                )

            st.markdown("#### Detailed 14-Rule Verification Checklist")
            for rule_title, rule_info in analysis["scores"].items():
                badge_class = "badge-pass" if rule_info["status"] == "PASS" else ("badge-caution" if rule_info["status"] == "CAUTION" else "badge-fail")
                st.markdown(
                    f"""
                <div class="rule-card">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.4rem;">
                        <span style="font-weight: 600; font-size: 0.95rem; color: #f1f5f9;">{rule_title}</span>
                        <div>
                            <span style="font-family: 'JetBrains Mono', monospace; font-size: 0.85rem; color: #94a3b8; margin-right: 0.8rem;">{rule_info.get('metric', '')}</span>
                            <span class="{badge_class}">{rule_info['status']}</span>
                        </div>
                    </div>
                    <div style="color: #cbd5e1; font-size: 0.88rem; line-height: 1.4;">
                        {rule_info['detail']}
                    </div>
                </div>
                """,
                    unsafe_allow_html=True,
                )

    with tab_logs:
        st.markdown("### 🔍 Trade Logs & LLM Reasoning Trail")
        st.markdown("Every single trade executed in the simulation contains the persona's explicit reasoning and rule triggers.")

        log_persona = st.selectbox(
            "Filter by Persona",
            options=personas_to_run + ["buy_and_hold_benchmark"],
            format_func=lambda x: x.replace("_", " ").title(),
            index=0,
        )

        action_filter = st.radio("Filter Actions", options=["All", "Buy & Sell Only", "Holds Only"], horizontal=True)

        selected_portfolio = portfolios.get(log_persona) or benchmark
        trade_logs = selected_portfolio.trade_log

        if trade_logs:
            df_logs = pd.DataFrame(trade_logs)
            if action_filter == "Buy & Sell Only":
                df_logs = df_logs[df_logs["action"].isin(["buy", "sell"])]
            elif action_filter == "Holds Only":
                df_logs = df_logs[df_logs["action"] == "hold"]

            st.dataframe(
                df_logs[["date", "ticker", "action", "shares", "price", "cash_after", "confidence", "reasoning"]],
                use_container_width=True,
                hide_index=True,
            )

            csv_data = df_logs.to_csv(index=False).encode("utf-8")
            st.download_button(
                label=f"📥 Download {log_persona.title()} Trades CSV",
                data=csv_data,
                file_name=f"{log_persona}_trades.csv",
                mime="text/csv",
            )
        else:
            st.info("No trades logged for this persona.")

    with tab_personas:
        st.markdown("### 📖 Persona Strategies & Evaluation Cadence")
        st.markdown(
            """
        #### 1. Why Frequency Matters: Long-Term Horizon vs Daily Noise
        A crucial realization in building this experiment is that **Dad's rules are not day-trading rules**.
        - Dad evaluates **15-20 years of financial statements**, debt deleveraging, working capital ratios, and 5-year sales doubling.
        - Evaluating Dad daily in a short-term trading loop creates artificial churn and transaction drag.
        - Therefore, **Dad evaluates quarterly (or monthly/yearly)**, holding compounders for the long term.
        - In contrast, the **Momentum Trader** trades daily price trends and RSI momentum.

        #### 2. Nuances in Dad's 14 Rules
        - **Rule 6 (Short-Term Debt vs Working Capital)**: Short-term borrowings are acceptable for operational growth, but should ideally remain *less than working capital* and grow no faster than top-line revenues.
        - **Rule 11 (Other Income 3-4% Threshold)**: Cash parked in investments or bonds should earn productive yield (at least 3-4%), demonstrating disciplined treasury management rather than idle capital.
        - **Rule 12 (Balance Sheet Coherence)**: Capex spent must visibly translate into revenue expansion in subsequent years; otherwise, heavy investment is a red flag.
        """
        )

    with tab_host:
        st.markdown("### 🚀 How to Host This Application Online (Free in 2 Minutes)")
        st.markdown(
            """
        You can host this interactive dashboard for free so anyone (including your Dad!) can open it on their phone or laptop.

        #### Option 1: Streamlit Community Cloud (Recommended & Free)
        1. Go to **[share.streamlit.io](https://share.streamlit.io)** and log in with your GitHub account.
        2. Click **"New app"**.
        3. Select your repository: `kavya-sajjit/acha-vs-ai`
        4. Main file path: `app.py`
        5. Click **"Deploy"**!
        6. *(Optional)* Add your `GEMINI_API_KEY` or `OPENAI_API_KEY` in Streamlit's **App Settings -> Secrets**.

        #### Option 2: Render / Railway / Hugging Face Spaces
        - Render: Create a new **Web Service**, link `https://github.com/kavya-sajjit/acha-vs-ai`, set Start Command: `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`.
        """
        )
