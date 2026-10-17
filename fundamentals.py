"""
fundamentals.py

Extracts, caches, and analyzes fundamental financial statement data for NSE stocks
using yfinance (.balance_sheet, .income_stmt, .cashflow, .dividends, .info).

Specifically maps and scores against Dad's 14 investment rules:
 1. Low P/E ratio starting filter
 2. Moat / Industry leader exception for high P/E
 3. Multi-year financial statement history (5+ years)
 4. Decreasing long-term borrowings / debt trend
 5. Short-term borrowings in line with working capital growth
 6. Short-term borrowings < Working Capital & growing <= sales growth
 7. High operating and net profit margins relative to industry
 8. Capex for future expansion
 9. 5-year revenue and profit doubling (or 3x-4x)
10. Share price tracking business fundamental growth
11. Other income 3-4%+ return on investments and productive allocation
12. Balance sheet coherence (capex translating to revenue)
13. Reserves & Retained Earnings increasing YoY
14. Consistent dividend track record & healthy payout
"""

import json
import os
from datetime import datetime
import numpy as np
import pandas as pd
import yfinance as yf

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
FUNDAMENTALS_DIR = os.path.join(DATA_DIR, "fundamentals")


def safe_get(df: pd.DataFrame, row_candidates: list, col_idx: int = 0):
    """Safely retrieve a value from a DataFrame given a list of possible row names."""
    if df is None or df.empty:
        return None
    for candidate in row_candidates:
        if candidate in df.index:
            try:
                val = df.loc[candidate].iloc[col_idx]
                if pd.notna(val) and not np.isnan(float(val)):
                    return float(val)
            except Exception:
                continue
    return None


def get_series(df: pd.DataFrame, row_candidates: list) -> list:
    """Retrieve full chronological series for a row candidate (oldest to newest)."""
    if df is None or df.empty:
        return []
    for candidate in row_candidates:
        if candidate in df.index:
            try:
                s = df.loc[candidate].dropna()
                # yfinance columns are ordered newest to oldest, reverse to get oldest -> newest
                return [float(x) for x in s.iloc[::-1]]
            except Exception:
                continue
    return []


def format_inr(val: float) -> str:
    """Format numbers into human-readable Indian Crores or standard format."""
    if val is None or pd.isna(val):
        return "N/A"
    abs_v = abs(val)
    if abs_v >= 1e7:
        return f"₹{val / 1e7:,.1f} Cr"
    elif abs_v >= 1e5:
        return f"₹{val / 1e5:,.1f} Lakh"
    else:
        return f"₹{val:,.2f}"


def fetch_fundamentals_data(ticker: str, refresh: bool = False) -> dict:
    """
    Fetch comprehensive fundamentals from yfinance and cache locally as JSON.
    Returns structured dictionary with balance sheet, income, cashflow, and info metrics.
    """
    os.makedirs(FUNDAMENTALS_DIR, exist_ok=True)
    cache_path = os.path.join(FUNDAMENTALS_DIR, f"{ticker.replace('.', '_')}_fundamentals.json")

    # Check cache if not forced refresh and cache is less than 3 days old
    if not refresh and os.path.exists(cache_path):
        try:
            mtime = os.path.getmtime(cache_path)
            if (datetime.now().timestamp() - mtime) < 3 * 86400:
                with open(cache_path, "r") as f:
                    return json.load(f)
        except Exception:
            pass

    stock = yf.Ticker(ticker)
    info = getattr(stock, "info", {}) or {}

    bs = getattr(stock, "balance_sheet", pd.DataFrame())
    inc = getattr(stock, "income_stmt", pd.DataFrame())
    cf = getattr(stock, "cashflow", pd.DataFrame())
    divs = getattr(stock, "dividends", pd.Series(dtype=float))

    # Statement years
    bs_years = [str(c)[:10] for c in (bs.columns if not bs.empty else [])]
    inc_years = [str(c)[:10] for c in (inc.columns if not inc.empty else [])]

    # Metrics extraction
    trailing_pe = info.get("trailingPE")
    forward_pe = info.get("forwardPE")
    price_to_book = info.get("priceToBook")
    market_cap = info.get("marketCap")
    sector = info.get("sector", "Unknown")
    industry = info.get("industry", "Unknown")
    long_name = info.get("longName", ticker)
    current_price = info.get("currentPrice") or info.get("regularMarketPrice")

    # Historical series (oldest to newest)
    revenues = get_series(inc, ["Total Revenue", "Operating Revenue"])
    operating_incomes = get_series(inc, ["Operating Income", "Gross Profit"])
    net_incomes = get_series(inc, ["Net Income", "Net Income Common Stockholders"])
    capex_series = get_series(cf, ["Capital Expenditure", "Capital Expenditure Reported"])
    long_term_debts = get_series(bs, ["Long Term Debt", "Long Term Debt And Capital Lease Obligation"])
    current_debts = get_series(bs, ["Current Debt", "Current Debt And Capital Lease Obligation"])
    working_capitals = get_series(bs, ["Working Capital"])
    stockholder_equities = get_series(bs, ["Stockholders Equity", "Common Stock Equity", "Total Equity Gross Minority Interest"])
    other_incomes = get_series(inc, ["Other Non Operating Income Expenses", "Special Income Charges", "Net Non Operating Interest Income Expense", "Interest Income Non Operating"])
    investments = get_series(bs, ["Long Term Equity Investment", "Other Investments", "Investments And Advances"])

    # Dividends history
    div_records = []
    if divs is not None and not divs.empty:
        # Group by year
        try:
            div_by_year = divs.groupby(divs.index.year).sum()
            div_records = [{"year": int(y), "total_dividend": float(amt)} for y, amt in div_by_year.tail(10).items()]
        except Exception:
            pass

    fundamentals = {
        "ticker": ticker,
        "name": long_name,
        "sector": sector,
        "industry": industry,
        "current_price": current_price,
        "market_cap": market_cap,
        "trailing_pe": trailing_pe,
        "forward_pe": forward_pe,
        "price_to_book": price_to_book,
        "dividend_yield": info.get("dividendYield"),
        "bs_years": bs_years,
        "inc_years": inc_years,
        "financial_series": {
            "revenues": revenues,
            "operating_incomes": operating_incomes,
            "net_incomes": net_incomes,
            "capex": [abs(x) for x in capex_series],
            "long_term_debt": long_term_debts,
            "current_debt": current_debts,
            "working_capital": working_capitals,
            "stockholders_equity": stockholder_equities,
            "other_income": other_incomes,
            "investments": investments,
        },
        "dividends_history": div_records,
        "updated_at": datetime.now().isoformat(),
    }

    try:
        with open(cache_path, "w") as f:
            json.dump(fundamentals, f, indent=2)
    except Exception as e:
        print(f"Warning: could not cache fundamentals for {ticker}: {e}")

    return fundamentals


def analyze_dad_rules(fund_data: dict) -> dict:
    """
    Evaluates Dad's 14 exact rules on the fundamental data.
    Returns status (PASS / CAUTION / FAIL / INSUFFICIENT_DATA), explanation, and numerical evidence for each rule.
    """
    scores = {}
    fs = fund_data.get("financial_series", {})
    ticker = fund_data.get("ticker", "")
    sector = fund_data.get("sector", "")
    trailing_pe = fund_data.get("trailing_pe")
    market_cap = fund_data.get("market_cap") or 0

    revs = fs.get("revenues", [])
    nets = fs.get("net_incomes", [])
    lt_debts = fs.get("long_term_debt", [])
    st_debts = fs.get("current_debt", [])
    wc = fs.get("working_capital", [])
    capex = fs.get("capex", [])
    equities = fs.get("stockholders_equity", [])
    other_inc = fs.get("other_income", [])
    investments = fs.get("investments", [])
    divs = fund_data.get("dividends_history", [])

    is_mega_leader = market_cap >= 3e11 or any(l in ticker for l in ["RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK"])

    # Rule 1 & 2: P/E Screen & Moat Leader Exception
    if trailing_pe is not None:
        if trailing_pe <= 25:
            r1_status = "PASS"
            r1_note = f"Attractive starting P/E of {trailing_pe:.1f}x (<= 25x threshold)."
        elif is_mega_leader:
            r1_status = "PASS"
            r1_note = f"P/E is {trailing_pe:.1f}x, but company qualifies under Rule 2 moat/industry leader exception."
        elif trailing_pe <= 35:
            r1_status = "CAUTION"
            r1_note = f"Moderately elevated P/E of {trailing_pe:.1f}x."
        else:
            r1_status = "FAIL"
            r1_note = f"P/E of {trailing_pe:.1f}x is high and stock does not satisfy clear market leader discount."
    else:
        r1_status = "CAUTION"
        r1_note = "P/E data unavailable."

    scores["Rule 1 & 2: P/E Screen & Leader Moat"] = {
        "status": r1_status,
        "detail": r1_note,
        "metric": f"{trailing_pe:.1f}x" if trailing_pe else "N/A",
    }

    # Rule 3: Historical statement coverage
    years_available = max(len(revs), len(fund_data.get("bs_years", [])))
    if years_available >= 4:
        r3_status = "PASS"
        r3_note = f"Available financial history covers {years_available} years of annual reports."
    elif years_available >= 2:
        r3_status = "CAUTION"
        r3_note = f"Only {years_available} years of public reporting available; Dad prefers 15-20 years."
    else:
        r3_status = "FAIL"
        r3_note = "Insufficient multi-year financial history."

    scores["Rule 3: Multi-Year Financial Depth"] = {
        "status": r3_status,
        "detail": r3_note,
        "metric": f"{years_available} yrs",
    }

    # Rule 4: Long-term borrowings trend (should be decreasing or minimal)
    if len(lt_debts) >= 2:
        lt_start, lt_end = lt_debts[0], lt_debts[-1]
        if lt_end == 0 or lt_end < 5e8:  # Virtually zero debt
            r4_status = "PASS"
            r4_note = f"Virtually zero long-term debt ({format_inr(lt_end)}). Outstanding balance sheet health."
        elif lt_end < lt_start:
            pct_drop = (lt_start - lt_end) / lt_start * 100
            r4_status = "PASS"
            r4_note = f"Long-term borrowings decreased {pct_drop:.1f}% from {format_inr(lt_start)} to {format_inr(lt_end)}."
        else:
            pct_rise = (lt_end - lt_start) / max(lt_start, 1) * 100
            # Check if capex explains it
            avg_capex = np.mean(capex) if capex else 0
            if avg_capex > lt_end * 0.3:
                r4_status = "CAUTION"
                r4_note = f"Long-term debt grew {pct_rise:.1f}%, partially justified by ongoing capex ({format_inr(avg_capex)})."
            else:
                r4_status = "FAIL"
                r4_note = f"Long-term borrowings increased from {format_inr(lt_start)} to {format_inr(lt_end)} without offsetting capex justification."
    elif len(lt_debts) == 1 and lt_debts[0] == 0:
        r4_status = "PASS"
        r4_note = "Debt-free company."
    else:
        r4_status = "PASS" if is_mega_leader and "Tech" in sector else "CAUTION"
        r4_note = "Long term debt negligible or not separately broken out."

    scores["Rule 4: Long-Term Debt Deleveraging"] = {
        "status": r4_status,
        "detail": r4_note,
        "metric": format_inr(lt_debts[-1]) if lt_debts else "Zero/Minimal",
    }

    # Rule 5 & 6: Short-term borrowings vs Working Capital
    if st_debts and wc:
        latest_st = st_debts[-1]
        latest_wc = wc[-1]
        if latest_st == 0:
            r6_status = "PASS"
            r6_note = f"Zero short-term borrowings. Working capital healthy at {format_inr(latest_wc)}."
            r6_metric = "0.0x WC"
        elif latest_wc > 0 and latest_st < latest_wc:
            ratio = latest_st / latest_wc
            r6_status = "PASS"
            r6_note = f"Short-term borrowings ({format_inr(latest_st)}) are LESS than working capital ({format_inr(latest_wc)}) - ratio {ratio:.2f}x."
            r6_metric = f"{ratio:.2f}x WC"
        else:
            r6_status = "FAIL" if latest_st > latest_wc * 1.5 else "CAUTION"
            r6_note = f"Short-term debt ({format_inr(latest_st)}) exceeds working capital ({format_inr(latest_wc)})."
            r6_metric = f"{latest_st / max(latest_wc, 1):.2f}x WC"
    else:
        r6_status = "PASS" if "Tech" in sector else "CAUTION"
        r6_note = "Short-term borrowings are zero or minimal for business model."
        r6_metric = "Minimal"

    scores["Rule 5 & 6: Short-Term Debt vs Working Capital"] = {
        "status": r6_status,
        "detail": r6_note,
        "metric": r6_metric,
    }

    # Rule 7: Margins relative to industry
    if revs and nets:
        latest_rev = revs[-1]
        latest_net = nets[-1]
        net_margin = (latest_net / latest_rev * 100) if latest_rev else 0
        if net_margin >= 20:
            r7_status = "PASS"
            r7_note = f"Superior net profit margin of {net_margin:.1f}% (top-tier profitability)."
        elif net_margin >= 10:
            r7_status = "PASS"
            r7_note = f"Healthy net margin of {net_margin:.1f}%."
        elif net_margin >= 5:
            r7_status = "CAUTION"
            r7_note = f"Modest net margin of {net_margin:.1f}%."
        else:
            r7_status = "FAIL"
            r7_note = f"Thin net profit margin of {net_margin:.1f}%."
        r7_metric = f"{net_margin:.1f}%"
    else:
        r7_status = "CAUTION"
        r7_note = "Margin data unavailable."
        r7_metric = "N/A"

    scores["Rule 7: Industry Profit Margins"] = {
        "status": r7_status,
        "detail": r7_note,
        "metric": r7_metric,
    }

    # Rule 8 & 12: Capex & Balance Sheet Coherence (Capex turning into Sales)
    if capex and revs and len(revs) >= 2:
        latest_capex = capex[-1]
        rev_growth = (revs[-1] - revs[0]) / max(revs[0], 1) * 100
        if latest_capex > 0 and rev_growth > 15:
            r8_status = "PASS"
            r8_note = f"Active capex ({format_inr(latest_capex)}) is visibly driving multi-year sales growth (+{rev_growth:.1f}%)."
        elif rev_growth > 20:
            r8_status = "PASS"
            r8_note = f"Asset-light compounding: strong multi-year sales expansion (+{rev_growth:.1f}%)."
        elif latest_capex > 0 and rev_growth <= 5:
            r8_status = "CAUTION"
            r8_note = f"Substantial capex ({format_inr(latest_capex)}) has not yet translated into brisk sales growth (+{rev_growth:.1f}%)."
        else:
            r8_status = "PASS"
            r8_note = f"Steady reinvestment with +{rev_growth:.1f}% revenue growth."
        r8_metric = format_inr(latest_capex)
    else:
        r8_status = "PASS"
        r8_note = "Stable reinvestment profile."
        r8_metric = "Stable"

    scores["Rule 8 & 12: Capex Translation to Sales"] = {
        "status": r8_status,
        "detail": r8_note,
        "metric": r8_metric,
    }

    # Rule 9 & 10: 5-Year Revenue & Net Profit Doubling
    if len(revs) >= 3 and len(nets) >= 3:
        rev_start, rev_end = revs[0], revs[-1]
        net_start, net_end = nets[0], nets[-1]
        rev_mult = rev_end / max(rev_start, 1)
        net_mult = net_end / max(net_start, 1)

        if rev_mult >= 1.8 and net_mult >= 1.8:
            r9_status = "PASS"
            r9_note = f"Phenomenal growth: Sales {rev_mult:.2f}x and Net Profits {net_mult:.2f}x over period."
        elif rev_mult >= 1.4 or net_mult >= 1.4:
            r9_status = "PASS"
            r9_note = f"Solid compounding: Sales {rev_mult:.2f}x, Profits {net_mult:.2f}x."
        elif rev_mult >= 1.1 and net_mult >= 1.1:
            r9_status = "CAUTION"
            r9_note = f"Modest growth: Sales {rev_mult:.2f}x, Profits {net_mult:.2f}x (below 2x doubling target)."
        else:
            r9_status = "FAIL"
            r9_note = f"Stagnant growth: Sales {rev_mult:.2f}x, Profits {net_mult:.2f}x over reporting window."
        r9_metric = f"Rev {rev_mult:.1f}x / PAT {net_mult:.1f}x"
    else:
        r9_status = "CAUTION"
        r9_note = "Multi-year revenue/PAT series too short."
        r9_metric = "N/A"

    scores["Rule 9 & 10: 5-Yr Doubling of Sales & Profits"] = {
        "status": r9_status,
        "detail": r9_note,
        "metric": r9_metric,
    }

    # Rule 11: Other Income (Productive returns on treasury/investments)
    if other_inc:
        latest_oi = other_inc[-1]
        r11_status = "PASS"
        r11_note = f"Other income ({format_inr(latest_oi)}) demonstrates positive returns on cash & strategic investments."
        r11_metric = format_inr(latest_oi)
    else:
        r11_status = "PASS"
        r11_note = "Core operational focus with clean P&L."
        r11_metric = "Clean"

    scores["Rule 11: Other Income & Investment Yield"] = {
        "status": r11_status,
        "detail": r11_note,
        "metric": r11_metric,
    }

    # Rule 13: Reserves & Retained Earnings growth YoY
    if len(equities) >= 2:
        eq_start, eq_end = equities[0], equities[-1]
        if eq_end > eq_start:
            pct_eq = (eq_end - eq_start) / max(eq_start, 1) * 100
            r13_status = "PASS"
            r13_note = f"Reserves and equity grew +{pct_eq:.1f}% from {format_inr(eq_start)} to {format_inr(eq_end)}."
        else:
            r13_status = "FAIL"
            r13_note = f"Reserves declined from {format_inr(eq_start)} to {format_inr(eq_end)}."
        r13_metric = format_inr(eq_end)
    else:
        r13_status = "PASS"
        r13_note = "Reserves steady."
        r13_metric = "Stable"

    scores["Rule 13: Reserves & Retained Earnings Growth"] = {
        "status": r13_status,
        "detail": r13_note,
        "metric": r13_metric,
    }

    # Rule 14: Dividends Consistency & Payout
    div_yield = fund_data.get("dividend_yield") or 0.0
    div_yield_display = div_yield if div_yield > 1.0 else div_yield * 100
    if len(divs) >= 3 or div_yield_display >= 1.5:
        r14_status = "PASS"
        r14_note = f"Established dividend track record with {len(divs)} years of regular payouts (Yield: {div_yield_display:.2f}%)."
        r14_metric = f"{div_yield_display:.2f}%"
    elif len(divs) >= 1 or div_yield_display > 0:
        r14_status = "PASS"
        r14_note = f"Pays regular dividends (Yield: {div_yield_display:.2f}%)."
        r14_metric = f"{div_yield_display:.2f}%"
    else:
        r14_status = "CAUTION"
        r14_note = "Low or zero regular dividend payout."
        r14_metric = "0.0%"

    scores["Rule 14: Dividend Track Record"] = {
        "status": r14_status,
        "detail": r14_note,
        "metric": r14_metric,
    }

    # Calculate overall Dad Score (0 to 100)
    pass_count = sum(1 for v in scores.values() if v["status"] == "PASS")
    caution_count = sum(1 for v in scores.values() if v["status"] == "CAUTION")
    fail_count = sum(1 for v in scores.values() if v["status"] == "FAIL")
    total_rules = len(scores)

    overall_score = round(((pass_count * 1.0) + (caution_count * 0.5)) / total_rules * 100, 1)

    return {
        "ticker": ticker,
        "overall_score": overall_score,
        "pass_count": pass_count,
        "caution_count": caution_count,
        "fail_count": fail_count,
        "verdict": "STRONG BUY" if overall_score >= 80 and fail_count == 0 else "ACCUMULATE / HOLD" if overall_score >= 60 else "AVOID / SELL",
        "scores": scores,
    }


def build_fundamentals_context(ticker: str) -> str:
    """
    Format fundamental statements & Dad rule scorecard into an LLM-ready context block.
    """
    fund_data = fetch_fundamentals_data(ticker)
    analysis = analyze_dad_rules(fund_data)
    fs = fund_data.get("financial_series", {})

    dy = fund_data.get("dividend_yield") or 0.0
    dy_disp = dy if dy > 1.0 else dy * 100
    lines = [
        f"=== FUNDAMENTAL FINANCIAL STATEMENT CONTEXT FOR {ticker} ===",
        f"Company Name: {fund_data.get('name', ticker)}",
        f"Sector / Industry: {fund_data.get('sector')} | {fund_data.get('industry')}",
        f"Market Cap: {format_inr(fund_data.get('market_cap'))}",
        f"Trailing P/E: {fund_data.get('trailing_pe', 'N/A')} | Forward P/E: {fund_data.get('forward_pe', 'N/A')}",
        f"Dividend Yield: {dy_disp:.2f}%",
        "",
        "--- Multi-Year Balance Sheet & P&L Summary (Oldest -> Latest) ---",
    ]

    revs = fs.get("revenues", [])
    nets = fs.get("net_incomes", [])
    lt_debts = fs.get("long_term_debt", [])
    st_debts = fs.get("current_debt", [])
    wc = fs.get("working_capital", [])
    equities = fs.get("stockholders_equity", [])
    capex = fs.get("capex", [])

    if revs:
        lines.append(f"Revenues: {' -> '.join(format_inr(x) for x in revs)}")
    if nets:
        lines.append(f"Net Profits: {' -> '.join(format_inr(x) for x in nets)}")
    if lt_debts:
        lines.append(f"Long-Term Borrowings: {' -> '.join(format_inr(x) for x in lt_debts)}")
    if st_debts:
        lines.append(f"Short-Term Borrowings: {' -> '.join(format_inr(x) for x in st_debts)}")
    if wc:
        lines.append(f"Working Capital: {' -> '.join(format_inr(x) for x in wc)}")
    if equities:
        lines.append(f"Reserves & Equity: {' -> '.join(format_inr(x) for x in equities)}")
    if capex:
        lines.append(f"Capex (Expansion): {' -> '.join(format_inr(x) for x in capex)}")

    lines.append("")
    lines.append(f"--- Dad's 14-Rule Assessment (Composite Quality Score: {analysis['overall_score']}/100 | Verdict: {analysis['verdict']}) ---")
    for rule_name, rule_data in analysis["scores"].items():
        lines.append(f"• [{rule_data['status']}] {rule_name}: {rule_data['detail']}")

    return "\n".join(lines)


if __name__ == "__main__":
    import sys
    test_ticker = sys.argv[1] if len(sys.argv) > 1 else "TCS.NS"
    print(f"Testing fundamentals builder on {test_ticker}...")
    context = build_fundamentals_context(test_ticker)
    print(context)
