"""
test_portfolio.py

Sanity check for portfolio.py using a few hardcoded trades - no network,
no AI. If this prints numbers that match what you'd expect by hand, the
core bookkeeping is correct and safe to build the LLM layer on top of.

Run:
    python test_portfolio.py
"""

from portfolio import Portfolio


def main():
    p = Portfolio(starting_cash=1000, name="sanity_check")

    print("Starting cash: 1000")

    # Buy 2 shares of a stock at 100 each -> should leave 800 cash
    p.buy("TEST.NS", shares=2, price=100)
    print(f"After buying 2 @ 100: cash = {p.cash}")
    assert p.cash == 800, "cash should be 800 after this buy"

    # Price goes up to 120, sell 1 share -> should leave 920 cash
    p.sell("TEST.NS", shares=1, price=120)
    print(f"After selling 1 @ 120: cash = {p.cash}")
    assert p.cash == 920, "cash should be 920 after this sell"

    # Try to overspend - should be refused, cash unchanged
    bought = p.buy("TEST.NS", shares=100, price=100)
    assert bought is False, "should refuse to buy without enough cash"

    # Try to oversell - should be refused, holdings unchanged
    sold = p.sell("TEST.NS", shares=10, price=100)
    assert sold is False, "should refuse to sell more than held"

    # Log a hold decision
    p.hold("TEST.NS")

    # Final value check: 920 cash + 1 remaining share at current price 130
    value = p.get_value({"TEST.NS": 130})
    print(f"Final portfolio value at price 130: {value}")
    assert value == 920 + 130, "portfolio value should be cash + holdings value"

    p.save_log()

    print("\nAll checks passed - portfolio.py math is correct.")


if __name__ == "__main__":
    main()
