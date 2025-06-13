from zoneinfo import ZoneInfo

import yfinance as yf
from datetime import datetime


def calculate_stock_earnings(symbol, purchase_price, shares, purchase_date, target_date):
    # Convert strings to datetime
    purchase_date = datetime.strptime(purchase_date, '%Y-%m-%d')
    target_date = datetime.strptime(target_date, '%Y-%m-%d')
    purchase_date = purchase_date.replace(tzinfo=ZoneInfo("America/New_York"))
    target_date = target_date.replace(tzinfo=ZoneInfo("America/New_York"))

    # Download historical data
    stock = yf.Ticker(symbol)

    # Get historical closing prices
    hist = stock.history(start=purchase_date.strftime('%Y-%m-%d'), end=target_date.strftime('%Y-%m-%d'))

    # Ensure data is available
    if hist.empty:
        raise ValueError("No stock price data found for given dates.")

    # Get the first and last available closing prices
    target_price = hist['Close'].iloc[-1]

    # Get dividend data
    dividends = stock.dividends.loc[purchase_date:target_date]
    total_dividends = dividends.sum() * shares

    # Calculate values
    initial_investment = purchase_price * shares
    final_value = target_price * shares
    total_earnings = final_value + total_dividends - initial_investment

    # Return result as dictionary
    return {
        'symbol': symbol,
        'shares': shares,
        'purchase_price': round(purchase_price, 2),
        'target_price': round(target_price, 2),
        'initial_investment': round(initial_investment, 2),
        'final_value': round(final_value, 2),
        'total_dividends': round(total_dividends, 2),
        'total_earnings': round(total_earnings, 2)
    }

def main():
    result = calculate_stock_earnings('RIO', 64.91, 220, '2024-07-16', '2025-06-12')
    print(result)

if __name__ == "__main__":
    main()
# Example usage:
# result = calculate_stock_earnings('AAPL', 10, '2022-01-01', '2023-01-01')
# print(result)
