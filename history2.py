import requests
import pandas as pd
from datetime import datetime, timedelta

def fetch_coingecko_data(symbol, vs_currency, days):
    """
    Fetch historical price data from CoinGecko.

    Parameters:
        symbol (str): Cryptocurrency symbol (e.g., "ethereum").
        vs_currency (str): Fiat currency (e.g., "usd").
        days (int): Number of past days to fetch data for.

    Returns:
        pd.DataFrame: DataFrame with timestamp, price, and volume.
    """
    url = f"https://api.coingecko.com/api/v3/coins/{symbol}/market_chart"
    params = {"vs_currency": vs_currency, "days": days}
    response = requests.get(url, params=params)
    data = response.json()

    # Convert to DataFrame
    prices = pd.DataFrame(data["prices"], columns=["timestamp", "price"])
    prices["timestamp"] = pd.to_datetime(prices["timestamp"], unit="ms")
    return prices

# Fetch data for the past 2 days
symbol = "ethereum"
vs_currency = "usd"
days = 2

df = fetch_coingecko_data(symbol, vs_currency, days)
df.to_csv("eth_coingecko_data.csv", index=False)  # Save to file
print(df.head())
