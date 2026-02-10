import requests
import pandas as pd
from datetime import datetime, timedelta

# Binance API endpoint
BASE_URL = "https://api.binance.com/api/v3/klines"

def fetch_binance_historical_data(symbol, interval, lookback_hours):
    """
    Fetch historical candlestick data from Binance.
    
    Parameters:
        symbol (str): The trading pair symbol (e.g., ETHUSDT).
        interval (str): Time interval (e.g., '1m', '15m', '1h').
        lookback_hours (int): Number of hours to look back.
    
    Returns:
        pd.DataFrame: DataFrame with timestamp, open, high, low, close, volume.
    """
    # Calculate the start time (48 hours ago)
    end_time = int(datetime.utcnow().timestamp() * 1000)
    start_time = int((datetime.utcnow() - timedelta(hours=lookback_hours)).timestamp() * 1000)

    # Request parameters
    params = {
        "symbol": symbol,
        "interval": interval,
        "startTime": start_time,
        "endTime": end_time
    }

    # Make the API call
    response = requests.get(BASE_URL, params=params)
    print("Response Status Code:", response.status_code)
    print("Response Data", response.text)
    data = response.json()

    # Convert the data to a DataFrame
    df = pd.DataFrame(data, columns=[
        "timestamp", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "number_of_trades",
        "taker_buy_base_volume", "taker_buy_quote_volume", "ignore"
    ])

    # Keep only relevant columns and convert to numeric
    df = df[["timestamp", "open", "high", "low", "close", "volume"]]
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit='ms')
    df[["open", "high", "low", "close", "volume"]] = df[["open", "high", "low", "close", "volume"]].apply(pd.to_numeric)

    return df

# Fetch data
if __name__ == "__main__":
    symbol = "ETHUSDT"  # ETH/USD trading pair
    interval = "1m"     # 1-minute interval
    lookback_hours = 48  # Past 48 hours

    df = fetch_binance_historical_data(symbol, interval, lookback_hours)

    # Save to CSV (optional)
    df.to_csv("eth_historical_data.csv", index=False)

    print(df.head())
