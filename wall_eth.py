import requests
import pandas as pd
import time
from datetime import datetime

# === CONFIG ===
SYMBOL = "ethusd"
CANDLE_URL = f"https://www.bitstamp.net/api/v2/ohlc/{SYMBOL}/"
VWAP_PERIOD = 20  # Period in minutes
ALERT_THRESHOLD = 0.5  # Price wall detection threshold
VWAP_ALERT_THRESHOLD = 0.5  # VWAP distance threshold for alerts

# === FETCH VWAP ===
def fetch_vwap_data(period):
    """
    Fetch VWAP by aggregating smaller intervals if needed.
    """
    try:
        # Adjust period to nearest supported step
        if period <= 5:
            step = 60
        elif period <= 15:
            step = 300
        elif period <= 30:
            step = 900
        elif period <= 60:
            step = 1800
        else:
            step = 3600
        
        limit = (period * 60) // step  # Determine how many candles are needed
        response = requests.get(CANDLE_URL, params={"step": step, "limit": limit})
        response.raise_for_status()
        data = response.json()
        
        if "data" in data and "ohlc" in data["data"]:
            candles = data["data"]["ohlc"]
            df = pd.DataFrame(candles)
            df['price'] = df[['high', 'low', 'close']].astype(float).mean(axis=1)
            df['volume'] = df['volume'].astype(float)
            vwap = (df['price'] * df['volume']).sum() / df['volume'].sum()
            return vwap
        return None
    except Exception as e:
        print(f"Error fetching VWAP: {e}")
        return None

# === MAIN LOGIC ===
def monitor_market():
    """
    Monitors the market and identifies buy/sell signals based on VWAP and order book walls.
    """
    while True:
        # Fetch VWAP
        vwap = fetch_vwap_data(VWAP_PERIOD)
        if vwap is None:
            print("VWAP could not be fetched. Retrying...")
            time.sleep(60)
            continue
        
        # Simulate fetching current price and order book data (replace with real API call)
        current_price = 2000.00  # Placeholder for current price
        bid_wall = 1950.00  # Placeholder for strongest buy wall
        ask_wall = 2050.00  # Placeholder for strongest sell wall
        
        # Evaluate alerts
        if abs(current_price - vwap) / vwap > VWAP_ALERT_THRESHOLD:
            if current_price < vwap and current_price < bid_wall:
                print(f"📈 BUY SIGNAL - Current Price: {current_price}, VWAP: {vwap:.2f}, Bid Wall: {bid_wall}")
            elif current_price > vwap and current_price > ask_wall:
                print(f"📉 SELL SIGNAL - Current Price: {current_price}, VWAP: {vwap:.2f}, Ask Wall: {ask_wall}")
        
        time.sleep(60)  # Wait for 1 minute before the next check

# === ENTRY POINT ===
if __name__ == "__main__":
    monitor_market()
