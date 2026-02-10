import requests
import pandas as pd
from datetime import datetime, timedelta

# Configuration
BASE_URL = "https://www.bitstamp.net/api/v2/transactions/xrpusd/"
OUTPUT_FILE = "xrp_bitstamp_data.csv"
DAYS_BACK = 1  # Number of days of historical data to fetch

# Fetch data function
def fetch_trade_data(days_back=1):
    end_time = datetime.utcnow()
    start_time = end_time - timedelta(days=days_back)
    params = {
        "time": "day",  # Fetch trades for the last day
    }

    response = requests.get(BASE_URL, params=params)
    if response.status_code == 200:
        trades = response.json()
        trades_df = pd.DataFrame(trades)
        # Convert timestamps to readable dates
        trades_df['date'] = pd.to_datetime(trades_df['date'], unit='s')
        trades_df = trades_df.rename(columns={"price": "Price", "amount": "Volume", "date": "Timestamp"})
        trades_df = trades_df[["Timestamp", "Price", "Volume"]]
        trades_df.to_csv(OUTPUT_FILE, index=False)
        print(f"Data saved to {OUTPUT_FILE}")
    else:
        print(f"Failed to fetch data. Status code: {response.status_code}")

# Run the function
fetch_trade_data(DAYS_BACK)
