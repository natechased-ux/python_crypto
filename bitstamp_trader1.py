import requests
import time
import hmac
import hashlib
import json
import pandas as pd
from datetime import datetime
import telegram

# === CONFIG ===
API_KEY = 'qJFV9q7knofXxNRyZexiZsjO0KoZm7GG'
API_SECRET = '2wNvNcxz7PLBog3Ah7G5T1W69i5CLpwC'
CUSTOMER_ID = 'bcdu2450'
SYMBOL = 'xrpusd'
TELEGRAM_TOKEN = '8177096945:AAHTg5nxVTA6hcPkAL4MLAAHPsZ1at7Ywmw'
TELEGRAM_CHAT_ID = '7967738614'
VWAP_LENGTH = 14
IMBALANCE_THRESHOLD = 10.0
POSITION_SIZE = 20  # Size of the trade in USD
LEVERAGE = 5  # Leverage for futures trading

# === TELEGRAM ALERTS ===
def send_telegram_alert(message):
    bot = telegram.Bot(token=TELEGRAM_TOKEN)
    bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=message)

# === BITSTAMP AUTH ===
def create_signature(nonce, endpoint):
    message = f"{nonce}{CUSTOMER_ID}{API_KEY}"
    signature = hmac.new(API_SECRET.encode(), msg=message.encode(), digestmod=hashlib.sha256).hexdigest()
    return signature.upper()

def place_order(side, amount, price=None, order_type="market"):
    url = "https://www.bitstamp.net/api/v2/{}/".format("buy" if side == "buy" else "sell")
    nonce = str(int(time.time() * 1000))
    headers = {
        "Content-Type": "application/x-www-form-urlencoded"
    }
    payload = {
        "key": API_KEY,
        "signature": create_signature(nonce, url),
        "nonce": nonce,
        "amount": amount,
        "price": price,
        "type": "0" if order_type == "market" else "1"
    }
    response = requests.post(url, data=payload, headers=headers)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Order failed: {response.text}")
        return None

# === FETCH DATA ===
def fetch_bitstamp_data(symbol, interval='60', limit=100):
    url = f"https://www.bitstamp.net/api/v2/ohlc/{symbol}/"
    params = {'step': int(interval) * 60, 'limit': limit}
    response = requests.get(url, params=params)
    if response.status_code == 200:
        return response.json().get('data', {}).get('ohlc', [])
    else:
        print(f"Error fetching data: {response.status_code}")
        return []

# === INDICATORS ===
def calculate_vwap(data):
    data['VWAP'] = (data['close'] * data['volume']).cumsum() / data['volume'].cumsum()
    return data

def calculate_imbalance(data):
    data['BullVol'] = data.apply(lambda x: x['volume'] if x['close'] > x['open'] else 0, axis=1)
    data['BearVol'] = data.apply(lambda x: x['volume'] if x['close'] <= x['open'] else 0, axis=1)
    data['Imbalance'] = 100 * (data['BullVol'] - data['BearVol']) / data['volume']
    return data

# === SIGNAL CONDITIONS ===
def generate_signals(data):
    data['BuySignal'] = (data['Imbalance'] > IMBALANCE_THRESHOLD) & (data['close'] > data['VWAP'])
    data['SellSignal'] = (data['Imbalance'] < -IMBALANCE_THRESHOLD) & (data['close'] < data['VWAP'])
    return data

# === MAIN LOGIC ===
def main():
    raw_data = fetch_bitstamp_data(SYMBOL)
    if not raw_data:
        print("No data fetched.")
        return
    
    df = pd.DataFrame(raw_data)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    df = df.astype({'open': 'float', 'high': 'float', 'low': 'float', 'close': 'float', 'volume': 'float'})
    
    # Calculate Indicators
    df = calculate_vwap(df)
    df = calculate_imbalance(df)
    df = generate_signals(df)

    # Check for Buy/Sell Signals
    for index, row in df.iterrows():
        if row['BuySignal']:
            send_telegram_alert(f"BUY Signal at {row['timestamp']} - Price: {row['close']}")
            place_order("buy", POSITION_SIZE / row['close'])
        elif row['SellSignal']:
            send_telegram_alert(f"SELL Signal at {row['timestamp']} - Price: {row['close']}")
            place_order("sell", POSITION_SIZE / row['close'])

if __name__ == "__main__":
    while True:
        main()
        time.sleep(60)  # Check for signals every minute
