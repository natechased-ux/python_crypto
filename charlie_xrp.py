import requests
import time
import pandas as pd

# === CONFIG ===
SYMBOL = 'xrpusd'
ORDER_BOOK_URL = f'https://www.bitstamp.net/api/v2/order_book/{SYMBOL}/'
PRICE_DATA_URL = f'https://www.bitstamp.net/api/v2/ticker/{SYMBOL}/'
VWAP_PERIOD = 20  # Number of periods for VWAP calculation
IMBALANCE_THRESHOLD_BUY = 70  # % imbalance for BUY signal
IMBALANCE_THRESHOLD_SELL = 30  # % imbalance for SELL signal
ALERT_INTERVAL = 300  # Minimum seconds between alerts

# Telegram Config
TELEGRAM_TOKEN = '8177096945:AAHTg5nxVTA6hcPkAL4MLAAHPsZ1at7Ywmw'
TELEGRAM_CHAT_ID = '7967738614'

price_volume_data = pd.DataFrame(columns=['price', 'volume'])
last_alert_time = 0  # Timestamp of the last alert


def fetch_order_book():
    """Fetch order book data from Bitstamp."""
    try:
        response = requests.get(ORDER_BOOK_URL)
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Error fetching order book: {response.status_code}")
            return None
    except Exception as e:
        print(f"Exception fetching order book: {e}")
        return None


def fetch_price_data():
    """Fetch current price and volume data from Bitstamp."""
    try:
        response = requests.get(PRICE_DATA_URL)
        if response.status_code == 200:
            data = response.json()
            return {
                'price': float(data['last']),
                'volume': float(data['volume']),
            }
        else:
            print(f"Error fetching price data: {response.status_code}")
            return None
    except Exception as e:
        print(f"Exception fetching price data: {e}")
        return None


def calculate_imbalance(order_book):
    """Calculate order book imbalance as a percentage."""
    bids = order_book['bids'][:10]
    asks = order_book['asks'][:10]

    bid_volume = sum(float(qty) for price, qty in bids)
    ask_volume = sum(float(qty) for price, qty in asks)

    if bid_volume + ask_volume == 0:
        return 50  # Neutral imbalance

    return (bid_volume / (bid_volume + ask_volume)) * 100


def calculate_vwap(data):
    """Calculate VWAP based on historical price and volume."""
    data['price_volume'] = data['price'] * data['volume']
    cumulative_price_volume = data['price_volume'].rolling(VWAP_PERIOD).sum()
    cumulative_volume = data['volume'].rolling(VWAP_PERIOD).sum()
    vwap = cumulative_price_volume / cumulative_volume
    return vwap.iloc[-1]


def send_telegram_alert(message):
    """Send alert message to Telegram."""
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message}
    try:
        requests.post(url, json=payload)
    except Exception as e:
        print(f"Telegram Error: {e}")


def main():
    global price_volume_data, last_alert_time

    while True:
        order_book = fetch_order_book()
        price_data = fetch_price_data()

        if not order_book or not price_data:
            time.sleep(10)  # Retry after 10 seconds if data is unavailable
            continue

        # Update price and volume data
        price_volume_data = pd.concat([
            price_volume_data,
            pd.DataFrame([price_data])
        ]).tail(100)  # Keep only the last 100 records

        # Calculate VWAP
        if len(price_volume_data) >= VWAP_PERIOD:
            vwap = calculate_vwap(price_volume_data)
        else:
            vwap = None

        # Calculate order book imbalance
        imbalance = calculate_imbalance(order_book)

        # Check for buy/sell signals
        current_time = time.time()
        if vwap and current_time - last_alert_time >= ALERT_INTERVAL:
            if imbalance > IMBALANCE_THRESHOLD_BUY and price_data['price'] > vwap:
                message = (f"🚀 Strong BUY Signal\n"
                           f"Price: {price_data['price']:.2f}\n"
                           f"Imbalance: {imbalance:.2f}%\n"
                           f"VWAP: {vwap:.2f}")
                print(message)
                send_telegram_alert(message)
                last_alert_time = current_time

            elif imbalance < IMBALANCE_THRESHOLD_SELL and price_data['price'] < vwap:
                message = (f"⚠️ Strong SELL Signal\n"
                           f"Price: {price_data['price']:.2f}\n"
                           f"Imbalance: {imbalance:.2f}%\n"
                           f"VWAP: {vwap:.2f}")
                print(message)
                send_telegram_alert(message)
                last_alert_time = current_time

        time.sleep(10)  # Fetch data every 10 seconds


if __name__ == "__main__":
    main()
