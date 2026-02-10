import requests
import time
import hmac
import hashlib
import json

# Bitstamp API credentials
API_KEY = 'qJFV9q7knofXxNRyZexiZsjO0KoZm7GG'
API_SECRET = '2wNvNcxz7PLBog3Ah7G5T1W69i5CLpwC'
CUSTOMER_ID = 'bcdu2450'

# Constants
SYMBOL = 'xrpusd'  # XRP/USD pair
TRADE_SIZE = 50  # USD amount per trade
LEVERAGE = 5
VWAP_PERIOD = 15  # Adjust the period for VWAP calculation

# Helper functions
def create_signature(nonce):
    message = nonce + BITSTAMP_API_CLIENT_ID + BITSTAMP_API_KEY
    return hmac.new(
        BITSTAMP_API_SECRET.encode(),
        msg=message.encode(),
        digestmod=hashlib.sha256
    ).hexdigest()

def make_request(endpoint, method='GET', params=None, data=None):
    url = f"https://www.bitstamp.net/api/v2/{endpoint}"
    nonce = str(int(time.time() * 1000))
    headers = {
        'Content-Type': 'application/x-www-form-urlencoded',
        'X-Auth': BITSTAMP_API_KEY,
        'X-Auth-Signature': create_signature(nonce),
        'X-Auth-Nonce': nonce,
    }
    if method == 'GET':
        response = requests.get(url, params=params, headers=headers)
    else:
        response = requests.post(url, data=data, headers=headers)
    return response.json()

def fetch_order_book():
    return make_request(f'order_book/{SYMBOL}')

def fetch_ticker():
    return make_request(f'ticker/{SYMBOL}')

def place_order(order_type, amount, price=None):
    params = {
        'amount': amount,
        'price': price,
        'type': order_type,  # 0 for buy, 1 for sell
        'symbol': SYMBOL,
    }
    return make_request('buy/' if order_type == 'buy' else 'sell/', method='POST', data=params)

# Strategy logic
def calculate_vwap(order_book):
    bids = order_book['bids']
    asks = order_book['asks']
    bid_vwap = sum(float(bid[0]) * float(bid[1]) for bid in bids[:VWAP_PERIOD]) / sum(float(bid[1]) for bid in bids[:VWAP_PERIOD])
    ask_vwap = sum(float(ask[0]) * float(ask[1]) for ask in asks[:VWAP_PERIOD]) / sum(float(ask[1]) for ask in asks[:VWAP_PERIOD])
    return bid_vwap, ask_vwap

def trading_loop():
    position = None
    entry_price = 0
    while True:
        try:
            # Fetch market data
            order_book = fetch_order_book()
            ticker = fetch_ticker()

            # Calculate VWAP
            bid_vwap, ask_vwap = calculate_vwap(order_book)
            current_price = float(ticker['last'])

            # Trading logic
            if current_price > ask_vwap and position != 'long':
                print(f"Placing buy order at {current_price}")
                place_order('buy', TRADE_SIZE / current_price)
                position = 'long'
                entry_price = current_price
            elif current_price < bid_vwap and position == 'long':
                print(f"Placing sell order at {current_price}")
                place_order('sell', TRADE_SIZE / current_price)
                position = None

            # Wait before the next iteration
            time.sleep(5)  # Adjust as needed

        except Exception as e:
            print(f"Error: {e}")
            time.sleep(10)

# Start trading
if __name__ == '__main__':
    trading_loop()
