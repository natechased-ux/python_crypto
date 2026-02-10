import pandas as pd
import requests
import numpy as np
from ta.volatility import AverageTrueRange

# Configuration Parameters
symbols = ["btc-usd", "eth-usd", "xrp-usd", "ltc-usd", "ada-usd",
           "doge-usd", "sol-usd", "wif-usd", "ondo-usd",
           "sei-usd", "magic-usd", "ape-usd", "jasmy-usd", "wld-usd",
           "syrup-usd","fartcoin-usd", "aero-usd", "link-usd","hbar-usd",
           "aave-usd", "fet-usd", "crv-usd","tao-usd", "avax-usd", "xcn-usd",
           "uni-usd", "mkr-usd", "toshi-usd", "near-usd", "algo-usd", "trump-usd",
           "bch-usd", "inj-usd", "pepe-usd", "xlm-usd", "moodeng-usd", "bonk-usd",
           "dot-usd", "render-usd", "popcat-usd", "arb-usd", "icp-usd",
           "qnt-usd", "tia-usd", "ip-usd" , "pnut-usd", "apt-usd", "vet-usd",
           "ena-usd", "turbo-usd", "bera-usd", "pol-usd", "mask-usd", "ach-usd",
           "pyth-usd", "sand-usd", "morpho-usd", "mana-usd", "velo-usd", "coti-usd",
           "axs-usd"]  # Test with a smaller set
vwap_length = 14
atr_length = 14
risk_reward_ratio = 2.0
price_range_percentage = 10

# Dynamic Whale Threshold by Coin
def get_dynamic_whale_threshold(symbol):
    high_value_coins = ["btc-usd", "eth-usd"]
    mid_value_coins = ["xrp-usd", "ltc-usd", "ada-usd", "sol-usd", "link-usd", "dot-usd", "qnt-usd", "avax-usd", "near-usd", "uni-usd", "algo-usd"]

    low_value_coins = ["doge-usd", "sei-usd", "magic-usd", "ape-usd", "jasmy-usd", "wld-usd", "hbar-usd", "crv-usd", "inj-usd", 
 "pepe-usd", "xlm-usd", "arb-usd", "icp-usd", "mask-usd", "sand-usd", "mana-usd", "axs-usd", "morpho-usd", "coti-usd", 
 "vela-usd", "pyth-usd", "ach-usd", "turbo-usd", "bera-usd", "ena-usd", "vet-usd", "apt-usd", "pnut-usd", "ip-usd", 
 "tia-usd", "popcat-usd", "render-usd", "moodeng-usd", "bonk-usd", "toshi-usd", "trump-usd", "syrup-usd", "fartcoin-usd", 
 "aero-usd"]


    if symbol in high_value_coins:
        return 10.0  # Higher threshold for large-cap coins
    elif symbol in mid_value_coins:
        return 5.0  # Mid-level threshold
    elif symbol in low_value_coins:
        return 1.0  # Lower threshold for small-cap coins
    else:
        return 2.0  # Default threshold

# Helper Functions
def group_close_prices(orders, threshold):
    if not orders:
        return []
    orders.sort(key=lambda x: x[0])
    grouped, current = [], [orders[0]]
    for price, qty in orders[1:]:
        prev_price = current[-1][0]
        if abs(price - prev_price) / prev_price <= threshold:
            current.append((price, qty))
        else:
            total_qty = sum(q for _, q in current)
            avg_price = sum(p * q for p, q in current) / total_qty
            grouped.append((avg_price, total_qty))
            current = [(price, qty)]
    total_qty = sum(q for _, q in current)
    avg_price = sum(p * q for p, q in current) / total_qty
    grouped.append((avg_price, total_qty))
    return grouped

def calculate_vwap(data):
    data['typical_price'] = (data['high'] + data['low'] + data['close']) / 3
    data['vwap'] = (data['typical_price'] * data['volume']).rolling(vwap_length).sum() / data['volume'].rolling(vwap_length).sum()
    return data

def calculate_atr(data):
    atr_indicator = AverageTrueRange(high=data['high'], low=data['low'], close=data['close'], window=atr_length)
    data['atr'] = atr_indicator.average_true_range()
    return data

def analyze_whale_orders(symbol, mid_price):
    whale_threshold = get_dynamic_whale_threshold(symbol)
    url = f"https://api.exchange.coinbase.com/products/{symbol}/book?level=2"
    response = requests.get(url)
    if response.status_code == 200:
        order_book = response.json()
        bids = [(float(bid[0]), float(bid[1])) for bid in order_book['bids']
                if float(bid[1]) >= whale_threshold]
        asks = [(float(ask[0]), float(ask[1])) for ask in order_book['asks']
                if float(ask[1]) >= whale_threshold]

        grouped_bids = group_close_prices(bids, 0.001)
        grouped_asks = group_close_prices(asks, 0.001)

        # Filter bids and asks to be at least 1% away from the mid-price
        filtered_bids = [bid for bid in grouped_bids if abs(mid_price - bid[0]) / mid_price >= 0.01]
        filtered_asks = [ask for ask in grouped_asks if abs(mid_price - ask[0]) / mid_price >= 0.01]

        nearest_bid = min(filtered_bids, key=lambda x: abs(mid_price - x[0]), default=None)
        nearest_ask = min(filtered_asks, key=lambda x: abs(mid_price - x[0]), default=None)

        return nearest_bid, nearest_ask
    else:
        return None, None

def fetch_historical_data(symbol):
    url = f"https://api.exchange.coinbase.com/products/{symbol}/candles?granularity=3600"
    response = requests.get(url)
    if response.status_code == 200:
        columns = ["time", "low", "high", "open", "close", "volume"]
        data = pd.DataFrame(response.json(), columns=columns)
        data = data.sort_values(by="time")
        return data
    else:
        raise Exception(f"Failed to fetch data for {symbol}: {response.status_code}")

def backtest(symbol, data):
    data = calculate_vwap(data)
    data = calculate_atr(data)
    
    data['long_signal'] = False
    data['short_signal'] = False
    
    for i in range(len(data)):
        if i < max(vwap_length, atr_length):
            continue

        latest = data.iloc[i]
        vwap, atr = latest['vwap'], latest['atr']
        close_price = latest['close']

        nearest_bid, nearest_ask = analyze_whale_orders(symbol, close_price)

        if nearest_bid and close_price <= nearest_bid[0]:
            data.at[i, 'long_signal'] = True
        if nearest_ask and close_price >= nearest_ask[0]:
            data.at[i, 'short_signal'] = True

    long_trades = data[data['long_signal']]
    short_trades = data[data['short_signal']]

    return long_trades, short_trades

if __name__ == "__main__":
    results = {}
    for symbol in symbols:
        try:
            data = fetch_historical_data(symbol)
            long_trades, short_trades = backtest(symbol, data)
            results[symbol] = {
                "longs": len(long_trades),
                "shorts": len(short_trades)
            }
            print(f"{symbol.upper()} - Long Signals: {len(long_trades)}, Short Signals: {len(short_trades)}")
        except Exception as e:
            print(f"Error with {symbol}: {e}")
