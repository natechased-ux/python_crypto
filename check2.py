import asyncio
import requests
import numpy as np
import pandas as pd
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
           "axs-usd"]
price_range_percentage = 10
vwap_length = 14
atr_length = 14
risk_reward_ratio = 2.0
position_size = 1000  # Fixed position size for trades

def calculate_vwap(data):
    data['typical_price'] = (data['high'] + data['low'] + data['close']) / 3
    data['vwap'] = (data['typical_price'] * data['volume']).rolling(vwap_length).sum() / data['volume'].rolling(vwap_length).sum()
    return data

def calculate_atr(data):
    atr_indicator = AverageTrueRange(high=data['high'], low=data['low'], close=data['close'], window=atr_length)
    data['atr'] = atr_indicator.average_true_range()
    return data

def simulate_trades(symbol, data):
    try:
        # Calculate VWAP and ATR
        data = calculate_vwap(data)
        data = calculate_atr(data)

        profits = []
        num_signals = 0
        for i in range(len(data) - 1):
            row = data.iloc[i]
            next_rows = data.iloc[i + 1:]

            # Signal Logic
            if row['close'] > row['vwap']:  # Example Long Signal
                entry_price = row['close']
                tp = entry_price + (row['atr'] * risk_reward_ratio)
                sl = entry_price - row['atr']

                # Determine Exit Price
                exit_price = None
                for _, next_row in next_rows.iterrows():
                    if next_row['high'] >= tp:
                        exit_price = tp
                        break
                    if next_row['low'] <= sl:
                        exit_price = sl
                        break

                if exit_price is None:
                    exit_price = next_rows.iloc[-1]['close']

                # Calculate Profit
                profit = position_size * ((exit_price - entry_price) / entry_price)
                profits.append(profit)
                num_signals += 1

            elif row['close'] < row['vwap']:  # Example Short Signal
                entry_price = row['close']
                tp = entry_price - (row['atr'] * risk_reward_ratio)
                sl = entry_price + row['atr']

                # Determine Exit Price
                exit_price = None
                for _, next_row in next_rows.iterrows():
                    if next_row['low'] <= tp:
                        exit_price = tp
                        break
                    if next_row['high'] >= sl:
                        exit_price = sl
                        break

                if exit_price is None:
                    exit_price = next_rows.iloc[-1]['close']

                # Calculate Profit
                profit = position_size * ((entry_price - exit_price) / entry_price)
                profits.append(profit)
                num_signals += 1

        total_profit = sum(profits)
        avg_profit = total_profit / num_signals if num_signals > 0 else 0

        return {
            'symbol': symbol,
            'num_signals': num_signals,
            'total_profit': total_profit,
            'avg_profit': avg_profit,
        }

    except Exception as e:
        return {
            'symbol': symbol,
            'error': str(e)
        }

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

def evaluate_performance():
    results = []
    for symbol in symbols:
        try:
            data = fetch_historical_data(symbol)
            result = simulate_trades(symbol, data)
            results.append(result)
        except Exception as e:
            results.append({'symbol': symbol, 'error': str(e)})

    for result in results:
        if 'error' in result:
            print(f"{result['symbol']}: Error - {result['error']}")
        else:
            print(f"{result['symbol']}: Total Profit: ${result['total_profit']:.2f}, Avg Profit: ${result['avg_profit']:.2f}, Signals: {result['num_signals']}")

if __name__ == "__main__":
    evaluate_performance()
