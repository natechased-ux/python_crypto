import requests
import pandas as pd
import numpy as np
from ta.volatility import AverageTrueRange
from datetime import datetime

# Config
symbols = ["btc-usd", "eth-usd", "xrp-usd", "ltc-usd", "ada-usd",
           "doge-usd", "sol-usd", "wif-usd", "ondo-usd",
           "sei-usd", "magic-usd", "ape-usd", "jasmy-usd", "wld-usd",
           "syrup-usd", "fartcoin-usd", "aero-usd", "link-usd", "hbar-usd",
           "aave-usd", "fet-usd", "crv-usd", "tao-usd", "avax-usd", "xcn-usd",
           "uni-usd", "mkr-usd", "toshi-usd", "near-usd", "algo-usd", "trump-usd",
           "bch-usd", "inj-usd", "pepe-usd", "xlm-usd", "moodeng-usd", "bonk-usd",
           "dot-usd", "popcat-usd", "arb-usd", "icp-usd",
           "qnt-usd", "tia-usd", "ip-usd", "pnut-usd", "apt-usd", "vet-usd",
           "ena-usd", "turbo-usd", "bera-usd", "pol-usd", "mask-usd", "ach-usd",
           "pyth-usd", "sand-usd", "morpho-usd", "mana-usd", "velo-usd", "coti-usd",
           "axs-usd"]

granularity = 3600  # 1 hour
num_candles = 300
atr_length = 14
capital_per_trade = 2500
tp_mult = 2.0
sl_mult = 1.0
max_hold = 6  # bars

# Indicators
def calculate_atr(data):
    atr_indicator = AverageTrueRange(high=data['high'], low=data['low'], close=data['close'], window=atr_length)
    data['atr'] = atr_indicator.average_true_range()
    return data

def calculate_cvd(data):
    data['vol_delta'] = data['volume'] * np.sign(data['close'] - data['open'])
    data['cvd'] = data['vol_delta'].cumsum()
    return data

# Peaks & Troughs
def get_peaks(series, window=3):
    return [i for i in range(window, len(series)-window) if series[i] == max(series[i-window:i+window+1])]

def get_troughs(series, window=3):
    return [i for i in range(window, len(series)-window) if series[i] == min(series[i-window:i+window+1])]

# Divergences
def detect_divergences(data):
    price = data['close'].values
    cvd = data['cvd'].values
    price_highs = get_peaks(price)
    price_lows = get_troughs(price)
    cvd_highs = get_peaks(cvd)
    cvd_lows = get_troughs(cvd)

    divergences = []

    # Bearish
    for i in range(len(price_highs) - 1):
        ph1, ph2 = price_highs[i], price_highs[i + 1]
        if ph2 <= ph1:
            continue
        ch1 = min(cvd_highs, key=lambda x: abs(x - ph1)) if cvd_highs else None
        ch2 = min(cvd_highs, key=lambda x: abs(x - ph2)) if cvd_highs else None
        if ch1 is None or ch2 is None:
            continue
        if cvd[ch2] < cvd[ch1]:
            divergences.append((ph2, 'bearish'))

    # Bullish
    for i in range(len(price_lows) - 1):
        pl1, pl2 = price_lows[i], price_lows[i + 1]
        if price[pl2] >= price[pl1]:
            continue
        if pl1 in cvd_lows and pl2 in cvd_lows:
            if cvd[pl2] > cvd[pl1]:
                divergences.append((pl2, 'bullish'))

    return sorted(divergences, key=lambda x: x[0])

# Fetch data
def fetch_historical_data(symbol):
    url = f"https://api.exchange.coinbase.com/products/{symbol}/candles?granularity={granularity}"
    response = requests.get(url)
    if response.status_code == 200:
        columns = ["time", "low", "high", "open", "close", "volume"]
        data = pd.DataFrame(response.json(), columns=columns)
        data = data.sort_values(by="time").reset_index(drop=True)
        return data.head(num_candles)
    else:
        return None

# Simulate trade
def simulate_trade(data, idx, div_type):
    entry_price = data.loc[idx, 'close']
    atr = data.loc[idx, 'atr']
    if np.isnan(atr) or atr == 0:
        return None

    size = capital_per_trade / entry_price
    tp = entry_price + atr * tp_mult if div_type == 'bullish' else entry_price - atr * tp_mult
    sl = entry_price - atr * sl_mult if div_type == 'bullish' else entry_price + atr * sl_mult

    for i in range(1, max_hold + 1):
        if idx + i >= len(data):
            break
        high = data.loc[idx + i, 'high']
        low = data.loc[idx + i, 'low']

        if div_type == 'bullish':
            if low <= sl:
                return -atr * sl_mult * size
            if high >= tp:
                return atr * tp_mult * size
        else:
            if high >= sl:
                return -atr * sl_mult * size
            if low <= tp:
                return atr * tp_mult * size

    # Exit at last close
    exit_price = data.loc[min(idx + max_hold, len(data)-1), 'close']
    return (exit_price - entry_price) * size if div_type == 'bullish' else (entry_price - exit_price) * size

# Backtest Runner
def backtest():
    total_profit = 0
    total_trades = 0
    total_wins = 0
    total_losses = 0
    all_trades = []

    for symbol in symbols:
        data = fetch_historical_data(symbol)
        if data is None or len(data) < 50:
            print(f"⚠️ Skipping {symbol.upper()} (insufficient data)")
            continue

        data = calculate_atr(data)
        data = calculate_cvd(data)
        divergences = detect_divergences(data)

        profits = []
        wins = 0
        losses = 0
        timestamps = []

        for idx, div_type in divergences:
            profit = simulate_trade(data, idx, div_type)
            if profit is not None:
                profits.append(profit)
                timestamp = data.loc[idx, 'time']
                timestamps.append(timestamp)
                all_trades.append({
                    "symbol": symbol.upper(),
                    "timestamp": timestamp,
                    "direction": div_type,
                    "profit": round(profit, 2)
                })
                if profit > 0:
                    wins += 1
                else:
                    losses += 1

        if profits:
            total = sum(profits)
            avg = total / len(profits)
            win_rate = (wins / len(profits)) * 100
            print(f"\n📊 {symbol.upper()} — Trades: {len(profits)} | Wins: {wins} | Losses: {losses} | Win Rate: {win_rate:.1f}% | Total: ${total:.2f} | Avg: ${avg:.2f}")
            
            # Print last 10 timestamps
            print("🕒 10 Most Recent Trade Timestamps:")
            for t in timestamps[-10:]:
                print(f" - {datetime.utcfromtimestamp(t).strftime('%Y-%m-%d %H:%M:%S UTC')}")
                
            total_profit += total
            total_trades += len(profits)
            total_wins += wins
            total_losses += losses
        else:
            print(f"\n📊 {symbol.upper()} — No valid trades")

    # Final summary
    if total_trades > 0:
        final_avg = total_profit / total_trades
        win_rate = (total_wins / total_trades) * 100
        print("\n📈 TOTAL SUMMARY — All Coins")
        print(f"Trades: {total_trades} | Wins: {total_wins} | Losses: {total_losses} | Win Rate: {win_rate:.1f}% | Net Profit: ${total_profit:.2f} | Avg/Trade: ${final_avg:.2f}")
    else:
        print("\n📈 TOTAL SUMMARY — No trades found")

    # Show 10 most recent trades globally
    if all_trades:
        print("\n🧾 10 Most Recent Trades Across All Coins:")
        sorted_trades = sorted(all_trades, key=lambda x: x['timestamp'], reverse=True)[:100]
        for trade in sorted_trades:
            ts = datetime.utcfromtimestamp(trade['timestamp']).strftime('%Y-%m-%d %H:%M:%S UTC')
            print(f" - [{ts}] {trade['symbol']} | {trade['direction'].capitalize():7} | Profit: ${trade['profit']:.2f}")

# Run it
if __name__ == "__main__":
    backtest()
