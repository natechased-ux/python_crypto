
import pandas as pd
import requests
import numpy as np
from datetime import datetime

# Configuration
COINS = [
    "btc-usd", "eth-usd", "xrp-usd", "ltc-usd", "ada-usd", "doge-usd", "sol-usd", "wif-usd", "ondo-usd",
    "sei-usd", "magic-usd", "ape-usd", "jasmy-usd", "wld-usd", "syrup-usd", "fartcoin-usd", "aero-usd",
    "link-usd", "hbar-usd", "aave-usd", "fet-usd", "crv-usd", "tao-usd", "avax-usd", "xcn-usd", "uni-usd",
    "mkr-usd", "toshi-usd", "near-usd", "algo-usd", "trump-usd", "bch-usd", "inj-usd", "pepe-usd", "xlm-usd",
    "moodeng-usd", "bonk-usd", "dot-usd", "popcat-usd", "arb-usd", "icp-usd", "qnt-usd", "tia-usd", "ip-usd",
    "pnut-usd", "apt-usd", "vet-usd", "ena-usd", "turbo-usd", "bera-usd", "pol-usd", "mask-usd", "ach-usd",
    "pyth-usd", "sand-usd", "morpho-usd", "mana-usd", "velo-usd", "coti-usd", "axs-usd"
]
INTERVALS = {'1D': 86400, '4H': 14400, '1H': 3600}
BASE_URL = "https://api.exchange.coinbase.com/products"

# Indicators
def get_candles(symbol, granularity, limit=300):
    url = f"{BASE_URL}/{symbol}/candles?granularity={granularity}"
    response = requests.get(url)
    if response.status_code != 200:
        return pd.DataFrame()
    data = response.json()
    df = pd.DataFrame(data, columns=['time','low','high','open','close','volume'])
    df['time'] = pd.to_datetime(df['time'], unit='s')
    return df.sort_values('time')

def ema(series, period=50):
    return series.ewm(span=period, adjust=False).mean()

def macd(df, fast=12, slow=26, signal=9):
    fast_ema = df['close'].ewm(span=fast, adjust=False).mean()
    slow_ema = df['close'].ewm(span=slow, adjust=False).mean()
    macd_line = fast_ema - slow_ema
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def rsi(df, period=14):
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def atr(df, period=14):
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()

# Backtest
def backtest_swing_trades():
    results = []
    for coin in COINS:
        try:
            df_d = get_candles(coin.upper(), INTERVALS['1D'])
            df_4h = get_candles(coin.upper(), INTERVALS['4H'])
            df_1h = get_candles(coin.upper(), INTERVALS['1H'])

            if df_d.empty or df_4h.empty or df_1h.empty:
                continue

            df_d['ema50'] = ema(df_d['close'], 50)
            macd_line, signal_line, _ = macd(df_4h)
            df_4h['macd'] = macd_line
            df_4h['signal'] = signal_line
            df_1h['rsi'] = rsi(df_1h)
            df_4h['rsi'] = rsi(df_4h)
            df_1h['atr'] = atr(df_1h)

            trades, wins, losses, profit = 0, 0, 0, 0

            for i in range(50, len(df_4h) - 13):
                date = df_4h['time'].iloc[i]
                price = df_4h['close'].iloc[i]
                trend_up = df_d['close'].iloc[-1] > df_d['ema200'].iloc[-1]
                trend_down = df_d['close'].iloc[-1] < df_d['ema200'].iloc[-1]
                macd_cross_up = df_4h['macd'].iloc[i-1] < df_4h['signal'].iloc[i-1] and df_4h['macd'].iloc[i] > df_4h['signal'].iloc[i]
                macd_cross_down = df_4h['macd'].iloc[i-1] > df_4h['signal'].iloc[i-1] and df_4h['macd'].iloc[i] < df_4h['signal'].iloc[i]
                rsi1h_val = df_1h[df_1h['time'] <= date]['rsi'].iloc[-1]
                rsi4h_val = df_4h['rsi'].iloc[i]
                atr_val = df_1h[df_1h['time'] <= date]['atr'].iloc[-1]

                if atr_val == 0 or np.isnan(atr_val):
                    continue

                if trend_up and macd_cross_up and rsi1h_val < 70 and rsi4h_val < 70:
                    entry = price
                    sl = entry - 1.5 * atr_val
                    tp = entry + 3 * atr_val
                    trades += 1
                    future = df_4h['close'].iloc[i+1:i+13]
                    hit_tp = any(p >= tp for p in future)
                    hit_sl = any(p <= sl for p in future)
                    if hit_tp and not hit_sl:
                        wins += 1
                        profit += 2500 * ((tp - entry) / entry)
                    elif hit_sl and not hit_tp:
                        losses += 1
                        profit += 2500 * ((sl - entry) / entry)
                    else:
                        trades -= 1

                if trend_down and macd_cross_down and rsi1h_val > 30 and rsi4h_val > 30:
                    entry = price
                    sl = entry + 1.5 * atr_val
                    tp = entry - 3 * atr_val
                    trades += 1
                    future = df_4h['close'].iloc[i+1:i+13]
                    hit_tp = any(p <= tp for p in future)
                    hit_sl = any(p >= sl for p in future)
                    if hit_tp and not hit_sl:
                        wins += 1
                        profit += 2500 * ((entry - tp) / entry)
                    elif hit_sl and not hit_tp:
                        losses += 1
                        profit += 2500 * ((entry - sl) / entry)
                    else:
                        trades -= 1

            if trades > 0:
                results.append({
                    "symbol": coin.upper(),
                    "trades": trades,
                    "wins": wins,
                    "losses": losses,
                    "win_rate": round(100 * wins / trades, 2),
                    "net_profit": round(profit, 2)
                })

        except Exception as e:
            print(f"Error with {coin.upper()}: {e}")

    df = pd.DataFrame(results)
    df.to_csv("swing_trade_backtest_results.csv", index=False)
    print(df)

if __name__ == "__main__":
    backtest_swing_trades()
