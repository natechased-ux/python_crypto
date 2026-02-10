import requests
import pandas as pd
import numpy as np
from ta.momentum import StochRSIIndicator
from ta.trend import EMAIndicator

# === CONFIG ===
COINS = [
    'BTC-USD', 'ETH-USD', 'XRP-USD',
    'SOL-USD', 'ADA-USD', 'AVAX-USD',
    'DOGE-USD', 'LINK-USD', 'DOT-USD',
    'MATIC-USD', 'LTC-USD', 'BCH-USD',
    'ATOM-USD', 'FIL-USD', 'ICP-USD',
    'AAVE-USD', 'NEAR-USD', 'ALGO-USD',
    'XLM-USD', 'EOS-USD'
]
GOLDEN_ZONE_MARGIN = 0.5  # ±1%
START_AT = 200
SL_BUFFER = 0.005
LOOKBACK_HOURS = 168

def get_candles(symbol, granularity):
    url = f'https://api.exchange.coinbase.com/products/{symbol}/candles?granularity={granularity}'
    r = requests.get(url)
    if r.status_code != 200:
        print(f"Failed to fetch {symbol} candles.")
        return None
    df = pd.DataFrame(r.json(), columns=['time', 'low', 'high', 'open', 'close', 'volume'])
    df['time'] = pd.to_datetime(df['time'], unit='s')
    return df.sort_values('time')

def calc_fibs(high, low):
    diff = high - low
    return {
        '0.618': high - diff * 0.618,
        '0.66': high - diff * 0.66
    }

def detect_engulfing(prev, curr):
    if curr['close'] > curr['open'] and prev['close'] < prev['open']:
        return curr['close'] > prev['open'] and curr['open'] < prev['close']
    if curr['close'] < curr['open'] and prev['close'] > prev['open']:
        return curr['close'] < prev['open'] and curr['open'] > prev['close']
    return False

def detect_inside_bar(prev, curr):
    return curr['high'] < prev['high'] and curr['low'] > prev['low']

def backtest_coin(symbol):
    print(f"\nRunning backtest for {symbol}...")
    df_1h = get_candles(symbol, 3600)
    df_1d = get_candles(symbol, 86400)

    if df_1h is None or df_1d is None or len(df_1h) < START_AT + 20:
        print(f"{symbol}: Not enough data")
        return pd.DataFrame()

    df_1h['stoch_k'] = StochRSIIndicator(df_1h['close']).stochrsi_k()
    df_1h['stoch_d'] = StochRSIIndicator(df_1h['close']).stochrsi_d()
    df_1d['ema200'] = EMAIndicator(df_1d['close'], window=200).ema_indicator()
    df_1d.set_index('time', inplace=True)

    trades = []

    for i in range(START_AT, len(df_1h)):
        row = df_1h.iloc[i]
        time = row['time']
        price = row['close']
        recent_df = df_1h.iloc[i - LOOKBACK_HOURS:i]
        prev = df_1h.iloc[i - 1]

        daily_row = df_1d[df_1d.index <= time]
        if daily_row.empty:
            continue
        ema200 = daily_row['ema200'].iloc[-1]
        is_bull = df_1d['close'].iloc[-1] > ema200
        is_bear = not is_bull

        swing_high = recent_df['high'].max()
        swing_low = recent_df['low'].min()
        fibs = calc_fibs(swing_high, swing_low)
        golden_min = fibs['0.66'] * (1 - GOLDEN_ZONE_MARGIN)
        golden_max = fibs['0.618'] * (1 + GOLDEN_ZONE_MARGIN)

        k = row['stoch_k']
        d = row['stoch_d']
        crossed_up = k > d and k < 40
        crossed_down = k < d and k > 60

        engulfing = detect_engulfing(prev, row)
        inside_bar = detect_inside_bar(prev, row)
        pattern = "Engulfing" if engulfing else "Inside Bar" if inside_bar else "None"
        valid_candle = True  # relaxed

        result = None
        entry_price = price
        entry_time = row['time']
        tp = None
        sl = None
        direction = None

        if is_bull and golden_min <= price <= golden_max and crossed_up and valid_candle:
            direction = 'long'
            tp = swing_high
            sl = fibs['0.66'] * (1 - SL_BUFFER)
        elif is_bear and golden_min <= price <= golden_max and crossed_down and valid_candle:
            direction = 'short'
            tp = swing_low
            sl = fibs['0.618'] * (1 + SL_BUFFER)

        if direction:
            for j in range(i + 1, len(df_1h)):
                future = df_1h.iloc[j]
                high, low = future['high'], future['low']
                if direction == 'long':
                    if high >= tp:
                        result = 'win'
                        break
                    elif low <= sl:
                        result = 'loss'
                        break
                elif direction == 'short':
                    if low <= tp:
                        result = 'win'
                        break
                    elif high >= sl:
                        result = 'loss'
                        break
            if result:
                trades.append({
                    'symbol': symbol,
                    'entry_time': entry_time,
                    'exit_time': df_1h.iloc[j]['time'],
                    'entry': entry_price,
                    'tp': tp,
                    'sl': sl,
                    'direction': direction,
                    'result': result,
                    'pattern': pattern
                })

    return pd.DataFrame(trades)

# === MAIN BACKTEST LOOP ===
all_results = []

for coin in COINS:
    result_df = backtest_coin(coin)
    if not result_df.empty:
        all_results.append(result_df)

if all_results:
    df_all = pd.concat(all_results, ignore_index=True)

    def calc_pnl(row):
        if row['result'] == 'win':
            return abs(row['tp'] - row['entry'])
        elif row['result'] == 'loss':
            return -abs(row['entry'] - row['sl'])
        return 0

    df_all['pnl'] = df_all.apply(calc_pnl, axis=1)

    summary = df_all.groupby('symbol').agg(
        total_trades=('result', 'count'),
        wins=('result', lambda x: (x == 'win').sum()),
        losses=('result', lambda x: (x == 'loss').sum()),
        win_rate=('result', lambda x: (x == 'win').mean().round(2)),
        net_profit=('pnl', 'sum')
    ).reset_index()

    print("\n=== Backtest Summary ===")
    print(summary)

    df_all.to_csv("rigo_backtest_trades.csv", index=False)
else:
    print("No trades were triggered on any coins.")
