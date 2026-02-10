import requests
import pandas as pd
import numpy as np
from ta.momentum import StochRSIIndicator
from ta.trend import EMAIndicator, MACD
from ta.volatility import AverageTrueRange

# === CONFIG ===
COINS = ["btc-usd", "eth-usd", "xrp-usd", "ltc-usd", "ada-usd",
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
GOLDEN_ZONE_MARGIN = 0.5
START_AT = 200
LOOKBACK_HOURS = 168
POSITION_SIZE = 2500
SUP_RES_WINDOW = 20  # Number of bars to look back for zones


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

def find_zones(prices, window=SUP_RES_WINDOW):
    supports, resistances = [], []
    for i in range(window, len(prices)):
        segment = prices[i-window:i]
        high = segment['high'].max()
        low = segment['low'].min()
        close = segment.iloc[-1]['close']

        if abs(close - low) / close < 0.01:
            supports.append(low)
        if abs(close - high) / close < 0.01:
            resistances.append(high)

    return supports, resistances

def backtest_coin(symbol):
    print(f"\nRunning backtest for {symbol}...")
    df_1h = get_candles(symbol, 3600)
    df_1d = get_candles(symbol, 86400)

    if df_1h is None or df_1d is None or len(df_1h) < START_AT + 20:
        print(f"{symbol}: Not enough data")
        return pd.DataFrame()

    df_1h['stoch_k'] = StochRSIIndicator(df_1h['close']).stochrsi_k()
    df_1h['stoch_d'] = StochRSIIndicator(df_1h['close']).stochrsi_d()

    macd = MACD(df_1h['close'])
    df_1h['macd_diff'] = macd.macd_diff()

    df_1d['ema200'] = EMAIndicator(df_1d['close'], window=200).ema_indicator()
    df_1d.set_index('time', inplace=True)

    trades = []

    for i in range(START_AT, len(df_1h)):
        row = df_1h.iloc[i]
        time = row['time']
        price = row['close']

        if row['macd_diff'] < 0:
            continue

        recent_df = df_1h.iloc[i - LOOKBACK_HOURS:i]
        zones_df = df_1h.iloc[i - SUP_RES_WINDOW:i]
        supports, resistances = find_zones(zones_df)

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

        result = None
        entry_price = price
        entry_time = row['time']
        tp = None
        sl = None
        direction = None

        if is_bull and golden_min <= price <= golden_max and crossed_up:
            direction = 'long'
            tp = max([r for r in resistances if r > price], default=swing_high)
            sl = min([s for s in supports if s < price], default=swing_low)
        elif is_bear and golden_min <= price <= golden_max and crossed_down:
            direction = 'short'
            tp = min([s for s in supports if s < price], default=swing_low)
            sl = max([r for r in resistances if r > price], default=swing_high)

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
                    'result': result
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
    df_all['pnl_usd'] = df_all['pnl'] / df_all['entry'] * POSITION_SIZE

    summary = df_all.groupby('symbol').agg(
        total_trades=('result', 'count'),
        wins=('result', lambda x: (x == 'win').sum()),
        losses=('result', lambda x: (x == 'loss').sum()),
        win_rate=('result', lambda x: (x == 'win').mean().round(2)),
        net_profit_usd=('pnl_usd', 'sum')
    ).reset_index()

    totals = pd.DataFrame([{
        'symbol': 'TOTAL',
        'total_trades': summary['total_trades'].sum(),
        'wins': summary['wins'].sum(),
        'losses': summary['losses'].sum(),
        'win_rate': round(summary['wins'].sum() / summary['total_trades'].sum(), 2),
        'net_profit_usd': summary['net_profit_usd'].sum()
    }])

    summary = pd.concat([summary, totals], ignore_index=True)
    print("\n=== Backtest Summary ===")
    print(summary)

    df_all.to_csv("rigo_backtest_trades.csv", index=False)
else:
    print("No trades were triggered on any coins.")
