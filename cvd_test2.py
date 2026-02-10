import requests
import pandas as pd
import numpy as np
from ta.volatility import AverageTrueRange

granularity = 60  # 1d candles

def fetch_candle_data(symbol):
    url = f"https://api.exchange.coinbase.com/products/{symbol}/candles?granularity={granularity}"
    response = requests.get(url)
    if response.status_code == 200:
        columns = ["time", "low", "high", "open", "close", "volume"]
        df = pd.DataFrame(response.json(), columns=columns)
        df = df.sort_values(by="time").reset_index(drop=True)
        return df
    else:
        raise Exception(f"Failed to fetch data for {symbol}")

def calculate_atr(data, length=14):
    atr_indicator = AverageTrueRange(high=data['high'], low=data['low'], close=data['close'], window=length)
    data['atr'] = atr_indicator.average_true_range()
    return data

def calculate_cvd(data):
    data['vol_delta'] = data['volume'] * np.sign(data['close'] - data['open'])
    data['cvd'] = data['vol_delta'].cumsum()
    return data

def get_peaks(series, window=3):
    peaks = []
    for i in range(window, len(series) - window):
        if series[i] == max(series[i-window:i+window+1]):
            peaks.append(i)
    return peaks

def get_troughs(series, window=3):
    troughs = []
    for i in range(window, len(series) - window):
        if series[i] == min(series[i-window:i+window+1]):
            troughs.append(i)
    return troughs

def detect_divergences(data):
    price = data['close'].values
    cvd = data['cvd'].values

    price_highs = get_peaks(price)
    price_lows = get_troughs(price)
    cvd_highs = get_peaks(cvd)
    cvd_lows = get_troughs(cvd)

    divergences = []

    # Bearish divergence detection
    for i in range(len(price_highs)-1):
        ph1, ph2 = price_highs[i], price_highs[i+1]
        if ph2 <= ph1:
            continue
        cvd_h1 = min(cvd_highs, key=lambda x: abs(x - ph1)) if cvd_highs else None
        cvd_h2 = min(cvd_highs, key=lambda x: abs(x - ph2)) if cvd_highs else None
        if cvd_h1 is None or cvd_h2 is None:
            continue
        if cvd[cvd_h2] < cvd[cvd_h1]:
            divergences.append((ph2, 'bearish'))

    # Bullish divergence detection
    for i in range(len(price_lows) - 1):
        pl1, pl2 = price_lows[i], price_lows[i + 1]

        if price[pl2] >= price[pl1]:
            continue

        if pl1 in cvd_lows and pl2 in cvd_lows:
            cvd_l1, cvd_l2 = pl1, pl2

            if cvd[cvd_l2] > cvd[cvd_l1]:
                divergences.append((pl2, 'bullish'))
    return sorted(divergences, key=lambda x: x[0])

def backtest_divergence_with_sl_tp(symbol, hold_bars=6, trade_size=10000, sl_atr_mult=1.0, tp_atr_mult=2.0):
    data = fetch_candle_data(symbol)
    data = calculate_atr(data)
    data = calculate_cvd(data)

    divergences = detect_divergences(data)

    trades = []

    for idx, div_type in divergences:
        if idx + hold_bars >= len(data):
            continue

        entry_price = data.loc[idx, 'close']
        atr = data.loc[idx, 'atr']

        if pd.isna(atr) or atr == 0:
            continue

        if div_type == 'bullish':
            sl_price = entry_price - atr * sl_atr_mult
            tp_price = entry_price + atr * tp_atr_mult
        else:
            sl_price = entry_price + atr * sl_atr_mult
            tp_price = entry_price - atr * tp_atr_mult

        exit_price = None
        exit_idx = None
        exit_type = None

        for future_idx in range(idx + 1, idx + hold_bars + 1):
            low = data.loc[future_idx, 'low']
            high = data.loc[future_idx, 'high']

            if div_type == 'bullish':
                if high >= tp_price:
                    exit_price = tp_price
                    exit_idx = future_idx
                    exit_type = 'TP'
                    break
                elif low <= sl_price:
                    exit_price = sl_price
                    exit_idx = future_idx
                    exit_type = 'SL'
                    break
            else:
                if low <= tp_price:
                    exit_price = tp_price
                    exit_idx = future_idx
                    exit_type = 'TP'
                    break
                elif high >= sl_price:
                    exit_price = sl_price
                    exit_idx = future_idx
                    exit_type = 'SL'
                    break

        if exit_price is None:
            exit_idx = idx + hold_bars
            exit_price = data.loc[exit_idx, 'close']
            exit_type = 'End Hold'

        if div_type == 'bullish':
            ret = (exit_price - entry_price) / entry_price
        else:
            ret = (entry_price - exit_price) / entry_price
        pnl = ret * trade_size

        trades.append({
            'index': idx,
            'type': div_type,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'exit_idx': exit_idx,
            'exit_type': exit_type,
            'return': ret,
            'pnl': pnl
        })

    return pd.DataFrame(trades)

def backtest_multiple_symbols(symbols, hold_bars=6, trade_size=10000, sl_atr_mult=1.0, tp_atr_mult=2.0):
    all_trades = []

    for symbol in symbols:
        trades = backtest_divergence_with_sl_tp(symbol, hold_bars, trade_size, sl_atr_mult, tp_atr_mult)
        if not trades.empty:
            trades['symbol'] = symbol
            all_trades.append(trades)

    if not all_trades:
        return None

    return pd.concat(all_trades, ignore_index=True)

if __name__ == "__main__":
    symbols = ["btc-usd", "eth-usd", "xrp-usd", "ltc-usd", "ada-usd",
           "doge-usd", "sol-usd", "wif-usd", "ondo-usd",
           "sei-usd", "magic-usd", "ape-usd", "jasmy-usd", "wld-usd",
           "syrup-usd","fartcoin-usd", "aero-usd", "link-usd","hbar-usd",
           "aave-usd", "fet-usd", "crv-usd","tao-usd", "avax-usd", "xcn-usd",
           "uni-usd", "mkr-usd", "toshi-usd", "near-usd", "algo-usd", "trump-usd",
           "bch-usd", "inj-usd", "pepe-usd", "xlm-usd", "moodeng-usd", "bonk-usd",
           "dot-usd", "popcat-usd", "arb-usd", "icp-usd",
           "qnt-usd", "tia-usd", "ip-usd" , "pnut-usd", "apt-usd", "vet-usd",
           "ena-usd", "turbo-usd", "bera-usd", "pol-usd", "mask-usd", "ach-usd",
           "pyth-usd", "sand-usd", "morpho-usd", "mana-usd", "velo-usd", "coti-usd",
           "axs-usd"]
    df_all_trades = backtest_multiple_symbols(symbols, hold_bars=6, trade_size=10000, sl_atr_mult=1.0, tp_atr_mult=2.0)

    if df_all_trades is not None:
        print("\nTrade Summary:")
        print(df_all_trades)

        start_date = min(fetch_candle_data(symbol)["time"].min() for symbol in symbols)
        end_date = max(fetch_candle_data(symbol)["time"].max() for symbol in symbols)

        total_days = (end_date - start_date) / (24 * 3600)
        total_trades = len(df_all_trades)
        avg_alerts_per_day = total_trades / total_days

        print(f"\nAverage alerts per day: {avg_alerts_per_day:.2f}")

        recent_trades = df_all_trades.sort_values(by="index", ascending=False).head(10)
        print("\nDates of the 10 most recent trades:")
        for _, trade in recent_trades.iterrows():
            symbol = trade['symbol']
            index = int(trade['index'])
            data = fetch_candle_data(symbol)
            trade_date = pd.to_datetime(data.loc[index, 'time'], unit='s')
            print(f"Symbol: {symbol}, Date: {trade_date}")
