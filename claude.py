import requests
import pandas as pd
import numpy as np

# List of cryptocurrencies to monitor (use Coinbase ticker symbols)
CRYPTO_LIST = ["BTC-USD", "ETH-USD", "XRP-USD", "ADA-USD"]

# Stochastic RSI calculation
def calculate_stochastic_rsi(data, period=14):
    delta = data["close"].diff(1)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)

    avg_gain = pd.Series(gain).rolling(window=period).mean()
    avg_loss = pd.Series(loss).rolling(window=period).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    rsi_min = rsi.rolling(window=period).min()
    rsi_max = rsi.rolling(window=period).max()

    stochastic_rsi = 100 * (rsi - rsi_min) / (rsi_max - rsi_min)

    k = stochastic_rsi.rolling(window=3).mean()
    d = k.rolling(window=3).mean()

    return k, d

# MACD calculation
def calculate_macd(data, short_period=12, long_period=26, signal_period=9):
    short_ema = data["close"].ewm(span=short_period, adjust=False).mean()
    long_ema = data["close"].ewm(span=long_period, adjust=False).mean()
    macd = short_ema - long_ema
    signal = macd.ewm(span=signal_period, adjust=False).mean()
    return macd, signal

# Fetch daily data from Coinbase API
def fetch_data(symbol):
    url = f"https://api.exchange.coinbase.com/products/{symbol}/candles?granularity=86400"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        df = pd.DataFrame(data, columns=["time", "low", "high", "open", "close", "volume"])
        df["time"] = pd.to_datetime(df["time"], unit="s")
        df = df.sort_values("time")
        return df
    else:
        print(f"Failed to fetch data for {symbol}: {response.status_code}")
        return None

# ATR calculation
def calculate_atr(data, period=14):
    high_low = data["high"] - data["low"]
    high_close = abs(data["high"] - data["close"].shift())
    low_close = abs(data["low"] - data["close"].shift())

    true_range = pd.DataFrame({"high_low": high_low, "high_close": high_close, "low_close": low_close}).max(axis=1)
    atr = true_range.rolling(window=period).mean()
    return atr

# Backtesting function
def backtest(data, k, d, macd, signal):
    initial_balance = 1000
    balance = initial_balance
    trades = []
    position = None
    entry_price = None
    wins = 0
    losses = 0
    atr = calculate_atr(data)

    for i in range(1, len(data)):
        if position is None:
            # Check for buy signal
            if (
                k.iloc[i] < 20 and d.iloc[i] < 20 and
                k.iloc[i] > k.iloc[i - 1] and d.iloc[i] > d.iloc[i - 1] and
                abs(k.iloc[i] - d.iloc[i]) < abs(k.iloc[i - 1] - d.iloc[i - 1]) and
                macd.iloc[i] > signal.iloc[i]
            ):
                position = "long"
                entry_price = data["close"].iloc[i]
                take_profit = entry_price + 2 * atr.iloc[i]
                stop_loss = entry_price - 2 * atr.iloc[i]
                trades.append({"type": "buy", "price": entry_price, "time": data["time"].iloc[i]})

            # Check for sell signal
            elif (
                k.iloc[i] > 80 and d.iloc[i] > 80 and
                k.iloc[i] < k.iloc[i - 1] and d.iloc[i] < d.iloc[i - 1] and
                abs(k.iloc[i] - d.iloc[i]) < abs(k.iloc[i - 1] - d.iloc[i - 1]) and
                macd.iloc[i] < signal.iloc[i]
            ):
                position = "short"
                entry_price = data["close"].iloc[i]
                take_profit = entry_price - 2 * atr.iloc[i]
                stop_loss = entry_price + 2 * atr.iloc[i]
                trades.append({"type": "sell", "price": entry_price, "time": data["time"].iloc[i]})

        else:
            # Check long position for exit conditions
            if position == "long":
                if data["close"].iloc[i] >= take_profit or data["close"].iloc[i] <= stop_loss:
                    profit = data["close"].iloc[i] - entry_price
                    balance += profit
                    if profit > 0:
                        wins += 1
                    else:
                        losses += 1
                    trades.append({"type": "sell", "price": data["close"].iloc[i], "time": data["time"].iloc[i]})
                    position = None

            # Check short position for exit conditions
            elif position == "short":
                if data["close"].iloc[i] <= take_profit or data["close"].iloc[i] >= stop_loss:
                    profit = entry_price - data["close"].iloc[i]
                    balance += profit
                    if profit > 0:
                        wins += 1
                    else:
                        losses += 1
                    trades.append({"type": "buy", "price": data["close"].iloc[i], "time": data["time"].iloc[i]})
                    position = None

    return balance, trades, len(trades), wins

# Main function
def main():
    results = []
    for symbol in CRYPTO_LIST:
        data = fetch_data(symbol)
        if data is not None:
            k, d = calculate_stochastic_rsi(data)
            macd, signal = calculate_macd(data)
            final_balance, trades, num_trades, wins = backtest(data, k, d, macd, signal)
            print(f"{symbol} final balance: ${final_balance:.2f}")
            print(f"{symbol} total trades: {num_trades}, wins: {wins}")
            results.append({"symbol": symbol, "balance": final_balance, "trades": trades, "num_trades": num_trades, "wins": wins})

if __name__ == "__main__":
    main()
