import requests
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from datetime import datetime, timedelta, timezone
from ta.volatility import BollingerBands, AverageTrueRange
from ta.momentum import RSIIndicator
from ta.trend import ADXIndicator, MACD

# ===== CONFIG =====
COINS = ["btc-usd", "eth-usd", "sol-usd"]
BASE_URL = "https://api.exchange.coinbase.com"
GRANULARITY = 3600  # 1 hour candles
MAX_CANDLES = 300
TRAINING_WINDOW_DAYS = 180

# Strategy parameters
RISK_PER_TRADE = 0.02
ACCOUNT_BALANCE = 10000

# TA Parameters (match bot)
RSI_PERIOD = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
ATR_PERIOD = 14
ADX_PERIOD = 14
BB_PERIOD = 20
BB_STD = 2
def fetch_coinbase_candles(pair, start, end, granularity=GRANULARITY):
    """Fetch candles from Coinbase API in chunks."""
    all_data = []
    chunk_seconds = granularity * MAX_CANDLES
    current_start = start

    while current_start < end:
        current_end = min(current_start + timedelta(seconds=chunk_seconds), end)
        url = f"{BASE_URL}/products/{pair}/candles"
        params = {
            "granularity": granularity,
            "start": current_start.isoformat(),
            "end": current_end.isoformat()
        }
        try:
            r = requests.get(url, params=params)
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            print(f"Error fetching {pair}: {e}")
            break
        if not data:
            break
        all_data.extend(data)
        time.sleep(0.4)  # API throttle
        current_start = current_end

    if not all_data:
        return pd.DataFrame()

    df = pd.DataFrame(all_data, columns=["time", "low", "high", "open", "close", "volume"])
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    return df.sort_values("time")
def add_indicators(df):
    """Add TA indicators to DataFrame."""
    if df.empty:
        return df

    df["rsi"] = RSIIndicator(df["close"], window=RSI_PERIOD).rsi()

    bb = BollingerBands(df["close"], window=BB_PERIOD, window_dev=BB_STD)
    df["bb_high"] = bb.bollinger_hband()
    df["bb_low"] = bb.bollinger_lband()

    df["atr"] = AverageTrueRange(
        df["high"], df["low"], df["close"], window=ATR_PERIOD
    ).average_true_range()

    macd = MACD(
        df["close"], window_slow=MACD_SLOW, window_fast=MACD_FAST, window_sign=MACD_SIGNAL
    )
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()

    adx = ADXIndicator(df["high"], df["low"], df["close"], window=ADX_PERIOD)
    df["adx"] = adx.adx()

    return df


def get_daily_trend(coin):
    """Return daily trend: BULL or BEAR."""
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=250)
    df_daily = fetch_coinbase_candles(coin, start, end, granularity=86400)
    if df_daily.empty:
        return None
    ma50 = df_daily["close"].rolling(50).mean().iloc[-1]
    ma200 = df_daily["close"].rolling(200).mean().iloc[-1]
    return "BULL" if ma50 > ma200 else "BEAR"
def generate_signal(df, daily_trend):
    """Generate buy/sell/hold signal with TP and SL."""
    last_row = df.iloc[-1]
    atr = last_row["atr"]
    close_price = last_row["close"]

    # Multi-candle confirmation
    bullish = (df["macd"].iloc[-3:] > df["macd_signal"].iloc[-3:]).all()
    bearish = (df["macd"].iloc[-3:] < df["macd_signal"].iloc[-3:]).all()

    # Trend-adjusted TP/SL
    if last_row["adx"] > 25:  # trending
        tp_mult, sl_mult = 2.0, 1.2
    else:  # ranging
        tp_mult, sl_mult = 1.2, 0.8

    signal, tp_price, sl_price = "HOLD", None, None

    if bullish and last_row["rsi"] < 35 and daily_trend == "BULL":
        signal = "BUY"
        tp_price = close_price + tp_mult * atr
        sl_price = close_price - sl_mult * atr
    elif bearish and last_row["rsi"] > 65 and daily_trend == "BEAR":
        signal = "SELL"
        tp_price = close_price - tp_mult * atr
        sl_price = close_price + sl_mult * atr

    return signal, tp_price, sl_price
def backtest_coin(coin):
    """Run backtest for a single coin."""
    print(f"\n=== Backtesting {coin.upper()} ===")
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=365 * 2)

    # Fetch and prepare data
    df = fetch_coinbase_candles(coin, start, end)
    if df.empty:
        print(f"No data for {coin}")
        return None

    df = add_indicators(df)
    df.dropna(inplace=True)

    # Get daily trend
    daily_trend = get_daily_trend(coin)
    if not daily_trend:
        print(f"No daily trend for {coin}")
        return None

    # Initial balance
    balance = ACCOUNT_BALANCE
    equity_curve = []
    trade_results = []

    position_open = False
    entry_price = None
    sl_price = None
    tp_price = None

    for i in range(50, len(df)):  # skip first bars for indicators
        sub_df = df.iloc[: i + 1]
        signal, tp, sl = generate_signal(sub_df, daily_trend)

        price = sub_df.iloc[-1]["close"]
        atr = sub_df.iloc[-1]["atr"]

        if not position_open:
            if signal in ["BUY", "SELL"]:
                entry_price = price
                sl_price = sl
                tp_price = tp
                risk_amount = balance * RISK_PER_TRADE
                position_size = risk_amount / abs(entry_price - sl_price)
                position_open = True
                entry_signal = signal
        else:
            # Check exit conditions
            if entry_signal == "BUY":
                if price >= tp_price:
                    profit = (tp_price - entry_price) * position_size
                    balance += profit
                    trade_results.append(profit)
                    position_open = False
                elif price <= sl_price:
                    loss = (sl_price - entry_price) * position_size
                    balance += loss
                    trade_results.append(loss)
                    position_open = False
            elif entry_signal == "SELL":
                if price <= tp_price:
                    profit = (entry_price - tp_price) * position_size
                    balance += profit
                    trade_results.append(profit)
                    position_open = False
                elif price >= sl_price:
                    loss = (entry_price - sl_price) * position_size
                    balance += loss
                    trade_results.append(loss)
                    position_open = False

        equity_curve.append(balance)

    # Performance stats
    wins = [r for r in trade_results if r > 0]
    losses = [r for r in trade_results if r < 0]
    win_rate = len(wins) / len(trade_results) * 100 if trade_results else 0
    profit_factor = sum(wins) / abs(sum(losses)) if losses else float("inf")
    expectancy = np.mean(trade_results) if trade_results else 0
    max_drawdown = min(equity_curve) - max(equity_curve) if equity_curve else 0

    stats = {
        "coin": coin.upper(),
        "trades": len(trade_results),
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "expectancy": expectancy,
        "final_balance": balance,
        "max_drawdown": max_drawdown,
        "equity_curve": equity_curve,
    }
    return stats
def run_backtest():
    all_stats = []
    plt.figure(figsize=(12, 6))

    for coin in COINS:
        stats = backtest_coin(coin)
        if stats:
            all_stats.append(stats)
            plt.plot(stats["equity_curve"], label=stats["coin"])

    # Print summary
    print("\n=== BACKTEST SUMMARY ===")
    for s in all_stats:
        print(
            f"{s['coin']}: Trades={s['trades']} | "
            f"Win%={s['win_rate']:.1f}% | "
            f"PF={s['profit_factor']:.2f} | "
            f"Expectancy={s['expectancy']:.2f} | "
            f"Final=${s['final_balance']:.2f} | "
            f"MaxDD=${s['max_drawdown']:.2f}"
        )

    # Plot equity curves
    plt.title("Equity Curves — Hourly Swing Bot Backtest")
    plt.xlabel("Trades")
    plt.ylabel("Balance ($)")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    run_backtest()
