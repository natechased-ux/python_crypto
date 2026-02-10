import pandas as pd
import joblib
from datetime import datetime, timezone

# ==== CONFIG ====
COINS = ["btc-usd", "eth-usd", "sol-usd"]
LOOKAHEAD_HOURS = 24  # same as TP/SL training

# ==== LOAD TRAINED MODELS ====
print("📥 Loading ML models...")
ml_model, ml_features = joblib.load("hourly_entry_model.pkl")
tp_model, tp_features = joblib.load("tp_model.pkl")
sl_model, sl_features = joblib.load("sl_model.pkl")

# ==== LOAD HISTORICAL DATA ====
def load_historical_csv(coin):
    csv_file = f"{coin.replace('-', '_')}_historical_dataset.csv"
    print(f"📄 Loading historical data for {coin.upper()} from {csv_file}...")
    df = pd.read_csv(csv_file, parse_dates=["time"])
    df.dropna(inplace=True)
    df["coin"] = coin.upper()  # for model input
    return df

historical_data = {coin: load_historical_csv(coin) for coin in COINS}
import numpy as np

def predict_entry(df):
    latest = df.iloc[-1:][ml_features].copy()

    # Convert coin to category codes like in training
    if "coin" in latest.columns:
        latest["coin"] = latest["coin"].astype("category").cat.codes

    pred = ml_model.predict(latest)[0]
    prob = ml_model.predict_proba(latest)[0][1]
    return pred, prob



def predict_tp_sl(df):
    latest = df.iloc[-1:].copy()

    # Convert coin to category codes like in training
    if "coin" in latest.columns:
        latest["coin"] = latest["coin"].astype("category").cat.codes

    tp_mult = tp_model.predict(latest[tp_features])[0]
    sl_mult = sl_model.predict(latest[sl_features])[0]

    atr = latest["atr"].iloc[0]
    close_price = latest["close"].iloc[0]

    tp_price = close_price + tp_mult * atr
    sl_price = close_price - sl_mult * atr

    return tp_price, sl_price, tp_mult, sl_mult

def simulate_trade(df, entry_index, tp_price, sl_price):
    """
    Simulate a single trade starting at entry_index.
    Looks ahead LOOKAHEAD_HOURS candles to see if TP or SL is hit first.
    Returns: profit in %, outcome ("win"/"loss"/"hold")
    """
    entry_price = df.iloc[entry_index]["close"]

    # Slice future candles
    future_df = df.iloc[entry_index+1 : entry_index+LOOKAHEAD_HOURS+1]
    if future_df.empty:
        return 0.0, "hold"

    for _, row in future_df.iterrows():
        if row["high"] >= tp_price:
            # TP hit
            return (tp_price - entry_price) / entry_price * 100, "win"
        elif row["low"] <= sl_price:
            # SL hit
            return (sl_price - entry_price) / entry_price * 100, "loss"

    # Neither TP nor SL hit → close at last price
    final_price = future_df.iloc[-1]["close"]
    return (final_price - entry_price) / entry_price * 100, "hold"
def run_combined_backtest():
    RISK_PER_TRADE = 0.2  # 2% of equity per trade
    START_BALANCE = 10000

    results = []

    for coin, df in historical_data.items():
        print(f"\n📊 Backtesting {coin.upper()}...")

        trades_taken = 0
        wins = 0
        losses = 0
        holds = 0
        profit_list = []
        
        balance = START_BALANCE
        equity = [balance]

        i = 0
        while i < len(df) - LOOKAHEAD_HOURS - 1:
            # Data up to current candle
            window_df = df.iloc[:i+1]

            # Predict entry
            entry_pred, _ = predict_entry(window_df)
            if entry_pred != 1:
                i += 1
                continue

            # Predict TP & SL
            tp_price, sl_price, _, _ = predict_tp_sl(window_df)

            # Simulate trade
            profit_pct, outcome = simulate_trade(df, i, tp_price, sl_price)

            # Position sizing
            position_size = balance * RISK_PER_TRADE
            balance += position_size * (profit_pct / 100)
            equity.append(balance)
            profit_list.append(profit_pct)

            # Track outcome
            trades_taken += 1
            if outcome == "win":
                wins += 1
            elif outcome == "loss":
                losses += 1
            else:
                holds += 1

            # Skip forward until after this trade closes
            i += LOOKAHEAD_HOURS

        # Performance stats
        win_rate = wins / trades_taken if trades_taken > 0 else 0
        avg_return = np.mean(profit_list) if profit_list else 0
        pf = (sum([p for p in profit_list if p > 0]) /
              abs(sum([p for p in profit_list if p < 0])) if losses > 0 else float('inf'))
        max_dd = (min(equity) / max(equity) - 1) * 100 if equity else 0

        results.append({
            "coin": coin.upper(),
            "trades": trades_taken,
            "win_rate": win_rate,
            "profit_factor": pf,
            "avg_return_%": avg_return,
            "final_balance": balance,
            "max_drawdown_%": max_dd
        })

    return pd.DataFrame(results)



import matplotlib.pyplot as plt

def plot_equity_curves():
    RISK_PER_TRADE = 0.2  # Match backtest setting
    START_BALANCE = 10000

    for coin, df in historical_data.items():
        print(f"📈 Generating equity curve for {coin.upper()}...")

        balance = START_BALANCE
        equity_curve = [balance]
        i = 0

        while i < len(df) - LOOKAHEAD_HOURS - 1:
            # Data up to current candle
            window_df = df.iloc[:i+1]

            # Predict entry
            entry_pred, _ = predict_entry(window_df)
            if entry_pred != 1:
                equity_curve.append(balance)
                i += 1
                continue

            # Predict TP & SL
            tp_price, sl_price, _, _ = predict_tp_sl(window_df)

            # Simulate trade
            profit_pct, _ = simulate_trade(df, i, tp_price, sl_price)

            # Apply position sizing
            position_size = balance * RISK_PER_TRADE
            balance += position_size * (profit_pct / 100)
            equity_curve.append(balance)

            # Skip overlapping trades
            i += LOOKAHEAD_HOURS

        # Plot the equity curve
        plt.figure(figsize=(10, 5))
        plt.plot(equity_curve, label=f"{coin.upper()} Equity")
        plt.title(f"Equity Curve — {coin.upper()} (Fixed Risk)")
        plt.xlabel("Trade Progression")
        plt.ylabel("Balance ($)")
        plt.grid(True)
        plt.legend()
        plt.show()

if __name__ == "__main__":
    df_results = run_combined_backtest()

    print("\n📈 COMBINED ENTRY + TP/SL MODEL BACKTEST RESULTS")
    print(df_results.to_string(index=False,
                               formatters={
                                   "win_rate": "{:.2%}".format,
                                   "profit_factor": "{:.2f}".format,
                                   "avg_return_%": "{:.2f}".format,
                                   "final_balance": "${:,.2f}".format,
                                   "max_drawdown_%": "{:.2f}".format
                               }))

    plot_equity_curves()

        
