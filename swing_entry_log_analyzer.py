import pandas as pd
import matplotlib.pyplot as plt

def analyze_trade_log(file_path="live_trade_log.csv"):
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"⛔ Log file '{file_path}' not found.")
        return

    if df.empty:
        print("⚠ No trades found in log.")
        return

    # Calculate profit/loss in points
    df["pnl"] = 0.0
    df["risk"] = abs(df["entry"] - df["sl"])
    df["reward"] = abs(df["tp"] - df["entry"])
    df["rr_ratio"] = df["reward"] / df["risk"]

    # Outcome: Win if high >= TP (long) or low <= TP (short)
    for i, row in df.iterrows():
        if row["side"] == "long":
            if row["high"] >= row["tp"]:
                df.at[i, "pnl"] = row["reward"]
                df.at[i, "outcome"] = "WIN"
            elif row["low"] <= row["sl"]:
                df.at[i, "pnl"] = -row["risk"]
                df.at[i, "outcome"] = "LOSS"
            else:
                df.at[i, "outcome"] = "OPEN"
        elif row["side"] == "short":
            if row["low"] <= row["tp"]:
                df.at[i, "pnl"] = row["reward"]
                df.at[i, "outcome"] = "WIN"
            elif row["high"] >= row["sl"]:
                df.at[i, "pnl"] = -row["risk"]
                df.at[i, "outcome"] = "LOSS"
            else:
                df.at[i, "outcome"] = "OPEN"

    # Filter out open trades
    closed_df = df[df["outcome"].isin(["WIN", "LOSS"])]

    # Metrics
    total_trades = len(closed_df)
    win_rate = (closed_df["outcome"] == "WIN").mean() * 100
    avg_rr = closed_df["rr_ratio"].mean()
    expectancy = closed_df["pnl"].mean()

    print(f"📊 Trade Performance Summary")
    print(f"Total Trades: {total_trades}")
    print(f"Win Rate: {win_rate:.2f}%")
    print(f"Average R:R Ratio: {avg_rr:.2f}")
    print(f"Expectancy per trade: {expectancy:.4f}")

    # Best & Worst coins
    best_coins = closed_df.groupby("symbol")["pnl"].sum().sort_values(ascending=False).head(5)
    worst_coins = closed_df.groupby("symbol")["pnl"].sum().sort_values().head(5)

    print("\n🏆 Best Performing Symbols:")
    print(best_coins)

    print("\n💔 Worst Performing Symbols:")
    print(worst_coins)

    # Performance by side
    print("\n📈 Performance by Side:")
    print(closed_df.groupby("side")["pnl"].mean())

    # Plot
    closed_df.groupby("symbol")["pnl"].sum().sort_values(ascending=False).plot(kind="bar", figsize=(12,6))
    plt.title("Total PnL by Symbol")
    plt.xlabel("Symbol")
    plt.ylabel("PnL")
    plt.grid(True)
    plt.show()

# Run the analyzer
if __name__ == "__main__":
    analyze_trade_log("live_trade_log.csv")
