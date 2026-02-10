import requests
import pandas as pd
import datetime as dt
import time
import matplotlib.pyplot as plt

# ------------------------
# Fetch 1-min candles looped over N days
# ------------------------
def fetch_candles(symbol="fartcoin-USD", granularity=60, days=7):
    """
    Fetches up to `days` worth of 1-min candles by paging Coinbase API.
    """
    end = dt.datetime.utcnow()
    start = end - dt.timedelta(days=days)
    all_candles = []

    while end > start:
        # Coinbase max = 300 candles per call
        chunk_start = end - dt.timedelta(seconds=granularity*300)
        if chunk_start < start:
            chunk_start = start

        url = f"https://api.exchange.coinbase.com/products/{symbol}/candles"
        params = {
            "granularity": granularity,
            "start": chunk_start.isoformat(),
            "end": end.isoformat()
        }

        r = requests.get(url, params=params)
        data = r.json()

        if not data or isinstance(data, dict):  # API returns error as dict
            print("Error or empty response, stopping...")
            break

        all_candles.extend(data)

        # move window back
        end = chunk_start
        time.sleep(0.2)  # avoid rate limits

    # Coinbase returns [time, low, high, open, close, volume]
    df = pd.DataFrame(all_candles, columns=["time", "low", "high", "open", "close", "volume"])
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df = df.sort_values("time").reset_index(drop=True)
    return df

# ------------------------
# Spike analysis (split up vs down)
# ------------------------
def analyze_spikes(df, threshold=0.01, forward_range=15):
    forward_returns_up = {m: [] for m in range(1, forward_range+1)}
    forward_returns_down = {m: [] for m in range(1, forward_range+1)}

    for i in range(len(df) - forward_range):
        pct_change = (df.loc[i, "close"] - df.loc[i, "open"]) / df.loc[i, "open"]

        if pct_change >= threshold:  # UP spike
            for m in range(1, forward_range+1):
                ret = (df.loc[i+m, "close"] - df.loc[i, "close"]) / df.loc[i, "close"]
                forward_returns_up[m].append(ret * 100)
        elif pct_change <= -threshold:  # DOWN spike
            for m in range(1, forward_range+1):
                ret = (df.loc[i+m, "close"] - df.loc[i, "close"]) / df.loc[i, "close"]
                forward_returns_down[m].append(ret * 100)

    def summarize(forward_returns):
        results = {}
        for m in forward_returns:
            if forward_returns[m]:
                series = pd.Series(forward_returns[m])
                results[m] = {
                    "median": series.median(),
                    "mean": series.mean(),
                    "25th": series.quantile(0.25),
                    "75th": series.quantile(0.75),
                    "samples": len(series)
                }
        return results

    return summarize(forward_returns_up), summarize(forward_returns_down)

# ------------------------
# Plot results with IQR shading
# ------------------------
def plot_results(results_up, results_down):
    minutes = sorted(set(results_up.keys()).union(results_down.keys()))

    plt.figure(figsize=(10,6))

    if results_up:
        med_up = [results_up[m]["median"] for m in minutes if m in results_up]
        p25_up = [results_up[m]["25th"] for m in minutes if m in results_up]
        p75_up = [results_up[m]["75th"] for m in minutes if m in results_up]
        mins_up = [m for m in minutes if m in results_up]

        plt.plot(mins_up, med_up, marker="o", color="green", label="Median (Up Spikes)")
        plt.fill_between(mins_up, p25_up, p75_up, color="green", alpha=0.2, label="IQR Up (25–75%)")

    if results_down:
        med_down = [results_down[m]["median"] for m in minutes if m in results_down]
        p25_down = [results_down[m]["25th"] for m in minutes if m in results_down]
        p75_down = [results_down[m]["75th"] for m in minutes if m in results_down]
        mins_down = [m for m in minutes if m in results_down]

        plt.plot(mins_down, med_down, marker="o", color="red", label="Median (Down Spikes)")
        plt.fill_between(mins_down, p25_down, p75_down, color="red", alpha=0.2, label="IQR Down (25–75%)")

    plt.axhline(0, color="black", linewidth=1, linestyle="--")
    plt.title("Forward Returns After ≥1% 1-Min Move (Up vs Down)")
    plt.xlabel("Minutes Ahead")
    plt.ylabel("Return (%)")
    plt.legend()
    plt.grid(True)
    plt.show()

# ------------------------
# Run for XRP (example)
# ------------------------
if __name__ == "__main__":
    df = fetch_candles("sui-USD", granularity=900, days=90)  # last 30 days
    print("Candles fetched:", len(df))

    results_up, results_down = analyze_spikes(df, threshold=0.03, forward_range=15)

    print("\n--- UP Spikes ---")
    for m, stats in results_up.items():
        print(f"{m} min after (n={stats['samples']}): "
              f"Median {stats['median']:.4f}%, Mean {stats['mean']:.4f}%, "
              f"IQR {stats['25th']:.4f}%–{stats['75th']:.4f}%")

    print("\n--- DOWN Spikes ---")
    for m, stats in results_down.items():
        print(f"{m} min after (n={stats['samples']}): "
              f"Median {stats['median']:.4f}%, Mean {stats['mean']:.4f}%, "
              f"IQR {stats['25th']:.4f}%–{stats['75th']:.4f}%")

    plot_results(results_up, results_down)
