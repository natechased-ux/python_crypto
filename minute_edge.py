# minute_edge.py
# pip install pandas numpy requests ta python-dateutil

import time, math, requests, pandas as pd, numpy as np
from datetime import datetime, timedelta, timezone
from dateutil import parser as dtparser
import argparse
import ta

# ---------- Config defaults ----------
DEFAULT_SYMBOL      = "ETH-USD"
DEFAULT_GRAN        = 60                 # 1-minute candles
DEFAULT_DAYS        = 30                 # fetch this many days back
FEE_RT              = 0.0012             # 0.12% round-trip (both sides)
SLIPPAGE            = 0.0002             # 0.02% each side
REQUEST_TIMEOUT_S   = 15
BATCH_SECONDS       = 300 * DEFAULT_GRAN # 300 candles per call -> 5 hours for 1-min
PAUSE_BETWEEN_CALLS = 0.18               # be gentle on rate limits
MAX_RETRIES         = 5

# ---------- Data fetch ----------
def fetch_all_candles(product_id=DEFAULT_SYMBOL, granularity=DEFAULT_GRAN, days=DEFAULT_DAYS):
    """
    Fetch as many candles as requested by paging time windows of 300 points (API limit).
    Returns a DataFrame with columns: time, open, high, low, close, volume (UTC).
    """
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days)

    all_rows = []
    t0 = start

    while t0 < end:
        t1 = min(t0 + timedelta(seconds=BATCH_SECONDS), end)

        url = (
            f"https://api.exchange.coinbase.com/products/{product_id}/candles"
            f"?granularity={granularity}"
            f"&start={t0.isoformat()}&end={t1.isoformat()}"
        )

        # retry loop
        attempt = 0
        while True:
            try:
                r = requests.get(url, timeout=REQUEST_TIMEOUT_S)
                if r.status_code == 429:
                    # rate limited; back off a bit more
                    time.sleep(1.0 + attempt * 0.5)
                    attempt += 1
                    if attempt > MAX_RETRIES:
                        raise RuntimeError("Rate limit exceeded repeatedly.")
                    continue
                r.raise_for_status()
                data = r.json()
                break
            except Exception as e:
                attempt += 1
                if attempt > MAX_RETRIES:
                    raise
                time.sleep(0.6 * attempt)

        if not data:
            # Sometimes Coinbase returns empty windows near the most recent edge — move forward.
            t0 = t1
            time.sleep(PAUSE_BETWEEN_CALLS)
            continue

        # Coinbase returns: [ time, low, high, open, close, volume ] newest->oldest or oldest->newest depending on window.
        df_chunk = pd.DataFrame(data, columns=["time","low","high","open","close","volume"])
        all_rows.append(df_chunk)
        t0 = t1
        time.sleep(PAUSE_BETWEEN_CALLS)

    if not all_rows:
        raise RuntimeError("No data returned from Coinbase.")

    df = pd.concat(all_rows, ignore_index=True)
    # Deduplicate/normalize
    df = df.drop_duplicates(subset=["time"])
    df = df.sort_values("time").reset_index(drop=True)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    # Reorder to a common OHLCV column order
    df = df[["time","open","high","low","close","volume"]]
    return df

# ---------- Indicators ----------
def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    close = df["close"]
    high  = df["high"]
    low   = df["low"]

    df["ema9"]   = close.ewm(span=9,  adjust=False).mean()
    df["ema21"]  = close.ewm(span=21, adjust=False).mean()
    df["ema200"] = close.ewm(span=200, adjust=False).mean()

    prev_close = close.shift(1)
    tr = np.maximum(high - low, np.maximum((high - prev_close).abs(), (low - prev_close).abs()))
    df["atr"] = tr.rolling(14, min_periods=14).mean().fillna(method="bfill")

    mid = close.rolling(20, min_periods=20).mean()
    std = close.rolling(20, min_periods=20).std(ddof=0)
    df["bb_mid"] = mid
    df["bb_up"]  = mid + 2*std
    df["bb_lo"]  = mid - 2*std
    df["bb_bw"]  = (df["bb_up"] - df["bb_lo"]) / mid

    df["rsi7"] = ta.momentum.rsi(close, window=7)

    # Rolling VWAP over up to a day of minutes (fallback if fewer rows)
    window = min(1440, len(df))
    pv = (close * df["volume"]).rolling(window, min_periods=1).sum()
    vv = df["volume"].rolling(window, min_periods=1).sum()
    df["vwap"] = pv / vv

    dev = close - df["vwap"]
    df["dev_std"] = dev.rolling(120, min_periods=30).std(ddof=0)

    return df

# ---------- Backtest ----------
def simulate_trade(df, i, side, atr, tp_mult, sl_mult, timeout_bars):
    """
    Entry at next bar open; SL/TP as absolute price moves using ATR at signal bar.
    Conservative fill order: SL checked before TP inside each bar.
    Returns net return (fraction of entry), after fees + slippage.
    """
    if i+1 >= len(df):
        return 0.0

    o_next = df["open"].iloc[i+1]
    entry = o_next * (1 + SLIPPAGE if side=="long" else 1 - SLIPPAGE)
    tp = entry + (tp_mult*atr if side=="long" else -tp_mult*atr)
    sl = entry - (sl_mult*atr if side=="long" else -sl_mult*atr)

    for j in range(1, timeout_bars+1):
        k = i + j
        if k >= len(df):
            break
        h = df["high"].iloc[k]
        l = df["low"].iloc[k]
        # Prioritize SL fill first (conservative)
        if side=="long":
            if l <= sl:
                exit_px = sl * (1 - SLIPPAGE)
                gross = exit_px - entry
                return gross/entry - FEE_RT
            if h >= tp:
                exit_px = tp * (1 - SLIPPAGE)
                gross = exit_px - entry
                return gross/entry - FEE_RT
        else:
            if h >= sl:
                exit_px = sl * (1 + SLIPPAGE)
                gross = entry - exit_px
                return gross/entry - FEE_RT
            if l <= tp:
                exit_px = tp * (1 + SLIPPAGE)
                gross = entry - exit_px
                return gross/entry - FEE_RT

    # Timeout: exit at close of final bar
    exit_px = df["close"].iloc[min(i+timeout_bars, len(df)-1)]
    exit_px = exit_px * (1 - SLIPPAGE if side=="long" else 1 + SLIPPAGE)
    gross = (exit_px - entry) if side=="long" else (entry - exit_px)
    return gross/entry - FEE_RT

def backtest_all(df: pd.DataFrame):
    """
    Run three strategies independently.
    """
    results = {
        "trend_pb":  {"pnl": [], "trades": 0, "wins": 0, "losses": 0},
        "breakout":  {"pnl": [], "trades": 0, "wins": 0, "losses": 0},
        "vwap_mr":   {"pnl": [], "trades": 0, "wins": 0, "losses": 0},
    }

    for i in range(210, len(df)-1):
        row = df.iloc[i]
        atr = max(row["atr"], 1e-9)

        # --- A) Trend pullback (EMA9>EMA21 above EMA200, and tagged EMA21 last 5 bars)
        long_sig_a  = (row["close"] > row["ema200"]) and (row["ema9"] > row["ema21"]) \
                      and (df["low"].iloc[i-5:i].min() <= df["ema21"].iloc[i-5:i].max())
        short_sig_a = (row["close"] < row["ema200"]) and (row["ema9"] < row["ema21"]) \
                      and (df["high"].iloc[i-5:i].max() >= df["ema21"].iloc[i-5:i].min())

        if long_sig_a or short_sig_a:
            pnl = simulate_trade(df, i, "long" if long_sig_a else "short",
                                 atr, tp_mult=1.2, sl_mult=0.8, timeout_bars=20)
            r = results["trend_pb"]
            r["pnl"].append(pnl); r["trades"] += 1; r["wins"] += (pnl>0); r["losses"] += (pnl<=0)

        # --- B) Breakout from squeeze (BB width in lowest 20% of last 500 bars, then close breaks band)
        if i >= 520:
            bw_window = df["bb_bw"].iloc[i-500:i].dropna()
            if len(bw_window) >= 100:
                perc20 = np.percentile(bw_window, 20)
                squeeze = row["bb_bw"] <= perc20
            else:
                squeeze = False
        else:
            squeeze = False

        long_sig_b  = squeeze and (row["close"] > row["bb_up"])
        short_sig_b = squeeze and (row["close"] < row["bb_lo"])
        if long_sig_b or short_sig_b:
            pnl = simulate_trade(df, i, "long" if long_sig_b else "short",
                                 atr, tp_mult=1.5, sl_mult=1.0, timeout_bars=30)
            r = results["breakout"]
            r["pnl"].append(pnl); r["trades"] += 1; r["wins"] += (pnl>0); r["losses"] += (pnl<=0)

        # --- C) VWAP mean-reversion (±1σ from VWAP + RSI filter + reversal candle)
        dev_std = row["dev_std"]
        long_sig_c  = (dev_std and dev_std>0) and (row["close"] < row["vwap"] - 1.0*dev_std) and (row["rsi7"] < 25) and (row["close"] > row["open"])
        short_sig_c = (dev_std and dev_std>0) and (row["close"] > row["vwap"] + 1.0*dev_std) and (row["rsi7"] > 75) and (row["close"] < row["open"])
        if long_sig_c or short_sig_c:
            pnl = simulate_trade(df, i, "long" if long_sig_c else "short",
                                 atr, tp_mult=0.8, sl_mult=0.8, timeout_bars=15)
            r = results["vwap_mr"]
            r["pnl"].append(pnl); r["trades"] += 1; r["wins"] += (pnl>0); r["losses"] += (pnl<=0)

    # summarize
    def summarize(bucket):
        pnl = np.array(bucket["pnl"], dtype=float)
        trades = bucket["trades"]
        wins   = bucket["wins"]
        losses = bucket["losses"]
        winrate = (wins / trades * 100) if trades else 0.0
        avg    = (pnl.mean() * 100) if trades else 0.0
        med    = (np.median(pnl) * 100) if trades else 0.0
        sh     = (np.mean(pnl) / (np.std(pnl) + 1e-12) * np.sqrt(60*24*365)) if trades > 10 else None
        return {
            "trades": trades,
            "winrate_%": round(winrate, 2),
            "avg_return_%": round(avg, 3),
            "median_%": round(med, 3),
            "sharpe_like": round(sh, 2) if sh is not None else None,
            "expectancy_bp": round(avg * 100, 1)
        }

    return {k: summarize(v) for k, v in results.items()}

# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser(description="Fetch long-history 1m candles and backtest minute setups.")
    ap.add_argument("--symbol", default=DEFAULT_SYMBOL, help="e.g., BTC-USD, ETH-USD")
    ap.add_argument("--days", type=int, default=DEFAULT_DAYS, help="How many days back to fetch (e.g., 30, 90, 365)")
    ap.add_argument("--granularity", type=int, default=DEFAULT_GRAN, help="Seconds per candle; 60 for 1m")
    args = ap.parse_args()

    print(f"Fetching {args.symbol} {args.granularity}s candles for ~{args.days} days…")
    df = fetch_all_candles(product_id=args.symbol, granularity=args.granularity, days=args.days)
    print(f"Got {len(df)} candles from {df['time'].iloc[0]} to {df['time'].iloc[-1]}.")

    df = compute_indicators(df)
    report = backtest_all(df)

    print("\nBacktest results (after fees+slippage):")
    for name, stats in report.items():
        print(f"{name}: {stats}")

if __name__ == "__main__":
    main()
