#!/usr/bin/env python3
# outcome_fix_once.py (robust mixed-timestamp + no-overwrite)
import os, requests, urllib.parse as url
import pandas as pd
from datetime import datetime, timezone, timedelta

LOG_PATH    = "live_trade_log_fib6.csv"
KRAKEN_BASE = "https://api.kraken.com"
GRANULARITY_SEC = 60      # 1m
ENC = "utf-8-sig"         # Windows/Excel friendly

# Optional alias map (kept here in case any legacy symbols are present)
ALIAS = {
    "BTC-USD": "XBT/USD", "BTC/USD": "XBT/USD",
    "ETH-USD": "ETH/USD", "ETH/USD": "ETH/USD",
}

def kr_pair_for_rest(pair: str) -> str:
    # "XBT/USD" -> "XBTUSD"
    return pair.replace("/", "")

def fetch_ohlc(pair: str, seconds: int = GRANULARITY_SEC) -> pd.DataFrame:
    """Fetch Kraken OHLC and normalize 7/8 columns."""
    itv_map = {60:1, 300:5, 900:15, 1800:30, 3600:60, 21600:240, 86400:1440}
    interval = itv_map.get(seconds, 60)
    q = kr_pair_for_rest(pair)
    u = f"{KRAKEN_BASE}/0/public/OHLC?pair={url.quote(q)}&interval={interval}"
    r = requests.get(u, timeout=15); r.raise_for_status()
    res  = r.json().get("result", {})
    ohlc = next((v for k,v in res.items() if k != "last"), None)
    if not ohlc:
        return pd.DataFrame(columns=["time","open","high","low","close","volume"])
    df = pd.DataFrame(ohlc)
    if df.shape[1] == 8:
        df.columns = ["time","open","high","low","close","vwap","volume","count"]
    elif df.shape[1] == 7:
        df.columns = ["time","open","high","low","close","volume","count"]
        df["vwap"] = pd.NA
    else:
        # pad/trim defensively to 8
        while df.shape[1] < 8:
            df[df.shape[1]] = pd.NA
        df = df.iloc[:, :8]
        df.columns = ["time","open","high","low","close","vwap","volume","count"]
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df = df.astype({"open":"float64","high":"float64","low":"float64","close":"float64"}, errors="ignore")
    return df.sort_values("time").reset_index(drop=True)

def first_touch_outcome(side: str, entry: float, tp: float, sl: float, ohlc: pd.DataFrame):
    """Return 'TP' | 'SL' | None based on first touch scanning forward."""
    for _, c in ohlc.iterrows():
        h = float(c["high"]); l = float(c["low"])
        if side == "LONG":
            if h >= tp: return "TP"
            if l <= sl: return "SL"
        else:
            if l <= tp: return "TP"
            if h >= sl: return "SL"
    return None

def main():
    if not os.path.exists(LOG_PATH):
        print("No CSV found:", LOG_PATH); return

    df = pd.read_csv(LOG_PATH, encoding=ENC)
    required = ["timestamp_utc","outcome","hit_tp","hit_sl","symbol","side","entry","tp1","sl"]
    for col in required:
        if col not in df.columns:
            print("CSV missing column:", col); return

    # ---- tolerant, non-destructive timestamp parsing ----
    ts1 = pd.to_datetime(df["timestamp_utc"], utc=True, errors="coerce")
    bad = ts1.isna()
    if bad.any():
        # strip whitespace; accept both '...Z' and space-formats
        ts2 = pd.to_datetime(df.loc[bad, "timestamp_utc"].astype(str).str.strip().str.replace("Z","", regex=False),
                             utc=True, errors="coerce")
        ts1 = ts1.where(~bad, ts2)
    df["_ts_parsed"] = ts1  # keep original timestamp_utc untouched

    # rows that look open (no outcome) and have a parsable ts
    open_mask   = (df["outcome"].astype(str).str.len()==0) | ((df["hit_tp"].fillna(0)==0) & (df["hit_sl"].fillna(0)==0))
    valid_ts    = df["_ts_parsed"].notna()
    candidate_ix = df[open_mask & valid_ts].index.tolist()

    print(f"Scanning {len(candidate_ix)} open rows...")
    updated = 0

    for i in candidate_ix:
        row = df.loc[i]
        raw_sym = str(row["symbol"])
        sym     = ALIAS.get(raw_sym, raw_sym)  # fine if unchanged
        side    = str(row["side"]).upper()

        # Numerics
        try:
            entry = float(row["entry"]); tp = float(row["tp1"]); sl = float(row["sl"])
        except Exception:
            # Bad numeric fields; skip
            continue

        t0 = row["_ts_parsed"]
        if pd.isna(t0):
            # Unparseable timestamp; skip (we do NOT blank the original)
            continue

        # 2-minute prebuffer to avoid edge miss
        start = t0 - timedelta(minutes=2)

        # Fetch candles and filter to start
        try:
            ohlc = fetch_ohlc(sym, GRANULARITY_SEC)
        except Exception:
            # API hiccup; skip this row
            continue
        if ohlc.empty:    continue
        ohlc = ohlc[ohlc["time"] >= start]
        if ohlc.empty:    continue

        out = first_touch_outcome(side, entry, tp, sl, ohlc)
        if out is None:
            # Still open
            continue

        # Mark outcome + flags
        if out == "TP":
            df.at[i,"outcome"] = "TP"; df.at[i,"hit_tp"] = 1; df.at[i,"hit_sl"] = 0
        else:
            df.at[i,"outcome"] = "SL"; df.at[i,"hit_tp"] = 0; df.at[i,"hit_sl"] = 1

        # MAE/MFE in percent (optional, useful for tuning later)
        highs = ohlc["high"].astype(float); lows = ohlc["low"].astype(float)
        if side == "LONG":
            mfe_pct = (highs.max() - entry) / entry * 100.0
            mae_pct = (entry - lows.min()) / entry * 100.0
        else:
            mfe_pct = (entry - lows.min()) / entry * 100.0
            mae_pct = (highs.max() - entry) / entry * 100.0
        df.at[i,"mfe_pct"] = round(float(mfe_pct), 4)
        df.at[i,"mae_pct"] = round(float(mae_pct), 4)

        updated += 1

    # ---- write back ONLY outcome fields; preserve timestamp_utc strings ----
    if updated:
        tmp = LOG_PATH + ".tmp"
        df.drop(columns=["_ts_parsed"], errors="ignore").to_csv(tmp, index=False, encoding=ENC)
        os.replace(tmp, LOG_PATH)
        print(f"Updated {updated} rows → {LOG_PATH}")
    else:
        print("No rows updated. Likely still open, wrong symbols, or TP/SL not touched yet.")

if __name__ == "__main__":
    main()
