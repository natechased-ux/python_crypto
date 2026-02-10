# download_coinbase_candles.py
"""
Download historical OHLCV candles from Coinbase Exchange (public API)
and save to a single CSV compatible with session_cycle_backtest.py.

Usage (Windows / macOS / Linux):
--------------------------------
py download_coinbase_candles.py --symbols BTC-USD ETH-USD XRP-USD \
  --start 2024-01-01 --end 2025-08-25 --granularity 900 --out data.csv

Notes:
- Granularity must be one of: 60, 300, 900, 3600, 21600, 86400 (1m, 5m, 15m, 1h, 6h, 1d).
- The Coinbase endpoint returns at most 300 candles per request, so this script auto-chunks.
- Endpoint: https://api.exchange.coinbase.com/products/{product_id}/candles
- Response order is reverse-chronological; the script sorts ascending by time.
"""

import argparse
import time
from datetime import datetime, timedelta, timezone
import sys
import requests
import pandas as pd

CB_BASE = "https://api.exchange.coinbase.com"

ALLOWED = {60, 300, 900, 3600, 21600, 86400}

def parse_args():
    p = argparse.ArgumentParser(description="Download Coinbase candles to CSV")
    p.add_argument("--symbols", nargs="+", required=True, help="Symbols like BTC-USD ETH-USD")
    p.add_argument("--start", required=True, help="Start date (UTC) YYYY-MM-DD or YYYY-MM-DDTHH:MM:SSZ")
    p.add_argument("--end", required=True, help="End date (UTC) YYYY-MM-DD or YYYY-MM-DDTHH:MM:SSZ")
    p.add_argument("--granularity", type=int, default=900, help="Candle size in seconds (60,300,900,3600,21600,86400)")
    p.add_argument("--out", default="data.csv", help="Output CSV path")
    p.add_argument("--pause", type=float, default=0.2, help="Pause (seconds) between requests to be nice to rate limits")
    return p.parse_args()

def to_dt(s: str) -> datetime:
    # Accept date or datetime; assume UTC
    try:
        if "T" in s:
            # parse full ISO8601
            return datetime.fromisoformat(s.replace("Z","")).replace(tzinfo=timezone.utc)
        else:
            return datetime.fromisoformat(s).replace(tzinfo=timezone.utc)
    except Exception:
        print(f"Could not parse date: {s}", file=sys.stderr)
        raise

def fetch_chunk(symbol: str, start_dt: datetime, end_dt: datetime, granularity: int):
    url = f"{CB_BASE}/products/{symbol}/candles"
    params = {
        "granularity": granularity,
        "start": start_dt.isoformat().replace("+00:00","Z"),
        "end": end_dt.isoformat().replace("+00:00","Z"),
    }
    r = requests.get(url, params=params, timeout=30)
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code} for {symbol}: {r.text}")
    data = r.json()
    # rows are [time, low, high, open, close, volume]
    cols = ["time", "low", "high", "open", "close", "volume"]
    df = pd.DataFrame(data, columns=cols)
    if df.empty:
        return df
    df["timestamp"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df["symbol"] = symbol
    # sort ascending
    df = df.sort_values("timestamp")
    return df[["timestamp","open","high","low","close","volume","symbol"]]

def download_symbol(symbol: str, start_dt: datetime, end_dt: datetime, granularity: int, pause: float) -> pd.DataFrame:
    max_span = granularity * 300  # 300 candles per request
    all_parts = []
    cur_start = start_dt
    while cur_start < end_dt:
        cur_end = min(end_dt, cur_start + timedelta(seconds=max_span))
        try:
            part = fetch_chunk(symbol, cur_start, cur_end, granularity)
            if not part.empty:
                all_parts.append(part)
        except Exception as e:
            print(f"[{symbol}] error {e} for chunk {cur_start} -> {cur_end}", file=sys.stderr)
            # brief pause and continue
            time.sleep(max(pause, 0.5))
        time.sleep(pause)
        # advance; Coinbase returns inclusive ends; shift by one granularity to avoid overlap
        cur_start = cur_end + timedelta(seconds=granularity)
    if not all_parts:
        return pd.DataFrame(columns=["timestamp","open","high","low","close","volume","symbol"])
    out = pd.concat(all_parts, ignore_index=True)
    # de-dup & sort
    out = out.drop_duplicates(subset=["symbol","timestamp"]).sort_values(["symbol","timestamp"]).reset_index(drop=True)
    # ensure numeric
    for c in ["open","high","low","close","volume"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out = out.dropna(subset=["open","high","low","close","volume"])
    return out

def main():
    args = parse_args()
    if args.granularity not in ALLOWED:
        raise SystemExit(f"granularity must be one of {sorted(ALLOWED)}")
    start_dt = to_dt(args.start)
    end_dt = to_dt(args.end)
    if end_dt <= start_dt:
        raise SystemExit("--end must be after --start")

    frames = []
    for sym in args.symbols:
        print(f"Downloading {sym} ...")
        df = download_symbol(sym, start_dt, end_dt, args.granularity, args.pause)
        print(f"{sym}: {len(df)} rows")
        frames.append(df)

    if frames:
        out = pd.concat(frames, ignore_index=True).sort_values(["symbol","timestamp"]).reset_index(drop=True)
    else:
        out = pd.DataFrame(columns=["timestamp","open","high","low","close","volume","symbol"])

    out.to_csv(args.out, index=False)
    print(f"Saved {len(out)} rows to {args.out}")

if __name__ == "__main__":
    main()
