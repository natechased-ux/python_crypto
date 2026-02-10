
"""
Bybit Data Fetcher — OHLCV + Funding Rates (backtest-ready CSVs)

Requirements
------------
pip install ccxt pandas python-dateutil

What it does
------------
- Downloads 1H candles and funding rate history from Bybit (USDT perpetuals).
- Merges by timestamp (UTC). Funding is forward-filled to 1H.
- Saves one CSV per symbol with columns:
  timestamp, open, high, low, close, volume, funding_rate

Usage
-----
python bybit_data_fetcher.py \
  --symbols BTCUSDT ETHUSDT XRPUSDT \
  --since "2024-10-01" \
  --until "2025-08-25" \
  --out ./data

Notes
-----
- Symbols are USDT perps. You can pass BTCUSDT or BTC/USDT:USDT — both work.
- If you omit dates, it pulls ~6 months by default.
- Respects ccxt rate limits.
"""

from __future__ import annotations
import argparse
import time
from datetime import datetime, timezone
from dateutil import parser as dateparser
from typing import List, Dict
import pandas as pd
import ccxt

TIMEFRAME = '1h'

def parse_date(s: str | None) -> int | None:
    if not s:
        return None
    dt = dateparser.parse(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    return int(dt.timestamp() * 1000)

def norm_symbol_for_ccxt(sym: str) -> str:
    s = sym.strip().upper().replace('-', '').replace('_','')
    # Accept already ccxt style "BTC/USDT:USDT"
    if '/' in sym and ':USDT' in sym.upper():
        return sym
    # Map common forms to Bybit USDT perp notation
    # e.g., BTCUSDT -> BTC/USDT:USDT
    if s.endswith('USDT'):
        base = s[:-4]
        return f"{base}/USDT:USDT"
    # Fallback if user passed "BTC/USDT"
    if '/' in sym and ':USDT' not in sym.upper():
        return sym + ':USDT'
    return sym  # last resort

def file_symbol_name(sym: str) -> str:
    # Make a clean filename like btcusdt.csv
    s = sym.upper().replace('/', '').replace(':USDT','').replace('-','').replace('_','')
    return s.lower()

def fetch_all_ohlcv(ex, symbol: str, since_ms: int | None, until_ms: int | None) -> pd.DataFrame:
    limit = 1000  # bybit supports up to 1000
    all_rows = []
    cursor = since_ms
    while True:
        rows = ex.fetch_ohlcv(symbol, timeframe=TIMEFRAME, since=cursor, limit=limit)
        if not rows:
            break
        all_rows.extend(rows)
        last_ts = rows[-1][0]
        # Next since is last_ts + 1ms to avoid duplicates
        cursor = last_ts + 1
        # Stop if we reached until_ms
        if until_ms and last_ts >= until_ms:
            break
        # Respect rate limit
        time.sleep(ex.rateLimit / 1000.0)
        # Safety: if fewer than 2 rows returned repeatedly, break
        if len(rows) < 2 and cursor and since_ms:
            break
    if not all_rows:
        return pd.DataFrame(columns=['timestamp','open','high','low','close','volume'])
    df = pd.DataFrame(all_rows, columns=['timestamp','open','high','low','close','volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
    return df

def fetch_all_funding(ex, symbol: str, since_ms: int | None, until_ms: int | None) -> pd.DataFrame:
    limit = 200  # typical page size
    all_rows = []
    cursor = since_ms
    while True:
        data = ex.fetch_funding_rate_history(symbol, since=cursor, limit=limit, params={})
        if not data:
            break
        for r in data:
            # ccxt uses {'symbol','fundingRate','timestamp',...}
            all_rows.append([r.get('timestamp'), r.get('fundingRate')])
        last_ts = data[-1].get('timestamp')
        if last_ts is None:
            break
        cursor = last_ts + 1
        if until_ms and last_ts >= until_ms:
            break
        time.sleep(ex.rateLimit / 1000.0)
        if len(data) < 2 and cursor and since_ms:
            break
    if not all_rows:
        return pd.DataFrame(columns=['timestamp','funding_rate'])
    df = pd.DataFrame(all_rows, columns=['timestamp','funding_rate'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
    return df.drop_duplicates('timestamp').sort_values('timestamp')

def resample_to_1h(df: pd.DataFrame) -> pd.DataFrame:
    # Ensure 1h frequency and fill gaps if any (OHLCV)
    df = df.set_index('timestamp').sort_index()
    # If already 1h, just ensure types
    df = df[['open','high','low','close','volume']].astype(float)
    return df

def merge_ohlcv_funding(ohlcv: pd.DataFrame, funding: pd.DataFrame) -> pd.DataFrame:
    if funding is None or funding.empty:
        funding = pd.DataFrame(columns=['timestamp','funding_rate'])
    f = funding.set_index('timestamp').sort_index()
    # Forward-fill funding to 1H grid based on OHLCV timestamps
    merged = ohlcv.copy()
    f = f.reindex(merged.index.union(f.index)).sort_index().ffill()
    merged['funding_rate'] = f['funding_rate'].reindex(merged.index).ffill().fillna(0.0)
    merged = merged.reset_index().rename(columns={'index':'timestamp'})
    return merged

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--symbols', nargs='+', required=True, help='Symbols e.g. BTCUSDT ETHUSDT or BTC/USDT:USDT')
    ap.add_argument('--since', type=str, default=None, help='Start date (e.g., 2024-10-01)')
    ap.add_argument('--until', type=str, default=None, help='End date (e.g., 2025-08-25)')
    ap.add_argument('--out', type=str, default='./data', help='Output folder')
    args = ap.parse_args()

    since_ms = parse_date(args.since)
    until_ms = parse_date(args.until)
    if since_ms is None and until_ms is None:
        # default: ~6 months back
        now = int(datetime.now(timezone.utc).timestamp() * 1000)
        six_months_ms = 180 * 24 * 60 * 60 * 1000
        since_ms = now - six_months_ms

    ex = ccxt.bybit({
        'enableRateLimit': True,
        'options': {
            'defaultType': 'swap',  # perpetual swaps
            'fetchMarkets': ['swap'],  # IMPORTANT: avoid SPOT endpoints (403 in some regions)
        }
    })
    # Load only swap markets to avoid Spot endpoints being blocked (403)
    markets = ex.load_markets()

    os.makedirs(args.out, exist_ok=True)

    for raw in args.symbols:
        ccxt_sym = norm_symbol_for_ccxt(raw)
        if ccxt_sym not in markets:
            # Try uppercase base inference
            try_alt = ccxt_sym.upper()
            if try_alt in markets:
                ccxt_sym = try_alt
            else:
                print(f"[WARN] Symbol not found on Bybit: {raw} -> tried '{ccxt_sym}'")
                continue

        print(f"[INFO] Fetching {ccxt_sym} {TIMEFRAME} OHLCV ...")
        ohlcv = fetch_all_ohlcv(ex, ccxt_sym, since_ms, until_ms)
        if ohlcv.empty:
            print(f"[WARN] No OHLCV for {ccxt_sym}. Skipping.")
            continue
        ohlcv = resample_to_1h(ohlcv)

        print(f"[INFO] Fetching {ccxt_sym} funding rate history ...")
        funding = fetch_all_funding(ex, ccxt_sym, since_ms, until_ms)

        merged = merge_ohlcv_funding(ohlcv, funding)

        fname = file_symbol_name(ccxt_sym) + '.csv'
        out_path = os.path.join(args.out, fname)
        merged.to_csv(out_path, index=False)
        print(f"[OK] Saved {out_path}  rows={len(merged)}")

if __name__ == '__main__':
    main()
