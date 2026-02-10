
"""
Bybit Data Fetcher (USDT Perp, 'linear' market type) — OHLCV + Funding Rates

This version fixes the previous error by using Bybit's supported market type "linear"
instead of "swap", and avoids touching spot endpoints during market loading.

Usage (Windows):
py bybit_data_fetcher_linear.py --symbols BTCUSDT ETHUSDT XRPUSDT --since "2024-10-01" --until "2025-08-25" --out data
"""
from __future__ import annotations
import argparse, time, os
from datetime import datetime, timezone
from dateutil import parser as dateparser
import pandas as pd
import ccxt

TIMEFRAME = '1h'

def parse_date(s):
    if not s: return None
    dt = dateparser.parse(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    return int(dt.timestamp() * 1000)

def norm_symbol_for_ccxt(sym: str) -> str:
    s = sym.strip().upper().replace('-', '').replace('_','')
    if '/' in sym and ':USDT' in sym.upper():
        return sym
    if s.endswith('USDT'):
        base = s[:-4]
        return f"{base}/USDT:USDT"
    if '/' in sym and ':USDT' not in sym.upper():
        return sym + ':USDT'
    return sym

def file_symbol_name(sym: str) -> str:
    return sym.upper().replace('/', '').replace(':USDT','').replace('-','').replace('_','').lower()

def fetch_all_ohlcv(ex, symbol, since_ms, until_ms):
    limit = 1000
    out = []
    cursor = since_ms
    while True:
        rows = ex.fetch_ohlcv(symbol, timeframe=TIMEFRAME, since=cursor, limit=limit, params={'category':'linear'})
        if not rows:
            break
        out.extend(rows)
        last_ts = rows[-1][0]
        cursor = last_ts + 1
        if until_ms and last_ts >= until_ms:
            break
        time.sleep(ex.rateLimit / 1000.0)
        if len(rows) < 2:
            break
    if not out:
        return pd.DataFrame(columns=['timestamp','open','high','low','close','volume'])
    df = pd.DataFrame(out, columns=['timestamp','open','high','low','close','volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
    return df

def fetch_all_funding(ex, symbol, since_ms, until_ms):
    limit = 200
    out = []
    cursor = since_ms
    while True:
        data = ex.fetch_funding_rate_history(symbol, since=cursor, limit=limit, params={'category':'linear'})
        if not data:
            break
        for r in data:
            out.append([r.get('timestamp'), r.get('fundingRate')])
        last_ts = data[-1].get('timestamp')
        if last_ts is None:
            break
        cursor = last_ts + 1
        if until_ms and last_ts >= until_ms:
            break
        time.sleep(ex.rateLimit / 1000.0)
        if len(data) < 2:
            break
    if not out:
        return pd.DataFrame(columns=['timestamp','funding_rate'])
    df = pd.DataFrame(out, columns=['timestamp','funding_rate'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
    return df.drop_duplicates('timestamp').sort_values('timestamp')

def resample_ohlcv(df):
    df = df.set_index('timestamp').sort_index()
    return df[['open','high','low','close','volume']].astype(float)

def merge_ohlcv_funding(ohlcv, funding):
    f = funding.set_index('timestamp').sort_index() if not funding.empty else pd.DataFrame(columns=['funding_rate'])
    merged = ohlcv.copy()
    f = f.reindex(merged.index.union(f.index)).sort_index().ffill()
    merged['funding_rate'] = f['funding_rate'].reindex(merged.index).ffill().fillna(0.0)
    return merged.reset_index()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--symbols', nargs='+', required=True)
    ap.add_argument('--since', type=str, default=None)
    ap.add_argument('--until', type=str, default=None)
    ap.add_argument('--out', type=str, default='./data')
    args = ap.parse_args()

    since_ms = parse_date(args.since)
    until_ms = parse_date(args.until)
    if since_ms is None and until_ms is None:
        now = int(datetime.now(timezone.utc).timestamp() * 1000)
        since_ms = now - 180*24*60*60*1000

    ex = ccxt.bybit({
        'enableRateLimit': True,
        'options': {
            # Bybit supports 'linear' (USDT-margined swaps) and 'inverse' and 'spot'
            'defaultType': 'linear',
        }
    })
    # Load only linear markets (avoid spot)
    markets = ex.load_markets({'type': 'linear'})

    os.makedirs(args.out, exist_ok=True)

    for raw in args.symbols:
        sym = norm_symbol_for_ccxt(raw)
        if sym not in markets:
            print(f"[WARN] Symbol not found in loaded 'linear' markets: {raw} -> {sym}")
            continue
        print(f"[INFO] Fetching {sym} {TIMEFRAME} OHLCV (linear) ...")
        ohlcv = fetch_all_ohlcv(ex, sym, since_ms, until_ms)
        if ohlcv.empty:
            print(f"[WARN] No OHLCV for {sym}. Skipping.")
            continue
        ohlcv = resample_ohlcv(ohlcv)

        print(f"[INFO] Fetching {sym} funding history (linear) ...")
        funding = fetch_all_funding(ex, sym, since_ms, until_ms)

        merged = merge_ohlcv_funding(ohlcv, funding)
        out_path = os.path.join(args.out, file_symbol_name(sym) + '.csv')
        merged.to_csv(out_path, index=False)
        print(f"[OK] Saved {out_path}  rows={len(merged)}")

if __name__ == '__main__':
    main()
