#!/usr/bin/env python3
# backfill_stoch_rsi.py
# Adds Stoch RSI K/D (15m, 1H, 6H, 1D) + simple flags to a trade log CSV, using the last CLOSED candle <= entry time.
# Prefers using your bot module's fetch_candles(product_id, seconds). Falls back to CCXT if requested.

import argparse, os, sys, time, importlib
from datetime import datetime
from typing import Tuple, Optional, Dict, Any

import pandas as pd
import numpy as np

# ----------------------------
# Stoch RSI helpers (14,14,3,3)
# ----------------------------
def _rsi(close: pd.Series, length: int = 14) -> pd.Series:
    s = pd.Series(close, dtype="float64").copy()
    delta = s.diff()
    up = delta.clip(lower=0.0)
    dn = -delta.clip(upper=0.0)
    roll_up = up.ewm(alpha=1/length, adjust=False).mean()
    roll_dn = dn.ewm(alpha=1/length, adjust=False).mean()
    rs = roll_up / roll_dn.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def stoch_rsi_kd(close: pd.Series, rsi_len=14, stoch_len=14, k_len=3, d_len=3) -> Tuple[pd.Series, pd.Series]:
    rsi = _rsi(close, rsi_len)
    rmin = rsi.rolling(stoch_len, min_periods=stoch_len).min()
    rmax = rsi.rolling(stoch_len, min_periods=stoch_len).max()
    denom = (rmax - rmin).replace(0, np.nan)
    k = 100 * (rsi - rmin) / denom
    k = k.rolling(k_len, min_periods=k_len).mean()
    d = k.rolling(d_len, min_periods=d_len).mean()
    return k, d

def last_closed_stoch_at(
    df_ohlc: pd.DataFrame,
    ts_utc: pd.Timestamp,
    tf_seconds: int,
    rsi_len=14, stoch_len=14, k_len=3, d_len=3
) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float], Dict[str, int]]:
    """
    Compute Stoch RSI on the last CLOSED candle at or prior to ts_utc (UTC).
    df_ohlc columns expected: ['time','close'] (time ms or ISO/UTC).
    Returns (k_last, d_last, k_prev, d_prev, flags_dict) or (None,...).
    """
    if df_ohlc is None or df_ohlc.empty:
        return (None, None, None, None, {})
    # Normalize time column to UTC pandas Timestamps
    if "time" not in df_ohlc.columns:
        return (None, None, None, None, {})
    if np.issubdtype(df_ohlc["time"].dtype, np.integer):
        tcol = pd.to_datetime(df_ohlc["time"], unit="ms", utc=True)
    else:
        tcol = pd.to_datetime(df_ohlc["time"], utc=True, errors="coerce")

    df = pd.DataFrame({"time": tcol, "close": pd.to_numeric(df_ohlc["close"], errors="coerce")}).dropna()
    ts_utc = pd.to_datetime(ts_utc, utc=True, errors="coerce")
    if pd.isna(ts_utc) or df.empty:
        return (None, None, None, None, {})
    # Use only CLOSED candles at or before entry
    df = df[df["time"] <= ts_utc.floor("s")]
    if len(df) < (rsi_len + stoch_len + max(k_len, d_len) + 2):
        return (None, None, None, None, {})

    k, d = stoch_rsi_kd(df["close"], rsi_len, stoch_len, k_len, d_len)
    if k.isna().all() or d.isna().all():
        return (None, None, None, None, {})

    # Take last closed values; if NaN, back up one bar
    k_last, d_last = k.iloc[-1], d.iloc[-1]
    k_prev, d_prev = (k.iloc[-2] if len(k) >= 2 else np.nan), (d.iloc[-2] if len(d) >= 2 else np.nan)
    if any(np.isnan([k_last, d_last, k_prev, d_prev])):
        if len(k) >= 3 and not any(np.isnan([k.iloc[-2], d.iloc[-2], k.iloc[-3], d.iloc[-3]])):
            k_last, d_last = k.iloc[-2], d.iloc[-2]
            k_prev, d_prev = k.iloc[-3], d.iloc[-3]
        else:
            return (None, None, None, None, {})

    ob = (k_last > 80.0 and d_last > 80.0)
    os = (k_last < 20.0 and d_last < 20.0)
    cross_up = (k_prev <= d_prev) and (k_last > d_last)
    cross_dn = (k_prev >= d_prev) and (k_last < d_last)
    flags = {"overbought": int(ob), "oversold": int(os), "cross_up": int(cross_up), "cross_dn": int(cross_dn)}
    return (float(k_last), float(d_last), float(k_prev), float(d_prev), flags)

# ----------------------------
# Candle fetch plumbing
# ----------------------------
def import_fetcher(module_name: str):
    """
    Try to import fetch_candles(product_id, seconds) from a module (e.g., 'fib6' or 'fib5').
    """
    try:
        mod = importlib.import_module(module_name)
        if hasattr(mod, "fetch_candles"):
            return getattr(mod, "fetch_candles")
        # attempt nested path fallback
        parts = module_name.split(".")
        while parts:
            try:
                mod = importlib.import_module(".".join(parts))
                if hasattr(mod, "fetch_candles"):
                    return getattr(mod, "fetch_candles")
            except Exception:
                pass
            parts = parts[1:]
        return None
    except Exception:
        return None

class CCXTFetcher:
    def __init__(self, exchange_id: str = "kraken", rate_limit_sec: float = 0.4):
        try:
            import ccxt  # type: ignore
        except Exception:
            print("[ERROR] ccxt not installed. Run: pip install ccxt")
            raise
        self.ccxt = ccxt
        if not hasattr(ccxt, exchange_id):
            raise ValueError(f"Exchange {exchange_id} not found in ccxt")
        self.ex = getattr(ccxt, exchange_id)({"enableRateLimit": True})
        self.tf_map = {"15m": "15m", "1h": "1h", "6h": "6h", "1d": "1d"}
        self.rate_limit = rate_limit_sec

    def _norm_symbol(self, s: str) -> str:
        return str(s).replace("-", "/").upper()

    def fetch(self, symbol: str, tf_str: str, end_ts: pd.Timestamp, lookback: int = 200) -> pd.DataFrame:
        tf_ms_map = {"15m": 900_000, "1h": 3_600_000, "6h": 21_600_000, "1d": 86_400_000}
        tf_ms = tf_ms_map[tf_str]
        end_ms = int(pd.to_datetime(end_ts, utc=True).timestamp() * 1000)
        since = end_ms - (lookback + 5) * tf_ms
        mkt = self._norm_symbol(symbol)
        try:
            ohlcv = self.ex.fetch_ohlcv(mkt, timeframe=self.tf_map[tf_str], since=since, limit=lookback + 10)
        except Exception:
            # try BTC alias for XBT if needed
            if "XBT/" in mkt:
                mkt = mkt.replace("XBT/", "BTC/")
                ohlcv = self.ex.fetch_ohlcv(mkt, timeframe=self.tf_map[tf_str], since=since, limit=lookback + 10)
            else:
                raise
        time.sleep(self.rate_limit)
        if not ohlcv:
            return pd.DataFrame(columns=["time","open","high","low","close","volume"])
        return pd.DataFrame(ohlcv, columns=["time","open","high","low","close","volume"])

# ----------------------------
# Backfill core
# ----------------------------
def backfill(
    input_csv: str,
    output_csv: Optional[str],
    fetch_module: Optional[str],
    use_ccxt: bool,
    ccxt_exchange: str,
    lookback_15m: int,
    lookback_1h: int,
    lookback_6h: int,
    lookback_1d: int
):
    df = pd.read_csv(input_csv)
    if "timestamp_utc" not in df.columns:
        raise ValueError("CSV missing 'timestamp_utc' column.")
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True, errors="coerce")

    # Prepare columns (create if missing)
    new_cols = [
        "stoch15m_k","stoch15m_d","stoch15m_cross_up","stoch15m_cross_dn","stoch15m_overbought","stoch15m_oversold",
        "stoch1h_k","stoch1h_d","stoch1h_cross_up","stoch1h_cross_dn","stoch1h_overbought","stoch1h_oversold",
        "stoch6h_k","stoch6h_d","stoch1d_k","stoch1d_d"
    ]
    for c in new_cols:
        if c not in df.columns:
            df[c] = np.nan

    # Only rows with valid timestamps
    rows = df[df["timestamp_utc"].notna()].copy()

    # Choose fetcher path
    fetch_fn = None
    fetcher = None
    if fetch_module:
        fetch_fn = import_fetcher(fetch_module)
        if fetch_fn is None:
            print(f"[WARN] Could not import fetch_candles from '{fetch_module}'. Falling back to CCXT.")
            use_ccxt = True
    if use_ccxt:
        fetcher = CCXTFetcher(exchange_id=ccxt_exchange)

    # Simple per-call fetch helper
    def get_candles(symbol: str, tf_seconds: int, entry_ts: pd.Timestamp, lookback: int) -> pd.DataFrame:
        tf_map = {900:"15m", 3600:"1h", 21600:"6h", 86400:"1d"}
        tf_str = tf_map[tf_seconds]
        if fetch_fn is not None:
            try:
                candles = fetch_fn(symbol, tf_seconds)
                dfc = candles if isinstance(candles, pd.DataFrame) else pd.DataFrame(candles)
                if "time" not in dfc.columns:
                    if isinstance(dfc.index, pd.DatetimeIndex):
                        dfc = dfc.reset_index().rename(columns={"index":"time"})
                    elif "timestamp" in dfc.columns:
                        dfc = dfc.rename(columns={"timestamp":"time"})
                keep = [c for c in ["time","open","high","low","close","volume"] if c in dfc.columns]
                return dfc[keep]
            except Exception as e:
                print(f"[WARN] module fetch_candles failed for {symbol} tf={tf_seconds}: {e}. Trying CCXT.")
                if fetcher is None:
                    raise
        # CCXT path
        assert fetcher is not None, "CCXT fetcher not initialized"
        norm = symbol.replace("-", "/")
        return fetcher.fetch(norm, tf_str, entry_ts, lookback)

    total = len(rows)
    for i, (idx, r) in enumerate(rows.iterrows(), start=1):
        ts = r["timestamp_utc"]
        sym_mod = str(r.get("symbol", r.get("product_id", "")))  # module/bot uses "-" form
        sym_ccx = sym_mod.replace("-", "/")                      # ccxt uses "/"
        # 15m
        df15 = get_candles(sym_mod if fetch_fn else sym_ccx, 900, ts, lookback_15m)
        k15, d15, _, _, f15 = last_closed_stoch_at(df15, ts, 900)
        if k15 is not None:
            df.at[idx, "stoch15m_k"] = k15
            df.at[idx, "stoch15m_d"] = d15
            df.at[idx, "stoch15m_cross_up"] = f15.get("cross_up")
            df.at[idx, "stoch15m_cross_dn"] = f15.get("cross_dn")
            df.at[idx, "stoch15m_overbought"] = f15.get("overbought")
            df.at[idx, "stoch15m_oversold"]   = f15.get("oversold")
        # 1h
        df1h = get_candles(sym_mod if fetch_fn else sym_ccx, 3600, ts, lookback_1h)
        k1h, d1h, _, _, f1h = last_closed_stoch_at(df1h, ts, 3600)
        if k1h is not None:
            df.at[idx, "stoch1h_k"] = k1h
            df.at[idx, "stoch1h_d"] = d1h
            df.at[idx, "stoch1h_cross_up"] = f1h.get("cross_up")
            df.at[idx, "stoch1h_cross_dn"] = f1h.get("cross_dn")
            df.at[idx, "stoch1h_overbought"] = f1h.get("overbought")
            df.at[idx, "stoch1h_oversold"]   = f1h.get("oversold")
        # 6h
        df6h = get_candles(sym_mod if fetch_fn else sym_ccx, 21600, ts, lookback_6h)
        k6h, d6h, *_ = last_closed_stoch_at(df6h, ts, 21600)
        if k6h is not None:
            df.at[idx, "stoch6h_k"] = k6h
            df.at[idx, "stoch6h_d"] = d6h
        # 1d
        df1d = get_candles(sym_mod if fetch_fn else sym_ccx, 86400, ts, lookback_1d)
        k1d, d1d, *_ = last_closed_stoch_at(df1d, ts, 86400)
        if k1d is not None:
            df.at[idx, "stoch1d_k"] = k1d
            df.at[idx, "stoch1d_d"] = d1d

        if (i % 50 == 0) or (i == total):
            print(f"[{i}/{total}] backfilled")

    out_path = output_csv or input_csv.replace(".csv", "_stochfilled.csv")
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"[OK] wrote {out_path}")

# ----------------------------
# CLI
# ----------------------------
def main():
    ap = argparse.ArgumentParser(description="Backfill Stoch RSI (15m/1h/6h/1d) into trade log CSV (last closed candle at/<= entry time).")
    ap.add_argument("--input", required=True, help="Trade log CSV (e.g., live_trade_log_fib6.csv)")
    ap.add_argument("--output", default=None, help="Output CSV path (default: *_stochfilled.csv)")
    ap.add_argument("--fetch-module", default=None, help="Module exposing fetch_candles(product_id, seconds), e.g. fib6 or fib5")
    ap.add_argument("--use-ccxt", action="store_true", help="Use CCXT fallback for candles (pip install ccxt)")
    ap.add_argument("--ccxt-exchange", default="kraken", help="CCXT exchange id: kraken, coinbase, binance, etc.")
    ap.add_argument("--lookback-15m", type=int, default=400, help="Bars to fetch for 15m TF")
    ap.add_argument("--lookback-1h",  type=int, default=200, help="Bars to fetch for 1h TF")
    ap.add_argument("--lookback-6h",  type=int, default=200, help="Bars to fetch for 6h TF")
    ap.add_argument("--lookback-1d",  type=int, default=200, help="Bars to fetch for 1d TF")
    args = ap.parse_args()

    backfill(
        input_csv=args.input,
        output_csv=args.output,
        fetch_module=args.fetch_module,
        use_ccxt=args.use_ccxt,
        ccxt_exchange=args.ccxt_exchange,
        lookback_15m=args.lookback_15m,
        lookback_1h=args.lookback_1h,
        lookback_6h=args.lookback_6h,
        lookback_1d=args.lookback_1d
    )

if __name__ == "__main__":
    main()
