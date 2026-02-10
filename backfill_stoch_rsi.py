#!/usr/bin/env python3
# backfill_stoch_rsi.py
# Adds Stoch RSI (K/D) for 15m/1h/6h/1d into a trade log CSV using the last CLOSED candle <= timestamp_utc.
# Robust: symbol normalization (XBT->BTC), TF auto-resampling (6h from 1h), paging + retries to avoid DDoS.
# Requires: pip install ccxt pandas numpy

import os, sys, time, math, argparse, random
from typing import Optional, Tuple, Dict, List
import numpy as np
import pandas as pd
from pandas.api.types import is_integer_dtype, is_float_dtype

# ===========
# EDIT HERE 👇
# ===========
CSV_PATH               = "live_trade_log_fib6_maemfe.csv"   # <-- change to your CSV
OUTPUT_CSV             = None                        # None -> writes *_stochfilled.csv next to CSV_PATH
EXCHANGE_ID            = "kraken"                    # e.g. "kraken", "coinbase", "binance"
ONLY_BACKFILL_MISSING  = True                        # True: only fill rows missing stoch cols
LOOKBACK_15M           = 400
LOOKBACK_1H            = 200
LOOKBACK_6H            = 200
LOOKBACK_1D            = 200
RATE_LIMIT_SEC         = 0.4                         # polite pacing between HTTP calls
MAX_CALLS_PER_WINDOW   = 800                         # per symbol+tf; paging will chunk under the hood
MAX_RETRIES            = 7                           # exponential backoff retries
BACKOFF_BASE_SEC       = 0.8                         # base for backoff
JITTER_SEC             = 0.25                        # random jitter added to waits

# -------------
# StochRSI core
# -------------
def _rsi(close: pd.Series, length: int = 14) -> pd.Series:
    s = pd.Series(close, dtype="float64").copy()
    delta = s.diff()
    up = delta.clip(lower=0.0)
    dn = -delta.clip(upper=0.0)
    roll_up = up.ewm(alpha=1/length, adjust=False).mean()
    roll_dn = dn.ewm(alpha=1/length, adjust=False).mean()
    rs = roll_up / roll_dn.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def stoch_rsi_kd(close: pd.Series, rsi_len=14, stoch_len=14, k_len=3, d_len=3) -> Tuple[pd.Series, pd.Series]:
    rsi = _rsi(close, rsi_len)
    rmin = rsi.rolling(stoch_len, min_periods=stoch_len).min()
    rmax = rsi.rolling(stoch_len, min_periods=stoch_len).max()
    denom = (rmax - rmin).replace(0, np.nan)
    k = 100 * (rsi - rmin) / denom
    k = k.rolling(k_len, min_periods=k_len).mean()
    d = k.rolling(d_len, min_periods=d_len).mean()
    return k, d

def _to_utc(series: pd.Series) -> pd.Series:
    # robust coercion to UTC Timestamps (handles ms ints and tz-aware datetimes)
    if is_integer_dtype(series) or is_float_dtype(series):
        return pd.to_datetime(series, unit="ms", utc=True, errors="coerce")
    return pd.to_datetime(series, utc=True, errors="coerce")

def last_closed_stoch_at(
    df_ohlc: pd.DataFrame,
    ts_utc: pd.Timestamp,
    tf_seconds: int,
    rsi_len=14, stoch_len=14, k_len=3, d_len=3
) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float], Dict[str, int]]:
    if df_ohlc is None or df_ohlc.empty or "close" not in df_ohlc.columns:
        return (None, None, None, None, {})
    if "time" not in df_ohlc.columns:
        if isinstance(df_ohlc.index, pd.DatetimeIndex):
            df_ohlc = df_ohlc.reset_index().rename(columns={"index": "time"})
        elif "timestamp" in df_ohlc.columns:
            df_ohlc = df_ohlc.rename(columns={"timestamp": "time"})
        else:
            return (None, None, None, None, {})
    tcol = _to_utc(df_ohlc["time"])
    df = pd.DataFrame({"time": tcol, "close": pd.to_numeric(df_ohlc["close"], errors="coerce")}).dropna()
    ts_utc = pd.to_datetime(ts_utc, utc=True, errors="coerce")
    if pd.isna(ts_utc) or df.empty:
        return (None, None, None, None, {})
    df = df[df["time"] <= ts_utc.floor("s")]  # only CLOSED candles at/<= entry
    if len(df) < (rsi_len + stoch_len + max(k_len, d_len) + 2):
        return (None, None, None, None, {})
    k, d = stoch_rsi_kd(df["close"], rsi_len, stoch_len, k_len, d_len)
    if k.isna().all() or d.isna().all():
        return (None, None, None, None, {})
    k_last, d_last = k.iloc[-1], d.iloc[-1]
    k_prev, d_prev = (k.iloc[-2] if len(k) >= 2 else np.nan), (d.iloc[-2] if len(d) >= 2 else np.nan)
    if any(np.isnan([k_last, d_last, k_prev, d_prev])) and len(k) >= 3:
        k_last, d_last = k.iloc[-2], d.iloc[-2]
        k_prev, d_prev = k.iloc[-3], d.iloc[-3]
        if any(np.isnan([k_last, d_last, k_prev, d_prev])):
            return (None, None, None, None, {})
    ob = int(k_last > 80 and d_last > 80)
    os = int(k_last < 20 and d_last < 20)
    cross_up = int((k_prev <= d_prev) and (k_last > d_last))
    cross_dn = int((k_prev >= d_prev) and (k_last < d_last))
    return (float(k_last), float(d_last), float(k_prev), float(d_prev), {
        "overbought": ob, "oversold": os, "cross_up": cross_up, "cross_dn": cross_dn
    })

# -----------
# CCXT fetcher with paging, retries, resampling, symbol normalization
# -----------
class CCXTFetcher:
    def __init__(self, exchange_id: str, rate_limit_sec: float = RATE_LIMIT_SEC):
        try:
            import ccxt  # type: ignore
        except Exception:
            print("[ERROR] ccxt not installed. Run: pip install ccxt")
            raise
        self.ccxt = ccxt
        if not hasattr(ccxt, exchange_id):
            raise ValueError(f"Exchange {exchange_id} not found in ccxt")
        self.ex = getattr(ccxt, exchange_id)({"enableRateLimit": True})
        self.rate_limit = rate_limit_sec
        try:
            self.ex.load_markets()
        except Exception:
            pass

    def _norm_symbol(self, s: str) -> str:
        # "HBAR-USD" -> "HBAR/USD"; Kraken aliases -> CCXT unified
        s = str(s).replace("-", "/").upper()
        parts = s.split("/")
        base, quote = (parts + ["USD"])[:2]
        alias = {"XBT": "BTC", "XDG": "DOGE"}  # extend if you see others
        base = alias.get(base, base)
        quote = alias.get(quote, quote)
        norm = f"{base}/{quote}"
        # If still not found, try BTC/USDT for stablecoins
        if hasattr(self.ex, "markets") and self.ex.markets:
            if norm not in self.ex.markets and quote == "USD" and f"{base}/USDT" in self.ex.markets:
                norm = f"{base}/USDT"
        return norm

    def _supports(self, tf: str) -> bool:
        tfs = getattr(self.ex, "timeframes", None)
        return bool(tfs and tf in tfs)

    def _fetch_native_once(self, mkt: str, tf: str, since_ms: int, limit: int) -> pd.DataFrame:
        for attempt in range(MAX_RETRIES):
            try:
                ohlcv = self.ex.fetch_ohlcv(mkt, timeframe=tf, since=since_ms, limit=limit)
                time.sleep(self.rate_limit)
                break
            except (self.ccxt.DDoSProtection, self.ccxt.RateLimitExceeded, self.ccxt.NetworkError, self.ccxt.ExchangeNotAvailable) as e:
                wait = BACKOFF_BASE_SEC * (2 ** attempt) + random.uniform(0, JITTER_SEC)
                print(f"[RATE] {e.__class__.__name__} waiting {wait:.2f}s (attempt {attempt+1}/{MAX_RETRIES})")
                time.sleep(wait)
            except Exception as e:
                # Unknown; don't spin forever
                print(f"[WARN] fetch_ohlcv error {e}; breaking")
                ohlcv = []
                break
        else:
            ohlcv = []
        if not ohlcv:
            return pd.DataFrame(columns=["time","open","high","low","close","volume"])
        return pd.DataFrame(ohlcv, columns=["time","open","high","low","close","volume"])

    def _page_window(self, mkt: str, tf: str, since_ms: int, until_ms: int, tf_ms: int, per_call_limit: int) -> pd.DataFrame:
        frames = []
        cursor = since_ms
        calls = 0
        max_calls = MAX_CALLS_PER_WINDOW
        # empiric per-call cap
        limit = min(per_call_limit, 1500)
        while cursor <= until_ms and calls < max_calls:
            df = self._fetch_native_once(mkt, tf, cursor, limit)
            calls += 1
            if df.empty:
                # advance cursor to avoid stuck loop
                cursor += tf_ms * limit
            else:
                # append and move cursor forward to last+1
                last_ms = int(pd.to_datetime(df["time"], unit="ms", utc=True).astype("int64").iloc[-1] // 10**6)
                frames.append(df)
                cursor = last_ms + tf_ms
            # mild pacing
            # (already sleeping in _fetch_native_once)
        if frames:
            out = pd.concat(frames, ignore_index=True)
        else:
            out = pd.DataFrame(columns=["time","open","high","low","close","volume"])
        return out

    @staticmethod
    def _resample_ohlcv(df: pd.DataFrame, target_tf: str) -> pd.DataFrame:
        if df.empty: return df
        rule_map = {"1m":"1min","5m":"5min","15m":"15min","30m":"30min","1h":"1h","4h":"4h","6h":"6h","1d":"1d"}
        rule = rule_map.get(target_tf, None)
        if rule is None: return df
        df = df.copy()
        df["time"] = _to_utc(df["time"])
        df = df.dropna(subset=["time","open","high","low","close","volume"]).set_index("time")
        o = df["open"].resample(rule, label="right", closed="right").first()
        h = df["high"].resample(rule, label="right", closed="right").max()
        l = df["low"].resample(rule, label="right", closed="right").min()
        c = df["close"].resample(rule, label="right", closed="right").last()
        v = df["volume"].resample(rule, label="right", closed="right").sum()
        out = pd.DataFrame({"time": o.index, "open": o.values, "high": h.values, "low": l.values, "close": c.values, "volume": v.values})
        return out.dropna().reset_index(drop=True)

    def fetch_window(self, symbol: str, tf_str: str, start_ts: pd.Timestamp, end_ts: pd.Timestamp, lookback_bars: int) -> pd.DataFrame:
        mkt = self._norm_symbol(symbol)
        tf_ms_map = {"1m":60_000,"5m":300_000,"15m":900_000,"30m":1_800_000,"1h":3_600_000,"4h":14_400_000,"6h":21_600_000,"1d":86_400_000}
        # choose native or fallback base TF
        if tf_str == "6h" and not self._supports("6h"):
            base_tf = "1h" if self._supports("1h") else "30m"
            base_ms = tf_ms_map[base_tf]
            since_ms = int((pd.to_datetime(start_ts, utc=True) - pd.Timedelta(milliseconds=base_ms * (lookback_bars*6 + 10))).timestamp()*1000)
            until_ms = int(pd.to_datetime(end_ts, utc=True).timestamp()*1000)
            base = self._page_window(mkt, base_tf, since_ms, until_ms, base_ms, per_call_limit=1500)
            return self._resample_ohlcv(base, "6h")
        if tf_str == "15m" and not self._supports("15m"):
            base_tf = "5m" if self._supports("5m") else ("1m" if self._supports("1m") else "1h")
            base_ms = tf_ms_map.get(base_tf, 60_000)
            since_ms = int((pd.to_datetime(start_ts, utc=True) - pd.Timedelta(milliseconds=base_ms * (lookback_bars*3 + 10))).timestamp()*1000)
            until_ms = int(pd.to_datetime(end_ts, utc=True).timestamp()*1000)
            base = self._page_window(mkt, base_tf, since_ms, until_ms, base_ms, per_call_limit=1500)
            return self._resample_ohlcv(base, "15m")
        if tf_str == "1d" and not self._supports("1d"):
            base_tf = "1h"
            base_ms = tf_ms_map["1h"]
            since_ms = int((pd.to_datetime(start_ts, utc=True) - pd.Timedelta(milliseconds=base_ms * (lookback_bars*24 + 10))).timestamp()*1000)
            until_ms = int(pd.to_datetime(end_ts, utc=True).timestamp()*1000)
            base = self._page_window(mkt, base_tf, since_ms, until_ms, base_ms, per_call_limit=1000)
            return self._resample_ohlcv(base, "1d")
        # native path with paging
        if not self._supports(tf_str):
            # As a last resort, ask native anyway with 1h paging
            base_tf = "1h"
            base_ms = tf_ms_map["1h"]
            since_ms = int((pd.to_datetime(start_ts, utc=True) - pd.Timedelta(milliseconds=base_ms * (lookback_bars + 10))).timestamp()*1000)
            until_ms = int(pd.to_datetime(end_ts, utc=True).timestamp()*1000)
            base = self._page_window(mkt, base_tf, since_ms, until_ms, base_ms, per_call_limit=1000)
            return base
        tf_ms = tf_ms_map[tf_str]
        # add safety history to compute stoch smoothing
        since_ms = int((pd.to_datetime(start_ts, utc=True) - pd.Timedelta(milliseconds=tf_ms * (lookback_bars + 20))).timestamp()*1000)
        until_ms = int(pd.to_datetime(end_ts, utc=True).timestamp()*1000)
        return self._page_window(mkt, tf_str, since_ms, until_ms, tf_ms, per_call_limit=1000)

# ----------------
# Backfill routine (batched per symbol & TF)
# ----------------
TF_SECONDS = { "15m": 900, "1h": 3600, "6h": 21600, "1d": 86400 }

def backfill(csv_path: str,
             output_csv: Optional[str] = None,
             exchange_id: str = EXCHANGE_ID,
             lookbacks: Dict[int, int] = None,
             only_missing: bool = ONLY_BACKFILL_MISSING):
    if lookbacks is None:
        lookbacks = {900: LOOKBACK_15M, 3600: LOOKBACK_1H, 21600: LOOKBACK_6H, 86400: LOOKBACK_1D}

    df = pd.read_csv(csv_path)
    if "timestamp_utc" not in df.columns:
        raise ValueError("CSV missing 'timestamp_utc' column.")
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True, errors="coerce")

    # Ensure columns exist
    new_cols = [
        "stoch15m_k","stoch15m_d","stoch15m_cross_up","stoch15m_cross_dn","stoch15m_overbought","stoch15m_oversold",
        "stoch1h_k","stoch1h_d","stoch1h_cross_up","stoch1h_cross_dn","stoch1h_overbought","stoch1h_oversold",
        "stoch6h_k","stoch6h_d","stoch1d_k","stoch1d_d"
    ]
    for c in new_cols:
        if c not in df.columns:
            df[c] = np.nan

    mask = df["timestamp_utc"].notna()
    if only_missing:
        need_mask = mask & (
            df["stoch15m_k"].isna() | df["stoch1h_k"].isna() | df["stoch6h_k"].isna() | df["stoch1d_k"].isna()
        )
    else:
        need_mask = mask
    rows = df[need_mask].copy()
    if rows.empty:
        out = OUTPUT_CSV or output_csv or csv_path.replace(".csv", "_stochfilled.csv")
        df.to_csv(out, index=False, encoding="utf-8-sig")
        print(f"[OK] nothing to backfill; wrote {out}")
        return

    fetcher = CCXTFetcher(exchange_id=exchange_id, rate_limit_sec=RATE_LIMIT_SEC)

    # Group rows by symbol so we can batch-fetch per TF
    by_symbol = { sym: g.sort_values("timestamp_utc") for sym, g in rows.groupby("symbol") }

    # cache per (symbol, tf_str): DataFrame
    cache: Dict[Tuple[str,str], pd.DataFrame] = {}

    # safety bars for Stoch smoothing
    SAFETY_BARS = 40

    # Fetch windows per symbol & TF
    for sym, g in by_symbol.items():
        start_ts = g["timestamp_utc"].min() - pd.Timedelta(hours=48)
        end_ts   = g["timestamp_utc"].max() + pd.Timedelta(hours=48)
        for tf_str in ["15m","1h","6h","1d"]:
            tf_sec = TF_SECONDS[tf_str]
            look = lookbacks[tf_sec] + SAFETY_BARS
            try:
                df_tf = fetcher.fetch_window(sym, tf_str, start_ts, end_ts, look)
            except Exception as e:
                print(f"[WARN] fetch_window failed for {sym} {tf_str}: {e}; skipping TF")
                df_tf = pd.DataFrame(columns=["time","open","high","low","close","volume"])
            cache[(sym, tf_str)] = df_tf

    # Now compute per-row using the cached frames
    total = len(rows)
    for i, (idx, r) in enumerate(rows.iterrows(), start=1):
        ts = r["timestamp_utc"]; sym = r["symbol"]

        # 15m
        d15 = cache.get((sym, "15m"), pd.DataFrame())
        k15, d15k, _, _, f15 = last_closed_stoch_at(d15, ts, TF_SECONDS["15m"])
        if k15 is not None:
            df.at[idx, "stoch15m_k"] = k15
            df.at[idx, "stoch15m_d"] = d15k
            df.at[idx, "stoch15m_cross_up"] = f15.get("cross_up")
            df.at[idx, "stoch15m_cross_dn"] = f15.get("cross_dn")
            df.at[idx, "stoch15m_overbought"] = f15.get("overbought")
            df.at[idx, "stoch15m_oversold"]   = f15.get("oversold")

        # 1h
        d1h = cache.get((sym, "1h"), pd.DataFrame())
        k1h, d1hk, _, _, f1h = last_closed_stoch_at(d1h, ts, TF_SECONDS["1h"])
        if k1h is not None:
            df.at[idx, "stoch1h_k"] = k1h
            df.at[idx, "stoch1h_d"] = d1hk
            df.at[idx, "stoch1h_cross_up"] = f1h.get("cross_up")
            df.at[idx, "stoch1h_cross_dn"] = f1h.get("cross_dn")
            df.at[idx, "stoch1h_overbought"] = f1h.get("overbought")
            df.at[idx, "stoch1h_oversold"]   = f1h.get("oversold")

        # 6h
        d6h = cache.get((sym, "6h"), pd.DataFrame())
        k6h, d6hk, *_ = last_closed_stoch_at(d6h, ts, TF_SECONDS["6h"])
        if k6h is not None:
            df.at[idx, "stoch6h_k"] = k6h
            df.at[idx, "stoch6h_d"] = d6hk

        # 1d
        d1d = cache.get((sym, "1d"), pd.DataFrame())
        k1d, d1dk, *_ = last_closed_stoch_at(d1d, ts, TF_SECONDS["1d"])
        if k1d is not None:
            df.at[idx, "stoch1d_k"] = k1d
            df.at[idx, "stoch1d_d"] = d1dk

        if (i % 50 == 0) or (i == total):
            print(f"[{i}/{total}] backfilled")

    out = OUTPUT_CSV or output_csv or csv_path.replace(".csv", "_stochfilled.csv")
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"[OK] wrote {out}")

# ----
# CLI (optional). If you run without args, it uses the constants above.
# ----
def main():
    ap = argparse.ArgumentParser(description="Backfill Stoch RSI (15m/1h/6h/1d) into a trade log CSV (last closed candle at/<= timestamp).")
    ap.add_argument("--input", help=f"CSV path (default: {CSV_PATH})")
    ap.add_argument("--output", help="Output CSV path (default: *_stochfilled.csv)")
    ap.add_argument("--exchange", default=EXCHANGE_ID, help=f"CCXT exchange id (default: {EXCHANGE_ID})")
    ap.add_argument("--lookback-15m", type=int, default=LOOKBACK_15M)
    ap.add_argument("--lookback-1h",  type=int, default=LOOKBACK_1H)
    ap.add_argument("--lookback-6h",  type=int, default=LOOKBACK_6H)
    ap.add_argument("--lookback-1d",  type=int, default=LOOKBACK_1D)
    ap.add_argument("--all", action="store_true", help="Process all rows (ignore ONLY_BACKFILL_MISSING)")
    args = ap.parse_args()

    csv_path = args.input or CSV_PATH
    out_path = args.output or OUTPUT_CSV
    looks = {900: args.lookback_15m, 3600: args.lookback_1h, 21600: args.lookback_6h, 86400: args.lookback_1d}
    backfill(csv_path, out_path, args.exchange, looks, only_missing=(not args.all))

if __name__ == "__main__":
    main()
