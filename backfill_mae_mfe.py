#!/usr/bin/env python3
# backfill_mae_mfe.py
# Adds time-sliced MAE/MFE (both censored and uncensored) into a trade log CSV.
# Reuses CCXT window-fetching patterns from the Stoch RSI backfill script.
# Requires: pip install ccxt pandas numpy

import os, sys, time, math, argparse, random
from typing import Optional, Tuple, Dict, List
import numpy as np
import pandas as pd
from pandas.api.types import is_integer_dtype, is_float_dtype

# ===========
# EDIT HERE 👇
# ===========
CSV_PATH               = "live_trade_log_fib6.csv"  # your trade log CSV
OUTPUT_CSV             = None   # None -> writes *_maemfe.csv next to CSV_PATH
EXCHANGE_ID            = "kraken"
ONLY_BACKFILL_MISSING  = True   # True: only fill rows missing the mae/mfe columns
HORIZONS               = "5m,15m,30m,1h,2h,4h,6h,12h,24h"  # comma-separated list
RATE_LIMIT_SEC         = 0.4
MAX_RETRIES            = 7
BACKOFF_BASE_SEC       = 0.8
JITTER_SEC             = 0.25

# Optional: compute $-denominated excursions for a given notional size
NOTIONAL_DOLLARS       = None   # e.g., 1000 or 5000; None = skip $ columns

# ------------------
# Utilities
# ------------------
def _to_utc(series: pd.Series) -> pd.Series:
    if is_integer_dtype(series) or is_float_dtype(series):
        return pd.to_datetime(series, unit="ms", utc=True, errors="coerce")
    return pd.to_datetime(series, utc=True, errors="coerce")

def _parse_horizons(hstr: str) -> List[pd.Timedelta]:
    out = []
    for tok in str(hstr).split(","):
        t = tok.strip().lower()
        if not t: 
            continue
        if t.endswith("m"):
            out.append(pd.Timedelta(minutes=int(t[:-1])))
        elif t.endswith("h"):
            out.append(pd.Timedelta(hours=int(t[:-1])))
        elif t.endswith("d"):
            out.append(pd.Timedelta(days=int(t[:-1])))
        else:
            raise ValueError(f"Unsupported horizon token: {tok}")
    return out

def _hlabel(td: pd.Timedelta) -> str:
    s = int(td.total_seconds())
    if s % 86400 == 0: return f"{s//86400}d"
    if s % 3600 == 0:  return f"{s//3600}h"
    if s % 60 == 0:    return f"{s//60}m"
    return f"{s}s"

# -----------
# CCXT fetcher (subset, adapted)
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
        s = str(s).replace("-", "/").upper()
        parts = s.split("/")
        base, quote = (parts + ["USD"])[:2]
        alias = {"XBT": "BTC", "XDG": "DOGE"}
        base = alias.get(base, base)
        quote = alias.get(quote, quote)
        norm = f"{base}/{quote}"
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
                print(f"[WARN] fetch_ohlcv error {e}; breaking")
                ohlcv = []
                break
        else:
            ohlcv = []
        if not ohlcv:
            return pd.DataFrame(columns=["time","open","high","low","close","volume"])
        return pd.DataFrame(ohlcv, columns=["time","open","high","low","close","volume"])

    def fetch_5m_window(self, symbol: str, start_ts: pd.Timestamp, end_ts: pd.Timestamp, safety_bars: int = 20) -> pd.DataFrame:
        # Ensure we get enough 5m data even if not natively supported (resample from 1m if available).
        mkt = self._norm_symbol(symbol)
        if self._supports("5m"):
            base_tf = "5m"; base_ms = 300_000; limit = 1500
        else:
            # fallback to 1m if possible; otherwise 15m
            base_tf = "1m" if self._supports("1m") else "15m"
            base_ms = 60_000 if base_tf == "1m" else 900_000
            limit = 1500
        since_ms = int((pd.to_datetime(start_ts, utc=True) - pd.Timedelta(milliseconds=base_ms * (safety_bars + 10))).timestamp()*1000)
        until_ms = int(pd.to_datetime(end_ts, utc=True).timestamp()*1000)

        frames = []
        cursor = since_ms
        calls = 0
        while cursor <= until_ms and calls < 1000:
            df = self._fetch_native_once(mkt, base_tf, cursor, limit)
            calls += 1
            if df.empty:
                cursor += base_ms * limit
            else:
                last_ms = int(pd.to_datetime(df["time"], unit="ms", utc=True).astype("int64").iloc[-1] // 10**6)
                frames.append(df)
                cursor = last_ms + base_ms
        if not frames:
            return pd.DataFrame(columns=["time","open","high","low","close","volume"])
        out = pd.concat(frames, ignore_index=True)
        out["time"] = _to_utc(out["time"])
        out = out.dropna(subset=["time","open","high","low","close","volume"]).set_index("time")

        # if base is 1m or 15m, resample to 5m, right-closed
        if base_tf != "5m":
            o = out["open"].resample("5min", label="right", closed="right").first()
            h = out["high"].resample("5min", label="right", closed="right").max()
            l = out["low"].resample("5min", label="right", closed="right").min()
            c = out["close"].resample("5min", label="right", closed="right").last()
            v = out["volume"].resample("5min", label="right", closed="right").sum()
            out = pd.DataFrame({"time": o.index, "open": o.values, "high": h.values, "low": l.values, "close": c.values, "volume": v.values}).dropna()
            out = out.reset_index(drop=True)
        else:
            out = out.reset_index()
        return out

# ------------------
# MAE/MFE computation
# ------------------
def _first_exit_bar(df5: pd.DataFrame, start_idx: int, side: str, tp: Optional[float], sl: Optional[float]) -> Optional[int]:
    if start_idx >= len(df5): 
        return None
    highs = df5["high"].values
    lows  = df5["low"].values
    sideU = str(side).upper()
    for i in range(start_idx, len(df5)):
        hi = highs[i]; lo = lows[i]
        if sideU == "LONG":
            hit_tp = (tp is not None) and (hi >= tp)
            hit_sl = (sl is not None) and (lo <= sl)
        else:
            hit_tp = (tp is not None) and (lo <= tp)
            hit_sl = (sl is not None) and (hi >= sl)
        if hit_tp or hit_sl:
            return i
    return None

def _excursions_for_window(df5: pd.DataFrame, i0: int, i1: int, side: str, entry: float, sl: float) -> Tuple[float, float, float, float]:
    # returns (mfe_r, mae_r, mfe_pct, mae_pct) over bars [i0, i1] inclusive
    sub = df5.iloc[i0:i1+1]
    sideU = str(side).upper()
    risk = abs(entry - sl)
    if risk == 0 or np.isnan(risk) or sub.empty:
        return (np.nan, np.nan, np.nan, np.nan)
    mx = sub["high"].max()
    mn = sub["low"].min()
    if sideU == "LONG":
        mfe = (mx - entry); mae = (entry - mn)
    else:
        mfe = (entry - mn); mae = (mx - entry)
    mfe_r = mfe / risk
    mae_r = mae / risk
    mfe_pct = mfe / entry
    mae_pct = mae / entry
    return (float(mfe_r), float(mae_r), float(mfe_pct*100.0), float(mae_pct*100.0))

def backfill_mae_mfe(csv_path: str,
                     output_csv: Optional[str] = None,
                     exchange_id: str = EXCHANGE_ID,
                     horizons: str = HORIZONS,
                     notional_dollars: Optional[float] = NOTIONAL_DOLLARS,
                     only_missing: bool = ONLY_BACKFILL_MISSING):
    df = pd.read_csv(csv_path)
    if "timestamp_utc" not in df.columns:
        raise ValueError("CSV missing 'timestamp_utc' column.")
    req_cols = ["symbol","side","entry","sl","tp1"]
    for c in req_cols:
        if c not in df.columns:
            raise ValueError(f"CSV missing '{c}' column.")

    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True, errors="coerce")
    df["_sideU"] = df["side"].astype(str).str.upper().str.strip()

    # Parse horizons and ensure deterministic order
    hs = _parse_horizons(horizons)
    hs = sorted(hs, key=lambda t: t.total_seconds())

    # Prepare output columns
    new_cols = []
    for td in hs:
        lab = _hlabel(td)
        for prefix in ["mfe_r","mae_r","mfe_pct","mae_pct","c_mfe_r","c_mae_r","c_mfe_pct","c_mae_pct"]:
            col = f"{prefix}_{lab}"
            if col not in df.columns:
                df[col] = np.nan
            new_cols.append(col)
        if notional_dollars is not None:
            for prefix in ["mfe_$","mae_$","c_mfe_$","c_mae_$"]:
                col = f"{prefix}_{lab}"
                if col not in df.columns:
                    df[col] = np.nan
                new_cols.append(col)

    # Choose rows
    mask = df["timestamp_utc"].notna() & df["entry"].notna() & df["sl"].notna() & df["tp1"].notna()
    if only_missing:
        # only those where some mae/mfe field is missing
        miss_any = None
        for c in new_cols:
            miss_any = df[c].isna() if miss_any is None else (miss_any | df[c].isna())
        need_mask = mask & miss_any if miss_any is not None else mask
    else:
        need_mask = mask
    rows = df[need_mask].copy()

    if rows.empty:
        out = output_csv or (csv_path.replace(".csv", "_maemfe.csv"))
        df.to_csv(out, index=False, encoding="utf-8-sig")
        print(f"[OK] nothing to backfill; wrote {out}")
        return

    # Fetcher
    fetcher = CCXTFetcher(exchange_id=exchange_id, rate_limit_sec=RATE_LIMIT_SEC)

    # Cache 5m OHLC per symbol in a broad window around all its trades
    by_symbol = { sym: g.sort_values("timestamp_utc") for sym, g in rows.groupby("symbol") }
    cache: Dict[str, pd.DataFrame] = {}

    for sym, g in by_symbol.items():
        start_ts = g["timestamp_utc"].min() - pd.Timedelta(hours=48)
        end_ts   = g["timestamp_utc"].max() + pd.Timedelta(hours=48)
        try:
            df5 = fetcher.fetch_5m_window(sym, start_ts, end_ts, safety_bars=40)
        except Exception as e:
            print(f"[WARN] fetch_5m_window failed for {sym}: {e}; skipping symbol")
            df5 = pd.DataFrame(columns=["time","open","high","low","close","volume"])
        cache[sym] = df5

    # Iterate rows and compute excursions
    total = len(rows)
    for i, (idx, r) in enumerate(rows.iterrows(), start=1):
        ts = r["timestamp_utc"]; sym = r["symbol"]; sideU = r["_sideU"]
        entry = float(r["entry"]); sl = float(r["sl"]); tp1 = float(r["tp1"])
        tp2 = float(r["tp2"]) if "tp2" in r and pd.notna(r["tp2"]) else None

        df5 = cache.get(sym, pd.DataFrame())
        if df5.empty or "time" not in df5.columns:
            continue
        tcol = _to_utc(df5["time"])
        base = pd.DataFrame({
            "time": tcol,
            "open": pd.to_numeric(df5["open"], errors="coerce"),
            "high": pd.to_numeric(df5["high"], errors="coerce"),
            "low":  pd.to_numeric(df5["low"],  errors="coerce"),
            "close":pd.to_numeric(df5["close"],errors="coerce")
        }).dropna()

        # bar index strictly after entry (use right-labeled bars)
        after = base[base["time"] > ts.floor("s")]
        if after.empty:
            continue
        i0 = after.index.min()
        i0_pos = list(base.index).index(i0)  # position 0..N-1

        # earliest exit bar using TP1/SL (TP2 optional; include if present)
        tp = tp1
        if tp2 is not None:
            # prefer nearer target in the trade direction for exit detection? keep TP1 as target for consistency
            pass

        i_exit = _first_exit_bar(base, i0_pos, sideU, tp, sl)

        risk = abs(entry - sl)
        risk_pct = risk / entry if entry else np.nan
        for td in hs:
            lab = _hlabel(td)
            # Uncensored window
            end_time = ts + td
            # choose all bars with time <= end_time
            upto = base[base["time"] <= end_time]
            if upto.empty:
                continue
            i1_pos = list(base.index).index(upto.index.max())
            mfe_r, mae_r, mfe_pct, mae_pct = _excursions_for_window(base, i0_pos, i1_pos, sideU, entry, sl)
            df.at[idx, f"mfe_r_{lab}"] = mfe_r
            df.at[idx, f"mae_r_{lab}"] = mae_r
            df.at[idx, f"mfe_pct_{lab}"] = mfe_pct
            df.at[idx, f"mae_pct_{lab}"] = mae_pct
            if notional_dollars is not None and not np.isnan(risk_pct):
                df.at[idx, f"mfe_${lab}"] = mfe_r * risk_pct * notional_dollars
                df.at[idx, f"mae_${lab}"] = mae_r * risk_pct * notional_dollars

            # Censored window (stop at first exit if occurs before end_time)
            if i_exit is not None:
                i1c_pos = min(i1_pos, i_exit)
            else:
                i1c_pos = i1_pos
            mfe_r_c, mae_r_c, mfe_pct_c, mae_pct_c = _excursions_for_window(base, i0_pos, i1c_pos, sideU, entry, sl)
            df.at[idx, f"c_mfe_r_{lab}"] = mfe_r_c
            df.at[idx, f"c_mae_r_{lab}"] = mae_r_c
            df.at[idx, f"c_mfe_pct_{lab}"] = mfe_pct_c
            df.at[idx, f"c_mae_pct_{lab}"] = mae_pct_c
            if notional_dollars is not None and not np.isnan(risk_pct):
                df.at[idx, f"c_mfe_${lab}"] = mfe_r_c * risk_pct * notional_dollars
                df.at[idx, f"c_mae_${lab}"] = mae_r_c * risk_pct * notional_dollars

        if (i % 50 == 0) or (i == total):
            print(f"[{i}/{total}] backfilled MAE/MFE")

    out = output_csv or (csv_path.replace(".csv", "_maemfe.csv"))
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"[OK] wrote {out}")

# ----
# CLI
# ----
def main():
    ap = argparse.ArgumentParser(description="Backfill time-sliced MAE/MFE (censored & uncensored) into a trade log CSV.")
    ap.add_argument("--input", help=f"CSV path (default: {CSV_PATH})")
    ap.add_argument("--output", help="Output CSV path (default: *_maemfe.csv)")
    ap.add_argument("--exchange", default=EXCHANGE_ID, help=f"CCXT exchange id (default: {EXCHANGE_ID})")
    ap.add_argument("--horizons", default=HORIZONS, help="Comma list like '5m,15m,30m,1h,2h,4h,6h,12h,24h'")
    ap.add_argument("--all", action="store_true", help="Process all rows (ignore ONLY_BACKFILL_MISSING)")
    ap.add_argument("--notional", type=float, default=NOTIONAL_DOLLARS, help="If set, also compute $ excursions at this notional")
    args = ap.parse_args()

    csv_path = args.input or CSV_PATH
    out_path = args.output or OUTPUT_CSV
    backfill_mae_mfe(csv_path=csv_path,
                     output_csv=out_path,
                     exchange_id=args.exchange,
                     horizons=args.horizons,
                     notional_dollars=args.notional,
                     only_missing=(not args.all))

if __name__ == "__main__":
    main()
