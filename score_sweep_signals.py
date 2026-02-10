#!/usr/bin/env python3
"""
Score sweep setups with TP/SL outcomes and MAE/MFE windows.

Requirements:
    pip install pandas ccxt python-dateutil

Usage:
    python score_sweep_signals.py \
        --input /path/to/parsed_sweep_signals.csv \
        --output /path/to/sweep_signals_scored.csv \
        --exchange coinbase \
        --timeframe 1m \
        --max-lookahead 24h \
        --priority SL \
        --horizons 15m 1h 6h 24h

Notes:
- By default this fetches 1m OHLCV from the chosen exchange using CCXT.
- You can optionally provide local OHLCV CSVs via --local-candles DIR.
  Expect filename pattern: {SYMBOL_REPLACED_SLASH}_1m.csv (e.g., BTCUSD_1m.csv)
  with columns: timestamp, open, high, low, close, volume (timestamp ms or ISO).
- Tie-break within a candle is controlled by --priority (SL or TP).
  With candle data we can't know intra-candle order, so choose the convention you prefer.
"""
from pathlib import Path

import argparse
import math
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Tuple, Optional

import pandas as pd

# Lazy import ccxt only when needed (allows offline/local-csv mode without ccxt installed)
def _lazy_ccxt():
    try:
        import ccxt  # type: ignore
        return ccxt
    except Exception as e:
        raise RuntimeError("ccxt is required when not using --local-candles. Install via: pip install ccxt") from e


def timeframe_to_ms(tf: str) -> int:
    tf = tf.lower()
    if tf.endswith("ms"):
        return int(tf[:-2])
    if tf.endswith("s"):
        return int(tf[:-1]) * 1000
    if tf.endswith("m"):
        return int(tf[:-1]) * 60_000
    if tf.endswith("h"):
        return int(tf[:-1]) * 60 * 60_000
    if tf.endswith("d"):
        return int(tf[:-1]) * 24 * 60 * 60_000
    raise ValueError(f"Unsupported timeframe: {tf}")


def parse_period_to_minutes(s: str) -> int:
    s = s.lower().strip()
    if s.endswith("m"):
        return int(s[:-1])
    if s.endswith("h"):
        return int(s[:-1]) * 60
    if s.endswith("d"):
        return int(s[:-1]) * 60 * 24
    raise ValueError(f"Unsupported period: {s}. Use forms like 15m, 2h, 1d.")


def ensure_dt_utc(x) -> datetime:
    if isinstance(x, datetime):
        if x.tzinfo is None:
            return x.replace(tzinfo=timezone.utc)
        return x.astimezone(timezone.utc)
    # Try parse string
    try:
        dt = pd.to_datetime(x, utc=True)
        if isinstance(dt, pd.Timestamp):
            return dt.to_pydatetime()
        return dt
    except Exception:
        raise


def read_local_ohlcv_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Flexible timestamp parsing
    if "timestamp" in df.columns:
        ts = df["timestamp"]
        if pd.api.types.is_numeric_dtype(ts):
            dt = pd.to_datetime(ts, unit="ms", utc=True)
        else:
            dt = pd.to_datetime(ts, utc=True)
    elif "time" in df.columns:
        dt = pd.to_datetime(df["time"], utc=True)
    else:
        raise ValueError(f"Cannot find timestamp column in {path}; expected 'timestamp' or 'time'.")

    # Normalize columns
    cols = {c.lower(): c for c in df.columns}
    for needed in ["open", "high", "low", "close"]:
        if needed not in cols:
            raise ValueError(f"Missing column '{needed}' in {path}. Found: {list(df.columns)}")

    out = pd.DataFrame({
        "datetime": dt,
        "open": df[cols["open"]].astype(float),
        "high": df[cols["high"]].astype(float),
        "low": df[cols["low"]].astype(float),
        "close": df[cols["close"]].astype(float),
    })
    out = out.sort_values("datetime").reset_index(drop=True)
    return out


def fetch_ohlcv_range_ccxt(exchange_id: str, symbol: str, timeframe: str,
                           since_ms: int, until_ms: int) -> pd.DataFrame:
    """
    Fetch OHLCV between since_ms (inclusive) and until_ms (exclusive).
    """
    ccxt = _lazy_ccxt()
    tf_ms = timeframe_to_ms(timeframe)
    limit = 1000  # per request

    ex = getattr(ccxt, exchange_id)({
        "enableRateLimit": True,
        # "options": {"adjustForTimeDifference": True},  # optional
    })
    ex.load_markets()
    if symbol not in ex.symbols:
        # Try a couple of simple fallbacks for common quote variations
        alts = [symbol.replace("/USDT", "/USD"), symbol.replace("/USD", "/USDT")]
        found = None
        for alt in alts:
            if alt in ex.symbols:
                found = alt
                break
        if not found:
            raise ValueError(f"Symbol {symbol} not found on {exchange_id}. Available sample: {list(ex.symbols)[:10]}")
        symbol = found

    all_rows = []
    fetch_since = since_ms
    # Safety: don't loop forever
    hard_cap = 50_000
    while fetch_since < until_ms and hard_cap > 0:
        ohlcv = ex.fetch_ohlcv(symbol, timeframe=timeframe, since=fetch_since, limit=limit)
        if not ohlcv:
            break
        all_rows.extend(ohlcv)
        last_ts = ohlcv[-1][0]
        # Advance to next candle after the last returned
        fetch_since = last_ts + tf_ms
        hard_cap -= 1
        # If the exchange doesn't support since for the given market/timeframe well,
        # we may need to break if last_ts isn't advancing.
        if len(ohlcv) < limit or last_ts >= until_ms:
            break

    if not all_rows:
        return pd.DataFrame(columns=["datetime", "open", "high", "low", "close"])

    df = pd.DataFrame(all_rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.loc[(df["datetime"].astype("int64") // 1_000_000) >= since_ms]  # ensure lower bound
    df = df.loc[(df["datetime"].astype("int64") // 1_000_000) < until_ms]   # ensure upper bound
    df = df[["datetime", "open", "high", "low", "close"]].sort_values("datetime").reset_index(drop=True)
    return df


def subset_until(df: pd.DataFrame, start_dt: datetime, end_dt: datetime) -> pd.DataFrame:
    return df.loc[(df["datetime"] >= start_dt) & (df["datetime"] < end_dt)].copy()


def compute_mfe_mae(side: str, entry: float, highs: pd.Series, lows: pd.Series) -> Tuple[float, float]:
    """
    Returns (mfe_pct, mae_pct), both positive percentages.
    """
    if highs.empty or lows.empty or math.isnan(entry) or entry == 0:
        return (float("nan"), float("nan"))

    if side == "long":
        max_up = (highs.max() - entry) / entry * 100.0
        max_down = (entry - lows.min()) / entry * 100.0
    else:  # short
        max_up = (entry - lows.min()) / entry * 100.0   # favorable is down move
        max_down = (highs.max() - entry) / entry * 100.0

    return (float(max_up), float(max_down))


def detect_outcome(side: str, entry: float, target: float, stop: float,
                   ohlc: pd.DataFrame, priority: str = "SL") -> Tuple[str, Optional[datetime]]:
    """
    Walk forward through candles to see which level is touched first.
    Returns (outcome, outcome_time). outcome in {"TP", "SL", "Timeout"}.
    If no candles provided, returns ("NoData", None). Caller decides timeout policy.
    priority: "SL" or "TP" for resolving same-candle touches.
    """
    if ohlc.empty:
        return ("NoData", None)

    priority = priority.upper()
    if priority not in {"SL", "TP"}:
        raise ValueError("priority must be 'SL' or 'TP'")

    for _, row in ohlc.iterrows():
        high = row["high"]
        low = row["low"]
        t = row["datetime"].to_pydatetime()

        if side == "long":
            hit_tp = high >= target
            hit_sl = low <= stop
        else:  # short
            hit_tp = low <= target
            hit_sl = high >= stop

        if hit_tp and hit_sl:
            # same-candle conflict → use tie-break
            if priority == "SL":
                return ("SL", t)
            else:
                return ("TP", t)
        elif hit_tp:
            return ("TP", t)
        elif hit_sl:
            return ("SL", t)

    return ("Timeout", None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=str, default="/mnt/data/parsed_sweep_signals.csv",
                    help="Path to parsed_sweep_signals.csv")
    ap.add_argument("--output", type=str, default="/mnt/data/sweep_signals_scored.csv",
                    help="Where to write the scored CSV")
    ap.add_argument("--exchange", type=str, default="coinbase",
                    help="CCXT exchange id (e.g., coinbase, binanceus, kraken)")
    ap.add_argument("--timeframe", type=str, default="1m",
                    help="OHLC timeframe to fetch (default: 1m)")
    ap.add_argument("--max-lookahead", type=str, default="24h",
                    help="Max lookahead window for detecting TP/SL (e.g., 6h, 24h, 2d)")
    ap.add_argument("--priority", type=str, default="SL",
                    help="Tie-break if both TP and SL are hit in same candle (SL or TP)")
    ap.add_argument("--horizons", type=str, nargs="+", default=["15m", "1h", "6h", "24h"],
                    help="Windows for MAE/MFE columns")
    ap.add_argument("--local-candles", type=str, default=None,
                    help="Optional directory of local OHLCV CSVs instead of fetching with CCXT")
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    if "signal_time_utc" not in df.columns:
        raise ValueError("Input CSV must include 'signal_time_utc' column.")

    # Normalize datetimes
    df["signal_time_utc"] = pd.to_datetime(df["signal_time_utc"], utc=True)

    # Prepare horizon minutes
    horizon_minutes = {h: parse_period_to_minutes(h) for h in args.horizons}
    max_minutes = parse_period_to_minutes(args.max_lookahead)
    tf_ms = timeframe_to_ms(args.timeframe)

    # Prepare outputs
    outcome_list = []
    outcome_time_list = []
    tto_min_list = []

    # Dynamic MAE/MFE columns
    mae_cols = {h: [] for h in args.horizons}
    mfe_cols = {h: [] for h in args.horizons}

    # Optional local data dir
    local_dir = Path(args.local_candles) if args.local_candles else None

    # For each setup, fetch candles and score
    for idx, row in df.iterrows():
        pair = row.get("pair") or row.get("symbol")  # fallback
        side = str(row["side"]).lower()
        entry = float(row["entry_price"])
        target = float(row["target_price"])
        stop = float(row["stop_price"])
        start_dt = ensure_dt_utc(row["signal_time_utc"])
        end_dt = start_dt + timedelta(minutes=max_minutes)
        start_ms = int(start_dt.timestamp() * 1000)
        until_ms = int(end_dt.timestamp() * 1000)

        # Fetch OHLC
        ohlc: pd.DataFrame
        try:
            if local_dir:
                # filename BTC/USD -> BTCUSD
                fname = f"{pair.replace('/', '')}_1m.csv"
                path = local_dir / fname
                ohlc_full = read_local_ohlcv_csv(path)
                ohlc = subset_until(ohlc_full, start_dt, end_dt)
            else:
                # CCXT fetch
                ohlc = fetch_ohlcv_range_ccxt(args.exchange, pair, args.timeframe,
                                              max(0, start_ms - tf_ms), until_ms)
                # Subset to [start_dt, end_dt)
                ohlc = subset_until(ohlc, start_dt, end_dt)
        except Exception as e:
            # On any error, record NoData and NaNs
            outcome_list.append("NoData")
            outcome_time_list.append(pd.NaT)
            tto_min_list.append(float("nan"))
            for h in args.horizons:
                mae_cols[h].append(float("nan"))
                mfe_cols[h].append(float("nan"))
            print(f"[{idx}] {pair} ERROR: {e}")
            continue

        # Outcome detection
        outcome, when = detect_outcome(side, entry, target, stop, ohlc, priority=args.priority)
        outcome_list.append(outcome)

        if when is not None:
            outcome_time_list.append(when)
            tto = (when.replace(tzinfo=timezone.utc) - start_dt).total_seconds() / 60.0
            tto_min_list.append(tto)
        else:
            outcome_time_list.append(pd.NaT)
            tto_min_list.append(float("nan"))

        # MAE/MFE windows
        for h, mins in horizon_minutes.items():
            wnd_end = start_dt + timedelta(minutes=mins)
            sub = subset_until(ohlc, start_dt, wnd_end)
            mfe, mae = compute_mfe_mae(side, entry, sub["high"], sub["low"])
            mfe_cols[h].append(mfe)
            mae_cols[h].append(mae)

        print(f"[{idx}] {pair} {side} → outcome={outcome} tto_min={tto_min_list[-1]}")

    # Attach columns
    df["outcome"] = outcome_list
    df["outcome_time_utc"] = pd.to_datetime(outcome_time_list, utc=True)
    df["time_to_outcome_min"] = tto_min_list

    # Add MAE/MFE columns
    for h in args.horizons:
        df[f"mfe_{h}"] = mfe_cols[h]
        df[f"mae_{h}"] = mae_cols[h]

    # Save
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Saved scored CSV → {out_path}")


if __name__ == "__main__":
    main()
