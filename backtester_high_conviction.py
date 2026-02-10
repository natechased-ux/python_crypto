#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
High-Conviction Backtester (auto-defaults) — with MFE/MAE logging
- Run with no args to fetch last 3 months from Coinbase for BTC/ETH/XRP/SOL/DOGE.
- Keeps ATR-based TP/SL simulation AND logs MFE/MAE (max favorable/adverse excursion)
  up to the simulated trade exit (TP/SL/Timeout).
- Portable: use --out_dir to choose where CSVs are written (defaults to current folder).

MFE/MAE fields added per trade:
- max_high_after_entry / min_low_after_entry (raw price extremes during trade)
- mfe_pct: (favorable move) percent from entry to best price in trade direction
- mae_pct: (adverse move) percent from entry to worst price against direction

Examples:
  py backtester_high_conviction.py
  py backtester_high_conviction.py --out_dir .\output
"""
import argparse, json, os, sys, time
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple
import pandas as pd, numpy as np

# HTTP (for Coinbase loader)
try:
    import requests
except Exception:
    requests = None

CB_BASE = "https://api.exchange.coinbase.com"

# ---------- Utilities ----------
def ensure_datetime(df: pd.DataFrame, col: str = "time") -> pd.DataFrame:
    df = df.copy()
    if np.issubdtype(df[col].dtype, np.number):
        if df[col].max() > 10**12:
            df[col] = pd.to_datetime(df[col], unit="ms", utc=True)
        else:
            df[col] = pd.to_datetime(df[col], unit="s", utc=True)
    else:
        df[col] = pd.to_datetime(df[col], utc=True)
    return df

def resample_ohlcv(df: pd.DataFrame, tf: str) -> pd.DataFrame:
    o = df["open"].resample(tf).first()
    h = df["high"].resample(tf).max()
    l = df["low"].resample(tf).min()
    c = df["close"].resample(tf).last()
    v = df["volume"].resample(tf).sum()
    out = pd.DataFrame({"open": o, "high": h, "low": l, "close": c, "volume": v})
    return out.dropna(how="any")

def ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False).mean()

def rsi(series: pd.Series, length: int = 14) -> pd.Series:
    delta = series.diff()
    up = np.where(delta > 0, delta, 0.0)
    down = np.where(delta < 0, -delta, 0.0)
    roll_up = pd.Series(up, index=series.index).ewm(alpha=1/length, adjust=False).mean()
    roll_down = pd.Series(down, index=series.index).ewm(alpha=1/length, adjust=False).mean()
    rs = roll_up / (roll_down + 1e-12)
    return 100 - (100 / (1 + rs))

def true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr

def atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    tr = true_range(df)
    return tr.ewm(alpha=1/length, adjust=False).mean()

def macd(series: pd.Series, fast=12, slow=26, signal=9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    fast_ema = ema(series, fast)
    slow_ema = ema(series, slow)
    macd_line = fast_ema - slow_ema
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def adx(df: pd.DataFrame, length: int = 14) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    tr = true_range(df)
    atr_w = tr.ewm(alpha=1/length, adjust=False).mean()
    plus_di = 100 * pd.Series(plus_dm, index=df.index).ewm(alpha=1/length, adjust=False).mean() / (atr_w + 1e-12)
    minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(alpha=1/length, adjust=False).mean() / (atr_w + 1e-12)
    dx = (100 * (plus_di - minus_di).abs() / ((plus_di + minus_di) + 1e-12))
    return dx.ewm(alpha=1/length, adjust=False).mean()

def stoch_rsi(series: pd.Series, rsi_len=14, stoch_len=14, k=3, d=3) -> Tuple[pd.Series, pd.Series]:
    r = rsi(series, rsi_len)
    min_r = r.rolling(stoch_len).min()
    max_r = r.rolling(stoch_len).max()
    stoch = (r - min_r) / ((max_r - min_r) + 1e-12)
    k_line = stoch.rolling(k).mean() * 100.0
    d_line = k_line.rolling(d).mean()
    return k_line, d_line

def detect_fvg(df: pd.DataFrame) -> pd.DataFrame:
    fvg_up = df["low"] > df["high"].shift(2)
    fvg_dn = df["high"] < df["low"].shift(2)
    return pd.DataFrame({"fvg_up": fvg_up.fillna(False), "fvg_dn": fvg_dn.fillna(False)}, index=df.index)

# ---------- Coinbase loader ----------
def fetch_coinbase_candles(product_id: str, start: str, end: str, granularity: int = 3600, pause: float = 0.25) -> pd.DataFrame:
    if requests is None:
        raise RuntimeError("requests not available; run locally with 'pip install requests'.")
    start_ts = pd.to_datetime(start, utc=True)
    end_ts = pd.to_datetime(end, utc=True)
    step = pd.Timedelta(seconds=granularity * 300)  # 300 candles per call
    frames = []
    headers = {"User-Agent": "hc-backtester/1.0"}
    t0 = start_ts
    while t0 < end_ts:
        t1 = min(t0 + step, end_ts)
        url = f"{CB_BASE}/products/{product_id}/candles"
        params = {
            "granularity": granularity,
            "start": t0.isoformat().replace("+00:00", "Z"),
            "end": t1.isoformat().replace("+00:00", "Z"),
        }
        resp = requests.get(url, params=params, headers=headers, timeout=30)
        if resp.status_code == 429:
            time.sleep(1.0); resp = requests.get(url, params=params, headers=headers, timeout=30)
        resp.raise_for_status()
        data = resp.json() or []
        if data:
            arr = np.array(data, dtype=float)
            df = pd.DataFrame(arr, columns=["time","low","high","open","close","volume"])
            df["time"] = pd.to_datetime(df["time"].astype(int), unit="s", utc=True)
            df = df.sort_values("time").set_index("time")[["open","high","low","close","volume"]]
            frames.append(df)
        time.sleep(pause)
        t0 = t1
    if not frames:
        raise ValueError(f"No data from Coinbase for {product_id} in {start}..{end}")
    return pd.concat(frames).groupby(level=0).last().sort_index()

# ---------- Dataclasses ----------
@dataclass
class Params:
    adx_min: float = 25.0
    rsi_long_min: float = 55.0
    rsi_short_max: float = 45.0
    ema_len: int = 200
    atr_len: int = 14
    tp_mult: float = 2.0
    sl_mult: float = 1.5
    confirm_hours: int = 6
    one_trade_per_day: bool = True
    use_fvg: bool = True
    use_whale: bool = False
    whale_max_distance_pct: float = 0.03

@dataclass
class Trade:
    run_id: str
    symbol: str
    side: str
    entry_time: pd.Timestamp
    entry: float
    sl: float
    tp: float
    exit_time: pd.Timestamp
    exit_price: float
    outcome: str
    rr: float
    duration_hours: float
    adx: float
    rsi: float
    macd_hist: float
    ema200_htf: float
    htf_close: float
    max_high_after_entry: float
    min_low_after_entry: float
    mfe_pct: float
    mae_pct: float

# ---------- Core calcs ----------
def compute_bias_and_momentum(htf: pd.DataFrame, p: Params) -> pd.DataFrame:
    df = htf.copy()
    df["ema200"] = ema(df["close"], p.ema_len)
    _, _, hist = macd(df["close"])
    df["macd_hist"] = hist
    df["adx"] = adx(df)
    df["rsi"] = rsi(df["close"])
    df["uptrend"] = (df["close"] > df["ema200"]) & (df["macd_hist"] > 0)
    df["downtrend"] = (df["close"] < df["ema200"]) & (df["macd_hist"] < 0)
    df["long_ok"] = df["uptrend"] & (df["adx"] >= p.adx_min) & (df["rsi"] >= p.rsi_long_min)
    df["short_ok"] = df["downtrend"] & (df["adx"] >= p.adx_min) & (df["rsi"] <= p.rsi_short_max)
    return df

def stoch_rsi_series(ltf: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    k, d = stoch_rsi(ltf["close"]); return k, d

def simulate_trade_with_excursions(ltf: pd.DataFrame, entry_idx: int, side: str, entry_price: float,
                                   atr_val: float, swing_buffer: float, p: Params) -> Tuple[pd.Timestamp, float, str, float, float, float, float, float]:
    # Build TP/SL like before (for comparison)
    if side == "long":
        sl = entry_price - p.sl_mult * atr_val - swing_buffer
        tp = entry_price + p.tp_mult * atr_val
    else:
        sl = entry_price + p.sl_mult * atr_val + swing_buffer
        tp = entry_price - p.tp_mult * atr_val
    # excursions from next bar
    max_high, min_low = entry_price, entry_price
    for i in range(entry_idx + 1, len(ltf)):
        high = ltf["high"].iloc[i]; low = ltf["low"].iloc[i]; ts = ltf.index[i]
        if high > max_high: max_high = high
        if low < min_low: min_low = low
        if side == "long":
            if low <= sl:
                mfe_pct = (max_high - entry_price) / entry_price * 100.0
                mae_pct = (entry_price - min_low) / entry_price * 100.0
                return ts, sl, "SL", (tp - entry_price) / (entry_price - sl), max_high, min_low, mfe_pct, mae_pct
            if high >= tp:
                mfe_pct = (max_high - entry_price) / entry_price * 100.0
                mae_pct = (entry_price - min_low) / entry_price * 100.0
                return ts, tp, "TP", (tp - entry_price) / (entry_price - sl), max_high, min_low, mfe_pct, mae_pct
        else:
            if high >= sl:
                mfe_pct = (entry_price - min_low) / entry_price * 100.0
                mae_pct = (max_high - entry_price) / entry_price * 100.0
                return ts, sl, "SL", (entry_price - tp) / (sl - entry_price), max_high, min_low, mfe_pct, mae_pct
            if low <= tp:
                mfe_pct = (entry_price - min_low) / entry_price * 100.0
                mae_pct = (max_high - entry_price) / entry_price * 100.0
                return ts, tp, "TP", (entry_price - tp) / (sl - entry_price), max_high, min_low, mfe_pct, mae_pct
    # Timeout at final bar
    ts = ltf.index[-1]; price = ltf["close"].iloc[-1]
    if side == "long":
        rr = (price - entry_price) / (entry_price - sl)
        mfe_pct = (max_high - entry_price) / entry_price * 100.0
        mae_pct = (entry_price - min_low) / entry_price * 100.0
    else:
        rr = (entry_price - price) / (sl - entry_price)
        mfe_pct = (entry_price - min_low) / entry_price * 100.0
        mae_pct = (max_high - entry_price) / entry_price * 100.0
    return ts, price, "Timeout", rr, max_high, min_low, mfe_pct, mae_pct

# ---------- Data Loading (CSV or Coinbase) ----------
def load_csv_symbol(data_dir: str, symbol: str) -> Optional[pd.DataFrame]:
    candidates = [
        os.path.join(data_dir, f"{symbol}.csv"),
        os.path.join(data_dir, f"{symbol.replace('-', '_')}.csv"),
        os.path.join(data_dir, f"{symbol.replace('-', '').lower()}.csv"),
        os.path.join(data_dir, f"{symbol.lower()}.csv"),
    ]
    for path in candidates:
        if os.path.exists(path):
            df = pd.read_csv(path); df = ensure_datetime(df, "time")
            df = df.sort_values("time").set_index("time")
            needed = {"open", "high", "low", "close", "volume"}
            if not needed.issubset(df.columns):
                raise ValueError(f"Missing columns in {path}. Need: {needed}")
            return df
    return None

def fetch_symbol_candles(data_dir: Optional[str], data_source: str, symbol: str, start: Optional[str], end: Optional[str], granularity: int = 3600) -> pd.DataFrame:
    if data_source == "csv":
        if data_dir is None: raise ValueError("--data_dir required for csv data_source")
        df = load_csv_symbol(data_dir, symbol)
        if df is None: raise FileNotFoundError(f"No CSV found for {symbol} in {data_dir}.")
        return df
    elif data_source == "coinbase":
        if start is None or end is None: raise ValueError("--start and --end are required for coinbase data_source")
        return fetch_coinbase_candles(symbol, start, end, granularity=granularity)
    else:
        raise ValueError("Unsupported --data_source (use 'csv' or 'coinbase')")

# ---------- Backtest Runner ----------
def run_backtest(run_id: str,
                 data_dir: Optional[str],
                 data_source: str,
                 symbols: List[str],
                 start: Optional[str],
                 end: Optional[str],
                 grid: Dict,
                 out_dir: str) -> pd.DataFrame:
    os.makedirs(out_dir, exist_ok=True)
    results = []
    grid_items = dict(grid)
    grid_items.setdefault("adx", [25.0])
    grid_items.setdefault("tp_mult", [2.0])
    grid_items.setdefault("sl_mult", [1.5])

    run_idx = 0
    for adx_min in grid_items["adx"]:
        for tp_mult in grid_items["tp_mult"]:
            for sl_mult in grid_items["sl_mult"]:
                run_idx += 1
                run_tag = f"adx{adx_min}_tp{tp_mult}_sl{sl_mult}_{run_idx}"
                params = Params(adx_min=adx_min, tp_mult=tp_mult, sl_mult=sl_mult)

                all_trades: List[Trade] = []
                for symbol in symbols:
                    raw = fetch_symbol_candles(data_dir, data_source, symbol, start, end, granularity=3600)
                    if start: raw = raw[raw.index >= pd.to_datetime(start, utc=True)]
                    if end:   raw = raw[raw.index <= pd.to_datetime(end, utc=True)]
                    if len(raw) < 300:
                        print(f"[WARN] {symbol}: too few candles ({len(raw)}). Skipping."); continue

                    raw_1h = resample_ohlcv(raw, "1H"); raw_6h = resample_ohlcv(raw, "6H")
                    htf = compute_bias_and_momentum(raw_6h, params)
                    htf_fvg = detect_fvg(raw_6h) if params.use_fvg else pd.DataFrame(index=raw_6h.index)
                    ltf = raw_1h.copy(); ltf["atr"] = atr(ltf, params.atr_len)
                    k, d = stoch_rsi(ltf["close"]); ltf["stoch_k"], ltf["stoch_d"] = k, d

                    for idx in range(2, len(htf)-1):
                        ts = htf.index[idx]; row = htf.iloc[idx]
                        side = "long" if row["long_ok"] else ("short" if row["short_ok"] else None)
                        if side is None: continue
                        if params.use_fvg:
                            fvgr = htf_fvg.loc[ts]
                            if side == "long" and not fvgr.get("fvg_up", False): continue
                            if side == "short" and not fvgr.get("fvg_dn", False): continue

                        start_confirm = ts; end_confirm = ts + pd.Timedelta(hours=params.confirm_hours)
                        window = ltf[(ltf.index > start_confirm) & (ltf.index <= end_confirm)].copy()
                        if window.empty: continue
                        if side == "long":
                            trig = (window["stoch_k"] > window["stoch_d"]) & (window["stoch_k"] < 40)
                        else:
                            trig = (window["stoch_k"] < window["stoch_d"]) & (window["stoch_k"] > 60)
                        if not trig.any(): continue

                        trig_idx = trig.idxmax()
                        if params.one_trade_per_day:
                            same_day = [t for t in all_trades if (t.symbol == symbol) and (t.entry_time.date() == trig_idx.date())]
                            if same_day: continue

                        entry_pos = ltf.index.get_loc(trig_idx)
                        if entry_pos + 1 >= len(ltf): continue
                        entry_time = ltf.index[entry_pos + 1]
                        entry_price = ltf["open"].iloc[entry_pos + 1]
                        atr_val = ltf["atr"].iloc[entry_pos]
                        swing_buffer = 0.0

                        exit_time, exit_price, outcome, rr, max_high, min_low, mfe_pct, mae_pct = simulate_trade_with_excursions(
                            ltf, entry_pos, side, entry_price, atr_val, swing_buffer, params
                        )

                        all_trades.append(Trade(
                            run_id=f"{run_id}_{run_tag}", symbol=symbol, side=side,
                            entry_time=entry_time, entry=entry_price,
                            sl=(entry_price - params.sl_mult * atr_val) if side == "long" else (entry_price + params.sl_mult * atr_val),
                            tp=(entry_price + params.tp_mult * atr_val) if side == "long" else (entry_price - params.tp_mult * atr_val),
                            exit_time=exit_time, exit_price=exit_price, outcome=outcome, rr=rr,
                            duration_hours=float((exit_time - entry_time).total_seconds() / 3600.0),
                            adx=float(row["adx"]), rsi=float(row["rsi"]), macd_hist=float(row["macd_hist"]),
                            ema200_htf=float(row["ema200"]), htf_close=float(row["close"]),
                            max_high_after_entry=float(max_high), min_low_after_entry=float(min_low),
                            mfe_pct=float(mfe_pct), mae_pct=float(mae_pct),
                        ))

                trades_df = pd.DataFrame([asdict(t) for t in all_trades])
                trades_path = os.path.join(out_dir, f"trades_{run_id}_{run_tag}.csv")
                trades_df.to_csv(trades_path, index=False)

                if trades_df.empty:
                    results.append({
                        "run_id": f"{run_id}_{run_tag}", "adx_min": adx_min, "tp_mult": tp_mult, "sl_mult": sl_mult,
                        "trades": 0, "win_rate": np.nan, "avg_rr": np.nan,
                        "tp_rate": np.nan, "sl_rate": np.nan, "timeout_rate": np.nan, "avg_dur_hours": np.nan,
                    })
                else:
                    trades_df["win"] = (trades_df["outcome"] == "TP").astype(int)
                    trades_df["loss"] = (trades_df["outcome"] == "SL").astype(int)
                    trades_df["timeout"] = (trades_df["outcome"] == "Timeout").astype(int)
                    results.append({
                        "run_id": f"{run_id}_{run_tag}", "adx_min": adx_min, "tp_mult": tp_mult, "sl_mult": sl_mult,
                        "trades": int(len(trades_df)),
                        "win_rate": float(trades_df["win"].mean()), "avg_rr": float(trades_df["rr"].mean()),
                        "tp_rate": float(trades_df["win"].mean()), "sl_rate": float(trades_df["loss"].mean()),
                        "timeout_rate": float(trades_df["timeout"].mean()), "avg_dur_hours": float(trades_df["duration_hours"].mean()),
                    })

    leaderboard = pd.DataFrame(results).sort_values(["win_rate","avg_rr","trades"], ascending=[False, False, False])
    lb_path = os.path.join(out_dir, f"summary_{run_id}.csv")
    leaderboard.to_csv(lb_path, index=False)
    return leaderboard

# ---------- Main ----------
def main():
    DEFAULT_SYMBOLS = [   "BTC-USD", "ETH-USD", "XRP-USD", "SOL-USD", "DOGE-USD", "ADA-USD", "XLM-USD", "SUI-USD",
    "BCH-USD", "LINK-USD", "HBAR-USD", "LTC-USD", "AVAX-USD", "SHIB-USD", "UNI-USD", "DOT-USD",
    "CRO-USD",  "AAVE-USD", "NEAR-USD", "ETC-USD", "ONDO-USD",
    "APT-USD", "ICP-USD", "ALGO-USD", "ARB-USD", "VET-USD",
    "ATOM-USD", "POL-USD", "BONK-USD", "RENDER-USD",
    "FET-USD", "SEI-USD", "FLR-USD", "FIL-USD", "QNT-USD", "LSETH-USD", "INJ-USD",
    "CRV-USD", "STX-USD", "TIA-USD"]
    now_utc = pd.Timestamp.utcnow().normalize()
    start_def = (now_utc - pd.Timedelta(days=92)).strftime("%Y-%m-%d")
    end_def = now_utc.strftime("%Y-%m-%d")
    DEFAULT_GRID = {"adx":[22,25,28], "tp_mult":[2.0,2.5], "sl_mult":[1.25,1.5]}
    DEFAULTS = {
        "data_source": "coinbase",
        "symbols": DEFAULT_SYMBOLS,
        "start": start_def, "end": end_def,
        "grid": DEFAULT_GRID,
        "run_id": "hc_cb_autorun",
        "out_dir": os.getcwd(),
    }

    ap = argparse.ArgumentParser()
    ap.add_argument("--data_source", type=str, choices=["csv","coinbase"], help="csv or coinbase")
    ap.add_argument("--data_dir", type=str, help="Folder with CSVs per symbol (csv mode)")
    ap.add_argument("--symbols", nargs="+", help="Symbols, e.g. BTC-USD ETH-USD")
    ap.add_argument("--start", type=str, help="YYYY-MM-DD")
    ap.add_argument("--end", type=str, help="YYYY-MM-DD")
    ap.add_argument("--grid", type=str, help='JSON like {"adx":[25],"tp_mult":[2.0],"sl_mult":[1.5]}')
    ap.add_argument("--run_id", type=str, help="Run identifier")
    ap.add_argument("--out_dir", type=str, help="Where to write CSVs (default: current folder)")
    args = ap.parse_args()

    data_source = args.data_source or DEFAULTS["data_source"]
    symbols = args.symbols or DEFAULTS["symbols"]
    start = args.start or DEFAULTS["start"]
    end = args.end or DEFAULTS["end"]
    grid = json.loads(args.grid) if args.grid else DEFAULTS["grid"]
    run_id = args.run_id or DEFAULTS["run_id"]
    out_dir = args.out_dir or DEFAULTS["out_dir"]
    os.makedirs(out_dir, exist_ok=True)

    print("=== High-Conviction Backtester (MFE/MAE) ===")
    print(f"Data source : {data_source}")
    print(f"Symbols     : {', '.join(symbols)}")
    print(f"Date range  : {start} -> {end}")
    print(f"Grid        : {grid}")
    print(f"Run ID      : {run_id}")
    print(f"Output dir  : {out_dir}")

    leaderboard = run_backtest(run_id, args.data_dir, data_source, symbols, start, end, grid, out_dir)

    print("\n=== Leaderboard (best on top) ===")
    print(leaderboard.head(20).to_string(index=False))
    print(f"\nSaved: {os.path.join(out_dir, f'summary_{run_id}.csv')} and per-run trades CSVs.")

if __name__ == "__main__":
    main()
