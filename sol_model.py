"""
SOL 4H + 1m structure-flow classifier (v0.1)

Goal
----
Train a 3-class model (LONG / SHORT / NONE) using:
- 4-hour candles as "structure" (open/close, wicks, range, volatility, volume, HVN proximity)
- 1-minute candles as fine-grained flow (returns, volume z-scores, intrabar position, VWAP distance)

Default data source: Bybit via CCXT (spot SOL/USDT) → deeper history than Coinbase.
(You can flip to Coinbase if you prefer, but Coinbase limits 300 candles per request.)

Notes
-----
- This is an end-to-end single-file script: fetch → engineer features → label → train → evaluate → infer.
- Sensible defaults are provided; tune thresholds and horizons to taste.
- Uses sklearn only (no external gradient-boosting libs required).

Usage
-----
python sol_model.py --days 60 --horizon_min 15 --pos_thr 0.004 --neg_thr -0.004 --proba_thr 0.60

Later, we can wire this into your Telegram alert system and your existing logging pipeline.
"""
from __future__ import annotations
import argparse
import math
from dataclasses import dataclass
from typing import Tuple, Optional

import ccxt  # pip install ccxt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import TimeSeriesSplit
from sklearn.utils.class_weight import compute_class_weight

# ---------------------------
# Config
# ---------------------------
PAIR = "SOL/USD"
EXCHANGE = "coinbase"  # fallback-friendly; bybit may 403 in US
TIMEZONE = "UTC"     # keep UTC for modeling; convert for alerts later

@dataclass
class LabelConfig:
    horizon_min: int = 15   # fwd horizon for labeling (minutes)
    pos_thr: float = 0.004  # +0.4% -> LONG
    neg_thr: float = -0.004 # -0.4% -> SHORT

@dataclass
class TrainConfig:
    proba_thr: float = 0.60      # min class probability to issue LONG/SHORT; else NONE
    tscv_splits: int = 5         # time-aware CV splits
    random_state: int = 42

# ---------------------------
# Helpers
# ---------------------------

def _unix_ms(dt: pd.Timestamp) -> int:
    return int(pd.Timestamp(dt).timestamp() * 1000)


def _fetch_ohlcv_paginated(exchange: ccxt.Exchange, symbol: str, timeframe: str, since_ms: int, limit: int = 1000) -> pd.DataFrame:
    all_rows = []
    cursor = since_ms
    while True:
        batch = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=cursor, limit=limit)
        if not batch:
            break
        all_rows.extend(batch)
        # advance cursor by last ts + 1ms to avoid duplicates
        cursor = batch[-1][0] + 1
        # stop if no growth
        if len(batch) < limit:
            break
    cols = ["timestamp","open","high","low","close","volume"]
    df = pd.DataFrame(all_rows, columns=cols)
    if df.empty:
        return df
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df.set_index("timestamp", inplace=True)
    return df


# ---------------------------
# Resampling helpers
# ---------------------------

def resample_1h_to_4h(df1h: pd.DataFrame) -> pd.DataFrame:
    """Aggregate 1H OHLCV to 4H bars (OHLCV), aligned to UTC with right labels."""
    df = df1h.copy().sort_index()
    agg = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    df4h = df.resample("4h", label="right", closed="right").agg(agg).dropna()
    return df4h

# ---------------------------
# Feature Engineering
# ---------------------------

def wick_features(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    body = (df["close"] - df["open"]).abs()
    range_ = df["high"] - df["low"]
    upper_wick = df["high"] - df[["close","open"]].max(axis=1)
    lower_wick = df[["close","open"]].min(axis=1) - df["low"]
    out["body"] = body
    out["range"] = range_
    out["upper_wick"] = upper_wick.clip(lower=0)
    out["lower_wick"] = lower_wick.clip(lower=0)
    out["body_pct_of_range"] = (body / (range_.replace(0, np.nan))).fillna(0)
    out["upper_wick_pct"] = (out["upper_wick"] / (range_.replace(0, np.nan))).fillna(0)
    out["lower_wick_pct"] = (out["lower_wick"] / (range_.replace(0, np.nan))).fillna(0)
    out["bullish"] = (df["close"] > df["open"]).astype(int)
    return out


def volatility_features(df: pd.DataFrame, windows=(6, 12, 24)) -> pd.DataFrame:
    # For 4H candles, 6=1 day, 30=~5 days, etc.
    out = pd.DataFrame(index=df.index)
    tr = (df[["high","close"]].max(axis=1) - df[["low","close"]].min(axis=1)).abs()
    out["tr"] = tr
    for w in windows:
        out[f"atr_{w}"] = tr.rolling(window=w, min_periods=1).mean()
        out[f"vol_{w}"] = (df["close"].pct_change().rolling(window=w, min_periods=1).std()).fillna(0)
    return out


def volume_features(df: pd.DataFrame, windows=(6, 24, 60)) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for w in windows:
        out[f"vol_sma_{w}"] = df["volume"].rolling(window=w, min_periods=1).mean()
        out[f"vol_z_{w}"] = ((df["volume"] - out[f"vol_sma_{w}"]) / (df["volume"].rolling(window=w, min_periods=2).std().replace(0, np.nan))).fillna(0)
    out["is_vol_spike"] = (df["volume"] > (df["volume"].rolling(24).mean() * 1.75)).astype(int)
    return out


def intrabar_position_1m(mins: pd.DataFrame, df4h: pd.DataFrame) -> pd.DataFrame:
    # Position of 1m price within *current* 4H range
    aligned = mins.join(df4h[["high","low","open","close"]], how="left", rsuffix="_4h").ffill().bfill()
    rng = (aligned["high_4h"] - aligned["low_4h"]).replace(0, np.nan)
    pos = (aligned["close"] - aligned["low_4h"]) / rng
    out = pd.DataFrame(index=mins.index)
    out["pos_in_4h"] = pos.clip(0,1).fillna(0.5)
    out["dist_to_4h_open"] = (aligned["close"] - aligned["open_4h"]) / aligned["open_4h"]
    out["dist_to_4h_close"] = (aligned["close"] - aligned["close_4h"]) / aligned["close_4h"]
    out["dist_to_4h_high"] = (aligned["close"] - aligned["high_4h"]) / aligned["high_4h"]
    out["dist_to_4h_low"] = (aligned["close"] - aligned["low_4h"]) / aligned["low_4h"]
    return out


def vwap(series_close: pd.Series, series_vol: pd.Series) -> pd.Series:
    pv = (series_close * series_vol).cumsum()
    vv = series_vol.cumsum().replace(0, np.nan)
    return (pv / vv)


def vwap_features_1m(mins: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=mins.index)
    # Group by calendar day (timezone-preserving) and broadcast daily VWAP to each minute
    day = mins.index.normalize()
    pv_sum = (mins["close"] * mins["volume"]).groupby(day).transform("sum")
    v_sum = mins["volume"].groupby(day).transform("sum").replace(0, np.nan)
    out["vwap_day"] = pv_sum / v_sum
    out["dist_to_vwap_day"] = (mins["close"] - out["vwap_day"]) / out["vwap_day"]
    return out


def hvn_features(mins: pd.DataFrame, lookback_hours: int = 72, bins: int = 40) -> pd.DataFrame:
    """High Volume Node proximity via simple volume-by-price over rolling window.
    Compute histogram of 1m volumes by price bins over last N hours; take top-N nodes.
    Return distance from current price to nearest HVN and whether price is above/below nearest HVN.
    """
    out = pd.DataFrame(index=mins.index)
    window = pd.Timedelta(hours=lookback_hours)
    prices = mins["close"]
    vols = mins["volume"]

    bin_edges_cache = None
    hvn_price_series = []
    dist_series = []

    # Efficient rolling: update every 30 minutes (coarse), forward-fill in between
    step = 30
    idx_list = list(range(0, len(mins), step))
    for i, start_idx in enumerate(idx_list):
        end_idx = min(start_idx + step, len(mins))
        t_end = mins.index[end_idx-1]
        t_start = t_end - window
        mask = (mins.index > t_start) & (mins.index <= t_end)
        p = prices.loc[mask]
        v = vols.loc[mask]
        if p.empty:
            hvn_price = np.nan
        else:
            pmin, pmax = float(p.min()), float(p.max())
            if pmax == pmin:
                hvn_price = pmax
            else:
                edges = np.linspace(pmin, pmax, bins+1)
                hist, edges = np.histogram(p, bins=edges, weights=v)
                hvn_idx = hist.argmax()
                hvn_price = (edges[hvn_idx] + edges[hvn_idx+1]) / 2
        hvn_price_series.extend([hvn_price] * (end_idx - start_idx))

    # Align lengths
    hvn_price_series = pd.Series(hvn_price_series, index=mins.index)
    out["hvn_price"] = hvn_price_series.fillna(method="ffill").fillna(method="bfill")
    out["dist_to_hvn"] = (mins["close"] - out["hvn_price"]) / out["hvn_price"]
    out["dist_to_hvn"] = out["dist_to_hvn"].replace([np.inf, -np.inf], np.nan).fillna(0)
    out["above_hvn"] = (mins["close"] > out["hvn_price"]).astype(float)
    return out


def one_minute_flow_features(mins: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=mins.index)
    out["ret_1"] = mins["close"].pct_change()
    out["ret_5"] = mins["close"].pct_change(5)
    out["ret_15"] = mins["close"].pct_change(15)
    out["vol_sma_20"] = mins["volume"].rolling(20).mean()
    out["vol_z_20"] = ((mins["volume"] - out["vol_sma_20"]) / (mins["volume"].rolling(20).std().replace(0, np.nan))).fillna(0)
    out["range_15"] = (mins["close"].rolling(15).max() - mins["close"].rolling(15).min()) / mins["close"].rolling(15).mean()
    return out

# ---------------------------
# Labeling
# ---------------------------

def make_labels(mins: pd.DataFrame, cfg: LabelConfig) -> pd.Series:
    future = mins["close"].shift(-cfg.horizon_min)
    ret = (future - mins["close"]) / mins["close"]
    y = pd.Series(2, index=mins.index)  # default NONE=2
    y[ret >= cfg.pos_thr] = 1          # LONG=1
    y[ret <= cfg.neg_thr] = 0          # SHORT=0
    return y

# ---------------------------
# Modeling & Backtesting
# ---------------------------

def build_dataset(df4h: pd.DataFrame, mins: pd.DataFrame, label_cfg: LabelConfig) -> Tuple[pd.DataFrame, pd.Series]:
    # 4H features (structure)
    f4a = wick_features(df4h)
    f4b = volatility_features(df4h)
    f4c = volume_features(df4h)
    f4 = pd.concat([f4a, f4b, f4c], axis=1)

    # Broadcast 4H features to 1m via forward-fill
    f4_1m = mins[["close","volume"]].join(f4, how="left").ffill()

    # 1m features (flow)
    f1_pos = intrabar_position_1m(mins, df4h)
    f1_vwap = vwap_features_1m(mins)
    f1_hvn = hvn_features(mins)
    f1_flow = one_minute_flow_features(mins)

    X = pd.concat([f4_1m.drop(columns=["close","volume"]), f1_pos, f1_vwap, f1_hvn, f1_flow], axis=1)
    y = make_labels(mins, label_cfg)

    # Clean / impute light
    X = X.replace([np.inf, -np.inf], np.nan)
    valid = X.dropna().index.intersection(y.dropna().index)
    if len(valid) == 0:
        # fallback to avoid empty dataset on fresh histories
        X = X.fillna(method="ffill").fillna(method="bfill")
        valid = X.index.intersection(y.dropna().index)

    X = X.loc[valid]
    y = y.loc[valid]
    return X, y


def train_model(X: pd.DataFrame, y: pd.Series, tcfg: TrainConfig) -> Tuple[RandomForestClassifier, dict]:
    classes = np.array([0,1,2])
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y.values)
    class_weight = {cls: w for cls, w in zip(classes, weights)}

    clf = RandomForestClassifier(
        n_estimators=400,
        max_depth=None,
        min_samples_split=20,
        min_samples_leaf=10,
        random_state=tcfg.random_state,
        class_weight=class_weight,
        n_jobs=-1
    )

    # Time-aware CV evaluation
    tscv = TimeSeriesSplit(n_splits=tcfg.tscv_splits)
    reports = []
    for fold, (train_idx, test_idx) in enumerate(tscv.split(X), 1):
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]
        clf.fit(X_tr, y_tr)
        y_pred = clf.predict(X_te)
        report = classification_report(y_te, y_pred, output_dict=True, zero_division=0)
        reports.append({"fold": fold, "report": report})

    # Final fit on full data
    clf.fit(X, y)
    return clf, {"cv_reports": reports}


def walkforward_backtest(
    X: pd.DataFrame,
    y: pd.Series,
    price: pd.Series,
    horizon_min: int,
    proba_thr: float,
    fee_bps: float,
    min_train: int,
) -> pd.DataFrame:
    """Simple walk-forward: train up to t, predict at t, realize at t+horizon.
    Returns a DataFrame indexed by timestamp with columns side, p_long, p_short, ret, equity.
    Gracefully returns an empty DataFrame if no records are produced.
    """
    # Align inputs
    common_idx = X.index.intersection(y.index).intersection(price.index)
    X = X.loc[common_idx]
    y = y.loc[common_idx]
    price = price.loc[common_idx]

    # Ensure strictly increasing time
    X = X.sort_index()
    y = y.sort_index()
    price = price.sort_index()

    records = []
    equity = 1.0
    fee = (fee_bps or 0.0) / 10000.0  # bps to fraction per side

    # Walk-forward
    n = len(X)
    if n <= min_train + horizon_min:
        # Not enough data to simulate
        return pd.DataFrame(index=pd.DatetimeIndex([], name="ts"), columns=["side","p_long","p_short","ret","equity"]).astype({"equity":"float64"})

    for i in range(min_train, n - horizon_min):
        X_tr, y_tr = X.iloc[:i], y.iloc[:i]
        x_row = X.iloc[i]

        # Train a compact RF per step (balanced_subsample to avoid class-weight errors on missing classes)
        clf = RandomForestClassifier(
            n_estimators=250,
            max_depth=None,
            min_samples_split=20,
            min_samples_leaf=10,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced_subsample",
        )
        clf.fit(X_tr, y_tr)

        # Predict probabilities with aligned feature names
        x_in = pd.DataFrame([x_row])
        if hasattr(clf, "feature_names_in_"):
            x_in = x_in.reindex(columns=clf.feature_names_in_)
        x_in = x_in.fillna(0)
        proba = clf.predict_proba(x_in)[0]
        p_short, p_long, p_none = float(proba[0]), float(proba[1]), float(proba[2])

        # Decide side
        side = "NONE"
        conf = max(p_short, p_long)
        if p_long >= proba_thr and p_long > p_short:
            side = "LONG"
        elif p_short >= proba_thr and p_short > p_long:
            side = "SHORT"

        # Realize forward return
        px_entry = float(price.iloc[i])
        px_exit = float(price.iloc[i + horizon_min])
        raw_ret = (px_exit - px_entry) / px_entry
        trade_ret = 0.0
        if side == "LONG":
            trade_ret = raw_ret - 2.0 * fee
        elif side == "SHORT":
            trade_ret = (-raw_ret) - 2.0 * fee
        # carry equity (only changes if trade was taken)
        if side != "NONE":
            equity *= (1.0 + trade_ret)
        ts = X.index[i]
        records.append({"ts": ts, "side": side, "p_long": p_long, "p_short": p_short, "ret": trade_ret, "equity": equity})

    if not records:
        return pd.DataFrame(index=pd.DatetimeIndex([], name="ts"), columns=["side","p_long","p_short","ret","equity"]).astype({"equity":"float64"})

    bt = pd.DataFrame.from_records(records)
    bt.index = pd.DatetimeIndex(bt.pop("ts"), name="ts")
    return bt
    bt["equity"] = (1 + bt["ret"]).cumprod()
    bt["trade"] = (bt["side"] != "NONE").astype(int)
    return bt


def backtest_summary(bt: pd.DataFrame) -> dict:
    if bt.empty:
        return {"trades": 0}
    trades = bt[bt["trade"] == 1]
    wins = trades[trades["ret"] > 0]
    losses = trades[trades["ret"] <= 0]
    win_rate = (len(wins) / max(1, len(trades))) if len(trades) else 0
    avg_ret = trades["ret"].mean() if len(trades) else 0
    med_ret = trades["ret"].median() if len(trades) else 0
    # Rough max drawdown on equity curve
    eq = bt["equity"].copy()
    roll_max = eq.cummax()
    drawdown = (eq / roll_max) - 1.0
    mdd = drawdown.min()
    # Simple Sharpe per trade (no annualization)
    sharpe = trades["ret"].mean() / (trades["ret"].std() + 1e-9) if len(trades) > 1 else 0
    return {
        "trades": int(len(trades)),
        "win_rate": float(win_rate),
        "avg_ret": float(avg_ret),
        "median_ret": float(med_ret),
        "equity_final": float(eq.iloc[-1]),
        "max_drawdown": float(mdd),
        "sharpe_per_trade": float(sharpe),
    }


def predict_signal(model: RandomForestClassifier, X_row: pd.Series, proba_thr: float) -> Tuple[str, float]:
    # Build a single-row DataFrame with aligned feature names to avoid sklearn warnings
    if isinstance(X_row, pd.Series):
        X_in = pd.DataFrame([X_row])
    else:
        X_in = pd.DataFrame([pd.Series(X_row)])
    if hasattr(model, "feature_names_in_"):
        X_in = X_in.reindex(columns=model.feature_names_in_)
    X_in = X_in.fillna(0)

    proba = model.predict_proba(X_in)[0]
    # proba order matches classes_ (0=SHORT,1=LONG,2=NONE)
    p_short, p_long, p_none = proba[0], proba[1], proba[2]
    if p_long >= proba_thr and p_long > p_short:
        return "LONG", float(p_long)
    if p_short >= proba_thr and p_short > p_long:
        return "SHORT", float(p_short)
    return "NONE", float(p_none)

# ---------------------------
# Two-stage (1m-only) components: bias & execution both on 1-minute bars
# ---------------------------

@dataclass
class OneMinParams:
    bias_fast: int = 60     # ~1h EMA on 1m bars
    bias_mid: int = 120     # ~2h
    bias_slow: int = 240    # ~4h
    window_min: int = 240   # range window for position filter
    entry_ema: int = 20     # micro trigger EMA
    entry_retest_window: int = 5
    atr_1m: int = 14
    tp_atr: float = 2.0
    sl_atr: float = 1.5
    vol_z_entry: float = 0.0
    pos_low_thresh: float = 0.35
    pos_high_thresh: float = 0.65
    cooldown_min: int = 30
    conservative_bar_fill: bool = True


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def atr(df: pd.DataFrame, window: int) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        (df["high"] - df["low"]).abs(),
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    w = max(1, int(round(window)))
    return tr.rolling(window=w, min_periods=1).mean()


def compute_1m_bias(mins: pd.DataFrame, p: OneMinParams) -> pd.Series:
    c = mins["close"]
    ema_fast = ema(c, p.bias_fast)
    ema_mid  = ema(c, p.bias_mid)
    ema_slow = ema(c, p.bias_slow)
    bias = pd.Series("NONE", index=mins.index, dtype=object)
    bias[(c > ema_slow) & (ema_fast > ema_mid)] = "LONG"
    bias[(c < ema_slow) & (ema_fast < ema_mid)] = "SHORT"
    return bias.rename("bias")


def pos_in_window_features(mins: pd.DataFrame, window_min: int) -> pd.DataFrame:
    out = pd.DataFrame(index=mins.index)
    hi = mins["high"].rolling(window=window_min, min_periods=1).max()
    lo = mins["low"].rolling(window=window_min, min_periods=1).min()
    rng = (hi - lo).replace(0, np.nan)
    out["pos_in_win"] = ((mins["close"] - lo) / rng).clip(0,1).fillna(0.5)
    out["dist_to_win_hi"] = (mins["close"] - hi) / hi
    out["dist_to_win_lo"] = (mins["close"] - lo) / lo
    return out


def backtest_two_stage_1m(mins: pd.DataFrame, p: OneMinParams) -> pd.DataFrame:
    bias_1m = compute_1m_bias(mins, p)
    flow = one_minute_flow_features(mins)
    ema_entry = ema(mins["close"], p.entry_ema)
    atr1m = atr(mins, p.atr_1m)
    winpos = pos_in_window_features(mins, p.window_min)

    long_cross  = (mins["close"] > ema_entry) & (mins["close"].shift(1) <= ema_entry.shift(1))
    short_cross = (mins["close"] < ema_entry) & (mins["close"].shift(1) >= ema_entry.shift(1))
    # allow entries within N minutes after a cross if still on the correct side of EMA
    long_recent = long_cross.rolling(window=p.entry_retest_window, min_periods=1).max().astype(bool)
    short_recent = short_cross.rolling(window=p.entry_retest_window, min_periods=1).max().astype(bool)
    long_trig = long_recent & (mins["close"] > ema_entry)
    short_trig = short_recent & (mins["close"] < ema_entry)

    # diagnostics (pre-filter counts)
    mask_long_bias = (bias_1m == "LONG")
    mask_short_bias = (bias_1m == "SHORT")
    diag_long_trig = (long_trig & mask_long_bias)
    diag_short_trig = (short_trig & mask_short_bias)
    diag_long_zone = diag_long_trig & (winpos["pos_in_win"] <= p.pos_low_thresh)
    diag_short_zone = diag_short_trig & (winpos["pos_in_win"] >= p.pos_high_thresh)
    diag_long_vol = diag_long_zone & (flow["vol_z_20"] > p.vol_z_entry)
    diag_short_vol = diag_short_zone & (flow["vol_z_20"] > p.vol_z_entry)
    print(f"Diagnostics (1m two-stage): bias LONG mins={int(mask_long_bias.sum())}, SHORT mins={int(mask_short_bias.sum())}")
    print(f"  Long: triggers={int(diag_long_trig.sum())}, zone-pass={int(diag_long_zone.sum())}, vol-pass={int(diag_long_vol.sum())}")
    print(f"  Short: triggers={int(diag_short_trig.sum())}, zone-pass={int(diag_short_zone.sum())}, vol-pass={int(diag_short_vol.sum())}")

    cooldown = pd.Timedelta(minutes=p.cooldown_min)
    last_exit_time = pd.Timestamp.min.tz_localize("UTC")

    records = []
    i = 0
    idx = mins.index

    while i < len(mins):
        t = idx[i]
        b = bias_1m.iloc[i]
        if t < last_exit_time + cooldown:
            i += 1
            continue

        took_trade = False
        if b == "LONG":
            if (winpos.loc[t, "pos_in_win"] <= p.pos_low_thresh) and long_trig.iloc[i] and (flow.loc[t, "vol_z_20"] > p.vol_z_entry):
                entry = float(mins.loc[t, "close"]) ; sl = entry - p.sl_atr * float(atr1m.loc[t]) ; tp = entry + p.tp_atr * float(atr1m.loc[t])
                j = i + 1 ; exit_ts = None ; exit_reason = None ; exit_px = None
                while j < len(mins):
                    hi = float(mins.iloc[j]["high"]) ; lo = float(mins.iloc[j]["low"]) ; tsj = idx[j]
                    if p.conservative_bar_fill:
                        if lo <= sl: exit_ts, exit_reason, exit_px = tsj, "SL", sl ; break
                        if hi >= tp: exit_ts, exit_reason, exit_px = tsj, "TP", tp ; break
                    else:
                        if hi >= tp: exit_ts, exit_reason, exit_px = tsj, "TP", tp ; break
                        if lo <= sl: exit_ts, exit_reason, exit_px = tsj, "SL", sl ; break
                    if bias_1m.iloc[j] != "LONG":
                        exit_ts, exit_reason, exit_px = tsj, "BIAS_FLIP", float(mins.iloc[j]["close"]) ; break
                    j += 1
                if exit_ts is None:
                    exit_ts, exit_reason, exit_px = idx[-1], "EOD", float(mins.iloc[-1]["close"]) 
                r = (exit_px - entry) / (entry - sl)
                records.append({"entry_ts": t, "side": "LONG", "entry": entry, "sl": sl, "tp": tp, "exit_ts": exit_ts, "exit_px": exit_px, "reason": exit_reason, "R": r})
                last_exit_time = exit_ts
                i = j + 1 ; took_trade = True
        if (not took_trade) and b == "SHORT":
            if (winpos.loc[t, "pos_in_win"] >= p.pos_high_thresh) and short_trig.iloc[i] and (flow.loc[t, "vol_z_20"] > p.vol_z_entry):
                entry = float(mins.loc[t, "close"]) ; sl = entry + p.sl_atr * float(atr1m.loc[t]) ; tp = entry - p.tp_atr * float(atr1m.loc[t])
                j = i + 1 ; exit_ts = None ; exit_reason = None ; exit_px = None
                while j < len(mins):
                    hi = float(mins.iloc[j]["high"]) ; lo = float(mins.iloc[j]["low"]) ; tsj = idx[j]
                    if p.conservative_bar_fill:
                        if hi >= sl: exit_ts, exit_reason, exit_px = tsj, "SL", sl ; break
                        if lo <= tp: exit_ts, exit_reason, exit_px = tsj, "TP", tp ; break
                    else:
                        if lo <= tp: exit_ts, exit_reason, exit_px = tsj, "TP", tp ; break
                        if hi >= sl: exit_ts, exit_reason, exit_px = tsj, "SL", sl ; break
                    if bias_1m.iloc[j] != "SHORT":
                        exit_ts, exit_reason, exit_px = tsj, "BIAS_FLIP", float(mins.iloc[j]["close"]) ; break
                    j += 1
                if exit_ts is None:
                    exit_ts, exit_reason, exit_px = idx[-1], "EOD", float(mins.iloc[-1]["close"]) 
                r = (entry - exit_px) / (sl - entry)
                records.append({"entry_ts": t, "side": "SHORT", "entry": entry, "sl": sl, "tp": tp, "exit_ts": exit_ts, "exit_px": exit_px, "reason": exit_reason, "R": r})
                last_exit_time = exit_ts
                i = j + 1 ; took_trade = True
        if not took_trade:
            i += 1

    if not records:
        return pd.DataFrame(columns=["entry_ts","side","entry","sl","tp","exit_ts","exit_px","reason","R"]).astype({"entry":"float64","sl":"float64","tp":"float64","exit_px":"float64","R":"float64"})

    trades = pd.DataFrame.from_records(records)
    trades["entry_ts"] = pd.to_datetime(trades["entry_ts"], utc=True)
    trades["exit_ts"]  = pd.to_datetime(trades["exit_ts"], utc=True)
    trades.set_index("entry_ts", inplace=True)
    return trades

# ---------------------------
# Two-stage (4H bias + 1m execution) components
# ---------------------------

@dataclass
class TwoStageParams:
    ema_fast_4h: int = 20
    ema_mid_4h: int = 50
    ema_slow_4h: int = 200
    pos_low_thresh: float = 0.35
    pos_high_thresh: float = 0.65
    vol_z_entry: float = 0.0
    ema_fast_1m: int = 20
    atr_1m: int = 14
    tp_atr: float = 2.0
    sl_atr: float = 1.5
    cooldown_min: int = 30
    max_trades_per_4h: int = 1
    conservative_bar_fill: bool = True


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def atr(df: pd.DataFrame, window: int) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        (df["high"] - df["low"]).abs(),
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    w = max(1, int(round(window)))
    return tr.rolling(window=w, min_periods=1).mean()


def compute_4h_bias(df4h: pd.DataFrame, p: TwoStageParams) -> pd.Series:
    c = df4h["close"]
    ema_fast = ema(c, p.ema_fast_4h)
    ema_mid  = ema(c, p.ema_mid_4h)
    ema_slow = ema(c, p.ema_slow_4h)
    bias = pd.Series("NONE", index=df4h.index, dtype=object)
    bias[(c > ema_slow) & (ema_fast > ema_mid)] = "LONG"
    bias[(c < ema_slow) & (ema_fast < ema_mid)] = "SHORT"
    return bias.rename("bias")


def cross_up(series: pd.Series, ref: pd.Series) -> pd.Series:
    return (series > ref) & (series.shift(1) <= ref.shift(1))


def cross_down(series: pd.Series, ref: pd.Series) -> pd.Series:
    return (series < ref) & (series.shift(1) >= ref.shift(1))


def backtest_two_stage(df4h: pd.DataFrame, mins: pd.DataFrame, p: TwoStageParams) -> pd.DataFrame:
    """Backtest: use 4H bias, trade entries on 1m with EMA20 cross + volume and 4H position filters."""
    bias_4h = compute_4h_bias(df4h, p)
    pos = intrabar_position_1m(mins, df4h)
    flow = one_minute_flow_features(mins)
    ema20 = ema(mins["close"], p.ema_fast_1m)
    atr1m = atr(mins, p.atr_1m)

    bias_1m = mins.join(bias_4h, how="left").ffill()["bias"].fillna("NONE")
    long_cross  = cross_up(mins["close"], ema20)
    short_cross = cross_down(mins["close"], ema20)

    cooldown = pd.Timedelta(minutes=p.cooldown_min)
    last_exit_time = pd.Timestamp.min.tz_localize("UTC")

    per_4h_count = {}
    records = []
    i = 0
    idx = mins.index

    while i < len(mins):
        t = idx[i]
        b = bias_1m.iloc[i]
        if t < last_exit_time + cooldown:
            i += 1
            continue
        current_4h_pos = df4h.index.searchsorted(t, side='right')-1
        current_4h = df4h.index[max(current_4h_pos, 0)] if len(df4h.index) else None
        if current_4h is not None:
            per_4h_count.setdefault(current_4h, 0)
            if per_4h_count[current_4h] >= p.max_trades_per_4h:
                i += 1
                continue

        took_trade = False
        if b == "LONG":
            if (pos.loc[t, "pos_in_4h"] <= p.pos_low_thresh) and long_trig.iloc[i] and (flow.loc[t, "vol_z_20"] > p.vol_z_entry):
                entry = float(mins.loc[t, "close"]) ; sl = entry - p.sl_atr * float(atr1m.loc[t]) ; tp = entry + p.tp_atr * float(atr1m.loc[t])
                j = i + 1 ; exit_ts = None ; exit_reason = None ; exit_px = None
                while j < len(mins):
                    hi = float(mins.iloc[j]["high"]) ; lo = float(mins.iloc[j]["low"]) ; tsj = idx[j]
                    if p.conservative_bar_fill:
                        if lo <= sl: exit_ts, exit_reason, exit_px = tsj, "SL", sl ; break
                        if hi >= tp: exit_ts, exit_reason, exit_px = tsj, "TP", tp ; break
                    else:
                        if hi >= tp: exit_ts, exit_reason, exit_px = tsj, "TP", tp ; break
                        if lo <= sl: exit_ts, exit_reason, exit_px = tsj, "SL", sl ; break
                    if bias_1m.iloc[j] != "LONG":
                        exit_ts, exit_reason, exit_px = tsj, "BIAS_FLIP", float(mins.iloc[j]["close"]) ; break
                    j += 1
                if exit_ts is None:
                    exit_ts, exit_reason, exit_px = idx[-1], "EOD", float(mins.iloc[-1]["close"]) 
                r = (exit_px - entry) / (entry - sl)
                records.append({"entry_ts": t, "side": "LONG", "entry": entry, "sl": sl, "tp": tp, "exit_ts": exit_ts, "exit_px": exit_px, "reason": exit_reason, "R": r})
                last_exit_time = exit_ts
                if current_4h is not None: per_4h_count[current_4h] += 1
                i = j + 1 ; took_trade = True
        if (not took_trade) and b == "SHORT":
            if (pos.loc[t, "pos_in_4h"] >= p.pos_high_thresh) and short_trig.iloc[i] and (flow.loc[t, "vol_z_20"] > p.vol_z_entry):
                entry = float(mins.loc[t, "close"]) ; sl = entry + p.sl_atr * float(atr1m.loc[t]) ; tp = entry - p.tp_atr * float(atr1m.loc[t])
                j = i + 1 ; exit_ts = None ; exit_reason = None ; exit_px = None
                while j < len(mins):
                    hi = float(mins.iloc[j]["high"]) ; lo = float(mins.iloc[j]["low"]) ; tsj = idx[j]
                    if p.conservative_bar_fill:
                        if hi >= sl: exit_ts, exit_reason, exit_px = tsj, "SL", sl ; break
                        if lo <= tp: exit_ts, exit_reason, exit_px = tsj, "TP", tp ; break
                    else:
                        if lo <= tp: exit_ts, exit_reason, exit_px = tsj, "TP", tp ; break
                        if hi >= sl: exit_ts, exit_reason, exit_px = tsj, "SL", sl ; break
                    if bias_1m.iloc[j] != "SHORT":
                        exit_ts, exit_reason, exit_px = tsj, "BIAS_FLIP", float(mins.iloc[j]["close"]) ; break
                    j += 1
                if exit_ts is None:
                    exit_ts, exit_reason, exit_px = idx[-1], "EOD", float(mins.iloc[-1]["close"]) 
                r = (entry - exit_px) / (sl - entry)
                records.append({"entry_ts": t, "side": "SHORT", "entry": entry, "sl": sl, "tp": tp, "exit_ts": exit_ts, "exit_px": exit_px, "reason": exit_reason, "R": r})
                last_exit_time = exit_ts
                if current_4h is not None: per_4h_count[current_4h] += 1
                i = j + 1 ; took_trade = True
        if not took_trade:
            i += 1

    if not records:
        return pd.DataFrame(columns=["entry_ts","side","entry","sl","tp","exit_ts","exit_px","reason","R"]).astype({"entry":"float64","sl":"float64","tp":"float64","exit_px":"float64","R":"float64"})

    trades = pd.DataFrame.from_records(records)
    trades["entry_ts"] = pd.to_datetime(trades["entry_ts"], utc=True)
    trades["exit_ts"]  = pd.to_datetime(trades["exit_ts"], utc=True)
    trades.set_index("entry_ts", inplace=True)
    return trades

# ---------------------------
# Fetch
# ---------------------------

def get_exchange(name: str):
    if name.lower() == "coinbase":
        return ccxt.coinbase()
    if name.lower() == "bybit":
        ex = ccxt.bybit()
        ex.options["defaultType"] = "spot"
        return ex
    raise ValueError("Unsupported exchange")


def fetch_data(days: int = 60) -> Tuple[pd.DataFrame, pd.DataFrame]:
    ex = get_exchange(EXCHANGE)
    since = pd.Timestamp.utcnow() - pd.Timedelta(days=days)
    since_ms = _unix_ms(since)

    df1h = _fetch_ohlcv_paginated(ex, PAIR, timeframe="1h", since_ms=since_ms)
    mins = _fetch_ohlcv_paginated(ex, PAIR, timeframe="1m", since_ms=since_ms)
    df4h = resample_1h_to_4h(df1h)

    # Sanity
    if mins.empty or df1h.empty:
        raise RuntimeError("No data fetched; adjust 'days' or check exchange symbol.")
    if df4h.empty:
        raise RuntimeError("4H aggregation produced no rows; reduce --days or check source.")

    # Keep only needed cols, standard names
    for df in (df4h, mins):
        df[["open","high","low","close","volume"]] = df[["open","high","low","close","volume"]].astype(float)

    return df4h, mins

# ---------------------------
# Main
# ---------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=60)
    parser.add_argument("--horizon_min", type=int, default=15)
    parser.add_argument("--pos_thr", type=float, default=0.004)
    parser.add_argument("--neg_thr", type=float, default=-0.004)
    parser.add_argument("--proba_thr", type=float, default=0.60)
    parser.add_argument("--fee_bps", type=float, default=8.0, help="fee per side in bps (0.01% = 1 bps)")
    parser.add_argument("--min_train", type=int, default=500, help="minimum training rows before first prediction")
    parser.add_argument("--bt", action="store_true", help="run walk-forward backtest (single-stage ML)")
    parser.add_argument("--two_stage", action="store_true", help="run 1m-bias + 1m-execution backtest")
    parser.add_argument("--exchange_order", type=str, default="coinbase,bybit", help="comma-separated list, e.g. 'coinbase,bybit'")
    parser.add_argument("--pair", type=str, default=None, help="override pair, e.g. 'SOL/USDT'")
    parser.add_argument("--proxy", type=str, default=None, help="optional http(s) proxy, e.g. 'http://host:port'")
    # Two-stage params (1m-only)
    parser.add_argument("--ts_tp_atr", type=float, default=2.0)
    parser.add_argument("--ts_sl_atr", type=float, default=1.5)
    parser.add_argument("--ts_cooldown", type=int, default=30)
    parser.add_argument("--ts_pos_low", type=float, default=0.35)
    parser.add_argument("--ts_pos_high", type=float, default=0.65)
    parser.add_argument("--ts_vol_z", type=float, default=0.0)
    parser.add_argument("--ts_window_min", type=int, default=240)
    parser.add_argument("--ts_bias_fast", type=int, default=60)
    parser.add_argument("--ts_bias_mid", type=int, default=120)
    parser.add_argument("--ts_bias_slow", type=int, default=240)
    parser.add_argument("--ts_entry_ema", type=int, default=20)
    args = parser.parse_args()

    label_cfg = LabelConfig(horizon_min=args.horizon_min, pos_thr=args.pos_thr, neg_thr=args.neg_thr)
    train_cfg = TrainConfig(proba_thr=args.proba_thr)

    print("Fetching data…")
    df4h, mins = fetch_data(days=args.days)

    # EARLY two-stage branch: 1m-only bias + execution
    if args.two_stage:
        print("Running two-stage backtest (1m bias + 1m execution)…")
        om = OneMinParams(
            bias_fast=args.ts_bias_fast,
            bias_mid=args.ts_bias_mid,
            bias_slow=args.ts_bias_slow,
            window_min=args.ts_window_min,
            entry_ema=args.ts_entry_ema,
            atr_1m=14,
            tp_atr=args.ts_tp_atr,
            sl_atr=args.ts_sl_atr,
            vol_z_entry=args.ts_vol_z,
            pos_low_thresh=args.ts_pos_low,
            pos_high_thresh=args.ts_pos_high,
            cooldown_min=args.ts_cooldown,
        )
        trades = backtest_two_stage_1m(mins, om)
        if trades.empty:
            print("Two-stage backtest produced no trades. Try loosening thresholds (e.g., --ts_vol_z -0.2), widen pos window (--ts_pos_low 0.4 / --ts_pos_high 0.6), or extend --days.")
        else:
            n = len(trades)
            wins = (trades["reason"] == "TP").sum()
            win_rate = wins / n if n else 0.0
            avg_R = trades["R"].mean() if n else 0.0
            print(f"Trades: {n} | Win rate: {win_rate:.2%} | Avg R: {avg_R:.3f}")
            print("Last 5 trades:")
            print(trades.tail(5)[["side","entry","sl","tp","exit_px","reason","R"]])
        return

    print("Engineering features…")
    X, y = build_dataset(df4h, mins, label_cfg)
    print(f"Dataset rows: {len(X):,} | Features: {X.shape[1]} | Class balance: {y.value_counts().to_dict()}")

    # Two-stage path (4H bias + 1m execution backtest)
    if args.two_stage:
        print("Running two-stage backtest (4H bias + 1m execution)…")
        ts_params = TwoStageParams(
            tp_atr=args.ts_tp_atr,
            sl_atr=args.ts_sl_atr,
            cooldown_min=args.ts_cooldown,
            max_trades_per_4h=args.ts_max_trades_per_4h,
            pos_low_thresh=args.ts_pos_low,
            pos_high_thresh=args.ts_pos_high,
            vol_z_entry=args.ts_vol_z,
        )
        trades = backtest_two_stage(df4h, mins, ts_params)
        if trades.empty:
            print("Two-stage backtest produced no trades. Try loosening thresholds (e.g., --ts_vol_z -0.2) or extend --days.")
        else:
            n = len(trades)
            wins = (trades["reason"] == "TP").sum()
            win_rate = wins / n if n else 0.0
            avg_R = trades["R"].mean() if n else 0.0
            print(f"Trades: {n} | Win rate: {win_rate:.2%} | Avg R: {avg_R:.3f}")
            print("Last 5 trades:")
            print(trades.tail(5)[["side","entry","sl","tp","exit_px","reason","R"]])
        return

    print("Training model (time-aware CV)…")
    model, meta = train_model(X, y, train_cfg)
    for fold_info in meta["cv_reports"]:
        fold = fold_info["fold"]
        rep = fold_info["report"]
        print(f"Fold {fold} classification report:")
        print(pd.DataFrame(rep).T)

    # Inference on the most recent row
    latest_idx = X.index[-1]
    sig, conf = predict_signal(model, X.iloc[-1], train_cfg.proba_thr)
    print(f"Latest signal @ {latest_idx}: {sig} (confidence={conf:.2%})")

if __name__ == "__main__":
    main()
