
# feature_builder.py
# Build rich, model-friendly features from raw 1-minute OHLCV candles.
# Usage:
#   import pandas as pd
#   from feature_builder import build_features, make_target
#   df = ...  # DataFrame with columns: ["time","open","high","low","close","volume"]
#   X = build_features(df, use_optional_technicals=True)
#   y = make_target(df, horizon=3, target_type="classification", thr_bp=2.0)
#
# Dependencies: pandas, numpy. Optional: ta (for RSI/MACD/Stoch, etc.)

from __future__ import annotations
import math
import numpy as np
import pandas as pd
from pandas.api.types import is_datetime64_any_dtype, is_datetime64tz_dtype

try:
    import ta  # optional; we'll guard usage
    _HAS_TA = True
except Exception:
    _HAS_TA = False

# ----------------------------
# Helpers
# ----------------------------
def _safe_div(a, b):
    return a / (b + 1e-12)

def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def _rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    # Ensure min_periods <= window (works for tiny windows like 3)
    mp = min(window, max(2, window // 3))
    roll = series.rolling(window, min_periods=mp)
    mu = roll.mean()
    std = roll.std(ddof=0)
    return (series - mu) / (std + 1e-12)
# ----------------------------
# Core feature builders
# ----------------------------
def add_core_price_features(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    o, h, l, c, v = df["open"], df["high"], df["low"], df["close"], df["volume"]

    out["open"] = o
    out["high"] = h
    out["low"] = l
    out["close"] = c
    out["volume"] = v

    # Primitive differences
    out["hl_range"] = h - l
    out["oc_change"] = c - o
    out["return_1m"] = _safe_div(c, o) - 1.0
    out["log_return"] = np.log(_safe_div(c, o))

    return out

def add_candle_geometry(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    o, h, l, c = df["open"], df["high"], df["low"], df["close"]
    body = (c - o).abs()
    upper_wick = h - np.maximum(o, c)
    lower_wick = np.minimum(o, c) - l
    rng = (h - l).replace(0, np.nan)

    out["body_abs"] = body
    out["upper_wick"] = upper_wick
    out["lower_wick"] = lower_wick
    out["body_to_range"] = _safe_div(body, rng)
    out["upper_to_range"] = _safe_div(upper_wick, rng)
    out["lower_to_range"] = _safe_div(lower_wick, rng)
    out["dir"] = np.sign(c - o).fillna(0)  # +1 bull, -1 bear, 0 doji

    return out

def add_rolling_stats(df: pd.DataFrame, windows=(3,5,15,30)) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    c, v = df["close"], df["volume"]
    for w in windows:
        out[f"ret_{w}m"] = _safe_div(c, c.shift(w)) - 1.0
        out[f"ret_{w}m_log"] = np.log(_safe_div(c, c.shift(w)))
        out[f"close_sma_{w}"] = c.rolling(w, min_periods=max(2, w//2)).mean()
        out[f"close_std_{w}"] = c.rolling(w, min_periods=max(2, w//2)).std(ddof=0)
        out[f"vol_sma_{w}"] = v.rolling(w, min_periods=max(2, w//2)).mean()
        out[f"vol_std_{w}"] = v.rolling(w, min_periods=max(2, w//2)).std(ddof=0)
        out[f"ret_z_{w}"] = _rolling_zscore(out[f"ret_{w}m"].fillna(0), w)
        out[f"vol_z_{w}"] = _rolling_zscore(v.fillna(0), w)
        out[f"min_{w}"] = df["low"].rolling(w, min_periods=max(2, w//2)).min()
        out[f"max_{w}"] = df["high"].rolling(w, min_periods=max(2, w//2)).max()
        out[f"range_frac_{w}"] = _safe_div(out[f"max_{w}"] - out[f"min_{w}"], c)
    return out

def add_volatility_features(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    h, l, c = df["high"], df["low"], df["close"]
    prev_close = c.shift(1)
    tr = np.maximum(h - l, np.maximum((h - prev_close).abs(), (l - prev_close).abs()))
    atr = tr.rolling(14, min_periods=14).mean()
    out["atr14"] = atr
    out["range_frac"] = _safe_div(h - l, c)
    out["atr_change_1"] = _safe_div(atr, atr.shift(1)) - 1.0
    out["atr_z_100"] = _rolling_zscore(atr.fillna(0), 100)
    # Volume spike proxy
    vol = df["volume"]
    out["vol_spike_20"] = _safe_div(vol, vol.rolling(20, min_periods=5).mean())
    out["vol_spike_60"] = _safe_div(vol, vol.rolling(60, min_periods=10).mean())
    return out

def add_time_features(df: pd.DataFrame, time_col: str = "time") -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    # Normalize time to timezone-aware UTC
    if not is_datetime64_any_dtype(df[time_col]):
        t = pd.to_datetime(df[time_col], utc=True)
    else:
        t = df[time_col]
        try:
            if is_datetime64tz_dtype(t.dtype):
                t = t.dt.tz_convert("UTC")
            else:
                t = t.dt.tz_localize("UTC")
        except Exception:
            t = pd.to_datetime(t, utc=True)

    minute_of_day = t.dt.hour * 60 + t.dt.minute
    out["minute_of_day"] = minute_of_day
    out["sin_t"] = np.sin(2 * np.pi * minute_of_day / 1440.0)
    out["cos_t"] = np.cos(2 * np.pi * minute_of_day / 1440.0)
    out["day_of_week"] = t.dt.dayofweek
    out["sin_dw"] = np.sin(2 * np.pi * t.dt.dayofweek / 7.0)
    out["cos_dw"] = np.cos(2 * np.pi * t.dt.dayofweek / 7.0)
    return out

def add_lagged(df: pd.DataFrame, cols=("close","volume"), max_lag=5) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for col in cols:
        for L in range(1, max_lag+1):
            out[f"{col}_lag{L}"] = df[col].shift(L)
    return out

def add_optional_technicals(df: pd.DataFrame) -> pd.DataFrame:
    """Optionally include popular technicals. If 'ta' is not installed, we compute EMAs manually and skip the rest."""
    out = pd.DataFrame(index=df.index)
    c, v = df["close"], df["volume"]

    # EMAs always available
    for span in (5, 10, 20, 50, 200):
        out[f"ema_{span}"] = _ema(c, span)
        out[f"ema_slope_{span}"] = out[f"ema_{span}"].diff()

    if _HAS_TA:
        # RSI
        out["rsi14"] = ta.momentum.rsi(c, window=14)
        # Stoch RSI
        try:
            stoch_k = ta.momentum.stochrsi_k(c, window=14, smooth1=3, smooth2=3)
            stoch_d = ta.momentum.stochrsi_d(c, window=14, smooth1=3, smooth2=3)
            out["stoch_k"] = stoch_k
            out["stoch_d"] = stoch_d
            out["stoch_k_minus_d"] = stoch_k - stoch_d
        except Exception:
            pass
        # MACD
        try:
            macd = ta.trend.macd_diff(c, window_fast=12, window_slow=26, window_sign=9)
            out["macd_diff"] = macd
        except Exception:
            pass
        # Bollinger %b
        try:
            bb = ta.volatility.BollingerBands(close=c, window=20, window_dev=2)
            out["bb_perc_b"] = (c - bb.bollinger_lband()) / (bb.bollinger_hband() - bb.bollinger_lband() + 1e-12)
            out["bb_bw"] = _safe_div(bb.bollinger_hband() - bb.bollinger_lband(), bb.bollinger_mavg())
        except Exception:
            pass
        # VWAP deviation (rolling session approx: 1440 bars)
        w = min(1440, len(df))
        pv = (c * v).rolling(w, min_periods=1).sum()
        vv = v.rolling(w, min_periods=1).sum()
        vwap = pv / (vv + 1e-12)
        out["vwap_dev"] = _safe_div(c - vwap, vwap)
    else:
        # Minimal VWAP deviation without ta
        w = min(1440, len(df))
        pv = (c * v).rolling(w, min_periods=1).sum()
        vv = v.rolling(w, min_periods=1).sum()
        vwap = pv / (vv + 1e-12)
        out["vwap_dev"] = _safe_div(c - vwap, vwap)

    return out

# ----------------------------
# Public API
# ----------------------------
def build_features(
    df: pd.DataFrame,
    use_optional_technicals: bool = True,
    lag_max: int = 5,
    roll_windows = (3,5,15,30),
    time_col: str = "time"
) -> pd.DataFrame:
    """Return a feature matrix X aligned to df (index preserved).
    df must contain: time, open, high, low, close, volume.
    """
    required_cols = {"time","open","high","low","close","volume"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Ensure datetime
    # ensure datetime (tz-aware UTC) for the time column
    if not is_datetime64_any_dtype(df[time_col]):
        df = df.copy()
        df[time_col] = pd.to_datetime(df[time_col], utc=True)
    else:
        df = df.copy()
        try:
            if is_datetime64tz_dtype(df[time_col].dtype):
                df[time_col] = df[time_col].dt.tz_convert("UTC")
            else:
                df[time_col] = df[time_col].dt.tz_localize("UTC")
        except Exception:
            df[time_col] = pd.to_datetime(df[time_col], utc=True)


    feats = []
    feats.append(add_core_price_features(df))
    feats.append(add_candle_geometry(df))
    feats.append(add_rolling_stats(df, windows=roll_windows))
    feats.append(add_volatility_features(df))
    feats.append(add_time_features(df, time_col=time_col))
    feats.append(add_lagged(df, cols=("close","volume"), max_lag=lag_max))
    if use_optional_technicals:
        feats.append(add_optional_technicals(df))

    X = pd.concat(feats, axis=1)

    # Replace infs and drop rows that are too early for rolling stats
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.dropna(how="any")  # strict; you can opt for imputation in ML pipeline
    return X

def make_target(
    df: pd.DataFrame,
    horizon: int = 3,
    target_type: str = "classification",  # "classification" | "regression"
    thr_bp: float = 2.0
) -> pd.Series:
    """Build target labels from raw close prices.

    - horizon: number of minutes ahead for the label
    - target_type:
        * "classification": {-1, 0, +1} based on threshold in basis points
        * "regression": raw forward return over horizon (fraction)
    - thr_bp: threshold in basis points to define +1 / -1
    """
    c = df["close"].astype(float)
    fwd = c.shift(-horizon)
    ret = (fwd / c) - 1.0

    y = None
    if target_type == "regression":
        y = ret
    elif target_type == "classification":
        thr = thr_bp / 1e4
        y = pd.Series(0, index=df.index, dtype=int)
        y = y.where(ret.abs() < thr, other=np.where(ret > 0, 1, -1))
    else:
        raise ValueError("target_type must be 'classification' or 'regression'")

    # Align with feature dropna: trim NaNs at head/tail
    y = y.dropna()
    return y

# ----------------------------
# Convenience: aligned X, y
# ----------------------------
def build_dataset(
    df: pd.DataFrame,
    use_optional_technicals: bool = True,
    lag_max: int = 5,
    roll_windows = (3,5,15,30),
    time_col: str = "time",
    horizon: int = 3,
    target_type: str = "classification",
    thr_bp: float = 2.0
):
    X = build_features(df, use_optional_technicals=use_optional_technicals, lag_max=lag_max, roll_windows=roll_windows, time_col=time_col)
    y = make_target(df, horizon=horizon, target_type=target_type, thr_bp=thr_bp)
    # Align indices: keep intersection
    idx = X.index.intersection(y.index)
    return X.loc[idx], y.loc[idx]
