#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Swarm Phase-Transition Breakout — All-in-One Backtester (Debug Build)
====================================================================

What's inside:
- Chunked forward Coinbase fetcher (retries, rate-limit aware)
- Compression, Flow (robust hybrid + simple option), Cluster (VBP proxy)
- SwarmScore, ATR-buffered Donchian breakout
- Entry modes: close vs stop (intrabar)
- Full debug counters incl. raw intrabar breakout tags
- Loose mode & CLI overrides for quick tuning
"""

import argparse
import math
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import requests

# ---------- Utils ----------

def parse_date(s: str) -> datetime:
    try:
        return datetime.fromisoformat(s.replace('Z','+00:00'))
    except Exception:
        return datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=timezone.utc)

def to_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)

def pct_rank(series: np.ndarray) -> float:
    arr = np.asarray(series, dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size < 8:
        return np.nan
    last = arr[-1]
    less_eq = (arr <= last).sum()
    return float(less_eq) / float(arr.size)

def clamp01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))

def rolling_apply(series: pd.Series, window: int, func):
    return series.rolling(window=window, min_periods=window).apply(func, raw=True)

# ---------- Coinbase fetcher (chunked forward, retries) ----------

CB_BASE = "https://api.exchange.coinbase.com"

def _sleep_with_backoff(base_sec: float, attempt: int):
    import random
    time.sleep(base_sec * (2 ** attempt) + random.uniform(0, 0.2))

def fetch_candles(product_id: str,
                  start: datetime,
                  end: datetime,
                  granularity: int,
                  max_per_call: int = 300,
                  max_retries: int = 6,
                  polite_pause: float = 0.12) -> pd.DataFrame:
    """
    Chunked forward-paging downloader for Coinbase candles.
    - Steps from start->end in chunks of (max_per_call * granularity)
    - Retries on 429/5xx with exponential backoff
    - De-dupes and advances from last received candle (no gaps/overlaps)
    """
    start = to_utc(start)
    end = to_utc(end)
    if end <= start:
        raise ValueError("fetch_candles: end must be after start")

    span = timedelta(seconds=granularity * max_per_call)
    t1 = start
    frames: List[pd.DataFrame] = []

    while t1 < end:
        t2 = min(end, t1 + span)
        params = {"start": t1.isoformat(), "end": t2.isoformat(), "granularity": granularity}
        url = f"{CB_BASE}/products/{product_id}/candles"

        data = None
        for attempt in range(max_retries):
            try:
                r = requests.get(url, params=params, timeout=20)
                if r.status_code == 429:
                    ra = r.headers.get("Retry-After")
                    time.sleep(float(ra)) if ra else _sleep_with_backoff(0.5, attempt)
                    continue
                if 500 <= r.status_code < 600:
                    _sleep_with_backoff(0.5, attempt)
                    continue
                r.raise_for_status()
                data = r.json()
                break
            except Exception:
                _sleep_with_backoff(0.5, attempt)
                continue

        if data is None:
            t1 = t2 + timedelta(seconds=granularity)
            continue
        if not data:
            t1 = t2 + timedelta(seconds=granularity)
            time.sleep(polite_pause)
            continue

        df = pd.DataFrame(data, columns=["time","low","high","open","close","volume"])
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
        df = df.sort_values("time").set_index("time")
        frames.append(df[["open","high","low","close","volume"]])

        last_ts = df.index.max()
        t1 = (last_ts + timedelta(seconds=granularity)) if pd.notna(last_ts) else t2 + timedelta(seconds=granularity)
        time.sleep(polite_pause)

    if not frames:
        raise RuntimeError("No candles fetched (check symbol, dates, or connectivity).")

    out = pd.concat(frames).sort_index()
    out = out[~out.index.duplicated(keep="last")]
    out = out.loc[(out.index >= start) & (out.index <= end)]
    out = out.astype({"open": float, "high": float, "low": float, "close": float, "volume": float})
    return out

# ---------- Indicators ----------

def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    h, l, c = df["high"], df["low"], df["close"]
    prev_c = c.shift(1)
    tr = pd.concat([h - l, (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    return tr.rolling(period, min_periods=period).mean()

def bb_width_pct(df: pd.DataFrame, period: int = 20, stds: float = 2.0) -> pd.Series:
    mid = df["close"].rolling(period, min_periods=period).mean()
    sd = df["close"].rolling(period, min_periods=period).std(ddof=0)
    upper, lower = mid + stds*sd, mid - stds*sd
    width = (upper - lower).abs()
    return (width / mid.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0.0); down = -delta.clip(upper=0.0)
    roll_up = up.ewm(alpha=1/period, adjust=False).mean()
    roll_down = down.ewm(alpha=1/period, adjust=False).mean()
    rs = roll_up / roll_down.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def stoch_rsi(series: pd.Series, rsi_period: int = 14,
              stoch_period: int = 14, smooth_k: int = 3, smooth_d: int = 3):
    r = rsi(series, rsi_period)
    min_r = r.rolling(stoch_period, min_periods=stoch_period).min()
    max_r = r.rolling(stoch_period, min_periods=stoch_period).max()
    stoch = (r - min_r) / (max_r - min_r).replace(0, np.nan) * 100.0
    k = stoch.rolling(smooth_k, min_periods=smooth_k).mean()
    d = k.rolling(smooth_d, min_periods=smooth_d).mean()
    return k, d

def donchian(df: pd.DataFrame, period: int = 20):
    return df["high"].rolling(period, min_periods=period).max(), df["low"].rolling(period, min_periods=period).min()

# ---------- Feature Engineering ----------

@dataclass
class FeaturesConfig:
    compression_lookback_rank: int = 90
    clv_window_minutes: int = 10
    vbp_lookback_hours: int = 72
    cluster_min_pct: float = 0.0075
    cluster_max_pct: float = 0.03
    bin_width_pct: float = 0.002
    # Robust flow knobs
    clv_min_coverage: float = 0.5        # not used in per-15m version but kept for future
    flow_momentum_weight: float = 0.30   # 0..0.5 reasonable

def compute_compression_score(df15: pd.DataFrame, df1h: pd.DataFrame, cfg: FeaturesConfig) -> pd.Series:
    bbw1h = bb_width_pct(df1h).reindex(df15.index, method='ffill')
    bbw15 = bb_width_pct(df15)
    atrp15 = (atr(df15) / df15["close"]).replace([np.inf,-np.inf], np.nan)

    def rank_series(s: pd.Series) -> pd.Series:
        return rolling_apply(s, cfg.compression_lookback_rank, lambda a: pct_rank(a))

    r1 = rank_series(bbw1h)
    r2 = rank_series(bbw15)
    r3 = rank_series(atrp15)
    comp = 1.0 - pd.concat([r1, r2, r3], axis=1).mean(axis=1)
    return comp.clip(0,1)

def compute_clv_p(df1m: pd.DataFrame, window_min: int) -> pd.Series:
    """
    CLV proxy for buy fraction p in [0,1] over rolling window.
    Uses min_periods=1 so we emit values even with partial coverage.
    """
    high, low, close = df1m["high"], df1m["low"], df1m["close"]
    vol = df1m["volume"].fillna(0.0)
    rng = (high - low).replace(0, np.nan)
    clv = ((close - low) / rng).clip(0, 1).fillna(0.5)
    w = vol.replace(0, np.nan).fillna(1.0)
    num = (clv * w).rolling(window=window_min, min_periods=1).sum()
    den = w.rolling(window=window_min, min_periods=1).sum()
    return (num / den).reindex(df1m.index)

def compute_flow_p_hybrid(df1m: pd.DataFrame,
                          df15: pd.DataFrame,
                          cfg: FeaturesConfig) -> pd.Series:
    """
    Robust p aligned to 15m:
      - For each 15m end, use VW CLV over last clv_window_minutes of 1m.
      - Fallback to 15m CLV if window empty.
      - Blend momentum via normalized return sigmoid.
    """
    ends = df15.index
    win = int(cfg.clv_window_minutes)
    p_vals = []

    h15, l15, c15 = df15["high"], df15["low"], df15["close"]
    rng15 = (h15 - l15).replace(0, np.nan)
    p15_clv = ((c15 - l15) / rng15).clip(0, 1).fillna(0.5)

    for t in ends:
        start = t - pd.Timedelta(minutes=win)
        sl = df1m.loc[(df1m.index > start) & (df1m.index <= t)]
        if sl.empty:
            p_vals.append(float(p15_clv.loc[t]))
            continue
        rng = (sl["high"] - sl["low"]).replace(0, np.nan)
        clv = ((sl["close"] - sl["low"]) / rng).clip(0, 1).fillna(0.5)
        w = sl["volume"].fillna(0.0).replace(0, 1.0)
        p1m = float((clv * w).sum() / w.sum()) if w.sum() > 0 else 0.5
        p_vals.append(p1m)

    base_p = pd.Series(p_vals, index=ends).astype(float).clip(0, 1)

    atr15 = atr(df15, 14).reindex(ends)
    atr_pct = (atr15 / df15["close"]).replace(0, np.nan)
    ret = df15["close"].pct_change().reindex(ends).fillna(0.0)
    z = (ret / atr_pct).replace([np.inf, -np.inf], 0.0).fillna(0.0).clip(-3, 3)
    momentum_p = 1.0 / (1.0 + np.exp(-z))

    w_mom = float(getattr(cfg, "flow_momentum_weight", 0.30))
    p_final = (1.0 - w_mom) * base_p + w_mom * momentum_p
    return p_final.clip(0, 1)

def flow_metrics_from_p(p: pd.Series) -> Tuple[pd.Series, pd.Series]:
    """
    From p in [0,1] compute:
      - flow_dir ∈ {-1,0,1}
      - flow_strength ∈ [0,1] = (1 - H) * |p-0.5| * 2
    NaN-safe.
    """
    p = p.astype(float).clip(0, 1).fillna(0.5)
    eps = 1e-12
    H = -(p*np.log2(p+eps) + (1-p)*np.log2(1-p+eps))
    H = H.fillna(0.0).clip(0, 1)
    one_sided = 1.0 - H
    dir_vals = np.where(p > 0.5, 1, np.where(p < 0.5, -1, 0))
    dir_series = pd.Series(dir_vals, index=p.index).astype("int8")
    strength = (one_sided * (p - 0.5).abs() * 2.0).fillna(0.0).clip(0, 1)
    return dir_series, strength

def flow_metrics_simple(p: pd.Series) -> Tuple[pd.Series, pd.Series]:
    """Simpler flow strength: |p-0.5|*2 (no entropy)."""
    p = p.astype(float).clip(0, 1).fillna(0.5)
    dir_vals = np.where(p > 0.5, 1, np.where(p < 0.5, -1, 0))
    dir_series = pd.Series(dir_vals, index=p.index).astype("int8")
    strength = ((p - 0.5).abs() * 2.0).clip(0, 1)
    return dir_series, strength

def proximity_weight(dist_frac: float, cfg: FeaturesConfig) -> float:
    a, b = cfg.cluster_min_pct, cfg.cluster_max_pct
    if np.isnan(dist_frac) or dist_frac < a or dist_frac > b:
        return 0.0
    return max(0.0, min(1.0, 1.0 - (dist_frac - a)/(b - a)))

def compute_cluster_scores_vbp(df1m: pd.DataFrame, df15: pd.DataFrame, cfg: FeaturesConfig) -> Tuple[pd.Series, pd.Series]:
    """
    Approximate whale cluster scores using rolling Volume-by-Price (VBP) from 1m bars.
    Returns (score_long, score_short) aligned to df15 index. NaNs -> 0.0
    """
    out_long, out_short, idx15 = [], [], df15.index
    for t in idx15:
        price = float(df15.loc[t, "close"])
        if not np.isfinite(price) or price <= 0:
            out_long.append(np.nan); out_short.append(np.nan); continue
        start = t - timedelta(hours=cfg.vbp_lookback_hours)
        window = df1m.loc[(df1m.index > start) & (df1m.index <= t)]
        if window.empty:
            out_long.append(np.nan); out_short.append(np.nan); continue
        step = price * cfg.bin_width_pct
        bins = np.round(window["close"].values / step) * step
        vols = window["volume"].values
        vdict: Dict[float, float] = {}
        for b, v in zip(bins, vols):
            vdict[b] = vdict.get(b, 0.0) + (v if np.isfinite(v) else 0.0)
        if not vdict:
            out_long.append(np.nan); out_short.append(np.nan); continue
        prices, vvals = np.array(list(vdict.keys())), np.array(list(vdict.values()))
        dists = (prices - price) / price
        above_mask, below_mask = dists > 0, dists < 0

        def nearest(mask):
            if not mask.any(): return (np.nan, np.nan)
            cand_dists, cand_vols = np.abs(dists[mask]), vvals[mask]
            w = (cand_dists >= cfg.cluster_min_pct) & (cand_dists <= cfg.cluster_max_pct)
            if not w.any(): return (np.nan, np.nan)
            i = np.argmin(cand_dists[w])
            return float(cand_vols[w][i]), float(cand_dists[w][i])

        va, da = nearest(above_mask)
        vb, db = nearest(below_mask)

        def vol_rank(v):
            if not np.isfinite(v): return np.nan
            arr = vvals[np.isfinite(vvals)]
            if arr.size < 8: return np.nan
            return float((arr <= v).sum())/float(arr.size)

        dens_above, dens_below = vol_rank(va), vol_rank(vb)
        score_long = clamp01((dens_below if np.isfinite(dens_below) else 0.0) * proximity_weight(db, cfg))
        score_short = clamp01((dens_above if np.isfinite(dens_above) else 0.0) * proximity_weight(da, cfg))
        out_long.append(score_long); out_short.append(score_short)

    scores_long = pd.Series(out_long, index=idx15).fillna(0.0)
    scores_short = pd.Series(out_short, index=idx15).fillna(0.0)
    return scores_long, scores_short

# ---------- Strategy ----------

@dataclass
class StrategyConfig:
    swarm_thresh: float = 0.65
    min_compression: float = 0.60
    min_flow_strength: float = 0.50
    min_cluster_score: float = 0.40
    breakout_buffer_atr: float = 0.25
    donchian_period: int = 20
    atr_period: int = 14
    stoch_gate: bool = True
    cooldown_minutes: int = 60
    fast_fail_bars: int = 3
    fast_fail_flow_min: float = 0.25
    tp_atr: float = 2.0
    sl_buffer_atr: float = 0.25
    slippage_bps_in: float = 3.0
    slippage_bps_out: float = 3.0

def swarm_scores(comp: pd.Series, flow_dir: pd.Series, flow_strength: pd.Series,
                 cl_long: pd.Series, cl_short: pd.Series) -> Tuple[pd.Series, pd.Series]:
    long_score  = 0.40*comp + 0.35*flow_strength*(flow_dir == 1).astype(float) + 0.25*cl_long
    short_score = 0.40*comp + 0.35*flow_strength*(flow_dir == -1).astype(float) + 0.25*cl_short
    return long_score.clip(0,1), short_score.clip(0,1)

def stoch_gate_ok(k: float, d: float, side: str) -> bool:
    if np.isnan(k) or np.isnan(d): 
        return False
    if side == "long":
        return (k > d) and (k < 40.0)
    else:
        return (k < d) and (k > 60.0)

@dataclass
class Trade:
    side: str
    entry_time: pd.Timestamp
    entry_price: float
    sl: float
    tp: float
    exit_time: Optional[pd.Timestamp] = None
    exit_price: Optional[float] = None
    reason: Optional[str] = None

# ---------- Backtest Engine ----------

def backtest_symbol(symbol: str,
                    df1m: pd.DataFrame, df15: pd.DataFrame, df1h: pd.DataFrame,
                    fcfg: FeaturesConfig, scfg: StrategyConfig,
                    log_rows: List[Dict],
                    debug_gates: bool = False) -> Dict:
    df1m = df1m.sort_index()
    df15 = df15.sort_index()
    df1h = df1h.sort_index()

    # --- Features ---
    comp = compute_compression_score(df15, df1h, fcfg)

    p_15 = compute_flow_p_hybrid(df1m, df15, fcfg).fillna(0.5)
    if globals().get("__FLOW_SIMPLE__", False):
        flow_dir, flow_strength = flow_metrics_simple(p_15)
    else:
        flow_dir, flow_strength = flow_metrics_from_p(p_15)

    cl_long, cl_short = compute_cluster_scores_vbp(df1m, df15, fcfg)
    long_score, short_score = swarm_scores(comp, flow_dir, flow_strength, cl_long, cl_short)
    box_hi, box_lo = donchian(df15, scfg.donchian_period)
    box_hi = box_hi.shift(1)
    box_lo = box_lo.shift(1)
    atr15 = atr(df15, scfg.atr_period)
    k, d = stoch_rsi(df15["close"])

    # Debug counters
    gate_counts = {
        "bars": 0,
        "compression": 0,
        "flow": 0,
        "cluster_long": 0,
        "cluster_short": 0,
        "swarm_long": 0,
        "swarm_short": 0,
        "stoch_long": 0,
        "stoch_short": 0,
        "breakout_long": 0,
        "breakout_short": 0,
        # raw (ungated) intrabar tags
        "raw_break_hi": 0,
        "raw_break_lo": 0,
    }
    gate_counts.update({
    "pre_long": 0,     # all long gates except breakout
    "pre_short": 0,    # all short gates except breakout
    "overlap_long": 0, # pre_long && breakout_long
    "overlap_short": 0 # pre_short && breakout_short
    })

    open_trade: Optional[Trade] = None
    cooldown_until: Optional[pd.Timestamp] = None

    total_r = 0.0
    max_dd = 0.0
    equity = 0.0
    equity_peak = 0.0

    for i in range(len(df15)-1):
        t = df15.index[i]
        nxt = df15.index[i+1]
        row = df15.iloc[i]
        close_t = float(row["close"])
        high_t = float(row["high"])
        low_t  = float(row["low"])
        high_next = float(df15.iloc[i+1]["high"])
        low_next  = float(df15.iloc[i+1]["low"])

        # ---------------- Manage open trade ----------------
        if open_trade is not None:
            side = open_trade.side
            entry, sl, tp = open_trade.entry_price, open_trade.sl, open_trade.tp
            r = None
            reason = None

            if side == "long":
                if low_next <= sl:
                    exit_price = sl * (1 - scfg.slippage_bps_out/10000.0)
                    r = (exit_price - entry) / (tp - entry); reason = "SL"
                elif high_next >= tp:
                    exit_price = tp * (1 - scfg.slippage_bps_out/10000.0)
                    r = 1.0; reason = "TP"
                else:
                    if (nxt - open_trade.entry_time) <= pd.Timedelta(minutes=15*scfg.fast_fail_bars):
                        fs = float(flow_strength.loc[t]) if t in flow_strength.index else np.nan
                        kk = float(k.loc[t]) if t in k.index else np.nan
                        ddv = float(d.loc[t]) if t in d.index else np.nan
                        if (fs < scfg.fast_fail_flow_min) or (scfg.stoch_gate and not stoch_gate_ok(kk, ddv, "long")):
                            exit_price = close_t * (1 - scfg.slippage_bps_out/10000.0)
                            r = (exit_price - entry) / (tp - entry); reason = "FastFail"

            else:  # short
                if high_next >= sl:
                    exit_price = sl * (1 + scfg.slippage_bps_out/10000.0)
                    r = (entry - exit_price) / (entry - tp); reason = "SL"
                elif low_next <= tp:
                    exit_price = tp * (1 + scfg.slippage_bps_out/10000.0)
                    r = 1.0; reason = "TP"
                else:
                    if (nxt - open_trade.entry_time) <= pd.Timedelta(minutes=15*scfg.fast_fail_bars):
                        fs = float(flow_strength.loc[t]) if t in flow_strength.index else np.nan
                        kk = float(k.loc[t]) if t in k.index else np.nan
                        ddv = float(d.loc[t]) if t in d.index else np.nan
                        if (fs < scfg.fast_fail_flow_min) or (scfg.stoch_gate and not stoch_gate_ok(kk, ddv, "short")):
                            exit_price = close_t * (1 + scfg.slippage_bps_out/10000.0)
                            r = (entry - exit_price) / (entry - tp); reason = "FastFail"

            if r is not None:
                total_r += r
                equity += r
                equity_peak = max(equity_peak, equity)
                max_dd = min(max_dd, equity - equity_peak)
                tr = open_trade
                log_rows.append({
                    "symbol": symbol,
                    "side": tr.side,
                    "entry_time": tr.entry_time.isoformat(),
                    "entry_price": tr.entry_price,
                    "sl": tr.sl,
                    "tp": tr.tp,
                    "exit_time": nxt.isoformat(),
                    "exit_price": exit_price,
                    "reason": reason,
                    "R": r,
                })
                open_trade = None
                cooldown_until = t + pd.Timedelta(minutes=scfg.cooldown_minutes)
                continue

        # ---------------- Entry checks ----------------
        gate_counts["bars"] += 1
        comp_t  = float(comp.loc[t]) if t in comp.index else np.nan
        fs_t    = float(flow_strength.loc[t]) if t in flow_strength.index else np.nan
        lsc     = float(long_score.loc[t]) if t in long_score.index else np.nan
        ssc     = float(short_score.loc[t]) if t in short_score.index else np.nan
        cls_long_t = float(cl_long.loc[t]) if t in cl_long.index else np.nan
        cls_short_t= float(cl_short.loc[t]) if t in cl_short.index else np.nan
        boxhi   = float(box_hi.loc[t]) if t in box_hi.index else np.nan
        boxlo   = float(box_lo.loc[t]) if t in box_lo.index else np.nan
        atrv    = float(atr15.loc[t]) if t in atr15.index else np.nan
        kk      = float(k.loc[t]) if t in k.index else np.nan
        ddv     = float(d.loc[t]) if t in d.index else np.nan

        # Raw (ungated) intrabar breakout tags
        if np.isfinite(boxhi) and np.isfinite(atrv):
            if high_t > boxhi + scfg.breakout_buffer_atr*atrv:
                gate_counts["raw_break_hi"] += 1
        if np.isfinite(boxlo) and np.isfinite(atrv):
            if low_t < boxlo - scfg.breakout_buffer_atr*atrv:
                gate_counts["raw_break_lo"] += 1

        # Debug counters for gated conditions
        if np.isfinite(comp_t) and comp_t >= scfg.min_compression: gate_counts["compression"] += 1
        if np.isfinite(fs_t) and fs_t >= scfg.min_flow_strength: gate_counts["flow"] += 1
        if np.isfinite(cls_long_t) and cls_long_t >= scfg.min_cluster_score: gate_counts["cluster_long"] += 1
        if np.isfinite(cls_short_t) and cls_short_t >= scfg.min_cluster_score: gate_counts["cluster_short"] += 1
        if np.isfinite(lsc) and lsc >= scfg.swarm_thresh: gate_counts["swarm_long"] += 1
        if np.isfinite(ssc) and ssc >= scfg.swarm_thresh: gate_counts["swarm_short"] += 1
        if (not scfg.stoch_gate) or (np.isfinite(kk) and np.isfinite(ddv) and (kk > ddv) and (kk < 40.0)):
            gate_counts["stoch_long"] += 1
        if (not scfg.stoch_gate) or (np.isfinite(kk) and np.isfinite(ddv) and (kk < ddv) and (kk > 60.0)):
            gate_counts["stoch_short"] += 1

        # Breakout counters reflect selected entry mode
        entry_mode = globals().get("__ENTRY_MODE__", "close")
        # --- “pre” (all gates except breakout), then compute overlap with breakout
        pre_long_ok = (
            np.isfinite(lsc) and lsc >= scfg.swarm_thresh and
            np.isfinite(comp_t) and comp_t >= scfg.min_compression and
            np.isfinite(fs_t) and fs_t >= scfg.min_flow_strength and
            np.isfinite(cls_long_t) and cls_long_t >= scfg.min_cluster_score and
            (not scfg.stoch_gate or stoch_gate_ok(kk, ddv, "long")) and
            np.isfinite(boxhi) and np.isfinite(atrv)
        )
        pre_short_ok = (
            np.isfinite(ssc) and ssc >= scfg.swarm_thresh and
            np.isfinite(comp_t) and comp_t >= scfg.min_compression and
            np.isfinite(fs_t) and fs_t >= scfg.min_flow_strength and
            np.isfinite(cls_short_t) and cls_short_t >= scfg.min_cluster_score and
            (not scfg.stoch_gate or stoch_gate_ok(kk, ddv, "short")) and
            np.isfinite(boxlo) and np.isfinite(atrv)
        )
        if pre_long_ok:  gate_counts["pre_long"]  += 1
        if pre_short_ok: gate_counts["pre_short"] += 1

        break_long_ok  = (close_t > (boxhi + scfg.breakout_buffer_atr*atrv)) if entry_mode=="close" else (high_t > (boxhi + scfg.breakout_buffer_atr*atrv))
        break_short_ok = (close_t < (boxlo - scfg.breakout_buffer_atr*atrv)) if entry_mode=="close" else (low_t  < (boxlo - scfg.breakout_buffer_atr*atrv))

        if pre_long_ok  and break_long_ok:  gate_counts["overlap_long"]  += 1
        if pre_short_ok and break_short_ok: gate_counts["overlap_short"] += 1

        if entry_mode == "close":
            if np.isfinite(boxhi) and np.isfinite(atrv) and close_t > boxhi + scfg.breakout_buffer_atr*atrv:
                gate_counts["breakout_long"] += 1
            if np.isfinite(boxlo) and np.isfinite(atrv) and close_t < boxlo - scfg.breakout_buffer_atr*atrv:
                gate_counts["breakout_short"] += 1
        else:
            if np.isfinite(boxhi) and np.isfinite(atrv) and high_t > boxhi + scfg.breakout_buffer_atr*atrv:
                gate_counts["breakout_long"] += 1
            if np.isfinite(boxlo) and np.isfinite(atrv) and low_t < boxlo - scfg.breakout_buffer_atr*atrv:
                gate_counts["breakout_short"] += 1

        # Gated entries
        if open_trade is None and (cooldown_until is None or t >= cooldown_until):

            # LONG entry
            if (np.isfinite(lsc) and lsc >= scfg.swarm_thresh and
                np.isfinite(comp_t) and comp_t >= scfg.min_compression and
                np.isfinite(fs_t) and fs_t >= scfg.min_flow_strength and
                np.isfinite(cls_long_t) and cls_long_t >= scfg.min_cluster_score and
                np.isfinite(boxhi) and np.isfinite(atrv) and
                (not scfg.stoch_gate or stoch_gate_ok(kk, ddv, "long"))):

                breakout = boxhi + scfg.breakout_buffer_atr * atrv
                trig_long = (close_t > breakout) if entry_mode == "close" else (high_t > breakout)

                if trig_long:
                    entry = (close_t if entry_mode == "close" else breakout) * (1 + scfg.slippage_bps_in/10000.0)
                    sl = (boxlo - scfg.sl_buffer_atr * atrv)
                    tp = entry + scfg.tp_atr * atrv
                    open_trade = Trade(side="long", entry_time=t, entry_price=entry, sl=sl, tp=tp)

            # SHORT entry
            if open_trade is None and (np.isfinite(ssc) and ssc >= scfg.swarm_thresh and
                np.isfinite(comp_t) and comp_t >= scfg.min_compression and
                np.isfinite(fs_t) and fs_t >= scfg.min_flow_strength and
                np.isfinite(cls_short_t) and cls_short_t >= scfg.min_cluster_score and
                np.isfinite(boxlo) and np.isfinite(atrv) and
                (not scfg.stoch_gate or stoch_gate_ok(kk, ddv, "short"))):

                breakout = boxlo - scfg.breakout_buffer_atr * atrv
                trig_short = (close_t < breakout) if entry_mode == "close" else (low_t < breakout)

                if trig_short:
                    entry = (close_t if entry_mode == "close" else breakout) * (1 - scfg.slippage_bps_in/10000.0)
                    sl = (boxhi + scfg.sl_buffer_atr * atrv)
                    tp = entry - scfg.tp_atr * atrv
                    open_trade = Trade(side="short", entry_time=t, entry_price=entry, sl=sl, tp=tp)

    # --- Debug summary ---
    if debug_gates:
        tot = max(gate_counts["bars"], 1)
        print(f"\n[DEBUG {symbol}] Gate passes over {tot} x 15m bars:")
        for k in ["compression","flow","cluster_long","cluster_short","swarm_long","swarm_short",
                  "stoch_long","stoch_short","breakout_long","breakout_short","raw_break_hi","raw_break_lo"]:
            v = gate_counts[k]; pct = (100.0 * v / tot)
            print(f"  {k:14s}: {v:5d} ({pct:5.1f}%)")

    trades_df = pd.DataFrame(log_rows)
    n = len(trades_df)
    winrate = (trades_df["R"] > 0).mean() if n else 0.0
    avgR = trades_df["R"].mean() if n else 0.0
    stdR = trades_df["R"].std(ddof=0) if n else 0.0
    sharpe_like = (avgR / (stdR + 1e-9)) * math.sqrt(max(n,1))
    totalR = trades_df["R"].sum() if n else 0.0
    for k in ["compression","flow","cluster_long","cluster_short","swarm_long","swarm_short",
          "stoch_long","stoch_short","breakout_long","breakout_short",
          "raw_break_hi","raw_break_lo","pre_long","pre_short","overlap_long","overlap_short"]:
        v = gate_counts[k]; pct = (100.0 * v / tot)
        print(f"  {k:14s}: {v:5d} ({pct:5.1f}%)")

    return {
        "symbol": symbol,
        "trades": n,
        "winrate": winrate,
        "avgR": avgR,
        "sharpe_like": sharpe_like,
        "totalR": totalR,
        "max_drawdown_R": max_dd,
    }
    

# ---------- Data Loading Helpers ----------

def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "time" not in df.columns:
        raise ValueError(f"{path} must have a 'time' column")
    try:
        idx = pd.to_datetime(df["time"], utc=True)
    except Exception:
        idx = pd.to_datetime(df["time"], unit="s", utc=True)
    df = df.set_index(idx)
    cols = ["open","high","low","close","volume"]
    for c in cols:
        if c not in df.columns:
            raise ValueError(f"{path} missing column '{c}'")
    out = df[cols].astype(float).sort_index()
    return out

# ---------- Grid Search ----------

def grid_search(symbols: List[str],
                data: Dict[str, Dict[str, pd.DataFrame]],
                fcfg: FeaturesConfig,
                base_scfg: StrategyConfig,
                thresh_list: List[float],
                buffer_list: List[float],
                debug_gates: bool = False) -> pd.DataFrame:
    rows = []
    for th in thresh_list:
        for buf in buffer_list:
            scfg = StrategyConfig(**{**base_scfg.__dict__, "swarm_thresh": th, "breakout_buffer_atr": buf})
            log_rows: List[Dict] = []
            agg = []
            for sym in symbols:
                res = backtest_symbol(sym, data[sym]["1m"], data[sym]["15m"], data[sym]["1h"], fcfg, scfg, log_rows, debug_gates=debug_gates)
                agg.append(res)
            dfres = pd.DataFrame(agg)
            rows.append({
                "swarm_thresh": th,
                "breakout_buffer_atr": buf,
                "trades": int(dfres["trades"].sum()),
                "winrate": float(np.average(dfres["winrate"], weights=dfres["trades"].clip(lower=1))),
                "avgR": float(np.average(dfres["avgR"], weights=dfres["trades"].clip(lower=1))),
                "totalR": float(dfres["totalR"].sum()),
                "sharpe_like": float(np.average(dfres["sharpe_like"], weights=dfres["trades"].clip(lower=1))),
            })
    return pd.DataFrame(rows).sort_values(["sharpe_like","totalR"], ascending=[False, False])

# ---------- Main ----------

def main():
    ap = argparse.ArgumentParser(description="Swarm Phase-Transition Breakout — Backtester (Debug Build)")
    ap.add_argument("--symbols", nargs="+", required=True, help="e.g., BTC-USD ETH-USD XRP-USD")
    ap.add_argument("--start", type=str, required=True, help="Start date (YYYY-MM-DD or ISO8601)")
    ap.add_argument("--end", type=str, required=True, help="End date (YYYY-MM-DD or ISO8601)")
    ap.add_argument("--fetch", action="store_true", help="Fetch candles from Coinbase public REST")
    ap.add_argument("--data-dir", type=str, default=None, help="Directory containing <symbol>_1m.csv, _15m.csv, _1h.csv")
    ap.add_argument("--outdir", type=str, default="./out", help="Output directory for logs")
    ap.add_argument("--grid", action="store_true", help="Run a small grid search on thresholds")
    ap.add_argument("--no-stoch-gate", action="store_true", help="Disable stochastic RSI gate")
    ap.add_argument("--cooldown", type=int, default=60, help="Cooldown minutes between entries per symbol")

    # Diagnostics & behavior
    ap.add_argument("--debug-gates", action="store_true", help="Print gate counters")
    ap.add_argument("--loose", action="store_true", help="Loosen gates (diagnostic)")
    ap.add_argument("--flow-simple", action="store_true", help="Use simple flow strength = |p-0.5|*2 (no entropy)")
    ap.add_argument("--entry-mode", choices=["close","stop"], default="close",
                    help="Entry trigger mode: 'close' waits for close; 'stop' triggers intrabar")

    # Overrides
    ap.add_argument("--thresh", type=float, default=None, help="Swarm threshold override (e.g., 0.60)")
    ap.add_argument("--buffer", type=float, default=None, help="Breakout buffer ATR (e.g., 0.25)")
    ap.add_argument("--min-comp", type=float, default=None, help="Min CompressionScore (e.g., 0.55)")
    ap.add_argument("--min-flow", type=float, default=None, help="Min FlowStrength (e.g., 0.40)")
    ap.add_argument("--min-cluster", type=float, default=None, help="Min ClusterScore (e.g., 0.30)")
    ap.add_argument("--donchian", type=int, default=None, help="Donchian lookback (e.g., 14)")
    args = ap.parse_args()

    # Echo config
    print(f"[CONFIG] entry_mode={args.entry_mode} flow_simple={args.flow_simple} loose={args.loose}", flush=True)

    # Globals visible to backtest function
    globals()["__ENTRY_MODE__"] = args.entry_mode
    globals()["__FLOW_SIMPLE__"] = args.flow_simple

    start = parse_date(args.start)
    end = parse_date(args.end)

    # Load data per symbol
    data: Dict[str, Dict[str, pd.DataFrame]] = {}
    for sym in args.symbols:
        if args.fetch:
            print(f"[{sym}] Fetching 1m/15m/1h candles from Coinbase...", flush=True)
            df1m = fetch_candles(sym, start, end, 60)
            df15 = fetch_candles(sym, start, end, 900)
            df1h = fetch_candles(sym, start, end, 3600)
        elif args.data_dir:
            import os
            p1 = os.path.join(args.data_dir, f"{sym}_1m.csv")
            p2 = os.path.join(args.data_dir, f"{sym}_15m.csv")
            p3 = os.path.join(args.data_dir, f"{sym}_1h.csv")
            df1m = load_csv(p1)
            df15 = load_csv(p2)
            df1h = load_csv(p3)
            df1m = df1m.loc[(df1m.index >= start) & (df1m.index <= end)]
            df15 = df15.loc[(df15.index >= start) & (df15.index <= end)]
            df1h = df1h.loc[(df1h.index >= start) & (df1h.index <= end)]
        else:
            print("ERROR: Provide --fetch or --data-dir", file=sys.stderr)
            sys.exit(1)

        if df1m.empty or df15.empty or df1h.empty:
            print(f"ERROR: Missing data for {sym}", file=sys.stderr); sys.exit(2)

        data[sym] = {"1m": df1m, "15m": df15, "1h": df1h}

    # Feature & Strategy configs
    fcfg = FeaturesConfig()
    scfg = StrategyConfig(cooldown_minutes=args.cooldown, stoch_gate=not args.no_stoch_gate)

    # CLI overrides
    if args.thresh is not None: scfg.swarm_thresh = args.thresh
    if args.buffer is not None: scfg.breakout_buffer_atr = args.buffer
    if args.min_comp is not None: scfg.min_compression = args.min_comp
    if args.min_flow is not None: scfg.min_flow_strength = args.min_flow
    if args.min_cluster is not None: scfg.min_cluster_score = args.min_cluster
    if args.donchian is not None: scfg.donchian_period = args.donchian

    # Loose mode (diagnostic)
    if args.loose:
        scfg.swarm_thresh = min(scfg.swarm_thresh, 0.60)
        scfg.min_compression = min(scfg.min_compression, 0.55)
        scfg.min_flow_strength = min(scfg.min_flow_strength, 0.40)
        scfg.min_cluster_score = min(scfg.min_cluster_score, 0.30)
        scfg.breakout_buffer_atr = min(scfg.breakout_buffer_atr, 0.25)
        scfg.donchian_period = min(scfg.donchian_period, 14)

    # Prepare outputs
    import os
    os.makedirs(args.outdir, exist_ok=True)
    trades_csv = os.path.join(args.outdir, "swarm_trades.csv")
    grid_csv = os.path.join(args.outdir, "swarm_grid.csv")

    # Optional grid search
    if args.grid:
        print("Running grid search...", flush=True)
        thresh_list = [0.60, 0.65, 0.70, 0.75]
        buffer_list = [0.25, 0.30, 0.35, 0.40]
        gdf = grid_search(args.symbols, data, fcfg, scfg, thresh_list, buffer_list, debug_gates=args.debug_gates)
        gdf.to_csv(grid_csv, index=False)
        print(f"Grid results -> {grid_csv}")
        best = gdf.iloc[0].to_dict()
        scfg.swarm_thresh = float(best["swarm_thresh"])
        scfg.breakout_buffer_atr = float(best["breakout_buffer_atr"])
        print(f"Using best params: thresh={scfg.swarm_thresh:.2f}, buffer={scfg.breakout_buffer_atr:.2f}")

    # Run backtest for each symbol
    all_logs: List[Dict] = []
    summaries = []
    for sym in args.symbols:
        print(f"Backtesting {sym}...", flush=True)
        s = backtest_symbol(sym, data[sym]["1m"], data[sym]["15m"], data[sym]["1h"], fcfg, scfg, all_logs, debug_gates=args.debug_gates)
        summaries.append(s)

    # Save trades
    if all_logs:
        pd.DataFrame(all_logs).to_csv(trades_csv, index=False)
        print(f"Trades log -> {trades_csv}")
    else:
        print("No trades logged for the given parameters/timeframe.")

    # Summary
    sdf = pd.DataFrame(summaries)
    if not sdf.empty:
        print("\n=== SUMMARY ===")
        print(sdf.to_string(index=False))
        print("\nTotals:")
        print(f"  Trades: {int(sdf['trades'].sum())}")
        print(f"  Total R: {float(sdf['totalR'].sum()):.2f}")
        wavg_wr = float(np.average(sdf['winrate'], weights=sdf['trades'].clip(lower=1)))
        print(f"  Weighted WinRate: {100*wavg_wr:.1f}%")
        wavg_sh = float(np.average(sdf['sharpe_like'], weights=sdf['trades'].clip(lower=1)))
        print(f"  Weighted Sharpe-like: {wavg_sh:.2f}")
    else:
        print("No summary to show.")

# ---------- Run guard ----------
if __name__ == "__main__":
    main()
