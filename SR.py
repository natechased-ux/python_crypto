from __future__ import annotations

# key_sr_confluence_scanner_coinbase_all.py
# ------------------------------------------------------------
# Daily Key Support/Resistance + Confluence Scanner (Coinbase)
# ------------------------------------------------------------
# Builds zones from:
# - Swing S/R (prominent highs/lows clustered)
# - Range / touch clustering (multi-touch price bins)
# - Fibonacci retracements from latest major swing
# - Weekly pivots (P, S1/S2, R1/R2)
# - EMA-50 / EMA-200 dynamic S/R
# - Anchored VWAP from latest major swing anchor
# - Volume-by-Price HVNs (approx via close-price distribution)
#
# Scores each symbol by proximity + confluence overlap and prints a ranked table.
#
# Deps: pandas, numpy, requests
# ------------------------------------------------------------

import time
import math
import requests
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Dict, Tuple
from datetime import datetime, timedelta, timezone

# ==============================
# Config
# ==============================

@dataclass
class Config:
    # Data
    lookback_days: int = 400            # ~2 years of daily candles
    request_timeout: int = 20
    # Highlighting for entry status
    entry_now_near_pct: float = 0.003   # 0.3% from entry band counts as "near"


    # Swings
    swing_window: int = 3               # local extrema lookback on each side
    min_retrace_pct: float = 0.03       # prominence: must move away at least 3%

    # Clustering / zone sizing
    cluster_tolerance_pct: float = 0.004   # 0.40% of price
    cluster_tolerance_atr_mult: float = 0.35
    zone_halfwidth_atr_mult: float = 0.60
    zone_halfwidth_min_pct: float = 0.003  # 0.30% of price minimum

    # Range/touch clustering
    touch_bin_pct: float = 0.004        # bin width ~0.4% of price
    touch_near_pct: float = 0.0025      # count a "touch" if within 0.25% of bin center
    touch_min_count: int = 3            # require >=3 touches to form a zone

    # Fibonacci / swing anchor
    major_swing_thresh: float = 0.08    # 8% move defines latest major swing
    fib_ratios: Tuple[float, ...] = (0.236, 0.382, 0.5, 0.618, 0.786, 0.886)
    min_rr: float = 1.5               # minimum R:R to consider tradable
    show_only_tradable: bool = True

    # Volume-by-Price HVN
    vbp_bins: int = 30
    vbp_topn: int = 3

    # EMAs
    ema_fast: int = 50
    ema_slow: int = 200

    # Scoring
    proximity_in_zone_score: float = 6.0
    proximity_falloff_pct: float = 0.006     # add'l radius for partial credit
    proximity_falloff_points: float = 4.0
    confluence_overlap_pct: float = 0.0035   # zones whose centers within 0.35% of price
    confluence_per_overlap: float = 1.25
    confluence_cap: float = 5.0
    # Show only zones that are close enough to trade
    max_zone_distance_pct: float = 0.08   # 8% max distance from price to print (tune 0.03–0.10)
    keep_nearest_if_empty: bool = True    # if nothing passes the filter, keep the nearest S and R anyway

# Recent-regime bias for building “touch” & VBP zones
    recent_days_for_touch_vbp: int = 180  # only use last ~6 months for touch bins / HVNs
    recent_days_for_swings: int = 365     # (optional) limit swings to last ~1 year
    

    # Source weights (tweak to taste)
    weights: Dict[str, float] = None
    tight_cluster_pct: float = 0.006      # merge levels only if centers within ~0.6% of price
    halfwidth_pct_min: float = 0.0025     # zone halfwidth floor  = 0.25% of price
    halfwidth_pct_max: float = 0.0075     # zone halfwidth ceiling= 0.75% of price
    halfwidth_atr_mult: float = 0.20      # use a *small* ATR component (tight)
    stop_buffer_pct: float = 0.0025       # 0.25% beyond the zone for stops
    entry_padding_pct: float = 0.0005     # 0.05% inside the zone for entries



    # ATR
    atr_length: int = 14

CFG = Config()
if CFG.weights is None:
    CFG.weights = {
        "swing": 1.00,
        "touch": 1.00,
        "fib":   0.90,
        "pivot": 0.80,
        "ema":   0.70,
        "avwap": 0.80,
        "vbp":   0.90,
    }

@dataclass
class TradePlan:
    side: str
    entry_low: float
    entry_high: float
    stop: float
    target: float
    rr: float
    note: str = ""
    tradable: bool = False
    why: str = ""                      # reason for skip if not tradable


from typing import Optional, List

def nearest_opposite_cluster(clusters: List["ZoneCluster"], price: float, want: str) -> Optional["ZoneCluster"]:
    """
    Return nearest opposite cluster relative to price.
    want: "RESISTANCE" (for long targets) or "SUPPORT" (for short targets)
    """
    cands = []
    for c in clusters:
        if c.label_sr != want:
            continue
        if want == "RESISTANCE" and c.low >= price:
            cands.append((abs(c.low - price), c))
        elif want == "SUPPORT" and c.high <= price:
            cands.append((abs(price - c.high), c))
    if not cands:
        return None
    return sorted(cands, key=lambda x: x[0])[0][1]

def entry_status(price: float, entry_low: float, entry_high: float, near_pct: float) -> str:
    if entry_low <= price <= entry_high:
        return "NOW"           # inside the entry band
    band_w = max(entry_high - entry_low, 1e-12)
    near_abs = max(near_pct * price, 0.25 * band_w)  # allow tiny bands
    if (entry_low - near_abs) <= price <= (entry_high + near_abs):
        return "NEAR"          # within small tolerance
    return "WAIT"

def build_trade_plans(price: float, clusters: List["ZoneCluster"], cfg: "Config") -> List["TradePlan"]:
    """
    Build LONG/SHORT plans from the strongest nearby cluster.
    - Entry range is a tight slice *inside* the cluster (side-specific).
    - Stop sits just outside the cluster edge with buffer.
    - Target uses nearest opposite cluster; otherwise a structural fallback.
    - Plans are tagged tradable if R:R >= cfg.min_rr.
    """
    plans: List["TradePlan"] = []
    if not clusters:
        return plans

    top = clusters[0]  # strongest nearby cluster
    width = max(1e-12, top.high - top.low)
    pad   = cfg.entry_padding_pct * price  # tiny pad inside the band

    # Helper to finalize/append a plan with tradability tagging
    def _finalize(side: str, entry_low: float, entry_high: float, stop: float, target: float, note: str):
        # Conservative risk/reward: assume worst fill for risk, best for reward
        if side == "LONG":
            risk   = entry_high - stop
            reward = max(0.0, target - entry_low)
        else:  # SHORT
            risk   = stop - entry_low
            reward = max(0.0, entry_high - target)

        rr = (reward / risk) if risk > 0 else 0.0
        tp = TradePlan(side, entry_low, entry_high, stop, target, rr, note=note)
        tp.tradable = rr >= cfg.min_rr
        tp.why = "" if tp.tradable else f"R:R {rr:.2f} < {cfg.min_rr}"
        plans.append(tp)

    # LONG plan: cluster acts as SUPPORT or neutral (AT PRICE)
    if top.label_sr in ("SUPPORT", "AT PRICE"):
        # Use inner slice near the *bottom* of the cluster for longs
        inner_frac = 0.35  # ~35% of cluster width as entry band
        entry_low  = top.low  + pad
        entry_high = min(top.high - pad, top.low + inner_frac * width)

        # Ensure a non-degenerate band
        if entry_high - entry_low < max(1e-7, 0.0001 * price):
            mid = (top.low + top.high) / 2.0
            entry_low  = max(top.low + pad, mid - 0.0005 * price)
            entry_high = min(top.high - pad, mid + 0.0005 * price)

        stop = top.low - cfg.stop_buffer_pct * price

        # Target: nearest RESISTANCE above; otherwise structural fallback
        res = nearest_opposite_cluster(clusters, price, "RESISTANCE")
        if res is not None:
            target = max(res.low, entry_high + 2.0 * (entry_high - entry_low))  # ensure some payoff
            note = "Target = nearest resistance cluster."
        else:
            target = entry_high + max(2.0 * width, 3.0 * (entry_high - entry_low))
            note = "No clear resistance; using structural target."

        # Optional micro-tighten entry toward the lower edge to improve R:R
        tighten = 0.25
        entry_high = entry_low + (1.0 - tighten) * (entry_high - entry_low)

        _finalize("LONG", entry_low, entry_high, stop, target, note)

    # SHORT plan: cluster acts as RESISTANCE or neutral (AT PRICE)
    if top.label_sr in ("RESISTANCE", "AT PRICE"):
        # Use inner slice near the *top* of the cluster for shorts
        inner_frac = 0.35
        entry_low  = max(top.low + pad, top.high - inner_frac * width)
        entry_high = top.high - pad

        if entry_high - entry_low < max(1e-7, 0.0001 * price):
            mid = (top.low + top.high) / 2.0
            entry_low  = max(top.low + pad,  mid - 0.0005 * price)
            entry_high = min(top.high - pad, mid + 0.0005 * price)

        stop = top.high + cfg.stop_buffer_pct * price

        # Target: nearest SUPPORT below; otherwise structural fallback
        sup = nearest_opposite_cluster(clusters, price, "SUPPORT")
        if sup is not None:
            target = min(sup.high, entry_low - 2.0 * (entry_high - entry_low))
            note = "Target = nearest support cluster."
        else:
            target = entry_low - max(2.0 * width, 3.0 * (entry_high - entry_low))
            note = "No clear support; using structural target."

        # Optional micro-tighten entry: pull lower bound up a bit to improve R:R
        tighten = 0.25
        entry_low = entry_high - (1.0 - tighten) * (entry_high - entry_low)

        _finalize("SHORT", entry_low, entry_high, stop, target, note)

    return plans


# ==============================
# Data fetcher (Coinbase)
# ==============================
def tight_halfwidth(price_now: float, atr_abs: float, cfg: Config) -> float:
    # Base on small ATR + clamp to [min%, max%] of price
    hw = cfg.halfwidth_atr_mult * atr_abs
    hw = max(hw, cfg.halfwidth_pct_min * price_now)
    hw = min(hw, cfg.halfwidth_pct_max * price_now)
    return hw



def fetch_coinbase_ohlcv_1d(symbol: str, days: int = 730, timeout: int = 20) -> pd.DataFrame:
    """
    Coinbase daily OHLCV with paging (handles 300-candle limit).
    Returns ascending timestamp DataFrame with:
    ['timestamp','open','high','low','close','volume']
    """
    import time as _time
    import requests as _req
    import pandas as _pd
    from datetime import datetime, timezone, timedelta

    GRAN = 86400
    MAX_CANDLES = 300  # Coinbase per-request cap
    headers = {"User-Agent": "key-sr-scanner/1.1"}

    # end = now (UTC midnight) to keep on day boundaries
    now = datetime.now(timezone.utc)
    end_dt = datetime(now.year, now.month, now.day, tzinfo=timezone.utc)
    start_dt = end_dt - timedelta(days=days)

    out = []
    cursor_start = start_dt

    while cursor_start < end_dt:
        cursor_end = min(cursor_start + timedelta(seconds=GRAN * (MAX_CANDLES - 1)), end_dt)

        url = f"https://api.exchange.coinbase.com/products/{symbol}/candles"
        params = {
            "granularity": GRAN,
            "start": cursor_start.isoformat().replace("+00:00", "Z"),
            "end": cursor_end.isoformat().replace("+00:00", "Z"),
        }

        r = _req.get(url, params=params, timeout=timeout, headers=headers)
        if r.status_code == 429:
            _time.sleep(1.0)
            continue
        r.raise_for_status()
        arr = r.json()

        if arr:
            df_chunk = _pd.DataFrame(arr, columns=["time","low","high","open","close","volume"])
            df_chunk["timestamp"] = _pd.to_datetime(df_chunk["time"], unit="s", utc=True)
            df_chunk = df_chunk.astype({"open":"float","high":"float","low":"float","close":"float","volume":"float"})
            df_chunk = df_chunk[["timestamp","open","high","low","close","volume"]]
            out.append(df_chunk)

        # advance to next window (non-overlapping)
        cursor_start = cursor_end
        # brief politeness delay
        _time.sleep(0.12)

    if not out:
        raise ValueError(f"No data returned for {symbol}")

    df = _pd.concat(out, ignore_index=True)
    df = df.drop_duplicates(subset="timestamp", keep="last")
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


# ==============================
# Indicators & helpers
# ==============================

def ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False).mean()

def atr(df: pd.DataFrame, length: int) -> pd.Series:
    high, low, close = df['high'], df['low'], df['close']
    pc = close.shift(1)
    tr = np.maximum(high - low, np.maximum((high - pc).abs(), (low - pc).abs()))
    return tr.rolling(length, min_periods=1).mean()

def typical_price(df: pd.DataFrame) -> pd.Series:
    return (df["high"] + df["low"] + df["close"]) / 3.0

def is_local_max(series: pd.Series, i: int, w: int) -> bool:
    window = series.iloc[i-w:i+w+1]
    return series.iloc[i] >= window.max()

def is_local_min(series: pd.Series, i: int, w: int) -> bool:
    window = series.iloc[i-w:i+w+1]
    return series.iloc[i] <= window.min()

# ==============================
# Zone data structures
# ==============================

@dataclass
class Zone:
    center: float
    halfwidth: float
    touches: int
    source: str     # 'swing','touch','fib','pivot','ema','avwap','vbp'
    label: str = "" # e.g., 'EMA50', 'R1', 'Fib 0.618'

from dataclasses import dataclass, field

@dataclass
class ZoneCluster:
    low: float
    high: float
    center: float
    members: list = field(default_factory=list)        # list[Zone]
    sources: set = field(default_factory=set)          # unique zone.source types
    touches_sum: int = 0                               # sum of touches for swing/touch/vbp
    has_ema200: bool = False
    has_fib618: bool = False
    inside: bool = False                               # price is inside cluster
    dist_pct: float = 0.0                              # distance from price to cluster (0 if inside)
    label_sr: str = ""                                 # SUPPORT / RESISTANCE / AT PRICE
    score: float = 0.0                                 # cluster importance score

# ==============================
# Swing S/R (prominent) + clustering
# ==============================
def merge_zones_to_clusters(zones: List[Zone], price_now: float, cfg: Config) -> List[ZoneCluster]:
    """Merge all zone types into confluence clusters, compute SR label & rank."""
    if not zones:
        return []

    # Sort by center; clusters form when centers are within tol
    tol = cfg.tight_cluster_pct * price_now   # use the tighter 0.6% overlap

    zones_sorted = sorted(zones, key=lambda z: z.center)

    raw_clusters: List[List[Zone]] = []
    cur = [zones_sorted[0]]
    for z in zones_sorted[1:]:
        if abs(z.center - cur[-1].center) <= tol:
            cur.append(z)
        else:
            raw_clusters.append(cur)
            cur = [z]
    raw_clusters.append(cur)

    clusters: List[ZoneCluster] = []
    for group in raw_clusters:
        lows  = [g.center - g.halfwidth for g in group]
        highs = [g.center + g.halfwidth for g in group]
        low, high = float(min(lows)), float(max(highs))
        center = 0.5 * (low + high)

        sources = {g.source for g in group}
        touches_sum = sum(g.touches for g in group if g.source in ("swing","touch","vbp"))
        has_ema200 = any((g.source == "ema" and (g.label or "").upper() == "EMA200") for g in group)
        has_fib618 = any((g.source == "fib" and ("0.618" in (g.label or ""))) for g in group)

        inside = (low <= price_now <= high)
        if inside:
            dist_pct = 0.0
        else:
            edge = high if price_now > high else low
            dist_pct = abs(price_now - edge) / price_now

        # Label SUPPORT/RESISTANCE
        if inside:
            label_sr = "AT PRICE"
        elif high < price_now:
            label_sr = "SUPPORT"
        else:
            label_sr = "RESISTANCE"

        # Cluster importance score (tunable weights)
        # – confluence weight: unique sources
        # – structure weight: touches_sum
        # – premium signals: EMA200 / Fib 0.618
        # – immediacy: inside cluster gets bonus; near cluster gets small decay
        confluence_w = 6.0 * len(sources)
        touches_w    = 0.5 * touches_sum
        premium_w    = (1.5 if has_ema200 else 0.0) + (1.0 if has_fib618 else 0.0)
        proximity_w  = (4.0 if inside else max(0.0, 2.0 * (1.0 - min(dist_pct / (cfg.proximity_falloff_pct or 1e-9), 1.0))))

        score = confluence_w + touches_w + premium_w + proximity_w

        clusters.append(ZoneCluster(
            low=low, high=high, center=center, members=group, sources=sources,
            touches_sum=touches_sum, has_ema200=has_ema200, has_fib618=has_fib618,
            inside=inside, dist_pct=dist_pct, label_sr=label_sr, score=score
        ))

    # Sort strongest first
    clusters.sort(key=lambda c: c.score, reverse=True)
    return clusters

def find_prominent_swings(df: pd.DataFrame, w: int, min_retrace_pct: float) -> List[Tuple[float,int]]:
    highs, lows = df["high"], df["low"]
    swings: List[Tuple[float,int]] = []

    for i in range(w, len(df)-w):
        if is_local_max(highs, i, w):
            swings.append((float(highs.iloc[i]), i))
        if is_local_min(lows, i, w):
            swings.append((float(lows.iloc[i]), i))

    # Prominence: must move away by min_retrace_pct after the swing
    filt: List[Tuple[float,int]] = []
    for price, idx in swings:
        rest = df.iloc[idx+1:]
        if rest.empty:
            continue
        moved = False
        # If swing high: look for drop of >= pct
        if df["high"].iloc[idx] == price:
            moved = ((price - rest["low"]).max() / price) >= min_retrace_pct
        else:
            moved = ((rest["high"].max() - price) / price) >= min_retrace_pct
        if moved:
            filt.append((price, idx))
    return filt

def cluster_levels_to_zones(levels: List[Tuple[float,int]], price_now: float, atr_abs: float, cfg: Config) -> List[Zone]:
    if not levels:
        return []
    levels_sorted = sorted(levels, key=lambda x: x[0])
    tol_abs = max(cfg.cluster_tolerance_pct * price_now, cfg.cluster_tolerance_atr_mult * atr_abs)

    clusters: List[List[Tuple[float,int]]] = []
    cur = [levels_sorted[0]]
    for p, i in levels_sorted[1:]:
        if abs(p - cur[-1][0]) <= tol_abs:
            cur.append((p,i))
        else:
            clusters.append(cur)
            cur = [(p,i)]
    clusters.append(cur)

    halfwidth = max(cfg.zone_halfwidth_atr_mult * atr_abs, cfg.zone_halfwidth_min_pct * price_now)
    zones = [Zone(center=float(np.mean([p for p,_ in cl])), halfwidth=halfwidth, touches=len(cl), source="swing", label=f"Swing x{len(cl)}") for cl in clusters]
    return zones

def swing_zones(df: pd.DataFrame, cfg: Config) -> List[Zone]:
    price_now = df["close"].iloc[-1]
    atr_abs = float(atr(df, cfg.atr_length).iloc[-1])

    # 1) find prominent swings (unchanged)
    swings = find_prominent_swings(df, cfg.swing_window, cfg.min_retrace_pct)
    if not swings:
        return []

    # 2) tighter clustering by center distance
    tol_abs = cfg.tight_cluster_pct * price_now
    levels_sorted = sorted(swings, key=lambda x: x[0])
    clusters, cur = [], [levels_sorted[0]]
    for p, i in levels_sorted[1:]:
        if abs(p - cur[-1][0]) <= tol_abs:
            cur.append((p, i))
        else:
            clusters.append(cur); cur = [(p, i)]
    clusters.append(cur)

    # 3) tight zone width
    halfwidth = tight_halfwidth(price_now, atr_abs, cfg)
    return [Zone(center=float(np.mean([p for p,_ in cl])),
                 halfwidth=halfwidth, touches=len(cl), source="swing",
                 label=f"Swing x{len(cl)}") for cl in clusters]


# ==============================
# Range / touch clustering (multi-touch bins)
# ==============================

def touch_zones(df: pd.DataFrame, cfg: Config) -> List[Zone]:
    df_recent = df.tail(cfg.recent_days_for_touch_vbp)  # NEW: recent-only
    price_now = df_recent["close"].iloc[-1]
    atr_abs = float(atr(df_recent, cfg.atr_length).iloc[-1])

    bin_size = max(cfg.touch_bin_pct * price_now, 0.002 * price_now)
    near_abs = min(0.5 * bin_size, 0.0025 * price_now)

    counts: Dict[int, int] = {}
    for _, row in df_recent.iterrows():                      # use df_recent here
        for p in (row["close"], row["high"], row["low"]):
            idx = int(round(p / bin_size))
            center = idx * bin_size
            if abs(p - center) <= near_abs:
                counts[idx] = counts.get(idx, 0) + 1

    halfwidth = tight_halfwidth(price_now, atr_abs, cfg)
    zones = []
    for idx, cnt in counts.items():
        if cnt >= cfg.touch_min_count:
            center = idx * bin_size
            zones.append(Zone(center=center, halfwidth=halfwidth, touches=cnt, source="touch",
                              label=f"Touch x{cnt}"))

    # Tight merge
    zones.sort(key=lambda z: z.center)
    merged = []
    tol_abs = cfg.tight_cluster_pct * price_now
    for z in zones:
        if merged and abs(z.center - merged[-1].center) <= tol_abs:
            prev = merged[-1]
            c = (prev.center + z.center) / 2.0
            merged[-1] = Zone(center=c, halfwidth=halfwidth, touches=prev.touches + z.touches,
                              source="touch", label=f"Touch x{prev.touches + z.touches}")
        else:
            merged.append(z)
    return merged



# ==============================
# Latest major swing + Fibonacci zones
# ==============================

def latest_major_swing(df: pd.DataFrame, threshold_pct: float) -> Tuple[int, int]:
    """
    Find the most recent anchor index 'a' such that price moved >= threshold_pct
    from the last close between a..end. Returns (anchor_idx, last_idx).
    """
    closes = df["close"].values
    n = len(closes)
    last_idx = n - 1
    last = closes[last_idx]
    anchor_idx = None
    for i in range(n - 2, -1, -1):
        move = abs((closes[i] - last) / last)
        if move >= threshold_pct:
            anchor_idx = i
            break
    if anchor_idx is None:
        anchor_idx = max(0, n - 120)  # default to ~6 months back if no large move
    return anchor_idx, last_idx

def fib_zones(df: pd.DataFrame, cfg: Config) -> List[Zone]:
    a, b = latest_major_swing(df, cfg.major_swing_thresh)
    seg_high = float(df["high"].iloc[a:b+1].max())
    seg_low  = float(df["low"].iloc[a:b+1].min())
    top, bot = (seg_high, seg_low) if seg_high >= seg_low else (seg_low, seg_high)
    span = top - bot
    if span <= 0:
        return []

    price_now = df["close"].iloc[-1]
    atr_abs = float(atr(df, cfg.atr_length).iloc[-1])
    halfwidth = tight_halfwidth(price_now, atr_abs, cfg)
  


    # If trend up (top is high): retrace down from top; if trend down: retrace up from bot
    levels = [top - r * span if seg_high >= seg_low else bot + r * span for r in cfg.fib_ratios]
    zones = [Zone(center=float(lv), halfwidth=halfwidth, touches=1, source="fib", label=f"Fib {r:.3f}") for lv, r in zip(levels, cfg.fib_ratios)]
    return zones

# ==============================
# Weekly pivots (classic)
# ==============================

def weekly_pivots_zones(df: pd.DataFrame, cfg: Config) -> List[Zone]:
    d = df.set_index("timestamp")
    # last fully completed week (Sun end). Use 'W-SUN'
    wk = d.resample("W-SUN").agg({"high":"max","low":"min","close":"last"})
    if len(wk) < 2:
        return []
    prev = wk.iloc[-2]
    P = (prev["high"] + prev["low"] + prev["close"]) / 3.0
    R1 = 2*P - prev["low"]
    S1 = 2*P - prev["high"]
    R2 = P + (prev["high"] - prev["low"])
    S2 = P - (prev["high"] - prev["low"])

    price_now = df["close"].iloc[-1]
    atr_abs = float(atr(df, cfg.atr_length).iloc[-1])
    halfwidth = tight_halfwidth(price_now, atr_abs, cfg)
  


    zones = [
        Zone(center=float(P),  halfwidth=halfwidth, touches=1, source="pivot", label="P"),
        Zone(center=float(R1), halfwidth=halfwidth, touches=1, source="pivot", label="R1"),
        Zone(center=float(S1), halfwidth=halfwidth, touches=1, source="pivot", label="S1"),
        Zone(center=float(R2), halfwidth=halfwidth, touches=1, source="pivot", label="R2"),
        Zone(center=float(S2), halfwidth=halfwidth, touches=1, source="pivot", label="S2"),
    ]
    return zones

# ==============================
# EMAs as dynamic S/R
# ==============================

def ema_zones(df: pd.DataFrame, cfg: Config) -> List[Zone]:
    closes = df["close"]
    e50 = float(ema(closes, cfg.ema_fast).iloc[-1])
    e200 = float(ema(closes, cfg.ema_slow).iloc[-1])
    atr_abs = float(atr(df, cfg.atr_length).iloc[-1])
    price_now = closes.iloc[-1]
    halfwidth = max(0.005 * price_now, 0.0025 * price_now)  

    return [
        Zone(center=e50,  halfwidth=halfwidth, touches=1, source="ema", label="EMA50"),
        Zone(center=e200, halfwidth=halfwidth, touches=1, source="ema", label="EMA200"),
    ]

# ==============================
# Anchored VWAP from latest major swing
# ==============================

def anchored_vwap_zones(df: pd.DataFrame, cfg: Config) -> List[Zone]:
    a, b = latest_major_swing(df, cfg.major_swing_thresh)
    tp = typical_price(df).iloc[a:b+1]
    vol = df["volume"].iloc[a:b+1]
    cum_v = vol.cumsum()
    cum_pv = (tp * vol).cumsum()
    avwap_series = cum_pv / cum_v.replace(0, np.nan)
    level = float(avwap_series.iloc[-1])

    price_now = df["close"].iloc[-1]
    atr_abs = float(atr(df, cfg.atr_length).iloc[-1])
    halfwidth = tight_halfwidth(price_now, atr_abs, cfg)
  


    return [Zone(center=level, halfwidth=halfwidth, touches=1, source="avwap", label="Anchored VWAP")]

# ==============================
# Volume-by-Price HVNs (approx using close distribution)
# ==============================

def vbp_hvn_zones(df: pd.DataFrame, cfg: Config) -> List[Zone]:
    df_recent = df.tail(cfg.recent_days_for_touch_vbp)  # NEW: recent-only
    closes = df_recent["close"].values
    vols = df_recent["volume"].values
    if len(closes) < 10:
        return []

    pmin, pmax = float(np.min(closes)), float(np.max(closes))
    if pmax <= pmin:
        return []

    hist, edges = np.histogram(closes, bins=cfg.vbp_bins, range=(pmin, pmax), weights=vols)
    centers = (edges[:-1] + edges[1:]) / 2.0
    idxs = np.argsort(hist)[::-1][:cfg.vbp_topn]

    price_now = df_recent["close"].iloc[-1]
    atr_abs = float(atr(df_recent, cfg.atr_length).iloc[-1])
    bin_w = (pmax - pmin) / cfg.vbp_bins
    halfwidth = tight_halfwidth(price_now, atr_abs, cfg)

    zones = []
    for i in idxs:
        if hist[i] <= 0:
            continue
        zones.append(Zone(center=float(centers[i]), halfwidth=halfwidth, touches=int(hist[i] > 0),
                          source="vbp", label="HVN"))
    return zones


# ==============================
# Scoring & ranking
# ==============================

def proximity_score(price: float, zone: Zone, cfg: Config) -> Tuple[float, bool]:
    dist = abs(price - zone.center)
    falloff_abs = cfg.proximity_falloff_pct * price

    # HARD FILTER: if beyond max distance, no points
    if dist > (cfg.max_zone_distance_pct * price) and dist > (zone.halfwidth + falloff_abs):
        return 0.0, False

    if dist <= zone.halfwidth:
        base = cfg.proximity_in_zone_score
    elif dist <= zone.halfwidth + falloff_abs:
        rem = (zone.halfwidth + falloff_abs - dist) / falloff_abs
        base = cfg.proximity_falloff_points * max(0.0, rem)
    else:
        return 0.0, False

    touch_bonus = 0.0
    if zone.source in ("swing", "touch", "vbp"):
        touch_bonus = min(1.0, 0.25 * max(0, zone.touches - 1))

    # Extra decay by relative distance (keeps very near levels on top)
    rel = dist / max(price, 1e-9)
    proximity_decay = max(0.0, 1.0 - (rel / cfg.max_zone_distance_pct))  # 1→0 across the allowed band

    score = (base + touch_bonus) * CFG.weights.get(zone.source, 1.0) * proximity_decay
    return score, dist <= zone.halfwidth


def confluence_bonus(active_zones: List[Zone], price_now: float, cfg: Config) -> float:
    """
    Cluster active zones (those already giving >0 proximity points) by
    near-overlap, and award a bonus per additional zone in each cluster.
    """
    if len(active_zones) < 2:
        return 0.0
    tol = cfg.confluence_overlap_pct * price_now
    centers = sorted([z.center for z in active_zones])
    clusters: List[List[float]] = []
    cur = [centers[0]]
    for c in centers[1:]:
        if abs(c - cur[-1]) <= tol:
            cur.append(c)
        else:
            clusters.append(cur)
            cur = [c]
    clusters.append(cur)

    bonus = 0.0
    for cl in clusters:
        k = len(cl)
        if k >= 2:
            bonus += cfg.confluence_per_overlap * (k - 1)
    return min(bonus, cfg.confluence_cap)

def score_symbol(symbol: str, df: pd.DataFrame, cfg: Config) -> Dict:
    price_now = float(df["close"].iloc[-1])

    # Build all zones
    zones: List[Zone] = []
    zones += swing_zones(df, cfg)
    zones += touch_zones(df, cfg)
    zones += fib_zones(df, cfg)
    zones += weekly_pivots_zones(df, cfg)
    zones += ema_zones(df, cfg)
    zones += anchored_vwap_zones(df, cfg)
    zones += vbp_hvn_zones(df, cfg)

    # Score proximity
    parts: List[Tuple[str, float, float, str]] = []  # (source, score, center, label)
    active_zones: List[Zone] = []
    total = 0.0
    for z in zones:
        sc, inside = proximity_score(price_now, z, cfg)
        if sc > 0:
            parts.append((z.source, sc, z.center, z.label))
            active_zones.append(z)
            total += sc

    # Confluence bonus across nearby zone centers
    total += confluence_bonus(active_zones, price_now, cfg)

    # Sort part reasons by contribution
    parts.sort(key=lambda x: x[1], reverse=True)

    return {
        "symbol": symbol,
        "price": price_now,
        "total": total,
        "parts": parts,
    }

def normalize_to_10(results: List[Dict]) -> List[Dict]:
    if not results:
        return results
    max_total = max(r["total"] for r in results) or 1.0
    for r in results:
        r["score_10"] = round(10.0 * r["total"] / max_total, 1)
    return results

# ==============================
# Main
# ==============================

def scan(symbols: List[str], cfg: Config) -> List[Dict]:
    out = []
    for sym in symbols:
        try:
            df = fetch_coinbase_ohlcv_1d(sym, cfg.lookback_days, cfg.request_timeout)
            if len(df) < max(cfg.ema_slow + 10, 60):
                print(f"[skip] {sym}: not enough data.")
                continue
            res = score_symbol(sym, df, cfg)
            out.append(res)
        except Exception as e:
            print(f"[error] {sym}: {e}")
    out.sort(key=lambda x: x["total"], reverse=True)
    return normalize_to_10(out)

def _fmt_pct(x: float) -> str:
    return f"{x*100:.1f}%"

def _fmt_zone_range(low: float, high: float, price: float) -> str:
    # Use dynamic precision based on price magnitude
    def prec(p):
        return 6 if p < 0.01 else 4 if p < 1 else 2
    return f"{low:.{prec(low)}f}–{high:.{prec(high)}f}"

def action_hint(clusters: List[ZoneCluster], price: float) -> str:
    if not clusters:
        return "No key zones in play."
    top = clusters[0]
    if top.inside and top.label_sr in ("SUPPORT", "AT PRICE") and "resistance" not in [s for s in top.sources]:
        return "Sitting on support; watch for reversal signal or breakdown."
    if top.inside and top.label_sr == "RESISTANCE":
        return "Pressing resistance; look for breakout or rejection."
    if not top.inside and top.label_sr == "SUPPORT":
        return f"Nearest support { _fmt_pct(top.dist_pct) } below."
    if not top.inside and top.label_sr == "RESISTANCE":
        return f"Nearest resistance { _fmt_pct(top.dist_pct) } above."
    return "At key area; wait for confirmation."
def _fmt_pct(x: float) -> str:
    return f"{x*100:.1f}%"

def _fmt_zone_range(low: float, high: float, price: float) -> str:
    # Dynamic precision based on price magnitude
    def prec(p):
        return 6 if p < 0.01 else 4 if p < 1 else 2
    return f"{low:.{prec(low)}f}–{high:.{prec(high)}f}"

def print_results(results: List[Dict], top_n: int = 25, zones_per_coin: int = 3):
    """
    Trader-focused output:
      - Score (/10)
      - Top nearby clusters per coin (tight ranges, SR label, confluence, distance)
      - Auto-generated trade plans; prints only 'Tradable' (R:R ≥ CFG.min_rr) if CFG.show_only_tradable=True
    Requires:
      - fetch_coinbase_ohlcv_1d, swing_zones, touch_zones, fib_zones, weekly_pivots_zones,
        ema_zones, anchored_vwap_zones, vbp_hvn_zones, merge_zones_to_clusters, build_trade_plans
      - CFG with max_zone_distance_pct, keep_nearest_if_empty, min_rr, show_only_tradable set
    """
    print("\n=== Key SR Confluence Scan (Coinbase • Daily) ===")
    for r in results[:top_n]:
        if r.get("total", 0) <= 0:
            continue

        symbol = r["symbol"]
        price = r["price"]
        print(f"{symbol:>10s} | Price {price:.6f} | Score {r['score_10']}/10")

        # --- rebuild clusters for this symbol (tight zones) ---
        try:
            df = fetch_coinbase_ohlcv_1d(symbol, CFG.lookback_days, CFG.request_timeout)

            # Build zones (some may use recent-only windows per your earlier tweaks)
            zones: List[Zone] = []
            zones += swing_zones(df, CFG)
            zones += touch_zones(df, CFG)
            zones += fib_zones(df, CFG)
            zones += weekly_pivots_zones(df, CFG)
            zones += ema_zones(df, CFG)
            zones += anchored_vwap_zones(df, CFG)
            zones += vbp_hvn_zones(df, CFG)

            clusters = merge_zones_to_clusters(zones, price, CFG)
        except Exception as e:
            print(f"   (data error: {e})")
            continue

        # --- keep only clusters that are near/inside price ---
        near = [c for c in clusters if c.inside or c.dist_pct <= CFG.max_zone_distance_pct]

        # Fallback: if nothing near, keep nearest support & resistance (optional)
        if not near and CFG.keep_nearest_if_empty:
            sup = [(abs(c.high - price), c) for c in clusters if c.label_sr == "SUPPORT" and c.high <= price]
            res = [(abs(c.low  - price), c) for c in clusters if c.label_sr == "RESISTANCE" and c.low  >= price]
            keep = []
            if sup: keep.append(sorted(sup, key=lambda x: x[0])[0][1])
            if res: keep.append(sorted(res, key=lambda x: x[0])[0][1])
            near = keep

        clusters = near[:zones_per_coin]

        if not clusters:
            print("   (no actionable clusters near price)")
            continue

        # --- print top clusters ---
        for c in clusters:
            zrange = _fmt_zone_range(c.low, c.high, price)
            dist = "inside" if c.inside else _fmt_pct(c.dist_pct) + " away"
            srcs = ",".join(sorted(c.sources))
            reasons = []
            if c.has_ema200: reasons.append("EMA200")
            if c.has_fib618: reasons.append("Fib 0.618")
            if c.touches_sum >= 8: reasons.append(f"{c.touches_sum} touches")
            elif c.touches_sum >= 4: reasons.append(f"{c.touches_sum} touches")
            reason_str = (" | " + ", ".join(reasons)) if reasons else ""
            print(f"   · {c.label_sr:<10s} {zrange} "
                  f"| Confluence {len(c.sources)} ({srcs}) "
                  f"| {dist} "
                  f"| Strength {c.score:.1f}{reason_str}")

        # --- trade plan ideas from strongest nearby cluster(s) ---
        plans = build_trade_plans(price, clusters, CFG)

        # Compute entry status and prioritize NOW > NEAR > WAIT
        def _status_rank(s: str) -> int:
            return {"NOW": 0, "NEAR": 1, "WAIT": 2}.get(s, 3)

        enriched = []
        for p in plans:
            status = entry_status(price, p.entry_low, p.entry_high, CFG.entry_now_near_pct)
            enriched.append((status, p))
        enriched.sort(key=lambda x: _status_rank(x[0]))

        tradable = [(s, p) for (s, p) in enriched if getattr(p, "tradable", False)]
        skips    = [(s, p) for (s, p) in enriched if not getattr(p, "tradable", False)]

        def _print_plan(prefix: str, status: str, p: TradePlan):
            rng = f"{p.entry_low:.6f}–{p.entry_high:.6f}"
            tag = "🔔 IN ENTRY" if status == "NOW" else ("👀 NEAR" if status == "NEAR" else "⏳ WAIT")
            print(f"     {prefix} {tag}  {p.side:<5s} Entry {rng} | Stop {p.stop:.6f} | "
                  f"TP {p.target:.6f} | R:R {p.rr:.2f}  ({p.note})")

        if CFG.show_only_tradable:
            if tradable:
                for status, p in tradable:
                    _print_plan("✅", status, p)
            else:
                print("     ❌ No tradable plan (fails R:R filter).")
        else:
            for status, p in tradable:
                _print_plan("✅", status, p)
            for status, p in skips:
                why = getattr(p, "why", "") or f"R:R {p.rr:.2f} < {CFG.min_rr}"
                rng = f"{p.entry_low:.6f}–{p.entry_high:.6f}"
                tag = "🔔 IN ENTRY" if status == "NOW" else ("👀 NEAR" if status == "NEAR" else "⏳ WAIT")
                print(f"     ⚠️  {tag}  {p.side:<5s} Entry {rng} | Stop {p.stop:.6f} | "
                      f"TP {p.target:.6f} | R:R {p.rr:.2f}  — Skip ({why})")




if __name__ == "__main__":
    # Edit this list to your universe
    SYMBOLS = [
    "BTC-USD","ETH-USD","XRP-USD","SOL-USD","ADA-USD","AVAX-USD","DOGE-USD","DOT-USD",
    "LINK-USD","ATOM-USD","NEAR-USD","ARB-USD","OP-USD","MATIC-USD","SUI-USD",
    "INJ-USD","AAVE-USD","LTC-USD","BCH-USD","ETC-USD","ALGO-USD","FIL-USD","ICP-USD",
    "RNDR-USD","STX-USD","JTO-USD","PYTH-USD","GRT-USD","SEI-USD",
    "ENS-USD","FLOW-USD","KSM-USD","KAVA-USD",
    "WLD-USD","HBAR-USD","JUP-USD","STRK-USD",
    "ONDO-USD","SUPER-USD","LDO-USD","POL-USD",
    "ZETA-USD","ZRO-USD","TIA-USD",
    "WIF-USD","MAGIC-USD","APE-USD","JASMY-USD","SYRUP-USD","FARTCOIN-USD",
    "AERO-USD","FET-USD","CRV-USD","TAO-USD","XCN-USD","UNI-USD","MKR-USD",
    "TOSHI-USD","TRUMP-USD","PEPE-USD","XLM-USD","MOODENG-USD","BONK-USD",
    "POPCAT-USD","QNT-USD","IP-USD","PNUT-USD","APT-USD","ENA-USD","TURBO-USD",
    "BERA-USD","MASK-USD","SAND-USD","MORPHO-USD","MANA-USD","C98-USD","AXS-USD"
]
    results = scan(SYMBOLS, CFG)
    print_results(results, top_n=30)
