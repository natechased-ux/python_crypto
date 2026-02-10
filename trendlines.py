#!/usr/bin/env python3
"""
Trendline Bounce Planner (clean): plots BTC-USD 1H with pivot-snapped trendlines,
plus Entry / SL / TP and R:R. Telegram loop removed for clarity.
Install (if needed):  py -m pip install requests pandas numpy matplotlib
"""

from __future__ import annotations
import math, requests
from dataclasses import dataclass
from typing import List, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

CB = "https://api.exchange.coinbase.com"
ONE_HOUR = 3600

# ---------- Tunables ----------
PRODUCT = "BTC-USD"
GRANULARITY = ONE_HOUR
PIVOT_LEFT = 2
PIVOT_RIGHT = 2
MIN_TOUCHES = 3
MIN_SPAN = 36                 # bars between first and last pivot used
MAX_DEV_BP = 20               # max pivot distance from line (basis points) when fitting
PROXIMITY_BP = 40             # last price must be within this to annotate trade
ATR_LEN = 14
SL_ATR_MULT = 0.5             # SL buffer
SWING_LOOKBACK = 30           # fallback swing window for TP
# -------------------------------

def fetch_candles(product_id=PRODUCT, granularity=GRANULARITY):
    r = requests.get(f"{CB}/products/{product_id}/candles",
                     params={"granularity": granularity}, timeout=15)
    r.raise_for_status()
    df = pd.DataFrame(r.json(), columns=["time","low","high","open","close","volume"]).sort_values("time")
    df.reset_index(drop=True, inplace=True)
    return df

def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False, min_periods=span).mean()

def atr(df: pd.DataFrame, length=ATR_LEN):
    h,l,c = df["high"], df["low"], df["close"]
    tr = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(length).mean()

def pivots(series: pd.Series, left=PIVOT_LEFT, right=PIVOT_RIGHT):
    s = series.values
    hi = np.full(len(s), False); lo = np.full(len(s), False)
    for i in range(left, len(s)-right):
        win = s[i-left:i+right+1]
        if s[i] == win.max() and win.argmax()==left: hi[i]=True
        if s[i] == win.min() and win.argmin()==left: lo[i]=True
    return hi, lo

def fit_line(xs, ys) -> Tuple[float,float]:
    x = np.asarray(xs); y = np.asarray(ys)
    m = np.cov(x, y, bias=True)[0,1] / (np.var(x)+1e-12)
    b = y.mean() - m*x.mean()
    return float(m), float(b)

def bp_dist(px, ref) -> float:
    return abs((px - ref)/px) * 1e4

@dataclass
class TL:
    kind: str      # "support" or "resistance"
    slope: float
    intercept: float
    idx0: int
    idx1: int
    touches: int
    max_dev_bp: float

def build_trendlines(df: pd.DataFrame,
                     max_dev_bp=MAX_DEV_BP,
                     min_touches=MIN_TOUCHES,
                     min_span=MIN_SPAN) -> List[TL]:
    """Greedy pivot fitter with deviation control; returns clean lines."""
    hi_mask, _ = pivots(df["high"])
    _, lo_mask = pivots(df["low"])
    piv_highs = [(i, df["high"].iloc[i]) for i, v in enumerate(hi_mask) if v]
    piv_lows  = [(i, df["low"].iloc[i])  for i, v in enumerate(lo_mask) if v]
    lines: List[TL] = []

    def grow(pivots_, kind):
        n = len(pivots_)
        for s in range(n-2):
            xs = [pivots_[s][0], pivots_[s+1][0]]
            ys = [pivots_[s][1], pivots_[s+1][1]]
            m,b = fit_line(xs, ys)

            # loose slope sanity: supports usually rise; resistances usually fall (short lookbacks)
            if kind=="support" and m <= -1e-9:  continue
            if kind=="resistance" and m >=  1e-9:  continue

            inx, iny = xs[:], ys[:]
            for k in range(s+2, n):
                i, y_p = pivots_[k]
                y_hat = m*i + b
                if bp_dist(y_p, y_hat) <= max_dev_bp:
                    inx.append(i); iny.append(y_p)
                    m,b = fit_line(inx, iny)

            if len(inx) >= min_touches and (max(inx)-min(inx)) >= min_span:
                devs = []
                for i,y in zip(inx,iny):
                    y_hat = m*i+b
                    devs.append(bp_dist(y, y_hat))
                lines.append(TL(kind, m, b, min(inx), max(inx), len(inx), max(devs)))

    grow(piv_lows, "support")
    grow(piv_highs, "resistance")
    return lines

def line_at(ln: TL, idx: int) -> float:
    return ln.slope*idx + ln.intercept

def nearest_support_and_resistance(lines: List[TL], price_now: float, idx_now: int):
    """Pick the single nearest support below and resistance above current price."""
    supports = [(ln, line_at(ln, idx_now)) for ln in lines if ln.kind=="support"]
    resistances = [(ln, line_at(ln, idx_now)) for ln in lines if ln.kind=="resistance"]

    below = [t for t in supports if t[1] <= price_now]
    above = [t for t in resistances if t[1] >= price_now]

    sup = max(below, key=lambda t: t[1]) if below else None       # closest below
    res = min(above, key=lambda t: t[1]) if above else None       # closest above
    return sup, res

def propose_sl(side: str, line_px: float, df: pd.DataFrame, mult=SL_ATR_MULT) -> float:
    val = float(atr(df, ATR_LEN).iloc[-1])
    buf = val * mult
    return round(line_px - buf, 6) if side=="LONG" else round(line_px + buf, 6)

def tp_from_channel(sup, res, side: str, idx_now: int, df: pd.DataFrame, price_now: float) -> float:
    """TP = opposite side of channel; fallback to recent swing if missing."""
    if side=="LONG":
        if res is not None:
            return round(res[1], 6)
        return round(float(df["high"].tail(SWING_LOOKBACK).max()), 6)
    else:
        if sup is not None:
            return round(sup[1], 6)
        return round(float(df["low"].tail(SWING_LOOKBACK).min()), 6)

def plot_trendlines(product=PRODUCT, granularity=GRANULARITY):
    df = fetch_candles(product, granularity)
    lines = build_trendlines(df)
    idx_now = len(df)-1
    price_now = float(df["close"].iloc[idx_now])

    # choose neat pair: nearest support below & nearest resistance above
    sup, res = nearest_support_and_resistance(lines, price_now, idx_now)

    fig, ax = plt.subplots(figsize=(11,6))
    ax.plot(df.index, df["close"], linewidth=1.2, label="Close")

    # draw only the chosen two lines (if found)
    def draw_line(tpl, style="--"):
        if tpl is None: return
        ln, y_now = tpl
        xs = np.arange(ln.idx0, idx_now+1)
        ys = ln.slope*xs + ln.intercept
        ax.plot(xs, ys, linestyle=style, linewidth=1.2)

    draw_line(sup, "--")
    draw_line(res, "--")

    # If near either line, annotate planned trade
    signals = []
    if sup is not None:
        d_bp = bp_dist(price_now, sup[1])
        if d_bp <= PROXIMITY_BP and price_now >= sup[1]:
            side="LONG"
            sl = propose_sl(side, sup[1], df)
            tp = tp_from_channel(sup, res, side, idx_now, df, price_now)
            rr = abs((tp - price_now)/(price_now - sl)) if (price_now - sl)!=0 else float("nan")
            signals.append((side, sl, tp, rr))
    if res is not None:
        d_bp = bp_dist(price_now, res[1])
        if d_bp <= PROXIMITY_BP and price_now <= res[1]:
            side="SHORT"
            sl = propose_sl(side, res[1], df)
            tp = tp_from_channel(sup, res, side, idx_now, df, price_now)
            rr = abs((price_now - tp)/(sl - price_now)) if (sl - price_now)!=0 else float("nan")
            signals.append((side, sl, tp, rr))

    # annotate if we have any signal
    for side, sl, tp, rr in signals:
        ax.scatter([idx_now], [price_now], label="Entry")
        ax.scatter([idx_now], [sl], label="SL")
        ax.scatter([idx_now], [tp], label="TP")
        ax.annotate(
            f"{side} | Entry {price_now:.2f} | SL {sl:.2f} | TP {tp:.2f} | R:R {rr:.2f}",
            xy=(idx_now, price_now),
            xytext=(10,10), textcoords="offset points",
            fontsize=8, bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.5)
        )

    ax.set_title(f"{product} — {granularity//60}m trendlines (pivot-snapped)")
    ax.set_xlabel("Bars"); ax.set_ylabel("Price"); ax.legend(loc="best"); plt.tight_layout(); plt.show()

if __name__ == "__main__":
    plot_trendlines()
