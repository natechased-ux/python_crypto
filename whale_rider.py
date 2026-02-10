"""
Whale Overrun & Bounce Alerts — with Strength Scoring (Coinbase, REST polling)
-------------------------------------------------------------------------------
Detects two scenarios around large whale clusters and ranks every alert 0–100:
1) **Overrun (skipped liquidity)**: price blows through a large bid/ask cluster → trade with momentum.
2) **Bounce (support/resistance hold)**: price taps a large cluster and **does not** overrun it → fade in opposite direction.

Noise controls + scoring:
- Stricter clusters (top percentile, min notional, dominance, proximity window)
- Trend + Stoch filters
- **Strength score** blends: cluster dominance, trend distance (ATR), Stoch RSI quality, 1m volume spike, and pattern severity/quality.
- Telegram shows ⭐ rating and top 2 reasons.

Python ≥3.9 (works on 3.13.5). Coinbase REST only.
"""

import time
import csv
import requests
from datetime import datetime, timezone
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

# --------------------------- CONFIG ---------------------------------
BASE_URL = "https://api.exchange.coinbase.com"  # Coinbase Exchange (aka GDAX) REST
PRODUCTS = [   "BTC-USD", "ETH-USD", "XRP-USD", "SOL-USD", "DOGE-USD", "ADA-USD", "XLM-USD", "SUI-USD",
    "BCH-USD", "LINK-USD", "HBAR-USD", "LTC-USD", "AVAX-USD", "SHIB-USD", "UNI-USD", "DOT-USD",
    "CRO-USD",  "AAVE-USD", "NEAR-USD", "ETC-USD", "ONDO-USD",
    "APT-USD", "ICP-USD", "ALGO-USD", "ARB-USD", "VET-USD",
    "ATOM-USD", "POL-USD", "BONK-USD", "RENDER-USD",
    "FET-USD", "SEI-USD", "ENA-USD", "FIL-USD", "QNT-USD", "LSETH-USD", "INJ-USD",
    "CRV-USD", "STX-USD", "TIA-USD"]

# Run mode
POLL_SECONDS = 30                # How often to poll REST
ALERT_MODE = "both"              # "overrun", "bounce", or "both"
COOLDOWN_MINUTES = 30            # cooldown per product+signal-type

# Order book & cluster settings (stricter to reduce noise)
ORDERBOOK_LEVEL = 2              # aggregated book
BIN_WIDTH_PCT = 0.0015           # 0.15% of price per bin
PCT_WITHIN_PRICE = 0.0125        # only consider book levels within ±1.25% of last price (was 2%)
TOP_CLUSTER_PERCENTILE = 0.95    # keep top 5% by notional size per side (was 10%)
MIN_CLUSTER_NOTIONAL = 15_000    # ignore clusters below this notional (was 5k)
MIN_DOMINANCE_RATIO = 1.6        # top cluster notional must be ≥ this × second-top
MIN_RELATIVE_SIZE_PCT = 0.25     # top notional must be ≥ this % of total side notional within window

# Overrun & bounce detection
OVERRUN_MARGIN_PCT = 0.0005      # must close this % beyond cluster price to count as "overrun"
LOOKBACK_MINS = 2                # cluster must have been touched within this many minutes
TOUCH_MARGIN_PCT = 0.0007        # for bounces, close must be within this % beyond cluster (but not overrun)

# Filters (toggle True/False)
USE_TREND_FILTER = True          # 200 EMA on 1H: long above / short below
USE_STOCH_DIR_FILTER = True      # 1H Stoch RSI direction/position filter

# Risk (ATR on 15m)
ATR_PERIOD = 14
TP_ATR_MULT = 2.0
SL_ATR_MULT = 1.5

# Telegram (user provided credentials)
TELEGRAM_BOT_TOKEN = "8177096945:AAHTg5nxVTA6hcPkAL4MLAAHPsZ1at7Ywmw"
TELEGRAM_CHAT_ID = "7967738614"

# CSV logging
LOG_PATH = "live_overrun_bounce_alerts.csv"

# ---------------------------- MODELS --------------------------------
@dataclass
class Cluster:
    price_bin: float
    total_size: float  # base size
    notional: float    # size * price_bin

@dataclass
class Signal:
    product_id: str
    signal_type: str   # "OVERRUN_LONG", "OVERRUN_SHORT", "BOUNCE_LONG", "BOUNCE_SHORT"
    cluster_price: float
    entry: float
    tp: float
    sl: float
    ts_iso: str
    strength: int
    reasons: List[str]
    context: Dict

# ------------------------- UTILITIES --------------------------------

def now_iso():
    return datetime.now(timezone.utc).isoformat()


def get_json(url: str, params: Dict = None) -> dict:
    r = requests.get(url, params=params, timeout=15)
    r.raise_for_status()
    return r.json()


def get_ticker(product_id: str) -> float:
    data = get_json(f"{BASE_URL}/products/{product_id}/ticker")
    return float(data["price"])  # last trade price


def get_candles(product_id: str, granularity: int, limit: int) -> List[List[float]]:
    """Return list of candles: [time, low, high, open, close, volume], ascending by time."""
    params = {"granularity": granularity}
    data = get_json(f"{BASE_URL}/products/{product_id}/candles", params=params)
    data = sorted(data, key=lambda x: x[0])
    return data[-limit:]


def ema(values: List[float], period: int) -> List[float]:
    out = []
    k = 2 / (period + 1)
    ema_val = None
    for v in values:
        if ema_val is None:
            ema_val = v
        else:
            ema_val = v * k + ema_val * (1 - k)
        out.append(ema_val)
    return out


def stoch_rsi(closes: List[float], period: int = 14, smooth_k: int = 3, smooth_d: int = 3) -> Tuple[List[float], List[float]]:
    # RSI
    gains, losses = [], []
    for i in range(1, len(closes)):
        ch = closes[i] - closes[i-1]
        gains.append(max(ch, 0))
        losses.append(abs(min(ch, 0)))
    avg_gain, avg_loss = [], []
    g, l = None, None
    for i in range(len(gains)):
        if i < period:
            pass
        if i == period:
            g = sum(gains[i-period+1:i+1]) / period
            l = sum(losses[i-period+1:i+1]) / period
        elif i > period:
            g = (g*(period-1) + gains[i]) / period
            l = (l*(period-1) + losses[i]) / period
        if g is not None:
            avg_gain.append(g)
            avg_loss.append(l)
    rsi = []
    for g, l in zip(avg_gain, avg_loss):
        if l == 0:
            rsi.append(100.0)
        else:
            rs = g/l
            rsi.append(100 - (100/(1+rs)))
    # Stoch of RSI
    k_vals = []
    for i in range(len(rsi)):
        window = rsi[max(0, i-period+1):i+1]
        if not window:
            k_vals.append(50.0)
            continue
        mn, mx = min(window), max(window)
        if mx - mn == 0:
            k_vals.append(50.0)
        else:
            k_vals.append(100 * (rsi[i]-mn)/(mx-mn))
    def sma(v, n):
        out = []
        for i in range(len(v)):
            w = v[max(0, i-n+1):i+1]
            out.append(sum(w)/len(w))
        return out
    K = sma(k_vals, smooth_k)
    D = sma(K, smooth_d)
    return K, D


def true_range(h, l, c_prev):
    return max(h - l, abs(h - c_prev), abs(l - c_prev))


def atr_from_candles(candles: List[List[float]], period: int = 14) -> float:
    trs = []
    for i in range(1, len(candles)):
        _, low, high, _, close, _ = candles[i]
        _, _, _, _, prev_close, _ = candles[i-1]
        trs.append(true_range(high, low, prev_close))
    if len(trs) < period:
        return sum(trs)/max(1, len(trs)) if trs else 0.0
    atr = sum(trs[:period]) / period
    for tr in trs[period:]:
        atr = (atr * (period - 1) + tr) / period
    return atr

# -------------------- ORDER BOOK CLUSTERING --------------------------

def get_order_book(product_id: str) -> dict:
    return get_json(f"{BASE_URL}/products/{product_id}/book", params={"level": ORDERBOOK_LEVEL})


def bin_price(price: float, bin_width: float) -> float:
    return round(price / bin_width) * bin_width


def build_clusters(levels: List[List[str]], last_price: float, side: str, bin_width: float) -> Tuple[List[Cluster], float, float]:
    """
    Return (clusters_sorted, total_side_notional, dominance_ratio_top2).
    clusters_sorted are filtered by percentile, min notional, dominance & relative size, and sorted by price in side direction.
    """
    bins: Dict[float, float] = {}
    total_notional = 0.0
    for p_str, sz_str, _ in levels:
        p = float(p_str)
        if abs(p - last_price)/last_price > PCT_WITHIN_PRICE:
            continue
        sz = float(sz_str)
        if sz <= 0:
            continue
        b = bin_price(p, bin_width)
        bins[b] = bins.get(b, 0.0) + sz
    clusters: List[Cluster] = []
    for b, size in bins.items():
        notional = b * size
        total_notional += notional
        if notional >= MIN_CLUSTER_NOTIONAL:
            clusters.append(Cluster(price_bin=b, total_size=size, notional=notional))
    if not clusters:
        return [], total_notional, 0.0
    # percentile filter
    notionals_sorted = sorted([c.notional for c in clusters])
    cutoff_idx = max(0, int(len(notionals_sorted) * TOP_CLUSTER_PERCENTILE) - 1)
    cutoff = notionals_sorted[cutoff_idx] if notionals_sorted else 0
    clusters = [c for c in clusters if c.notional >= cutoff]
    # dominance & relative size filters
    clusters_by_notional = sorted(clusters, key=lambda c: c.notional, reverse=True)
    dominance_ratio = 0.0
    if len(clusters_by_notional) >= 2:
        top, second = clusters_by_notional[0].notional, clusters_by_notional[1].notional
        if second > 0:
            dominance_ratio = top/second
        if dominance_ratio < MIN_DOMINANCE_RATIO:
            return [], total_notional, dominance_ratio
    if total_notional > 0 and (clusters_by_notional[0].notional / total_notional) < MIN_RELATIVE_SIZE_PCT:
        return [], total_notional, dominance_ratio
    # Price ordering for nearest selection
    reverse = True if side == "bids" else False
    clusters_sorted = sorted(clusters_by_notional, key=lambda c: c.price_bin, reverse=reverse)
    return clusters_sorted, total_notional, dominance_ratio


def nearest_cluster(clusters: List[Cluster], last_price: float, side: str) -> Optional[Cluster]:
    if not clusters:
        return None
    if side == "bids":
        below = [c for c in clusters if c.price_bin <= last_price]
        return below[0] if below else None
    else:
        above = [c for c in clusters if c.price_bin >= last_price]
        return above[0] if above else None

# ------------------------- SCORING -----------------------------------

def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def m1_volume_spike_score(m1: List[List[float]], lookback: int = 30) -> float:
    if len(m1) < lookback + 2:
        return 0.0
    vols = [c[5] for c in m1[-(lookback+1):-1]]
    vols_sorted = sorted(vols)
    med = vols_sorted[len(vols_sorted)//2]
    if med <= 0:
        return 0.0
    last_vol = m1[-1][5]
    ratio = last_vol / med
    return clamp((ratio - 1) / 3, 0.0, 1.0)  # 1x→0, 4x→1


def trend_distance_score(last: float, ema200: float, atr: float, side: str) -> float:
    if atr <= 0:
        return 0.0
    dist_atr = (last - ema200) / atr
    return clamp((dist_atr/3.0) if side == "LONG" else (-dist_atr/3.0), 0.0, 1.0)


def stoch_quality_score(K: float, D: float, side: str) -> float:
    sep = abs(K - D)
    sep_s = clamp(sep / 30.0, 0.0, 1.0)
    if side == "LONG":
        region = 1.0 if (K < 60 and K > D) else 0.0
    else:
        region = 1.0 if (K > 40 and K < D) else 0.0
    return 0.5*sep_s + 0.5*region


def cluster_strength_score(cluster_notional: float, total_side_notional: float, dominance_ratio: float) -> float:
    if total_side_notional <= 0:
        return 0.0
    rel = cluster_notional / total_side_notional
    rel_s = clamp((rel - 0.15) / 0.35, 0.0, 1.0)   # 15%→0, 50%→1
    dom_s = clamp((dominance_ratio - 1.2) / 1.3, 0.0, 1.0)  # 1.2x→0, 2.5x→1
    return 0.6*rel_s + 0.4*dom_s


def overrun_severity_score(close: float, cprice: float) -> float:
    pct = (close - cprice) / max(cprice, 1e-9)
    mag = abs(pct)
    return clamp(mag / (OVERRUN_MARGIN_PCT*4), 0.0, 1.0)


def bounce_quality_score(open_: float, close: float, cprice: float, side: str) -> float:
    body = abs(close - open_)
    prox = abs(close - cprice) / max(cprice, 1e-9)
    body_s = clamp(body / (prox*5 + 1e-9), 0.0, 1.0)
    dir_ok = (close > open_) if side == "LONG" else (close < open_)
    return 0.7*body_s + (0.3 if dir_ok else 0.0)


def build_strength_score(kind: str, side: str, *,
                         last_price: float,
                         ema200: float,
                         atr: float,
                         K: float,
                         D: float,
                         m1: List[List[float]],
                         cluster: Cluster,
                         total_side_notional: float,
                         dominance_ratio: float,
                         cprice: float) -> Tuple[int, List[str]]:
    """Return (score 0..100, top_reasons)."""
    # weights sum to 1.0
    w_cluster = 0.30
    w_trend = 0.20
    w_stoch = 0.15
    w_vol = 0.15
    w_pattern = 0.20

    cluster_s = cluster_strength_score(cluster.notional, total_side_notional, dominance_ratio)
    trend_s = trend_distance_score(last_price, ema200, atr, side)
    stoch_s = stoch_quality_score(K, D, side)
    vol_s = m1_volume_spike_score(m1)

    if kind == "OVERRUN":
        pattern_s = overrun_severity_score(m1[-1][4], cprice)
    else:
        open_ = m1[-1][3]
        close = m1[-1][4]
        pattern_s = bounce_quality_score(open_, close, cprice, side)

    score01 = (w_cluster*cluster_s + w_trend*trend_s + w_stoch*stoch_s + w_vol*vol_s + w_pattern*pattern_s)
    score = int(round(100 * score01))

    parts = [
        (cluster_s, "large, dominant cluster"),
        (trend_s, "strong trend alignment"),
        (stoch_s, "Stoch RSI confirmation"),
        (vol_s, "volume spike"),
        (pattern_s, "overrun severity" if kind == "OVERRUN" else "bounce quality"),
    ]
    parts.sort(key=lambda x: x[0], reverse=True)
    reasons = [parts[0][1], parts[1][1]]
    return score, reasons

# -------------------- SIGNAL DETECTION -------------------------------

def detect_overrun(product_id: str, last_price: float, m1: List[List[float]], obook: dict, bin_width: float):
    if len(m1) < max(LOOKBACK_MINS, 3):
        return None
    close = m1[-1][4]
    bids = obook.get("bids", [])
    asks = obook.get("asks", [])
    bid_clusters, bid_total, bid_dom = build_clusters(bids, last_price, "bids", bin_width)
    ask_clusters, ask_total, ask_dom = build_clusters(asks, last_price, "asks", bin_width)

    nb = nearest_cluster(bid_clusters, last_price, "bids")
    na = nearest_cluster(ask_clusters, last_price, "asks")

    # Bid overrun → LONG
    if nb is not None:
        cprice = nb.price_bin
        if close >= cprice * (1 + OVERRUN_MARGIN_PCT):
            recent = m1[-LOOKBACK_MINS:]
            touched = any((min(c[2], c[1]) <= cprice <= max(c[2], c[1])) for c in recent)
            if touched:
                return {
                    "side": "LONG",
                    "cluster": nb,
                    "total": bid_total,
                    "dom": bid_dom,
                    "cprice": cprice,
                }

    # Ask overrun → SHORT
    if na is not None:
        cprice = na.price_bin
        if close <= cprice * (1 - OVERRUN_MARGIN_PCT):
            recent = m1[-LOOKBACK_MINS:]
            touched = any((min(c[2], c[1]) <= cprice <= max(c[2], c[1])) for c in recent)
            if touched:
                return {
                    "side": "SHORT",
                    "cluster": na,
                    "total": ask_total,
                    "dom": ask_dom,
                    "cprice": cprice,
                }
    return None


def detect_bounce(product_id: str, last_price: float, m1: List[List[float]], obook: dict, bin_width: float):
    if len(m1) < max(LOOKBACK_MINS, 3):
        return None
    _, low, high, open_, close, _ = m1[-1]
    bids = obook.get("bids", [])
    asks = obook.get("asks", [])
    bid_clusters, bid_total, bid_dom = build_clusters(bids, last_price, "bids", bin_width)
    ask_clusters, ask_total, ask_dom = build_clusters(asks, last_price, "asks", bin_width)

    nb = nearest_cluster(bid_clusters, last_price, "bids")
    na = nearest_cluster(ask_clusters, last_price, "asks")

    # Bid bounce → LONG (tap support and hold; do NOT overrun)
    if nb is not None:
        cprice = nb.price_bin
        overrun_bar = close >= cprice * (1 + OVERRUN_MARGIN_PCT)
        touched = (low <= cprice <= high)
        near_close = close >= cprice * (1 + TOUCH_MARGIN_PCT)
        green = close > open_
        if touched and near_close and not overrun_bar and green:
            return {
                "side": "LONG",
                "cluster": nb,
                "total": bid_total,
                "dom": bid_dom,
                "cprice": cprice,
            }

    # Ask bounce → SHORT (tap resistance and hold; do NOT overrun)
    if na is not None:
        cprice = na.price_bin
        overrun_bar = close <= cprice * (1 - OVERRUN_MARGIN_PCT)
        touched = (low <= cprice <= high)
        near_close = close <= cprice * (1 - TOUCH_MARGIN_PCT)
        red = close < open_
        if touched and near_close and not overrun_bar and red:
            return {
                "side": "SHORT",
                "cluster": na,
                "total": ask_total,
                "dom": ask_dom,
                "cprice": cprice,
            }

    return None

# -------------------- TELEGRAM & LOGGING -----------------------------

def send_telegram(msg: str):
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        payload = {"chat_id": TELEGRAM_CHAT_ID, "text": msg, "parse_mode": "Markdown"}
        requests.post(url, json=payload, timeout=10)
    except Exception:
        pass


def log_signal(sig: Signal):
    header = [
        "ts_iso","product","type","strength","reasons","cluster_price","entry","tp","sl",
        "last_price","ema_long_ok","ema_short_ok","stoch_long_ok","stoch_short_ok","K","D",
        "atr","bin_width","ovr_margin_pct","touch_margin_pct","lookback_mins"
    ]
    exists = False
    try:
        with open(LOG_PATH, "r", newline="") as _:
            exists = True
    except FileNotFoundError:
        exists = False
    with open(LOG_PATH, "a", newline="") as f:
        w = csv.writer(f)
        if not exists:
            w.writerow(header)
        ctx = sig.context
        w.writerow([
            sig.ts_iso, sig.product_id, sig.signal_type, sig.strength, ", ".join(sig.reasons), f"{sig.cluster_price:.8f}", f"{sig.entry:.8f}", f"{sig.tp:.8f}", f"{sig.sl:.8f}",
            f"{ctx.get('last_price', 0):.8f}", ctx.get('ema_long_ok'), ctx.get('ema_short_ok'), ctx.get('stoch_long_ok'), ctx.get('stoch_short_ok'),
            f"{ctx.get('K', 0):.2f}", f"{ctx.get('D', 0):.2f}", f"{ctx.get('atr', 0):.8f}", f"{ctx.get('bin_width', 0):.8f}",
            OVERRUN_MARGIN_PCT, TOUCH_MARGIN_PCT, LOOKBACK_MINS
        ])

# ------------------------- MAIN LOGIC --------------------------------

def format_price(p: float) -> str:
    if p >= 100:
        return f"{p:.2f}"
    if p >= 1:
        return f"{p:.4f}"
    if p >= 0.1:
        return f"{p:.5f}"
    return f"{p:.8f}"


def rating_from_score(score: int) -> str:
    if score >= 75:
        return "⭐⭐⭐ Strong"
    if score >= 50:
        return "⭐⭐ Medium"
    return "⭐ Weak"


def build_signal(product_id: str, side: str, kind: str, cluster_price: float, last_price: float,
                 ema_long_ok: bool, ema_short_ok: bool, stoch_long_ok: bool, stoch_short_ok: bool,
                 K: float, D: float, atr: float, bin_width: float,
                 strength: int, reasons: List[str]) -> Signal:
    entry = last_price
    if side == "LONG":
        tp = entry + TP_ATR_MULT * atr
        sl = entry - SL_ATR_MULT * atr
    else:
        tp = entry - TP_ATR_MULT * atr
        sl = entry + SL_ATR_MULT * atr

    return Signal(
        product_id=product_id,
        signal_type=f"{kind}_{side}",
        cluster_price=cluster_price,
        entry=entry,
        tp=tp,
        sl=sl,
        ts_iso=now_iso(),
        strength=strength,
        reasons=reasons,
        context={
            "last_price": last_price,
            "ema_long_ok": ema_long_ok,
            "ema_short_ok": ema_short_ok,
            "stoch_long_ok": stoch_long_ok,
            "stoch_short_ok": stoch_short_ok,
            "K": K,
            "D": D,
            "atr": atr,
            "bin_width": bin_width,
        }
    )


def send_and_log(sig: Signal):
    side = "LONG" if sig.signal_type.endswith("LONG") else "SHORT"
    kind = "OVERRUN" if sig.signal_type.startswith("OVERRUN") else "BOUNCE"
    ema_ok = sig.context["ema_long_ok"] if side == "LONG" else sig.context["ema_short_ok"]
    stoch_ok = sig.context["stoch_long_ok"] if side == "LONG" else sig.context["stoch_short_ok"]
    K, D = sig.context["K"], sig.context["D"]
    atr = sig.context["atr"]
    bin_w = sig.context["bin_width"]

    msg = (
        f"*Whale {kind}* {sig.product_id} — *{side}*\n"
        f"{rating_from_score(sig.strength)}  (Score: {sig.strength}/100)\n"
        f"Top: {sig.reasons[0]}; {sig.reasons[1]}\n"
        f"Cluster: `{format_price(sig.cluster_price)}`\n"
        f"Entry: `{format_price(sig.entry)}`\n"
        f"TP: `{format_price(sig.tp)}`  SL: `{format_price(sig.sl)}`\n"
        f"Filters — EMA200 ok: `{ema_ok}`  Stoch dir ok: `{stoch_ok}` (K={K:.1f}, D={D:.1f})\n"
        f"ATR(15m): `{format_price(atr)}`  Bin: `{format_price(bin_w)}`\n"
        f"Time: `{sig.ts_iso}`"
    )
    send_telegram(msg)
    log_signal(sig)


def run_once(product_id: str, last_alert_ts: Dict[str, float]) -> Optional[Signal]:
    last_price = get_ticker(product_id)
    bin_width = last_price * BIN_WIDTH_PCT
    m1 = get_candles(product_id, 60, max(LOOKBACK_MINS + 2, 50))
    obook = get_order_book(product_id)

    # Filters
    ema_long_ok, ema_short_ok = (True, True)
    if USE_TREND_FILTER:
        h1 = get_candles(product_id, 3600, 240)
        closes = [c[4] for c in h1]
        e200 = ema(closes, 200)[-1]
        ema_long_ok, ema_short_ok = (closes[-1] >= e200, closes[-1] < e200)
    else:
        h1 = get_candles(product_id, 3600, 240)
        closes = [c[4] for c in h1]
        e200 = ema(closes, 200)[-1]
    stoch_long_ok, stoch_short_ok, K, D = (True, True, 50.0, 50.0)
    if USE_STOCH_DIR_FILTER:
        K_series, D_series = stoch_rsi([c[4] for c in h1])
        K, D = K_series[-1], D_series[-1]
        stoch_long_ok = (K > D and K < 60)
        stoch_short_ok = (K < D and K > 40)

    def ok_for(side: str) -> bool:
        return (ema_long_ok and stoch_long_ok) if side == "LONG" else (ema_short_ok and stoch_short_ok)

    atr = atr_from_candles(get_candles(product_id, 900, ATR_PERIOD + 32), period=ATR_PERIOD)

    # Try overrun and/or bounce depending on mode
    picked = None
    kind = None
    if ALERT_MODE in ("overrun", "both"):
        res = detect_overrun(product_id, last_price, m1, obook, bin_width)
        if res and ok_for(res["side"]):
            picked = res
            kind = "OVERRUN"
    if picked is None and ALERT_MODE in ("bounce", "both"):
        res = detect_bounce(product_id, last_price, m1, obook, bin_width)
        if res and ok_for(res["side"]):
            picked = res
            kind = "BOUNCE"

    if picked is None:
        return None

    side = picked["side"]
    cluster = picked["cluster"]
    total = picked["total"]
    dom = picked["dom"]
    cprice = picked["cprice"]

    # Build strength
    strength, reasons = build_strength_score(
        kind, side,
        last_price=last_price,
        ema200=e200,
        atr=atr,
        K=K,
        D=D,
        m1=m1,
        cluster=cluster,
        total_side_notional=total,
        dominance_ratio=dom,
        cprice=cprice,
    )

    # Cooldown per product+type bucketed by strength tier
    tier = "S" if strength >= 75 else ("M" if strength >= 50 else "W")
    key = f"{product_id}:{kind}:{side}:{tier}"
    now_t = time.time()
    if key in last_alert_ts and (now_t - last_alert_ts[key] < COOLDOWN_MINUTES*60):
        return None

    sig = build_signal(product_id, side, kind, cprice, last_price,
                       ema_long_ok, ema_short_ok, stoch_long_ok, stoch_short_ok,
                       K, D, atr, bin_width, strength, reasons)

    send_and_log(sig)
    last_alert_ts[key] = now_t
    return sig


def main_loop():
    last_alert_ts: Dict[str, float] = {}
    print("Starting Whale Overrun & Bounce Alert loop with scoring…")
    while True:
        for pid in PRODUCTS:
            try:
                sig = run_once(pid, last_alert_ts)
                if sig:
                    print(f"{sig.ts_iso} ALERT {sig.product_id} {sig.signal_type} score={sig.strength} entry={format_price(sig.entry)} tp={format_price(sig.tp)} sl={format_price(sig.sl)}")
                else:
                    print(f"{datetime.utcnow().isoformat()} no-signal {pid}")
            except Exception as e:
                print(f"Error on {pid}: {e}")
        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main_loop()
