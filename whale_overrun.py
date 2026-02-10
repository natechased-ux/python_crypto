"""
Whale Overrun Momentum Alert (Coinbase, REST polling)
----------------------------------------------------
Detects "skipped liquidity" (overrun) events where price blows through a large
whale bid/ask cluster and continues with momentum. Sends Telegram alerts.

Design goals:
- Simple, dependency-light (REST polling instead of websockets)
- Same-direction entries after liquidity overrun (momentum continuation)
- Optional filters: 200 EMA trend, Stoch RSI direction
- ATR-based TP/SL, per-coin cooldown, CSV logging for ML later

Notes:
- Coinbase Exchange REST endpoints used. Adjust BASE_URL if needed.
- Uses 1m candles for overrun detection, 1H for trend/confirmation, 15m for ATR.
- Cluster binning: round(price / bin_width) * bin_width, per user's prior method.
- This is a runnable template; you may want to add retries/backoff and better error handling.

Python >=3.9 recommended.
"""

import time
import math
import csv
import hmac
import hashlib
import base64
import requests
from datetime import datetime, timezone, timedelta
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

POLL_SECONDS = 30  # How often to poll REST
COOLDOWN_MINUTES = 30

# Order book & cluster settings
ORDERBOOK_LEVEL = 2  # aggregated book
BIN_WIDTH_PCT = 0.0015  # 0.15% of price per bin (rounded using round(p/bin)*bin)
PCT_WITHIN_PRICE = 0.02   # prefilter: only consider book levels within ±2% of last price
TOP_CLUSTER_PERCENTILE = 0.9  # keep top 10% by notional size per side
MIN_CLUSTER_NOTIONAL = 5_000  # ignore tiny clusters

# Overrun detection
OVERRUN_MARGIN_PCT = 0.0005  # must close this % beyond the cluster level to count as "skipped"
LOOKBACK_MINS = 2            # require the cluster price to have been inside [open, close] within this many minutes prior

# Filters (toggle True/False)
USE_TREND_FILTER = True     # 200 EMA on 1H: only long above / short below
USE_STOCH_DIR_FILTER = True # 1H Stoch RSI slope/position filter

# Risk (ATR on 15m)
ATR_PERIOD = 14
TP_ATR_MULT = 2.0
SL_ATR_MULT = 1.5

# Telegram (user provided credentials)
TELEGRAM_BOT_TOKEN = "8177096945:AAHTg5nxVTA6hcPkAL4MLAAHPsZ1at7Ywmw"
TELEGRAM_CHAT_ID = "7967738614"

# CSV logging
LOG_PATH = "live_overrun_alerts.csv"

# --------------------------------------------------------------------

@dataclass
class Cluster:
    price_bin: float
    total_size: float  # base size
    notional: float    # size * price_bin

@dataclass
class OverrunSignal:
    product_id: str
    side: str  # "LONG" or "SHORT"
    cluster_price: float
    entry: float
    tp: float
    sl: float
    ts_iso: str
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
    """
    Returns list of candles: [time, low, high, open, close, volume]
    granularity in seconds. Coinbase supports 60, 300, 900, 3600, 21600, 86400.
    """
    params = {"granularity": granularity}
    data = get_json(f"{BASE_URL}/products/{product_id}/candles", params=params)
    # Coinbase returns in reverse chronological order
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
    # compute RSI first
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
    # smooth
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
        t, low, high, op, close, vol = candles[i]
        t0, low0, high0, op0, close0, vol0 = candles[i-1]
        trs.append(true_range(high, low, close0))
    if len(trs) < period:
        return sum(trs)/max(1, len(trs)) if trs else 0.0
    # Wilder's smoothing
    atr = sum(trs[:period]) / period
    for tr in trs[period:]:
        atr = (atr * (period - 1) + tr) / period
    return atr

# -------------------- ORDER BOOK CLUSTERING --------------------------

def get_order_book(product_id: str) -> dict:
    params = {"level": ORDERBOOK_LEVEL}
    return get_json(f"{BASE_URL}/products/{product_id}/book", params=params)


def bin_price(price: float, bin_width: float) -> float:
    return round(price / bin_width) * bin_width


def build_clusters(levels: List[List[str]], last_price: float, side: str, bin_width: float) -> List[Cluster]:
    # levels: [price, size, num_orders]
    pts: Dict[float, float] = {}
    for p_str, sz_str, _ in levels:
        p = float(p_str)
        if abs(p - last_price)/last_price > PCT_WITHIN_PRICE:
            continue
        sz = float(sz_str)
        if sz <= 0:
            continue
        b = bin_price(p, bin_width)
        pts[b] = pts.get(b, 0.0) + sz
    clusters: List[Cluster] = []
    for b, size in pts.items():
        notional = b * size
        if notional >= MIN_CLUSTER_NOTIONAL:
            clusters.append(Cluster(price_bin=b, total_size=size, notional=notional))
    # keep top percentile by notional
    if not clusters:
        return []
    notionals = sorted([c.notional for c in clusters])
    cutoff_idx = max(0, int(len(notionals) * TOP_CLUSTER_PERCENTILE) - 1)
    cutoff = notionals[cutoff_idx] if notionals else 0
    clusters = [c for c in clusters if c.notional >= cutoff]
    # Sort: bids descending price, asks ascending price
    reverse = True if side == "bids" else False
    clusters.sort(key=lambda c: c.price_bin, reverse=reverse)
    return clusters


def nearest_cluster(clusters: List[Cluster], last_price: float, side: str) -> Optional[Cluster]:
    if not clusters:
        return None
    if side == "bids":
        # nearest below price
        below = [c for c in clusters if c.price_bin <= last_price]
        return below[0] if below else None
    else:
        # nearest above price
        above = [c for c in clusters if c.price_bin >= last_price]
        return above[0] if above else None

# -------------------- OVERRUN DETECTION ------------------------------

def detect_overrun(product_id: str,
                    last_price: float,
                    m1_candles: List[List[float]],
                    obook: dict,
                    bin_width: float) -> Optional[Tuple[str, float]]:
    """
    Returns (side, cluster_price) where side is "LONG" (bid overrun) or "SHORT" (ask overrun),
    or None if no signal.

    Logic (bid overrun => LONG):
    - Identify nearest whale BID cluster below current price
    - If the latest 1m candle CLOSED at least OVERRUN_MARGIN_PCT above that cluster price, and
      within the last LOOKBACK_MINS minutes the candle range included the cluster price (i.e., price was at/near it),
      we consider it "skipped" liquidity and go LONG.
    Symmetric for ask overrun => SHORT.
    """
    if len(m1_candles) < max(LOOKBACK_MINS, 3):
        return None

    close = m1_candles[-1][4]

    bids = obook.get("bids", [])
    asks = obook.get("asks", [])
    bid_clusters = build_clusters(bids, last_price, "bids", bin_width)
    ask_clusters = build_clusters(asks, last_price, "asks", bin_width)

    nb = nearest_cluster(bid_clusters, last_price, "bids")
    na = nearest_cluster(ask_clusters, last_price, "asks")

    # Check bid overrun (LONG)
    if nb is not None:
        cprice = nb.price_bin
        if close >= cprice * (1 + OVERRUN_MARGIN_PCT):
            # Was price interacting with that level in the last LOOKBACK_MINS?
            recent = m1_candles[-LOOKBACK_MINS:]
            touched = any((min(c[2], c[1]) <= cprice <= max(c[2], c[1])) or (c[3] <= cprice <= c[4]) for c in recent)
            if touched:
                return ("LONG", cprice)

    # Check ask overrun (SHORT)
    if na is not None:
        cprice = na.price_bin
        if close <= cprice * (1 - OVERRUN_MARGIN_PCT):
            recent = m1_candles[-LOOKBACK_MINS:]
            touched = any((min(c[2], c[1]) <= cprice <= max(c[2], c[1])) or (c[3] >= cprice >= c[4]) for c in recent)
            if touched:
                return ("SHORT", cprice)

    return None

# -------------------- FILTERS & RISK --------------------------------

def ema200_trend_ok(product_id: str) -> Optional[bool]:
    h1 = get_candles(product_id, 3600, 240)
    closes = [c[4] for c in h1]
    e200 = ema(closes, 200)[-1]
    last = closes[-1]
    # long if above, short if below -> we just return None here and decide per side
    return (last >= e200, last < e200)


def stoch_dir_ok(product_id: str) -> Tuple[bool, bool, float, float]:
    h1 = get_candles(product_id, 3600, 200)
    closes = [c[4] for c in h1]
    K, D = stoch_rsi(closes)
    k, d = K[-1], D[-1]
    long_ok = (k > d and k < 60)  # rising from lower region
    short_ok = (k < d and k > 40) # falling from upper mid
    return long_ok, short_ok, k, d


def atr_15m(product_id: str, period: int = ATR_PERIOD) -> float:
    m15 = get_candles(product_id, 900, period + 32)
    return atr_from_candles(m15, period=period)

# -------------------- TELEGRAM & LOGGING -----------------------------

def send_telegram(msg: str):
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        payload = {"chat_id": TELEGRAM_CHAT_ID, "text": msg, "parse_mode": "Markdown"}
        requests.post(url, json=payload, timeout=10)
    except Exception:
        pass


def log_signal(sig: OverrunSignal):
    header = [
        "ts_iso","product","side","cluster_price","entry","tp","sl",
        "last_price","ema_long_ok","ema_short_ok","stoch_long_ok","stoch_short_ok","K","D",
        "atr","bin_width","ovr_margin_pct","lookback_mins"
    ]
    exists = False
    try:
        with open(LOG_PATH, "r", newline="") as f:
            exists = True
    except FileNotFoundError:
        exists = False
    with open(LOG_PATH, "a", newline="") as f:
        w = csv.writer(f)
        if not exists:
            w.writerow(header)
        ctx = sig.context
        w.writerow([
            sig.ts_iso, sig.product_id, sig.side, f"{sig.cluster_price:.8f}", f"{sig.entry:.8f}", f"{sig.tp:.8f}", f"{sig.sl:.8f}",
            f"{ctx.get('last_price', 0):.8f}", ctx.get('ema_long_ok'), ctx.get('ema_short_ok'), ctx.get('stoch_long_ok'), ctx.get('stoch_short_ok'),
            f"{ctx.get('K', 0):.2f}", f"{ctx.get('D', 0):.2f}", f"{ctx.get('atr', 0):.8f}", f"{ctx.get('bin_width', 0):.8f}",
            OVERRUN_MARGIN_PCT, LOOKBACK_MINS
        ])

# ------------------------- MAIN LOOP ---------------------------------

def format_price(p: float) -> str:
    if p >= 100:
        return f"{p:.2f}"
    if p >= 1:
        return f"{p:.4f}"
    if p >= 0.1:
        return f"{p:.5f}"
    return f"{p:.8f}"


def run_once(product_id: str, last_alert_ts: Dict[str, float]) -> Optional[OverrunSignal]:
    last_price = get_ticker(product_id)

    # Bin width tied to price
    bin_width = last_price * BIN_WIDTH_PCT

    # 1m candles
    m1 = get_candles(product_id, 60, max(LOOKBACK_MINS + 2, 50))

    # Order book snapshot
    obook = get_order_book(product_id)

    # Detect overrun
    res = detect_overrun(product_id, last_price, m1, obook, bin_width)
    if not res:
        return None

    side, cluster_price = res

    # Cooldown
    now_t = time.time()
    key = f"{product_id}:{side}"
    if key in last_alert_ts and (now_t - last_alert_ts[key] < COOLDOWN_MINUTES*60):
        return None

    # Filters
    ema_long_ok, ema_short_ok = True, True
    if USE_TREND_FILTER:
        ema_ok = ema200_trend_ok(product_id)
        if ema_ok is not None:
            ema_long_ok, ema_short_ok = ema_ok

    stoch_long_ok, stoch_short_ok, K, D = True, True, 50.0, 50.0
    if USE_STOCH_DIR_FILTER:
        stoch_long_ok, stoch_short_ok, K, D = stoch_dir_ok(product_id)

    if side == "LONG" and not (ema_long_ok and stoch_long_ok):
        return None
    if side == "SHORT" and not (ema_short_ok and stoch_short_ok):
        return None

    # Risk: ATR-based TP/SL
    atr = atr_15m(product_id, ATR_PERIOD)
    entry = last_price
    if side == "LONG":
        tp = entry + TP_ATR_MULT * atr
        sl = entry - SL_ATR_MULT * atr
    else:
        tp = entry - TP_ATR_MULT * atr
        sl = entry + SL_ATR_MULT * atr

    sig = OverrunSignal(
        product_id=product_id,
        side=side,
        cluster_price=cluster_price,
        entry=entry,
        tp=tp,
        sl=sl,
        ts_iso=now_iso(),
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

    # Build and send Telegram message
    msg = (
        f"*Whale Overrun* {sig.product_id} — *{sig.side}*\n"
        f"Cluster: `{format_price(sig.cluster_price)}`\n"
        f"Entry: `{format_price(sig.entry)}`\n"
        f"TP: `{format_price(sig.tp)}`  SL: `{format_price(sig.sl)}`\n"
        f"Filters — EMA200 ok: `{ema_long_ok if side=='LONG' else ema_short_ok}`  "
        f"Stoch dir ok: `{stoch_long_ok if side=='LONG' else stoch_short_ok}` (K={K:.1f}, D={D:.1f})\n"
        f"ATR(15m): `{format_price(atr)}`  Bin: `{format_price(bin_width)}`\n"
        f"Time: `{sig.ts_iso}`"
    )
    send_telegram(msg)

    # Log
    log_signal(sig)

    # update cooldown
    last_alert_ts[key] = now_t

    return sig


def main_loop():
    last_alert_ts: Dict[str, float] = {}
    print("Starting Whale Overrun Momentum Alert loop…")
    while True:
        for pid in PRODUCTS:
            try:
                sig = run_once(pid, last_alert_ts)
                if sig:
                    print(f"{sig.ts_iso} ALERT {sig.product_id} {sig.side} entry={format_price(sig.entry)} tp={format_price(sig.tp)} sl={format_price(sig.sl)}")
                else:
                    print(f"{datetime.utcnow().isoformat()} no-signal {pid}")
            except Exception as e:
                print(f"Error on {pid}: {e}")
        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main_loop()
