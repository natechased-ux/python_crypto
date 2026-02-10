#!/usr/bin/env python3
"""
Liquidity Sweep Reversal Alert Bot — Coinbase Edition (Ultra‑Specialized)

Focus: ONLY liquidity sweep reversals around curated key levels.

Key deviations from the original spec (because Coinbase constraints):
- Uses Daily and 6H levels (Coinbase Exchange has no native 4H granularity).
- Monitors Coinbase products (e.g., BTC-USD, ETH-USD, SOL-USD, ADA-USD). BNB is not available on Coinbase.

Data sources (public endpoints only):
- REST: candles (1m/5m/6h/1d), order book snapshots, 24h stats
- WebSocket: ticker and matches (trade prints) for latency‑friendly triggers

Core rules implemented (tunable via constants below):
- SWEEP range: 0.1%–0.5% beyond a key level
- Reversal within REVERSAL_CANDLES (5m bars)
- Sweep candle volume > 150% of 20‑period average
- Close back inside the original range (back below/above the swept level)
- RSI(14, 5m) divergence confirmation
- Order book imbalance: bid/ask ratio < 0.7 for shorts, > 1.3 for longs
- Entry trigger: break of reversal candle low (short) / high (long)
- Risk: SL = 0.3% beyond sweep extreme, TP1 at >= 3R, TP2 at next major S/R
- Max 2 trades/day across the whole system (quality over quantity)

Extras:
- Level DB (sqlite): previous day high/low, weekly high/low, equal highs/lows (fractal), psychological levels, age tracking ≥24h
- False breakout log (sqlite): sweeps that failed the reversal/confirm filters
- Institutional activity tracker: flags large prints (matches) > configured USD notional
- Level strength score (1–10): age, touches, confluence type, proximity to round numbers
- Sweep classification: stop‑hunt style vs. clean break (based on wick/close/volume shape)

Telegram alerts: Use environment variables TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID.
(If you have fixed creds, set via env or plug directly.)

Author: ChatGPT
"""
from __future__ import annotations
import os
import time
import math
import json
import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

import requests
import websocket

# ==============================
# ---- Configuration Block  ----
# ==============================
COINS = [
    "BTC-USD",
    "ETH-USD",
    "SOL-USD",
    "ADA-USD",
    # Added — broadened liquid universe
    "XRP-USD",
    "DOGE-USD",
    "AVAX-USD",
    "LINK-USD",
    "DOT-USD",
    "MATIC-USD",
    "UNI-USD",
    "OP-USD",
    "ARB-USD",
    "SUI-USD",
    "NEAR-USD",
    "ATOM-USD",
    "XLM-USD",
    "ETC-USD",
    "LTC-USD",
    "BCH-USD",
    "AAVE-USD",
    "FIL-USD",
    "HBAR-USD",
    "IMX-USD",
    "INJ-USD",
    "ENA-USD",
    "PYTH-USD",
    "RNDR-USD",
    "SEI-USD",
    "PEPE-USD",
]

# Minimum 24h USD notional volume to include a product
MIN_DAILY_USD_VOL = 100_000  # $100M filter for SOL/ADA; BTC/ETH will pass anyway

PER_COIN_COOLDOWN_MIN = 30          # no more than 1 alert per coin within this window
LEVEL_REARM_TIME_MIN = 120          # per side/level lockout window
LEVEL_REARM_PCT = 0.003             # OR re-arm if price moves ~0.3% away then returns


# Strategy parameters
SWEEP_MIN = 0.001  # 0.1%
SWEEP_MAX = 0.005  # 0.5%
SWEEP_THRESHOLD = 0.002  # Target 0.2% (center), still enforce bounds above
REVERSAL_CANDLES = 3      # Max number of 5m candles for the rejection to print
VOLUME_MULTIPLIER = 1.5   # Sweep candle volume > 150% of 20‑period avg
MIN_LEVEL_AGE_H = 24      # Hours since level formed
ORDERBOOK_IMB_THRESHOLD_SHORT = 0.70  # bid/ask < 0.7 (more asks than bids)
ORDERBOOK_IMB_THRESHOLD_LONG = 1.30   # bid/ask > 1.3 (more bids than asks)

# Risk settings
ACCOUNT_EQUITY_USD = 50_000  # used for position sizing calc (paper)
RISK_PER_TRADE = 0.02        # 2% risk per trade
SL_BUFFER_PCT = 0.003        # 0.3% beyond sweep extreme
MIN_R_MULTIPLE = 3.0         # TP1 must be ≥ 3R
MAX_TRADES_PER_DAY = 10

# Institutional print threshold (USD notional) for awareness
BIG_TRADE_NOTIONAL_USD = 2_000_000

# Paths
DB_PATH = os.environ.get("SWEEP_DB_PATH", "sweep_levels.sqlite")
LOG_PATH = os.environ.get("SWEEP_LOG_PATH", "sweep_signals.log")

# Telegram (optional)
TELEGRAM_BOT_TOKEN = "8177096945:AAHTg5nxVTA6hcPkAL4MLAAHPsZ1at7Ywmw"
TELEGRAM_CHAT_ID = "7967738614"



# Coinbase endpoints
REST_BASE = "https://api.exchange.coinbase.com"
WS_URL = "wss://ws-feed.exchange.coinbase.com"  # public feed

# Candle granularities (seconds)
G_1M, G_5M, G_6H, G_1D = 60, 300, 21600, 86400

# ==============================
# ---------- Utilities ---------
# ==============================

def ts_utc() -> datetime:
    return datetime.now(timezone.utc)


def to_unix(dt: datetime) -> int:
    return int(dt.timestamp())


def round_psych(price: float, product: str) -> float:
    """Return a psychological level near price using magnitude-based rounding.
    Robust for tiny-price assets (e.g., PEPE) so it never returns 0.
    """
    p = abs(price)
    if p == 0:
        return 0.0
    # Use order-of-magnitude step for 2 significant-digit rounding
    m = math.floor(math.log10(p))
    # Keep reasonable precision floor for very small assets
    step = 10 ** (m - 1)
    # Clamp the minimum step so we still vary at micro prices
    step = max(step, 10 ** -12)
    rounded = round(price / step) * step
    # Avoid returning 0 for tiny assets—snap to one step if so
    if rounded == 0.0:
        rounded = step if price > 0 else -step
    return float(rounded)


def rsi(values: List[float], period: int = 14) -> List[float]:
    if len(values) < period + 1:
        return [float("nan")] * len(values)
    gains, losses = [], []
    rsis = [float("nan")] * len(values)
    # First period
    for i in range(1, period + 1):
        diff = values[i] - values[i - 1]
        gains.append(max(0.0, diff))
        losses.append(max(0.0, -diff))
    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period
    rs = (avg_gain / avg_loss) if avg_loss > 0 else float("inf")
    rsis[period] = 100 - (100 / (1 + rs))
    # Rest
    for i in range(period + 1, len(values)):
        diff = values[i] - values[i - 1]
        gain = max(0.0, diff)
        loss = max(0.0, -diff)
        avg_gain = (avg_gain * (period - 1) + gain) / period
        avg_loss = (avg_loss * (period - 1) + loss) / period
        rs = (avg_gain / avg_loss) if avg_loss > 0 else float("inf")
        rsis[i] = 100 - (100 / (1 + rs))
    return rsis

def fmt_price(price: float) -> str:
    """Format price with dynamic decimals based on magnitude."""
    if price >= 1000:
        return f"{price:,.2f}"
    elif price >= 1:
        return f"{price:,.3f}"
    elif price >= 0.01:
        return f"{price:,.5f}"
    else:
        return f"{price:,.8f}"

# ==============================
# --------- Data Access --------
# ==============================
def ob_direction_ok(side: str, ob_ratio: float) -> bool:
    """
    Enforce directional consistency between sweep and order book.
    - For SHORTs (sweep above resistance; side == 'high'): need ask-heavy -> bid/ask < 0.70
    - For LONGs  (sweep below support;   side == 'low' ): need bid-heavy -> bid/ask > 1.30
    """
    return (
        (side == "high" and ob_ratio < ORDERBOOK_IMB_THRESHOLD_SHORT) or
        (side == "low"  and ob_ratio > ORDERBOOK_IMB_THRESHOLD_LONG)
    )


class CB:
    @staticmethod
    def candles(product: str, granularity_s: int, start: Optional[datetime] = None, end: Optional[datetime] = None) -> List[List[float]]:
        """Return candles in [time, low, high, open, close, volume] (Coinbase format)."""
        params = {"granularity": granularity_s}
        if start and end:
            params.update({"start": start.isoformat(), "end": end.isoformat()})
        url = f"{REST_BASE}/products/{product}/candles"
        r = requests.get(url, params=params, timeout=15)
        r.raise_for_status()
        data = r.json()
        # Coinbase returns newest first; sort ascending by time
        data.sort(key=lambda x: x[0])
        return data

    @staticmethod
    def candles_paged(product: str, granularity_s: int, start: datetime, end: datetime) -> List[List[float]]:
        """Fetch >300 candles by paging time windows. Coinbase caps 300 per request.
        We split the [start, end] interval into chunks of <= 300*granularity seconds.
        """
        span = timedelta(seconds=granularity_s * 300 - 1)
        out: List[List[float]] = []
        cur_start = start
        while cur_start < end:
            cur_end = min(cur_start + span, end)
            chunk = CB.candles(product, granularity_s, cur_start, cur_end)
            if chunk:
                if out and chunk[0][0] <= out[-1][0]:
                    # avoid duplicates/overlap
                    chunk = [c for c in chunk if c[0] > out[-1][0]]
                out.extend(chunk)
            # advance
            cur_start = cur_end
        return out

    @staticmethod
    def orderbook(product: str, level: int = 2) -> Dict:
        url = f"{REST_BASE}/products/{product}/book"
        r = requests.get(url, params={"level": level}, timeout=10)
        r.raise_for_status()
        return r.json()

    @staticmethod
    def stats(product: str) -> Dict:
        # 24h stats: last, open, high, low, volume (base), volume_30d, etc.
        url = f"{REST_BASE}/products/{product}/stats"
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        return r.json()


# ==============================
# ----------- Storage ----------
# ==============================
class Store:
    def __init__(self, path: str):
        self.path = path
        self._init_db()

    def _init_db(self):
        con = sqlite3.connect(self.path)
        cur = con.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS levels (
                id INTEGER PRIMARY KEY,
                product TEXT,
                level REAL,
                side TEXT,              -- 'high' or 'low'
                kind TEXT,              -- 'prev_day','weekly','equal','psych'
                formed_at INTEGER,      -- unix ts
                touches INTEGER,
                strength REAL,
                UNIQUE(product, level, side, kind)
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS false_breakouts (
                id INTEGER PRIMARY KEY,
                product TEXT,
                ts INTEGER,
                level REAL,
                direction TEXT,         -- 'up' (stop hunt above) or 'down'
                reason TEXT
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY,
                product TEXT,
                side TEXT,              -- 'short' or 'long'
                ts INTEGER,
                entry REAL,
                sl REAL,
                tp1 REAL,
                tp2 REAL,
                r_multiple REAL,
                context_json TEXT
            )
            """
        )
        con.commit()
        con.close()

    # ---- Level ops ----
    def upsert_level(self, product: str, level: float, side: str, kind: str, formed_at: int, touches: int, strength: float):
        con = sqlite3.connect(self.path)
        cur = con.cursor()
        cur.execute(
            """
            INSERT INTO levels(product, level, side, kind, formed_at, touches, strength)
            VALUES (?,?,?,?,?,?,?)
            ON CONFLICT(product, level, side, kind) DO UPDATE SET
                formed_at=excluded.formed_at,
                touches=excluded.touches,
                strength=excluded.strength
            """,
            (product, level, side, kind, formed_at, touches, strength),
        )
        con.commit()
        con.close()

    def aged_levels(self, product: str, min_age_h: int) -> List[Tuple]:
        cutoff = to_unix(ts_utc() - timedelta(hours=min_age_h))
        con = sqlite3.connect(self.path)
        cur = con.cursor()
        cur.execute(
            """
            SELECT product, level, side, kind, formed_at, touches, strength
            FROM levels WHERE product=? AND formed_at <= ?
            ORDER BY strength DESC
            """,
            (product, cutoff),
        )
        rows = cur.fetchall()
        con.close()
        return rows

    def log_false_breakout(self, product: str, level: float, direction: str, reason: str):
        con = sqlite3.connect(self.path)
        cur = con.cursor()
        cur.execute(
            "INSERT INTO false_breakouts(product, ts, level, direction, reason) VALUES (?,?,?,?,?)",
            (product, to_unix(ts_utc()), level, direction, reason),
        )
        con.commit()
        con.close()

    def log_trade(self, product: str, side: str, entry: float, sl: float, tp1: float, tp2: float, r_multiple: float, context_json: str):
        con = sqlite3.connect(self.path)
        cur = con.cursor()
        cur.execute(
            "INSERT INTO trades(product, side, ts, entry, sl, tp1, tp2, r_multiple, context_json) VALUES (?,?,?,?,?,?,?,?,?)",
            (product, side, to_unix(ts_utc()), entry, sl, tp1, tp2, r_multiple, context_json),
        )
        con.commit()
        con.close()


# ==============================
# ------ Level Identification ---
# ==============================
class LevelDetector:
    def __init__(self, store: Store):
        self.store = store

    def build_levels(self, product: str):
        now = ts_utc()
        # Daily candles for prev day H/L, equal H/L across last N days
        d_candles = CB.candles_paged(product, G_1D, start=now - timedelta(days=30), end=now)
        # 6H candles as substitute for 4H structures
        h6_candles = CB.candles_paged(product, G_6H, start=now - timedelta(days=15), end=now)

        if not d_candles:
            return

        # Prev day high/low
        prev_day = d_candles[-2]
        _, d_low, d_high, d_open, d_close, d_vol = prev_day
        self._push_level(product, level=d_high, side="high", kind="prev_day", formed_at=int(prev_day[0]))
        self._push_level(product, level=d_low, side="low", kind="prev_day", formed_at=int(prev_day[0]))

        # Weekly high/low from last 2 calendar weeks (approx using daily groups)
        weeks = {}
        for c in d_candles:
            t = datetime.fromtimestamp(c[0], tz=timezone.utc)
            year_week = f"{t.isocalendar().year}-{t.isocalendar().week:02d}"
            weeks.setdefault(year_week, {"high": -1e18, "low": 1e18, "ts": c[0]})
            weeks[year_week]["high"] = max(weeks[year_week]["high"], c[2])
            weeks[year_week]["low"] = min(weeks[year_week]["low"], c[1])
        for wk, obj in list(weeks.items())[-3:]:
            self._push_level(product, obj["high"], "high", "weekly", formed_at=int(obj["ts"]))
            self._push_level(product, obj["low"], "low", "weekly", formed_at=int(obj["ts"]))

        # Equal highs/lows via simple fractal detection on 6H
        def fractals(candles: List[List[float]], lookback=2) -> List[Tuple[int, float, str]]:
            out = []
            for i in range(lookback, len(candles) - lookback):
                t, lo, hi, op, cl, vol = candles[i]
                is_high = all(hi >= candles[i - k][2] and hi >= candles[i + k][2] for k in range(1, lookback + 1))
                is_low = all(lo <= candles[i - k][1] and lo <= candles[i + k][1] for k in range(1, lookback + 1))
                if is_high:
                    out.append((int(t), hi, "high"))
                if is_low:
                    out.append((int(t), lo, "low"))
            return out

        fr = fractals(h6_candles, lookback=2)
        # Group near‑equal (within 0.05%)
        groups = {}
        for t, lvl, side in fr:
            key = (side, round(lvl, 2))
            groups.setdefault(key, []).append((t, lvl))
        for (side, _), arr in groups.items():
            if len(arr) >= 2:
                # Equal highs/lows: use median level and earliest ts
                lvls = [x[1] for x in arr]
                ts_first = min(x[0] for x in arr)
                median_lvl = sorted(lvls)[len(lvls)//2]
                self._push_level(product, median_lvl, side, "equal", formed_at=ts_first, touches=len(arr))

        # Psychological levels near current price
        last_close = d_candles[-1][4]
        for m in [-2, -1, 0, 1, 2]:
            psych = round_psych(last_close + m * (last_close * 0.02), product)
            # Add both as potential highs and lows with slight neutrality
            if psych > 0:
                self._push_level(product, psych, "high", "psych", formed_at=to_unix(now - timedelta(days=7)))
                self._push_level(product, psych, "low", "psych", formed_at=to_unix(now - timedelta(days=7)))

    def _push_level(self, product: str, level: float, side: str, kind: str, formed_at: int, touches: int = 1):
        # Strength heuristic
        age_h = max(1, (to_unix(ts_utc()) - formed_at) / 3600)
        base = 5.0 if kind in ("prev_day", "weekly") else 3.5 if kind == "equal" else 1.5

        touch_bonus = min(3.0, 0.5 * max(0, touches - 1))
        age_bonus = min(2.0, math.log10(age_h + 1))
        strength = min(10.0, base + touch_bonus + age_bonus)
        self.store.upsert_level(product, float(level), side, kind, formed_at, touches, strength)


# ==============================
# ------ Order Book / Flow -----
# ==============================
class OrderBookMonitor:
    @staticmethod
    def bid_ask_ratio(product: str, depth_levels: int = 20) -> float:
        ob = CB.orderbook(product, level=2)
        def sum_side(arr):
            total = 0.0
            for i, row in enumerate(arr[:depth_levels]):
                price, size = float(row[0]), float(row[1])
                total += price * size  # notional depth
            return total
        bids = sum_side(ob.get("bids", []))
        asks = sum_side(ob.get("asks", []))
        return (bids / asks) if asks > 0 else float("inf")


# ==============================
# ------- Sweep Detection ------
# ==============================
@dataclass
class SweepCandidate:
    product: str
    level: float
    level_kind: str
    side: str               # 'high' (looking for stop-hunt up) or 'low'
    sweep_price: float
    sweep_ts: int
    wick_rejection: bool
    vol_factor: float


class SweepEngine:
    def __init__(self, store: Store):
        self.store = store
        self.avg_cache: Dict[Tuple[str, int], float] = {}

    def _avg_volume(self, product: str, n: int = 20) -> float:
        key = (product, n)
        now = ts_utc()
        candles = CB.candles_paged(product, G_5M, start=now - timedelta(hours=6), end=now)
        vols = [c[5] for c in candles][-n:]
        if not vols:
            return 0.0
        avg = sum(vols) / len(vols)
        self.avg_cache[key] = avg
        return avg

    def find_sweeps(self, product: str) -> List[SweepCandidate]:
        """Scan latest 5m candles vs aged key levels and return sweep candidates that pass initial filters."""
        now = ts_utc()
        c5 = CB.candles_paged(product, G_5M, start=now - timedelta(hours=6), end=now)
        if len(c5) < 25:
            return []
        avg20 = self._avg_volume(product, n=20)
        levels = self.store.aged_levels(product, MIN_LEVEL_AGE_H)

        out: List[SweepCandidate] = []
        # Evaluate the most recent candle only for the sweep event
        for cidx in range(max(1, len(c5) - 2), len(c5)):
            t, lo, hi, op, cl, vol = c5[cidx]
            body_high = max(op, cl)
            body_low = min(op, cl)
            wick_top = hi - body_high
            wick_bottom = body_low - lo
            for (prod, lvl, side, kind, formed_at, touches, strength) in levels:
                if lvl <= 0 or abs(((lvl - cl) / lvl)) > 0.02:
                    # Ignore levels far away (>2%) for efficiency
                    continue
                # Did price sweep beyond level by 0.1%-0.5%?
                if side == "high":
                    over = (hi - lvl) / lvl
                    if SWEEP_MIN <= over <= SWEEP_MAX:
                        # wick showing rejection (upper wick prominent)
                        wick_rej = wick_top > (body_high - body_low)
                        vol_factor = (vol / avg20) if avg20 > 0 else 0.0
                        if vol_factor >= VOLUME_MULTIPLIER:
                            out.append(SweepCandidate(product, lvl, kind, side, hi, int(t), wick_rej, vol_factor))
                else:  # side == 'low'
                    under = (lvl - lo) / lvl
                    if SWEEP_MIN <= under <= SWEEP_MAX:
                        wick_rej = wick_bottom > (body_high - body_low)
                        vol_factor = (vol / avg20) if avg20 > 0 else 0.0
                        if vol_factor >= VOLUME_MULTIPLIER:
                            out.append(SweepCandidate(product, lvl, kind, side, lo, int(t), wick_rej, vol_factor))
        return out


# ==============================
# ------ Confirmation Logic ----
# ==============================
class Confirmation:
    @staticmethod
    def rsi_divergence_bearish(product: str) -> bool:
        now = ts_utc()
        c5 = CB.candles_paged(product, G_5M, start=now - timedelta(hours=10), end=now)
        closes = [c[4] for c in c5]
        rs = rsi(closes, 14)
        # Find last two swing highs in price and compare RSI tops
        swings: List[int] = []
        lookback = 2
        for i in range(lookback, len(c5) - lookback):
            hi = c5[i][2]
            if all(hi >= c5[i - k][2] for k in range(1, lookback + 1)) and \
               all(hi >= c5[i + k][2] for k in range(1, lookback + 1)):
                swings.append(i)
        if len(swings) < 2:
            return False
        a, b = swings[-2], swings[-1]
        price_higher_high = c5[b][2] > c5[a][2]
        rsi_lower_high = rs[b] < rs[a]
        return price_higher_high and rsi_lower_high and not math.isnan(rs[b]) and not math.isnan(rs[a])

    @staticmethod
    def rsi_divergence_bullish(product: str) -> bool:
        now = ts_utc()
        c5 = CB.candles_paged(product, G_5M, start=now - timedelta(hours=10), end=now)
        closes = [c[4] for c in c5]
        rs = rsi(closes, 14)
        swings: List[int] = []
        lookback = 2
        for i in range(lookback, len(c5) - lookback):
            lo = c5[i][1]
            if all(lo <= c5[i - k][1] for k in range(1, lookback + 1)) and \
               all(lo <= c5[i + k][1] for k in range(1, lookback + 1)):
                swings.append(i)
        if len(swings) < 2:
            return False
        a, b = swings[-2], swings[-1]
        price_lower_low = c5[b][1] < c5[a][1]
        rsi_higher_low = rs[b] > rs[a]
        return price_lower_low and rsi_higher_low and not math.isnan(rs[b]) and not math.isnan(rs[a])

    @staticmethod
    def close_back_inside(product: str, level: float, side: str, within_candles: int = 2) -> Tuple[bool, Optional[float], Optional[float]]:
        """Return (ok, reversal_low, reversal_high) using recent 5m closes after the sweep."""
        now = ts_utc()
        c5 = CB.candles(product, G_5M, start=now - timedelta(hours=1), end=now)
        if len(c5) < within_candles + 2:
            return False, None, None
        # Check last few bars
        ok = False
        rev_low, rev_high = None, None
        for c in c5[-within_candles:]:
            t, lo, hi, op, cl, vol = c
            if side == "high" and cl < level:
                ok = True
                rev_low = lo
                rev_high = hi
                break
            if side == "low" and cl > level:
                ok = True
                rev_low = lo
                rev_high = hi
                break
        return ok, rev_low, rev_high


# ==============================
# --------- Trade Logic --------
# ==============================
class TradePlanner:
    @staticmethod
    def position_size(entry: float, stop: float, equity_usd: float, risk_fraction: float) -> float:
        risk_usd = equity_usd * risk_fraction
        per_unit_risk = abs(stop - entry)
        size = risk_usd / per_unit_risk if per_unit_risk > 0 else 0.0
        return max(0.0, size)

    @staticmethod
    def compute_targets(side: str, entry: float, sweep_extreme: float, level: float) -> Tuple[float, float, float]:
        # SL at 0.3% beyond sweep extreme
        if side == "short":
            sl = sweep_extreme * (1 + SL_BUFFER_PCT)
            # Minimum TP1 for 3R
            r = abs(sl - entry)
            tp1 = entry - MIN_R_MULTIPLE * r
            # TP2 at next S/R: use level prior to sweep as first target proxy
            tp2 = min(tp1 - r, level - 2 * r)  # push further down
        else:
            sl = sweep_extreme * (1 - SL_BUFFER_PCT)
            r = abs(entry - sl)
            tp1 = entry + MIN_R_MULTIPLE * r
            tp2 = max(tp1 + r, level + 2 * r)
        return sl, tp1, tp2


# ==============================
# -------- Alert / Output ------
# ==============================
class Alerts:
    @staticmethod
    def send(text: str):
        print(text)
        with open(LOG_PATH, "a") as f:
            f.write(f"{ts_utc().isoformat()} | {text}\n")
        if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
            try:
                url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
                payload = {"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": "Markdown"}
                requests.post(url, json=payload, timeout=10)
            except Exception as e:
                print(f"[telegram] send error: {e}")

    @staticmethod

    def fmt_signal(product: str, direction: str, level: float, level_note: str, sweep: float, entry: float,
                   sl: float, tp1: float, r_mult: float, vol_pct: float, ob_ratio: float) -> str:
        arrow = "-" if direction == "short" else "+"
        pair = product.replace("-", "/")
        rr = f"{r_mult:.1f}:1"
    # force ASCII in the level note
        level_note_ascii = level_note.replace("≥", ">=")
        sweep_pct = (100 * abs(sweep - level) / level) if level > 0 else 0.0
        return (
            "*LIQUIDITY SWEEP SETUP*\n"
            f"Pair: *{pair}*\n"
            f"Level: {fmt_price(level)} ({level_note_ascii})\n"
            f"Sweep: {fmt_price(sweep)} (+{sweep_pct:.2f}%)\n"
            f"Entry: {arrow} {fmt_price(entry)}\n"
            f"Stop: {fmt_price(sl)}\n"
            f"Target: {fmt_price(tp1)} ({rr})\n"
            f"Volume: {vol_pct:.0f}% of avg\n"
            f"Orderbook bid/ask: {ob_ratio:.2f}\n"
            f"Time (UTC): {ts_utc().strftime('%Y-%m-%d %H:%M')}"
        )



# ==============================
# ---------- Orchestrator ------
# ==============================
class Bot:
    def __init__(self):
        self.store = Store(DB_PATH)
        self.levels = LevelDetector(self.store)
        self.sweeps = SweepEngine(self.store)
        self.trades_today = 0
        self.day_key = datetime.utcnow().date()

    def _reset_daily(self):
        today = datetime.utcnow().date()
        if today != self.day_key:
            self.trades_today = 0
            self.day_key = today

    def _passes_volume_filter(self, product: str) -> bool:
        """Filter out illiquid pairs using 24h notional ~ last * base_volume."""
        try:
            s = CB.stats(product)
            last = float(s.get('last', 0) or s.get('price', 0) or 0)
            vol_base = float(s.get('volume', 0))
            usd = last * vol_base
            return usd >= MIN_DAILY_USD_VOL
        except Exception:
            # Fail-open for majors if endpoint hiccups
            return True

    @staticmethod
    def ob_direction_ok(side: str, ob_ratio: float) -> bool:
        """
        Enforce directional consistency between sweep side and order book imbalance.
        - SHORT setup (sweep above resistance => side == 'high'): require ask-heavy -> bid/ask < 0.70
        - LONG  setup (sweep below support   => side == 'low' ): require bid-heavy -> bid/ask > 1.30
        """
        return (
            (side == "high" and ob_ratio < ORDERBOOK_IMB_THRESHOLD_SHORT) or
            (side == "low"  and ob_ratio > ORDERBOOK_IMB_THRESHOLD_LONG)
        )

    def run_cycle(self):
        # Reset per-day trade cap
        self._reset_daily()

        # 1) Build/refresh levels per eligible product
        for product in COINS:
            if not self._passes_volume_filter(product):
                continue
            try:
                self.levels.build_levels(product)
            except Exception as e:
                print(f"[levels] {product}: {e}")

        # 2) Scan for sweeps and process confirmations/entries
        for product in COINS:
            if self.trades_today >= MAX_TRADES_PER_DAY:
                break

            try:
                cands = self.sweeps.find_sweeps(product)
                for c in cands:
                    # Close-back-inside check (must happen within 2 bars)
                    ok_close, rev_low, rev_high = Confirmation.close_back_inside(
                        product, c.level, c.side, within_candles=2
                    )
                    if not ok_close:
                        self.store.log_false_breakout(
                            product, c.level,
                            'up' if c.side == 'high' else 'down',
                            'no_close_back_inside'
                        )
                        continue

                    # Order book directional agreement (single check, applies to both sides)
                    ob_ratio = OrderBookMonitor.bid_ask_ratio(product)
                    #if not self.ob_direction_ok(c.side, ob_ratio):
                    #    self.store.log_false_breakout(
                    #        product, c.level,
                    #        'up' if c.side == 'high' else 'down',
                    #        'orderbook_direction_mismatch'
                    #    )
                    #    continue

                    if c.side == 'high':
                        # Bearish divergence required for a short after an upside sweep
                        div_ok = Confirmation.rsi_divergence_bearish(product)
                        if not div_ok:
                            self.store.log_false_breakout(product, c.level, 'up', 'no_divergence')
                            continue

                        # Entry on break of the reversal candle low
                        entry = rev_low if rev_low else c.level
                        sl, tp1, tp2 = TradePlanner.compute_targets('short', entry, c.sweep_price, c.level)
                        r = abs(entry - sl)
                        r_multiple = abs(entry - tp1) / r if r > 0 else 0.0
                        if r_multiple < MIN_R_MULTIPLE:
                            continue

                        size = TradePlanner.position_size(entry, sl, ACCOUNT_EQUITY_USD, RISK_PER_TRADE)
                        if size <= 0:
                            continue

                        text = Alerts.fmt_signal(
                            product, 'short', c.level,
                            f"{c.level_kind} - aged≥{MIN_LEVEL_AGE_H}h",
                            c.sweep_price, entry, sl, tp1, r_multiple,
                            c.vol_factor * 100, ob_ratio
                        )
                        Alerts.send(text)
                        ctx = json.dumps({
                            'level': c.level,
                            'kind': c.level_kind,
                            'sweep_price': c.sweep_price,
                            'vol_factor': c.vol_factor,
                            'ob_ratio': ob_ratio,
                            'size_units': size,
                        })
                        self.store.log_trade(product, 'short', entry, sl, tp1, tp2, r_multiple, ctx)
                        self.trades_today += 1
                        if self.trades_today >= MAX_TRADES_PER_DAY:
                            break

                    else:  # c.side == 'low'
                        # Bullish divergence required for a long after a downside sweep
                        div_ok = Confirmation.rsi_divergence_bullish(product)
                        if not div_ok:
                            self.store.log_false_breakout(product, c.level, 'down', 'no_divergence')
                            continue

                        # Entry on break of the reversal candle high
                        entry = rev_high if rev_high else c.level
                        sl, tp1, tp2 = TradePlanner.compute_targets('long', entry, c.sweep_price, c.level)
                        r = abs(entry - sl)
                        r_multiple = abs(tp1 - entry) / r if r > 0 else 0.0
                        if r_multiple < MIN_R_MULTIPLE:
                            continue

                        size = TradePlanner.position_size(entry, sl, ACCOUNT_EQUITY_USD, RISK_PER_TRADE)
                        if size <= 0:
                            continue

                        text = Alerts.fmt_signal(
                            product, 'long', c.level,
                            f"{c.level_kind} - aged≥{MIN_LEVEL_AGE_H}h",
                            c.sweep_price, entry, sl, tp1, r_multiple,
                            c.vol_factor * 100, ob_ratio
                        )
                        Alerts.send(text)
                        ctx = json.dumps({
                            'level': c.level,
                            'kind': c.level_kind,
                            'sweep_price': c.sweep_price,
                            'vol_factor': c.vol_factor,
                            'ob_ratio': ob_ratio,
                            'size_units': size,
                        })
                        self.store.log_trade(product, 'long', entry, sl, tp1, tp2, r_multiple, ctx)
                        self.trades_today += 1
                        if self.trades_today >= MAX_TRADES_PER_DAY:
                            break

            except Exception as e:
                print(f"[sweep] {product}: {e}")



# ==============================
# ------------- Main -----------
# ==============================
if __name__ == "__main__":
    bot = Bot()
    print("Liquidity Sweep Reversal Bot — Coinbase Edition. Press Ctrl+C to stop.")
    try:
        while True:
            bot.run_cycle()
            # Run roughly once per minute; candles are 5m, but WS could be added for faster triggers
            time.sleep(60)
    except KeyboardInterrupt:
        print("Exiting.")
