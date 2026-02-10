#!/usr/bin/env python3
"""\
Crypto Alert Bot (Coinbase WS via matches/trades) — SEEDED VERSION

This is your original bot with one key upgrade:
✅ It seeds BOTH 15m and 4H history from Coinbase Exchange REST candles at startup,
   so the 4H EMA trend filter and 15m RelVol baseline work immediately.

How seeding works (Coinbase doesn't provide native 4H candles):
  - Pull 1H candles (granularity=3600) via REST and aggregate them into 4H candles.
  - Pull 15m candles (granularity=900) via REST for RelVol + breakout range.

Install:
  pip install websockets requests

Note: Coinbase candle endpoint returns at most ~300 candles per call, so the 1H
history fetch is paged.
"""

import asyncio
import json
import time
from dataclasses import dataclass
from collections import deque
from datetime import datetime, timezone, timedelta
from typing import Deque, Dict, List, Optional, Tuple

import requests
import websockets

# =========================================================
# CONFIG — HARD CODE YOUR SETTINGS HERE
# =========================================================

# (Copied from your current file)
COINS: List[str] = [
    "1INCH-USD", "ACH-USD", "ADA-USD", "AERGO-USD", "AERO-USD",
    "AKT-USD", "ALEO-USD", "ALGO-USD", "AMP-USD", "ANKR-USD", "APE-USD", "API3-USD",
    "APR-USD", "APT-USD", "ARB-USD", "ASM-USD", "ASTER-USD", "ATH-USD", "ATOM-USD",
    "AUCTION-USD", "AURORA-USD", "AVAX-USD", "AVNT-USD", "AXL-USD", "AXS-USD",
    "BAL-USD", "BAT-USD", "BERA-USD", "BCH-USD", "BIRB-USD", "BNB-USD", "BNKR-USD",
    "BONK-USD", "BREV-USD", "BTC-USD", "CBETH-USD", "CHZ-USD", "CLANKER-USD",
    "C98-USD", "COMP-USD", "CRV-USD", "CRO-USD", "CTSI-USD", "CVX-USD", "DASH-USD",
    "DEXT-USD", "DOGE-USD", "DOT-USD", "EDGE-USD", "EIGEN-USD", "ELSA-USD", "ENA-USD",
    "ENS-USD", "ETC-USD", "ETH-USD", "ETHFI-USD", "FARTCOIN-USD", "FET-USD",
    "FIGHT-USD", "FIL-USD", "FLR-USD", "FLUID-USD", "GFI-USD", "GHST-USD", "GIGA-USD",
    "GLM-USD", "GRT-USD", "HBAR-USD", "HFT-USD", "HNT-USD", "IMU-USD", "IMX-USD",
    "INJ-USD", "INX-USD", "IP-USD", "IOTX-USD", "IRYS-USD", "JASMY-USD", "JITOSOL-USD",
    "JTO-USD", "JUPITER-USD", "KAITO-USD", "KERNEL-USD", "KITE-USD", "KMNO-USD",
    "KSM-USD", "KTA-USD", "LCX-USD", "LDO-USD", "LIGHTER-USD", "LINK-USD", "LPT-USD",
    "LRC-USD", "LTC-USD", "MAGIC-USD", "MANA-USD", "MANTLE-USD", "MATH-USD",
    "MET-USD", "MINA-USD", "MOG-USD", "MON-USD", "MOODENG-USD", "MORPHO-USD",
    "NEAR-USD", "NKN-USD", "NOICE-USD", "NMR-USD", "OGN-USD", "ONDO-USD", "OP-USD",
    "ORCA-USD", "PAXG-USD", "PEPE-USD", "PENDLE-USD", "PENGU-USD", "PERP-USD",
    "PIRATE-USD", "PLUME-USD", "POPCAT-USD", "POL-USD", "PRIME-USD", "PROMPT-USD",
    "PROVE-USD", "PUMP-USD", "PYTH-USD", "QNT-USD", "RED-USD", "RENDER-USD",
    "REZ-USD", "RLS-USD", "ROSE-USD", "RSR-USD", "SAFE-USD", "SAND-USD",
    "SAPIEN-USD", "SEI-USD", "SENT-USD", "SHIB-USD", "SKL-USD", "SKR-USD", "SKY-USD",
    "SNX-USD", "SOL-USD", "SPK-USD", "SPX-USD", "STX-USD", "STRK-USD", "SUPER-USD",
    "SWFTC-USD", "SYRUP-USD", "TAO-USD", "THQ-USD", "TIA-USD", "TON-USD",
    "TOSHI-USD", "TRAC-USD", "TREE-USD", "TRB-USD", "TRUMP-USD", "TRUST-USD",
    "TROLL-USD", "TURBO-USD", "UNI-USD", "USELESS-USD", "USD1-USD", "VARA-USD",
    "VET-USD", "VOXEL-USD", "VVV-USD", "W-USD", "WET-USD", "WIF-USD", "WLD-USD",
    "WLFI-USD", "XCN-USD", "XLM-USD", "XPL-USD", "XRP-USD", "XTZ-USD", "XYO-USD",
    "YFI-USD", "ZAMA-USD", "ZEC-USD", "ZEN-USD", "ZK-USD", "ZKC-USD", "ZKP-USD",
    "ZORA-USD", "ZRO-USD", "ZRX-USD"
]

# Keep your credentials as-is (hard-coded), but consider env vars later.
TELEGRAM_BOT_TOKEN = "REPLACE_WITH_YOUR_TOKEN"
TELEGRAM_CHAT_ID = "REPLACE_WITH_YOUR_CHAT_ID"

# --- 15m RelVol breakout params ---
TF_ENTRY_MINUTES = 15
RELVOL_LOOKBACK_15M = 96           # 96*15m = 24h baseline
RELVOL_THRESHOLD_15M = 7.0
BREAKOUT_LOOKBACK_15M = 16         # breakout above last 16 closed 15m highs (~4h)

# --- 4H trend filter ---
TF_TREND_MINUTES = 240
EMA_FAST = 50
EMA_SLOW = 200
EMA_FAST_MUST_BE_RISING = True

# --- pullback entry rules ---
PULLBACK_ZONE_PCT = 0.006
RECLAIM_CONFIRM_CLOSE = True
BOUNCE_VOL_MAX_FRAC = 0.85

# --- alert gating ---
ALERT_COOLDOWN_MINUTES = 30

# --- websocket subscription chunking ---
SUBSCRIBE_CHUNK_SIZE = 80

# --- REST seeding ---
REST_BASE = "https://api.exchange.coinbase.com"
SEED_15M_DAYS = 3                  # 3 days of 15m candles (288) fits in one call
SEED_1H_DAYS_FOR_4H = 40            # ~40 days of 1h candles, enough to warm EMA200 on 4H (needs ~34 days)
SEED_CONCURRENCY = 10
REST_TIMEOUT = 15

# =========================================================

WS_URL = "wss://ws-feed.exchange.coinbase.com"
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "relvol-breakout-bot/seeded-1.0"})


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def iso_z(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def bucket_start(dt: datetime, minutes: int) -> datetime:
    dt = dt.astimezone(timezone.utc)
    m = (dt.minute // minutes) * minutes
    return dt.replace(minute=m, second=0, microsecond=0)


@dataclass
class Candle:
    start: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float

    @staticmethod
    def new(start: datetime, price: float) -> "Candle":
        return Candle(start=start, open=price, high=price, low=price, close=price, volume=0.0)

    def update(self, price: float, size: float) -> None:
        self.close = price
        if price > self.high:
            self.high = price
        if price < self.low:
            self.low = price
        self.volume += size


class EMA:
    def __init__(self, period: int):
        self.period = period
        self.k = 2 / (period + 1)
        self.value: Optional[float] = None
        self.prev: Optional[float] = None

    def update(self, x: float) -> float:
        self.prev = self.value
        if self.value is None:
            self.value = x
        else:
            self.value = (x * self.k) + (self.value * (1 - self.k))
        return self.value

    def rising(self) -> bool:
        return self.value is not None and self.prev is not None and self.value > self.prev


@dataclass
class StrategyState:
    candle_15m: Optional[Candle]
    closed_15m: Deque[Candle]

    candle_4h: Optional[Candle]
    closed_4h: Deque[Candle]
    ema_fast: EMA
    ema_slow: EMA

    mode: str
    breakout_level: Optional[float]
    breakout_candle_vol: Optional[float]
    breakout_time: Optional[datetime]
    pullback_tag_time: Optional[datetime]

    last_price: Optional[float]
    last_alert: Optional[datetime]


def in_pullback_zone(price: float, level: float, zone_pct: float) -> bool:
    return abs(price - level) / level <= zone_pct


def relvol_15m(state: StrategyState) -> Optional[float]:
    if state.candle_15m is None:
        return None
    if len(state.closed_15m) < max(12, RELVOL_LOOKBACK_15M // 4):
        return None
    vols = [c.volume for c in list(state.closed_15m)[-RELVOL_LOOKBACK_15M:] if c.volume > 0]
    if len(vols) < max(12, RELVOL_LOOKBACK_15M // 4):
        return None
    avg = sum(vols) / len(vols)
    if avg <= 0:
        return None
    return state.candle_15m.volume / avg


def trend_4h_ok(state: StrategyState) -> bool:
    if state.ema_fast.value is None or state.ema_slow.value is None:
        return False
    if state.ema_fast.value <= state.ema_slow.value:
        return False
    if EMA_FAST_MUST_BE_RISING and not state.ema_fast.rising():
        return False
    return True


def recent_range_high_15m(state: StrategyState) -> Optional[float]:
    if len(state.closed_15m) < BREAKOUT_LOOKBACK_15M:
        return None
    recent = list(state.closed_15m)[-BREAKOUT_LOOKBACK_15M:]
    return max(c.high for c in recent)


def rollover_candle(cur: Optional[Candle], dt: datetime, tf_minutes: int, closed: Deque[Candle], price: float) -> Candle:
    bs = bucket_start(dt, tf_minutes)
    if cur is None:
        return Candle.new(bs, price)
    if bs > cur.start:
        closed.append(cur)
        return Candle.new(bs, price)
    return cur


def on_close_4h(state: StrategyState, c: Candle) -> None:
    state.ema_fast.update(c.close)
    state.ema_slow.update(c.close)


async def telegram_worker(q: asyncio.Queue):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    while True:
        text = await q.get()
        try:
            for attempt in range(6):
                try:
                    r = await asyncio.to_thread(
                        requests.post,
                        url,
                        data={"chat_id": TELEGRAM_CHAT_ID, "text": text, "disable_web_page_preview": True},
                        timeout=15,
                        headers={"Connection": "close"},
                    )
                    r.raise_for_status()
                    js = r.json()
                    if not js.get("ok", False):
                        raise RuntimeError(js)
                    break
                except Exception:
                    await asyncio.sleep(min(2 ** attempt, 30) + 0.1 * attempt)
        finally:
            q.task_done()


def format_watch(pid: str, rv: float, level: float, price: float, now: datetime) -> str:
    return (
        f"👀 WATCH: 15m RelVol Breakout\n\n"
        f"{pid}\n"
        f"RelVol(15m): {rv:.2f} > {RELVOL_THRESHOLD_15M}\n"
        f"Breakout level: {level:.8f}\n"
        f"Price: {price:.8f}\n"
        f"Next: wait for first pullback + reclaim\n"
        f"UTC: {now.strftime('%Y-%m-%d %H:%M:%S')}Z"
    )


def format_entry(pid: str, rv: float, level: float, price: float, now: datetime) -> str:
    return (
        f"✅ ENTRY: First Pullback Reclaim\n\n"
        f"{pid}\n"
        f"4H Trend: UP (EMA{EMA_FAST}>{EMA_SLOW})\n"
        f"RelVol(15m): {rv:.2f}\n"
        f"Reclaimed level: {level:.8f}\n"
        f"Price: {price:.8f}\n"
        f"UTC: {now.strftime('%Y-%m-%d %H:%M:%S')}Z"
    )


def cooldown_ok(state: StrategyState, now: datetime) -> bool:
    if state.last_alert is None:
        return True
    return (now - state.last_alert).total_seconds() >= (ALERT_COOLDOWN_MINUTES * 60)


def mark_alerted(state: StrategyState, now: datetime) -> None:
    state.last_alert = now


async def subscribe(ws, products: List[str]) -> None:
    for i in range(0, len(products), SUBSCRIBE_CHUNK_SIZE):
        chunk = products[i:i + SUBSCRIBE_CHUNK_SIZE]
        msg = {"type": "subscribe", "product_ids": chunk, "channels": [{"name": "matches", "product_ids": chunk}]}
        await ws.send(json.dumps(msg))
        await asyncio.sleep(0.25)


def init_state() -> StrategyState:
    return StrategyState(
        candle_15m=None,
        closed_15m=deque(maxlen=RELVOL_LOOKBACK_15M + 400),
        candle_4h=None,
        closed_4h=deque(maxlen=EMA_SLOW + 200),
        ema_fast=EMA(EMA_FAST),
        ema_slow=EMA(EMA_SLOW),
        mode="IDLE",
        breakout_level=None,
        breakout_candle_vol=None,
        breakout_time=None,
        pullback_tag_time=None,
        last_price=None,
        last_alert=None,
    )


# --------------------------
# REST SEEDING
# --------------------------

def rest_get_candles(product_id: str, start: datetime, end: datetime, granularity: int) -> List[list]:
    params = {"start": iso_z(start), "end": iso_z(end), "granularity": granularity}
    r = SESSION.get(f"{REST_BASE}/products/{product_id}/candles", params=params, timeout=REST_TIMEOUT)
    r.raise_for_status()
    return r.json()  # [time, low, high, open, close, volume]


def fetch_candles_paged(product_id: str, start: datetime, end: datetime, granularity: int, max_per_call: int = 290) -> List[list]:
    """Page candles by stepping time windows so we never exceed Coinbase's per-call candle limits."""
    secs = (end - start).total_seconds()
    step = timedelta(seconds=granularity * max_per_call)
    out: List[list] = []
    t0 = start
    while t0 < end:
        t1 = min(t0 + step, end)
        data = rest_get_candles(product_id, t0, t1, granularity)
        out.extend(data)
        t0 = t1
        time.sleep(0.05)  # gentle throttle
    # Coinbase returns newest-first; normalize to oldest-first and de-dupe by timestamp
    seen = set()
    cleaned = []
    for row in sorted(out, key=lambda x: x[0]):
        ts = row[0]
        if ts in seen:
            continue
        seen.add(ts)
        cleaned.append(row)
    return cleaned


def rows_to_candles(rows: List[list], tf_minutes: int) -> List[Candle]:
    """Convert exchange candle rows to Candle objects (assumes row[0] is epoch seconds)."""
    candles: List[Candle] = []
    for t, low, high, open_, close, vol in rows:
        start = datetime.fromtimestamp(int(t), tz=timezone.utc)
        candles.append(Candle(start=start, open=float(open_), high=float(high), low=float(low), close=float(close), volume=float(vol)))
    # already oldest-first
    return candles


def aggregate_1h_to_4h(one_h: List[Candle]) -> List[Candle]:
    """Aggregate 1H candles into 4H candles."""
    if not one_h:
        return []
    buckets: Dict[datetime, Candle] = {}
    for c in one_h:
        bs = bucket_start(c.start, 240)
        if bs not in buckets:
            buckets[bs] = Candle(start=bs, open=c.open, high=c.high, low=c.low, close=c.close, volume=c.volume)
        else:
            b = buckets[bs]
            b.high = max(b.high, c.high)
            b.low = min(b.low, c.low)
            b.close = c.close
            b.volume += c.volume
    return [buckets[k] for k in sorted(buckets.keys())]


def seed_one_product(pid: str) -> Tuple[str, Optional[List[Candle]], Optional[List[Candle]], Optional[str]]:
    """Returns: (pid, candles_15m, candles_4h, error)"""
    try:
        now = utc_now()

        # --- 15m candles ---
        end_15 = bucket_start(now, TF_ENTRY_MINUTES)
        start_15 = end_15 - timedelta(days=SEED_15M_DAYS)
        rows_15 = rest_get_candles(pid, start_15, end_15, 900)  # 15m
        c15 = rows_to_candles(sorted(rows_15, key=lambda x: x[0]), TF_ENTRY_MINUTES)

        # --- 1h candles -> aggregate to 4h ---
        end_1h = bucket_start(now, 60)
        start_1h = end_1h - timedelta(days=SEED_1H_DAYS_FOR_4H)
        rows_1h = fetch_candles_paged(pid, start_1h, end_1h, 3600)
        c1h = rows_to_candles(rows_1h, 60)
        c4h = aggregate_1h_to_4h(c1h)

        return pid, c15, c4h, None
    except Exception as e:
        return pid, None, None, str(e)


async def seed_states(states: Dict[str, StrategyState]) -> None:
    """Seed 15m history + 4H EMAs before WS starts."""
    sem = asyncio.Semaphore(SEED_CONCURRENCY)

    async def _work(pid: str):
        async with sem:
            return await asyncio.to_thread(seed_one_product, pid)

    tasks = [asyncio.create_task(_work(pid)) for pid in states.keys()]
    done = 0
    for t in asyncio.as_completed(tasks):
        pid, c15, c4h, err = await t
        done += 1
        if err:
            print(f"[SEED] {pid}: failed: {err}")
            continue

        st = states[pid]

        # Seed closed 15m candles (keep only the most recent lookback+buffer)
        for c in c15[-(RELVOL_LOOKBACK_15M + 200):]:
            st.closed_15m.append(c)

        # Seed closed 4H candles + build EMA state
        for c in c4h[-(EMA_SLOW + 150):]:
            st.closed_4h.append(c)
            on_close_4h(st, c)

        # Set last_price to last close (helps early logic)
        if st.closed_15m:
            st.last_price = st.closed_15m[-1].close

        if done % 25 == 0 or done == len(states):
            print(f"[SEED] progress: {done}/{len(states)}")


# --------------------------
# STRATEGY
# --------------------------

def breakout_logic(pid: str, state: StrategyState, now: datetime, tg_q: asyncio.Queue) -> None:
    if state.candle_15m is None or state.last_price is None:
        return

    if not trend_4h_ok(state):
        if state.mode != "IDLE":
            state.mode = "IDLE"
            state.breakout_level = None
            state.breakout_candle_vol = None
            state.breakout_time = None
            state.pullback_tag_time = None
        return

    rv = relvol_15m(state)
    if rv is None:
        return

    price = state.last_price

    if state.mode == "IDLE":
        rng_high = recent_range_high_15m(state)
        if rng_high is None:
            return

        if rv >= RELVOL_THRESHOLD_15M and price > rng_high:
            state.mode = "BREAKOUT"
            state.breakout_level = rng_high
            state.breakout_candle_vol = state.candle_15m.volume
            state.breakout_time = now
            state.pullback_tag_time = None

            if cooldown_ok(state, now):
                tg_q.put_nowait(format_watch(pid, rv, rng_high, price, now))
                mark_alerted(state, now)
        return

    if state.mode == "BREAKOUT":
        if state.breakout_level is None:
            state.mode = "IDLE"
            return
        if in_pullback_zone(price, state.breakout_level, PULLBACK_ZONE_PCT):
            state.mode = "PULLBACK_TAGGED"
            state.pullback_tag_time = now
        return

    if state.mode == "PULLBACK_TAGGED":
        if state.breakout_level is None:
            state.mode = "IDLE"
            return

        if not RECLAIM_CONFIRM_CLOSE:
            reclaimed = price > state.breakout_level
            bounce_ok = True
        else:
            if len(state.closed_15m) < 2:
                return
            last_closed = state.closed_15m[-1]
            reclaimed = last_closed.close > state.breakout_level
            if state.breakout_candle_vol:
                bounce_ok = last_closed.volume <= (state.breakout_candle_vol * BOUNCE_VOL_MAX_FRAC)
            else:
                bounce_ok = True

        if reclaimed and bounce_ok:
            if cooldown_ok(state, now):
                tg_q.put_nowait(format_entry(pid, rv, state.breakout_level, price, now))
                mark_alerted(state, now)

            state.mode = "IDLE"
            state.breakout_level = None
            state.breakout_candle_vol = None
            state.breakout_time = None
            state.pullback_tag_time = None


async def run():
    if not COINS:
        raise SystemExit("COINS list is empty.")

    tg_q: asyncio.Queue = asyncio.Queue(maxsize=5000)
    asyncio.create_task(telegram_worker(tg_q))

    states: Dict[str, StrategyState] = {pid: init_state() for pid in COINS}

    print(f"Seeding REST candles (15m + 1h->4h) for {len(COINS)} products...")
    await seed_states(states)
    print("Seeding complete. Connecting WS...")

    backoff = 1.0
    while True:
        try:
            async with websockets.connect(
                WS_URL,
                ping_interval=20,
                ping_timeout=20,
                close_timeout=10,
                max_size=2**23,
            ) as ws:
                print(f"Connected. Subscribing to {len(COINS)} products...")
                await subscribe(ws, COINS)
                print("Subscribed. Running strategy...")
                backoff = 1.0

                while True:
                    raw = await ws.recv()
                    now = utc_now()

                    try:
                        msg = json.loads(raw)
                    except Exception:
                        continue

                    if msg.get("type") != "match":
                        continue

                    pid = msg.get("product_id")
                    if pid not in states:
                        continue

                    try:
                        price = float(msg["price"])
                        size = float(msg["size"])
                    except Exception:
                        continue

                    if size <= 0 or price <= 0:
                        continue

                    st = states[pid]
                    st.last_price = price

                    # 15m candles
                    st.candle_15m = rollover_candle(st.candle_15m, now, TF_ENTRY_MINUTES, st.closed_15m, price)
                    st.candle_15m.update(price, size)

                    # If 15m rolled, the previous candle was appended; nothing else needed.

                    # 4H candles (live) — EMAs already seeded; update on 4H close
                    prev_4h_start = st.candle_4h.start if st.candle_4h else None
                    st.candle_4h = rollover_candle(st.candle_4h, now, TF_TREND_MINUTES, st.closed_4h, price)
                    if prev_4h_start is not None and st.candle_4h.start > prev_4h_start:
                        if st.closed_4h:
                            on_close_4h(st, st.closed_4h[-1])
                    st.candle_4h.update(price, size)

                    breakout_logic(pid, st, now, tg_q)

        except Exception as e:
            print(f"[WS] error/disconnect: {e}. Reconnecting in {backoff:.1f}s...")
            await asyncio.sleep(backoff)
            backoff = min(backoff * 1.8, 60.0)


def main():
    print(
        f"Strategy: 4H uptrend (EMA{EMA_FAST}/{EMA_SLOW}) + 15m RelVol breakout (> {RELVOL_THRESHOLD_15M}) "
        f"+ first pullback reclaim.\n"
        f"Coins: {len(COINS)} | TF: {TF_ENTRY_MINUTES}m entry, {TF_TREND_MINUTES}m trend\n"
    )
    asyncio.run(run())


if __name__ == "__main__":
    main()
