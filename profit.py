"""
CVD Liquidity Flush Detector — flush/absorption LONG + SHORT signals

What it does
------------
Listens to Coinbase's public WebSocket "matches" + "ticker" for multiple symbols.
Computes:
1) CVD (Cumulative Volume Delta) via Lee–Ready tick rule
2) 1m net delta and rolling Z-score to spot capitulation ("flush")
3) Day-anchored VWAP + deviation std
4) Liquidity sweep + wick absorption using 1m bars
5) Warm-start from recent 1m candles so stats are ready immediately

When conditions align, emits a LONG or SHORT "flush" signal with entry + SL, and
optionally sends a Telegram alert.

Designed to be:
- Simple: one file, no DB
- Real-time: aggregates trades to 1m bars
- Robust: uses minute-close data for sweep/absorption; avoids repainting
- Multi-coin & two-sided: evaluates LONG and SHORT per symbol independently

Dependencies
------------
Python 3.9+
    pip install websockets pandas numpy requests pytz

Quick start
-----------
1) Put your Telegram creds (token/chat_id) in TELEGRAM_TOKEN / TELEGRAM_CHAT_ID, or leave None to just print.
2) Edit PRODUCTS to your coins.
3) Run:  python flush_detector.py

Tuning knobs (search below for "CONFIG")
---------------------------------------
FLUSH_Z (default -2.5)  -> long capitulation requires z <= -2.5; shorts use z >= +2.5
SWEEP_LOOKBACK (30)     -> prior-low/high lookback (minutes) for sweep condition
DEVIATION_K (1.25)      -> std-devs from AVWAP (below for longs, above for shorts)
WICK_RECOVERY (0.6)     -> absorption threshold: (close-low)/(high-low) or (high-close)/(high-low)
ATR_MINUTES (50)        -> ATR lookback (1m bars)

Security
--------
Keep your Telegram token private (consider env vars).
"""

import asyncio
import json
import statistics
from collections import deque, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Optional, List, Tuple

import numpy as np
import pytz
import requests
import websockets

# ===================== CONFIG =====================
PRODUCTS = ["HBAR-USD", "eth-usd","xrp-usd","ltc-usd","ada-usd","doge-usd",
    "sol-usd","wif-usd","ondo-usd","sei-usd","magic-usd","ape-usd",
    "jasmy-usd","wld-usd"]  # add/remove symbols freely

# Thresholds / parameters
FLUSH_Z = -2.5            # LONG requires delta_z <= -2.5; SHORT uses delta_z >= +2.5
SWEEP_LOOKBACK = 30       # minutes to look back for prior high/low sweep
DEVIATION_K = 1.25        # std-devs from AVWAP (below for LONG, above for SHORT)
WICK_RECOVERY = 0.60      # absorption threshold (wick close location)
ATR_MINUTES = 50          # ATR window in minutes

# New knobs
SL_ATR_BUFFER = 0.25      # how far beyond the sweep extreme to place SL (in ATR)
FRESH_MAX_MIN = 3         # sweep must be <= this many minutes old
REVERT_GUARD_ATR = 0.50

# ---- Reference TP config ----
TP_MODE = "VWAP"   # options: "VWAP", "ATR", "SWING"
TP_ATR_MULT = 1.0  # only used if TP_MODE == "ATR"


# Telegram (set to None to disable sending)
TELEGRAM_TOKEN: Optional[str] = "8177096945:AAHTg5nxVTA6hcPkAL4MLAAHPsZ1at7Ywmw"
TELEGRAM_CHAT_ID: Optional[str] = "7967738614"

# Timezone for human-readable timestamps
LOCAL_TZ = pytz.timezone("America/Los_Angeles")

WS_URL = "wss://ws-feed.exchange.coinbase.com"



# ===================== UTILITIES =====================
def now_ts() -> datetime:
    return datetime.now(timezone.utc)


def to_local(ts: datetime) -> str:
    return ts.astimezone(LOCAL_TZ).strftime("%Y-%m-%d %H:%M:%S %Z")


def safe_float(x) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return None


# ===================== DATA STRUCTURES =====================
@dataclass
class MinuteBar:
    t: int  # epoch minute
    open: float
    high: float
    low: float
    close: float
    volume: float
    net_delta: float  # signed vol in this minute


class AVWAP:
    """Day-anchored VWAP with std-dev tracker (minute-close deviations)."""

    def __init__(self):
        self.sum_pv = 0.0
        self.sum_v = 0.0
        self.deviations = deque(maxlen=2400)  # store minute-close deviations
        self.current_day = None  # YYYY-MM-DD

    def reset_if_new_day(self, ts: datetime):
        day = ts.strftime("%Y-%m-%d")
        if self.current_day != day:
            self.current_day = day
            self.sum_pv = 0.0
            self.sum_v = 0.0
            self.deviations.clear()

    def update(self, price: float, vol: float, minute_close: Optional[float] = None):
        """Update PV sums each trade; optionally append deviation at minute close."""
        if vol > 0:
            self.sum_pv += price * vol
            self.sum_v += vol
        if self.sum_v > 0 and minute_close is not None:
            vwap = self.value
            self.deviations.append(minute_close - vwap)

    @property
    def value(self) -> Optional[float]:
        return self.sum_pv / self.sum_v if self.sum_v > 0 else None

    @property
    def dev_std(self) -> Optional[float]:
        if len(self.deviations) < 20:
            return None
        return statistics.pstdev(self.deviations)


class MinuteBook:
    """Aggregates trades into 1m bars and computes delta z-score + ATR."""

    def __init__(self, atr_minutes: int = ATR_MINUTES):
        self.current_minute = None
        self.buffer: List[MinuteBar] = []
        self.minute_open = None
        self.minute_high = None
        self.minute_low = None
        self.minute_close = None
        self.minute_vol = 0.0
        self.minute_delta = 0.0
        self.delta_hist = deque(maxlen=2400)  # 1m net delta history
        self.atr_minutes = atr_minutes

    def on_trade(self, ts: datetime, price: float, size: float, signed_vol: float) -> bool:
        """
        Update the ongoing minute; return True if the previous minute was closed.
        """
        m = int(ts.timestamp() // 60)
        closed_prev = False
        if self.current_minute is None:
            self._start_minute(m, price)
        elif m != self.current_minute:
            self._close_minute()
            closed_prev = True
            self._start_minute(m, price)

        self.minute_close = price
        self.minute_vol += size
        self.minute_delta += signed_vol
        if price > self.minute_high:
            self.minute_high = price
        if price < self.minute_low:
            self.minute_low = price
        return closed_prev

    def _start_minute(self, m: int, price: float):
        self.current_minute = m
        self.minute_open = price
        self.minute_high = price
        self.minute_low = price
        self.minute_close = price
        self.minute_vol = 0.0
        self.minute_delta = 0.0

    def _close_minute(self):
        bar = MinuteBar(
            t=self.current_minute,
            open=self.minute_open,
            high=self.minute_high,
            low=self.minute_low,
            close=self.minute_close,
            volume=self.minute_vol,
            net_delta=self.minute_delta,
        )
        self.buffer.append(bar)
        self.delta_hist.append(bar.net_delta)

    def last_n(self, n: int) -> List[MinuteBar]:
        return self.buffer[-n:]

    def delta_z(self) -> Optional[float]:
        if len(self.delta_hist) < 50:
            return None
        arr = np.array(self.delta_hist, dtype=float)
        mu = arr.mean()
        std = arr.std()
        if std == 0:
            return None
        return (arr[-1] - mu) / std

    def atr(self) -> Optional[float]:
        bars = self.last_n(self.atr_minutes)
        if len(bars) < self.atr_minutes:
            return None
        trs = []
        prev_close = None
        for b in bars:
            if prev_close is None:
                tr = b.high - b.low
            else:
                tr = max(b.high - b.low, abs(b.high - prev_close), abs(b.low - prev_close))
            trs.append(tr)
            prev_close = b.close
        return float(np.mean(trs)) if trs else None

    def prior_low(self, lookback: int) -> Optional[float]:
        bars = self.last_n(lookback)
        if len(bars) < lookback:
            return None
        return min(b.low for b in bars[:-1]) if len(bars) > 1 else None

    def prior_high(self, lookback: int) -> Optional[float]:
        bars = self.last_n(lookback)
        if len(bars) < lookback:
            return None
        return max(b.high for b in bars[:-1]) if len(bars) > 1 else None


class TickSign:
    """Lee–Ready style tick rule to infer trade aggressor side."""

    def __init__(self):
        self.last_price: Optional[float] = None
        self.last_sign: int = 0

    def classify(self, price: float, mid: Optional[float] = None) -> int:
        if self.last_price is None:
            self.last_price = price
            self.last_sign = 0
            return 0
        if price > self.last_price:
            s = +1
        elif price < self.last_price:
            s = -1
        else:
            if mid is None:
                s = self.last_sign
            else:
                s = +1 if price >= mid else -1
        self.last_price = price
        self.last_sign = s
        return s


# ===================== DETECTOR =====================
@dataclass
class FlushSignal:
    ts: datetime
    symbol: str
    entry: float
    sl: float
    reason: str
    side: str  # "LONG" or "SHORT"


class FlushDetector:
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.book = MinuteBook()
        self.vwap = AVWAP()
        self.tick_sign = TickSign()
        self.last_mid = None
        self.last_price = None

    def on_trade(self, ts: datetime, price: float, size: float):
        # Update AVWAP daily anchor
        self.vwap.reset_if_new_day(ts)

        # Infer signed volume via tick rule
        sign = self.tick_sign.classify(price, self.last_mid)
        signed_vol = size * (1 if sign > 0 else -1) if sign != 0 else 0.0

        # Update per-minute aggregation (and detect minute close)
        closed_prev = self.book.on_trade(ts, price, size, signed_vol)

        # Update AVWAP PV sums (per trade)
        self.vwap.update(price, size)

        # If a minute just closed, append minute-close deviation for dev_std
        if closed_prev and self.book.buffer:
            last_closed = self.book.buffer[-1]
            self.vwap.update(price=last_closed.close, vol=0.0, minute_close=last_closed.close)

        self.last_price = price

    def on_ticker(self, best_bid: Optional[float], best_ask: Optional[float]):
        if best_bid is not None and best_ask is not None:
            self.last_mid = 0.5 * (best_bid + best_ask)

    def _fresh_enough(self, bar_minute: int) -> bool:
        """Ensure the sweep happened recently."""
        cur_min = int(now_ts().timestamp() // 60)
        return (cur_min - bar_minute) <= FRESH_MAX_MIN

    def _not_reverted_too_far(self, side: str, price: float, vwap: float, atr: float) -> bool:
        """Reject if price has mean-reverted too far toward AVWAP since sweep."""
        if atr is None or atr == 0 or vwap is None:
            return True
        dist = abs(vwap - price)
        # We want price still away from AVWAP by at least REVERT_GUARD_ATR * ATR
        return dist >= REVERT_GUARD_ATR * atr

    def check_signal(self) -> Optional[FlushSignal]:
        # Need enough history
        if len(self.book.buffer) < max(ATR_MINUTES, SWEEP_LOOKBACK, 10):
            return None

        last_bar = self.book.buffer[-1]
        prior_low = self.book.prior_low(SWEEP_LOOKBACK)
        prior_high = self.book.prior_high(SWEEP_LOOKBACK)
        delta_z = self.book.delta_z()
        atr = self.book.atr()
        vwap = self.vwap.value
        dev_std = self.vwap.dev_std

        if None in (delta_z, atr, vwap, dev_std) or (prior_low is None and prior_high is None):
            return None

        if not self._fresh_enough(last_bar.t):
            return None

        candidates = []

        # LONG flush
        if prior_low is not None:
            swept_l = last_bar.low < prior_low and last_bar.close > prior_low
            capit_l = delta_z <= FLUSH_Z
            extended_l = (vwap - last_bar.close) >= (DEVIATION_K * max(dev_std, 1e-9))
            absorbed_l = (last_bar.high - last_bar.low) > 0 and (last_bar.close - last_bar.low) / (last_bar.high - last_bar.low) >= WICK_RECOVERY
            if swept_l and capit_l and extended_l and absorbed_l and self._not_reverted_too_far("LONG", self.last_price or last_bar.close, vwap, atr):
                entry = float(self.last_price or last_bar.close)
                sl = float(last_bar.low - SL_ATR_BUFFER * atr)
                reason = f"sweep+capitulation (z={delta_z:.2f}) + AVWAP dev + absorption"
                candidates.append((abs(delta_z), FlushSignal(now_ts(), self.symbol, entry, sl, reason, "LONG")))

        # SHORT flush
        if prior_high is not None:
            swept_s = last_bar.high > prior_high and last_bar.close < prior_high
            capit_s = delta_z >= abs(FLUSH_Z)
            extended_s = (last_bar.close - vwap) >= (DEVIATION_K * max(dev_std, 1e-9))
            absorbed_s = (last_bar.high - last_bar.low) > 0 and (last_bar.high - last_bar.close) / (last_bar.high - last_bar.low) >= WICK_RECOVERY
            if swept_s and capit_s and extended_s and absorbed_s and self._not_reverted_too_far("SHORT", self.last_price or last_bar.close, vwap, atr):
                entry = float(self.last_price or last_bar.close)
                sl = float(last_bar.high + SL_ATR_BUFFER * atr)
                reason = f"sweep+capitulation (z={delta_z:.2f}) + AVWAP dev + absorption"
                candidates.append((abs(delta_z), FlushSignal(now_ts(), self.symbol, entry, sl, reason, "SHORT")))

        if not candidates:
            return None
        candidates.sort(key=lambda x: x[0], reverse=True)  # choose most extreme by |z|
        return candidates[0][1]

def suggest_tp(sig: FlushSignal, det: FlushDetector) -> Optional[float]:
    """
    Suggest a reference TP based on TP_MODE. Returns a float price or None.
    - VWAP: use current day-anchored VWAP
    - ATR:  use entry +/- (TP_ATR_MULT * ATR)
    - SWING: for LONGs target prior_high; for SHORTs target prior_low (fallback to VWAP)
    """
    try:
        vwap = det.vwap.value
        atr  = det.book.atr()
        last_bar = det.book.buffer[-1] if det.book.buffer else None

        if TP_MODE == "VWAP" and vwap is not None:
            return float(vwap)

        if TP_MODE == "ATR" and atr is not None:
            delta = TP_ATR_MULT * atr
            return float(sig.entry + (delta if sig.side == "LONG" else -delta))

        if TP_MODE == "SWING" and last_bar is not None:
            if sig.side == "LONG":
                ph = det.book.prior_high(SWEEP_LOOKBACK)
                return float(ph) if ph is not None else (float(vwap) if vwap is not None else None)
            else:
                pl = det.book.prior_low(SWEEP_LOOKBACK)
                return float(pl) if pl is not None else (float(vwap) if vwap is not None else None)

        return None
    except Exception:
        return None


# ===================== ALERTING =====================
class TelegramAlerter:
    def __init__(self, token: Optional[str], chat_id: Optional[str]):
        self.token = token
        self.chat_id = chat_id

    def send(self, text: str):
        if not self.token or not self.chat_id:
            print("[ALERT]\n" + text)
            return
        url = f"https://api.telegram.org/bot{self.token}/sendMessage"
        try:
            requests.post(url, json={"chat_id": self.chat_id, "text": text, "parse_mode": "Markdown"}, timeout=10)
        except Exception as e:
            print("[Telegram error]", e)
            print("[ALERT FALLBACK]\n" + text)

            


# ===================== WARM START (historical seeding) =====================
def fetch_recent_candles(product: str, limit: int = 240) -> List[Tuple[int, float, float, float, float, float]]:
    """
    Returns list of (t_epoch_minute, open, high, low, close, volume) sorted ascending.
    Coinbase Exchange public endpoint; no auth required.
    """
    url = f"https://api.exchange.coinbase.com/products/{product}/candles?granularity=60"
    r = requests.get(url, timeout=10)
    r.raise_for_status()
    data = r.json()  # [ time, low, high, open, close, volume ] (time in seconds)
    data = list(reversed(data))
    if len(data) > limit:
        data = data[-limit:]
    out = []
    for t, low, high, open_, close, vol in data:
        out.append((int(t // 60), float(open_), float(high), float(low), float(close), float(vol)))
    return out


def warm_seed_detector(det: FlushDetector, product: str, limit: int = 240):
    candles = fetch_recent_candles(product, limit=limit)
    # reset seed
    det.book.buffer.clear()
    det.book.delta_hist.clear()
    det.vwap.sum_pv = 0.0
    det.vwap.sum_v = 0.0
    det.vwap.deviations.clear()
    det.vwap.current_day = None

    for (t_min, o, h, l, c, v) in candles:
        ts = datetime.fromtimestamp(t_min * 60, tz=timezone.utc)
        det.vwap.reset_if_new_day(ts)
        # OBV-style signed volume seed for delta history
        sign = 1.0 if c >= o else -1.0
        net_delta = v * sign
        det.book.buffer.append(MinuteBar(t=t_min, open=o, high=h, low=l, close=c, volume=v, net_delta=net_delta))
        det.book.delta_hist.append(net_delta)
        # AVWAP PV sums + minute-close deviation
        det.vwap.update(c, v, minute_close=c)
        det.last_price = c


# ===================== WS CLIENT =====================
class CoinbaseWS:
    def __init__(self, symbols: List[str], alerter: TelegramAlerter):
        self.symbols = symbols
        self.detectors: Dict[str, FlushDetector] = {s: FlushDetector(s) for s in symbols}
        self.best_bid: Dict[str, Optional[float]] = defaultdict(lambda: None)
        self.best_ask: Dict[str, Optional[float]] = defaultdict(lambda: None)
        self.alerter = alerter
        # per-symbol cooldown: one signal per minute per symbol
        self.last_signal_minute: Dict[str, Optional[int]] = defaultdict(lambda: None)

    async def run(self):
        sub_msg = {
            "type": "subscribe",
            "product_ids": self.symbols,
            "channels": [
                {"name": "matches", "product_ids": self.symbols},
                {"name": "ticker", "product_ids": self.symbols},
            ],
        }
        async for ws in websockets.connect(WS_URL, ping_interval=20, ping_timeout=20):
            try:
                await ws.send(json.dumps(sub_msg))
                async for raw in ws:
                    msg = json.loads(raw)
                    t = msg.get("type")
                    if t == "ticker":
                        prod = msg.get("product_id")
                        self.best_bid[prod] = safe_float(msg.get("best_bid"))
                        self.best_ask[prod] = safe_float(msg.get("best_ask"))
                        self.detectors[prod].on_ticker(self.best_bid[prod], self.best_ask[prod])
                    elif t == "match":
                        prod = msg.get("product_id")
                        price = safe_float(msg.get("price"))
                        size = safe_float(msg.get("size")) or 0.0
                        time_str = msg.get("time")  # e.g., '2025-08-19T04:20:10.123456Z'
                        ts = datetime.fromisoformat(time_str.replace("Z", "+00:00")) if time_str else now_ts()
                        if price is None:
                            continue
                        self.detectors[prod].on_trade(ts, price, size)

                        # Try signal for this same product (per-symbol cooldown)
                        sig = self._maybe_signal(prod)
                        if sig:
                            self._emit(sig)

            except Exception as e:
                print("[WS error]", e, "— reconnecting...")
                await asyncio.sleep(2)
                continue

    def _maybe_signal(self, symbol: str) -> Optional[FlushSignal]:
        cur_min = int(now_ts().timestamp() // 60)
        if self.last_signal_minute[symbol] == cur_min:
            return None
        sig = self.detectors[symbol].check_signal()
        if sig:
            self.last_signal_minute[symbol] = cur_min
        return sig

    def _emit(self, sig: FlushSignal):
        ts_local = to_local(sig.ts)
        det = self.detectors[sig.symbol]
        atr = det.book.atr()

        tp = suggest_tp(sig, det)
        rr_txt = ""
        if tp is not None and sig.sl is not None and sig.sl != sig.entry:
            risk = abs(sig.entry - sig.sl)
            reward = abs(tp - sig.entry)
            if risk > 0:
                rr = reward / risk
                rr_txt = f"  (~{rr:.2f}R)"

        tp_line = f"Ref TP ({TP_MODE}): {tp:.6f}{rr_txt}\n" if tp is not None else ""

        text = (
            f"*Flush {sig.side}* on `{sig.symbol}`\n"
            f"Time: {ts_local}\n"
            f"Entry: {sig.entry:.6f}\n"
            f"SL: {sig.sl:.6f}\n"
            f"{tp_line}"
            f"ATR(1m x {ATR_MINUTES}): {atr:.6f}  (ref)\n"
            f"Reason: {sig.reason}\n"
            f"Params: z≤{FLUSH_Z} | dev≥{DEVIATION_K}σ | wick≥{WICK_RECOVERY} | SL={SL_ATR_BUFFER}×ATR\n"
        )
        self.alerter.send(text)



# ===================== MAIN =====================
async def main():
    alerter = TelegramAlerter(TELEGRAM_TOKEN, TELEGRAM_CHAT_ID)
    client = CoinbaseWS(PRODUCTS, alerter)
    # Warm-start each detector so stats are ready immediately
    seed_len = max(ATR_MINUTES, SWEEP_LOOKBACK, 300)
    for sym in PRODUCTS:
        try:
            # polite stagger if you add many coins: await asyncio.sleep(0.1)
            warm_seed_detector(client.detectors[sym], sym, limit=seed_len)
            print(f"[warm-start] {sym}: seeded {seed_len} minutes")
        except Exception as e:
            print(f"[warm-start] {sym} failed: {e}")
    await client.run()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Stopped.")
