
"""
Liquidity Sweep Entry Finder (with Limit Order Zones + Scoring)
----------------------------------------------------------------
Extends the original tool by adding:
1) A *limit order zone* around the swept level (ATR-based), plus 3 laddered limit prices.
2) A *score system* (0–100) to rank the limit candidates by R:R, confluence, and context.
3) Ranked output so you can choose the best limit placements per signal.

Original behavior (detection, TP/SL, Telegram) is preserved.

Designed for Coinbase Exchange symbols (e.g., "BTC-USD", "ETH-USD", "XRP-USD")
using the public market data REST endpoint. No API key needed for candles.
"""

from __future__ import annotations

import time
import math
import json
import requests
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime, timezone

# -------------------------------
# Config
# -------------------------------
SYMBOLS =  [
    "btc-usd","eth-usd","xrp-usd","ltc-usd","ada-usd","doge-usd",
    "sol-usd","wif-usd","ondo-usd","sei-usd","magic-usd","ape-usd",
    "jasmy-usd","wld-usd","syrup-usd","fartcoin-usd","aero-usd",
    "link-usd","hbar-usd","aave-usd","fet-usd","crv-usd","tao-usd",
    "avax-usd","xcn-usd","uni-usd","mkr-usd","toshi-usd","near-usd",
    "algo-usd","trump-usd","bch-usd","inj-usd","pepe-usd","xlm-usd",
    "moodeng-usd","bonk-usd","dot-usd","popcat-usd","arb-usd","icp-usd",
    "qnt-usd","tia-usd","ip-usd","pnut-usd","apt-usd","ena-usd","turbo-usd",
    "bera-usd","pol-usd","mask-usd","pyth-usd","sand-usd","morpho-usd",
    "mana-usd","c98-usd","axs-usd"
]

GRANULARITY = 3600  # 1H candles
CANDLE_LIMIT = 500

# Entry logic params
SWING_LOOKBACK = 3          # candles on each side to confirm a swing high/low
SWEEP_PCT = 0.0005          # 0.05% min penetration beyond swing to count as a sweep
STOCH_RSI_LEN = 14
EMA_TREND_FILTER = True
USE_STOCH_RSI = True

# Risk/targeting
ATR_LEN = 14
SL_ATR_BUFFER = 0.25        # add 0.25 * ATR beyond swept extreme
TP_ATR_MULT = 2.0           # fallback if no opposing swing target

# NEW: Limit zone + scoring configuration
LIMIT_ZONE_ATR = 0.75       # half-width of the zone as a multiple of ATR
# Three laddered limit levels inside the zone (0=at zone start, 1=at zone end)
LADDER_STEPS = [0.0, 0.5, 1.0]

# Scoring weights (sum doesn't have to be 100; the function normalizes)
W_RR          = 40.0   # weight for reward:risk (higher R:R, higher score up to a cap)
W_TARGET_ATR  = 20.0   # reward distance measured in ATRs (capped)
W_STOCH_TREND = 20.0   # confluence from stoch + trend filter
W_EMA_BUFFER  = 10.0   # penalty if entry is too close to ema200 (whipsaw risk)
W_ZONE_POS    = 10.0   # bonus for conservative placement deeper in the zone

# Cooldown to avoid duplicate alerts (seconds) per symbol+direction
COOLDOWN_SEC = 60 * 30
STATE_FILE = "liquidity_sweep_state.json"

# Telegram (optional)
TELEGRAM_BOT_TOKEN = "8177096945:AAHTg5nxVTA6hcPkAL4MLAAHPsZ1at7Ywmw"   # e.g. "8177096945:AAHTg5nxVTA6hcPkAL4MLAAHPsZ1at7Ywmw"
TELEGRAM_CHAT_ID = "7967738614"    # e.g. "7967738614"

# -------------------------------
# Utilities
# -------------------------------

def _load_state(path: str) -> dict:
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_state(path: str, state: dict) -> None:
    try:
        with open(path, "w") as f:
            json.dump(state, f, indent=2)
    except Exception:
        pass


def send_telegram(text: str) -> None:
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    try:
        requests.post(url, data={"chat_id": TELEGRAM_CHAT_ID, "text": text})
    except Exception:
        pass


# -------------------------------
# Data Fetch
# -------------------------------

def fetch_candles(product_id: str, granularity: int = 3600, limit: int = 500) -> pd.DataFrame:
    """Fetch OHLCV candles from Coinbase Exchange public endpoint.
    Returns newest->oldest, so we reverse to chronological (oldest->newest).
    API returns rows: [ time, low, high, open, close, volume ]
    """
    url = f"https://api.exchange.coinbase.com/products/{product_id}/candles"
    params = {"granularity": granularity, "limit": limit}
    r = requests.get(url, params=params, timeout=15)
    r.raise_for_status()
    data = r.json()
    if not data:
        raise RuntimeError("No candle data returned.")

    df = pd.DataFrame(data, columns=["time", "low", "high", "open", "close", "volume"]).sort_values("time")
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    return df.reset_index(drop=True)


# -------------------------------
# Indicators
# -------------------------------

def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"].shift(1)
    tr = pd.concat([
        (df["high"] - df["low"]).abs(),
        (df["high"] - close).abs(),
        (df["low"] - close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1/length, adjust=False).mean()


def rsi(series: pd.Series, length: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    ma_up = up.ewm(alpha=1/length, adjust=False).mean()
    ma_down = down.ewm(alpha=1/length, adjust=False).mean()
    rs = ma_up / ma_down.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))
    return out


def stoch_rsi_kd(close: pd.Series, rsi_len: int = 14, stoch_len: int = 14) -> tuple[pd.Series, pd.Series]:
    r = rsi(close, rsi_len)
    min_r = r.rolling(stoch_len).min()
    max_r = r.rolling(stoch_len).max()
    stoch = 100 * (r - min_r) / (max_r - min_r)
    k = stoch.rolling(3).mean()
    d = k.rolling(3).mean()
    return k, d


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ema200"] = ema(df["close"], 200)
    df["ATR"] = atr(df, ATR_LEN)
    k, d = stoch_rsi_kd(df["close"], 14, STOCH_RSI_LEN)
    df["stochK"], df["stochD"] = k, d
    return df


# -------------------------------
# Swing / Liquidity helpers
# -------------------------------

def swing_highs_lows(df: pd.DataFrame, lb: int = 3) -> tuple[list[int], list[int]]:
    """Return indices of swing highs and swing lows using a simple lookback."""
    sh, sl = [], []
    for i in range(lb, len(df) - lb):
        if df["high"].iloc[i] == df["high"].iloc[i-lb:i+lb+1].max():
            if df["high"].iloc[i] > df["high"].iloc[i-lb:i].max() and df["high"].iloc[i] > df["high"].iloc[i+1:i+lb+1].max():
                sh.append(i)
        if df["low"].iloc[i] == df["low"].iloc[i-lb:i+lb+1].min():
            if df["low"].iloc[i] < df["low"].iloc[i-lb:i].min() and df["low"].iloc[i] < df["low"].iloc[i+1:i+lb+1].min():
                sl.append(i)
    return sh, sl


def nearest_opposing_swing_target(df: pd.DataFrame, sh: list[int], sl: list[int], idx: int, side: str) -> float | None:
    """Pick the nearest opposing swing as a liquidity target relative to the entry index."""
    price = df["close"].iloc[idx]
    if side == "long":
        candidates = [df["high"].iloc[i] for i in sh if df["high"].iloc[i] > price]
        return min(candidates) if candidates else None
    else:
        candidates = [df["low"].iloc[i] for i in sl if df["low"].iloc[i] < price]
        return max(candidates) if candidates else None


@dataclass
class LimitCandidate:
    name: str
    price: float
    zone_low: float
    zone_high: float
    rr: float
    target_atrs: float
    ema_buffer_atrs: float
    stoch_trend_points: float
    zone_pos: float
    raw_score: float
    score: float


@dataclass
class Signal:
    symbol: str
    side: str  # "long" or "short"
    time: pd.Timestamp
    entry: float
    stop: float
    target: float
    swept_level: float
    limit_zone_low: float
    limit_zone_high: float
    limits: list[LimitCandidate] = field(default_factory=list)

    def text(self) -> str:
        ts = self.time.tz_convert("America/Los_Angeles").strftime("%Y-%m-%d %H:%M %Z")
        rr = abs((self.target - self.entry) / (self.entry - self.stop)) if self.entry != self.stop else float("nan")
        header = (
            f"[{self.symbol}] {self.side.upper()} after liquidity sweep\n"
            f"Time: {ts}\n"
            f"Entry (market ref): {self.entry:.8f}\nStop: {self.stop:.8f}\nTP: {self.target:.8f}\n"
            f"Swept level: {self.swept_level:.8f}\n"
            f"Limit Zone: {self.limit_zone_low:.8f} → {self.limit_zone_high:.8f}\n"
            f"Market R:R ≈ {rr:.2f}\n"
        )
        if not self.limits:
            return header
        lines = ["\nRanked limit orders:"]
        for i, c in enumerate(sorted(self.limits, key=lambda x: -x.score), 1):
            lines.append(
                f"{i}) {c.name}: {c.price:.8f}  | Score {c.score:.1f}  | R:R {c.rr:.2f}  "
                f"| Target~ATR {c.target_atrs:.2f}  | EMA buffer ATR {c.ema_buffer_atrs:.2f}"
            )
        return header + "\n".join(lines)


# -------------------------------
# Scoring
# -------------------------------

def _score_candidate(side: str, entry: float, stop: float, target: float, atr_v: float,
                     ema200: float, stochK: float, stochD: float, zone_low: float, zone_high: float) -> LimitCandidate:
    # Reward:Risk
    risk = abs(entry - stop)
    reward = abs(target - entry)
    rr = reward / risk if risk > 0 else 0.0

    # cap R:R contribution at 3:1 for scoring purposes
    rr_contrib = min(rr / 3.0, 1.0) * W_RR

    # Target distance in ATRs
    target_atrs = reward / atr_v if atr_v > 0 else 0.0
    target_contrib = min(target_atrs / 3.0, 1.0) * W_TARGET_ATR  # cap at 3 ATRs

    # Stoch + Trend
    stoch_ok = 0.0
    if side == "long":
        if stochK > stochD and stochK < 40:  # already filtered, but reward if strong
            stoch_ok = 1.0
        trend_ok = 1.0 if entry > ema200 else 0.0 if EMA_TREND_FILTER else 0.5
    else:
        if stochK < stochD and stochK > 60:
            stoch_ok = 1.0
        trend_ok = 1.0 if entry < ema200 else 0.0 if EMA_TREND_FILTER else 0.5
    stoch_trend_points = (0.6 * stoch_ok + 0.4 * trend_ok) * W_STOCH_TREND

    # EMA buffer (prefer not too close to ema200): measure distance in ATRs
    ema_buffer_atrs = abs(entry - ema200) / atr_v if atr_v > 0 else 0.0
    ema_penalty = 0.0
    if ema_buffer_atrs < 0.5:   # very close: bigger penalty
        ema_penalty = 1.0 * W_EMA_BUFFER
    elif ema_buffer_atrs < 1.0: # somewhat close
        ema_penalty = 0.5 * W_EMA_BUFFER
    elif ema_buffer_atrs < 1.5:
        ema_penalty = 0.25 * W_EMA_BUFFER
    # Zone position (bonus for deeper/safer fills)
    # normalize position 0..1 from zone edge nearest stop to farthest
    zone_range = max(zone_high - zone_low, 1e-12)
    if side == "long":
        zone_pos = (entry - zone_low) / zone_range  # 0=deep (near stop), 1=shallow
        zone_bonus = (1.0 - zone_pos) * W_ZONE_POS
    else:
        zone_pos = (zone_high - entry) / zone_range
        zone_bonus = (1.0 - zone_pos) * W_ZONE_POS

    raw = rr_contrib + target_contrib + stoch_trend_points + zone_bonus - ema_penalty
    # Normalize to 0..100
    # Estimate a plausible max = sum of weights + small margin
    max_possible = W_RR + W_TARGET_ATR + W_STOCH_TREND + W_ZONE_POS
    score = max(0.0, min(100.0, 100.0 * raw / max_possible))

    return LimitCandidate(
        name="",
        price=entry,
        zone_low=zone_low,
        zone_high=zone_high,
        rr=rr,
        target_atrs=target_atrs,
        ema_buffer_atrs=ema_buffer_atrs,
        stoch_trend_points=stoch_trend_points,
        zone_pos=zone_pos,
        raw_score=raw,
        score=score,
    )


def _build_limit_zone(side: str, swept_level: float, atr_v: float) -> tuple[float, float]:
    half = LIMIT_ZONE_ATR * atr_v
    if side == "long":
        # Zone from just below the swept low up to above it
        return swept_level - 0.05 * atr_v, swept_level + half
    else:
        return swept_level - half, swept_level + 0.05 * atr_v


def _ladder_prices(side: str, zone_low: float, zone_high: float) -> list[tuple[str, float]]:
    """Create named ladder levels across the zone from 'deep' to 'shallow'."""
    names = ["Deep", "Mid", "Shallow"]
    if len(LADDER_STEPS) != 3:
        # Fall back generically if user changed steps
        names = [f"Ladder {i+1}" for i in range(len(LADDER_STEPS))]
    prices = []
    for step, nm in zip(LADDER_STEPS, names):
        if side == "long":
            price = zone_low + step * (zone_high - zone_low)
        else:
            price = zone_high - step * (zone_high - zone_low)
        prices.append((nm, price))
    return prices


# -------------------------------
# Core detection
# -------------------------------

def detect_sweeps(df: pd.DataFrame) -> list[Signal]:
    df = add_indicators(df)
    sh, sl = swing_highs_lows(df, SWING_LOOKBACK)
    if not sh and not sl:
        return []

    out: list[Signal] = []
    i = len(df) - 2  # use last *closed* candle

    # --- Long: sweep of a prior swing low ---
    recent_sl = [idx for idx in sl if idx < i]
    if recent_sl:
        last_sl = recent_sl[-1]
        swept = df["low"].iloc[i] < df["low"].iloc[last_sl] * (1 - SWEEP_PCT)
        closed_back_above = df["close"].iloc[i] > df["low"].iloc[last_sl]
        trend_ok = (not EMA_TREND_FILTER) or (df["close"].iloc[i] > df["ema200"].iloc[i])
        stoch_ok = (not USE_STOCH_RSI) or (df["stochK"].iloc[i] > df["stochD"].iloc[i] and df["stochK"].iloc[i] < 40)
        if swept and closed_back_above and trend_ok and stoch_ok:
            entry_ref = float(df["close"].iloc[i])
            atr_v = float(df["ATR"].iloc[i])
            stop = min(float(df["low"].iloc[i]), float(df["low"].iloc[last_sl])) - SL_ATR_BUFFER * atr_v
            target = nearest_opposing_swing_target(df, sh, sl, i, side="long")
            if target is None:
                target = entry_ref + TP_ATR_MULT * atr_v

            # Build limit zone + ladder
            swept_level = float(df["low"].iloc[last_sl])
            zone_low, zone_high = _build_limit_zone("long", swept_level, atr_v)
            ladder = _ladder_prices("long", zone_low, zone_high)

            # Score each ladder entry
            limits: list[LimitCandidate] = []
            for nm, px in ladder:
                cand = _score_candidate(
                    side="long",
                    entry=float(px),
                    stop=stop,
                    target=float(target),
                    atr_v=atr_v,
                    ema200=float(df["ema200"].iloc[i]),
                    stochK=float(df["stochK"].iloc[i]),
                    stochD=float(df["stochD"].iloc[i]),
                    zone_low=zone_low,
                    zone_high=zone_high,
                )
                cand.name = nm
                limits.append(cand)

            out.append(Signal(
                symbol="",
                side="long",
                time=df["time"].iloc[i],
                entry=entry_ref,
                stop=stop,
                target=float(target),
                swept_level=swept_level,
                limit_zone_low=zone_low,
                limit_zone_high=zone_high,
                limits=sorted(limits, key=lambda x: -x.score),
            ))

    # --- Short: sweep of a prior swing high ---
    recent_sh = [idx for idx in sh if idx < i]
    if recent_sh:
        last_sh = recent_sh[-1]
        swept = df["high"].iloc[i] > df["high"].iloc[last_sh] * (1 + SWEEP_PCT)
        closed_back_below = df["close"].iloc[i] < df["high"].iloc[last_sh]
        trend_ok = (not EMA_TREND_FILTER) or (df["close"].iloc[i] < df["ema200"].iloc[i])
        stoch_ok = (not USE_STOCH_RSI) or (df["stochK"].iloc[i] < df["stochD"].iloc[i] and df["stochK"].iloc[i] > 60)
        if swept and closed_back_below and trend_ok and stoch_ok:
            entry_ref = float(df["close"].iloc[i])
            atr_v = float(df["ATR"].iloc[i])
            stop = max(float(df["high"].iloc[i]), float(df["high"].iloc[last_sh])) + SL_ATR_BUFFER * atr_v
            target = nearest_opposing_swing_target(df, sh, sl, i, side="short")
            if target is None:
                target = entry_ref - TP_ATR_MULT * atr_v

            swept_level = float(df["high"].iloc[last_sh])
            zone_low, zone_high = _build_limit_zone("short", swept_level, atr_v)
            ladder = _ladder_prices("short", zone_low, zone_high)

            limits: list[LimitCandidate] = []
            for nm, px in ladder:
                cand = _score_candidate(
                    side="short",
                    entry=float(px),
                    stop=stop,
                    target=float(target),
                    atr_v=atr_v,
                    ema200=float(df["ema200"].iloc[i]),
                    stochK=float(df["stochK"].iloc[i]),
                    stochD=float(df["stochD"].iloc[i]),
                    zone_low=zone_low,
                    zone_high=zone_high,
                )
                cand.name = nm
                limits.append(cand)

            out.append(Signal(
                symbol="",
                side="short",
                time=df["time"].iloc[i],
                entry=entry_ref,
                stop=stop,
                target=float(target),
                swept_level=swept_level,
                limit_zone_low=zone_low,
                limit_zone_high=zone_high,
                limits=sorted(limits, key=lambda x: -x.score),
            ))

    return out


# -------------------------------
# Scanning loop
# -------------------------------

def key_for(symbol: str, side: str) -> str:
    return f"{symbol}:{side}"


def scan_once(symbols: list[str]) -> list[Signal]:
    signals: list[Signal] = []
    for sym in symbols:
        try:
            df = fetch_candles(sym, GRANULARITY, CANDLE_LIMIT)
            sigs = detect_sweeps(df)
            for s in sigs:
                s.symbol = sym
            signals.extend(sigs)
        except Exception as e:
            print(f"[WARN] {sym}: {e}")
            continue
    return signals


def main_loop():
    state = _load_state(STATE_FILE)
    while True:
        now = time.time()
        signals = scan_once(SYMBOLS)

        for s in signals:
            k = key_for(s.symbol, s.side)
            last = state.get(k, 0)
            if now - last < COOLDOWN_SEC:
                continue

            msg = s.text()
            print("\n" + msg + "\n")
            send_telegram(msg)
            state[k] = now

        _save_state(STATE_FILE, state)

        # Wait before next scan (e.g. every 5 minutes)
        time.sleep(300)   # 300 seconds = 5 minutes



if __name__ == "__main__":
    main_loop()
