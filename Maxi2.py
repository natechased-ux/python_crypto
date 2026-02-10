#!/usr/bin/env python3
"""
Regime-Adaptive Crypto Alert System (Python + Telegram)

What it does
- Classifies regime per coin: TRENDING vs RANGING (ADX + Bollinger Band Width)
- TRENDING → Breakout strategy (with strict momentum + volume + clearance checks)
- RANGING → Mean-reversion strategy (band + RSI extreme + candle confirmation)
- Uses ATR-based TP/SL (SL=1.5×ATR, TP=2.5×ATR)
- Non-async Telegram alerts (simple + reliable)
- Clean, compact alert message with a 0–100 confidence score

Notes
- Default timeframe: 1H (granularity=3600)
- Data: Coinbase Exchange public API (no auth)
- Keep deps light: requests, pandas, numpy
"""

from __future__ import annotations
import os
import time
import math
import json
import signal
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from typing import Dict, Optional

import requests
import numpy as np
import pandas as pd

# -------------------------------
# Configuration
# -------------------------------
CONFIG = {
    # Coins to scan (Coinbase Exchange product IDs)
    "SYMBOLS": ["eth-usd","xrp-usd","ltc-usd","ada-usd","doge-usd",
    "sol-usd","wif-usd","ondo-usd","sei-usd","magic-usd","ape-usd",
    "jasmy-usd","wld-usd","syrup-usd","fartcoin-usd","aero-usd",
    "link-usd","hbar-usd","aave-usd","fet-usd","crv-usd","tao-usd",
    "avax-usd","xcn-usd","uni-usd","mkr-usd","toshi-usd","near-usd",
    "algo-usd","trump-usd","bch-usd","inj-usd","pepe-usd","xlm-usd",
    "moodeng-usd","bonk-usd","dot-usd","popcat-usd","arb-usd","icp-usd",
    "tia-usd","ip-usd","pnut-usd","apt-usd","ena-usd","turbo-usd",
    "bera-usd","pol-usd","mask-usd","pyth-usd","sand-usd","morpho-usd",
    "mana-usd","c98-usd","axs-usd"],

    # Candle timeframe in seconds (Coinbase granularity): 60, 300, 900, 3600, 21600, 86400
    "GRANULARITY": 3600,           # 1H
    "LOOP_INTERVAL": 60,           # scan every minute
    "ALERT_COOLDOWN_MIN": 30,      # per symbol cooldown (minutes)
    "MIN_BARS": 120,               # minimum bars needed for indicators

    # Breakout lookback for trending regime
    "BREAKOUT_LOOKBACK": 20,

    # Bollinger and ATR/ADX settings
    "BB_PERIOD": 20,
    "BB_STD": 2.0,
    "ATR_PERIOD": 14,
    "ADX_PERIOD": 14,

    # Regime thresholds
    "ADX_TREND": 25.0,             # trending if ADX >= this + BBW expanding
    "ADX_RANGE": 20.0,             # ranging if ADX <= this + BBW compressed
    "BBW_COMPRESS_PCTL": 25,       # last BBW in bottom 25% of last 100

    # Trend momentum thresholds
    "RSI_TREND_MIN_LONG": 55.0,
    "RSI_TREND_MAX_SHORT": 45.0,

    # Range extremes for mean reversion
    "RSI_OVERSOLD": 30.0,
    "RSI_OVERBOUGHT": 70.0,

    # ATR-based risk settings
    "SL_ATR_MULT": 1.5,
    "TP_ATR_MULT": 2.5,

    # Volume confirmation for breakouts
    "VOL_MULT_BREAKOUT": 1.2,      # baseline
    "VOL_MULT_BREAKOUT_STRICT": 1.3,  # stricter of the two is used

    # Tightening knobs for trend breakouts
    "MIN_ADX_SLOPE": 0.1,          # require ADX rising at least this much vs previous bar
    "ADX_UP_BARS": 3,              # consider last N bars for ADX rising check
    "ADX_UP_MIN": 2,               # require ADX rising in at least this many of last N
    "MIN_MACD_SLOPE": 0.0,         # MACD histogram must be increasing by at least this
    "BREAKOUT_BUFFER_ATR": 0.25,   # must clear high/low by this many ATRs
    "MAX_DISTANCE_EMA_ATR": 2.0,   # skip if |price-EMA20| > this many ATRs (late-stage)

    # Timezone for timestamps in alerts
    "DISPLAY_TZ": "America/Los_Angeles",

    # Telegram
    "TELEGRAM_BOT_TOKEN": os.getenv("TELEGRAM_BOT_TOKEN", "8177096945:AAHTg5nxVTA6hcPkAL4MLAAHPsZ1at7Ywmw"),
    "TELEGRAM_CHAT_ID": os.getenv("TELEGRAM_CHAT_ID", "7967738614"),
}

CBX_BASE = "https://api.exchange.coinbase.com"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("regime_adaptive_alerts")


# -------------------------------
# Utilities
# -------------------------------
def fmt_price(p: float) -> str:
    if p <= 0:
        return f"{p:.2f}"
    if p < 0.001:
        return f"{p:.8f}"
    if p < 0.01:
        return f"{p:.6f}"
    if p < 1:
        return f"{p:.4f}"
    if p < 100:
        return f"{p:.2f}"
    return f"{p:.2f}"


def now_pt() -> str:
    tz = ZoneInfo(CONFIG["DISPLAY_TZ"])
    return datetime.now(tz=tz).strftime("%Y-%m-%d %H:%M:%S %Z")


# -------------------------------
# Data Fetching
# -------------------------------
def fetch_candles(symbol: str, granularity: int) -> pd.DataFrame:
    url = f"{CBX_BASE}/products/{symbol}/candles?granularity={granularity}"
    for attempt in range(3):
        try:
            r = requests.get(url, timeout=20)
            r.raise_for_status()
            break
        except requests.exceptions.Timeout:
            if attempt < 2:
                logger.warning("Timeout for %s, retrying...", symbol)
                time.sleep(1)
                continue
            else:
                raise
    data = r.json()
    df = pd.DataFrame(data, columns=["time","low","high","open","close","volume"])
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    return df[["time","open","high","low","close","volume"]].sort_values("time").reset_index(drop=True)



# -------------------------------
# Indicators (no external TA libs)
# -------------------------------
def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = (delta.clip(lower=0)).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    rs = gain / (loss.replace(0, np.nan))
    return 100 - (100 / (1 + rs))


def true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["close"].shift(1)
    ranges = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    )
    return ranges.max(axis=1)


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    tr = true_range(df)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    up_move = df["high"].diff()
    down_move = -df["low"].diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    tr = true_range(df)
    atr_series = tr.ewm(alpha=1 / period, adjust=False).mean()

    plus_di = 100 * (pd.Series(plus_dm, index=df.index).ewm(alpha=1 / period, adjust=False).mean() / atr_series)
    minus_di = 100 * (pd.Series(minus_dm, index=df.index).ewm(alpha=1 / period, adjust=False).mean() / atr_series)

    dx = ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)) * 100
    adx_vals = dx.ewm(alpha=1 / period, adjust=False).mean()
    return adx_vals


def bollinger(df: pd.DataFrame, period: int = 20, std_mult: float = 2.0):
    mid = df["close"].rolling(period).mean()
    sd = df["close"].rolling(period).std(ddof=0)
    upper = mid + std_mult * sd
    lower = mid - std_mult * sd
    width = (upper - lower) / mid
    return mid, upper, lower, width


def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = ema(series, fast)
    ema_slow = ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


# -------------------------------
# Candle patterns
# -------------------------------
def bullish_engulfing(df: pd.DataFrame) -> bool:
    if len(df) < 2:
        return False
    prev = df.iloc[-2]
    last = df.iloc[-1]
    return (
        prev["close"] < prev["open"]
        and last["close"] > last["open"]
        and last["close"] >= prev["open"]
        and last["open"] <= prev["close"]
    )


def bearish_engulfing(df: pd.DataFrame) -> bool:
    if len(df) < 2:
        return False
    prev = df.iloc[-2]
    last = df.iloc[-1]
    return (
        prev["close"] > prev["open"]
        and last["close"] < last["open"]
        and last["close"] <= prev["open"]
        and last["open"] >= prev["close"]
    )


# -------------------------------
# Regime + Signal Logic
# -------------------------------
@dataclass
class Signal:
    symbol: str
    regime: str  # "TRENDING" or "RANGING"
    side: str    # "LONG" or "SHORT"
    entry: float
    tp: float
    sl: float
    atr: float
    adx: float
    rsi: float
    macd_hist: float
    macd_slope: float
    bb_width: float
    extras: Dict
    asof: datetime


class StrategyEngine:
    def __init__(self, config: Dict):
        self.cfg = config
        self.cooldowns: Dict[str, float] = {}  # symbol -> epoch seconds

    def in_cooldown(self, symbol: str) -> bool:
        last = self.cooldowns.get(symbol, 0)
        return (time.time() - last) < self.cfg["ALERT_COOLDOWN_MIN"] * 60

    def mark_cooldown(self, symbol: str):
        self.cooldowns[symbol] = time.time()

    def compute(self, df: pd.DataFrame, symbol: str) -> Optional[Signal]:
        if len(df) < max(self.cfg["MIN_BARS"], self.cfg["BB_PERIOD"] + 30):
            return None

        # Indicators
        adx_series = adx(df, self.cfg["ADX_PERIOD"])
        atr_series = atr(df, self.cfg["ATR_PERIOD"])
        mid, upper, lower, bbw = bollinger(df, self.cfg["BB_PERIOD"], self.cfg["BB_STD"])
        rsi_series = rsi(df["close"], 14)
        _, _, hist = macd(df["close"], 12, 26, 9)

        last = df.iloc[-1]
        adx_last = float(adx_series.iloc[-1])
        adx_prev = float(adx_series.iloc[-2])
        adx_slope = adx_last - adx_prev
        atr_last = float(atr_series.iloc[-1])
        bbw_last = float(bbw.iloc[-1])
        bbw_prev = float(bbw.iloc[-2])
        rsi_last = float(rsi_series.iloc[-1])
        hist_last = float(hist.iloc[-1])
        hist_prev = float(hist.iloc[-2])
        macd_slope = hist_last - hist_prev
        ema20_last = float(ema(df["close"], 20).iloc[-1])
        dist_from_ema_atr = abs(float(last["close"]) - ema20_last) / max(1e-9, atr_last)

        # BB compression percentile (last 100)
        bbw_tail = bbw.dropna().iloc[-100:]
        if len(bbw_tail) < 20:
            return None
        bbw_rank = (bbw_tail.rank(pct=True).iloc[-1]) * 100.0
        compressed = bbw_rank <= self.cfg["BBW_COMPRESS_PCTL"]

        # Regime detection
        trending = (adx_last >= self.cfg["ADX_TREND"]) and (bbw_last > bbw_prev)
        ranging = (adx_last <= self.cfg["ADX_RANGE"]) and compressed

        extras = {
            "adx_slope": adx_slope,
            "ema20": ema20_last,
            "bbw_rank_pct": float(bbw_rank),
        }

        # Momentum health checks for trend mode
        adx_up_recent = (adx_series.diff().tail(self.cfg["ADX_UP_BARS"]) > 0).sum() >= self.cfg["ADX_UP_MIN"]
        momentum_ok_long = (
            macd_slope >= self.cfg["MIN_MACD_SLOPE"]
            and adx_up_recent
            and adx_slope >= self.cfg["MIN_ADX_SLOPE"]
            and dist_from_ema_atr <= self.cfg["MAX_DISTANCE_EMA_ATR"]
        )
        momentum_ok_short = (
            -macd_slope >= self.cfg["MIN_MACD_SLOPE"]
            and adx_up_recent
            and adx_slope >= self.cfg["MIN_ADX_SLOPE"]
            and dist_from_ema_atr <= self.cfg["MAX_DISTANCE_EMA_ATR"]
        )

        # TRENDING → Breakout logic (tightened)
        if trending:
            n = self.cfg["BREAKOUT_LOOKBACK"]
            recent_high = df["high"].rolling(n).max().shift(1)
            recent_low = df["low"].rolling(n).min().shift(1)
            hi_n = float(recent_high.iloc[-1])
            lo_n = float(recent_low.iloc[-1])

            vol_sma = df["volume"].rolling(20).mean().iloc[-1]
            vol_ok = True
            if not math.isnan(vol_sma):
                vol_ok = last["volume"] >= max(self.cfg["VOL_MULT_BREAKOUT"], self.cfg["VOL_MULT_BREAKOUT_STRICT"]) * vol_sma

            # LONG breakout (tightened)
            hi_buf = hi_n + self.cfg["BREAKOUT_BUFFER_ATR"] * atr_last
            if (
                last["close"] > hi_buf
                and hist_last > 0
                and macd_slope > 0
                and rsi_last >= self.cfg["RSI_TREND_MIN_LONG"]
                and vol_ok
                and momentum_ok_long
            ):
                entry = float(last["close"])
                sl = entry - self.cfg["SL_ATR_MULT"] * atr_last
                tp = entry + self.cfg["TP_ATR_MULT"] * atr_last
                extras.update({"lookback_high": hi_n})
                return Signal(symbol, "TRENDING", "LONG", entry, tp, sl, atr_last, adx_last, rsi_last, hist_last, macd_slope, bbw_last, extras, last["time"])

            # SHORT breakout (tightened)
            lo_buf = lo_n - self.cfg["BREAKOUT_BUFFER_ATR"] * atr_last
            if (
                last["close"] < lo_buf
                and hist_last < 0
                and macd_slope < 0
                and rsi_last <= self.cfg["RSI_TREND_MAX_SHORT"]
                and vol_ok
                and momentum_ok_short
            ):
                entry = float(last["close"])
                sl = entry + self.cfg["SL_ATR_MULT"] * atr_last
                tp = entry - self.cfg["TP_ATR_MULT"] * atr_last
                extras.update({"lookback_low": lo_n})
                return Signal(symbol, "TRENDING", "SHORT", entry, tp, sl, atr_last, adx_last, rsi_last, hist_last, macd_slope, bbw_last, extras, last["time"])

        # RANGING → Mean-reversion logic
        if ranging:
            band_width = (upper.iloc[-1] - lower.iloc[-1])
            near_upper = last["close"] >= (upper.iloc[-1] - 0.15 * band_width)
            near_lower = last["close"] <= (lower.iloc[-1] + 0.15 * band_width)

            # LONG fade
            if near_lower and rsi_last <= self.cfg["RSI_OVERSOLD"] and bullish_engulfing(df.tail(2)):
                entry = float(last["close"])
                sl = entry - self.cfg["SL_ATR_MULT"] * atr_last
                tp = entry + self.cfg["TP_ATR_MULT"] * atr_last
                extras.update({"near_band": "lower"})
                return Signal(symbol, "RANGING", "LONG", entry, tp, sl, atr_last, adx_last, rsi_last, hist_last, macd_slope, bbw_last, extras, last["time"])

            # SHORT fade
            if near_upper and rsi_last >= self.cfg["RSI_OVERBOUGHT"] and bearish_engulfing(df.tail(2)):
                entry = float(last["close"])
                sl = entry + self.cfg["SL_ATR_MULT"] * atr_last
                tp = entry - self.cfg["TP_ATR_MULT"] * atr_last
                extras.update({"near_band": "upper"})
                return Signal(symbol, "RANGING", "SHORT", entry, tp, sl, atr_last, adx_last, rsi_last, hist_last, macd_slope, bbw_last, extras, last["time"])

        return None


# -------------------------------
# Telegram Notifier
# -------------------------------
class Notifier:
    def __init__(self, token: str, chat_id: str):
        self.api = f"https://api.telegram.org/bot{token}/sendMessage"
        self.chat_id = chat_id

    def send(self, text: str):
        try:
            resp = requests.post(
                self.api,
                json={
                    "chat_id": self.chat_id,
                    "text": text,
                    "parse_mode": "Markdown",
                    "disable_web_page_preview": True,
                },
                timeout=10,
            )
            if resp.status_code != 200:
                logger.warning("Telegram send failed: %s", resp.text)
        except Exception as e:
            logger.warning("Telegram error: %s", e)


# -------------------------------
# Runner (+ scoring + message)
# -------------------------------
class Runner:
    def __init__(self, config: Dict):
        self.cfg = config
        self.engine = StrategyEngine(config)
        self.notifier = Notifier(config["TELEGRAM_BOT_TOKEN"], config["TELEGRAM_CHAT_ID"])
        self._stop = False

    def stop(self, *_):
        logger.info("Stopping runner...")
        self._stop = True

    def compute_score(self, s: Signal) -> tuple[int, Dict[str, int]]:
        """
        Compute a compact 0–100 confidence score with a transparent breakdown.
        Parts:
          ADX level (0..20), ADX slope (0..15), MACD (0..15), RSI buffer (0..15),
          Breakout clearance (0..15, trending only), EMA distance (0..15)
        """
        score = 0.0
        parts: Dict[str, int] = {}

        # 1) ADX level (reward 25..40)
        adx_part = max(0, min(1.0, (s.adx - 25.0) / 15.0)) * 20
        parts["ADX"] = round(adx_part); score += adx_part

        # 2) ADX slope (acceleration)
        adx_slope = float(s.extras.get("adx_slope", 0.0))
        slope_part = max(0, min(1.0, adx_slope / 0.5)) * 15 if adx_slope > 0 else 0
        parts["Slope"] = round(slope_part); score += slope_part

        # 3) MACD histogram + slope
        macd_part = 0
        macd_part += 10 if ((s.side == "LONG" and s.macd_hist > 0) or (s.side == "SHORT" and s.macd_hist < 0)) else 0
        macd_part += 5  if ((s.side == "LONG" and s.macd_slope > 0) or (s.side == "SHORT" and s.macd_slope < 0)) else 0
        parts["MACD"] = macd_part; score += macd_part

        # 4) RSI buffer beyond thresholds
        if s.regime == "TRENDING":
            rsi_part = max(0, min(1.0, (s.rsi - 55.0) / 10.0)) * 15 if s.side == "LONG" else max(0, min(1.0, (45.0 - s.rsi) / 10.0)) * 15
        else:
            rsi_part = max(0, min(1.0, (30.0 - s.rsi) / 10.0)) * 15 if s.side == "LONG" else max(0, min(1.0, (s.rsi - 70.0) / 10.0)) * 15
        parts["RSI"] = round(rsi_part); score += rsi_part

        # 5) Breakout clearance (only for trending signals)
        clr_part = 0.0
        if s.regime == "TRENDING":
            hi = s.extras.get("lookback_high")
            lo = s.extras.get("lookback_low")
            if hi is not None and s.side == "LONG":
                clearance = (s.entry - float(hi)) / max(1e-9, s.atr)
                clr_part = max(0, min(1.0, (clearance - 0.25) / 0.75)) * 15  # 0.25..1.0 ATR → 0..15
            if lo is not None and s.side == "SHORT":
                clearance = (float(lo) - s.entry) / max(1e-9, s.atr)
                clr_part = max(0, min(1.0, (clearance - 0.25) / 0.75)) * 15
        parts["Breakout"] = round(clr_part); score += clr_part

        # 6) Distance to EMA20 (closer = better)
        ema_part = 0.0
        ema20 = s.extras.get("ema20")
        if ema20 is not None and s.atr > 0:
            dist_atr = abs(s.entry - float(ema20)) / s.atr
            ema_part = max(0, 1.0 - min(dist_atr / 2.0, 1.0)) * 15  # <=2 ATR → up to 15 pts
        parts["EMA"] = round(ema_part); score += ema_part

        final = int(round(min(100, score)))
        return final, parts

    def build_message(self, s: Signal) -> str:
        tz = ZoneInfo(self.cfg["DISPLAY_TZ"])
        asof_local = s.asof.tz_convert(tz) if s.asof.tzinfo else s.asof.replace(tzinfo=timezone.utc).astimezone(tz)

        final_score, parts = self.compute_score(s)
        arrow = "↗" if s.macd_slope > 0 else ("↘" if s.macd_slope < 0 else "→")

        lines = []
        lines.append(f"*{s.symbol}* — *{s.side}* ({s.regime})  |  *Score:* {final_score}/100")
        lines.append(f"Time: {asof_local:%Y-%m-%d %H:%M:%S %Z}")
        lines.append("")
        lines.append(f"`Entry` {fmt_price(s.entry)}    `TP` {fmt_price(s.tp)}    `SL` {fmt_price(s.sl)}")
        lines.append(f"`ATR` {fmt_price(s.atr)}    `ADX` {s.adx:.1f} (Δ {float(s.extras.get('adx_slope',0)):+.2f})    `RSI` {s.rsi:.1f}")
        lines.append(f"`MACD` {s.macd_hist:.5f} {arrow} (Δ {s.macd_slope:+.5f})    `BB Width` {s.bb_width:.5f}")

        if s.regime == "TRENDING":
            hi = s.extras.get("lookback_high")
            lo = s.extras.get("lookback_low")
            ema20 = s.extras.get("ema20")
            if s.side == "LONG" and hi is not None:
                clearance = (s.entry - float(hi)) / max(1e-9, s.atr)
                lines.append(f"Breakout over `{fmt_price(float(hi))}` by `{clearance:.2f}` ATR; EMA20 `{fmt_price(float(ema20)) if ema20 else 'n/a'}`.")
            if s.side == "SHORT" and lo is not None:
                clearance = (float(lo) - s.entry) / max(1e-9, s.atr)
                lines.append(f"Breakdown under `{fmt_price(float(lo))}` by `{clearance:.2f}` ATR; EMA20 `{fmt_price(float(ema20)) if ema20 else 'n/a'}`.")

        if s.regime == "RANGING" and s.extras.get("near_band"):
            lines.append(f"Mean-reversion near {s.extras['near_band']} Bollinger band.")

        lines.append("")
        lines.append("_Risk model: SL=1.5×ATR, TP=2.5×ATR. Size accordingly._")

        # compact score breakdown footer
        parts_txt = " ".join([f"{k}:{v}" for k, v in parts.items()])
        lines.append(f"_{parts_txt}_")

        return "\n".join(lines)

    def scan_symbol(self, symbol: str):
        try:
            if self.engine.in_cooldown(symbol):
                return
            df = fetch_candles(symbol, self.cfg["GRANULARITY"])
            sig = self.engine.compute(df, symbol)
            if sig:
                self.engine.mark_cooldown(symbol)
                text = self.build_message(sig)
                logger.info("ALERT %s %s %s entry=%.8f", sig.symbol, sig.regime, sig.side, sig.entry)
                self.notifier.send(text)
        except Exception as e:
            logger.exception("scan_symbol error for %s: %s", symbol, e)

    def loop(self):
        logger.info("Starting scan loop — interval=%ss, timeframe=%ss", self.cfg["LOOP_INTERVAL"], self.cfg["GRANULARITY"])
        while not self._stop:
            start = time.time()
            for sym in self.cfg["SYMBOLS"]:
                self.scan_symbol(sym)
                time.sleep(0.5)  # gentle pacing
            elapsed = time.time() - start
            sleep_for = max(0, self.cfg["LOOP_INTERVAL"] - elapsed)
            time.sleep(sleep_for)


# -------------------------------
# Entrypoint
# -------------------------------
if __name__ == "__main__":
    runner = Runner(CONFIG)
    signal.signal(signal.SIGINT, runner.stop)
    signal.signal(signal.SIGTERM, runner.stop)
    logger.info("Regime-Adaptive Crypto Alerts booting… (%s)", now_pt())
    logger.info("Symbols: %s", ", ".join(CONFIG["SYMBOLS"]))
    try:
        runner.loop()
    except KeyboardInterrupt:
        pass
    finally:
        logger.info("Exited.")
