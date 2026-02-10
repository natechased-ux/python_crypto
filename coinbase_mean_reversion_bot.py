#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import math
import logging
import requests
import threading
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from dateutil import tz

# ------------------------------ Config ------------------------------
Z_SCORE_THRESHOLD = 2.8
RSI_OVERSOLD = 15.0
RSI_OVERBOUGHT = 85.0
VOLUME_SPIKE_THRESHOLD = 2.0
VOLUME_EXHAUSTION = 0.5
BB_PERIOD = 20
BB_STD_DEV = 2.5
MAX_CONCURRENT_SIGNALS = 3
SCAN_INTERVAL_SEC = 60 * 15
HOLD_TIME_LIMIT_HOURS = 24

PRIMARY_GRANULARITY = 900
CONFIRM_GRANULARITY = 300
CONTEXT_GRANULARITY = 3600

MIN_DAILY_USD_VOLUME = 10_000_000
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "8177096945:AAHTg5nxVTA6hcPkAL4MLAAHPsZ1at7Ywmw")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "7967738614")

COINBASE_API = "https://api.exchange.coinbase.com"
REQUEST_TIMEOUT = 15

COINBASE_PAIRS = [
    'BTC-USD', 'ETH-USD', 'ADA-USD', 'SOL-USD', 'DOT-USD',
    'AVAX-USD', 'MATIC-USD', 'LINK-USD', 'ATOM-USD', 'ALGO-USD',
    'XTZ-USD', 'COMP-USD', 'UNI-USD', 'AAVE-USD', 'MKR-USD',
    'SNX-USD', 'SUSHI-USD', 'YFI-USD', 'BAT-USD', '1INCH-USD',
    'BAND-USD', 'CRV-USD', 'GRT-USD', 'SKL-USD', 'NMR-USD',
    'OP-USD', 'ARB-USD', 'INJ-USD', 'SEI-USD', 'NEAR-USD',
    'RNDR-USD', 'BONK-USD', 'JTO-USD', 'SUI-USD',
    'DOGE-USD', 'SHIB-USD', 'APT-USD', 'WIF-USD',
    'SAGA-USD', 'PYTH-USD', 'STRK-USD', 'JUP-USD', 'ENA-USD', 'ONDO-USD'
]

UNDERPERFORMERS = {
    'VELO-USD','APT-USD','SYRUP-USD','AXS-USD','SAND-USD'
}
COINBASE_PAIRS = [p for p in COINBASE_PAIRS if p.upper() not in UNDERPERFORMERS]

# Tunable strictness (can be overridden by CLI flags)
REQUIRE_DIVERGENCE = True
REQUIRE_PATTERN = True
REQUIRE_VOLUME_EXH = True
EXTREME_REQUIRES_BB_AND_Z = True  # both Z and BB extreme required

# Backtest/scan thresholds (defaults copied from base)
TH_Z = Z_SCORE_THRESHOLD
TH_RSI_OVERSOLD = RSI_OVERSOLD
TH_RSI_OVERBOUGHT = RSI_OVERBOUGHT
TH_BB_STD_EXTREME = BB_STD_DEV
TH_VOL_SPIKE = VOLUME_SPIKE_THRESHOLD
TH_VOL_DECLINE = VOLUME_EXHAUSTION

# Bypass/relax flags (set by CLI)
BYPASS_TREND = False
BYPASS_STOCH = False
RELAX_STOCH_20 = False

# ------------------------------ Utils ------------------------------
def to_pacific(ts_utc: datetime) -> str:
    pacific = tz.gettz("America/Los_Angeles")
    return ts_utc.astimezone(pacific).strftime("%Y-%m-%d %H:%M %Z")

def pct(a: float, b: float) -> float:
    if b == 0: return 0.0
    return 100.0 * (a - b) / b

def safe_div(a: float, b: float) -> float:
    return a / b if b else 0.0

# -------------------------- Coinbase Client -------------------------
class CoinbaseClient:
    def __init__(self, api_url: str = COINBASE_API):
        self.api_url = api_url
        self.session = requests.Session()
        self.session.headers.update({
            "Accept": "application/json",
            "User-Agent": "MeanReversionBot/1.0",
        })

    def get_candles(self, product_id: str, granularity: int, start: Optional[datetime] = None, end: Optional[datetime] = None) -> pd.DataFrame:
        params = {"granularity": granularity}
        if start: params["start"] = start.replace(tzinfo=timezone.utc).isoformat()
        if end: params["end"] = end.replace(tzinfo=timezone.utc).isoformat()
        url = f"{self.api_url}/products/{product_id}/candles"
        r = self.session.get(url, params=params, timeout=REQUEST_TIMEOUT)
        r.raise_for_status()
        data = r.json()
        if not data: return pd.DataFrame()
        df = pd.DataFrame(data, columns=["time","low","high","open","close","volume"])
        df.sort_values("time", inplace=True)
        return df.reset_index(drop=True)

    def product_stats_24h(self, product_id: str) -> Optional[Dict]:
        url = f"{self.api_url}/products/{product_id}/stats"
        r = self.session.get(url, timeout=REQUEST_TIMEOUT)
        if r.status_code == 200: return r.json()
        return None

    def estimate_daily_usd_volume(self, product_id: str) -> float:
        stats = self.product_stats_24h(product_id)
        if stats and "volume" in stats and "last" in stats:
            try:
                base_vol = float(stats["volume"]); last_px = float(stats.get("last") or stats.get("last_trade", 0.0))
                return base_vol * last_px
            except Exception:
                pass
        try:
            df = self.get_candles(product_id, PRIMARY_GRANULARITY)
            if df.empty: return 0.0
            recent = df.tail(96); v = float(recent["volume"].sum()); px = float(recent["close"].iloc[-1])
            return v * px
        except Exception:
            return 0.0

# --------------------------- Indicators ----------------------------
class IndicatorUtils:
    @staticmethod
    def rsi(series: pd.Series, period: int = 14) -> pd.Series:
        delta = series.diff()
        gain = np.where(delta > 0, delta, 0.0)
        loss = np.where(delta < 0, -delta, 0.0)
        roll_up = pd.Series(gain, index=series.index).ewm(alpha=1/period, adjust=False).mean()
        roll_down = pd.Series(loss, index=series.index).ewm(alpha=1/period, adjust=False).mean()
        rs = roll_up / roll_down.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50.0)

    @staticmethod
    def stochastic_k(close: pd.Series, high: pd.Series, low: pd.Series, period: int = 14) -> pd.Series:
        lowest_low = low.rolling(period).min()
        highest_high = high.rolling(period).max()
        k = 100 * (close - lowest_low) / (highest_high - lowest_low)
        return k.replace([np.inf, -np.inf], np.nan).fillna(50.0)

    @staticmethod
    def bollinger_bands(series: pd.Series, period: int = BB_PERIOD, stds: float = BB_STD_DEV) -> Tuple[pd.Series, pd.Series, pd.Series]:
        sma = series.rolling(period).mean()
        std = series.rolling(period).std(ddof=0)
        upper = sma + stds * std
        lower = sma - stds * std
        return lower, sma, upper

    @staticmethod
    def z_score(series: pd.Series, window: int = BB_PERIOD) -> pd.Series:
        mean = series.rolling(window).mean()
        std = series.rolling(window).std(ddof=0)
        return (series - mean) / std.replace(0, np.nan)

    @staticmethod
    def ema(series: pd.Series, period: int) -> pd.Series:
        return series.ewm(span=period, adjust=False).mean()

# --------------------- Patterns & Volume/Divergence ------------------
class PatternAnalyzer:
    @staticmethod
    def is_hammer(row) -> bool:
        o, h, l, c = row["open"], row["high"], row["low"], row["close"]
        body = abs(c - o); rng = h - l
        lower_wick = min(o, c) - l; upper_wick = h - max(o, c)
        if rng <= 0: return False
        return (lower_wick > 2*body) and (upper_wick < body) and (c > o)

    @staticmethod
    def is_shooting_star(row) -> bool:
        o, h, l, c = row["open"], row["high"], row["low"], row["close"]
        body = abs(c - o); rng = h - l
        lower_wick = min(o, c) - l; upper_wick = h - max(o, c)
        if rng <= 0: return False
        return (upper_wick > 2*body) and (lower_wick < body) and (c < o)

    @staticmethod
    def is_doji(row, threshold: float = 0.1) -> bool:
        o, h, l, c = row["open"], row["high"], row["low"], row["close"]
        rng = h - l; body = abs(c - o)
        if rng <= 0: return False
        return body <= threshold * rng

class VolumeAnalyzer:
    @staticmethod
    def exhaustion_pattern(vol_series: pd.Series) -> Tuple[bool, bool, float, float]:
        if len(vol_series) < 25: return False, False, 0.0, 0.0
        avg20 = vol_series.tail(20).mean()
        current = vol_series.iloc[-1]
        spike = current > TH_VOL_SPIKE * avg20
        last3 = vol_series.tail(3).mean()
        decline = last3 < TH_VOL_DECLINE * avg20
        spike_ratio = safe_div(current, avg20)
        decline_ratio = safe_div(last3, avg20)
        return spike, decline, spike_ratio, decline_ratio

class DivergenceDetector:
    @staticmethod
    def bullish_divergence(price: pd.Series, rsi: pd.Series, lookback: int = 20) -> bool:
        if len(price) < lookback + 5: return False
        p = price.tail(lookback); r = rsi.tail(lookback)
        p_min1_idx = p.idxmin(); p1 = p[p_min1_idx]
        p_before = p[:p_min1_idx]; 
        if p_before.empty: return False
        p_min0_idx = p_before.idxmin(); p0 = p[p_min0_idx]
        r1 = r.get(p_min1_idx, np.nan); r0 = r.get(p_min0_idx, np.nan)
        if np.isnan(r1) or np.isnan(r0): return False
        return (p1 < p0) and (r1 > r0)

    @staticmethod
    def bearish_divergence(price: pd.Series, rsi: pd.Series, lookback: int = 20) -> bool:
        if len(price) < lookback + 5: return False
        p = price.tail(lookback); r = rsi.tail(lookback)
        p_max1_idx = p.idxmax(); p1 = p[p_max1_idx]
        p_before = p[:p_max1_idx]; 
        if p_before.empty: return False
        p_max0_idx = p_before.idxmax(); p0 = p[p_max0_idx]
        r1 = r.get(p_max1_idx, np.nan); r0 = r.get(p_max0_idx, np.nan)
        if np.isnan(r1) or np.isnan(r0): return False
        return (p1 > p0) and (r1 < r0)

# ------------------------- Ranking & Risk/Alerts ----------------------
@dataclass
class SetupContext:
    pair: str
    score: float
    direction: str
    stats: Dict[str, float] = field(default_factory=dict)
    message: str = ""

class RankingEngine:
    def rank(self, candidates: List[SetupContext]) -> List[SetupContext]:
        return sorted(candidates, key=lambda x: x.score, reverse=True)

class RiskManager:
    @staticmethod
    def calc_targets(current: float, sma20: float, bb_lower: float, bb_upper: float, direction: str) -> Tuple[float, float, float]:
        if direction == "LONG":
            halfway = current + 0.5 * (sma20 - current)
            sl = min(current, halfway) - abs(current) * 0.0005
            tp1 = sma20
            tp2 = bb_upper if not math.isnan(bb_upper) else sma20
        else:
            halfway = current + 0.5 * (sma20 - current)
            sl = max(current, halfway) + abs(current) * 0.0005
            tp1 = sma20
            tp2 = bb_lower if not math.isnan(bb_lower) else sma20
        return float(sl), float(tp1), float(tp2)

    @staticmethod
    def position_size(balance_usd: float, risk_pct: float, entry: float, stop: float) -> float:
        risk_amount = balance_usd * risk_pct
        per_unit_risk = abs(entry - stop)
        if per_unit_risk <= 0: return 0.0
        return max(risk_amount / per_unit_risk, 0.0)

class AlertManager:
    def __init__(self, bot_token: str = TELEGRAM_BOT_TOKEN, chat_id: str = TELEGRAM_CHAT_ID):
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.session = requests.Session()

    def send(self, text: str):
        if not self.bot_token or not self.chat_id:
            logging.warning("Telegram not configured; skipping alert:\n%s", text)
            return
        try:
            url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
            payload = {"chat_id": self.chat_id, "text": text, "parse_mode": "Markdown"}
            r = self.session.post(url, json=payload, timeout=REQUEST_TIMEOUT)
            if r.status_code != 200:
                logging.error("Telegram send error: %s %s", r.status_code, r.text[:2000])
        except Exception as e:
            logging.exception("Telegram send exception: %s", e)

# ------------------------ Mean Reversion Scanner ----------------------
class MeanReversionScanner:
    def __init__(self, client: 'CoinbaseClient'):
        self.client = client

    def _fetch_frames(self, pair: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        primary = self.client.get_candles(pair, PRIMARY_GRANULARITY)
        confirm = self.client.get_candles(pair, CONFIRM_GRANULARITY)
        context = self.client.get_candles(pair, CONTEXT_GRANULARITY)
        return primary, confirm, context

    def _context_trend_ok(self, df_ctx: pd.DataFrame, direction: str) -> bool:
        if BYPASS_TREND: return True
        if df_ctx.empty: return True
        close = df_ctx["close"]
        ema50 = IndicatorUtils.ema(pd.Series(close), 50)
        ema200 = IndicatorUtils.ema(pd.Series(close), 200)
        if len(ema50) < 200: return True
        return (ema50.iloc[-1] >= ema200.iloc[-1]) if direction == "LONG" else (ema50.iloc[-1] <= ema200.iloc[-1])

    def check_pair(self, pair: str) -> Optional[SetupContext]:
        try:
            # volume gate
            if CoinbaseClient().estimate_daily_usd_volume(pair) < MIN_DAILY_USD_VOLUME:
                return None

            primary, confirm, context = self._fetch_frames(pair)
            if primary.empty or len(primary) < 60: return None

            close = primary["close"]; high = primary["high"]; low = primary["low"]; open_ = primary["open"]
            rsi = IndicatorUtils.rsi(close, 14)
            k = IndicatorUtils.stochastic_k(close, high, low, 14)
            lower, sma20, upper = IndicatorUtils.bollinger_bands(close, BB_PERIOD, BB_STD_DEV)
            z = IndicatorUtils.z_score(close, BB_PERIOD)

            current = float(close.iloc[-1]); sma_v = float(sma20.iloc[-1])
            bb_lower = float(lower.iloc[-1]); bb_upper = float(upper.iloc[-1])
            z_now = float(z.iloc[-1]); rsi_now = float(rsi.iloc[-1])
            k_now = float(k.iloc[-1])

            spike, decline, spike_ratio, decline_ratio = VolumeAnalyzer.exhaustion_pattern(primary["volume"])
            row_last = primary.iloc[-1]
            hammer = PatternAnalyzer.is_hammer(row_last); shooting = PatternAnalyzer.is_shooting_star(row_last); doji = PatternAnalyzer.is_doji(row_last)
            bb_extreme = (current < bb_lower) or (current > bb_upper)
            bul_div = DivergenceDetector.bullish_divergence(close, rsi, 30)
            ber_div = DivergenceDetector.bearish_divergence(close, rsi, 30)

            long_ok = (((z_now <= -TH_Z) and (bb_extreme if EXTREME_REQUIRES_BB_AND_Z else True)) or (not EXTREME_REQUIRES_BB_AND_Z and bb_extreme)) \
                      and (rsi_now <= TH_RSI_OVERSOLD) \
                      and ((not REQUIRE_DIVERGENCE) or bul_div) \
                      and ((not REQUIRE_VOLUME_EXH) or (spike and decline)) \
                      and ((not REQUIRE_PATTERN) or (hammer or doji))

            short_ok = (((z_now >= TH_Z) and (bb_extreme if EXTREME_REQUIRES_BB_AND_Z else True)) or (not EXTREME_REQUIRES_BB_AND_Z and bb_extreme)) \
                       and (rsi_now >= TH_RSI_OVERBOUGHT) \
                       and ((not REQUIRE_DIVERGENCE) or ber_div) \
                       and ((not REQUIRE_VOLUME_EXH) or (spike and decline)) \
                       and ((not REQUIRE_PATTERN) or (shooting or doji))

            direction = None
            if long_ok and self._context_trend_ok(context, "LONG"): direction = "LONG"
            elif short_ok and self._context_trend_ok(context, "SHORT"): direction = "SHORT"
            else: return None

            if not BYPASS_STOCH:
                if confirm.empty or len(confirm) < 30: return None
                k5 = IndicatorUtils.stochastic_k(confirm["close"], confirm["high"], confirm["low"], 14)
                k5_now = float(k5.iloc[-1])
                thr_low = 20.0 if RELAX_STOCH_20 else 10.0
                thr_high = 80.0 if RELAX_STOCH_20 else 90.0
                if direction == "LONG" and not (k5_now <= thr_low): return None
                if direction == "SHORT" and not (k5_now >= thr_high): return None

            score = 0.0
            score += min(abs(z_now) * 10.0, 40.0)
            if rsi_now <= 20.0 or rsi_now >= 80.0: score += 25.0
            if spike and decline: score += 20.0
            if bb_extreme: score += 15.0

            sl, tp1, tp2 = RiskManager.calc_targets(current, sma_v, bb_lower, bb_upper, direction)
            rr = safe_div(abs(tp1 - current), abs(current - sl))

            stats = {
                "current": current, "sma20": sma_v, "z": z_now, "rsi": rsi_now, "k15": k_now,
                "bb_lower": bb_lower, "bb_upper": bb_upper, "bb_extreme": float(bb_extreme),
                "spike_ratio": spike_ratio, "decline_ratio": decline_ratio, "rr_to_tp1": rr,
            }

            deviation = pct(current, sma_v)
            title = "EXTREME MEAN REVERSION DETECTED"
            side = "Severely Oversold" if direction == "LONG" else "Severely Overbought"
            text = (
                f"🎯 *{title}*\n"
                f"*Pair:* `{pair}`\n"
                f"*Setup:* *{direction}* ({side})\n\n"
                f"📊 *Statistics:*\n"
                f"Current Price: `${current:.6f}`\n"
                f"20-SMA: `${sma_v:.6f}` ({deviation:+.2f}% deviation)\n"
                f"Z-Score: `{z_now:.2f}`\n"
                f"RSI(14): `{rsi_now:.1f}`\n"
                f"Bollinger Position: {'below lower' if direction=='LONG' else 'above upper'} band\n\n"
                f"🕐 Time: {to_pacific(datetime.now(timezone.utc))}\n"
            )

            return SetupContext(pair=pair, score=score, direction=direction, stats=stats, message=text)

        except Exception as e:
            logging.exception("check_pair error for %s: %s", pair, e)
            return None

# --------------------------- Orchestrator ---------------------------
class CoinbaseMeanReversionBot:
    def __init__(self, pairs: List[str] = None):
        self.client = CoinbaseClient()
        self.scanner = MeanReversionScanner(self.client)
        self.ranker = RankingEngine()
        self.alerts = AlertManager()
        self.pairs = pairs or COINBASE_PAIRS
        self._lock = threading.Lock()

    def scan_all_pairs_once(self) -> List[SetupContext]:
        results: List[SetupContext] = []

        def worker(pair: str):
            ctx = self.scanner.check_pair(pair)
            if ctx:
                with self._lock:
                    results.append(ctx)

        threads = []
        for pair in self.pairs:
            t = threading.Thread(target=worker, args=(pair,), daemon=True)
            t.start()
            threads.append(t)
            time.sleep(0.05)

        for t in threads:
            t.join()

        ranked = self.ranker.rank(results)
        return ranked

    def execute_top_setups(self, ranked: List[SetupContext], balance_usd: float = 10000.0, risk_per_trade: float = 0.02):
        top = ranked[:MAX_CONCURRENT_SIGNALS]
        for ctx in top:
            s = ctx.stats
            current = s["current"]
            sl, tp1, tp2 = RiskManager.calc_targets(current, s["sma20"], s["bb_lower"], s["bb_upper"], ctx.direction)
            qty = RiskManager.position_size(balance_usd, risk_per_trade, current, sl)
            addendum = (
                f"\n💰 *Sizing:* Risk `{risk_per_trade*100:.1f}%` of ${balance_usd:,.0f} | Qty: `{qty:.6f}`"
                f"\n🔒 *Hold Limit:* {HOLD_TIME_LIMIT_HOURS} hours max"
                f"\n🏁 *Max Concurrent Positions:* {MAX_CONCURRENT_SIGNALS}"
            )
            self.alerts.send(ctx.message + addendum)

    def loop(self):
        logging.info("Starting scanner loop. Pairs: %d", len(self.pairs))
        while True:
            try:
                ranked = self.scan_all_pairs_once()
                if ranked:
                    self.execute_top_setups(ranked)
                else:
                    logging.info("No extreme setups this cycle.")
            except Exception as e:
                logging.exception("scan loop error: %s", e)
            time.sleep(SCAN_INTERVAL_SEC)

# ------------------------------ Backtester ------------------------------
class Backtester:
    FEE_RATE = 0.005  # 0.5% per side

    def __init__(self, client: CoinbaseClient):
        self.client = client

    def _paginate(self, pair: str, granularity: int, months: int) -> pd.DataFrame:
        end = datetime.now(timezone.utc).replace(microsecond=0)
        start = end - timedelta(days=int(30.4375 * months))
        step_sec = granularity * 300
        out = []
        cur = start
        while cur < end:
            nxt = min(cur + timedelta(seconds=step_sec), end)
            df = self.client.get_candles(pair, granularity, start=cur, end=nxt)
            if not df.empty: out.append(df)
            cur = nxt
        if not out: return pd.DataFrame()
        df_all = pd.concat(out, ignore_index=True).drop_duplicates(subset=["time"]).sort_values("time").reset_index(drop=True)
        return df_all

    def _trend_ok(self, df1h: pd.DataFrame, when_ts: pd.Timestamp, direction: str) -> bool:
        if BYPASS_TREND: return True
        if df1h.empty: return True
        df1h = df1h.copy()
        df1h["dt"] = pd.to_datetime(df1h["time"], unit="s", utc=True)
        df1h = df1h[df1h["dt"] <= when_ts]
        if len(df1h) < 210: return True
        ema50 = IndicatorUtils.ema(df1h["close"], 50)
        ema200 = IndicatorUtils.ema(df1h["close"], 200)
        return (ema50.iloc[-1] >= ema200.iloc[-1]) if direction == "LONG" else (ema50.iloc[-1] <= ema200.iloc[-1])

    def _stoch5_ok(self, df5: pd.DataFrame, when_ts: pd.Timestamp, direction: str) -> bool:
        if BYPASS_STOCH: return True
        if df5.empty: return False
        df5 = df5.copy()
        df5["dt"] = pd.to_datetime(df5["time"], unit="s", utc=True)
        df5 = df5[df5["dt"] <= when_ts]
        if len(df5) < 20: return False
        k5 = IndicatorUtils.stochastic_k(df5["close"], df5["high"], df5["low"], 14)
        k_now = float(k5.iloc[-1])
        thr_low = 20.0 if RELAX_STOCH_20 else 10.0
        thr_high = 80.0 if RELAX_STOCH_20 else 90.0
        return (k_now <= thr_low) if direction == "LONG" else (k_now >= thr_high)

    def run_pair(self, pair: str, months: int = 12) -> Dict[str, float]:
        df15 = self._paginate(pair, PRIMARY_GRANULARITY, months)
        if df15.empty or len(df15) < 120:
            return {"pair": pair, "trades": 0, "win_rate": np.nan, "avg_rr": np.nan, "avg_hold_hours": np.nan, "max_concurrent": 0}
        df5 = self._paginate(pair, CONFIRM_GRANULARITY, months)
        df1h = self._paginate(pair, CONTEXT_GRANULARITY, months)

        close = df15["close"]; high = df15["high"]; low = df15["low"]; open_ = df15["open"]; vol = df15["volume"]
        lower, sma20, upper = IndicatorUtils.bollinger_bands(close, BB_PERIOD, BB_STD_DEV)
        rsi = IndicatorUtils.rsi(close, 14); z = IndicatorUtils.z_score(close, BB_PERIOD)
        dtcol = pd.to_datetime(df15["time"], unit="s", utc=True)

        trades = []; open_positions = []; max_conc = 0

        def vol_exhaustion(idx: int):
            if idx < 25: return (False, False)
            avg20_prev = vol.iloc[:idx].tail(20).mean()
            spike = vol.iloc[idx-1] > TH_VOL_SPIKE * avg20_prev if avg20_prev > 0 else False
            recent3 = vol.iloc[idx-2:idx+1].mean() if idx-2 >= 0 else vol.iloc[:idx+1].mean()
            decline = recent3 < TH_VOL_DECLINE * avg20_prev if avg20_prev > 0 else False
            return (bool(spike), bool(decline))

        for i in range(60, len(df15) - 2):
            cur_high = float(high.iloc[i]); cur_low = float(low.iloc[i]); cur_time = dtcol.iloc[i]

            # Update open positions
            next_open_positions = []
            for pos in open_positions:
                if cur_time >= pos["expiry"]:
                    exit_px = float(close.iloc[i])
                    exit_px_net = exit_px * (1 - self.FEE_RATE) if pos["side"] == "LONG" else exit_px * (1 + self.FEE_RATE)
                    rr = (exit_px_net - pos["entry_net"]) / (abs(pos["entry"] - pos["sl"]) + 1e-12) if pos["side"] == "LONG" \
                         else (pos["entry_net"] - exit_px_net) / (abs(pos["entry"] - pos["sl"]) + 1e-12)
                    trades.append({"entry_time": pos["entry_time"], "exit_time": cur_time, "side": pos["side"], "outcome": "timeout", "rr": rr, "hold_h": (cur_time - pos["entry_time"]).total_seconds()/3600.0})
                    continue
                if pos["side"] == "LONG":
                    if cur_low <= pos["sl"]:
                        exit_px_net = pos["sl"] * (1 - self.FEE_RATE); rr = (exit_px_net - pos["entry_net"]) / (abs(pos["entry"] - pos["sl"]) + 1e-12)
                        trades.append({"entry_time": pos["entry_time"], "exit_time": cur_time, "side": "LONG", "outcome": "sl", "rr": rr, "hold_h": (cur_time - pos["entry_time"]).total_seconds()/3600.0}); continue
                    if cur_high >= pos["tp1"]:
                        exit_px_net = pos["tp1"] * (1 - self.FEE_RATE); rr = (exit_px_net - pos["entry_net"]) / (abs(pos["entry"] - pos["sl"]) + 1e-12)
                        trades.append({"entry_time": pos["entry_time"], "exit_time": cur_time, "side": "LONG", "outcome": "tp1", "rr": rr, "hold_h": (cur_time - pos["entry_time"]).total_seconds()/3600.0}); continue
                else:
                    if cur_high >= pos["sl"]:
                        exit_px_net = pos["sl"] * (1 + self.FEE_RATE); rr = (pos["entry_net"] - exit_px_net) / (abs(pos["entry"] - pos["sl"]) + 1e-12)
                        trades.append({"entry_time": pos["entry_time"], "exit_time": cur_time, "side": "SHORT", "outcome": "sl", "rr": rr, "hold_h": (cur_time - pos["entry_time"]).total_seconds()/3600.0}); continue
                    if cur_low <= pos["tp1"]:
                        exit_px_net = pos["tp1"] * (1 + self.FEE_RATE); rr = (pos["entry_net"] - exit_px_net) / (abs(pos["entry"] - pos["sl"]) + 1e-12)
                        trades.append({"entry_time": pos["entry_time"], "exit_time": cur_time, "side": "SHORT", "outcome": "tp1", "rr": rr, "hold_h": (cur_time - pos["entry_time"]).total_seconds()/3600.0}); continue
                next_open_positions.append(pos)
            open_positions = next_open_positions
            if len(open_positions) > max_conc: max_conc = len(open_positions)
            if len(open_positions) >= 3: continue  # concurrency cap

            sig_idx = i - 1
            price_sig = float(close.iloc[sig_idx])
            sma_sig = float(sma20.iloc[sig_idx]) if not math.isnan(sma20.iloc[sig_idx]) else np.nan
            z_sig = float(z.iloc[sig_idx]) if not math.isnan(z.iloc[sig_idx]) else np.nan
            rsi_sig = float(rsi.iloc[sig_idx]) if not math.isnan(rsi.iloc[sig_idx]) else np.nan

            # Flexible BB extreme using std at signal index
            std_sig = float(close.iloc[:sig_idx+1].rolling(BB_PERIOD).std(ddof=0).iloc[-1]) if sig_idx+1 >= BB_PERIOD else float("nan")
            if not math.isnan(std_sig) and not math.isnan(sma_sig):
                lower_flex = sma_sig - TH_BB_STD_EXTREME * std_sig
                upper_flex = sma_sig + TH_BB_STD_EXTREME * std_sig
                bb_extreme = (price_sig < lower_flex) or (price_sig > upper_flex)
            else:
                bb_extreme = False

            # Divergences via truncated windows
            bul_div = DivergenceDetector.bullish_divergence(close.iloc[:sig_idx+1], rsi.iloc[:sig_idx+1], 30)
            ber_div = DivergenceDetector.bearish_divergence(close.iloc[:sig_idx+1], rsi.iloc[:sig_idx+1], 30)

            # Volume & pattern
            spike, decline = vol_exhaustion(sig_idx)
            pat_row = df15.iloc[sig_idx]
            pattern = "hammer" if PatternAnalyzer.is_hammer(pat_row) else "shooting_star" if PatternAnalyzer.is_shooting_star(pat_row) else ("doji" if PatternAnalyzer.is_doji(pat_row) else "")

            # Checklists
            long_ok = (((z_sig <= -TH_Z) and (bb_extreme if EXTREME_REQUIRES_BB_AND_Z else True)) or (not EXTREME_REQUIRES_BB_AND_Z and bb_extreme)) \
                      and (rsi_sig <= TH_RSI_OVERSOLD) \
                      and ((not REQUIRE_DIVERGENCE) or bul_div) \
                      and ((not REQUIRE_VOLUME_EXH) or (spike and decline)) \
                      and ((not REQUIRE_PATTERN) or (pattern in ("hammer","doji")))

            short_ok = (((z_sig >= TH_Z) and (bb_extreme if EXTREME_REQUIRES_BB_AND_Z else True)) or (not EXTREME_REQUIRES_BB_AND_Z and bb_extreme)) \
                       and (rsi_sig >= TH_RSI_OVERBOUGHT) \
                       and ((not REQUIRE_DIVERGENCE) or ber_div) \
                       and ((not REQUIRE_VOLUME_EXH) or (spike and decline)) \
                       and ((not REQUIRE_PATTERN) or (pattern in ("shooting_star","doji")))

            if not (long_ok or short_ok): continue
            direction = "LONG" if long_ok else "SHORT"

            entry_time = dtcol.iloc[i]
            if not self._trend_ok(df1h, entry_time, direction): continue
            if not self._stoch5_ok(df5, entry_time, direction): continue

            entry = float(open_.iloc[i])
            # compute sl/tp using signal-time SMA and std-based bands
            sl, tp1, _ = RiskManager.calc_targets(entry, sma_sig, lower_flex if not math.isnan(std_sig) else entry, upper_flex if not math.isnan(std_sig) else entry, direction)
            entry_net = entry * (1 + self.FEE_RATE) if direction == "LONG" else entry * (1 - self.FEE_RATE)

            open_positions.append({
                "side": direction,
                "entry_time": entry_time.to_pydatetime(),
                "entry": entry,
                "entry_net": entry_net,
                "sl": sl,
                "tp1": tp1,
                "expiry": entry_time.to_pydatetime() + timedelta(hours=HOLD_TIME_LIMIT_HOURS)
            })

        if open_positions:
            last_time = pd.to_datetime(df15["time"].iloc[-1], unit="s", utc=True).to_pydatetime()
            last_close = float(close.iloc[-1])
            for pos in open_positions:
                exit_px_net = last_close * (1 - self.FEE_RATE) if pos["side"] == "LONG" else last_close * (1 + self.FEE_RATE)
                rr = (exit_px_net - pos["entry_net"]) / (abs(pos["entry"] - pos["sl"]) + 1e-12) if pos["side"] == "LONG" else (pos["entry_net"] - exit_px_net) / (abs(pos["entry"] - pos["sl"]) + 1e-12)
                trades.append({"entry_time": pos["entry_time"], "exit_time": last_time, "side": pos["side"], "outcome": "end", "rr": rr, "hold_h": (last_time - pos["entry_time"]).total_seconds()/3600.0})

        if not trades:
            return {"pair": pair, "trades": 0, "win_rate": np.nan, "avg_rr": np.nan, "avg_hold_hours": np.nan, "max_concurrent": max_conc}

        df_tr = pd.DataFrame(trades)
        wins = (df_tr["outcome"] == "tp1").sum()
        win_rate = 100.0 * wins / len(df_tr)
        avg_rr = df_tr["rr"].replace([np.inf, -np.inf], np.nan).mean()
        avg_hold = df_tr["hold_h"].mean()

        return {"pair": pair, "trades": int(len(df_tr)), "win_rate": float(win_rate), "avg_rr": float(avg_rr), "avg_hold_hours": float(avg_hold), "max_concurrent": int(max_conc)}

    def run(self, pairs: List[str], months: int = 12) -> pd.DataFrame:
        rows = []
        for p in pairs:
            try:
                rows.append(self.run_pair(p, months=months))
            except Exception as e:
                logging.error("Backtest error for %s: %s", p, e)
        return pd.DataFrame(rows)

# ------------------------------ CLI ------------------------------
def main():
    import argparse
    parser = argparse.ArgumentParser(description="Coinbase Extreme Mean Reversion Scanner Bot")
    parser.add_argument("--loop", action="store_true", help="Run continuous scanning loop (every 15 minutes).")
    parser.add_argument("--once", action="store_true", help="Run one scan and send alerts for top setups.")
    parser.add_argument("--pairs", type=str, default="", help="Comma-separated list of pairs to scan/backtest.")
    parser.add_argument("--backtest", action="store_true", help="Run backtest over historical data for selected pairs.")
    parser.add_argument("--months", type=int, default=6, help="Backtest months lookback (approx).")
    parser.add_argument("--balance", type=float, default=10000.0, help="Account balance USD for sizing.")
    parser.add_argument("--risk", type=float, default=0.02, help="Risk per trade (e.g., 0.02 for 2%).")

    # Relaxation & bypass flags (single definition)
    parser.add_argument("--loose", action="store_true", help="Relax thresholds to generate more signals.")
    parser.add_argument("--veryloose", action="store_true", help="Relax even further (Z=2.0, RSI 25/75, BB=1.8σ).")
    parser.add_argument("--either", action="store_true", help="Accept either Z extreme or BB extreme (not both).")
    parser.add_argument("--nodev", action="store_true", help="Do not require RSI divergence.")
    parser.add_argument("--nopattern", action="store_true", help="Do not require candle pattern.")
    parser.add_argument("--novol", action="store_true", help="Do not require volume exhaustion.")
    parser.add_argument("--notrend", action="store_true", help="Disable 1h EMA50/200 trend filter.")
    parser.add_argument("--nostoch", action="store_true", help="Disable 5m Stoch K confirmation.")
    parser.add_argument("--stoch20", action="store_true", help="Relax 5m Stoch K to 20/80 instead of 10/90.")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    # Apply CLI toggles
    global REQUIRE_DIVERGENCE, REQUIRE_PATTERN, REQUIRE_VOLUME_EXH, EXTREME_REQUIRES_BB_AND_Z
    global TH_Z, TH_RSI_OVERSOLD, TH_RSI_OVERBOUGHT, TH_BB_STD_EXTREME, TH_VOL_SPIKE, TH_VOL_DECLINE
    global BYPASS_TREND, BYPASS_STOCH, RELAX_STOCH_20

    if args.nodev: REQUIRE_DIVERGENCE = False
    if args.nopattern: REQUIRE_PATTERN = False
    if args.novol: REQUIRE_VOLUME_EXH = False
    if args.either: EXTREME_REQUIRES_BB_AND_Z = False
    if args.loose:
        TH_Z = 2.3; TH_RSI_OVERSOLD = 20.0; TH_RSI_OVERBOUGHT = 80.0
        TH_BB_STD_EXTREME = 2.0; TH_VOL_SPIKE = 1.5; TH_VOL_DECLINE = 0.7
    if args.veryloose:
        TH_Z = 2.0; TH_RSI_OVERSOLD = 25.0; TH_RSI_OVERBOUGHT = 75.0
        TH_BB_STD_EXTREME = 1.8; TH_VOL_SPIKE = 1.2; TH_VOL_DECLINE = 0.9

    BYPASS_TREND = args.notrend
    BYPASS_STOCH = args.nostoch
    RELAX_STOCH_20 = args.stoch20

    pairs = [p.strip().upper() for p in args.pairs.split(",") if p.strip()] if args.pairs else COINBASE_PAIRS
    bot = CoinbaseMeanReversionBot(pairs=pairs)

    if args.backtest:
        bt = Backtester(bot.client)
        df = bt.run(pairs, months=args.months).sort_values("win_rate", ascending=False)
        total_trades = int(df["trades"].sum()) if not df.empty else 0
        avg_win = float(df["win_rate"].replace([np.inf, -np.inf], np.nan).mean()) if not df.empty else float("nan")
        avg_rr = float(df["avg_rr"].replace([np.inf, -np.inf], np.nan).mean()) if not df.empty else float("nan")
        print("=== Backtest Summary ===")
        print(f"Pairs: {', '.join(pairs)}")
        print(f"Months: {args.months}")
        print(f"Total trades: {total_trades}")
        print(f"Average win rate: {avg_win:.2f}%")
        print(f"Average RR (to TP1): {avg_rr:.2f}")
        print("\nPer-pair results:")
        print(df.to_string(index=False))
        return

    if args.once:
        ranked = bot.scan_all_pairs_once()
        if ranked:
            bot.execute_top_setups(ranked, balance_usd=args.balance, risk_per_trade=args.risk)
        else:
            print("No setups found.")
        return

    if args.loop or not (args.once or args.backtest):
        bot.loop()

if __name__ == "__main__":
    main()
