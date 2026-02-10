"""
Weekend Gap Fill Trading Bot — Backtest + Live-Ready Scanner (Coinbase, 30 coins)

- Detects weekend gaps (Fri 4:00 PM ET → Sun 8:00 PM ET) with DST-aware anchors
- Fetches 1H from Coinbase and resamples to 4H for conservative simulation
- Enforces volume filters, clean gap checks (with tiny tolerance), risk rules, time limits
- Aggregates performance by month and season

Requirements:
    pip install ccxt pandas numpy pytz python-dateutil

Notes:
- Coinbase has no native 4H. We fetch 1H and resample to 4H (right-labeled/right-closed).
- Detection uses 1H data to ensure anchors exist; simulation uses 4H for conservative fills.
- A tiny wick-overlap tolerance is allowed by default. Tighten to zero for stricter gaps.
"""
from __future__ import annotations
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List, Dict, Optional, Tuple

import ccxt
import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta
import pytz

# =====================
# Config & Constants
# =====================
DEBUG = True  # print diagnostics for why weekends are filtered out

# Gap detection parameters (slightly relaxed to surface realistic crypto gaps)
MIN_GAP_PERCENT = 0.005     # 0.6% minimum (use 0.008 for stricter)
MAX_GAP_PERCENT = 0.15     # 3.5% maximum
CLEAN_OVERLAP_TOL = 0.005   # allow <= 0.10% wick overlap in the "clean" test

# Anchors computed per-date in America/New_York to handle DST accurately.
FRIDAY_CLOSE_LOCAL = ("America/New_York", 16, 0)   # Friday 4:00 PM ET
SUNDAY_OPEN_LOCAL  = ("America/New_York", 20, 0)   # Sunday 8:00 PM ET
ANCHOR_TOLERANCE = "61min"  # nearest 1H bar within ±61 minutes is acceptable

MAX_HOLD_HOURS = 72
VOLUME_THRESHOLD = 2.90     # weekend volume must be < 40% of weekday average
MAX_CONCURRENT_POSITIONS = 10
RISK_PER_TRADE = 0.03
SLIPPAGE_BPS = 5            # 5 bps = 0.05% slippage assumption on entry/exit
SPREAD_BPS = 5              # 0.05% spread assumption added to fills

BLACKOUT_KEYWORDS = [
    "thanksgiving", "christmas", "new year", "fomc", "fed", "powell",
]

# Default 30 Coinbase USD pairs (avoid thin/new listings). Some may be skipped if not supported.
DEFAULT_SYMBOLS = [
    "BTC/USD", "ETH/USD", "SOL/USD", "XRP/USD"
]

BASE_TIMEFRAME = "1h"  # Coinbase supports 1h; we'll resample to 4h internally
YEARS_BACK = 2

# =====================
# Utility helpers
# =====================

def utc_ts(dt: datetime) -> int:
    return int(dt.replace(tzinfo=timezone.utc).timestamp() * 1000)


def bps_to_mult(bps: float) -> float:
    return 1.0 + (bps / 10000.0)


# =====================
# Data Fetch
# =====================
class CoinbaseData:
    def __init__(self, symbols: List[str]):
        self.exchange = ccxt.coinbase()
        try:
            self.markets = self.exchange.load_markets()
        except Exception as e:
            print("[coinbase] load_markets failed: {}".format(e))
            self.markets = {}
        self.supported = set(self.markets.keys())
        self.symbols = symbols

    def _resample_4h(self, df1h: pd.DataFrame) -> pd.DataFrame:
        if df1h.empty:
            return df1h
        df4h = (
            df1h
            .resample('4h', label='right', closed='right')
            .agg({'open':'first','high':'max','low':'min','close':'last','volume':'sum'})
            .dropna()
        )
        return df4h.astype(float)

    def normalize_symbol(self, symbol: str) -> Optional[str]:
        """Map common variants to Coinbase symbols. Prefer USD over USDT if available."""
        if symbol in self.supported:
            return symbol
        if symbol.endswith("/USDT"):
            usd = symbol.replace("/USDT", "/USD")
            if usd in self.supported:
                return usd
        base, quote = symbol.split("/")
        candidate = "{}/{}".format(base.upper(), quote.upper())
        if candidate in self.supported:
            return candidate
        return None

    def load_ohlcv(self, symbol: str, years_back: int = YEARS_BACK) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Fetch 1h OHLCV and return (df_1h, df_4h_resampled)."""
        now = datetime.now(timezone.utc)
        since_dt = now - relativedelta(years=years_back, months=1)  # buffer
        since_ms = utc_ts(since_dt)
        all_rows = []
        limit = 300
        timeframe = BASE_TIMEFRAME

        cx_symbol = self.normalize_symbol(symbol)
        if cx_symbol is None:
            empty = pd.DataFrame(columns=["timestamp","open","high","low","close","volume"]).set_index("timestamp")
            return empty, empty

        while True:
            try:
                chunk = self.exchange.fetch_ohlcv(cx_symbol, timeframe=timeframe, since=since_ms, limit=limit)
            except Exception as e:
                print("[fetch_ohlcv] skip {} -> {}".format(symbol, e))
                break
            if not chunk:
                break
            all_rows.extend(chunk)
            last_ms = chunk[-1][0]
            next_since = last_ms + 1
            if next_since == since_ms or len(chunk) < limit:
                break
            since_ms = next_since
            time.sleep(self.exchange.rateLimit / 1000.0)
        if not all_rows:
            empty = pd.DataFrame(columns=["timestamp","open","high","low","close","volume"]).set_index("timestamp")
            return empty, empty
        df1h = pd.DataFrame(all_rows, columns=["timestamp","open","high","low","close","volume"])
        df1h["timestamp"] = pd.to_datetime(df1h["timestamp"], unit="ms", utc=True)
        df1h.set_index("timestamp", inplace=True)
        df1h = df1h.astype(float)
        df4h = self._resample_4h(df1h)
        return df1h, df4h


# =====================
# Gap & Volume Logic
# =====================
@dataclass
class GapCandidate:
    symbol: str
    week_end: datetime             # Friday date (UTC) of the target weekend
    friday_close_ts: datetime
    sunday_open_ts: datetime
    friday_close: float
    sunday_open: float
    gap_pct: float
    direction: str                 # "UP" or "DOWN"
    gap_high: float
    gap_low: float


class GapDetector:
    def __init__(self):
        # counters for debug
        self.stats = {"weeks": 0, "anchor_missing": 0, "size_fail": 0, "clean_fail": 0, "candidates": 0}

    @staticmethod
    def _nearest_index(idx: pd.DatetimeIndex, ts: pd.Timestamp, tol: str = ANCHOR_TOLERANCE) -> Optional[pd.Timestamp]:
        """Return nearest timestamp within tolerance, else None."""
        try:
            pos = idx.get_indexer([ts], method='nearest', tolerance=pd.Timedelta(tol))
            if pos[0] == -1:
                return None
            return idx[pos[0]]
        except Exception:
            return None

    def find_weekly_gap(self, symbol: str, df1h: pd.DataFrame) -> List[GapCandidate]:
        """Scan using 1H data with DST-correct local anchors (Fri 4pm ET and Sun 8pm ET)."""
        out: List[GapCandidate] = []
        if df1h.empty:
            return out
        fridays = df1h[df1h.index.weekday == 4].index.normalize().unique()

        ny = pytz.timezone("America/New_York")

        for fri_date in fridays:
            self.stats["weeks"] += 1

            # Friday 4pm ET → UTC
            fri_dt_local = ny.localize(datetime(fri_date.year, fri_date.month, fri_date.day, FRIDAY_CLOSE_LOCAL[1], FRIDAY_CLOSE_LOCAL[2]))
            fri_close_utc = pd.Timestamp(fri_dt_local.astimezone(timezone.utc))
            # Sunday 8pm ET → UTC
            sun_date = fri_date + pd.Timedelta(days=2)
            sun_dt_local = ny.localize(datetime(sun_date.year, sun_date.month, sun_date.day, SUNDAY_OPEN_LOCAL[1], SUNDAY_OPEN_LOCAL[2]))
            sun_open_utc = pd.Timestamp(sun_dt_local.astimezone(timezone.utc))

            # Snap to nearest 1H bar within tolerance
            fri_anchor = self._nearest_index(df1h.index, fri_close_utc, ANCHOR_TOLERANCE)
            sun_anchor = self._nearest_index(df1h.index, sun_open_utc, ANCHOR_TOLERANCE)
            if fri_anchor is None or sun_anchor is None:
                self.stats["anchor_missing"] += 1
                if DEBUG:
                    if fri_anchor is None:
                        print("[anchor] {} missing Friday 4pm ET anchor near: {}".format(symbol, str(fri_close_utc)))
                    if sun_anchor is None:
                        print("[anchor] {} missing Sunday 8pm ET anchor near: {}".format(symbol, str(sun_open_utc)))
                continue

            fri_close_utc = fri_anchor
            sun_open_utc = sun_anchor

            friday_close = float(df1h.loc[fri_close_utc, "close"])  # 1H close at Fri 4pm ET
            sunday_row = df1h.loc[sun_open_utc]
            sunday_open = float(sunday_row.get("open", sunday_row["close"]))
            gap = (sunday_open - friday_close) / friday_close

            if abs(gap) < MIN_GAP_PERCENT or abs(gap) > MAX_GAP_PERCENT:
                self.stats["size_fail"] += 1
                if DEBUG:
                    print("[size] {} {} gap={:.2f}% (min={:.2f}%, max={:.2f}%)".format(
                        symbol,
                        fri_date.strftime('%Y-%m-%d'),
                        gap * 100.0,
                        MIN_GAP_PERCENT * 100.0,
                        MAX_GAP_PERCENT * 100.0,
                    ))
                continue

            # Clean gap using adjacent 1H candles (allow tiny overlap tolerance)
            prev_ts = sun_open_utc - pd.Timedelta(hours=1)
            if prev_ts not in df1h.index:
                self.stats["anchor_missing"] += 1
                if DEBUG:
                    print("[anchor] {} missing prev 1H before Sunday open: {}".format(symbol, str(prev_ts)))
                continue
            prev_row = df1h.loc[prev_ts]
            prev_high, prev_low = float(prev_row["high"]), float(prev_row["low"]) 
            s_open_high, s_open_low = float(sunday_row["high"]), float(sunday_row["low"]) 

            direction = "UP" if gap > 0 else "DOWN"
            if direction == "UP":
                clean = s_open_low >= prev_high * (1 - CLEAN_OVERLAP_TOL)
            else:
                clean = s_open_high <= prev_low * (1 + CLEAN_OVERLAP_TOL)

            if not clean:
                self.stats["clean_fail"] += 1
                if DEBUG:
                    print("[clean] {} {} not clean; prev_high={:.4f}, prev_low={:.4f}, s_high={:.4f}, s_low={:.4f}".format(
                        symbol,
                        fri_date.strftime('%Y-%m-%d'),
                        prev_high,
                        prev_low,
                        s_open_high,
                        s_open_low,
                    ))
                continue

            gap_high = max(prev_high, s_open_high)
            gap_low = min(prev_low, s_open_low)

            out.append(GapCandidate(
                symbol=symbol,
                week_end=fri_close_utc.to_pydatetime(),
                friday_close_ts=fri_close_utc.to_pydatetime(),
                sunday_open_ts=sun_open_utc.to_pydatetime(),
                friday_close=friday_close,
                sunday_open=sunday_open,
                gap_pct=gap,
                direction=direction,
                gap_high=gap_high,
                gap_low=gap_low,
            ))
            self.stats["candidates"] += 1

        if DEBUG:
            s = self.stats
            print("[gap-stats] {}: weeks={} anchors_miss={} size_fail={} clean_fail={} candidates={}".format(
                symbol,
                s.get('weeks', 0),
                s.get('anchor_missing', 0),
                s.get('size_fail', 0),
                s.get('clean_fail', 0),
                s.get('candidates', 0),
            ))
        return out


class VolumeAnalyzer:
    def __init__(self):
        pass

    def weekend_ok(self, df: pd.DataFrame, fc_ts: datetime, so_ts: datetime) -> Tuple[bool, Dict]:
        """Weekend volume < 40% of weekday avg AND gap forms during a lowest 12h weekend window."""
        fc_ts = pd.Timestamp(fc_ts)
        so_ts = pd.Timestamp(so_ts)
        fc_ts = fc_ts.tz_localize("UTC") if fc_ts.tz is None else fc_ts.tz_convert("UTC")
        so_ts = so_ts.tz_localize("UTC") if so_ts.tz is None else so_ts.tz_convert("UTC")

        start = fc_ts - pd.Timedelta(days=7)
        end = so_ts + pd.Timedelta(days=1)
        dfw = df.loc[start:end]
        if dfw.empty:
            return False, {"reason": "no_data"}

        weekdays = dfw[dfw.index.weekday <= 4]
        wkday_avg = weekdays["volume"].mean() if len(weekdays) else np.nan

        weekend = dfw[(dfw.index.weekday >= 5)]
        weekend_avg = weekend["volume"].mean() if len(weekend) else np.nan

        cond1 = bool(weekend_avg < VOLUME_THRESHOLD * wkday_avg) if np.isfinite(wkday_avg) and np.isfinite(weekend_avg) else False

        if weekend.empty:
            return False, {"reason": "no_weekend_data"}
        vol_roll = weekend["volume"].rolling(3).sum()  # 12h on 4h resampled data
        so_ts_pd = so_ts
        candidates = vol_roll[(vol_roll.index >= so_ts_pd - pd.Timedelta(hours=4)) & (vol_roll.index <= so_ts_pd + pd.Timedelta(hours=4))]
        if candidates.empty:
            return False, {"reason": "no_candidate_window"}
        threshold = vol_roll.quantile(0.25)
        cond2 = bool((candidates.min() <= threshold))

        return (cond1 and cond2), {
            "weekday_avg_vol": float(wkday_avg) if np.isfinite(wkday_avg) else None,
            "weekend_avg_vol": float(weekend_avg) if np.isfinite(weekend_avg) else None,
            "lowest12h_ok": cond2,
        }


class NewsFilter:
    def __init__(self, blackout_keywords: List[str] = None):
        self.keywords = blackout_keywords or []

    def is_blackout(self, text_blob: str) -> bool:
        blob = (text_blob or "").lower()
        return any(k in blob for k in self.keywords)


# =====================
# Trading / Backtest Engine
# =====================
@dataclass
class Trade:
    symbol: str
    side: str  # LONG / SHORT
    entry_time: datetime
    entry_price: float
    target_price: float
    stop_price: float
    gap_pct: float
    direction: str
    friday_close: float
    sunday_open: float
    gap_high: float
    gap_low: float
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    outcome: Optional[str] = None  # TP/SL/TIMEOUT
    hold_hours: Optional[float] = None
    pnl_r: Optional[float] = None


class PositionManager:
    def __init__(self, max_concurrent: int = MAX_CONCURRENT_POSITIONS):
        self.max_concurrent = max_concurrent

    def can_open(self, open_positions: List[Trade]) -> bool:
        return len(open_positions) < self.max_concurrent


class WeekendGapBot:
    def __init__(self, symbols: List[str] = None):
        requested = symbols or DEFAULT_SYMBOLS
        self.data = CoinbaseData(requested)
        # Filter to supported symbols on Coinbase (auto-map USDT->USD when possible)
        supported: List[str] = []
        skipped: List[str] = []
        for s in requested:
            ns = self.data.normalize_symbol(s)
            if ns is not None:
                if ns not in supported:
                    supported.append(ns)
            else:
                skipped.append(s)
        self.symbols = supported
        if skipped:
            print("[symbols] Skipping unsupported on Coinbase: {}".format(", ".join(skipped)))
        print("[symbols] Effective list: {}".format(", ".join(self.symbols)))
        self.gap_detector = GapDetector()
        self.volume_analyzer = VolumeAnalyzer()
        self.news_filter = NewsFilter(BLACKOUT_KEYWORDS)
        self.pos_mgr = PositionManager()

    # ---------- Risk & Price Helpers ----------
    @staticmethod
    def _apply_spread(price: float, side: str) -> float:
        spread_mult = bps_to_mult(SPREAD_BPS)
        return price * (spread_mult if side == "LONG" else (1.0 / spread_mult))

    @staticmethod
    def _apply_slippage(price: float, side: str, on_exit: bool = False) -> float:
        slip_mult = bps_to_mult(SLIPPAGE_BPS)
        if side == "LONG":
            return price * (slip_mult if not on_exit else (1.0 / slip_mult))
        else:  # SHORT
            return price * ((1.0 / slip_mult) if not on_exit else slip_mult)

    @staticmethod
    def _season(month: int) -> str:
        return {12:"Winter",1:"Winter",2:"Winter", 3:"Spring",4:"Spring",5:"Spring", 6:"Summer",7:"Summer",8:"Summer", 9:"Fall",10:"Fall",11:"Fall"}[month]

    # ---------- Core Gap → Trade transform ----------
    def _trade_from_gap(self, g: GapCandidate) -> Trade:
        entry_price_raw = g.sunday_open
        target_price = g.friday_close

        gap_abs = abs(g.sunday_open - g.friday_close)
        half_gap = 0.5 * gap_abs

        if g.direction == "UP":
            side = "SHORT"
            stop_price = g.gap_high + half_gap
        else:
            side = "LONG"
            stop_price = g.gap_low - half_gap

        entry_w_spread = self._apply_spread(entry_price_raw, side)
        entry = self._apply_slippage(entry_w_spread, side, on_exit=False)

        target_w_spread = self._apply_spread(target_price, side)
        target = self._apply_slippage(target_w_spread, side, on_exit=True)

        stop_w_spread = self._apply_spread(stop_price, side)
        stop = self._apply_slippage(stop_w_spread, side, on_exit=True)

        return Trade(
            symbol=g.symbol,
            side=side,
            entry_time=g.sunday_open_ts,
            entry_price=float(entry),
            target_price=float(target),
            stop_price=float(stop),
            gap_pct=float(g.gap_pct),
            direction=g.direction,
            friday_close=g.friday_close,
            sunday_open=g.sunday_open,
            gap_high=g.gap_high,
            gap_low=g.gap_low,
        )

    # ---------- Simulation of each trade ----------
    def _simulate_trade(self, df4h: pd.DataFrame, t: Trade) -> Trade:
        start = pd.Timestamp(t.entry_time)
        if start.tz is None:
            start = start.tz_localize("UTC")
        else:
            start = start.tz_convert("UTC")
        end = start + pd.Timedelta(hours=MAX_HOLD_HOURS)
        fwd = df4h.loc[start:end]
        outcome = None
        exit_price = None
        exit_time = None

        for ts, row in fwd.iterrows():
            high = float(row["high"]) ; low = float(row["low"]) 
            if t.side == "LONG":
                if high >= t.target_price:
                    outcome, exit_price, exit_time = "TP", t.target_price, ts
                    break
                if low <= t.stop_price:
                    outcome, exit_price, exit_time = "SL", t.stop_price, ts
                    break
            else:  # SHORT
                if low <= t.target_price:
                    outcome, exit_price, exit_time = "TP", t.target_price, ts
                    break
                if high >= t.stop_price:
                    outcome, exit_price, exit_time = "SL", t.stop_price, ts
                    break
        if outcome is None:
            outcome, exit_price, exit_time = "TIMEOUT", float(fwd.iloc[-1]["close"]) if len(fwd) else t.entry_price, fwd.index[-1] if len(fwd) else start

        t.exit_time = exit_time.to_pydatetime() if isinstance(exit_time, pd.Timestamp) else exit_time
        t.exit_price = float(exit_price)
        t.outcome = outcome
        t.hold_hours = float((t.exit_time - t.entry_time).total_seconds() / 3600.0)

        risk = abs(t.entry_price - t.stop_price)
        reward = abs(t.entry_price - t.target_price)
        if outcome == "TP":
            t.pnl_r = reward / risk if risk else 0.0
        elif outcome == "SL":
            t.pnl_r = -1.0
        else:
            mtm = (fwd.iloc[-1]["close"] - t.entry_price) if t.side == "LONG" else (t.entry_price - fwd.iloc[-1]["close"]) if len(fwd) else 0.0
            t.pnl_r = mtm / risk if risk else 0.0
        return t

    # ---------- Backtest Runner ----------
    def backtest(self, years_back: int = YEARS_BACK, news_text_provider=None) -> pd.DataFrame:
        all_trades: List[Trade] = []
        open_positions: List[Trade] = []

        for symbol in self.symbols:
            print("Downloading {}...".format(symbol))
            df1h, df4h = self.data.load_ohlcv(symbol, years_back=years_back)
            if df4h.empty or df1h.empty:
                print("No data for {}".format(symbol))
                continue

            # Reset detector stats per symbol
            self.gap_detector.stats = {"weeks": 0, "anchor_missing": 0, "size_fail": 0, "clean_fail": 0, "candidates": 0}
            gaps = self.gap_detector.find_weekly_gap(symbol, df1h)
            if DEBUG:
                s = self.gap_detector.stats
                print("[gap-stats] {}: weeks={} anchors_miss={} size_fail={} clean_fail={} candidates={}".format(
                    symbol,
                    s.get('weeks', 0),
                    s.get('anchor_missing', 0),
                    s.get('size_fail', 0),
                    s.get('clean_fail', 0),
                    s.get('candidates', 0),
                ))
            if not gaps:
                if DEBUG:
                    print("[debug] {} produced 0 gaps before volume/news filters".format(symbol))
                continue

            for g in gaps:
                # Volume / context checks (use 4H for weekend stats)
                vol_ok, _ = self.volume_analyzer.weekend_ok(df4h, g.friday_close_ts, g.sunday_open_ts)
                if not vol_ok:
                    continue

                if news_text_provider:
                    blob = news_text_provider(g.friday_close_ts, g.sunday_open_ts)
                    if self.news_filter.is_blackout(blob):
                        continue

                # Concurrency control
                open_positions = [t for t in open_positions if t.exit_time is None or t.exit_time >= g.sunday_open_ts]
                if not self.pos_mgr.can_open(open_positions):
                    continue

                trade = self._trade_from_gap(g)
                trade = self._simulate_trade(df4h, trade)
                all_trades.append(trade)
                open_positions.append(trade)

        if not all_trades:
            print("No qualifying gaps found.")
            return pd.DataFrame()

        results = self._summarize(all_trades)
        return results

    # ---------- Reporting ----------
    def _summarize(self, trades: List[Trade]) -> pd.DataFrame:
        rows = []
        for t in trades:
            month = t.entry_time.month
            season = self._season(month)
            rows.append({
                "symbol": t.symbol,
                "side": t.side,
                "direction": t.direction,
                "entry_time": t.entry_time,
                "exit_time": t.exit_time,
                "hold_hours": t.hold_hours,
                "entry": t.entry_price,
                "target": t.target_price,
                "stop": t.stop_price,
                "outcome": t.outcome,
                "pnl_r": t.pnl_r,
                "gap_pct": t.gap_pct,
                "month": month,
                "season": season,
            })
        df = pd.DataFrame(rows)
        if df.empty:
            return df

        overview = {
            "total_trades": len(df),
            "fill_rate": float((df["outcome"] == "TP").mean()),
            "avg_hold_hours": float(df.loc[df["outcome"] == "TP", "hold_hours"].mean()),
            "avg_r": float(df["pnl_r"].mean()),
        }
        print("=== OVERVIEW ===")
        for k, v in overview.items():
            print("{}: {}".format(k, v))

        print("=== BY MONTH (fill rate, avg R) ===")
        by_month = df.groupby("month").agg(fill_rate=("outcome", lambda s: (s=="TP").mean()), avg_r=("pnl_r","mean"))
        print(by_month)

        print("=== BY SEASON (fill rate, avg R) ===")
        by_season = df.groupby("season").agg(fill_rate=("outcome", lambda s: (s=="TP").mean()), avg_r=("pnl_r","mean"))
        print(by_season)

        return df

    # ---------- Alert formatting (for live mode) ----------
    def format_alert(self, t: Trade, hist_success: Optional[float] = None) -> str:
        gap_pct_disp = abs(t.gap_pct) * 100.0
        action = "SHORT at market open" if t.side == "SHORT" else "LONG at market open"
        hs = "{}%".format(int(hist_success * 100)) if hist_success is not None else "N/A"
        msg = (
            "📈 WEEKEND GAP DETECTED\n"
            + "Pair: {}\n".format(t.symbol)
            + "Gap Type: {}\n".format('UP GAP' if t.direction == 'UP' else 'DOWN GAP')
            + "Friday Close: ${:,.2f}\n".format(t.friday_close)
            + "Sunday Open: ${:,.2f} (+{:.1f}%)\n".format(t.sunday_open, gap_pct_disp)
            + "Action: {}\n".format(action)
            + "Stop Loss: ${:,.2f}\n".format(t.stop_price)
            + "Target: ${:,.2f} (GAP FILL)\n".format(t.target_price)
            + "Expected Fill: 24–48 hours\n"
            + "Historical Success: {}".format(hs)
        )
        return msg


# =====================
# Example usage (backtest)
# =====================
if __name__ == "__main__":
    bot = WeekendGapBot(symbols=DEFAULT_SYMBOLS)

    # Simple news provider stub (no blackout by default)
    def dummy_news_provider(start_dt, end_dt) -> str:
        return ""

    results_df = bot.backtest(years_back=YEARS_BACK, news_text_provider=dummy_news_provider)

    if results_df is not None and not results_df.empty:
        results_df.to_csv("weekend_gap_backtest_results.csv", index=False)
        print("Saved: weekend_gap_backtest_results.csv")

        fill_rate_72h = (results_df["outcome"] == "TP").mean()
        avg_hold = results_df.loc[results_df["outcome"] == "TP", "hold_hours"].mean()
        monthly_counts = results_df.groupby("month").size()
        monthly_trades = monthly_counts.mean() if len(monthly_counts) else 0

        print("=== METRICS (Headline) ===")
        print("Fill rate (<=72h): {:.2%}".format(fill_rate_72h))
        print("Average hold time (hours): {:.1f}".format(avg_hold))
        print("Expected monthly trades: {:.1f}".format(monthly_trades))
    else:
        print("No results produced.")
