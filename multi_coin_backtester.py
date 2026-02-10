"""
Multi-coin Coinbase backtester (6H candles by default)
------------------------------------------------------
- Fetches historical OHLCV from Coinbase public API (chunked to bypass 300-candle limit)
- Computes EMA(200), EMA(50), ATR(14), Stochastic RSI (14, 3, 3)
- Strategy:
    Long when (price > EMA200) AND (Stoch RSI K crosses above D AND K < 40)
    Short when (price < EMA200) AND (Stoch RSI K crosses below D AND K > 60)
- TP/SL via ATR multiples (TP=2.0 * ATR, SL=1.5 * ATR) in the trade direction
- One trade per coin per day (configurable)
- Cooldown: optional N candles after exit (default 0)
- Outputs per-coin CSV of trades + overall summary
- Includes a 0–10 setup score for each signal (trend, K/D slope, EMA stack, ATR regime)

Usage:
    python multi_coin_backtester.py \
        --products BTC-USD ETH-USD XRP-USD \
        --start 2024-01-01 --end 2025-09-01 \
        --granularity 21600 \
        --tp_mult 2.0 --sl_mult 1.5 \
        --max_trades_per_day 1

Notes:
- Coinbase candles endpoint returns candles in reverse chronological order.
- Granularity must be one of Coinbase's allowed values (in seconds):
  {60, 300, 900, 3600, 21600, 86400}. Use 21600 (6H) or 86400 (1D) for multi‑month spans.
- Backtest uses last completed candle only for entries.
- This script is self-contained (pandas/numpy/requests only).
"""
from __future__ import annotations
import argparse
import time
import math
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import requests

COINBASE_API = "https://api.exchange.coinbase.com"  # Formerly GDAX. Public candles.

# ----------------------------
# Utilities
# ----------------------------

def parse_date(s: str) -> datetime:
    return datetime.fromisoformat(s).replace(tzinfo=timezone.utc) if "T" in s else datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=timezone.utc)


def to_iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


# ----------------------------
# Data fetching
# ----------------------------

def fetch_candles(product_id: str, start: datetime, end: datetime, granularity: int) -> pd.DataFrame:
    """Fetch candles from Coinbase in chunks to bypass the 300-candle/server window.
    Returns DataFrame with columns: time, low, high, open, close, volume
    """
    assert granularity in {60, 300, 900, 3600, 21600, 86400}, "Invalid granularity"
    max_points = 300
    window_seconds = granularity * max_points

    # Coinbase returns [ time, low, high, open, close, volume ] descending by time
    all_rows: List[List[float]] = []

    chunk_start = start
    while chunk_start < end:
        chunk_end = min(chunk_start + timedelta(seconds=window_seconds), end)
        params = {
            "start": to_iso(chunk_start),
            "end": to_iso(chunk_end),
            "granularity": granularity,
        }
        url = f"{COINBASE_API}/products/{product_id}/candles"
        r = requests.get(url, params=params, timeout=20)
        if r.status_code != 200:
            raise RuntimeError(f"HTTP {r.status_code} fetching candles: {r.text}")
        rows = r.json()
        if not isinstance(rows, list):
            raise RuntimeError(f"Unexpected response: {rows}")
        all_rows.extend(rows)
        # Move window forward (avoid overlap):
        chunk_start = chunk_end
        time.sleep(0.2)  # be polite

    if not all_rows:
        raise RuntimeError("No candle data fetched.")

    df = pd.DataFrame(all_rows, columns=["time", "low", "high", "open", "close", "volume"])  # descending order
    df = df.sort_values("time").reset_index(drop=True)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    return df


# ----------------------------
# Indicators
# ----------------------------

def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr1 = df["high"] - df["low"]
    tr2 = (df["high"] - prev_close).abs()
    tr3 = (df["low"] - prev_close).abs()
    return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    return true_range(df).rolling(window=period, min_periods=period).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(window=period, min_periods=period).mean()
    loss = -delta.clip(upper=0).rolling(window=period, min_periods=period).mean()
    rs = gain / loss.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))
    return out


def stoch_rsi(series: pd.Series, rsi_period: int = 14, stoch_period: int = 14, k_period: int = 3, d_period: int = 3) -> Tuple[pd.Series, pd.Series]:
    r = rsi(series, rsi_period)
    min_r = r.rolling(window=stoch_period, min_periods=stoch_period).min()
    max_r = r.rolling(window=stoch_period, min_periods=stoch_period).max()
    stoch = (r - min_r) / (max_r - min_r)
    k = stoch.rolling(window=k_period, min_periods=k_period).mean() * 100
    d = k.rolling(window=d_period, min_periods=d_period).mean()
    return k, d


# ----------------------------
# Strategy
# ----------------------------
@dataclass
class StrategyConfig:
    ema_fast: int = 50
    ema_slow: int = 200
    atr_period: int = 14
    st_rsi_len: int = 14
    st_k: int = 3
    st_d: int = 3
    tp_mult: float = 2.0
    sl_mult: float = 1.5
    max_trades_per_day: int = 1
    cooldown_candles_after_exit: int = 0


def compute_indicators(df: pd.DataFrame, cfg: StrategyConfig) -> pd.DataFrame:
    df = df.copy()
    df["ema_fast"] = ema(df["close"], cfg.ema_fast)
    df["ema_slow"] = ema(df["close"], cfg.ema_slow)
    df["atr"] = atr(df, cfg.atr_period)
    k, d = stoch_rsi(df["close"], cfg.st_rsi_len, cfg.st_rsi_len, cfg.st_k, cfg.st_d)
    df["stoch_k"] = k
    df["stoch_d"] = d
    # Cross signals (using previous candle to ensure closed signal):
    df["kd_cross_up"] = (df["stoch_k"].shift(1) < df["stoch_d"].shift(1)) & (df["stoch_k"] >= df["stoch_d"])  # K crosses above D
    df["kd_cross_dn"] = (df["stoch_k"].shift(1) > df["stoch_d"].shift(1)) & (df["stoch_k"] <= df["stoch_d"])  # K crosses below D
    return df


def setup_score(row: pd.Series) -> float:
    """0–10 score: trend(0-4) + stoch location(0-3) + EMA stack(0-2) + ATR regime(0-1)."""
    score = 0.0
    # Trend weight (0-4): distance from EMA200
    if not np.isnan(row.get("ema_slow", np.nan)):
        dist = (row["close"] - row["ema_slow"]) / row["ema_slow"]
        if dist >= 0.02:
            score += 4
        elif dist >= 0.0:
            score += 2
        elif dist <= -0.02:
            score += 4
        else:
            score += 2
    # Stoch location (0-3)
    k = row.get("stoch_k", np.nan)
    d = row.get("stoch_d", np.nan)
    if not np.isnan(k) and not np.isnan(d):
        if k < 20 and k > d:
            score += 3
        elif k > 80 and k < d:
            score += 3
        else:
            score += 1
    # EMA stack (0-2)
    if not np.isnan(row.get("ema_fast", np.nan)) and not np.isnan(row.get("ema_slow", np.nan)):
        score += 2 if (row["ema_fast"] > row["ema_slow"]) == (row["close"] > row["ema_slow"]) else 0
    # ATR regime (0-1): avoid ultra-low ATR; award 1 if ATR > 0
    score += 1 if row.get("atr", 0) > 0 else 0
    return round(min(score, 10.0), 2)


@dataclass
class Trade:
    symbol: str
    entry_time: pd.Timestamp
    side: str  # "long" or "short"
    entry: float
    tp: float
    sl: float
    exit_time: Optional[pd.Timestamp] = None
    exit_price: Optional[float] = None
    outcome: Optional[str] = None  # "TP" | "SL"
    r_multiple: Optional[float] = None
    setup_score: Optional[float] = None


# ----------------------------
# Backtester
# ----------------------------

def backtest_product(df: pd.DataFrame, product: str, cfg: StrategyConfig, granularity: int) -> Tuple[pd.DataFrame, List[Trade]]:
    df = compute_indicators(df, cfg)
    df["date"] = df["time"].dt.tz_convert("America/Los_Angeles").dt.date

    trades: List[Trade] = []
    candles_since_exit_cooldown = 0
    last_trade_date: Optional[datetime.date] = None

    i = 1  # start at 1 to have previous candle for cross detection
    while i < len(df):
        row = df.iloc[i]
        prev = df.iloc[i - 1]

        # Enforce cooldown after exit
        if candles_since_exit_cooldown > 0:
            candles_since_exit_cooldown -= 1
            i += 1
            continue

        # Limit trades per day
        if last_trade_date == row["date"] and cfg.max_trades_per_day <= sum(1 for t in trades if t.entry_time.date() == last_trade_date):
            i += 1
            continue

        price = row["close"]
        ema200 = row["ema_slow"]
        atr_val = row["atr"]
        k, d = row["stoch_k"], row["stoch_d"]
        setup = setup_score(row)

        open_new_trade = None  # (side, entry, tp, sl)

        # Long conditions
        if (
            price > ema200
            and row["kd_cross_up"]
            and k < 40
            and not np.isnan(atr_val) and atr_val > 0
        ):
            entry = price
            tp = entry + cfg.tp_mult * atr_val
            sl = entry - cfg.sl_mult * atr_val
            open_new_trade = ("long", entry, tp, sl)

        # Short conditions
        if (
            price < ema200
            and row["kd_cross_dn"]
            and k > 60
            and not np.isnan(atr_val) and atr_val > 0
        ):
            entry = price
            tp = entry - cfg.tp_mult * atr_val
            sl = entry + cfg.sl_mult * atr_val
            open_new_trade = ("short", entry, tp, sl)

        if open_new_trade:
            side, entry, tp, sl = open_new_trade
            trade = Trade(
                symbol=product,
                entry_time=row["time"],
                side=side,
                entry=float(entry),
                tp=float(tp),
                sl=float(sl),
                setup_score=setup,
            )
            # Simulate forward to exit
            j = i + 1
            exit_found = False
            while j < len(df):
                hi = df.iloc[j]["high"]
                lo = df.iloc[j]["low"]
                if side == "long":
                    # Hit SL first?
                    if lo <= trade.sl:
                        trade.exit_time = df.iloc[j]["time"]
                        trade.exit_price = trade.sl
                        trade.outcome = "SL"
                        risk = trade.entry - trade.sl
                        trade.r_multiple = (trade.exit_price - trade.entry) / risk if risk != 0 else -1
                        exit_found = True
                        break
                    if hi >= trade.tp:
                        trade.exit_time = df.iloc[j]["time"]
                        trade.exit_price = trade.tp
                        trade.outcome = "TP"
                        risk = trade.entry - trade.sl
                        trade.r_multiple = (trade.exit_price - trade.entry) / risk if risk != 0 else 2
                        exit_found = True
                        break
                else:  # short
                    if hi >= trade.sl:
                        trade.exit_time = df.iloc[j]["time"]
                        trade.exit_price = trade.sl
                        trade.outcome = "SL"
                        risk = trade.sl - trade.entry
                        trade.r_multiple = (trade.entry - trade.exit_price) / risk if risk != 0 else -1
                        exit_found = True
                        break
                    if lo <= trade.tp:
                        trade.exit_time = df.iloc[j]["time"]
                        trade.exit_price = trade.tp
                        trade.outcome = "TP"
                        risk = trade.sl - trade.entry
                        trade.r_multiple = (trade.entry - trade.exit_price) / risk if risk != 0 else 2
                        exit_found = True
                        break
                j += 1

            if exit_found:
                trades.append(trade)
                last_trade_date = trade.entry_time.tz_convert("America/Los_Angeles").date()
                candles_since_exit_cooldown = cfg.cooldown_candles_after_exit
                # Move i to j to avoid overlapping trades
                i = j + 1
                continue

        i += 1

    # Build trades DataFrame
    trades_df = pd.DataFrame([t.__dict__ for t in trades]) if trades else pd.DataFrame(columns=[
        "symbol", "entry_time", "side", "entry", "tp", "sl", "exit_time", "exit_price", "outcome", "r_multiple", "setup_score"
    ])
    return trades_df, trades


# ----------------------------
# Main orchestration
# ----------------------------

def run_backtest(products: List[str], start: datetime, end: datetime, granularity: int, cfg: StrategyConfig, out_dir: str = "backtests") -> pd.DataFrame:
    os.makedirs(out_dir, exist_ok=True)
    all_results: List[pd.DataFrame] = []

    for product in products:
        print(f"Fetching {product} candles...")
        df = fetch_candles(product, start, end, granularity)
        print(f"{product}: {len(df)} candles fetched from {df['time'].iloc[0]} to {df['time'].iloc[-1]}")
        trades_df, _ = backtest_product(df, product, cfg, granularity)
        if not trades_df.empty:
            trades_df.to_csv(os.path.join(out_dir, f"{product.replace('-', '_')}_trades.csv"), index=False)
        all_results.append(trades_df)

    combined = pd.concat(all_results, ignore_index=True) if all_results else pd.DataFrame()
    if not combined.empty:
        combined["entry_time"] = pd.to_datetime(combined["entry_time"], utc=True)
        combined["exit_time"] = pd.to_datetime(combined["exit_time"], utc=True)
        # Summary
        summary = (
            combined.assign(win=lambda d: (d["outcome"] == "TP").astype(int))
                    .groupby("symbol")
                    .agg(trades=("symbol", "count"),
                         win_rate=("win", lambda x: 100 * x.mean() if len(x) else 0.0),
                         avg_r=("r_multiple", "mean"))
                    .reset_index()
        )
        summary.to_csv(os.path.join(out_dir, "summary_by_symbol.csv"), index=False)
        overall = {
            "trades": int(len(combined)),
            "win_rate": round(100 * (combined["outcome"] == "TP").mean(), 2),
            "avg_r": round(combined["r_multiple"].mean(), 3),
        }
        print("\nOverall:", overall)
        print("Saved:", os.path.join(out_dir, "summary_by_symbol.csv"))
    else:
        print("No trades generated.")

    return combined


def load_products(args) -> List[str]:
    # Priority: explicit --products > products file > default curated list
    removed = {
        "velo-usd","apt-usd","syrup-usd","axs-usd","sand-usd","pepe-usd","wif-usd","toshi-usd","moodeng-usd","tao-usd","morpho-usd","tia-usd","fartcoin-usd","popcat-usd","fet-usd","turbo-usd","crv-usd","magic-usd","pnut-usd",
    }
    def norm(s: str) -> str:
        return s.strip()

    if args.products:
        prods = [norm(p) for p in args.products]
    elif args.products_file:
        with open(args.products_file, "r") as f:
            prods = [norm(line) for line in f if line.strip() and not line.strip().startswith("#")]
    else:
        # Curated default: broad majors + liquid alts; underperformers removed.
        default = [
            "BTC-USD","ETH-USD","XRP-USD","SOL-USD","ADA-USD","AVAX-USD","LINK-USD","LTC-USD","BCH-USD","DOT-USD","MATIC-USD",
            "SHIB-USD","OP-USD","ARB-USD","INJ-USD","NEAR-USD","ATOM-USD","FIL-USD","AAVE-USD","UNI-USD","ETC-USD","XLM-USD",
            "SEI-USD","RUNE-USD","SUI-USD","RNDR-USD","TIA-USD","FET-USD","CRV-USD","AXS-USD","SAND-USD","PEPE-USD","WIF-USD",
        ]
        prods = default

    # Filter out removed, but keep case as provided
    prods_filtered = []
    for p in prods:
        if p.lower() in removed:
            continue
        prods_filtered.append(p)

    # De-duplicate preserving order
    seen = set()
    final = []
    for p in prods_filtered:
        if p not in seen:
            final.append(p)
            seen.add(p)
    if not final:
        raise ValueError("Product list ended up empty after filtering. Provide --products or adjust products file.")
    return final


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--products", nargs="+", help="e.g., BTC-USD ETH-USD XRP-USD (overrides default)")
    parser.add_argument("--products_file", help="Path to a newline-separated list of products (e.g., coins.txt)")
    parser.add_argument("--start", required=True, help="YYYY-MM-DD or ISO8601")
    parser.add_argument("--end", required=True, help="YYYY-MM-DD or ISO8601")
    parser.add_argument("--granularity", type=int, default=21600, help="Seconds (21600 = 6H)")
    parser.add_argument("--tp_mult", type=float, default=2.0)
    parser.add_argument("--sl_mult", type=float, default=1.5)
    parser.add_argument("--max_trades_per_day", type=int, default=1)
    parser.add_argument("--cooldown_after_exit", type=int, default=0, help="Candles to wait after exit")
    args = parser.parse_args()

    cfg = StrategyConfig(
        tp_mult=args.tp_mult,
        sl_mult=args.sl_mult,
        max_trades_per_day=args.max_trades_per_day,
        cooldown_candles_after_exit=args.cooldown_after_exit,
    )

    start = parse_date(args.start)
    end = parse_date(args.end)

    products = load_products(args)

    run_backtest(products, start, end, args.granularity, cfg)


if __name__ == "__main__":
    main()

    main()
