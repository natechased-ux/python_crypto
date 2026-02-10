# BB SR/Structure Strategy — V2: Key SR Zones (1D zones, 1H entries)
# -------------------------------------------------------------------
# What changed vs V1:
# - SR zones are built ONLY from Daily pivots (1D), merged by ATR(1D), scored by touches,
#   and limited to top-K zones (MUCH fewer + stronger).
# - Entries happen on 1H, only when price interacts with a key 1D zone.
# - Optional relative-volume spike confirmation on 1H to cut noise.
# - Exits: TP1=1R (move SL to BE), TP2=5R (skew expectancy), optional ATR trail.
# - Hygiene: one-trade-per-zone-per-day.
#
# Usage:
#   pip install pandas numpy requests
#   python bb_sr_keyzones_backtest.py
#
# Outputs:
#   bb_sr_v2_trades.csv
#   bb_sr_v2_summary.csv
#
# NOTE: Requires internet (Coinbase public API). This environment might not fetch;
#       run locally.

import time
from datetime import datetime, timedelta, timezone, date
from typing import List, Tuple, Dict, Optional
import numpy as np
import pandas as pd

try:
    import requests
except Exception:
    requests = None

# ------------------------------
# Config
# ------------------------------
COINS = [
    "BTC-USD", "ETH-USD", "XRP-USD", "SOL-USD", "ADA-USD", "AVAX-USD", "DOGE-USD", "LTC-USD"
]
# Coinbase granularities (seconds): 60, 300, 900, 3600, 21600, 86400
TF_TRADE = 3600   # 1H entries
TF_ZONES = 86400  # 1D zones
DAYS_BACK = 365   # how much history to fetch

# Zone construction (1D)
PIVOT_LEFT_1D = 5
PIVOT_RIGHT_1D = 5
ATR_LEN_1D = 14
ZONE_THICKNESS_ATR = 0.40       # thickness of a zone = ±0.40*ATR(1D) around pivot price
ZONE_MERGE_WITHIN_ATR = 1.50    # merge zones whose centers are within 1.5*ATR(1D)
MIN_TOUCHES_1D = 2              # require >=2 touches (multi-test) to keep a zone
TOP_K_ZONES = 6                 # keep only the strongest K zones (by touch count, then recency)

# Entry filters (1H)
EMA_LONG = 200
ATR_LEN_1H = 14
ZONE_TOUCH_ATR_1H = 0.60        # consider “touch” if price within 0.60*ATR(1H) of zone band
REQ_REL_VOL_SPIKE = True
REL_VOL_LOOKBACK = 20
REL_VOL_MULT = 1.5              # vol spike if vol > 1.5 * median(20)

# Confirmation on 1H candle
USE_STOPHUNT_CONF = True        # wick through zone & close back inside counts as confirmation
USE_CANDLE_CONF = True          # engulfing or pin bar

# Exits / Risk
TP1_R = 1.0                     # take partial at 1R
TP1_PART = 0.5                  # 50% off at TP1
TP2_R = 5.0                     # final target far to create expectancy skew
TRAIL_AFTER_TP1 = True
TRAIL_ATR_MULT = 2.0
ATR_BUFFER_ENTRY_SL = 0.10      # add/subtract 0.10*ATR(1H) beyond wick/zone to place SL

# Hygiene
ONE_TRADE_PER_ZONE_PER_DAY = True

# ------------------------------
# Data fetch
# ------------------------------
def coinbase_fetch_candles(product_id: str, granularity: int, start: datetime, end: datetime) -> pd.DataFrame:
    if requests is None:
        raise RuntimeError("requests not available. Install requests and run with internet.")
    url = f"https://api.exchange.coinbase.com/products/{product_id}/candles"
    params = {
        "granularity": granularity,
        "start": start.replace(tzinfo=timezone.utc).isoformat(),
        "end": end.replace(tzinfo=timezone.utc).isoformat(),
    }
    for _ in range(5):
        r = requests.get(url, params=params, headers={"User-Agent": "bb-sr-keyzones"})
        if r.status_code == 200:
            data = r.json()
            if not data:
                return pd.DataFrame(columns=["time", "low", "high", "open", "close", "volume"])
            df = pd.DataFrame(data, columns=["time", "low", "high", "open", "close", "volume"])
            df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
            df = df.sort_values("time").reset_index(drop=True)
            return df
        time.sleep(0.4)
    raise RuntimeError(f"Failed fetching candles for {product_id}: {r.status_code} {r.text}")

def coinbase_fetch_days(product_id: str, granularity: int, days_back: int) -> pd.DataFrame:
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days_back)
    bars_per_call = 300
    step = timedelta(seconds=granularity * bars_per_call)
    out = []
    cursor = start
    while cursor < end:
        w_end = min(cursor + step, end)
        df = coinbase_fetch_candles(product_id, granularity, cursor, w_end)
        if not df.empty:
            out.append(df)
        cursor = w_end
        time.sleep(0.2)
    if not out:
        return pd.DataFrame(columns=["time", "low", "high", "open", "close", "volume"])
    df = pd.concat(out, ignore_index=True).drop_duplicates(subset=["time"]).sort_values("time")
    return df.reset_index(drop=True)

# ------------------------------
# Indicators
# ------------------------------
def ema(s: pd.Series, length: int) -> pd.Series:
    return s.ewm(span=length, adjust=False).mean()

def atr(df: pd.DataFrame, length: int) -> pd.Series:
    high = df["high"]; low = df["low"]; close = df["close"].shift(1)
    tr = pd.DataFrame({
        "hl": high - low,
        "hc": (high - close).abs(),
        "lc": (low - close).abs()
    }).max(axis=1)
    return tr.rolling(length).mean()

# ------------------------------
# Pivots / Zones on 1D
# ------------------------------
def is_pivot_high(df: pd.DataFrame, i: int, left: int, right: int) -> bool:
    if i - left < 0 or i + right >= len(df): return False
    hi = df.loc[i, "high"]
    return all(df.loc[i-k, "high"] < hi for k in range(1, left+1)) and \
           all(df.loc[i+k, "high"] < hi for k in range(1, right+1))

def is_pivot_low(df: pd.DataFrame, i: int, left: int, right: int) -> bool:
    if i - left < 0 or i + right >= len(df): return False
    lo = df.loc[i, "low"]
    return all(df.loc[i-k, "low"] > lo for k in range(1, left+1)) and \
           all(df.loc[i+k, "low"] > lo for k in range(1, right+1))

def build_daily_key_zones(df_1d: pd.DataFrame) -> List[Tuple[float, float, int, int]]:
    """
    Build SR zones from 1D pivots, merge, count touches, filter to top-K.
    Return list of tuples: (zlo, zhi, last_touch_idx, touches)
    """
    d = df_1d.copy()
    d["atr1d"] = atr(d, ATR_LEN_1D)
    zones = []
    for i in range(len(d)):
        a = d.loc[i, "atr1d"]
        if np.isnan(a) or a <= 0: continue
        if is_pivot_high(d, i, PIVOT_LEFT_1D, PIVOT_RIGHT_1D):
            p = d.loc[i, "high"]
            zlo, zhi = p - ZONE_THICKNESS_ATR*a, p + ZONE_THICKNESS_ATR*a
            zones.append((zlo, zhi, i))
        if is_pivot_low(d, i, PIVOT_LEFT_1D, PIVOT_RIGHT_1D):
            p = d.loc[i, "low"]
            zlo, zhi = p - ZONE_THICKNESS_ATR*a, p + ZONE_THICKNESS_ATR*a
            zones.append((zlo, zhi, i))

    if not zones: return []

    # Merge zones whose centers are close (within MERGE_ATR * latest ATR)
    latest_atr = d["atr1d"].iloc[-1] if not np.isnan(d["atr1d"].iloc[-1]) else d["atr1d"].dropna().iloc[-1]
    zones.sort(key=lambda z: (z[0]+z[1])/2.0)
    merged: List[Tuple[float, float, int]] = []
    for z in zones:
        if not merged:
            merged.append(z); continue
        lo, hi, idx = z
        mlo, mhi, midx = merged[-1]
        if abs(((lo+hi)/2)-((mlo+mhi)/2)) <= ZONE_MERGE_WITHIN_ATR * latest_atr:
            merged[-1] = (min(mlo, lo), max(mhi, hi), max(midx, idx))
        else:
            merged.append(z)

    # Score zones by number of 1D touches (body/close near zone)
    scored = []
    for (zlo, zhi, last_idx) in merged:
        # touches: count days where close is inside band or within 0.25*ATR(1D) from band
        touches = 0
        for j in range(len(d)):
            a = d.loc[j, "atr1d"]
            if np.isnan(a) or a <= 0: continue
            close = d.loc[j, "close"]
            # distance to band if outside
            if zlo <= close <= zhi:
                touches += 1
            else:
                dist = min(abs(close - zlo), abs(close - zhi))
                if dist <= 0.25*a:
                    touches += 1
        scored.append((zlo, zhi, last_idx, touches))

    # Filter: require multi-test (touches >= MIN_TOUCHES_1D), then keep TOP_K_ZONES by (touches desc, recency)
    filtered = [z for z in scored if z[3] >= MIN_TOUCHES_1D]
    filtered.sort(key=lambda z: (z[3], z[2]), reverse=True)
    return filtered[:TOP_K_ZONES]

# ------------------------------
# 1H confirmation helpers
# ------------------------------
def candlestick_signals(df: pd.DataFrame, i: int) -> Dict[str, bool]:
    if i < 1:
        return {"bull_engulf": False, "bear_engulf": False, "bull_pin": False, "bear_pin": False}
    o1, c1 = df.loc[i-1, "open"], df.loc[i-1, "close"]
    o2, c2, h2, l2 = df.loc[i, "open"], df.loc[i, "close"], df.loc[i, "high"], df.loc[i, "low"]
    body_prev = abs(c1 - o1)
    body = abs(c2 - o2)
    bull_engulf = c2 > o2 and o2 <= c1 and c2 >= o1 and body > 0.5 * max(body_prev, 1e-9)
    bear_engulf = c2 < o2 and o2 >= c1 and c2 <= o1 and body > 0.5 * max(body_prev, 1e-9)
    lower_wick = min(o2, c2) - l2
    upper_wick = h2 - max(o2, c2)
    bull_pin = lower_wick >= 2 * body and c2 > o2
    bear_pin = upper_wick >= 2 * body and c2 < o2
    return {"bull_engulf": bool(bull_engulf), "bear_engulf": bool(bear_engulf),
            "bull_pin": bool(bull_pin), "bear_pin": bool(bear_pin)}

def rel_vol_spike(df: pd.DataFrame, i: int, lookback: int, mult: float) -> bool:
    if i < lookback: return False
    med = df["volume"].iloc[i-lookback:i].median()
    return df.loc[i, "volume"] > mult * max(med, 1e-12)

def stophunt_wick_through_zone_and_close_inside(df: pd.DataFrame, i: int, zlo: float, zhi: float) -> Dict[str, bool]:
    """
    Long stop-hunt: low < zlo and close > zlo
    Short stop-hunt: high > zhi and close < zhi
    """
    o, c, h, l = df.loc[i, "open"], df.loc[i, "close"], df.loc[i, "high"], df.loc[i, "low"]
    long_ok = (l < zlo) and (c > zlo)
    short_ok = (h > zhi) and (c < zhi)
    return {"long": long_ok, "short": short_ok}

# ------------------------------
# Backtest core
# ------------------------------
def backtest_coin(product: str, df_1h: pd.DataFrame, df_1d: pd.DataFrame) -> pd.DataFrame:
    h = df_1h.copy()
    d = df_1d.copy()

    # Indicators
    h["ema200"] = ema(h["close"], EMA_LONG)
    h["atr1h"] = atr(h, ATR_LEN_1H)
    # build key zones from 1D
    key_zones = build_daily_key_zones(d)  # list of (zlo, zhi, last_idx, touches)
    if not key_zones:
        return pd.DataFrame(columns=[])

    trades = []
    in_position = False
    side = None
    entry = sl = tp1 = tp2 = None
    be_moved = False
    trail_active = False
    trail_stop = None
    zone_last_trade_day: Dict[int, date] = {}  # zone_id -> last trade date

    def can_trade_zone_today(zone_id: int, t_utc: pd.Timestamp) -> bool:
        if not ONE_TRADE_PER_ZONE_PER_DAY: return True
        day = t_utc.tz_convert(None).date() if t_utc.tzinfo else t_utc.date()
        last = zone_last_trade_day.get(zone_id)
        return (last is None) or (last != day)

    for i in range(len(h)):
        if i < max(EMA_LONG, ATR_LEN_1H) + 5:
            continue
        a = h.loc[i, "atr1h"]
        if np.isnan(a) or a <= 0: continue

        price = h.loc[i, "close"]
        ema200 = h.loc[i, "ema200"]
        tstamp = h.loc[i, "time"]

        # find the nearest key zone in reach (within band + touch tolerance)
        touched = None  # (zone_id, zlo, zhi)
        for zone_id, (zlo, zhi, _idx, _touches) in enumerate(key_zones):
            # consider inside band as touch; or close to edges within ZONE_TOUCH_ATR_1H * ATR(1H)
            dist = 0.0
            if price < zlo: dist = zlo - price
            elif price > zhi: dist = price - zhi
            else: dist = 0.0
            if dist <= ZONE_TOUCH_ATR_1H * a:
                if can_trade_zone_today(zone_id, tstamp):
                    touched = (zone_id, zlo, zhi)
                    break
        if touched is None:
            continue
        zone_id, zlo, zhi = touched

        # confirmations
        sigs = candlestick_signals(h, i)
        wick_conf = stophunt_wick_through_zone_and_close_inside(h, i, zlo, zhi)
        vol_ok = (not REQ_REL_VOL_SPIKE) or rel_vol_spike(h, i, REL_VOL_LOOKBACK, REL_VOL_MULT)

        # LONG setup: above EMA200 and (stop-hunt long OR (candle conf bullish AND vol_ok))
        if (price > ema200) and (
            (USE_STOPHUNT_CONF and wick_conf["long"]) or
            (USE_CANDLE_CONF and (sigs["bull_engulf"] or sigs["bull_pin"]) and vol_ok)
        ):
            stop = min(h.loc[i, "low"], zlo) - ATR_BUFFER_ENTRY_SL * a
            if stop < price:
                R = price - stop
                take1 = price + TP1_R * R
                take2 = price + TP2_R * R
                trades.append({
                    "product": product, "open_time": tstamp, "side": "long",
                    "entry": price, "sl": stop, "tp1": take1, "tp2": take2,
                    "R": R, "status": "open", "pnl_R": 0.0, "partial1": False, "zone_id": zone_id
                })
                in_position = True; side = "long"; entry, sl, tp1, tp2 = price, stop, take1, take2
                be_moved = False; trail_active = False; trail_stop = None
                zone_last_trade_day[zone_id] = (tstamp.tz_convert(None).date() if tstamp.tzinfo else tstamp.date())
                continue

        # SHORT setup: below EMA200 and (stop-hunt short OR (bearish candle conf AND vol_ok))
        if (price < ema200) and (
            (USE_STOPHUNT_CONF and wick_conf["short"]) or
            (USE_CANDLE_CONF and (sigs["bear_engulf"] or sigs["bear_pin"]) and vol_ok)
        ):
            stop = max(h.loc[i, "high"], zhi) + ATR_BUFFER_ENTRY_SL * a
            if stop > price:
                R = stop - price
                take1 = price - TP1_R * R
                take2 = price - TP2_R * R
                trades.append({
                    "product": product, "open_time": tstamp, "side": "short",
                    "entry": price, "sl": stop, "tp1": take1, "tp2": take2,
                    "R": R, "status": "open", "pnl_R": 0.0, "partial1": False, "zone_id": zone_id
                })
                in_position = True; side = "short"; entry, sl, tp1, tp2 = price, stop, take1, take2
                be_moved = False; trail_active = False; trail_stop = None
                zone_last_trade_day[zone_id] = (tstamp.tz_convert(None).date() if tstamp.tzinfo else tstamp.date())
                continue

        # manage open position on subsequent bars
        if in_position and trades:
            hi = h.loc[i, "high"]; lo = h.loc[i, "low"]; cur_time = tstamp
            # SL check
            if side == "long" and lo <= sl:
                trades[-1].update({"status": "closed", "close_time": cur_time, "pnl_R": -1.0})
                in_position = False; side = None; continue
            if side == "short" and hi >= sl:
                trades[-1].update({"status": "closed", "close_time": cur_time, "pnl_R": -1.0})
                in_position = False; side = None; continue
            # TP1
            if side == "long" and hi >= tp1 and not trades[-1]["partial1"]:
                trades[-1]["partial1"] = True
                trades[-1]["pnl_R"] += TP1_R * TP1_PART
                sl = entry; be_moved = True
                if TRAIL_AFTER_TP1: trail_active = True
            if side == "short" and lo <= tp1 and not trades[-1]["partial1"]:
                trades[-1]["partial1"] = True
                trades[-1]["pnl_R"] += TP1_R * TP1_PART
                sl = entry; be_moved = True
                if TRAIL_AFTER_TP1: trail_active = True
            # trail
            if trail_active:
                a = h.loc[i, "atr1h"]
                if side == "long":
                    trail_stop = max((trail_stop or -1e9), (h.loc[i, "close"] - TRAIL_ATR_MULT * a))
                    sl = max(sl, trail_stop)
                else:
                    trail_stop = min((trail_stop or 1e9), (h.loc[i, "close"] + TRAIL_ATR_MULT * a))
                    sl = min(sl, trail_stop)
            # TP2
            if side == "long" and hi >= tp2:
                trades[-1]["pnl_R"] += TP2_R * (1.0 - TP1_PART)
                trades[-1].update({"status": "closed", "close_time": cur_time})
                in_position = False; side = None; continue
            if side == "short" and lo <= tp2:
                trades[-1]["pnl_R"] += TP2_R * (1.0 - TP1_PART)
                trades[-1].update({"status": "closed", "close_time": cur_time})
                in_position = False; side = None; continue
            # BE stop after TP1
            if be_moved:
                if side == "long" and lo <= sl:
                    trades[-1].update({"status": "closed", "close_time": cur_time})
                    in_position = False; side = None; continue
                if side == "short" and hi >= sl:
                    trades[-1].update({"status": "closed", "close_time": cur_time})
                    in_position = False; side = None; continue

    return pd.DataFrame(trades)

# ------------------------------
# Metrics
# ------------------------------
def profit_factor(pnl: pd.Series) -> float:
    wins = pnl[pnl > 0].sum()
    losses = pnl[pnl < 0].sum()
    return wins / abs(losses) if losses < 0 else float("nan")

def max_drawdown_R(pnl: pd.Series) -> float:
    eq = pnl.cumsum()
    peak = eq.cummax()
    dd = eq - peak
    return float(dd.min())

def summarize_results(all_trades: pd.DataFrame) -> pd.DataFrame:
    if all_trades.empty:
        return pd.DataFrame(columns=["product","trades","win_rate","profit_factor","total_R","avg_R","max_dd_R"])
    closed = all_trades[all_trades["status"] == "closed"].copy()
    if closed.empty:
        return pd.DataFrame(columns=["product","trades","win_rate","profit_factor","total_R","avg_R","max_dd_R"])
    closed["win"] = closed["pnl_R"] > 0
    summary = closed.groupby("product").apply(
        lambda g: pd.Series({
            "trades": len(g),
            "win_rate": g["win"].mean(),
            "profit_factor": profit_factor(g["pnl_R"]),
            "total_R": g["pnl_R"].sum(),
            "avg_R": g["pnl_R"].mean(),
            "max_dd_R": max_drawdown_R(g["pnl_R"]),
        })
    ).reset_index()

    overall = pd.Series({
        "product": "ALL",
        "trades": len(closed),
        "win_rate": closed["win"].mean(),
        "profit_factor": profit_factor(closed["pnl_R"]),
        "total_R": closed["pnl_R"].sum(),
        "avg_R": closed["pnl_R"].mean(),
        "max_dd_R": max_drawdown_R(closed["pnl_R"]),
    })
    return pd.concat([summary, overall.to_frame().T], ignore_index=True)

# ------------------------------
# Runner
# ------------------------------
def run():
    all_trades = []
    for product in COINS:
        print(f"[{product}] fetching 1D & 1H ...")
        try:
            df_1d = coinbase_fetch_days(product, TF_ZONES, DAYS_BACK)
            df_1h = coinbase_fetch_days(product, TF_TRADE, DAYS_BACK)
        except Exception as e:
            print(f"Fetch failed for {product}: {e}")
            continue
        if df_1d.empty or df_1h.empty:
            print(f"No data for {product}")
            continue

        # normalize columns
        df_1d = df_1d[["time","open","high","low","close","volume"]].copy()
        df_1h = df_1h[["time","open","high","low","close","volume"]].copy()

        print(f"[{product}] building zones & backtesting on {len(df_1h)} x 1H bars ...")
        tdf = backtest_coin(product, df_1h, df_1d)
        if not tdf.empty:
            all_trades.append(tdf)

    if not all_trades:
        print("No trades generated.")
        return

    trades = pd.concat(all_trades, ignore_index=True)
    summary = summarize_results(trades)
    trades.to_csv("bb_sr_v2_trades.csv", index=False)
    summary.to_csv("bb_sr_v2_summary.csv", index=False)
    print("Saved: bb_sr_v2_trades.csv, bb_sr_v2_summary.csv")
    print(summary)

if __name__ == "__main__":
    run()
