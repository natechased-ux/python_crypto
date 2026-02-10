
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Live Zone-to-Zone Winners-Only Alert Bot

- Trades only "winner" zone->zone paths (based on Avg R from your backtest).
- Treats zones as support/resistance and aims for the next zone in the trend direction.
- Telegram alerts (non-async) with entry, SL, TP, zone path, and context.

Requirements:
  pip install requests pandas numpy python-dateutil

Run:
  python live_zone2zone_winners.py

Notes:
- Polls every 5 minutes and uses 1H candles with a "hold" rule (2 consecutive closes within the zone).
- Daily bias via EMA200 (1D), trend via 1H EMA50>EMA200 and ADX>=18, momentum via Stoch RSI strict.
- Winner whitelist can be static (below) or dynamically loaded from a summary CSV of recent performance.
"""

import os
import time
import math
import json
import queue
import threading
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta

# ===================== CONFIG =====================
COINS = [
    "BTC-USD","ETH-USD","XRP-USD","SOL-USD","ADA-USD","AVAX-USD","DOGE-USD","DOT-USD",
    "LINK-USD","ATOM-USD","NEAR-USD","ARB-USD","OP-USD","MATIC-USD","SUI-USD",
    "INJ-USD","AAVE-USD","LTC-USD","BCH-USD","ETC-USD","ALGO-USD","FIL-USD","ICP-USD",
    "RNDR-USD","STX-USD","JTO-USD","PYTH-USD","GRT-USD","SEI-USD",
    "ENS-USD","FLOW-USD","KSM-USD","KAVA-USD",
    "WLD-USD","HBAR-USD","JUP-USD","STRK-USD",
    "ONDO-USD","SUPER-USD","LDO-USD","POL-USD",
    "ZETA-USD","ZRO-USD","TIA-USD",
    # Trimmed tail coins that often underperform can be dropped here if desired
]

# Polling & cooldowns
POLL_SECONDS = 300   # 5 minutes
COOLDOWN_MINUTES = 30
MAX_CONCURRENT_THREADS = 6

# Zones & indicators
EMA_BAND_PCT = 0.005      # 0.5%
SWEEP_LOOKBACK_HOURS = 48
SWEEP_BAND_PCT = 0.0015
VWAP_BAND_PCT = 0.003
GOLDEN_MIN, GOLDEN_MAX, GOLDEN_TOL = 0.618, 0.66, 0.0025

# Trend & momentum filters
TREND_ADX_MIN = 18.0
STOCH_RSI_PERIOD = 14
STOCH_SMOOTH_K = 3
STOCH_SMOOTH_D = 3
LONG_K_MAX  = 40.0  # LONG: K>D & K<40
SHORT_K_MIN = 60.0  # SHORT: K<D & K>60

# Exits
SL_BUFFER = 0.01     # SL beyond current zone boundary
ATR_MULT_TP1 = 1.0   # fallback if no next zone
MAX_HOLD_HOURS_SIM = 48  # only used for sanity checks before alert (not for live mgmt)

# Telegram (non-async)
TELEGRAM_BOT_TOKEN = "8177096945:AAHTg5nxVTA6hcPkAL4MLAAHPsZ1at7Ywmw"
TELEGRAM_CHAT_ID   = "7967738614"

# Winners-only control
USE_WINNER_WHITELIST = True
ALLOWED_ZONE_PATHS = {
    "FIB_TINY->LIQ_SWEEP_LOW_1H",
    "FIB_MEDIUM->FIB_LONG",
    "FIB_MEDIUM->EMA20_1D",
    "EMA50_1D->FIB_SHORT",
    "EMA200_1D->EMA200_1H",
    "FIB_SHORT->FIB_MEDIUM",
    "FIB_MEDIUM->LIQ_SWEEP_HIGH_1H",
    "EMA100_1D->EMA50_1D",
    "EMA50_1H->FIB_MEDIUM",
    "EMA20_1D->FIB_SHORT",
    "FIB_TINY->EMA100_1D",
    "VWAP_SESSION_1H->FIB_LONG",
    "VWAP_SESSION_1H->LIQ_SWEEP_HIGH_1H",
    "FIB_LONG->EMA50_1D",
    "VWAP_SESSION_1H->EMA50_1D",
    "EMA50_1D->EMA200_1H",
    "FIB_SHORT->EMA200_1H",
    "EMA50_1H->EMA20_1D",
}

# Optional: dynamic whitelist updates from a rolling summary CSV
DYNAMIC_SUMMARY_CSV = None  # e.g., "summary_zone2zone_recent.csv"
DYN_MIN_TRADES = 10
DYN_MIN_AVG_R  = 0.20

# I/O
CB_BASE = "https://api.exchange.coinbase.com"
MAX_BARS_PER_CALL = 300
LOG_FILE = "live_zone2zone_alerts.csv"

# =============== TA / HELPERS ===============
def rsi(series: pd.Series, period=14):
    d = series.diff()
    up = d.clip(lower=0.0); down = -d.clip(upper=0.0)
    ru = up.ewm(alpha=1/period, adjust=False).mean()
    rd = down.ewm(alpha=1/period, adjust=False).mean()
    rs = ru / (rd.replace(0, 1e-10))
    return 100 - (100/(1+rs))

def stoch_rsi(close: pd.Series, period: int, k: int, d_: int):
    r = rsi(close, period)
    lo = r.rolling(period).min(); hi = r.rolling(period).max()
    st = (r - lo)/(hi - lo + 1e-10)*100.0
    K = st.rolling(k).mean(); D = K.rolling(d_).mean()
    return K, D

def ema(series: pd.Series, period: int):
    return series.ewm(span=period, adjust=False).mean()

def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14):
    prev_close = close.shift(1)
    tr = pd.concat([(high-low),(high-prev_close).abs(),(low-prev_close).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, adjust=False).mean()

def adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14):
    up = high.diff(); down = -low.diff()
    plus_dm  = np.where((up>down)&(up>0), up, 0.0)
    minus_dm = np.where((down>up)&(down>0), down, 0.0)
    atr_val  = atr(high, low, close, period=period)
    plus_di  = 100*pd.Series(plus_dm, index=high.index).ewm(alpha=1/period, adjust=False).mean()/atr_val.replace(0,np.nan)
    minus_di = 100*pd.Series(minus_dm,index=high.index).ewm(alpha=1/period,adjust=False).mean()/atr_val.replace(0,np.nan)
    dx = (plus_di - minus_di).abs()/ (plus_di + minus_di).replace(0,np.nan) * 100
    return dx.ewm(alpha=1/period, adjust=False).mean().fillna(0)

def fib_golden_zone(lo: float, hi: float, tol: float = GOLDEN_TOL):
    span = hi - lo; f618 = lo+span*GOLDEN_MIN; f66 = lo+span*GOLDEN_MAX
    return f618*(1-tol), f66*(1+tol)

def session_vwap_band(df_1h: pd.DataFrame, idx: int, band_pct: float):
    ts = df_1h["time"].iloc[idx]
    s = ts.normalize()
    d = df_1h[(df_1h["time"] >= s) & (df_1h["time"] <= ts)]
    if d.empty or d["volume"].sum() <= 0: return (None, None, None)
    pv = (d["close"]*d["volume"]).sum(); v = d["volume"].sum()
    vwap = float(pv/v)
    return (vwap*(1-band_pct), vwap*(1+band_pct), vwap)

def recent_swing_levels(df_1h: pd.DataFrame, hours: int):
    w = df_1h.tail(hours)
    return float(w["high"].max()), float(w["low"].min())

def collect_zones(df_1h, df_1d, i):
    t = df_1h["time"].iloc[i]
    curr = float(df_1h["close"].iloc[i])
    zones = []
    def add_zone(name, zmin, zmax, prio):
        if zmin is None or zmax is None: return
        mid = 0.5*(zmin+zmax)
        zones.append({"name":name, "zmin":zmin, "zmax":zmax, "mid":mid, "priority":prio})
    # FIB 1H tiny/short
    for name, look, pr in [("FIB_TINY", 2*24, 1), ("FIB_SHORT", 7*24, 2)]:
        w = df_1h.iloc[:i+1].tail(look)
        sl, sh = float(w["low"].min()), float(w["high"].max())
        zmin, zmax = fib_golden_zone(sl, sh)
        add_zone(name, zmin, zmax, pr)
    # FIB 1D medium/long
    df_1d_hist = df_1d[df_1d["time"] <= t]
    if len(df_1d_hist) >= 14:
        w = df_1d_hist.tail(14); sl, sh = float(w["low"].min()), float(w["high"].max())
        zmin, zmax = fib_golden_zone(sl, sh); add_zone("FIB_MEDIUM", zmin, zmax, 6)
    if len(df_1d_hist) >= 30:
        w = df_1d_hist.tail(30); sl, sh = float(w["low"].min()), float(w["high"].max())
        zmin, zmax = fib_golden_zone(sl, sh); add_zone("FIB_LONG", zmin, zmax, 7)
    # EMA 1H
    if i+1 >= 50:
        e50 = float(ema(df_1h["close"].iloc[:i+1], 50).iloc[-1])
        zmin, zmax = e50*(1-EMA_BAND_PCT), e50*(1+EMA_BAND_PCT)
        add_zone("EMA50_1H", zmin, zmax, 2)
    if i+1 >= 200:
        e200 = float(ema(df_1h["close"].iloc[:i+1], 200).iloc[-1])
        zmin, zmax = e200*(1-EMA_BAND_PCT), e200*(1+EMA_BAND_PCT)
        add_zone("EMA200_1H", zmin, zmax, 3)
    # EMA 1D
    def add_ema_1d(period, nm, prio):
        if len(df_1d_hist) >= period:
            val = float(ema(df_1d_hist["close"], period).iloc[-1])
            zmin, zmax = val*(1-EMA_BAND_PCT), val*(1+EMA_BAND_PCT)
            add_zone(nm, zmin, zmax, prio)
    add_ema_1d(20,  "EMA20_1D", 9)
    add_ema_1d(50,  "EMA50_1D", 8)
    add_ema_1d(100, "EMA100_1D", 10)
    add_ema_1d(200, "EMA200_1D", 11)
    # Sweep bands
    hi, lo = recent_swing_levels(df_1h.iloc[:i+1], SWEEP_LOOKBACK_HOURS)
    hi_min,hi_max = hi*(1-SWEEP_BAND_PCT), hi*(1+SWEEP_BAND_PCT)
    lo_min,lo_max = lo*(1-SWEEP_BAND_PCT), lo*(1+SWEEP_BAND_PCT)
    add_zone("LIQ_SWEEP_HIGH_1H", hi_min, hi_max, 5)
    add_zone("LIQ_SWEEP_LOW_1H",  lo_min, lo_max,  5)
    # Session VWAP band
    vmin, vmax, _ = session_vwap_band(df_1h, i, VWAP_BAND_PCT)
    if vmin is not None:
        add_zone("VWAP_SESSION_1H", vmin, vmax, 4)
    zones.sort(key=lambda z: z["mid"])
    return zones

# =============== DATA ===============
def fetch_candles_range(product_id: str, granularity: int, start_dt: pd.Timestamp, end_dt: pd.Timestamp, session=None):
    sess = session or requests.Session()
    url = f"{CB_BASE}/products/{product_id}/candles"
    out = []; step_secs = granularity*MAX_BARS_PER_CALL; curr_end = end_dt
    while curr_end > start_dt:
        curr_start = max(start_dt, curr_end - pd.Timedelta(seconds=step_secs))
        params = {"granularity": granularity, "start": curr_start.isoformat(), "end": curr_end.isoformat()}
        r = sess.get(url, params=params, timeout=20); r.raise_for_status()
        data = r.json()
        if not isinstance(data, list) or not data: break
        df_chunk = pd.DataFrame(data, columns=["time","low","high","open","close","volume"])
        df_chunk["time"] = pd.to_datetime(df_chunk["time"], unit="s", utc=True)
        out.append(df_chunk)
        earliest = df_chunk["time"].min()
        if pd.isna(earliest): break
        curr_end = earliest - pd.Timedelta(seconds=1)
        time.sleep(0.05)
    if not out: raise ValueError(f"No data for {product_id}@{granularity}")
    df = pd.concat(out, ignore_index=True).drop_duplicates(subset=["time"]).sort_values("time")
    df = df[(df["time"] >= start_dt) & (df["time"] < end_dt)].reset_index(drop=True)
    return df

def fetch_with_warmup(product_id: str, granularity: int, start_hours_ago: int = 24*30):
    e = pd.Timestamp.utcnow().tz_localize("UTC")
    s = e - pd.Timedelta(hours=start_hours_ago + 250)  # warm bars
    return fetch_candles_range(product_id, granularity, s, e)

def fetch_ticker_price(product_id: str) -> float:
    url = f"{CB_BASE}/products/{product_id}/ticker"
    r = requests.get(url, timeout=10)
    r.raise_for_status()
    data = r.json()
    return float(data["price"])

# =============== TELEGRAM ===============
def tg_send(text: str):
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        payload = {"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": "Markdown"}
        r = requests.post(url, json=payload, timeout=15)
        return r.ok
    except Exception as e:
        print(f"[ERR] Telegram: {e}")
        return False

# =============== DYNAMIC WHITELIST ===============
def load_dynamic_winners(path, min_trades=10, min_avg_R=0.20):
    try:
        df = pd.read_csv(path)
        df = df[df["section"]=="by_zone"].copy()
        winners = df[(df["trades"]>=min_trades) & (df["avg_R"]>=min_avg_R)]["zone_source"].dropna().unique().tolist()
        return set(winners)
    except Exception as e:
        print(f"[WARN] Dynamic whitelist load failed: {e}")
        return None

# =============== CORE LOGIC ===============
_last_alert_time = {}      # (coin, zone_source) -> timestamp
_coin_cooldown_until = {}  # coin -> timestamp

def price_decimals(px: float) -> int:
    if px >= 100: return 2
    if px >= 1:   return 4
    return 6

def process_coin(coin: str):
    # cooldown check
    now = datetime.now(timezone.utc)
    if coin in _coin_cooldown_until and now < _coin_cooldown_until[coin]:
        return

    # fetch data
    df_1h = fetch_with_warmup(coin, 3600, start_hours_ago=24*21)  # 3 weeks of 1H
    df_1d = fetch_with_warmup(coin, 86400, start_hours_ago=24*400) # ~400 days of 1D
    if len(df_1h) < 300 or len(df_1d) < 220:
        return

    # indicators
    K, D = stoch_rsi(df_1h["close"], STOCH_RSI_PERIOD, STOCH_SMOOTH_K, STOCH_SMOOTH_D)
    ema50_1h = ema(df_1h["close"], 50)
    ema200_1h = ema(df_1h["close"], 200)
    adx_1h = adx(df_1h["high"], df_1h["low"], df_1h["close"], 14)
    atr_1h_series = atr(df_1h["high"], df_1h["low"], df_1h["close"], 14)

    i = len(df_1h) - 1
    curr = float(df_1h["close"].iloc[i])
    kv, dv = float(K.iloc[i]), float(D.iloc[i])

    # zones (current)
    zones = collect_zones(df_1h, df_1d, i)
    if not zones: 
        return

    inside = [z for z in zones if z["zmin"] <= curr <= z["zmax"]]
    if not inside:
        return
    inside.sort(key=lambda z: z["priority"], reverse=True)
    cz = inside[0]

    # hold rule: last two closes in zone
    prev_close = float(df_1h["close"].iloc[i-1])
    if not (cz["zmin"] <= prev_close <= cz["zmax"] and cz["zmin"] <= curr <= cz["zmax"]):
        return

    # daily EMA200 bias at latest daily bar <= now
    df_1d_hist = df_1d[df_1d["time"] <= df_1h["time"].iloc[i]]
    if len(df_1d_hist) < 200:
        return
    ema200_1d_val = float(ema(df_1d_hist["close"], 200).iloc[-1])

    long_bias  = (curr > ema200_1d_val) and (ema50_1h.iloc[i] > ema200_1h.iloc[i]) and (adx_1h.iloc[i] >= TREND_ADX_MIN)
    short_bias = (curr < ema200_1d_val) and (ema50_1h.iloc[i] < ema200_1h.iloc[i]) and (adx_1h.iloc[i] >= TREND_ADX_MIN)

    stoch_long_ok  = (kv > dv and kv < LONG_K_MAX)
    stoch_short_ok = (kv < dv and kv > SHORT_K_MIN)

    side = None
    if long_bias and stoch_long_ok:   side = "LONG"
    if short_bias and stoch_short_ok: side = "SHORT"
    if side is None:
        return

    # choose next zone
    if side == "LONG":
        next_candidates = [z for z in zones if z["mid"] > cz["mid"]]
        if not next_candidates:
            zone_source = f'{cz["name"]}->ATR'
            tp1 = curr + ATR_MULT_TP1 * float(atr_1h_series.iloc[i])
            sl = cz["zmin"]*(1 - SL_BUFFER); risk = curr - sl
        else:
            nz = min(next_candidates, key=lambda z: z["mid"])
            zone_source = f'{cz["name"]}->{nz["name"]}'
            tp1 = nz["mid"]
            sl = cz["zmin"]*(1 - SL_BUFFER); risk = curr - sl
    else:
        next_candidates = [z for z in zones if z["mid"] < cz["mid"]]
        if not next_candidates:
            zone_source = f'{cz["name"]}->ATR'
            tp1 = curr - ATR_MULT_TP1 * float(atr_1h_series.iloc[i])
            sl = cz["zmax"]*(1 + SL_BUFFER); risk = sl - curr
        else:
            nz = max(next_candidates, key=lambda z: z["mid"])
            zone_source = f'{cz["name"]}->{nz["name"]}'
            tp1 = nz["mid"]
            sl = cz["zmax"]*(1 + SL_BUFFER); risk = sl - curr

    if risk <= 0:
        return

    # winners-only filter
    if USE_WINNER_WHITELIST:
        if "->" in zone_source:
            if zone_source not in ALLOWED_ZONE_PATHS:
                return
        else:
            return  # skip ->ATR when strict winners-only

    # coin cooldown & duplicate suppression (per zone path)
    key = (coin, zone_source, side)
    last = _last_alert_time.get(key)
    if last and (now - last) < timedelta(minutes=COOLDOWN_MINUTES):
        return

    # live price sanity (use ticker)
    try:
        live_px = fetch_ticker_price(coin)
    except Exception:
        live_px = curr

    # Format message
    dec = price_decimals(live_px)
    entry_txt = f"{live_px:.{dec}f}"
    sl_txt = f"{sl:.{dec}f}"
    tp_txt = f"{tp1:.{dec}f}"
    tstamp = df_1h["time"].iloc[i].strftime("%Y-%m-%d %H:%M UTC")

    msg = (
        f"*{coin}* — *{side}* (zone-to-zone winner)\n"
        f"*Path:* `{zone_source}`\n"
        f"*Entry:* {entry_txt}\n"
        f"*SL:* {sl_txt}  |  *TP:* {tp_txt}\n"
        f"*1D EMA200:* {ema200_1d_val:.{dec}f} | *ADX(1H):* {adx_1h.iloc[i]:.1f}\n"
        f"*StochRSI:* K={kv:.1f}, D={dv:.1f}\n"
        f"*Zone:* {cz['name']}  ({cz['zmin']:.{dec}f}–{cz['zmax']:.{dec}f})\n"
        f"*Signal time:* {tstamp}"
    )

    # Log and alert
    row = {
        "time": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
        "coin": coin, "side": side, "zone_source": zone_source,
        "entry": live_px, "sl": sl, "tp1": tp1,
        "ema200_1d": ema200_1d_val, "adx_1h": float(adx_1h.iloc[i]),
        "stoch_k": float(kv), "stoch_d": float(dv)
    }
    log_row(row)
    tg_send(msg)
    print(msg)

    # set cooldowns
    _last_alert_time[key] = now
    _coin_cooldown_until[coin] = now + timedelta(minutes=COOLDOWN_MINUTES)

def log_row(row: dict):
    write_header = not os.path.exists(LOG_FILE)
    df = pd.DataFrame([row])
    df.to_csv(LOG_FILE, mode="a", header=write_header, index=False)

def maybe_refresh_dynamic_whitelist():
    global ALLOWED_ZONE_PATHS
    if DYNAMIC_SUMMARY_CSV:
        dyn = load_dynamic_winners(DYNAMIC_SUMMARY_CSV, DYN_MIN_TRADES, DYN_MIN_AVG_R)
        if dyn:
            ALLOWED_ZONE_PATHS = dyn

def worker(q: "queue.Queue[str]"):
    while True:
        coin = q.get()
        if coin is None: break
        try:
            process_coin(coin)
        except Exception as e:
            print(f"[ERR] {coin}: {e}")
        finally:
            q.task_done()

def main_loop():
    maybe_refresh_dynamic_whitelist()

    q = queue.Queue()
    threads = []
    for _ in range(MAX_CONCURRENT_THREADS):
        t = threading.Thread(target=worker, args=(q,), daemon=True)
        t.start()
        threads.append(t)

    try:
        while True:
            start = time.time()
            for c in COINS:
                q.put(c)
            q.join()
            # periodic dynamic refresh (every hour)
            if int(time.time()) % 3600 < POLL_SECONDS:
                maybe_refresh_dynamic_whitelist()
            # sleep until next poll
            elapsed = time.time() - start
            sleep_s = max(5, POLL_SECONDS - elapsed)
            time.sleep(sleep_s)
    except KeyboardInterrupt:
        print("Shutting down...")
    finally:
        for _ in threads:
            q.put(None)
        for t in threads:
            t.join(timeout=1)

if __name__ == "__main__":
    main_loop()
