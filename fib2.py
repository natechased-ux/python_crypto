#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Live Fib + EMA Reversal Alert Bot (Coinbase + Telegram)

Zones:
  FIB_TINY   (2d/1H, confluence required)
  FIB_SHORT  (7d/1H)
  FIB_MEDIUM (14d/6H)
  FIB_LONG   (30d/1D)
  EMA50_1H   (confluence required)
  EMA200_1H
  EMA20_1D
  EMA50_1D
  EMA100_1D
  EMA200_1D

Rules:
  - Reversal entries:
      * Ascend into zone => SHORT
      * Descend into zone => LONG
  - Stoch RSI REQUIRED (strict):
      LONG  = K > D and K < 40
      SHORT = K < D and K > 60
  - TINY + EMA50_1H require ≥ 2 zones active
  - Alerts include star ranking (★☆☆ → ★★★)
"""

import time, requests, pandas as pd
from datetime import datetime, timedelta, timezone

# ======= CONFIG =======
BOT_TOKEN = "8177096945:AAHTg5nxVTA6hcPkAL4MLAAHPsZ1at7Ywmw"
CHAT_ID   = "-4916911067"  # channel or user id

INTERVAL_SECONDS = 300
COOLDOWN_MINUTES = 30

# Fib lookbacks
TINY_LOOKBACK_DAYS   = 2   # 2d / 1H
SHORT_LOOKBACK_DAYS  = 7   # 7d / 1H
MEDIUM_LOOKBACK_DAYS = 14  # 14d / 6H
LONG_LOOKBACK_DAYS   = 30  # 30d / 1D

# Fib pocket
GOLDEN_MIN = 0.618
GOLDEN_MAX = 0.66
GOLDEN_TOL = 0.0025

# EMA band width (±)
EMA_BAND_PCT = 0.005

# Risk
SL_BUFFER = 0.01
TP1_MULT  = 1.0
TP2_MULT  = 1.5

COINS = [
    "BTC-USD","ETH-USD","XRP-USD","SOL-USD","ADA-USD","AVAX-USD","DOGE-USD","DOT-USD",
    "LINK-USD","ATOM-USD","NEAR-USD","ARB-USD","OP-USD","MATIC-USD","SUI-USD",
    "INJ-USD","AAVE-USD","LTC-USD","BCH-USD","ETC-USD","ALGO-USD","FIL-USD","ICP-USD",
    "RNDR-USD","STX-USD","JTO-USD","PYTH-USD","GRT-USD","SEI-USD",
    "ENS-USD","FLOW-USD","KSM-USD","KAVA-USD",
    "WLD-USD","HBAR-USD","JUP-USD","STRK-USD",
    "ONDO-USD","SUPER-USD","LDO-USD","POL-USD",
    "ZETA-USD","ZRO-USD","TIA-USD",
    "WIF-USD","MAGIC-USD","APE-USD","JASMY-USD","SYRUP-USD","FARTCOIN-USD",
    "AERO-USD","FET-USD","CRV-USD","TAO-USD","XCN-USD","UNI-USD","MKR-USD",
    "TOSHI-USD","TRUMP-USD","PEPE-USD","XLM-USD","MOODENG-USD","BONK-USD",
    "POPCAT-USD","QNT-USD","IP-USD","PNUT-USD","APT-USD","ENA-USD","TURBO-USD",
    "BERA-USD","MASK-USD","SAND-USD","MORPHO-USD","MANA-USD","C98-USD","AXS-USD"
]
EXCLUDE = {"VELO-USD","SYRUP-USD","AERO-USD"}
COINS = [c for c in COINS if c.upper() not in EXCLUDE]

CB_BASE = "https://api.exchange.coinbase.com"
# ======= END CONFIG =======

# --- Helpers ---
def send_telegram(msg):
    try:
        requests.post(
            f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage",
            json={"chat_id": CHAT_ID, "text": msg, "parse_mode": "Markdown"},
            timeout=12
        )
    except Exception:
        pass

def fmt_price(px: float) -> str:
    if px >= 100: return f"{px:.2f}"
    if px >= 10:  return f"{px:.3f}"
    if px >= 1:   return f"{px:.4f}"
    if px >= 0.1: return f"{px:.5f}"
    return f"{px:.6f}"

def fetch_candles(product_id: str, granularity: int) -> pd.DataFrame:
    url = f"{CB_BASE}/products/{product_id}/candles"
    r = requests.get(url, params={"granularity": granularity}, timeout=15)
    r.raise_for_status()
    data = r.json()
    df = pd.DataFrame(data, columns=["time","low","high","open","close","volume"])
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df.sort_values("time", inplace=True)
    return df.reset_index(drop=True)

def rsi(series: pd.Series, period=14):
    d = series.diff()
    up, down = d.clip(lower=0), -d.clip(upper=0)
    ru = up.ewm(alpha=1/period, adjust=False).mean()
    rd = down.ewm(alpha=1/period, adjust=False).mean()
    rs = ru / (rd.replace(0,1e-10))
    return 100 - (100 / (1 + rs))

def stoch_rsi(close: pd.Series, period=14, k=3, d=3):
    r = rsi(close, period)
    lo, hi = r.rolling(period).min(), r.rolling(period).max()
    st = (r - lo) / (hi - lo + 1e-10) * 100
    K = st.rolling(k).mean()
    D = K.rolling(d).mean()
    return K, D

def ema(series: pd.Series, period: int):
    return series.ewm(span=period, adjust=False).mean()

def fib_golden_zone(lo, hi, tol=GOLDEN_TOL):
    span = hi - lo
    f618, f66 = lo + span*GOLDEN_MIN, lo + span*GOLDEN_MAX
    return f618*(1-tol), f66*(1+tol)

def price_in_zone(px, zmin, zmax):
    return (zmin is not None) and (zmax is not None) and (zmin <= px <= zmax)

def detect_reversal(prev, curr, zmin, zmax):
    if (zmin is None) or (zmax is None): return None
    if prev < zmin and zmin <= curr <= zmax: return "SHORT"
    if prev > zmax and zmin <= curr <= zmax: return "LONG"
    return None

def starrating(zone: str, conf_count: int) -> str:
    # Core zones are always ★★★ even alone
    CORE = {"EMA100_1D", "EMA20_1D", "FIB_MEDIUM", "FIB_LONG"}
    if zone in CORE or conf_count >= 3:
        return "★★★"
    # Secondary zones get ★★☆; but the two weak zones need confluence to reach ★★☆
    if zone in {"FIB_TINY","EMA50_1H"}:
        return "★★☆" if conf_count >= 2 else "★☆☆"
    # Other secondary zones default to ★★☆
    return "★★☆"

# --- Core logic ---
def check_signal(product_id, cooldowns):
    df1h = fetch_candles(product_id, 3600)
    df6h = fetch_candles(product_id, 21600)
    df1d = fetch_candles(product_id, 86400)
    if len(df1h) < SHORT_LOOKBACK_DAYS*24 + 2: return None

    curr, prev = float(df1h["close"].iloc[-1]), float(df1h["close"].iloc[-2])

    entries, zone_flags = [], {}

    # FIB_TINY / FIB_SHORT (1H windows)
    for name, look in [("FIB_TINY", TINY_LOOKBACK_DAYS*24), ("FIB_SHORT", SHORT_LOOKBACK_DAYS*24)]:
        sl = df1h["low"].tail(look).min()
        sh = df1h["high"].tail(look).max()
        zmin, zmax = fib_golden_zone(sl, sh)
        side = detect_reversal(prev, curr, zmin, zmax)
        entries.append((name, sl, sh, zmin, zmax, side))
        zone_flags[name] = price_in_zone(curr, zmin, zmax)

    # FIB_MEDIUM (6H window)
    df6h_hist = df6h[df6h["time"] <= df1h["time"].iloc[-1]]
    if len(df6h_hist) >= MEDIUM_LOOKBACK_DAYS*4:
        sl = df6h_hist["low"].tail(MEDIUM_LOOKBACK_DAYS*4).min()
        sh = df6h_hist["high"].tail(MEDIUM_LOOKBACK_DAYS*4).max()
        zmin, zmax = fib_golden_zone(sl, sh)
        side = detect_reversal(prev, curr, zmin, zmax)
        entries.append(("FIB_MEDIUM", sl, sh, zmin, zmax, side))
        zone_flags["FIB_MEDIUM"] = price_in_zone(curr, zmin, zmax)

    # FIB_LONG (1D window)
    df1d_hist = df1d[df1d["time"] <= df1h["time"].iloc[-1]]
    if len(df1d_hist) >= LONG_LOOKBACK_DAYS:
        sl = df1d_hist["low"].tail(LONG_LOOKBACK_DAYS).min()
        sh = df1d_hist["high"].tail(LONG_LOOKBACK_DAYS).max()
        zmin, zmax = fib_golden_zone(sl, sh)
        side = detect_reversal(prev, curr, zmin, zmax)
        entries.append(("FIB_LONG", sl, sh, zmin, zmax, side))
        zone_flags["FIB_LONG"] = price_in_zone(curr, zmin, zmax)

    # EMA50_1H / EMA200_1H
    for ema_p, name in [(50, "EMA50_1H"), (200, "EMA200_1H")]:
        if len(df1h) >= ema_p:
            val = float(ema(df1h["close"], ema_p).iloc[-1])
            zmin, zmax = val*(1-EMA_BAND_PCT), val*(1+EMA_BAND_PCT)
            side = detect_reversal(prev, curr, zmin, zmax)
            entries.append((name, val, val, zmin, zmax, side))
            zone_flags[name] = price_in_zone(curr, zmin, zmax)

    # EMA20/50/100/200 on 1D
    for ema_p, name in [(20,"EMA20_1D"), (50,"EMA50_1D"), (100,"EMA100_1D"), (200,"EMA200_1D")]:
        if len(df1d_hist) >= ema_p:
            val = float(ema(df1d_hist["close"], ema_p).iloc[-1])
            zmin, zmax = val*(1-EMA_BAND_PCT), val*(1+EMA_BAND_PCT)
            side = detect_reversal(prev, curr, zmin, zmax)
            entries.append((name, val, val, zmin, zmax, side))
            zone_flags[name] = price_in_zone(curr, zmin, zmax)

    # Filter for directional hits
    entries = [e for e in entries if e[5] is not None]
    if not entries: return None

    # Priority (higher-TF zones first; you can tweak)
    priority = {
        "EMA200_1D":10,"EMA100_1D":9,"EMA50_1D":8,"EMA20_1D":7,
        "FIB_LONG":6,"FIB_MEDIUM":5,
        "EMA200_1H":4,"EMA50_1H":3,
        "FIB_SHORT":2,"FIB_TINY":1
    }
    entries.sort(key=lambda x: priority[x[0]], reverse=True)
    zone_source, sl, sh, zmin, zmax, side = entries[0]

    # Confluence
    zone_count = sum(int(v) for v in zone_flags.values())
    zones_list = "/".join([z for z, v in zone_flags.items() if v]) or "None"

    # Stoch RSI REQUIRED (strict)
    K, D = stoch_rsi(df1h["close"])
    kv, dv = float(K.iloc[-1]), float(D.iloc[-1])
    stoch_ok = (kv > dv and kv < 40.0) if side == "LONG" else (kv < dv and kv > 60.0)
    if not stoch_ok:
        return None

    # TINY + EMA50_1H need ≥ 2 zones active
    if zone_source in {"FIB_TINY","EMA50_1H"} and zone_count < 2:
        return None

    # Cooldown
    now = datetime.now(timezone.utc)
    if (product_id in cooldowns) and (now - cooldowns[product_id] < timedelta(minutes=COOLDOWN_MINUTES)):
        return None

    # Risk / targets
    entry = curr
    if side == "LONG":
        slp = zmin*(1-SL_BUFFER); risk = entry - slp
        tp1, tp2 = entry + TP1_MULT*risk, entry + TP2_MULT*risk
    else:
        slp = zmax*(1+SL_BUFFER); risk = slp - entry
        tp1, tp2 = entry - TP1_MULT*risk, entry - TP2_MULT*risk
    if risk <= 0: return None

    stars = starrating(zone_source, zone_count)
    msg = (
        f"{'🔻' if side=='SHORT' else '🚀'} *{side}* on *{product_id}* — Strength: *{stars}*\n"
        f"Confluence: {zone_count} zone(s) — {zones_list}\n"
        f"Active zone: {zone_source} — zmin {fmt_price(zmin)} / zmax {fmt_price(zmax)}\n"
        f"Entry: ${fmt_price(entry)} | SL: ${fmt_price(slp)} | TP1: ${fmt_price(tp1)} | TP2: ${fmt_price(tp2)}\n"
        f"StochRSI (strict): K={kv:.1f}, D={dv:.1f} — ✅\n"
        f"Time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}"
    )

    cooldowns[product_id] = now
    return msg

# --- Loop ---
def main_loop():
    cooldowns = {}
    print(f"Monitoring {len(COINS)} coins every {INTERVAL_SECONDS}s | Stoch REQUIRED | Tiny/EMA50_1H need confluence")
    while True:
        start = time.time()
        for coin in COINS:
            try:
                alert = check_signal(coin, cooldowns)
                if alert:
                    send_telegram(alert)
            except Exception as e:
                print(f"[ERR] {coin}: {e}")
            time.sleep(0.15)
        time.sleep(max(0.0, INTERVAL_SECONDS - (time.time() - start)))

if __name__ == "__main__":
    main_loop()
