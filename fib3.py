#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Live Fib + EMA Reversal Alert Bot (Coinbase + Telegram) with per-coin playbooks.

Implements:
- EMA zones => SCALP (percent TP/SL/timebox per coin)
- FIB zones (SHORT/MED/LONG) => REVERSAL (TP1% partial + TP2% runner, timebox per coin)
- EXCLUDE => suppress or mark 'weak'

Stoch RSI strict (40/60) remains required.
"""

import time, requests, pandas as pd
from datetime import datetime, timedelta, timezone
from coin_rules import set_all_coins, load_rules, get_styles_for


# ======= CONFIG (same as your current) =======
BOT_TOKEN = "8177096945:AAHTg5nxVTA6hcPkAL4MLAAHPsZ1at7Ywmw"
CHAT_ID   = "-4916911067"
INTERVAL_SECONDS = 300
COOLDOWN_MINUTES = 30

TINY_LOOKBACK_DAYS   = 2
SHORT_LOOKBACK_DAYS  = 7
MEDIUM_LOOKBACK_DAYS = 14
LONG_LOOKBACK_DAYS   = 30

GOLDEN_MIN = 0.618
GOLDEN_MAX = 0.66
GOLDEN_TOL = 0.0025

EMA_BAND_PCT = 0.005

SL_BUFFER = 0.01  # still used if a coin's REVERSAL SL_mode='zone'

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
EXCLUDE = {"VELO-USD","SYRUP-USD","AERO-USD","FLOW-USD","QNT-USD"}
COINS = [c for c in COINS if c.upper() not in EXCLUDE]

CB_BASE = "https://api.exchange.coinbase.com"
# ======= END CONFIG =======

# --- Helpers (same as before) ---
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
    CORE = {"EMA100_1D", "EMA20_1D", "FIB_MEDIUM", "FIB_LONG"}
    if zone in CORE or conf_count >= 3: return "★★★"
    if zone in {"FIB_TINY","EMA50_1H"}: return "★★☆" if conf_count >= 2 else "★☆☆"
    return "★★☆"

# --- Core logic (updated to apply per-coin rules) ---
def check_signal(product_id: str, cooldowns: dict):
    # ------------- fetch data -------------
    df1h = fetch_candles(product_id, 3600)
    df6h = fetch_candles(product_id, 21600)
    df1d = fetch_candles(product_id, 86400)
    if len(df1h) < SHORT_LOOKBACK_DAYS*24 + 2:
        return None

    curr, prev = float(df1h["close"].iloc[-1]), float(df1h["close"].iloc[-2])
    now_bar_time = df1h["time"].iloc[-1]

    # ------------- build zones -------------
    entries = []
    zone_flags = {}

    # FIB_TINY (2d / 1H)
    sl = df1h["low"].tail(TINY_LOOKBACK_DAYS*24).min()
    sh = df1h["high"].tail(TINY_LOOKBACK_DAYS*24).max()
    zmin, zmax = fib_golden_zone(sl, sh)
    side = detect_reversal(prev, curr, zmin, zmax)
    entries.append(("FIB_TINY", sl, sh, zmin, zmax, side))
    zone_flags["FIB_TINY"] = price_in_zone(curr, zmin, zmax)

    # FIB_SHORT (7d / 1H)
    sl = df1h["low"].tail(SHORT_LOOKBACK_DAYS*24).min()
    sh = df1h["high"].tail(SHORT_LOOKBACK_DAYS*24).max()
    zmin, zmax = fib_golden_zone(sl, sh)
    side = detect_reversal(prev, curr, zmin, zmax)
    entries.append(("FIB_SHORT", sl, sh, zmin, zmax, side))
    zone_flags["FIB_SHORT"] = price_in_zone(curr, zmin, zmax)

    # FIB_MEDIUM (14d / 6H)
    df6h_hist = df6h[df6h["time"] <= now_bar_time]
    if len(df6h_hist) >= MEDIUM_LOOKBACK_DAYS*4:
        sl = df6h_hist["low"].tail(MEDIUM_LOOKBACK_DAYS*4).min()
        sh = df6h_hist["high"].tail(MEDIUM_LOOKBACK_DAYS*4).max()
        zmin, zmax = fib_golden_zone(sl, sh)
        side = detect_reversal(prev, curr, zmin, zmax)
        entries.append(("FIB_MEDIUM", sl, sh, zmin, zmax, side))
        zone_flags["FIB_MEDIUM"] = price_in_zone(curr, zmin, zmax)

    # FIB_LONG (30d / 1D)
    df1d_hist = df1d[df1d["time"] <= now_bar_time]
    if len(df1d_hist) >= LONG_LOOKBACK_DAYS:
        sl = df1d_hist["low"].tail(LONG_LOOKBACK_DAYS).min()
        sh = df1d_hist["high"].tail(LONG_LOOKBACK_DAYS).max()
        zmin, zmax = fib_golden_zone(sl, sh)
        side = detect_reversal(prev, curr, zmin, zmax)
        entries.append(("FIB_LONG", sl, sh, zmin, zmax, side))
        zone_flags["FIB_LONG"] = price_in_zone(curr, zmin, zmax)

    # EMA 1H (50/200)
    for per, name in [(50, "EMA50_1H"), (200, "EMA200_1H")]:
        if len(df1h) >= per:
            val = float(ema(df1h["close"], per).iloc[-1])
            zmin, zmax = val*(1-EMA_BAND_PCT), val*(1+EMA_BAND_PCT)
            side = detect_reversal(prev, curr, zmin, zmax)
            entries.append((name, val, val, zmin, zmax, side))
            zone_flags[name] = price_in_zone(curr, zmin, zmax)

    # EMA 1D (20/50/100/200)
    for per, name in [(20,"EMA20_1D"),(50,"EMA50_1D"),(100,"EMA100_1D"),(200,"EMA200_1D")]:
        if len(df1d_hist) >= per:
            val = float(ema(df1d_hist["close"], per).iloc[-1])
            zmin, zmax = val*(1-EMA_BAND_PCT), val*(1+EMA_BAND_PCT)
            side = detect_reversal(prev, curr, zmin, zmax)
            entries.append((name, val, val, zmin, zmax, side))
            zone_flags[name] = price_in_zone(curr, zmin, zmax)

    # keep only directional candidates
    entries = [e for e in entries if e[5] is not None]
    if not entries:
        return None

    # ------------- choose active zone (priority) -------------
    priority = {
        "EMA200_1D":10,"EMA100_1D":9,"EMA50_1D":8,"EMA20_1D":7,
        "FIB_LONG":6,"FIB_MEDIUM":5,
        "EMA200_1H":4,"EMA50_1H":3,
        "FIB_SHORT":2,"FIB_TINY":1
    }
    entries.sort(key=lambda x: priority.get(x[0],1), reverse=True)
    zone_source, swing_low, swing_high, zmin, zmax, side = entries[0]

    # confluence snapshot
    zone_count = sum(int(v) for v in zone_flags.values())
    zones_list = "/".join([z for z, v in zone_flags.items() if v]) or "None"

    # ------------- stoch RSI strict (required) -------------
    K, D = stoch_rsi(df1h["close"])
    kv, dv = float(K.iloc[-1]), float(D.iloc[-1])
    stoch_ok = (kv > dv and kv < 40.0) if side == "LONG" else (kv < dv and kv > 60.0)
    if not stoch_ok:
        return None

    # tiny & EMA50_1H need confluence ≥2
    if zone_source in {"FIB_TINY","EMA50_1H"} and zone_count < 2:
        return None

    # ------------- cooldown -------------
    now = datetime.now(timezone.utc)
    if (product_id in cooldowns) and (now - cooldowns[product_id] < timedelta(minutes=COOLDOWN_MINUTES)):
        return None

    # ------------- per-zone, per-coin style + TP/SL -------------
    style, params = get_styles_for(product_id, zone_source)
    if params.get("suppress"):
        return None

    entry = curr
    # SCALP on EMA zones
    if zone_source.startswith("EMA") and style == "SCALP":
        tp_pct = params["tp_pct"]
        sl_pct = params["sl_pct"]
        timebox_h = params["timebox_h"]

        if side == "LONG":
            tp1_price = entry * (1 + tp_pct/100)
            sl_price  = entry * (1 - sl_pct/100)
        else:
            tp1_price = entry * (1 - tp_pct/100)
            sl_price  = entry * (1 + sl_pct/100)

        msg_plan = f"SCALP • TP {tp_pct:.2f}% | SL {sl_pct:.2f}% | timebox {timebox_h}h"

    # REVERSAL on FIB zones
    elif zone_source.startswith("FIB") and style == "REVERSAL":
        tp1_pct = params["tp1_pct"]
        tp1_size = params["tp1_size_pct"]
        tp2_pct = params["tp2_pct"]
        timebox_h = params["timebox_h"]
        sl_mode = params.get("sl_mode","zone")

        if side == "LONG":
            tp1_price = entry * (1 + tp1_pct/100)
            tp2_price = entry * (1 + tp2_pct/100)
            if sl_mode == "atr":
                sl_price = entry * (1 - params.get("atr_sl_pct",0.70)/100)
            else:
                sl_price = zmin*(1 - SL_BUFFER)
        else:
            tp1_price = entry * (1 - tp1_pct/100)
            tp2_price = entry * (1 - tp2_pct/100)
            if sl_mode == "atr":
                sl_price = entry * (1 + params.get("atr_sl_pct",0.70)/100)
            else:
                sl_price = zmax*(1 + SL_BUFFER)

        msg_plan = (
            f"REVERSAL • TP1 {tp1_pct:.2f}% ({int(tp1_size*100)}%) "
            f"| TP2 {tp2_pct:.2f}% | SL {sl_mode.upper()} | timebox {timebox_h}h"
        )

    # Fallback: treat anything else as a SCALP
    else:
        tp_pct = params.get("tp_pct", 0.55)
        sl_pct = params.get("sl_pct", 0.30)
        timebox_h = params.get("timebox_h", 3)
        if side == "LONG":
            tp1_price = entry * (1 + tp_pct/100)
            sl_price  = entry * (1 - sl_pct/100)
        else:
            tp1_price = entry * (1 - tp_pct/100)
            sl_price  = entry * (1 + sl_pct/100)
        msg_plan = f"ADAPTIVE • TP {tp_pct:.2f}% | SL {sl_pct:.2f}% | timebox {timebox_h}h"

    # ------------- star rating & message -------------
    stars = starrating(zone_source, zone_count)
    msg = (
        f"{'🔻' if side=='SHORT' else '🚀'} *{side}* on *{product_id}* — Strength: *{stars}*\n"
        f"Zones active: {zone_count} — {zones_list}\n"
        f"Active zone: {zone_source} — zmin {fmt_price(zmin)} / zmax {fmt_price(zmax)}\n"
        f"Entry: ${fmt_price(entry)} | SL: ${fmt_price(sl_price)} | TP1: ${fmt_price(tp1_price)}"
    )

    if zone_source.startswith("FIB") and style == "REVERSAL":
        msg += f" | TP2: ${fmt_price(tp2_price)}"

    msg += (
        f"\n{msg_plan}"
        f"\nStochRSI (strict): K={kv:.1f}, D={dv:.1f} — ✅"
        f"\nTime: {now.strftime('%Y-%m-%d %H:%M UTC')}"
    )

    cooldowns[product_id] = now
    return msg


# --- Loop ---
def main_loop():
    set_all_coins(COINS)            # seed heuristics (future use)
    load_rules()
    cooldowns = {}
    print("Per-coin per-zone rules active • EMA=SCALP • FIB=REVERSAL • ATR-stop on selected FIB coins")
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
