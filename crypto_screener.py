# wa_friendly_screener.py
# pip install ccxt pandas numpy ta requests

import os, json, math, time, datetime as dt
import pandas as pd, numpy as np, ccxt, requests
from ta.trend import EMAIndicator
from ta.momentum import RSIIndicator, StochRSIIndicator
from ta.volatility import AverageTrueRange

# =========================
# CONFIG
# =========================
RUN_EVERY_MIN = 60            # use 15 or 30 for tighter scans

EXCHANGES = ["okx", "coinbase", "kraken"]  # WA-friendly set
QUOTE_FILTERS = {"USDT", "USD"}            # scan only these quote currencies
EXCLUDE_KEYWORDS = ("UP", "DOWN", "BULL", "BEAR", "3L", "3S", "5L", "5S", "PERP")

# ---- Coarse filter thresholds (tune here) ----
MAX_PER_EXCHANGE      = 30
MIN_NOTIONAL_24H_USD  = 1e7      # $10M
MIN_RVOL_24H          = 1.5
PRICE_CHANGE_RANGE    = (0.03, 0.12)   # 3%..12% magnitude
ATR1H_RANGE           = (0.006, 0.03)  # 0.6%..3.0% of price

ALWAYS_INCLUDE = {
    "okx":     ["SOL/USDT","BCH/USDT","ONDO/USDT","LTC/USDT","MKR/USDT","XRP/USDT","DOGE/USDT"],
    "coinbase":["BTC/USDT","ETH/USDT","SOL/USDT","LTC/USDT","BCH/USDT"],
    "kraken":  ["BTC/USDT","ETH/USDT","SOL/USDT"]
}


LONG_SCORE_THRESHOLD  = 7.5
SHORT_SCORE_THRESHOLD = 7.5
MIN_NOTIONAL_15M_AVG  = 5e5   # ~$500k avg notional over last 96×15m; tweak as you like
TOP_N_SAVE = 100              # how many to keep in outputs

# Telegram (optional)
TG_BOT_TOKEN = os.getenv("TG_BOT_TOKEN", "")
TG_CHAT_ID   = os.getenv("TG_CHAT_ID", "")
MIN_BARS = {"15m": 60, "1h": 60, "4h": 60}   # skip pairs with too little history

# =========================
# HELPERS
# =========================
def round_tick(x, tick=0.0001):
    if x == 0: return 0.0
    k = max(0, -int(math.floor(math.log10(tick))))
    return round(float(x), k)

def safe_fetch_ohlcv(ex, symbol, tf, limit=400):
    """Fetch OHLCV with graceful fallback if exchange lacks timeframe."""
    try:
        return ex.fetch_ohlcv(symbol, timeframe=tf, limit=limit)
    except Exception:
        # Some exchanges alias 4h as '240m'; try common alternates
        alt = {"4h":"240m", "1h":"60m", "15m":"15m"}.get(tf, tf)
        if alt != tf:
            return ex.fetch_ohlcv(symbol, timeframe=alt, limit=limit)
        raise

def df_from_ohlcv(ohlcv):
    df = pd.DataFrame(ohlcv, columns=["ts","open","high","low","close","volume"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    return df
from ta.volatility import AverageTrueRange

def add_indicators_1h(df):
    df = df.copy()
    df["ema200"] = EMAIndicator(df.close, 200).ema_indicator()
    df["rsi14"]  = RSIIndicator(df.close, 14).rsi()
    df["atr"]    = AverageTrueRange(df.high, df.low, df.close, 14).average_true_range()
    return df

def pct_change_24h(df1h):
    if len(df1h) < 25: return 0.0
    a = float(df1h.close.iloc[-25])
    b = float(df1h.close.iloc[-1])
    return (b/a - 1.0) if a else 0.0

def atrpct_1h(df1h):
    last = df1h.iloc[-1]
    return float(last.atr / last.close) if last.close and pd.notna(last.atr) else 0.0

def rvol_24h_1h(df1h):
    """24h sum vs avg of previous 5 days; needs ~6 days of data."""
    if len(df1h) < 24*6: return 1.0
    last = float(df1h.volume.iloc[-24:].sum())
    prev = float(df1h.volume.iloc[-(24*6):-24].sum()) / 5.0
    return (last / prev) if prev > 0 else 1.0

def notional_24h(df1h):
    d = df1h.tail(24)
    return float((d.close * d.volume).sum())

def add_indicators(df):
    df = df.copy()
    # EMAs / RSI / Stoch RSI are fine with short series
    df["ema20"]  = EMAIndicator(df.close, 20).ema_indicator()
    df["ema50"]  = EMAIndicator(df.close, 50).ema_indicator()
    df["ema200"] = EMAIndicator(df.close, 200).ema_indicator()
    df["rsi14"]  = RSIIndicator(df.close, 14).rsi()
    st = StochRSIIndicator(df.close, window=14, smooth1=3, smooth2=3)
    df["k"] = st.stochrsi_k()
    df["d"] = st.stochrsi_d()

    # ---- SAFE ATR ----
    if len(df) >= 14:
        df["atr"] = AverageTrueRange(df.high, df.low, df.close, 14).average_true_range()
    else:
        # not enough bars → set NaN so we can skip later
        df["atr"] = np.nan

    return df


def rvol(series, window=96):  # 96×15m ≈ 24h
    cur = series.iloc[-1]
    avg = series.tail(window).mean()
    return float(cur/avg) if avg and np.isfinite(avg) else 1.0

def enumerate_symbols(ex):
    ex.load_markets()
    syms = []
    for m in ex.markets.values():
        if not m.get("active", True): 
            continue
        if m.get("type") not in (None, "spot"): 
            continue
        base, quote = m.get("base"), m.get("quote")
        symbol = m.get("symbol")
        if not symbol or not base or not quote: 
            continue
        if quote not in QUOTE_FILTERS: 
            continue
        if any(k in symbol for k in EXCLUDE_KEYWORDS): 
            continue
        syms.append(symbol)
    return sorted(set(syms))

def notional_volume(df15, lookback=96):
    """Approx 24h notional using 15m candles: sum(close*volume)."""
    d = df15.tail(lookback)
    return float((d.close * d.volume).sum())
def coarse_filter_candidates(ex, symbols):
    survivors = []
    for s in symbols:
        try:
            o1h = safe_fetch_ohlcv(ex, s, "1h", 24*8)  # ~8 days
        except Exception:
            continue
        if len(o1h) < 24*3:   # need some history
            continue

        d1h = add_indicators_1h(df_from_ohlcv(o1h))

        notional = notional_24h(d1h)
        if notional < MIN_NOTIONAL_24H_USD:
            continue

        rv = rvol_24h_1h(d1h)
        if rv < MIN_RVOL_24H:
            continue

        pc = abs(pct_change_24h(d1h))
        if not (PRICE_CHANGE_RANGE[0] <= pc <= PRICE_CHANGE_RANGE[1]):
            continue

        atrp = atrpct_1h(d1h)
        if not (ATR1H_RANGE[0] <= atrp <= ATR1H_RANGE[1]):
            continue

        # Composite priority for ranking (bigger is better)
        priority = rv * pc * (notional / 1e7)
        survivors.append((s, priority))

    # Always-include symbols (if present in markets)
    ex.load_markets()
    for keep in ALWAYS_INCLUDE.get(ex.id, []):
        if keep in ex.markets and ex.markets[keep].get("active", True):
            if keep not in [x[0] for x in survivors]:
                survivors.append((keep, float("inf")))  # pin to top

    survivors.sort(key=lambda x: x[1], reverse=True)
    return [s for s, _ in survivors[:MAX_PER_EXCHANGE]]

def telegram(msg):
    if not TG_BOT_TOKEN or not TG_CHAT_ID: 
        return
    try:
        requests.get(
            f"https://api.telegram.org/bot{TG_BOT_TOKEN}/sendMessage",
            params={"chat_id": TG_CHAT_ID, "text": msg, "parse_mode": "Markdown"}
        )
    except Exception as e:
        print("Telegram error:", e)

# =========================
# SCORING MODEL (0–10)
# =========================
def score_setup(sym, df15, df1h, df4h, btc1h_ctx):
    """
    Weighted model reflecting our playbook:
      4H trend/context, 1H bridge, 15m trigger, RVOL, ATR regime,
      distance-to-EMA (anti-chase), BTC tailwind/headwind.
    """
    last15, last1h, last4h = df15.iloc[-1], df1h.iloc[-1], df4h.iloc[-1]
    price = float(last15.close)

    # ---- Features ----
    up4  = last4h.close > last4h.ema200
    up1  = last1h.close > last1h.ema200
    stack4 = last4h.ema20 > last4h.ema50
    stack1 = last1h.ema20 > last1h.ema50
    rsi4 = float(last4h.rsi14)
    rsi1 = float(last1h.rsi14)
    kd_long  = (last15.k > last15.d) and (last15.k < 70)
    kd_short = (last15.k < last15.d) and (last15.k > 30)
    dist20 = abs(price - float(last15.ema20)) / price
    atrp = float(last15.atr / price)  # ATR as % of price
    rv = rvol(df15.volume, 96)

    # BTC context
    btc_up   = btc1h_ctx["up"]
    btc_down = btc1h_ctx["down"]

    # ---- Long score ----
    ls = 0.0
    ls += 2.5 if up4 else 0.0
    ls += 1.0 if stack4 else 0.0
    ls += 1.0 if rsi4 > 52 else 0.0
    ls += 1.5 if up1 else 0.0
    ls += 0.5 if stack1 else 0.0
    ls += 1.5 if rsi1 > 50 else 0.0
    ls += 1.5 if kd_long else 0.0
    # Distance/anti-chase
    if dist20 < 0.004: ls += 0.7        # near EMA20/50
    if last15.k > 90:  ls -= 0.8        # overbought -> wait for curl
    # Volatility regime
    if 0.008 <= atrp <= 0.02: ls += 0.7 # sweet spot
    elif atrp < 0.004:        ls -= 0.5 # too quiet -> chop risk
    elif atrp > 0.03:         ls -= 0.5 # too wild -> slippage
    # Relative volume
    if rv >= 1.5: ls += 0.8
    elif rv < 0.7: ls -= 0.5
    # BTC tailwind
    if btc_up:   ls += 0.6
    if btc_down: ls -= 0.8

    # ---- Short score (mirror) ----
    ss = 0.0
    ss += 2.5 if (last4h.close < last4h.ema200) else 0.0
    ss += 1.0 if (last4h.ema20 < last4h.ema50) else 0.0
    ss += 1.0 if rsi4 < 48 else 0.0
    ss += 1.5 if (last1h.close < last1h.ema200) else 0.0
    ss += 0.5 if (last1h.ema20 < last1h.ema50) else 0.0
    ss += 1.5 if rsi1 < 50 else 0.0
    ss += 1.5 if kd_short else 0.0
    if dist20 < 0.004: ss += 0.7
    if last15.k < 10:  ss -= 0.8        # oversold -> wait for curl
    if 0.008 <= atrp <= 0.02: ss += 0.7
    elif atrp < 0.004:        ss -= 0.5
    elif atrp > 0.03:         ss -= 0.5
    if rv >= 1.5: ss += 0.8
    elif rv < 0.7: ss -= 0.5
    if btc_down: ss += 0.6
    if btc_up:   ss -= 0.8

    # Clamp 0..10
    long_score  = float(np.clip(ls, 0, 10))
    short_score = float(np.clip(ss, 0, 10))

    # Suggested levels (EMA-blend entry + ATR sizing)
    ema_blend = float((last15.ema20 + last15.ema50)/2.0)
    atr = float(last15.atr)
    entry_long  = round_tick(ema_blend)
    sl_long     = round_tick(entry_long - 1.5*atr)
    tp1_long    = round_tick(entry_long + 1.5*atr)
    tp2_long    = round_tick(entry_long + 2.5*atr)

    entry_short = round_tick(ema_blend)
    sl_short    = round_tick(entry_short + 1.5*atr)
    tp1_short   = round_tick(entry_short - 1.5*atr)
    tp2_short   = round_tick(entry_short - 2.5*atr)

    return {
        "long_score": long_score,
        "short_score": short_score,
        "atrp": atrp, "rvol_15m": rv,
        "entry_long": entry_long, "sl_long": sl_long, "tp1_long": tp1_long, "tp2_long": tp2_long,
        "entry_short": entry_short, "sl_short": sl_short, "tp1_short": tp1_short, "tp2_short": tp2_short
    }

def btc_context(ex):
    """BTC 1H context used as a market filter."""
    try:
        o1h = safe_fetch_ohlcv(ex, "BTC/USDT", "1h", 400)
    except Exception:
        # fallback to USD if USDT pair not present
        o1h = safe_fetch_ohlcv(ex, "BTC/USD", "1h", 400)
    d1h = add_indicators(df_from_ohlcv(o1h))
    last = d1h.iloc[-1]
    up   = (last.close > last.ema200) and (last.rsi14 > 50)
    down = (last.close < last.ema200) and (last.rsi14 < 50)
    return {"up": bool(up), "down": bool(down)}

# =========================
# MAIN SCAN
# =========================
def scan_once():
    rows = []
    now = dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")

    for ex_id in EXCHANGES:
        ex = getattr(ccxt, ex_id)({"enableRateLimit": True})
        try:
            symbols_all = enumerate_symbols(ex)
            symbols = coarse_filter_candidates(ex, symbols_all)

        except Exception as e:
            print(f"[{ex_id}] load_markets failed:", e)
            continue

        # BTC context per exchange
        try:
            ctx = btc_context(ex)
        except Exception as e:
            print(f"[{ex_id}] btc_context failed:", e)
            ctx = {"up": False, "down": False}

        for s in symbols:
            try:
                o15 = safe_fetch_ohlcv(ex, s, "15m", 400)
                o1h = safe_fetch_ohlcv(ex, s, "1h", 400)
                o4h = safe_fetch_ohlcv(ex, s, "4h", 400)

                if len(o15) < MIN_BARS["15m"] or len(o1h) < MIN_BARS["1h"] or len(o4h) < MIN_BARS["4h"]:
    # print(f"[skip bars] {ex_id}:{s} -> 15m:{len(o15)} 1h:{len(o1h)} 4h:{len(o4h)}")
                    continue
            except Exception as e:
                # skip pairs that lack one of the timeframes
                # print(f"[skip tf] {ex_id}:{s} -> {e}")
                continue

            d15 = add_indicators(df_from_ohlcv(o15))
            d1h = add_indicators(df_from_ohlcv(o1h))
            d4h = add_indicators(df_from_ohlcv(o4h))

            need = ["ema20","ema50","ema200","rsi14"]
            if (pd.isna(d15.iloc[-1]["atr"]) or
                any(pd.isna(d15.iloc[-1][need])) or
                any(pd.isna(d1h.iloc[-1][need])) or
                any(pd.isna(d4h.iloc[-1][need]))):
    # print(f"[skip nan] {ex_id}:{s}")
                continue

            # liquidity filter
            notional = notional_volume(d15, 96)
            if notional < MIN_NOTIONAL_15M_AVG:
                continue

            plan = score_setup(s, d15, d1h, d4h, ctx)
            best_dir   = "LONG" if plan["long_score"] >= plan["short_score"] else "SHORT"
            best_score = max(plan["long_score"], plan["short_score"])

            rows.append({
                "exchange": ex_id,
                "symbol": s,
                "close_15m": float(d15.iloc[-1].close),
                "best_dir": best_dir,
                "best_score": round(best_score, 2),
                **plan
            })

    if not rows:
        print("No candidates found.")
        return

    df = pd.DataFrame(rows).sort_values("best_score", ascending=False).head(TOP_N_SAVE)
    df.to_csv("wa_screener_results.csv", index=False)
    with open("wa_screener_results.json","w") as f:
        json.dump({"timestamp": now, "results": df.to_dict(orient="records")}, f, indent=2)

    print(f"\n=== {now} — Top {min(TOP_N_SAVE,len(df))} ===")
    print(df[["exchange","symbol","best_dir","best_score","long_score","short_score","close_15m",
              "entry_long","sl_long","tp1_long","tp2_long","entry_short","sl_short","tp1_short","tp2_short"]].head(20))

    # Telegram alerts (optional)
    for _, r in df.iterrows():
        if r["best_dir"]=="LONG" and r["best_score"]>=LONG_SCORE_THRESHOLD and TG_BOT_TOKEN:
            telegram(f"🟢 *LONG {r['exchange']} {r['symbol']}* score {r['long_score']}/10\n"
                     f"Entry {r['entry_long']} | SL {r['sl_long']} | TP1 {r['tp1_long']} | TP2 {r['tp2_long']}")
        if r["best_dir"]=="SHORT" and r["best_score"]>=SHORT_SCORE_THRESHOLD and TG_BOT_TOKEN:
            telegram(f"🔴 *SHORT {r['exchange']} {r['symbol']}* score {r['short_score']}/10\n"
                     f"Entry {r['entry_short']} | SL {r['sl_short']} | TP1 {r['tp1_short']} | TP2 {r['tp2_short']}")

if __name__ == "__main__":
    scan_once()
    # or run as a daemon:
    # while True:
    #     scan_once()
    #     time.sleep(RUN_EVERY_MIN*60)
