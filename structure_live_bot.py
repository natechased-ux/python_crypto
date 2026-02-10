
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Structure Live Alert Bot (Time-Decaying Confirmation + Scoring)
---------------------------------------------------------------
- 1H structure setup with minimal hard filters (ADX + swing-based SL/TP)
- 15m Stoch RSI confirmation with time-decaying bands (strict → relaxed)
- Score (0–4) from soft filters only; entry depends on confirmation, not score
  +1 Trend bias (1D EMA200)
  +1 1H EMA stacking
  +1 MACD histogram zero-cross on last closed 1H candle
  +1 RSI regime (≥50 long, ≤50 short)
- Cooldown per symbol/side to avoid duplicate alerts
- Telegram message matches your original format (+ Score + EMA guidance)

Fill in TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID, then run:
    python structure_live_bot.py
"""
import os, time, datetime as dt
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import requests
import numpy as np
import pandas as pd
# ================== User Config ==================

TELEGRAM_BOT_TOKEN = "8177096945:AAHTg5nxVTA6hcPkAL4MLAAHPsZ1at7Ywmw"
TELEGRAM_CHAT_ID = "7967738614"   # set env var or paste string

SYMBOLS = [
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
# Best combo parameters from backtest
BASE_R_MULT = 1.1
ADX_MIN     = 14
BB_PCTL_MIN = 0.0  # disabled effectively

# Confirmation (time-decaying)
CONFIRM_TIMEOUT_MIN = 120
STOCH_RSI_LEN = 14
STOCH_RSI_SMOOTH_K = 3
STOCH_RSI_SMOOTH_D = 3

# Soft filters for scoring (informational only)
USE_TREND_BIAS   = True
USE_EMA_STACKING = True
USE_MACD_ZERO    = True
USE_RSI_REGIME   = True

# Core indicator settings
ATR_LEN = 14
ADX_LEN = 14
RSI_LEN = 14
MACD_FAST, MACD_SLOW, MACD_SIG = 12, 26, 9
EMA_SET = [10,20,50,200]
BB_LEN, BB_STD, BB_LOOKBACK = 20, 2.0, 120

# Anti-micro-stop guard
MIN_STOP_ATR = 0.25   # stop distance must be ≥ 0.25 * ATR(14)

# Cooldown per symbol/side
COOLDOWN_HOURS = 6

# Poll cadence and pacing
POLL_SECONDS = 60
SYMBOL_SLEEP = 0.8  # pause after each symbol to reduce rate limits

# Data source & batching
CB_BASE = "https://api.exchange.coinbase.com"
MAX_PER_REQ = 250  # smaller chunks are friendlier

# ================== TA helpers ==================
def ema(s:pd.Series,n:int)->pd.Series: return s.ewm(span=n,adjust=False).mean()
def rsi(s:pd.Series,n:int)->pd.Series:
    d=s.diff(); up=d.clip(lower=0); dn=-d.clip(upper=0)
    ru=up.ewm(alpha=1/n,adjust=False).mean(); rd=dn.ewm(alpha=1/n,adjust=False).mean()
    rs=ru/(rd+1e-12); return 100-(100/(1+rs))
def macd(close:pd.Series, f:int, sl:int, sg:int):
    fe=ema(close,f); se=ema(close,sl); line=fe-se; sig=ema(line,sg); hist=line-sig; return line,sig,hist
def true_range(h:pd.Series,l:pd.Series,c:pd.Series)->pd.Series:
    pc=c.shift(1); return pd.concat([h-l,(h-pc).abs(),(l-pc).abs()],axis=1).max(axis=1)
def atr(h:pd.Series,l:pd.Series,c:pd.Series,n:int)->pd.Series: return true_range(h,l,c).ewm(alpha=1/n,adjust=False).mean()
def adx(h:pd.Series,l:pd.Series,c:pd.Series,n:int):
    up=h.diff(); dn=-l.diff()
    plus=np.where((up>dn)&(up>0),up,0.0); minus=np.where((dn>up)&(dn>0),dn,0.0)
    tr=true_range(h,l,c); atr_v=tr.ewm(alpha=1/n,adjust=False).mean()
    pdi=100*pd.Series(plus,index=h.index).ewm(alpha=1/n,adjust=False).mean()/(atr_v+1e-12)
    mdi=100*pd.Series(minus,index=h.index).ewm(alpha=1/n,adjust=False).mean()/(atr_v+1e-12)
    dx=100*(pdi-mdi).abs()/((pdi+mdi)+1e-12); adx_v=dx.ewm(alpha=1/n,adjust=False).mean()
    return adx_v,pdi,mdi
def stoch_rsi(close:pd.Series,n:int,sk:int,sd:int):
    r=rsi(close,n); lo=r.rolling(n,min_periods=n).min(); hi=r.rolling(n,min_periods=n).max()
    st=(r-lo)/(hi-lo+1e-12)*100.0; k=st.rolling(sk,min_periods=sk).mean(); d=k.rolling(sd,min_periods=sd).mean()
    return k,d
def bb_width(close:pd.Series,n:int,m:float)->pd.Series:
    ma=close.rolling(n).mean(); sd=close.rolling(n).std(); up=ma+m*sd; lo=ma-m*sd
    return (up-lo)/(ma.abs()+1e-12)

# ================== Utilities ==================
def tg_send(text: str):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("[TELEGRAM] Missing token/chat_id. Message:\n", text)
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": "Markdown"}
    try:
        r = requests.post(url, json=payload, timeout=15)
        if r.status_code != 200:
            print("[TELEGRAM] Error:", r.text)
    except Exception as e:
        print("[TELEGRAM] Exception:", e)

def fmt_price(p: float) -> str:
    if p >= 100: return f"{p:,.2f}"
    if p >= 1:   return f"{p:,.4f}"
    return f"{p:,.6f}"

# Product cache
PRODUCTS_CACHE: Optional[set] = None
def product_exists(symbol: str) -> bool:
    """Check if symbol is a valid Coinbase product (cached)."""
    global PRODUCTS_CACHE
    if PRODUCTS_CACHE is None:
        url = f"{CB_BASE}/products"
        r = requests.get(url, timeout=20)
        r.raise_for_status()
        ids = [p.get("id","") for p in r.json() if isinstance(p, dict)]
        PRODUCTS_CACHE = set(ids)
    return symbol in PRODUCTS_CACHE

# ================== Data Fetch ==================
def chunk_timerange(start: dt.datetime, end: dt.datetime, gran_s: int, max_points: int):
    step = dt.timedelta(seconds=gran_s*max_points - gran_s)
    cur = start; out = []
    while cur < end:
        nxt = min(end, cur + step); out.append((cur, nxt)); cur = nxt
    return out

def fetch_candles_range(symbol: str, granularity: int, start: dt.datetime, end: dt.datetime) -> pd.DataFrame:
    now = dt.datetime.now(dt.timezone.utc)
    safe_end = min(end, now - dt.timedelta(seconds=granularity))
    if safe_end <= start:
        return pd.DataFrame(columns=["time","low","high","open","close","volume","dt"])

    frames: List[pd.DataFrame] = []
    for s,e in chunk_timerange(start, safe_end, granularity, MAX_PER_REQ):
        params = {"granularity": granularity, "start": int(s.timestamp()), "end": int(e.timestamp())}
        url = f"{CB_BASE}/products/{symbol}/candles"
        backoff = 0.5
        for attempt in range(8):
            try:
                r = requests.get(url, params=params, timeout=20)
                if r.status_code in (429,500,502,503,504):
                    raise requests.HTTPError(f"{r.status_code} {r.text}")
                r.raise_for_status()
                arr = r.json()
                if not isinstance(arr, list):
                    raise ValueError(f"Unexpected response: {arr}")
                df = pd.DataFrame(arr, columns=["time","low","high","open","close","volume"])
                if not df.empty:
                    frames.append(df)
                break
            except Exception as ex:
                if attempt == 7:
                    print(f"[{symbol}] fetch retry exhausted {s}->{e}: {ex}")
                else:
                    time.sleep(backoff)
                    backoff = min(30.0, backoff * 2.0)  # stronger backoff
        time.sleep(0.1)  # intra-chunk pacing

    if not frames:
        return pd.DataFrame(columns=["time","low","high","open","close","volume","dt"])

    df = pd.concat(frames, ignore_index=True)
    if df.empty:
        return pd.DataFrame(columns=["time","low","high","open","close","volume","dt"])
    df = df.drop_duplicates("time").sort_values("time").reset_index(drop=True)
    df["dt"] = pd.to_datetime(df["time"], unit="s", utc=True)
    for c in ["open","high","low","close","volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

# ================== Strategy helpers ==================
def last_swing_levels(df: pd.DataFrame, lookback: int = 25):
    if len(df) < lookback + 3: return None, None
    H=df["high"].values; L=df["low"].values; sh=sl=None
    for i in range(len(df)-2,2,-1):
        if sh is None and H[i]>H[i-1] and H[i]>H[i-2] and H[i]>H[i+1]: sh=float(H[i])
        if sl is None and L[i]<L[i-1] and L[i]<L[i-2] and L[i]<L[i+1]: sl=float(L[i])
        if sh is not None and sl is not None: break
    return sh, sl

def ema_guidance(df1h: pd.DataFrame, df1d: pd.DataFrame) -> Tuple[Optional[Dict[str,float]], Optional[Dict[str,float]]]:
    # Require enough bars for stable EMAs (and .iloc[-2])
    if len(df1d) < 202 or len(df1h) < 60:
        return None, None
    e1h = {f"EMA{L}": float(ema(df1h["close"], L).iloc[-2]) for L in EMA_SET}
    e1d = {f"EMA{L}": float(ema(df1d["close"], L).iloc[-2]) for L in EMA_SET}
    return e1h, e1d

def ema_stacking_ok(e1h: Dict[str, float], side: str) -> bool:
    e10,e20,e50 = e1h["EMA10"], e1h["EMA20"], e1h["EMA50"]
    return (e10>e20>e50) if side=="LONG" else (e10<e20<e50)

@dataclass
class Signal:
    symbol: str; side: str; entry: float; tp: float; sl: float; sig_time: dt.datetime
    adx: float; rsi: float; macd_hist: float; atr: float; bb_width: float
    e1h: Optional[Dict[str,float]]; e1d: Optional[Dict[str,float]]; score: int

def build_structure_signal(df1h: pd.DataFrame, df1d: pd.DataFrame) -> Optional[Signal]:
    # Strict guards to avoid .iloc[-2] issues
    if len(df1h) < 120:
        return None

    c=df1h["close"]; h=df1h["high"]; l=df1h["low"]
    adx_v,pdi,mdi = adx(h,l,c,ADX_LEN)
    _,_,hist = macd(c,MACD_FAST,MACD_SLOW,MACD_SIG)
    r = rsi(c,RSI_LEN)
    atr1h = atr(h,l,c,ATR_LEN)
    bbw = bb_width(c,BB_LEN,BB_STD)

    # last closed 1H bar (index -2), previous (-3)
    try:
        adx2=float(adx_v.iloc[-2]); pdi2=float(pdi.iloc[-2]); mdi2=float(mdi.iloc[-2])
        hist2=float(hist.iloc[-2]); hist3=float(hist.iloc[-3]); r2=float(r.iloc[-2])
        atr2=float(atr1h.iloc[-2]); bbw2=float(bbw.iloc[-2])
        last_close=float(c.iloc[-2]); ts=pd.to_datetime(df1h["dt"].iloc[-2]).to_pydatetime()
    except Exception:
        return None

    # Minimal hard filters: ADX + swing + ATR guard
    sh, slv = last_swing_levels(df1h, 25)
    long_setup  = (adx2 > ADX_MIN and pdi2 > mdi2 and slv is not None)
    short_setup = (adx2 > ADX_MIN and mdi2 > pdi2 and sh  is not None)

    if long_setup:
        side="LONG"; entry=last_close; sl=slv; risk=max(1e-8, entry - sl)
    elif short_setup:
        side="SHORT"; entry=last_close; sl=sh;  risk=max(1e-8, sl - entry)
    else:
        return None

    # ATR-based micro-stop guard
    if (risk / max(1e-12, atr2)) < MIN_STOP_ATR:
        return None

    # Dynamic R-multiple (same as backtest)
    rmult = BASE_R_MULT
    ratio = risk/max(1e-12, atr2)
    if ratio < 0.5: rmult = min(1.2, BASE_R_MULT+0.3)
    elif ratio > 1.8: rmult = max(0.5, BASE_R_MULT-0.25)
    tp = entry + rmult*risk if side=="LONG" else entry - rmult*risk

    e1h, e1d = ema_guidance(df1h, df1d)

    # Score (informational only)
    score = 0
    if USE_TREND_BIAS and e1d:
        if (side=="LONG" and entry>e1d["EMA200"]) or (side=="SHORT" and entry<e1d["EMA200"]): score += 1
    if USE_EMA_STACKING and e1h:
        if ema_stacking_ok(e1h, side): score += 1
    if USE_MACD_ZERO:
        if (side=="LONG" and hist2>0 and hist3<=0) or (side=="SHORT" and hist2<0 and hist3>=0): score += 1
    if USE_RSI_REGIME:
        if (side=="LONG" and r2>=50) or (side=="SHORT" and r2<=50): score += 1

    return Signal("", side, entry, tp, sl, ts, adx2, r2, hist2, atr2, bbw2, e1h, e1d, score)

# Time-decaying 15m confirmation
def confirm_time_15m(df15: pd.DataFrame, side: str, start_ts: dt.datetime, timeout_min: int) -> Optional[dt.datetime]:
    if len(df15) < 30:
        return None
    end_ts = start_ts + dt.timedelta(minutes=timeout_min)
    d = df15[df15["dt"] > start_ts].copy()
    if d.empty: return None
    k, dd = stoch_rsi(d["close"], STOCH_RSI_LEN, STOCH_RSI_SMOOTH_K, STOCH_RSI_SMOOTH_D)
    d["K"], d["D"] = k, dd
    d = d.dropna().reset_index(drop=True)
    for i in range(1, len(d)):
        ts = d.loc[i,"dt"].to_pydatetime()
        if ts > end_ts: break
        k1, d1 = float(d.loc[i,"K"]), float(d.loc[i,"D"])
        k0 = float(d.loc[i-1,"K"]); slope = k1 - k0
        long_ceiling  = 40.0 + min(20.0, 2.0*i)   # 40→60
        short_floor   = 60.0 - min(20.0, 2.0*i)   # 60→40
        long_ok  = ((k1 > d1) and (k1 < long_ceiling)) or (slope > 0 and k1 < 70.0)
        short_ok = ((k1 < d1) and (k1 > short_floor)) or (slope < 0 and k1 > 30.0)
        if (side=="LONG" and long_ok) or (side=="SHORT" and short_ok):
            return ts
    return None

# ================== Alert formatting ==================
def alert_text(sym: str, sig: Signal, confirm_ts: dt.datetime) -> str:
    side_icon = "✅big *LONG Confirmed*" if sig.side=="LONG" else "✅big *SHORT Confirmed*"
    lines = [
        f"{side_icon} on {sym.replace('-', '')}",
        f"Entry: {fmt_price(sig.entry)}",
        f"TP: {fmt_price(sig.tp)}",
        f"SL: {fmt_price(sig.sl)}",
        confirm_ts.strftime("%Y-%m-%d %I:%M %p UTC"),
        ""
    ]

    # EMA guidance (optional — only if we have EMAs)
    if sig.e1d and sig.e1h:
        lines.append("📊 EMA Guidance (within 10% of entry):")
        above, below = [], []
        for label in ["EMA200","EMA50","EMA20","EMA10"]:
            v1d = sig.e1d.get(label)
            v1h = sig.e1h.get(label)
            if v1d is not None:
                (above if v1d >= sig.entry else below).append(f"• 1D {label}: {fmt_price(v1d)}")
            if v1h is not None:
                (above if v1h >= sig.entry else below).append(f"• 1H {label}: {fmt_price(v1h)}")
        if above:
            lines.append("⬆️ *Resistance EMAs Above Entry:*"); lines.extend(above)
        if below:
            lines.append("⬇️ *Support EMAs Below Entry:*"); lines.extend(below)
        lines.append("")

    # Score block
    lines.append("📊 ")
    lines.append(f"*Score*: +{sig.score}")
    if USE_TREND_BIAS:
        tb = "✅" if (sig.e1d and ((sig.side=="LONG" and sig.entry>sig.e1d.get("EMA200", float('inf'))) or (sig.side=="SHORT" and sig.entry<sig.e1d.get("EMA200", float('-inf'))))) else "—"
        lines.append(f"{tb} Trend vs 1D EMA200")
    if USE_EMA_STACKING:
        st = "✅" if (sig.e1h and ema_stacking_ok(sig.e1h, sig.side)) else "—"
        lines.append(f"{st} 1H EMA stacking")
    if USE_MACD_ZERO:
        mz = "✅" if ((sig.side=="LONG" and sig.macd_hist>0) or (sig.side=="SHORT" and sig.macd_hist<0)) else "—"
        lines.append(f"{mz} MACD hist sign/zero-cross")
    if USE_RSI_REGIME:
        rg = "✅" if ((sig.side=="LONG" and sig.rsi>=50) or (sig.side=="SHORT" and sig.rsi<=50)) else "—"
        lines.append(f"{rg} RSI regime")
    return "\n".join(lines)

# ================== Runner ==================
def run_once(cooldown_map: Dict[Tuple[str,str], dt.datetime]):
    now = dt.datetime.now(dt.timezone.utc)

    # Lighter lookbacks (enough for indicators, kinder to API)
    start_1d = now - dt.timedelta(days=250)  # EMA200 + buffer
    start_1h = now - dt.timedelta(days=10)   # ~240 bars
    start_15 = now - dt.timedelta(days=2)    # ~192 bars

    for sym_raw in SYMBOLS:
        sym = sym_raw.upper()
        try:
            # Validate product once
            if not product_exists(sym):
                print(f"[{sym}] not a valid Coinbase product; skipping")
                continue

            df1d = fetch_candles_range(sym, 86400, start_1d, now)
            df1h = fetch_candles_range(sym, 3600,  start_1h, now)
            df15 = fetch_candles_range(sym, 900,   start_15, now)

            # History guards to avoid .iloc issues
            if len(df1h) < 120 or len(df15) < 30:
                # 1D can be thin; we handle EMA guidance optionally later
                continue

            sig = build_structure_signal(df1h, df1d)
            if not sig: 
                time.sleep(SYMBOL_SLEEP); 
                continue
            sig.symbol = sym

            # Cooldown check (per symbol & side)
            key = (sym, sig.side)
            last = cooldown_map.get(key)
            if last and (sig.sig_time - last).total_seconds() < COOLDOWN_HOURS*3600:
                time.sleep(SYMBOL_SLEEP)
                continue

            cts = confirm_time_15m(df15, sig.side, sig.sig_time, CONFIRM_TIMEOUT_MIN)
            if not cts:
                time.sleep(SYMBOL_SLEEP)
                continue  # no entry without confirmation

            txt = alert_text(sym, sig, cts)
            tg_send(txt)
            cooldown_map[key] = sig.sig_time

        except Exception as e:
            print(f"[{sym}] error: {e}")

        # Per-symbol pacing
        time.sleep(SYMBOL_SLEEP)

def main():
    print("Structure Live Alert Bot starting...")
    cooldown_map: Dict[Tuple[str,str], dt.datetime] = {}
    while True:
        run_once(cooldown_map)
        time.sleep(POLL_SECONDS)

if __name__ == "__main__":
    main()
