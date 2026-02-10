"""
Unified Crypto Alert Bot — hardened single-file

What changed vs previous draft (be blunt):
- Kept: 6H structure ➜ lower-TF confirmation; regime-adaptive 1H signals.
- Added: whale flow confluence (Coinbase WebSocket 'matches'), time‑decaying StochRSI confirmation (15m),
         breakout clearance (≥0.25–1.0 ATR beyond prior high/low), volume gating, 0–100 score, partial‑TP + trail blueprint,
         robust candle fetch with backoff, cleaner cooldowns & logging.
- Cut (default OFF): Fibonacci "golden zone" and overly chatty EMA dumps; they add noise without clear edge unless proven.

Dependencies: requests, pandas, numpy. Optional: websocket-client (enables whale flow). No TA libs required.
"""
from __future__ import annotations
import time
import math
import json
import threading
import datetime as dt
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests

try:
    import websocket  # websocket-client
except Exception:
    websocket = None

# ============================
# ---- Configuration ---------
# ============================
TELEGRAM_BOT_TOKEN = "8177096945:AAHTg5nxVTA6hcPkAL4MLAAHPsZ1at7Ywmw"
TELEGRAM_CHAT_ID   = "7967738614"

COINS = ["eth-usd","xrp-usd","ltc-usd","ada-usd","doge-usd",
    "sol-usd","wif-usd","ondo-usd","sei-usd","magic-usd","ape-usd",
    "jasmy-usd","wld-usd","syrup-usd","fartcoin-usd","aero-usd",
    "link-usd","hbar-usd","aave-usd","fet-usd","crv-usd","tao-usd",
    "avax-usd","xcn-usd","uni-usd","mkr-usd","toshi-usd","near-usd",
    "algo-usd","trump-usd","bch-usd","inj-usd","pepe-usd","xlm-usd",
    "moodeng-usd","bonk-usd","dot-usd","popcat-usd","arb-usd","icp-usd",
    "tia-usd","ip-usd","pnut-usd","apt-usd","ena-usd","turbo-usd",
    "bera-usd","pol-usd","mask-usd","pyth-usd","sand-usd","morpho-usd",
    "mana-usd","c98-usd","axs-usd"]

# Modules
ENABLE_STRUCTURE_6H   = True    # 6H structure ➜ 15m decaying StochRSI confirmation
ENABLE_REGIME_1H      = True    # Regime-adaptive 1H (breakout with ATR clearance, or mean-reversion in ranges)
ENABLE_WHALE_FILTER   = True    # Use WebSocket whale flow as confluence (auto-disables if websocket-client missing)
REQUIRE_WHALE_FOR_MR  = True   # If True, require whale alignment for mean-reversion entries
ENABLE_FVG_CONFLUENCE = False   # Off by default; low proven edge without backtest

# Structure confirmation window (like your best-performing versions)
CONFIRM_TIMEOUT_MIN = 360    # 6 hours

# Risk
SL_ATR_MULT = 1.5
TP_ATR_MULT = 2.0
PARTIAL_TP_SHARE = 0.5       # Mentioned in alert; you still manage execution
TRAIL_ATR_AFTER_TP = 1.0     # Trail blueprint after TP1

# Cadence & cooldowns
SCAN_INTERVAL_SECONDS = 60
COOLDOWN_MINUTES = 60        # per-symbol cooldown after an alert (tight but sane)

# ============================
# ---- Utilities -------------
# ============================
CB_BASE = "https://api.exchange.coinbase.com"


def now_utc() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def fmt_price(p: float) -> str:
    if p <= 0: return f"{p:.2f}"
    if p < 0.001: return f"{p:.8f}"
    if p < 0.01:  return f"{p:.6f}"
    if p < 1:     return f"{p:.4f}"
    if p < 100:   return f"{p:.2f}"
    return f"{p:.2f}"

# ============================
# ---- Data Fetch (robust) ---
# ============================
MAX_PER_REQ = 250

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
        return pd.DataFrame(columns=["time","open","high","low","close","volume","dt"])    
    frames: List[pd.DataFrame] = []
    for s,e in chunk_timerange(start, safe_end, granularity, MAX_PER_REQ):
        params = {"granularity": granularity, "start": int(s.timestamp()), "end": int(e.timestamp())}
        url = f"{CB_BASE}/products/{symbol}/candles"
        backoff = 0.5
        for attempt in range(8):
            try:
                r = requests.get(url, params=params, timeout=20)
                if r.status_code in (429,500,502,503,504):
                    raise requests.HTTPError(f"{r.status_code}")
                r.raise_for_status()
                arr = r.json() or []
                if arr:
                    frames.append(pd.DataFrame(arr, columns=["time","low","high","open","close","volume"]))
                break
            except Exception:
                if attempt == 7:
                    break
                time.sleep(backoff); backoff = min(30.0, backoff*2)
        time.sleep(0.08)
    if not frames:
        return pd.DataFrame(columns=["time","open","high","low","close","volume","dt"])    
    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates("time").sort_values("time").reset_index(drop=True)
    df["dt"] = pd.to_datetime(df["time"], unit="s", utc=True)
    for c in ["open","high","low","close","volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df[["dt","open","high","low","close","volume"]]

# ============================
# ---- Indicators ------------
# ============================

def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = (delta.clip(lower=0)).ewm(alpha=1/period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/period, adjust=False).mean()
    rs = gain / (loss.replace(0, np.nan))
    return 100 - (100 / (1 + rs))


def true_range(h: pd.Series, l: pd.Series, c: pd.Series) -> pd.Series:
    pc = c.shift(1)
    return pd.concat([h-l,(h-pc).abs(),(l-pc).abs()], axis=1).max(axis=1)


def atr(h: pd.Series, l: pd.Series, c: pd.Series, n: int = 14) -> pd.Series:
    return true_range(h,l,c).ewm(alpha=1/n, adjust=False).mean()


def adx(h: pd.Series, l: pd.Series, c: pd.Series, n: int = 14) -> Tuple[pd.Series,pd.Series,pd.Series]:
    up = h.diff(); dn = -l.diff()
    plus_dm  = np.where((up>dn)&(up>0), up, 0.0)
    minus_dm = np.where((dn>up)&(dn>0), dn, 0.0)
    tr = true_range(h,l,c); atr_w = tr.ewm(alpha=1/n, adjust=False).mean()
    pdi = 100*pd.Series(plus_dm, index=h.index).ewm(alpha=1/n, adjust=False).mean()/(atr_w+1e-12)
    mdi = 100*pd.Series(minus_dm,index=h.index).ewm(alpha=1/n, adjust=False).mean()/(atr_w+1e-12)
    dx = (100*(pdi-mdi).abs()/((pdi+mdi)+1e-12)).ewm(alpha=1/n, adjust=False).mean()
    return dx,pdi,mdi


def macd(series: pd.Series, fast=12, slow=26, signal=9) -> Tuple[pd.Series,pd.Series,pd.Series]:
    ema_fast = ema(series, fast); ema_slow = ema(series, slow)
    line = ema_fast - ema_slow; sig = ema(line, signal); hist = line - sig
    return line, sig, hist


def bollinger(close: pd.Series, n: int = 20, m: float = 2.0) -> Tuple[pd.Series,pd.Series,pd.Series,pd.Series]:
    mid = close.rolling(n).mean(); sd = close.rolling(n).std(ddof=0)
    up = mid + m*sd; lo = mid - m*sd; width = (up - lo) / (mid + 1e-12)
    return mid, up, lo, width

# ============================
# ---- Whale Flow ------------
# ============================
class WhaleWatcher:
    WS_URL = "wss://ws-feed.exchange.coinbase.com"
    def __init__(self, symbols: List[str], min_usd=100_000.0, near_price_pct=0.25, max_age_sec=600):
        self.symbols = symbols
        self.min_usd = float(min_usd)
        self.near_pct = float(near_price_pct) / 100.0
        self.max_age = int(max_age_sec)
        self._data: Dict[str, List[dict]] = {s: [] for s in symbols}
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self):
        if websocket is None:
            return
        def run():
            ws = None
            while not self._stop.is_set():
                try:
                    ws = websocket.create_connection(self.WS_URL, timeout=30)
                    sub = {"type":"subscribe","channels":[{"name":"matches","product_ids":self.symbols}]}
                    ws.send(json.dumps(sub))
                    while not self._stop.is_set():
                        raw = ws.recv()
                        if not raw: continue
                        msg = json.loads(raw)
                        if msg.get("type") != "match":
                            continue
                        product = msg.get("product_id"); price = float(msg.get("price",0) or 0)
                        size = float(msg.get("size",0) or 0); side = msg.get("side"); ts = msg.get("time")
                        notional = price*size
                        if product in self._data and notional >= self.min_usd:
                            self._data[product].append({"ts": ts, "price": price, "side": side, "usd": notional, "t": now_utc()})
                            # prune
                            self._data[product] = [x for x in self._data[product] if (now_utc()-x["t"]).total_seconds() <= self.max_age]
                except Exception:
                    time.sleep(1)
                finally:
                    try:
                        if ws: ws.close()
                    except Exception:
                        pass
        self._thread = threading.Thread(target=run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2)

    def aligned(self, symbol: str, side: str, ref_price: float) -> bool:
        dq = self._data.get(symbol, [])
        for p in reversed(dq):
            if abs(p["price"] - ref_price)/max(ref_price,1e-9) > self.near_pct:
                continue
            if side == "LONG" and p["side"] == "buy": return True
            if side == "SHORT" and p["side"] == "sell": return True
        return False

# ============================
# ---- Scoring 0–100 ---------
# ============================
@dataclass
class ScoreParts:
    adx: int; slope: int; macd: int; rsi: int; clearance: int; ema_dist: int


def score_signal(side: str, adx_val: float, adx_slope: float, macd_hist: float, macd_slope: float,
                 rsi_val: float, ema20: float, entry: float, atr_val: float, clearance_atr: float) -> Tuple[int, ScoreParts]:
    s = 0
    adx_part = int(max(0, min(1.0, (adx_val - 25.0)/15.0)) * 20); s += adx_part
    slope_part = int(max(0, min(1.0, adx_slope/0.5)) * 15) if adx_slope>0 else 0; s += slope_part
    macd_part = (10 if ((side=="LONG" and macd_hist>0) or (side=="SHORT" and macd_hist<0)) else 0) + \
                (5  if ((side=="LONG" and macd_slope>0) or (side=="SHORT" and macd_slope<0)) else 0); s += macd_part
    if side=="LONG": rsi_part = int(max(0, min(1.0, (rsi_val-55.0)/10.0))*15)
    else:            rsi_part = int(max(0, min(1.0, (45.0-rsi_val)/10.0))*15); s += rsi_part
    clr_part = int(max(0, min(1.0, (clearance_atr-0.25)/0.75))*15); s += clr_part
    ema_part = 0
    if atr_val>0 and ema20>0:
        dist_atr = abs(entry - ema20)/atr_val
        ema_part = int(max(0, 1.0 - min(dist_atr/2.0, 1.0))*15); s += ema_part
    return min(100, s), ScoreParts(adx_part, slope_part, macd_part, rsi_part, clr_part, ema_part)

# ============================
# ---- Confirm (15m decaying) -
# ============================

def stoch_rsi(close: pd.Series, rsi_len=14, stoch_len=14, smooth_k=3, smooth_d=3) -> Tuple[pd.Series, pd.Series]:
    r = rsi(close, rsi_len)
    lo = r.rolling(stoch_len).min(); hi = r.rolling(stoch_len).max()
    st = 100*(r - lo)/(hi - lo + 1e-12)
    k = st.rolling(smooth_k).mean(); d_ = k.rolling(smooth_d).mean()
    return k, d_


def confirm_time_15m(df15: pd.DataFrame, side: str, start_ts: dt.datetime, timeout_min: int) -> Optional[dt.datetime]:
    if len(df15) < 30: return None
    end_ts = start_ts + dt.timedelta(minutes=timeout_min)
    d = df15[df15["dt"] > start_ts].copy()
    if d.empty: return None
    k, dd = stoch_rsi(d["close"])
    d["K"], d["D"] = k, dd
    d = d.dropna().reset_index(drop=True)
    for i in range(1, len(d)):
        ts = d.loc[i, "dt"].to_pydatetime()
        if ts > end_ts: break
        k1, d1 = float(d.loc[i,"K"]), float(d.loc[i,"D"])
        k0 = float(d.loc[i-1,"K"]); slope = k1 - k0
        long_ceiling  = 40.0 + min(20.0, 2.0*i)  # 40→60
        short_floor   = 60.0 - min(20.0, 2.0*i)  # 60→40
        long_ok  = ((k1 > d1) and (k1 < long_ceiling)) or (slope > 0 and k1 < 70.0)
        short_ok = ((k1 < d1) and (k1 > short_floor)) or (slope < 0 and k1 > 30.0)
        if (side=="LONG" and long_ok) or (side=="SHORT" and short_ok):
            return ts
    return None

# ============================
# ---- Signals ----------------
# ============================
@dataclass
class Signal:
    symbol: str; module: str; side: str; entry: float; tp: float; sl: float; sig_time: dt.datetime
    atr: float; adx: float; rsi: float; macd_hist: float; macd_slope: float; bbw: float
    ema20: float; clearance_atr: float; score: int; parts: ScoreParts


def structure_signal_6h(df6: pd.DataFrame, df1h: pd.DataFrame, df15: pd.DataFrame) -> Optional[Signal]:
    if len(df6) < 80 or len(df15) < 30: return None
    close6 = df6["close"]; h6 = df6["high"]; l6 = df6["low"]
    adx6, pdi6, mdi6 = adx(h6,l6,close6,14); r6 = rsi(close6,14); _,_,hist6 = macd(close6)
    ema200_6 = ema(close6, 200)

    # Trend bias
    long_ok  = (close6 > ema200_6) & (hist6 > 0) & (adx6 >= 25) & (r6 >= 55)
    short_ok = (close6 < ema200_6) & (hist6 < 0) & (adx6 >= 25) & (r6 <= 45)

    # Last closed 6H bar
    t = df6["dt"].iloc[-2].to_pydatetime(); side = None
    if bool(long_ok.iloc[-2]): side = "LONG"
    elif bool(short_ok.iloc[-2]): side = "SHORT"
    if side is None: return None

    # Confirmation on 15m (time‑decaying)
    cts = confirm_time_15m(df15, side, t, CONFIRM_TIMEOUT_MIN)
    if not cts: return None

    # Entry/ATR from 1H context near sig_time
    if len(df1h) < 50: return None
    atr1h = atr(df1h["high"], df1h["low"], df1h["close"], 14)
    entry = float(df1h["close"].iloc[-1])
    atr_val = float(atr1h.iloc[-1])
    tp = entry + TP_ATR_MULT*atr_val if side=="LONG" else entry - TP_ATR_MULT*atr_val
    sl = entry - SL_ATR_MULT*atr_val if side=="LONG" else entry + SL_ATR_MULT*atr_val

    # Scoring bits from 1H
    _,_,hist = macd(df1h["close"]) ; hist_slope = float(hist.iloc[-1] - hist.iloc[-2])
    a1h,_,_ = adx(df1h["high"], df1h["low"], df1h["close"], 14)
    r1h = rsi(df1h["close"],14)
    mid,up,lo,bbw = bollinger(df1h["close"]) ; ema20_v = float(ema(df1h["close"],20).iloc[-1])
    # clearance not applicable here → set small value for score
    scr, parts = score_signal(side, float(a1h.iloc[-1]), float(a1h.iloc[-1]-a1h.iloc[-2]), float(hist.iloc[-1]), hist_slope,
                              float(r1h.iloc[-1]), ema20_v, entry, atr_val, 0.3)

    return Signal("", "STRUCTURE 6H", side, entry, float(tp), float(sl), t, atr_val,
                  float(a1h.iloc[-1]), float(r1h.iloc[-1]), float(hist.iloc[-1]), hist_slope,
                  float(bbw.iloc[-1]), ema20_v, 0.3, scr, parts)


def regime_signal_1h(df1h: pd.DataFrame) -> Optional[Signal]:
    if len(df1h) < 120: return None
    c = df1h["close"]; h = df1h["high"]; l = df1h["low"]; v = df1h["volume"]
    a,_,_ = adx(h,l,c,14); r = rsi(c,14); mid,up,lo,bbw = bollinger(c,20,2.0)
    m_line, m_sig, hist = macd(c)
    atr1 = atr(h,l,c,14)

    adx_last = float(a.iloc[-1]); adx_prev = float(a.iloc[-2]); adx_slope = adx_last - adx_prev
    hist_last = float(hist.iloc[-1]); hist_prev = float(hist.iloc[-2]); macd_slope = hist_last - hist_prev
    bbw_last = float(bbw.iloc[-1]); bbw_prev = float(bbw.iloc[-2])
    ema20_v = float(ema(c,20).iloc[-1])

    trending = (adx_last >= 25) and (bbw_last > bbw_prev)
    ranging  = (adx_last <= 20) and ((bbw.dropna().iloc[-100:].rank(pct=True).iloc[-1])*100.0 <= 25)

    entry = float(c.iloc[-1]); atr_val = float(atr1.iloc[-1])

    if trending:
        # Breakout with clearance + volume gating
        n = 20
        hi_n = float(h.rolling(n).max().shift(1).iloc[-1])
        lo_n = float(l.rolling(n).min().shift(1).iloc[-1])
        vol_sma = float(v.rolling(20).mean().iloc[-1])
        vol_ok = (not math.isnan(vol_sma)) and (float(v.iloc[-1]) >= 1.3*vol_sma)
        side = None; clearance = 0.0
        hi_buf = hi_n + 0.25*atr_val
        lo_buf = lo_n - 0.25*atr_val
        if entry > hi_buf and hist_last>0 and macd_slope>0 and float(r.iloc[-1])>=55 and adx_slope>0 and vol_ok:
            side = "LONG"; clearance = (entry - hi_n)/max(atr_val,1e-9)
        if entry < lo_buf and hist_last<0 and macd_slope<0 and float(r.iloc[-1])<=45 and adx_slope>0 and vol_ok:
            side = "SHORT"; clearance = (lo_n - entry)/max(atr_val,1e-9)
        if side is None:
            return None
        tp = entry + TP_ATR_MULT*atr_val if side=="LONG" else entry - TP_ATR_MULT*atr_val
        sl = entry - SL_ATR_MULT*atr_val if side=="LONG" else entry + SL_ATR_MULT*atr_val
        scr, parts = score_signal(side, adx_last, adx_slope, hist_last, macd_slope, float(r.iloc[-1]), ema20_v, entry, atr_val, clearance)
        return Signal("", "REGIME 1H (TREND)", side, entry, float(tp), float(sl), df1h["dt"].iloc[-1].to_pydatetime(),
                      atr_val, adx_last, float(r.iloc[-1]), hist_last, macd_slope, bbw_last, ema20_v, clearance, scr, parts)

    if ranging:
        # Mean-reversion near bands + RSI extreme; confirmation strengthened by optional whale require
        band_w = (up.iloc[-1] - lo.iloc[-1])
        near_upper = entry >= (up.iloc[-1] - 0.15*band_w)
        near_lower = entry <= (lo.iloc[-1] + 0.15*band_w)
        side = None
        if near_lower and float(r.iloc[-1]) <= 30: side = "LONG"
        if near_upper and float(r.iloc[-1]) >= 70: side = "SHORT"
        if side is None: return None
        tp = entry + TP_ATR_MULT*atr_val if side=="LONG" else entry - TP_ATR_MULT*atr_val
        sl = entry - SL_ATR_MULT*atr_val if side=="LONG" else entry + SL_ATR_MULT*atr_val
        scr, parts = score_signal(side, adx_last, adx_slope, hist_last, macd_slope, float(r.iloc[-1]), ema20_v, entry, atr_val, 0.0)
        return Signal("", "REGIME 1H (RANGE)", side, entry, float(tp), float(sl), df1h["dt"].iloc[-1].to_pydatetime(),
                      atr_val, adx_last, float(r.iloc[-1]), hist_last, macd_slope, bbw_last, ema20_v, 0.0, scr, parts)

    return None

# ============================
# ---- Telegram --------------
# ============================

def tg_send(text: str) -> None:
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("[TG]", text)
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": "Markdown", "disable_web_page_preview": True}
    try:
        requests.post(url, json=payload, timeout=15)
    except Exception:
        pass

# ============================
# ---- Logging & Cooldowns ---
# ============================
LOG_PATH = "live_trade_log.csv"

def init_log():
    try:
        with open(LOG_PATH, "x") as f:
            f.write("ts,symbol,module,side,entry,tp,sl,atr,score,adx,rsi,macd_hist,macd_slope,bbw,ema20,clearance_atr")
    except FileExistsError:
        pass

def log_trade(sym: str, sig: Signal):
    with open(LOG_PATH, "a") as f:
        f.write(
            f"{now_utc().isoformat()},{sym},{sig.module},{sig.side},{sig.entry},{sig.tp},{sig.sl},{sig.atr},{sig.score},{sig.adx},{sig.rsi},{sig.macd_hist},{sig.macd_slope},{sig.bbw},{sig.ema20},{sig.clearance_atr}" )

last_alert_at: Dict[str, dt.datetime] = {}

def on_cooldown(sym: str) -> bool:
    t = last_alert_at.get(sym)
    return t is not None and (now_utc()-t).total_seconds() < COOLDOWN_MINUTES*60

# ============================
# ---- Messaging --------------
# ============================

def message_for(sym: str, sig: Signal, whale_ok: Optional[bool]) -> str:
    parts = sig.parts
    whale_tag = (" | WHALE✅" if whale_ok else " | WHALE–") if whale_ok is not None else ""
    lines = []
    lines.append(f"*{sym}* — *{sig.side}* {whale_tag}")
    lines.append(f"{sig.module}  |  Score {sig.score}/100  (ADX:{parts.adx} Slope:{parts.slope} MACD:{parts.macd} RSI:{parts.rsi} Brk:{parts.clearance} EMA:{parts.ema_dist})")
    lines.append("")
    lines.append(f"`Entry` {fmt_price(sig.entry)}    `TP` {fmt_price(sig.tp)}    `SL` {fmt_price(sig.sl)}")
    lines.append(f"`ATR` {fmt_price(sig.atr)}    `ADX` {sig.adx:.1f}    `RSI` {sig.rsi:.1f}    `MACD` {sig.macd_hist:.5f}")
    lines.append("")
    lines.append(f"_Partial TP:_ {int(PARTIAL_TP_SHARE*100)}% at TP, then trail {TRAIL_ATR_AFTER_TP}×ATR.")
    return "\n".join(lines)

# ============================
# ---- Runner -----------------
# ============================

def run_once(whales: Optional[WhaleWatcher]):
    now = now_utc()
    start_6h = now - dt.timedelta(days=20)
    start_1h = now - dt.timedelta(days=10)
    start_15 = now - dt.timedelta(days=2)

    for sym in COINS:
        if on_cooldown(sym):
            continue
        try:
            df6  = fetch_candles_range(sym, 21600, start_6h, now)
            df1h = fetch_candles_range(sym, 3600,  start_1h, now)
            df15 = fetch_candles_range(sym, 900,   start_15, now)
            if df1h.empty: continue

            picked: Optional[Signal] = None

            if ENABLE_STRUCTURE_6H and not df6.empty and not df15.empty:
                s = structure_signal_6h(df6, df1h, df15)
                if s: picked = s

            if picked is None and ENABLE_REGIME_1H:
                s = regime_signal_1h(df1h)
                if s: picked = s

            if picked is None:
                continue

            picked.symbol = sym
            whale_ok = None
            if ENABLE_WHALE_FILTER and websocket is not None and whales is not None:
                whale_ok = whales.aligned(sym, picked.side, picked.entry)
                if REQUIRE_WHALE_FOR_MR and picked.module.endswith("RANGE") and not whale_ok:
                    continue  # skip weak range fades without whales

            # Send + log
            tg_send(message_for(sym, picked, whale_ok))
            log_trade(sym, picked)
            last_alert_at[sym] = now_utc()

        except Exception as e:
            tg_send(f"⚠️ {sym} error: {e}")
        time.sleep(0.3)


def main():
    init_log()
    tg_send("✅ Unified Crypto Alert Bot — hardened build online.")
    whales = None
    if ENABLE_WHALE_FILTER and websocket is not None:
        whales = WhaleWatcher(COINS, min_usd=100_000.0, near_price_pct=0.25, max_age_sec=600)
        whales.start()
    while True:
        run_once(whales)
        time.sleep(SCAN_INTERVAL_SECONDS)


if __name__ == "__main__":
    main()
