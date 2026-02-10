#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Dual-Mode Backtest (Per-coin TP1 ATR multiple + Per-coin TP2 for FIB)
Pass 1:
  - Run entries (Stoch strict) on EMA/FIB zones (no other filters)
  - Record per-trade MFE_R, ATR, risk
  - For each coin:
      * TP1_ATR_MULT[coin] = P40( theta ), where theta = mfe_R * risk / ATR
        (so ≈60% of that coin's trades would hit TP1)
      * TP2_R[coin] = P75(MFE_R) computed from FIB trades only
Pass 2:
  - EMA zones => SCALP: TP1 = ATR * TP1_ATR_MULT[coin], FLAT 100% at TP1
  - FIB zones (SHORT/MEDIUM/LONG) => REVERSAL: TP1 partial (50%) at ATR * TP1_ATR_MULT[coin], runner -> TP2_R[coin]
  - SL = zone boundary (± SL_BUFFER)
  - Stoch RSI strict (40/60) REQUIRED; no other filters
Outputs:
  - trades_dual_pass1.csv, summary_dual_pass1.csv
  - trades_dual_pass2.csv, summary_dual_pass2.csv
"""

import csv, time
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone

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
    "WIF-USD","MAGIC-USD","APE-USD","JASMY-USD","SYRUP-USD","FARTCOIN-USD",
    "AERO-USD","FET-USD","CRV-USD","TAO-USD","XCN-USD","UNI-USD","MKR-USD",
    "TOSHI-USD","TRUMP-USD","PEPE-USD","XLM-USD","MOODENG-USD","BONK-USD",
    "POPCAT-USD","QNT-USD","IP-USD","PNUT-USD","APT-USD","ENA-USD","TURBO-USD",
    "BERA-USD","MASK-USD","SAND-USD","MORPHO-USD","MANA-USD","C98-USD","AXS-USD"
]


# Backtest window (UTC)
START_DATE = "2025-02-01"
END_DATE   = "2025-08-25"

# Fib windows (bars)
SHORT_LOOKBACK_HOURS = 7 * 24    # 7d on 1H
MEDIUM_LOOKBACK_DAYS = 14        # 14d on 1D
LONG_LOOKBACK_DAYS   = 30        # 30d on 1D

# EMA band width (±)
EMA_BAND_PCT = 0.005             # 0.5%

# Stoch RSI strict (only filter in play)
STOCH_RSI_PERIOD = 14
STOCH_SMOOTH_K   = 3
STOCH_SMOOTH_D   = 3
LONG_K_MAX  = 40.0               # LONG: K>D & K<40
SHORT_K_MIN = 60.0               # SHORT: K<D & K>60

# Golden pocket
GOLDEN_MIN = 0.618
GOLDEN_MAX = 0.66
GOLDEN_TOL = 0.0025

# Risk / exits
SL_BUFFER       = 0.01           # SL at zone boundary ±1%
TP1_QTL_TARGET  = 0.40           # target ≈60% hit → TP1_ATR_MULT = P40(theta)
TP2_QTL_FIB     = 0.75           # per-coin P75(MFE_R) for Fib
TP1_PCT_FIB     = 0.5            # Fib: take 50% at TP1
TP1_MULT_BOUNDS = (0.3, 1.8)     # reasonable bounds for ATR multiple
DEFAULT_TP1_MULT= 1.0            # fallback if insufficient data
DEFAULT_TP2_R   = 1.5            # fallback TP2 (in R) if insufficient data

MAX_HOLD_HOURS  = 7 * 24
SLIPPAGE_PCT    = 0.0
FEES_PCT        = 0.0

# I/O
TRADES_P1  = "trades_dual_pass1.csv"
SUMMARY_P1 = "summary_dual_pass1.csv"
TRADES_P2  = "trades_dual_pass2.csv"
SUMMARY_P2 = "summary_dual_pass2.csv"

# Coinbase API & chunking
CB_BASE = "https://api.exchange.coinbase.com"
MAX_BARS_PER_CALL = 300
WARMUP_BARS = { 3600: 250, 21600: 80, 86400: 220 }  # warmup for indicators

# ================== TA / UTILS ==================
def rsi(series: pd.Series, period=14):
    d = series.diff()
    up = d.clip(lower=0.0); down = -d.clip(upper=0.0)
    ru = up.ewm(alpha=1/period, adjust=False).mean()
    rd = down.ewm(alpha=1/period, adjust=False).mean()
    rs = ru / (rd.replace(0,1e-10))
    return 100 - (100/(1+rs))

def stoch_rsi(close: pd.Series, period: int, k: int, d_: int):
    r = rsi(close, period)
    lo = r.rolling(period).min(); hi = r.rolling(period).max()
    st = (r - lo)/(hi - lo + 1e-10)*100.0
    K = st.rolling(k).mean(); D = K.rolling(d_).mean()
    return K, D

def ema(series: pd.Series, period: int): return series.ewm(span=period, adjust=False).mean()

def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14):
    prev_close = close.shift(1)
    tr = pd.concat([(high-low),(high-prev_close).abs(),(low-prev_close).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, adjust=False).mean()

def fib_golden_zone(lo: float, hi: float, tol: float = GOLDEN_TOL):
    span = hi - lo; f618 = lo+span*GOLDEN_MIN; f66 = lo+span*GOLDEN_MAX
    return f618*(1-tol), f66*(1+tol)

def detect_reversal(prev: float, curr: float, zmin: float, zmax: float):
    if zmin is None or zmax is None: return None
    if prev < zmin and zmin <= curr <= zmax: return "SHORT"
    if prev > zmax and zmin <= curr <= zmax: return "LONG"
    return None

# --------- Chunked fetching + warmup ---------
def fetch_candles_range(product_id: str, granularity: int,
                        start_dt: pd.Timestamp, end_dt: pd.Timestamp,
                        session=None) -> pd.DataFrame:
    sess = session or requests.Session()
    url = f"{CB_BASE}/products/{product_id}/candles"
    out=[]; step_secs=granularity*MAX_BARS_PER_CALL; curr_end=end_dt
    while curr_end > start_dt:
        curr_start = max(start_dt, curr_end - pd.Timedelta(seconds=step_secs))
        params={"granularity":granularity,"start":curr_start.isoformat(),"end":curr_end.isoformat()}
        r=sess.get(url, params=params, timeout=20); r.raise_for_status()
        data=r.json()
        if not isinstance(data,list) or not data: break
        df_chunk=pd.DataFrame(data, columns=["time","low","high","open","close","volume"])
        df_chunk["time"]=pd.to_datetime(df_chunk["time"], unit="s", utc=True)
        out.append(df_chunk)
        earliest=df_chunk["time"].min()
        if pd.isna(earliest): break
        curr_end=earliest - pd.Timedelta(seconds=1)
        time.sleep(0.05)
    if not out: raise ValueError(f"No data for {product_id}@{granularity}")
    df=pd.concat(out, ignore_index=True).drop_duplicates(subset=["time"]).sort_values("time")
    return df[(df["time"]>=start_dt) & (df["time"]<end_dt)].reset_index(drop=True)

def fetch_with_warmup(product_id: str, granularity: int, start_date: str, end_date: str) -> pd.DataFrame:
    s=pd.Timestamp(start_date, tz="UTC"); e=pd.Timestamp(end_date, tz="UTC")
    warm_bars=WARMUP_BARS.get(granularity,0); s_warm=s-pd.Timedelta(seconds=warm_bars*granularity)
    return fetch_candles_range(product_id, granularity, s_warm, e)

# ---------- ZONES (EMA + FIB SHORT/MED/LONG only) ----------
def build_zones(df_1h, df_1d, i):
    t = df_1h["time"].iloc[i]
    curr = float(df_1h["close"].iloc[i])
    prev = float(df_1h["close"].iloc[i-1])
    entries = []

    # FIB_SHORT (7d/1H)
    w = df_1h.iloc[:i+1].tail(SHORT_LOOKBACK_HOURS)
    sl_s, sh_s = float(w["low"].min()), float(w["high"].max())
    zmin_s, zmax_s = fib_golden_zone(sl_s, sh_s)
    side_s = detect_reversal(prev, curr, zmin_s, zmax_s)
    if side_s: entries.append(("FIB_SHORT", sl_s, sh_s, zmin_s, zmax_s, side_s))

    # FIB_MEDIUM (14d/1D) + FIB_LONG (30d/1D)
    df_1d_hist = df_1d[df_1d["time"] <= t]
    if len(df_1d_hist) >= MEDIUM_LOOKBACK_DAYS:
        wm = df_1d_hist.tail(MEDIUM_LOOKBACK_DAYS)
        sl_m, sh_m = float(wm["low"].min()), float(wm["high"].max())
        zmin_m, zmax_m = fib_golden_zone(sl_m, sh_m)
        side_m = detect_reversal(prev, curr, zmin_m, zmax_m)
        if side_m: entries.append(("FIB_MEDIUM", sl_m, sh_m, zmin_m, zmax_m, side_m))
    if len(df_1d_hist) >= LONG_LOOKBACK_DAYS:
        wl = df_1d_hist.tail(LONG_LOOKBACK_DAYS)
        sl_l, sh_l = float(wl["low"].min()), float(wl["high"].max())
        zmin_l, zmax_l = fib_golden_zone(sl_l, sh_l)
        side_l = detect_reversal(prev, curr, zmin_l, zmax_l)
        if side_l: entries.append(("FIB_LONG", sl_l, sh_l, zmin_l, zmax_l, side_l))

    # EMA 1H
    for period, name in [(50,"EMA50_1H"), (200,"EMA200_1H")]:
        if i+1 >= period:
            val = float(ema(df_1h["close"].iloc[:i+1], period).iloc[-1])
            zmin, zmax = val*(1-EMA_BAND_PCT), val*(1+EMA_BAND_PCT)
            side = detect_reversal(prev, curr, zmin, zmax)
            if side: entries.append((name, val, val, zmin, zmax, side))

    # EMA 1D
    def add_ema_1d(period, nm):
        if len(df_1d_hist) >= period:
            val = float(ema(df_1d_hist["close"], period).iloc[-1])
            zmin, zmax = val*(1-EMA_BAND_PCT), val*(1+EMA_BAND_PCT)
            side = detect_reversal(prev, curr, zmin, zmax)
            if side: entries.append((nm, val, val, zmin, zmax, side))
    add_ema_1d(20,"EMA20_1D"); add_ema_1d(50,"EMA50_1D"); add_ema_1d(100,"EMA100_1D"); add_ema_1d(200,"EMA200_1D")

    return entries

# ---------- PASS RUNNERS ----------
def run_pass(coin, df_1h, df_1d, pass_mode, tp1_mult_map=None, tp2_map=None):
    K, D = stoch_rsi(df_1h["close"], STOCH_RSI_PERIOD, STOCH_SMOOTH_K, STOCH_SMOOTH_D)
    atr_1h = atr(df_1h["high"], df_1h["low"], df_1h["close"], 14)

    trades=[]
    warmup = max(SHORT_LOOKBACK_HOURS, 250)

    # Zone priority: 1D EMAs > FIB long/med > 1H EMAs > FIB short
    priority = {
        "EMA200_1D":11,"EMA100_1D":10,"EMA50_1D":9,"EMA20_1D":8,
        "FIB_LONG":7,"FIB_MEDIUM":6,
        "EMA200_1H":4,"EMA50_1H":3,
        "FIB_SHORT":2
    }

    for i in range(warmup, len(df_1h)-1):
        entries = build_zones(df_1h, df_1d, i)
        if not entries: continue
        entries = [e for e in entries if e[5] is not None]
        if not entries: continue
        entries.sort(key=lambda x: priority.get(x[0],1), reverse=True)
        zone_source, swing_low, swing_high, zmin, zmax, side = entries[0]

        # Only Stoch strict gate (no trend/ADX/time filters)
        kv = float(K.iloc[i]) if not pd.isna(K.iloc[i]) else None
        dv = float(D.iloc[i]) if not pd.isna(D.iloc[i]) else None
        if kv is None or dv is None: continue
        stoch_ok = (kv>dv and kv<LONG_K_MAX) if side=="LONG" else (kv<dv and kv>SHORT_K_MIN)
        if not stoch_ok: continue

        curr = float(df_1h["close"].iloc[i])
        # Risk from zone boundary
        if side=="LONG":
            sl = zmin*(1-SL_BUFFER); risk = curr - sl
        else:
            sl = zmax*(1+SL_BUFFER); risk = sl - curr
        if risk <= 0: continue

        # --- TP1 per-coin ATR multiple ---
        atr_now = float(atr_1h.iloc[i]) if not pd.isna(atr_1h.iloc[i]) else 0.0
        tp1_mult = tp1_mult_map.get(coin, DEFAULT_TP1_MULT) if (tp1_mult_map is not None) else DEFAULT_TP1_MULT
        tp1 = curr + tp1_mult*atr_now if side=="LONG" else curr - tp1_mult*atr_now

        # --- TP2 for FIB trades (pass2 only) ---
        if pass_mode == "pass2" and zone_source.startswith("FIB"):
            tp2_R = tp2_map.get(coin, DEFAULT_TP2_R) if tp2_map is not None else DEFAULT_TP2_R
            tp2 = curr + tp2_R*risk if side=="LONG" else curr - tp2_R*risk
        else:
            tp2 = None; tp2_R = np.nan

        # Manage forward
        highest=curr; lowest=curr
        sl_work=sl; tp1_hit=False; tp2_hit=False; tp1_bar=-1; tp2_bar=-1
        exit_price=curr; outcome="Timeout"; hold_hours=0

        for j in range(i+1, len(df_1h)):
            bar=df_1h.iloc[j]; bh=float(bar["high"]); bl=float(bar["low"])
            if bh>highest: highest=bh
            if bl<lowest:  lowest=bl

            # TP1
            if not tp1_hit:
                if (side=="LONG" and bh>=tp1) or (side=="SHORT" and bl<=tp1):
                    tp1_hit=True; tp1_bar=j-i
                    if not zone_source.startswith("FIB") or pass_mode=="pass1":
                        # EMA scalp (any pass) OR pass1 (both types): flat all at TP1
                        exit_price=tp1; outcome="TP1"; hold_hours=j-i; break

            # FIB runner (pass2 only)
            if pass_mode=="pass2" and zone_source.startswith("FIB") and tp1_hit and not tp2_hit:
                if (side=="LONG" and bh>=tp2) or (side=="SHORT" and bl<=tp2):
                    tp2_hit=True; tp2_bar=j-i; exit_price=tp2; outcome="TP1_TP2"; hold_hours=j-i; break

            # SL
            if (side=="LONG" and bl<=sl_work) or (side=="SHORT" and bh>=sl_work):
                exit_price=sl_work; outcome=("TP1_BE" if (tp1_hit and zone_source.startswith("FIB")) else "SL"); hold_hours=j-i; break

            # Timeout
            if (j - i) >= MAX_HOLD_HOURS:
                exit_price=float(df_1h["close"].iloc[j]); outcome=("Timeout_after_TP1" if (tp1_hit and zone_source.startswith("FIB")) else "Timeout"); hold_hours=j-i; break

        # R calculator
        def R(px): return (px - curr)/risk if side=="LONG" else (curr - px)/risk

        netR = R(exit_price)

        # FIB partial in pass2
        if pass_mode=="pass2" and zone_source.startswith("FIB"):
            if tp1_hit:
                tp1_R = R(tp1)
                if tp2_hit:
                    partial_R = TP1_PCT_FIB*tp1_R + (1-TP1_PCT_FIB)*tp2_R
                else:
                    partial_R = TP1_PCT_FIB*tp1_R + (1-TP1_PCT_FIB)*netR
            else:
                partial_R = netR
            netR = partial_R

        # MFE/MAE for pass1 calibration
        mfe_R = R(highest) if side=="LONG" else R(lowest)
        mae_R = R(lowest)  if side=="LONG" else R(highest)

        trades.append({
            "coin": coin, "time": df_1h["time"].iloc[i].strftime("%Y-%m-%d %H:%M"),
            "side": side, "zone_source": zone_source,
            "entry": curr, "sl": sl, "tp1": tp1, "tp2_R_used": tp2_R,
            "outcome": outcome, "R_multiple": round(float(netR - FEES_PCT),4),
            "tp1_hit": bool(tp1_hit), "tp2_hit": bool(tp2_hit),
            "tp1_bar": tp1_bar, "tp2_bar": tp2_bar, "hold_hours": hold_hours,
            "mfe_R": round(float(mfe_R),4), "mae_R": round(float(mae_R),4),
            # extras for calibration
            "risk": risk, "atr_now": atr_now
        })

    return trades

# ------------- Save + Summary -------------
def save_trades_and_summary(trades, trades_path, summary_path, title=""):
    if not trades:
        print("No trades"); return
    df = pd.DataFrame(trades)
    df.to_csv(trades_path, index=False)

    def win_rate(s): return (s>0).mean() if len(s) else 0.0
    overall = pd.DataFrame([{
        "trades": len(df),
        "win_rate": win_rate(df["R_multiple"]),
        "TP1_rate": df["tp1_hit"].mean(),
        "TP2_rate": df["tp2_hit"].mean() if "tp2_hit" in df.columns else 0.0,
        "avg_R": df["R_multiple"].mean(),
        "median_R": df["R_multiple"].median(),
        "avg_hold_hours": df["hold_hours"].mean(),
        "section": "overall"
    }])

    by_zone = df.groupby("zone_source").agg(
        trades=("zone_source","count"),
        win_rate=("R_multiple", win_rate),
        TP1_rate=("tp1_hit","mean"),
        TP2_rate=("tp2_hit","mean"),
        avg_R=("R_multiple","mean"),
        median_R=("R_multiple","median"),
        avg_hold_hours=("hold_hours","mean")
    ).reset_index()
    by_zone["section"] = "by_zone"

    summary = pd.concat([overall, by_zone], ignore_index=True)
    summary.to_csv(summary_path, index=False)

    if title: print(title)
    print(f"Saved {len(df)} trades to {trades_path}")
    print(f"Saved summary to {summary_path}")
    print("\n=== Overall ==="); print(overall.to_string(index=False))
    print("\n=== By zone ==="); print(by_zone.sort_values("avg_R", ascending=False).to_string(index=False))

# ------------- Main (two-pass) -------------
def main():
    # Pass 1: run and gather MFE/ATR/risk to learn per-coin TP1 ATR multiples and per-coin TP2_R for FIB
    all_p1=[]
    for coin in COINS:
        try:
            print(f"[Pass1] {coin} ...")
            df_1h = fetch_with_warmup(coin, 3600,  START_DATE, END_DATE)
            df_1d = fetch_with_warmup(coin, 86400, START_DATE, END_DATE)
            all_p1.extend(run_pass(coin, df_1h, df_1d, pass_mode="pass1"))
        except Exception as e:
            print(f"[Pass1 ERR] {coin}: {e}")
    save_trades_and_summary(all_p1, TRADES_P1, SUMMARY_P1, title="== PASS 1 ==")

    tp1_mult_map = {}
    tp2_map = {}
    if all_p1:
        dfp1 = pd.DataFrame(all_p1)

        # --- per-coin TP1 ATR multiple: TP1_MULT = P40(theta); theta = mfe_R * risk / ATR ---
        for coin, g in dfp1.groupby("coin"):
            g = g.copy()
            g = g[(g["atr_now"]>0) & (g["risk"]>0) & g["mfe_R"].notna()]
            if len(g) >= 15:
                theta = (g["mfe_R"] * g["risk"]) / g["atr_now"]
                # bound and take quantile
                mult = float(theta.quantile(TP1_QTL_TARGET))
                lo, hi = TP1_MULT_BOUNDS
                mult = max(lo, min(hi, mult)) if np.isfinite(mult) else DEFAULT_TP1_MULT
                tp1_mult_map[coin] = mult
            else:
                tp1_mult_map[coin] = DEFAULT_TP1_MULT

        # --- per-coin TP2 for FIB trades only (P75 of MFE_R) ---
        fibp1 = dfp1[dfp1["zone_source"].str.startswith("FIB")]
        for coin, g in fibp1.groupby("coin"):
            g = g[g["mfe_R"].notna()]
            if len(g) >= 10:
                tp2_map[coin] = float(g["mfe_R"].quantile(TP2_QTL_FIB))
            else:
                tp2_map[coin] = DEFAULT_TP2_R

    print("Per-coin TP1 ATR multiples (P40 θ):", {k:round(v,3) for k,v in tp1_mult_map.items()})
    print("Per-coin TP2_R for FIB (P75 MFE_R):", {k:round(v,3) for k,v in tp2_map.items()})

    # Pass 2: apply per-coin TP1 (for both EMA & FIB) and per-coin TP2 (for FIB)
    all_p2=[]
    for coin in COINS:
        try:
            print(f"[Pass2] {coin} ...")
            df_1h = fetch_with_warmup(coin, 3600,  START_DATE, END_DATE)
            df_1d = fetch_with_warmup(coin, 86400, START_DATE, END_DATE)
            all_p2.extend(run_pass(coin, df_1h, df_1d, pass_mode="pass2",
                                   tp1_mult_map=tp1_mult_map, tp2_map=tp2_map))
        except Exception as e:
            print(f"[Pass2 ERR] {coin}: {e}")
    save_trades_and_summary(all_p2, TRADES_P2, SUMMARY_P2, title="== PASS 2 ==")

if __name__ == "__main__":
    main()
