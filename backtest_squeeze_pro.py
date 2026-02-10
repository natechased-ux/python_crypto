#!/usr/bin/env python3
"""
Backtest • TTM Squeeze (LazyBear) • Pro Triggers (small-cap 15m friendly)

What’s included
- k-of-3 quality: need 2 of 3 (DI spread, squeeze duration, tight BBW)
- Power path: ADX rising OR absolute-strong |val| OR relative-strong |val|
- Relative momentum: |val| percentile vs recent window
- Soft-ADX: slope informs score/power but doesn’t hard-block alone
- Scoring fully tunable from CLI (targets + weights)
- Optional: --disable-score-gate for diagnostics
- Per-symbol debug counters

Outputs
- <prefix>_trades.csv
- <prefix>_summary.csv
"""

import argparse
import csv
import datetime as dt
import math
import requests
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

# ===================== CLI =====================
def parse_args():
    p = argparse.ArgumentParser(description="Backtest Pro Squeeze rules on Coinbase data.")
    p.add_argument("--symbols", type=str, required=True, help="Comma-separated tickers (e.g., BTC-USD,ETH-USD)")
    p.add_argument("--granularity", type=int, required=True, choices=[900, 3600], help="900=15m, 3600=1H")
    p.add_argument("--start", type=str, required=True, help="YYYY-MM-DD (UTC)")
    p.add_argument("--end", type=str, required=True, help="YYYY-MM-DD (UTC)")
    p.add_argument("--cooldown-min", type=int, default=25)

    # Pro gates
    p.add_argument("--score-min", type=int, default=70)
    p.add_argument("--allow-fading-if-score-ge", type=int, default=85)
    p.add_argument("--bbw-lookback", type=int, default=200)
    p.add_argument("--bbw-pctl-max", type=int, default=30)
    p.add_argument("--adx-len", type=int, default=14)
    p.add_argument("--adx-slope-lookback", type=int, default=3)
    p.add_argument("--adx-slope-min", type=float, default=2.0)  # advisory
    p.add_argument("--di-spread-min", type=float, default=5.0)
    p.add_argument("--min-sqz-bars", type=int, default=4)
    p.add_argument("--release-window", type=int, default=2)
    p.add_argument("--abs-val-min", type=float, default=0.5)
    p.add_argument("--alt-val-strong", type=float, default=1.2)

    # Relative momentum
    p.add_argument("--val-lookback", type=int, default=200)
    p.add_argument("--val-pctl-min", type=float, default=80.0)

    # Risk
    p.add_argument("--atr-len", type=int, default=14)
    p.add_argument("--tp-mult", type=float, default=2.0)
    p.add_argument("--sl-mult", type=float, default=1.5)

    # SMI / squeeze
    p.add_argument("--bb-len", type=int, default=20)
    p.add_argument("--bb-mult", type=float, default=2.0)
    p.add_argument("--kc-len", type=int, default=20)
    p.add_argument("--kc-mult", type=float, default=1.5)
    p.add_argument("--use-true-range", action="store_true", default=True)

    # Score calibration (NEW)
    p.add_argument("--val-target", type=float, default=1.2)
    p.add_argument("--adx-target", type=float, default=25.0)
    p.add_argument("--w-val", type=float, default=0.45)
    p.add_argument("--w-adx", type=float, default=0.35)
    p.add_argument("--w-mom", type=float, default=0.10)
    p.add_argument("--w-fresh", type=float, default=0.07)
    p.add_argument("--w-di", type=float, default=0.03)

    # Diagnostics
    p.add_argument("--disable-score-gate", action="store_true", default=False)
    p.add_argument("--sl-first", action="store_true", default=True)
    p.add_argument("--out-prefix", type=str, default="backtest_pro")

    return p.parse_args()

# ===================== Math =====================
def sma(series: List[float], length: int) -> List[float]:
    out, q, s = [], [], 0.0
    for v in series:
        q.append(v); s += v
        if len(q) > length: s -= q.pop(0)
        out.append(s/length if len(q) == length else float('nan'))
    return out

def stdev(series: List[float], length: int) -> List[float]:
    out, window = [], []
    for v in series:
        window.append(v)
        if len(window) > length: window.pop(0)
        if len(window) == length:
            m = sum(window)/length
            var = sum((x-m)**2 for x in window)/length
            out.append(math.sqrt(var))
        else:
            out.append(float('nan'))
    return out

def highest(series: List[float], length: int) -> List[float]:
    out, window = [], []
    for v in series:
        window.append(v)
        if len(window) > length: window.pop(0)
        out.append(max(window) if len(window) == length else float('nan'))
    return out

def lowest(series: List[float], length: int) -> List[float]:
    out, window = [], []
    for v in series:
        window.append(v)
        if len(window) > length: window.pop(0)
        out.append(min(window) if len(window) == length else float('nan'))
    return out

def linreg_last(series: List[float], length: int) -> List[float]:
    out = []
    xs = list(range(length))
    sum_x = sum(xs); sum_x2 = sum(x*x for x in xs)
    denom = length*sum_x2 - sum_x*sum_x
    window = []
    for v in series:
        window.append(v)
        if len(window) > length: window.pop(0)
        if len(window) < length:
            out.append(float('nan')); continue
        sum_y = sum(window)
        sum_xy = sum(x*y for x,y in zip(xs, window))
        m = (length*sum_xy - sum_x*sum_y)/denom if denom != 0 else 0.0
        b = (sum_y - m*sum_x)/length
        out.append(m*(length-1)+b)
    return out

def true_range(highs: List[float], lows: List[float], closes: List[float]) -> List[float]:
    trs, prev_close = [], None
    for h,l,c in zip(highs,lows,closes):
        if prev_close is None:
            tr = h - l
        else:
            tr = max(h - l, abs(h - prev_close), abs(l - prev_close))
        trs.append(tr); prev_close = c
    return trs

def atr_sma(highs: List[float], lows: List[float], closes: List[float], length: int) -> List[float]:
    return sma(true_range(highs, lows, closes), length)

def adx_di(highs: List[float], lows: List[float], closes: List[float], length: int):
    n = len(closes)
    tr = [0.0]*n; plus_dm = [0.0]*n; minus_dm = [0.0]*n
    for i in range(n):
        h, l = highs[i], lows[i]
        if i == 0:
            tr[i] = h - l; plus_dm[i] = 0.0; minus_dm[i] = 0.0
        else:
            prev_close = closes[i-1]; prev_high = highs[i-1]; prev_low = lows[i-1]
            tr[i] = max(h - l, abs(h - prev_close), abs(l - prev_close))
            up = h - prev_high; dn = prev_low - l
            plus_dm[i] = up if (up > dn and up > 0) else 0.0
            minus_dm[i] = dn if (dn > up and dn > 0) else 0.0

    def wilder_ema(vals, L):
        out = [float('nan')]*n; alpha = 1.0/L; acc = None
        for i,v in enumerate(vals):
            if i < L:
                if i == L-1:
                    acc = sum(vals[:L])/L; out[i] = acc
            else:
                acc = acc + alpha*(v - acc); out[i] = acc
        return out

    tr_s = wilder_ema(tr, length)
    plus_dm_s = wilder_ema(plus_dm, length)
    minus_dm_s = wilder_ema(minus_dm, length)

    plus_di = [100*(p/t) if (not math.isnan(p) and not math.isnan(t) and t!=0) else float('nan')
               for p,t in zip(plus_dm_s, tr_s)]
    minus_di = [100*(m/t) if (not math.isnan(m) and not math.isnan(t) and t!=0) else float('nan')
                for m,t in zip(minus_dm_s, tr_s)]
    dx = [100*abs(p-m)/(p+m) if (not math.isnan(p) and not math.isnan(m) and (p+m)!=0) else float('nan')
          for p,m in zip(plus_di, minus_di)]
    adx_vals = wilder_ema(dx, length)
    return plus_di, minus_di, adx_vals

# ================= Coinbase fetch =================
def iso_to_unix(s: str) -> int:
    return int(dt.datetime.fromisoformat(s).replace(tzinfo=dt.timezone.utc).timestamp())

def fetch_candles_range(product: str, granularity: int, ts_start: int, ts_end: int, max_per_call: int = 300):
    product = product.upper()
    out = []
    step = granularity * max_per_call
    t0 = ts_start
    while t0 < ts_end:
        t1 = min(ts_end, t0 + step)
        url = f"https://api.exchange.coinbase.com/products/{product}/candles"
        params = {
            "start": dt.datetime.fromtimestamp(t0, dt.timezone.utc).isoformat(),
            "end":   dt.datetime.fromtimestamp(t1, dt.timezone.utc).isoformat(),
            "granularity": granularity,
        }
        r = requests.get(url, params=params, timeout=20)
        if r.status_code == 429:
            time.sleep(1.5); continue
        if r.status_code == 404:
            return []
        r.raise_for_status()
        data = r.json()
        if not isinstance(data, list):
            raise RuntimeError(f"Unexpected response for {product}: {data}")
        out.extend(data)
        t0 = t1
        time.sleep(0.12)
    out.sort(key=lambda x: x[0])
    # de-dup
    seen=set(); ordered=[]
    for row in out:
        ts=row[0]
        if ts in seen: continue
        seen.add(ts); ordered.append(row)
    return ordered

# ================== Indicator Pack ==================
def compute_pack(candles, bb_len, bb_mult, kc_len, kc_mult, use_true_range, adx_len, atr_len):
    times  = [c[0] for c in candles]
    lows   = [float(c[1]) for c in candles]
    highs  = [float(c[2]) for c in candles]
    closes = [float(c[4]) for c in candles]

    source = closes
    basis = sma(source, bb_len)
    devs  = [bb_mult*d if not math.isnan(d) else float('nan') for d in stdev(source, bb_len)]
    upperBB = [b + d if not (math.isnan(b) or math.isnan(d)) else float('nan') for b,d in zip(basis, devs)]
    lowerBB = [b - d if not (math.isnan(b) or math.isnan(d)) else float('nan') for b,d in zip(basis, devs)]

    ma = sma(source, kc_len)
    rng = true_range(highs, lows, closes) if use_true_range else [h-l for h,l in zip(highs, lows)]
    rangema = sma(rng, kc_len)
    upperKC = [m + kc_mult*rm if not (math.isnan(m) or math.isnan(rm)) else float('nan') for m,rm in zip(ma, rangema)]
    lowerKC = [m - kc_mult*rm if not (math.isnan(m) or math.isnan(rm)) else float('nan') for m,rm in zip(ma, rangema)]

    sqzOn, sqzOff = [], []
    for lb,ub,lkc,ukc in zip(lowerBB, upperBB, lowerKC, upperKC):
        if any(math.isnan(x) for x in (lb, ub, lkc, ukc)):
            sqzOn.append(False); sqzOff.append(False)
        else:
            on  = (lb > lkc) and (ub < ukc)
            off = (lb < lkc) and (ub > ukc)
            sqzOn.append(on); sqzOff.append(off)

    hh = highest(highs, kc_len)
    ll = lowest(lows, kc_len)
    mid_hl = [(a+b)/2.0 if not (math.isnan(a) or math.isnan(b)) else float('nan') for a,b in zip(hh, ll)]
    sma_close_kc = sma(closes, kc_len)
    inner_avg = [(a+b)/2.0 if not (math.isnan(a) or math.isnan(b)) else float('nan') for a,b in zip(mid_hl, sma_close_kc)]
    detrended = [s - ia if not math.isnan(ia) else float('nan') for s, ia in zip(source, inner_avg)]
    detrended_clean = [x if not math.isnan(x) else 0.0 for x in detrended]
    val = linreg_last(detrended_clean, kc_len)

    atr_vals = atr_sma(highs, lows, closes, atr_len)
    di_plus, di_minus, adx_vals = adx_di(highs, lows, closes, adx_len)

    bbw = []
    for u,l,b in zip(upperBB, lowerBB, basis):
        if any(math.isnan(x) for x in (u,l,b)) or b == 0:
            bbw.append(float('nan'))
        else:
            bbw.append((u - l) / abs(b))

    pack = {
        "times": times, "lows": lows, "highs": highs, "closes": closes,
        "sqzOn": sqzOn, "sqzOff": sqzOff,
        "val": val, "atr": atr_vals,
        "di_plus": di_plus, "di_minus": di_minus, "adx": adx_vals,
        "bbw": bbw, "basis": basis,
    }
    return pack

# =============== Helpers & Scoring ===============
def momentum_state(curr, prev):
    if math.isnan(curr) or math.isnan(prev): return "n/a"
    if curr > 0 and curr > prev: return "LIME"
    if curr > 0 and curr <= prev: return "GREEN"
    if curr < 0 and curr < prev: return "RED"
    if curr < 0 and curr >= prev: return "MAROON"
    return "n/a"

def percentile_rank(window_vals: List[float], target: float) -> float:
    vals = [v for v in window_vals if not math.isnan(v)]
    if not vals: return float('nan')
    less = sum(1 for v in vals if v <= target)
    return 100.0 * less / len(vals)

def di_spread(dip, dim):
    if math.isnan(dip) or math.isnan(dim): return float('nan')
    return abs(dip - dim)

def compute_entry_score(val_now, val_prev, adx_now, dip, dim, strict_release, side,
                        val_target=1.2, adx_target=25.0,
                        w_val=0.45, w_adx=0.35, w_mom=0.10, w_fresh=0.07, w_di=0.03):
    ms = momentum_state(val_now, val_prev)
    val_comp = min(abs(val_now)/val_target, 1.0) if not math.isnan(val_now) else 0.0
    adx_comp = min((adx_now or 0.0)/adx_target, 1.0) if not math.isnan(adx_now) else 0.0
    mom_comp = 1.0 if (ms in ("LIME","RED")) else (0.4 if (ms in ("GREEN","MAROON")) else 0.0)
    fresh_comp = 1.0 if strict_release else 0.5
    di_comp = 0.0
    if side == "long" and not math.isnan(dip) and not math.isnan(dim) and dip > dim: di_comp = 1.0
    if side == "short" and not math.isnan(dip) and not math.isnan(dim) and dim > dip: di_comp = 1.0
    score01 = (w_val*val_comp + w_adx*adx_comp + w_mom*mom_comp + w_fresh*fresh_comp + w_di*di_comp)
    return max(0.0, min(1.0, score01))

# =============== Backtest Engine ===============
@dataclass
class Trade:
    symbol: str
    granularity: int
    entry_ts: int
    exit_ts: int
    side: str
    entry: float
    exit: float
    reason: str
    bars_held: int
    r_multiple: float
    score: int
    grade: str
    val: float
    adx: float
    di_spread: float
    bbw_pct: float
    sqz_dur: int

def backtest_symbol(symbol: str, args) -> List[Trade]:
    ts_start = iso_to_unix(args.start)
    ts_end   = iso_to_unix(args.end)

    candles = fetch_candles_range(symbol, args.granularity, ts_start, ts_end)
    if not candles:
        print(f"[{symbol}] no data; skipping")
        return []

    P = compute_pack(candles, args.bb_len, args.bb_mult, args.kc_len, args.kc_mult,
                     args.use_true_range, args.adx_len, args.atr_len)

    n = len(P["times"])
    trades: List[Trade] = []
    last_entry_ts = 0
    in_pos = False; pos_side = None
    entry_price = 0.0; sl = float('nan'); tp = float('nan')
    entry_idx = -1; last_state = "n/a"

    # Debug counters
    stats = {"releases":0,"bbw_ok":0,"di_ok":0,"adx_rise_ok":0,"dur_ok":0,"base_ok":0,"score_ok":0,"rel_val_ok":0}
    blocks = {"bbw":0,"bbw_nan":0,"di":0,"adx_rise":0,"dur":0,"score":0,"power":0}

    for i in range(2, n):
        val_now = P["val"][i]; val_prev = P["val"][i-1]
        ms = momentum_state(val_now, val_prev)
        adx_i = P["adx"][i]; dip_i = P["di_plus"][i]; dim_i = P["di_minus"][i]
        di_gap = di_spread(dip_i, dim_i)

        # BBW percentile
        start_bbw = max(0, i - args.bbw_lookback + 1)
        bbw_vals = P["bbw"][start_bbw:i+1]
        bbw_pctl = percentile_rank(bbw_vals, P["bbw"][i])

        # Relative |val| percentile
        start_val = max(0, i - args.val_lookback + 1)
        abs_val_window = [abs(x) for x in P["val"][start_val:i+1]]
        val_abs_pctl = percentile_rank(abs_val_window, abs(val_now))
        rel_val_ok = (not math.isnan(val_abs_pctl)) and (val_abs_pctl >= args.val_pctl_min)
        if rel_val_ok: stats["rel_val_ok"] += 1

        # Release logic
        strict_rel_up = P["sqzOff"][i] and P["sqzOn"][i-1] and val_now > 0
        strict_rel_dn = P["sqzOff"][i] and P["sqzOn"][i-1] and val_now < 0

        def released_within_window(idx) -> Tuple[bool,int]:
            k=None
            for j in range(idx-1, max(idx-20,0), -1):
                if P["sqzOn"][j]: k=j; break
            if k is None: return False,0
            dur=1; t=k-1
            while t>=0 and P["sqzOn"][t]: dur+=1; t-=1
            for t in range(k+1, min(k+1+args.release_window, idx+1)):
                if t<n and P["sqzOff"][t]: return True, dur
            return False, dur

        rel_up_win, dur_up = released_within_window(i)
        rel_dn_win, dur_dn = released_within_window(i)
        is_rel_up   = strict_rel_up or (rel_up_win and val_now > 0)
        is_rel_down = strict_rel_dn or (rel_dn_win and val_now < 0)
        sqz_dur = dur_up if is_rel_up else (dur_dn if is_rel_down else 0)
        if is_rel_up or is_rel_down: stats["releases"] += 1

        # Cooldown
        if (P["times"][i] - last_entry_ts) < args.cooldown_min*60:
            pass
        else:
            # Power (soft-ADX)
            val_strong = abs(val_now) >= args.alt_val_strong
            adx_rise = (not math.isnan(adx_i)) and (i-args.adx_slope_lookback >= 0) and \
                       ((P["adx"][i] - P["adx"][i-args.adx_slope_lookback]) >= args.adx_slope_min)
            if adx_rise: stats["adx_rise_ok"] += 1

            # Quality (k-of-3)
            di_ok  = (not math.isnan(di_gap)) and (di_gap >= args.di_spread_min)
            dur_ok = sqz_dur >= args.min_sqz_bars
            bbw_ok = (not math.isnan(bbw_pctl)) and (bbw_pctl <= args.bbw_pctl_max)

            if bbw_ok: stats["bbw_ok"] += 1
            else:
                if math.isnan(bbw_pctl): blocks["bbw_nan"] += 1
                else: blocks["bbw"] += 1
            if di_ok: stats["di_ok"] += 1
            else: blocks["di"] += 1
            if not adx_rise: blocks["adx_rise"] += 1
            if dur_ok: stats["dur_ok"] += 1
            else: blocks["dur"] += 1

            power_ok = adx_rise or val_strong or rel_val_ok
            if not power_ok: blocks["power"] += 1

            quality_hits = (1 if di_ok else 0) + (1 if dur_ok else 0) + (1 if bbw_ok else 0)
            quality_ok = quality_hits >= 2

            base_ok = power_ok and quality_ok
            if base_ok: stats["base_ok"] += 1

            # Attempt entries
            if not in_pos and base_ok:
                if is_rel_up and val_now > 0:
                    side = "long"; strict_flag = strict_rel_up
                elif is_rel_down and val_now < 0:
                    side = "short"; strict_flag = strict_rel_dn
                else:
                    side = None

                if side is not None:
                    score01 = compute_entry_score(
                        val_now, val_prev, adx_i, dip_i, dim_i, strict_flag, side,
                        val_target=args.val_target, adx_target=args.adx_target,
                        w_val=args.w_val, w_adx=args.w_adx, w_mom=args.w_mom,
                        w_fresh=args.w_fresh, w_di=args.w_di
                    )
                    score_pct = int(round(score01*100))
                    def momentum_preference_ok(sp:int)->bool:
                        if ms in ("LIME","RED"): return True
                        return sp >= args.allow_fading_if_score_ge

                    if score_pct >= args.score_min: stats["score_ok"] += 1
                    else: blocks["score"] += 1

                    if (args.disable_score_gate or score_pct >= args.score_min) and momentum_preference_ok(score_pct):
                        # open
                        in_pos=True; pos_side=side; entry_idx=i; last_entry_ts=P["times"][i]; last_state=ms
                        entry_price = P["closes"][i]
                        atr_i = P["atr"][i]
                        if not math.isnan(atr_i) and atr_i>0:
                            if side=="long":
                                tp = entry_price + args.tp_mult*atr_i
                                sl = entry_price - args.sl_mult*atr_i
                            else:
                                tp = entry_price - args.tp_mult*atr_i
                                sl = entry_price + args.sl_mult*atr_i
                        else:
                            tp = sl = float('nan')
                        open_meta = (score_pct, "A" if score01>=0.8 else "B" if score01>=0.65 else "C" if score01>=0.5 else "D",
                                     val_now, adx_i, di_gap, bbw_pctl, sqz_dur)

        # Manage exits
        if in_pos:
            lo = P["lows"][i]; hi = P["highs"][i]
            exit_now=False; reason=""; px_exit=P["closes"][i]

            if pos_side=="long":
                if not math.isnan(sl) and not math.isnan(tp):
                    if args.sl_first:
                        if lo<=sl: exit_now=True; reason="SL"; px_exit=sl
                        elif hi>=tp: exit_now=True; reason="TP"; px_exit=tp
                    else:
                        if hi>=tp: exit_now=True; reason="TP"; px_exit=tp
                        elif lo<=sl: exit_now=True; reason="SL"; px_exit=sl
                if not exit_now:
                    if last_state=="LIME" and ms=="GREEN": exit_now=True; reason="ColorFlip"
                    elif val_now<0: exit_now=True; reason="ZeroCross"
            else:
                if not math.isnan(sl) and not math.isnan(tp):
                    if args.sl_first:
                        if hi>=sl: exit_now=True; reason="SL"; px_exit=sl
                        elif lo<=tp: exit_now=True; reason="TP"; px_exit=tp
                    else:
                        if lo<=tp: exit_now=True; reason="TP"; px_exit=tp
                        elif hi>=sl: exit_now=True; reason="SL"; px_exit=sl
                if not exit_now:
                    if last_state=="RED" and ms=="MAROON": exit_now=True; reason="ColorFlip"
                    elif val_now>0: exit_now=True; reason="ZeroCross"

            if exit_now:
                risk = (entry_price - sl) if pos_side=="long" else (sl - entry_price)
                r_mult = float('nan') if (risk<=0 or math.isnan(risk)) else \
                         ((px_exit - entry_price)/risk if pos_side=="long" else (entry_price - px_exit)/risk)
                score_pct, grade, v0, adx0, di0, bbw0, dur0 = open_meta
                trades.append(Trade(
                    symbol=symbol, granularity=args.granularity,
                    entry_ts=P["times"][entry_idx], exit_ts=P["times"][i],
                    side=pos_side, entry=entry_price, exit=px_exit, reason=reason,
                    bars_held=(i-entry_idx), r_multiple=r_mult, score=score_pct, grade=grade,
                    val=v0, adx=adx0, di_spread=di0, bbw_pct=bbw0, sqz_dur=dur0
                ))
                in_pos=False; pos_side=None; entry_idx=-1
            else:
                if ms!="n/a": last_state=ms

    # Debug summary
    print(f"[{symbol}] releases={stats['releases']} "
          f"bbw_ok={stats['bbw_ok']} di_ok={stats['di_ok']} adx_rise_ok={stats['adx_rise_ok']} "
          f"dur_ok={stats['dur_ok']} base_ok={stats['base_ok']} score_ok={stats['score_ok']} "
          f"rel_val_ok={stats['rel_val_ok']} trades={len(trades)} "
          f"blocks: bbw={blocks['bbw']} bbw_nan={blocks['bbw_nan']} di={blocks['di']} "
          f"adx_rise={blocks['adx_rise']} dur={blocks['dur']} score={blocks['score']} power={blocks['power']}")
    return trades

def write_trades_csv(path: str, trades: List[Trade]):
    header = ["symbol","granularity","entry_ts","exit_ts","side","entry","exit","reason",
              "bars_held","r_multiple","score","grade","val","adx","di_spread","bbw_pct","sqz_dur"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(header)
        for t in trades:
            w.writerow([t.symbol, t.granularity, t.entry_ts, t.exit_ts, t.side,
                        f"{t.entry:.8f}", f"{t.exit:.8f}", t.reason, t.bars_held,
                        f"{t.r_multiple:.4f}" if not math.isnan(t.r_multiple) else "nan",
                        t.score, t.grade, f"{t.val:.4f}", f"{t.adx:.2f}",
                        f"{t.di_spread:.2f}", f"{t.bbw_pct:.2f}", t.sqz_dur])

def summarize(trades: List[Trade]) -> Dict[str, float]:
    if not trades: return {"trades": 0}
    valid = [t for t in trades if not math.isnan(t.r_multiple)]
    wins = [t for t in valid if t.r_multiple > 0]
    avg_r = (sum(t.r_multiple for t in valid)/len(valid)) if valid else float('nan')
    winrate = 100.0*len(wins)/len(valid) if valid else 0.0
    return {
        "trades": len(trades),
        "winrate_pct": winrate,
        "avg_R": avg_r,
        "avg_bars": (sum(t.bars_held for t in trades)/len(trades)) if trades else 0.0,
        "tp_count": sum(1 for t in trades if t.reason=="TP"),
        "sl_count": sum(1 for t in trades if t.reason=="SL"),
        "flip_exits": sum(1 for t in trades if t.reason=="ColorFlip"),
        "zerox_exits": sum(1 for t in trades if t.reason=="ZeroCross"),
    }

def write_summary_csv(path: str, trades: List[Trade]):
    s = summarize(trades)
    keys = list(s.keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(keys); w.writerow([s[k] for k in keys])

def main():
    args = parse_args()
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    all_trades: List[Trade] = []
    for sym in symbols:
        try:
            t0 = time.time()
            tlist = backtest_symbol(sym, args)
            all_trades.extend(tlist)
            print(f"[{sym}] trades={len(tlist)} in {time.time()-t0:.1f}s")
        except Exception as e:
            print(f"[{sym}] error: {e}")

    out_trades = f"{args.out_prefix}_trades.csv"
    out_summary = f"{args.out_prefix}_summary.csv"
    write_trades_csv(out_trades, all_trades)
    write_summary_csv(out_summary, all_trades)
    print(f"Wrote trades: {out_trades}")
    print(f"Wrote summary: {out_summary}")

if __name__ == "__main__":
    main()
