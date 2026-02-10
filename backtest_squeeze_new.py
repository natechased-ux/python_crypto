#!/usr/bin/env python3
"""
Squeeze Backtester (enhanced + configurable scale-out) — multi-symbol, Coinbase, 15m/1H

Core:
- Squeeze ON: BB inside KC
- Entry: first bar AFTER a squeeze of at least `--min-sqz-bars`, within `--release-window`
- Direction: momentum = {lb, slope, roc}
- Scale-out exits (configurable):
  * SL = -1R
  * Take PARTIAL at +1R: --partial-size (default 0.4) → partial_size * 1R
  * Move stop on remainder to BE
  * Final TP on remainder at --tp-mult R (default 1.8R)
  * Total on full win = partial_size*1 + (1-partial_size)*tp_mult

Filters (CLI): BBW cap (percentile), allowed UTC hours/days, exclude ATR bottom quartile
Logs per trade: atr_entry, bbw_pct_entry, sqz_dur_entry, release_lag, entry_hour_utc, entry_dow_utc, mfe_r, mae_r
Outputs: <prefix>_trades.csv, <prefix>_summary.csv
"""

import argparse, csv, datetime as dt, math, time, requests
from dataclasses import dataclass
from typing import List, Dict

# ----------------------- CLI -----------------------
def parse_args():
    p = argparse.ArgumentParser(description="Clean Squeeze backtester on Coinbase candles.")
    p.add_argument("--symbols", required=True, type=str)
    p.add_argument("--granularity", required=True, type=int, choices=[900, 3600])
    p.add_argument("--start", required=True, type=str)
    p.add_argument("--end", required=True, type=str)

    # Squeeze params
    p.add_argument("--bb-len", type=int, default=20)
    p.add_argument("--bb-mult", type=float, default=2.0)
    p.add_argument("--kc-len", type=int, default=20)
    p.add_argument("--kc-mult", type=float, default=1.5)
    p.add_argument("--kc-use-atr", action="store_true", default=True)

    p.add_argument("--min-sqz-bars", type=int, default=3)
    p.add_argument("--release-window", type=int, default=1)

    # Momentum & risk
    p.add_argument("--momentum", choices=["lb", "slope", "roc"], default="lb")
    p.add_argument("--mom-len", type=int, default=20)
    p.add_argument("--atr-len", type=int, default=14)

    # SCALE-OUT exits (configurable)
    p.add_argument("--partial-size", type=float, default=0.4, help="Fraction to close at +1R (0..1).")
    p.add_argument("--tp-mult", type=float, default=1.8, help="Final-leg target in R for remainder.")
    p.add_argument("--sl-mult", type=float, default=1.0, help="SL in R")
    p.add_argument("--sl-first", action="store_true", default=True, help="When thresholds touch in same bar, SL/BE first.")

    # Trade flow
    p.add_argument("--cooldown-bars", type=int, default=0)
    p.add_argument("--time-stop-bars", type=int, default=0)

    # Filters & logging
    p.add_argument("--log-bbw-lookback", type=int, default=200)
    p.add_argument("--bbw-max-pct", type=float, default=100.0)
    p.add_argument("--hours-allow", type=str, default="")
    p.add_argument("--dows-allow", type=str, default="")
    p.add_argument("--exclude-atr-bottom-quartile", action="store_true", default=False)

    p.add_argument("--out-prefix", type=str, default="squeeze_bt")
    return p.parse_args()

# ----------------------- Math utils -----------------------
def sma(x: List[float], n: int) -> List[float]:
    out, q, s = [], [], 0.0
    for v in x:
        q.append(v); s += v
        if len(q) > n: s -= q.pop(0)
        out.append(s/n if len(q)==n else float("nan"))
    return out

def stdev(x: List[float], n: int) -> List[float]:
    out, q = [], []
    for v in x:
        q.append(v)
        if len(q) > n: q.pop(0)
        if len(q) == n:
            m = sum(q)/n
            var = sum((y-m)**2 for y in q)/n
            out.append(math.sqrt(var))
        else: out.append(float("nan"))
    return out

def linreg_slope(series: List[float], n: int) -> List[float]:
    out, xs = [], list(range(n))
    sx, sx2 = sum(xs), sum(x*x for x in xs)
    denom = n*sx2 - sx*sx
    if denom==0: return [float("nan")]*len(series)
    win=[]
    for v in series:
        win.append(v)
        if len(win)>n: win.pop(0)
        if len(win)<n: out.append(float("nan")); continue
        sy=sum(win); sxy=sum(x*y for x,y in zip(xs,win))
        m=(n*sxy - sx*sy)/denom
        out.append(m)
    return out

def true_range(h,l,c): 
    out=[]; prev_c=None
    for hi,lo,cl in zip(h,l,c):
        tr = (hi-lo) if prev_c is None else max(hi-lo, abs(hi-prev_c), abs(lo-prev_c))
        out.append(tr); prev_c=cl
    return out

def atr_sma(h,l,c,n): return sma(true_range(h,l,c),n)

def percentile_rank(vals: List[float], target: float) -> float:
    nums=[v for v in vals if not math.isnan(v)]
    if not nums: return float("nan")
    less=sum(1 for v in nums if v<=target)
    return 100.0*less/len(nums)

# ----------------------- Indicators -----------------------
def bollinger_bands(src,n,mult):
    mid=sma(src,n); sd=stdev(src,n)
    up,lo=[],[]
    for m,s in zip(mid,sd):
        if math.isnan(m) or math.isnan(s): up.append(float("nan")); lo.append(float("nan"))
        else: up.append(m+mult*s); lo.append(m-mult*s)
    return up,mid,lo

def keltner_channels_close(h,l,c,n,mult,use_atr=True):
    mid=sma(c,n); band=atr_sma(h,l,c,n) if use_atr else sma([hi-lo for hi,lo in zip(h,l)],n)
    up=[m+mult*b if not (math.isnan(m) or math.isnan(b)) else float("nan") for m,b in zip(mid,band)]
    lo=[m-mult*b if not (math.isnan(m) or math.isnan(b)) else float("nan") for m,b in zip(mid,band)]
    return up,mid,lo

def lb_momentum(h,l,c,n):
    mid_hl=[(hi+lo)/2 for hi,lo in zip(h,l)]
    avg_inner=[(mh+sc)/2 if not (math.isnan(mh) or math.isnan(sc)) else float("nan") for mh,sc in zip(mid_hl, sma(c,n))]
    detr=[(ci-ai) if not math.isnan(ai) else float("nan") for ci,ai in zip(c,avg_inner)]
    detr_clean=[0.0 if math.isnan(x) else x for x in detr]
    slope=linreg_slope(detr_clean,n)
    return [s*n if not math.isnan(s) else float("nan") for s in slope]

# ----------------------- Data -----------------------
def iso_to_unix(s:str)->int:
    return int(dt.datetime.fromisoformat(s).replace(tzinfo=dt.timezone.utc).timestamp())

def fetch_candles(product,granularity,ts_start,ts_end,max_per_call=300):
    product=product.upper(); out=[]; step=granularity*max_per_call; t0=ts_start
    while t0<ts_end:
        t1=min(ts_end,t0+step)
        url=f"https://api.exchange.coinbase.com/products/{product}/candles"
        params={"start":dt.datetime.fromtimestamp(t0,dt.timezone.utc).isoformat(),
                "end":dt.datetime.fromtimestamp(t1,dt.timezone.utc).isoformat(),
                "granularity":granularity}
        r=requests.get(url,params=params,timeout=20)
        if r.status_code==429: time.sleep(1.0); continue
        if r.status_code==404: return []
        r.raise_for_status(); data=r.json()
        if not isinstance(data,list): raise RuntimeError(f"Unexpected {data}")
        out.extend(data); t0=t1; time.sleep(0.05)
    out.sort(key=lambda x:x[0])
    seen=set(); ordered=[]
    for row in out:
        ts=row[0]
        if ts in seen: continue
        seen.add(ts); ordered.append(row)
    return ordered

# ----------------------- Core backtest -----------------------
@dataclass
class Trade:
    symbol:str; granularity:int
    entry_ts:int; entry:float; side:str
    exit_ts:int; exit:float; reason:str
    bars_held:int; r_multiple:float
    atr_entry:float; bbw_pct_entry:float
    sqz_dur_entry:int; release_lag:int
    entry_hour_utc:int; entry_dow_utc:int
    mfe_r:float; mae_r:float

def run_symbol(sym,args)->List[Trade]:
    ts_start=iso_to_unix(args.start); ts_end=iso_to_unix(args.end)
    raw=fetch_candles(sym,args.granularity,ts_start,ts_end)
    if not raw: print(f"[{sym}] no data"); return []
    t=[r[0] for r in raw]; lo=[float(r[1]) for r in raw]; hi=[float(r[2]) for r in raw]; cl=[float(r[4]) for r in raw]
    bb_u,bb_m,bb_l=bollinger_bands(cl,args.bb_len,args.bb_mult)
    kc_u,kc_m,kc_l=keltner_channels_close(hi,lo,cl,args.kc_len,args.kc_mult,use_atr=args.kc_use_atr)
    atr=atr_sma(hi,lo,cl,args.atr_len)

    sqz_on=[(bu<ku and bl>kl) if not any(math.isnan(x) for x in (bu,bl,ku,kl)) else False
            for bu,bl,ku,kl in zip(bb_u,bb_l,kc_u,kc_l)]
    bbw=[(bu-bl)/abs(m) if not any(math.isnan(x) for x in (bu,bl,m)) and m!=0 else float("nan")
         for bu,bl,m in zip(bb_u,bb_l,bb_m)]

    mom = lb_momentum(hi,lo,cl,args.mom_len) if args.momentum=="lb" else \
          linreg_slope(cl,args.mom_len) if args.momentum=="slope" else \
          [(cl[i]-cl[i-args.mom_len]) if i-args.mom_len>=0 else float("nan") for i in range(len(cl))]

    # ATR Q1 (per symbol) for low-ATR filtering
    atr_valid=[a for a in atr if not math.isnan(a)]
    atr_q1=None
    if atr_valid and args.exclude_atr_bottom_quartile:
        atr_sorted=sorted(atr_valid); q1_idx=int(0.25*(len(atr_sorted)-1)); atr_q1=atr_sorted[q1_idx]

    hours_allow = set(int(h) for h in args.hours_allow.split(",") if h.strip().isdigit()) if args.hours_allow else None
    dows_allow  = set(int(d) for d in args.dows_allow.split(",")  if d.strip().isdigit()) if args.dows_allow  else None

    trades=[]; in_pos=False; pos_side=None
    entry_price=0.0; entry_idx=-1; sl=float("nan"); tp2=float("nan")
    last_entry_idx=-10_000; open_meta={}
    has_partial=False; r1_price=float("nan"); be_price=float("nan")

    n=len(t); i0=max(args.bb_len,args.kc_len,args.mom_len,args.atr_len)+2
    i=i0
    while i<n:
        # ---------- manage exits ----------
        if in_pos:
            lo_i,hi_i=lo[i],hi[i]; px_exit=cl[i]; exit_now=False; reason=""
            risk = (entry_price-sl) if pos_side=="long" else (sl-entry_price)

            if risk>0 and not math.isnan(risk):
                if not has_partial:
                    r1_price = entry_price + ( risk if pos_side=="long" else -risk )
                    tp2      = entry_price + ( args.tp_mult*risk if pos_side=="long" else -args.tp_mult*risk )
                    be_price = entry_price

                if not has_partial:
                    if args.sl_first:
                        if (pos_side=="long" and lo_i<=sl) or (pos_side=="short" and hi_i>=sl):
                            px_exit=sl; reason="SL"; exit_now=True
                        elif (pos_side=="long" and hi_i>=r1_price) or (pos_side=="short" and lo_i<=r1_price):
                            has_partial=True
                            if (pos_side=="long" and lo_i<=be_price) or (pos_side=="short" and hi_i>=be_price):
                                px_exit=be_price; reason="BE_after_partial"; exit_now=True
                            elif (pos_side=="long" and hi_i>=tp2) or (pos_side=="short" and lo_i<=tp2):
                                px_exit=tp2; reason="TP2_after_partial"; exit_now=True
                    else:
                        if (pos_side=="long" and hi_i>=r1_price) or (pos_side=="short" and lo_i<=r1_price):
                            has_partial=True
                            if (pos_side=="long" and hi_i>=tp2) or (pos_side=="short" and lo_i<=tp2):
                                px_exit=tp2; reason="TP2_after_partial"; exit_now=True
                            elif (pos_side=="long" and lo_i<=be_price) or (pos_side=="short" and hi_i>=be_price):
                                px_exit=be_price; reason="BE_after_partial"; exit_now=True
                        elif (pos_side=="long" and lo_i<=sl) or (pos_side=="short" and hi_i>=sl):
                            px_exit=sl; reason="SL"; exit_now=True
                else:
                    if args.sl_first:
                        if (pos_side=="long" and lo_i<=be_price) or (pos_side=="short" and hi_i>=be_price):
                            px_exit=be_price; reason="BE_after_partial"; exit_now=True
                        elif (pos_side=="long" and hi_i>=tp2) or (pos_side=="short" and lo_i<=tp2):
                            px_exit=tp2; reason="TP2_after_partial"; exit_now=True
                    else:
                        if (pos_side=="long" and hi_i>=tp2) or (pos_side=="short" and lo_i<=tp2):
                            px_exit=tp2; reason="TP2_after_partial"; exit_now=True
                        elif (pos_side=="long" and lo_i<=be_price) or (pos_side=="short" and hi_i>=be_price):
                            px_exit=be_price; reason="BE_after_partial"; exit_now=True

            # time stop
            if not exit_now and args.time_stop_bars>0 and (i-entry_idx)>=args.time_stop_bars:
                if has_partial and risk>0 and not math.isnan(risk):
                    if pos_side=="long": px_exit=max(cl[i], be_price)
                    else:                px_exit=min(cl[i], be_price)
                    reason="TimeStop_after_partial"
                else:
                    px_exit=cl[i]; reason="TimeStop"
                exit_now=True

            if exit_now:
                # total R with scale-out
                ps = max(0.0, min(1.0, args.partial_size))
                if risk>0 and not math.isnan(risk):
                    if reason=="SL":
                        r_total=-1.0
                    elif reason in ("BE_after_partial","TimeStop_after_partial"):
                        r_total=ps * 1.0
                    elif reason=="TP2_after_partial":
                        r_total=ps * 1.0 + (1.0 - ps) * args.tp_mult
                    else:
                        r_total=(px_exit-entry_price)/risk if pos_side=="long" else (entry_price-px_exit)/risk
                else:
                    r_total=float("nan")

                # MFE/MAE in R
                h_slice=hi[entry_idx:i+1]; l_slice=lo[entry_idx:i+1]
                mfe_r=mae_r=float("nan")
                if risk>0 and not math.isnan(risk):
                    if pos_side=="long":
                        mfe_r=(max(h_slice)-entry_price)/risk
                        mae_r=(entry_price-min(l_slice))/risk
                    else:
                        mfe_r=(entry_price-min(l_slice))/risk
                        mae_r=(max(h_slice)-entry_price)/risk

                trades.append(Trade(
                    symbol=sym, granularity=args.granularity,
                    entry_ts=t[entry_idx], entry=entry_price, side=pos_side,
                    exit_ts=t[i], exit=px_exit, reason=reason,
                    bars_held=(i-entry_idx), r_multiple=r_total,
                    atr_entry=open_meta.get("atr_entry", float("nan")),
                    bbw_pct_entry=open_meta.get("bbw_pct_entry", float("nan")),
                    sqz_dur_entry=open_meta.get("sqz_dur_entry", 0),
                    release_lag=open_meta.get("release_lag", 0),
                    entry_hour_utc=open_meta.get("entry_hour_utc", 0),
                    entry_dow_utc=open_meta.get("entry_dow_utc", 0),
                    mfe_r=mfe_r, mae_r=mae_r
                ))
                in_pos=False; pos_side=None; entry_idx=-1
                has_partial=False; r1_price=float("nan"); be_price=float("nan"); tp2=float("nan")

        # ---------- try entry ----------
        if not in_pos:
            entered=False
            for w in range(1, args.release_window+1):
                k=i-w
                if k-1<0 or k>=n: continue
                if sqz_on[k-1] and not sqz_on[k]:
                    dur=1; j=k-2
                    while j>=0 and sqz_on[j]: dur+=1; j-=1
                    if dur < args.min_sqz_bars: continue

                    mom_i = mom[i]
                    if math.isnan(mom_i): continue
                    dir_long, dir_short = (mom_i>0), (mom_i<0)

                    a = atr[i]; atr_entry = a if not math.isnan(a) else float("nan")
                    start_bbw=max(0, i-args.log_bbw_lookback+1)
                    bbw_vals = bbw[start_bbw:i+1]
                    bbw_curr = bbw[i]
                    bbw_pct_entry = percentile_rank(bbw_vals, bbw_curr) if not math.isnan(bbw_curr) else float("nan")
                    release_lag = i - k
                    dt_entry = dt.datetime.fromtimestamp(t[i], dt.timezone.utc)
                    entry_hour_utc = dt_entry.hour
                    entry_dow_utc = dt_entry.weekday()

                    # filters
                    if args.bbw_max_pct < 100.0 and not math.isnan(bbw_pct_entry) and bbw_pct_entry > args.bbw_max_pct:
                        continue
                    if hours_allow is not None and entry_hour_utc not in hours_allow:
                        continue
                    if dows_allow is not None and entry_dow_utc not in dows_allow:
                        continue
                    if args.exclude_atr_bottom_quartile:
                        atr_valid=[v for v in atr if not math.isnan(v)]
                        if atr_valid:
                            atr_sorted=sorted(atr_valid); q1_idx=int(0.25*(len(atr_sorted)-1)); atr_q1=atr_sorted[q1_idx]
                            if math.isnan(a) or a < atr_q1: continue
                    if (i - last_entry_idx) < args.cooldown_bars:
                        continue

                    # place trade
                    if dir_long:
                        entry_price=cl[i]; entry_idx=i; pos_side="long"
                        sl = entry_price - args.sl_mult * a if not math.isnan(a) and a>0 else float("nan")
                        tp2 = entry_price + args.tp_mult * a if not math.isnan(a) and a>0 else float("nan")
                        in_pos=True; last_entry_idx=i; entered=True
                    elif dir_short:
                        entry_price=cl[i]; entry_idx=i; pos_side="short"
                        sl = entry_price + args.sl_mult * a if not math.isnan(a) and a>0 else float("nan")
                        tp2 = entry_price - args.tp_mult * a if not math.isnan(a) and a>0 else float("nan")
                        in_pos=True; last_entry_idx=i; entered=True

                    if entered:
                        open_meta={"atr_entry":atr_entry,"bbw_pct_entry":bbw_pct_entry,
                                   "sqz_dur_entry":dur,"release_lag":release_lag,
                                   "entry_hour_utc":entry_hour_utc,"entry_dow_utc":entry_dow_utc}
                        has_partial=False; r1_price=float("nan"); be_price=float("nan")
                        break
        i+=1
    return trades

# ----------------------- IO & summary -----------------------
def write_trades(path:str,trades:List[Trade]):
    with open(path,"w",newline="",encoding="utf-8") as f:
        w=csv.writer(f)
        w.writerow([
            "symbol","granularity","entry_ts","exit_ts","side","entry","exit","reason",
            "bars_held","r_multiple",
            "atr_entry","bbw_pct_entry","sqz_dur_entry","release_lag","entry_hour_utc","entry_dow_utc",
            "mfe_r","mae_r"
        ])
        for tr in trades:
            w.writerow([
                tr.symbol,tr.granularity,tr.entry_ts,tr.exit_ts,tr.side,
                f"{tr.entry:.8f}",f"{tr.exit:.8f}",tr.reason,tr.bars_held,
                f"{tr.r_multiple:.4f}" if not math.isnan(tr.r_multiple) else "nan",
                f"{tr.atr_entry:.6f}" if not math.isnan(tr.atr_entry) else "nan",
                f"{tr.bbw_pct_entry:.2f}" if not math.isnan(tr.bbw_pct_entry) else "nan",
                tr.sqz_dur_entry,tr.release_lag,tr.entry_hour_utc,tr.entry_dow_utc,
                f"{tr.mfe_r:.3f}" if not math.isnan(tr.mfe_r) else "nan",
                f"{tr.mae_r:.3f}" if not math.isnan(tr.mae_r) else "nan",
            ])

def summarize(trades:List[Trade])->Dict[str,float]:
    if not trades: return {"trades":0}
    valid=[t for t in trades if not math.isnan(t.r_multiple)]
    wins=[t for t in valid if t.r_multiple>0]
    avg_r = (sum(t.r_multiple for t in valid)/len(valid)) if valid else float("nan")
    winrate = 100.0*len(wins)/len(valid) if valid else 0.0
    by_reason={}
    for t in trades:
        by_reason[t.reason]=by_reason.get(t.reason,0)+1
    return {"trades":len(trades),"winrate_pct":winrate,"avg_R":avg_r,
            "avg_bars":(sum(t.bars_held for t in trades)/len(trades)) if trades else 0.0,
            "tp2_count":by_reason.get("TP2_after_partial",0),
            "sl_count":by_reason.get("SL",0),
            "timed_count":by_reason.get("TimeStop",0)+by_reason.get("TimeStop_after_partial",0),
            "symbols_traded":len(set(t.symbol for t in trades))}

def write_summary(path:str,trades:List[Trade]):
    s=summarize(trades)
    with open(path,"w",newline="",encoding="utf-8") as f:
        w=csv.writer(f); keys=list(s.keys()); w.writerow(keys); w.writerow([s[k] for k in keys])

# ----------------------- Main -----------------------
def main():
    args=parse_args()
    symbols=[s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    all_trades=[]
    for sym in symbols:
        t0=time.time()
        trs=run_symbol(sym,args)
        all_trades.extend(trs)
        print(f"[{sym}] trades={len(trs)} in {time.time()-t0:.1f}s")
    write_trades(f"{args.out_prefix}_trades.csv", all_trades)
    write_summary(f"{args.out_prefix}_summary.csv", all_trades)
    print(f"Wrote: {args.out_prefix}_trades.csv\nWrote: {args.out_prefix}_summary.csv")

if __name__=="__main__":
    main()
