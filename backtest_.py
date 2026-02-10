#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Historical Backtester (Looser/Toggleable) for the Structure Strategy
- Adds switches to relax filters: trend bias, EMA stacking, MACD zero-cross, weekend skip, confirmation.
- Wider parameter grids to actually produce trades when the baseline is too strict.
- Still uses Coinbase historical batching + cache.

Run:
  python structure_backtest_historical_loose.py
"""

import os, time, datetime as dt
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import requests, numpy as np, pandas as pd

# ------------- User params (looser defaults) -------------
COINS = ["BTC-USD","ETH-USD","XRP-USD","APE-USD","SOL-USD"]

START_DATE = "2025-07-01"   # inclusive UTC
END_DATE   = "2025-08-15"   # exclusive  UTC

# Wider grids (so something fires)
GRID_BASE_R_MULT = [0.6, 0.75, 0.9, 1.1]
GRID_ADX_MIN = [14, 18, 21]
GRID_BB_WIDTH_MIN_PCTL = [0.0, 0.10, 0.20]

# Toggles (flip to False to loosen further)
USE_TREND_BIAS = False          # require 1D EMA200 bias
USE_EMA_STACKING = False        # require 1H EMA10>20>50 (or reverse for shorts)
REQUIRE_MACD_ZERO_CROSS = False # if False, allow hist sign without strict cross
SKIP_WEEKEND = False           # allow weekend/session trades
USE_CONFIRMATION = False        # Stoch RSI confirmation (turn off to see raw structure hits)

# RSI regime (slightly relaxed)
RSI_LONG_FLOOR = 48
RSI_SHORT_CEIL = 52

# Confirmation config (looser)
CONFIRMATION_TIMEOUT_MIN = 120
STOCH_RSI_LEN = 14
STOCH_RSI_SMOOTH_K = 3
STOCH_RSI_SMOOTH_D = 3

# Indicators / calc params
ATR_LEN = 14
ADX_LEN = 14
MACD_FAST, MACD_SLOW, MACD_SIG = 12, 26, 9
EMA_SET = [10,20,50,200]
BB_LEN, BB_STD, BB_WIDTH_LOOKBACK = 20, 2.0, 120

# Coinbase + IO
CB_BASE = "https://api.exchange.coinbase.com"
MAX_PER_REQ = 300
CACHE_DIR = "cache"
OUT_TRADES = "bt_trades.csv"
OUT_SUMMARY = "bt_summary.csv"

# ------------- Utils -------------
def ensure_dir(p): os.makedirs(p, exist_ok=True)
def parse_date(s:str)->dt.datetime: return dt.datetime.strptime(s,"%Y-%m-%d").replace(tzinfo=dt.timezone.utc)

def chunk_timerange(start:dt.datetime, end:dt.datetime, gran_s:int, max_points:int):
    step = dt.timedelta(seconds=gran_s*max_points - gran_s)
    cur=start; out=[]
    while cur<end:
        nxt=min(end, cur+step); out.append((cur,nxt)); cur=nxt
    return out

def fetch_candles_range(symbol:str, granularity:int, start:dt.datetime, end:dt.datetime)->pd.DataFrame:
    now = dt.datetime.now(dt.timezone.utc)
    safe_end = min(end, now - dt.timedelta(seconds=granularity))
    if safe_end <= start:
        return pd.DataFrame(columns=["time","low","high","open","close","volume","dt"])
    frames=[]
    for s,e in chunk_timerange(start, safe_end, granularity, MAX_PER_REQ):
        params={"granularity":granularity,"start":int(s.timestamp()),"end":int(e.timestamp())}
        url=f"{CB_BASE}/products/{symbol}/candles"
        backoff=0.5
        for attempt in range(6):
            try:
                r=requests.get(url,params=params,timeout=20)
                if r.status_code in (429,500,502,503,504): raise requests.HTTPError(f"{r.status_code} {r.text}")
                r.raise_for_status()
                arr=r.json()
                if not isinstance(arr,list): raise ValueError(f"Unexpected response: {arr}")
                frames.append(pd.DataFrame(arr,columns=["time","low","high","open","close","volume"]))
                break
            except Exception as ex:
                if attempt==5: print(f"[{symbol}] fetch retry exhausted {s}->{e}: {ex}")
                else: time.sleep(backoff); backoff=min(8.0, backoff*2)
        time.sleep(0.15)
    if not frames: return pd.DataFrame(columns=["time","low","high","open","close","volume","dt"])
    df=pd.concat(frames,ignore_index=True).drop_duplicates("time").sort_values("time")
    df["dt"]=pd.to_datetime(df["time"],unit="s",utc=True)
    for c in ["open","high","low","close","volume"]:
        df[c]=pd.to_numeric(df[c],errors="coerce")
    return df.reset_index(drop=True)

def load_or_fetch(symbol:str, tf:str, gran:int, start:dt.datetime, end:dt.datetime)->pd.DataFrame:
    ensure_dir(CACHE_DIR)
    path=os.path.join(CACHE_DIR,f"{symbol.replace('/','_')}-{tf}.csv")
    if os.path.exists(path):
        df=pd.read_csv(path); df["dt"]=pd.to_datetime(df["dt"],utc=True)
        if df["dt"].min()<=start and df["dt"].max()>=end:
            return df[(df["dt"]>=start-dt.timedelta(days=10))&(df["dt"]<=end+dt.timedelta(days=10))].reset_index(drop=True)
    df=fetch_candles_range(symbol,gran,start-dt.timedelta(days=30),end+dt.timedelta(days=30))
    df.to_csv(path,index=False); return df

def to_pacific(ts:dt.datetime)->dt.datetime: return ts.astimezone(dt.timezone(dt.timedelta(hours=-7)))
def in_weekend_quiet(ts:dt.datetime)->bool:
    if not SKIP_WEEKEND: return False
    pt=to_pacific(ts); wd=pt.weekday(); h=pt.hour
    if wd==4 and h>=17: return True
    if wd==5: return True
    if wd==6 and h<14: return True
    return False

# TA helpers
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
def rolling_percentile(series:pd.Series,val:float,look:int)->float:
    if len(series)<look: return 1.0
    ref=series.iloc[-look:]; return (ref<val).sum()/max(1,len(ref))

def ema_guidance(df1h:pd.DataFrame, df1d:pd.DataFrame):
    e1h={f"EMA{L}": float(ema(df1h["close"],L).iloc[-2]) for L in EMA_SET}
    e1d={f"EMA{L}": float(ema(df1d["close"],L).iloc[-2]) for L in EMA_SET}
    return e1h,e1d
def ema_stacking_ok(e1h:Dict[str,float], side:str)->bool:
    e10,e20,e50=e1h["EMA10"],e1h["EMA20"],e1h["EMA50"]
    return (e10>e20>e50) if side=="LONG" else (e10<e20<e50)

def last_swing_levels(df:pd.DataFrame, look:int=25):
    if len(df)<look+3: return None,None
    H=df["high"].values; L=df["low"].values; sh=sl=None
    for i in range(len(df)-2,2,-1):
        if sh is None and H[i]>H[i-1] and H[i]>H[i-2] and H[i]>H[i+1]: sh=float(H[i])
        if sl is None and L[i]<L[i-1] and L[i]<L[i-2] and L[i]<L[i+1]: sl=float(L[i])
        if sh is not None and sl is not None: break
    return sh,sl

# ------------- Strategy -------------
@dataclass
class StructureSignal:
    symbol:str; side:str; entry:float; tp:float; sl:float; sig_time:dt.datetime
    adx:float; pdi:float; mdi:float; macd_hist:float; rsi:float; atr:float; bbw:float
    params:Dict[str,float]

def structure_signal_1h(df1h:pd.DataFrame, df1d:pd.DataFrame, params:Dict[str,float])->Optional[StructureSignal]:
    ADX_MIN=params["ADX_MIN"]; BASE_R_MULT=params["BASE_R_MULT"]; BB_PCTL=params["BB_PCTL"]
    if len(df1h)<120: return None
    c=df1h["close"]; h=df1h["high"]; l=df1h["low"]
    adx_v,pdi,mdi=adx(h,l,c,ADX_LEN); _,_,hist=macd(c,MACD_FAST,MACD_SLOW,MACD_SIG)
    r=rsi(c,14); atr1h=atr(h,l,c,ATR_LEN); bbw=bb_width(c,BB_LEN,BB_STD)

    adx2=float(adx_v.iloc[-2]); pdi2=float(pdi.iloc[-2]); mdi2=float(mdi.iloc[-2])
    hist2=float(hist.iloc[-2]); hist3=float(hist.iloc[-3]); r2=float(r.iloc[-2]); atr2=float(atr1h.iloc[-2]); bbw2=float(bbw.iloc[-2])

    # Volatility gate (can be zero to disable)
    perc=rolling_percentile(bbw,bbw2,BB_WIDTH_LOOKBACK)
    if perc <= BB_PCTL: return None

    last_close=float(c.iloc[-2]); sh,sl=last_swing_levels(df1h,25)
    e1h,e1d=ema_guidance(df1h,df1d)
    # Trend bias optional
    trend_long_ok = (last_close > e1d["EMA200"]) if USE_TREND_BIAS else True
    trend_short_ok= (last_close < e1d["EMA200"]) if USE_TREND_BIAS else True

    # MACD condition: strict zero-cross OR just sign in direction
    macd_long_ok  = (hist2>0 and (hist3<=0 if REQUIRE_MACD_ZERO_CROSS else True))
    macd_short_ok = (hist2<0 and (hist3>=0 if REQUIRE_MACD_ZERO_CROSS else True))

    long_setup  = (adx2>ADX_MIN and pdi2>mdi2 and macd_long_ok  and r2>=RSI_LONG_FLOOR  and trend_long_ok)
    short_setup = (adx2>ADX_MIN and mdi2>pdi2 and macd_short_ok and r2<=RSI_SHORT_CEIL and trend_short_ok)

    if long_setup and sl is not None:
        side="LONG"; risk=max(1e-8,last_close-sl); entry=last_close; slp=sl
    elif short_setup and sh is not None:
        side="SHORT"; risk=max(1e-8,sh-last_close); entry=last_close; slp=sh
    else:
        return None

    if USE_EMA_STACKING and not ema_stacking_ok(e1h, side):
        return None

    # Dynamic R-multiple (kept from advanced logic)
    ratio=risk/max(1e-12,atr2); rmult=BASE_R_MULT
    if ratio<0.5: rmult=min(1.2, BASE_R_MULT+0.3)
    elif ratio>1.8: rmult=max(0.5, BASE_R_MULT-0.25)
    tp = entry + rmult*risk if side=="LONG" else entry - rmult*risk
    sig_time=pd.to_datetime(df1h["dt"].iloc[-2]).to_pydatetime()
    return StructureSignal("",side,entry,tp,slp,sig_time,adx2,pdi2,mdi2,hist2,r2,atr2,bbw2,params)

def stoch_confirm_time(df15:pd.DataFrame, side:str, start_ts:dt.datetime, timeout_min:int)->Optional[dt.datetime]:
    end_ts=start_ts+dt.timedelta(minutes=timeout_min)
    d=df15[df15["dt"]>start_ts].copy()
    if d.empty: return None
    k,dline=stoch_rsi(d["close"],STOCH_RSI_LEN,STOCH_RSI_SMOOTH_K,STOCH_RSI_SMOOTH_D)
    d["K"]=k; d["D"]=dline; d=d.dropna().reset_index(drop=True)
    for i in range(1,len(d)):
        k1=float(d.loc[i,"K"]); d1=float(d.loc[i,"D"]); k0=float(d.loc[i-1,"K"]); slope=k1-k0
        long_ok=(k1>d1) and (k1<50.0) and (slope>0.0)   # slightly looser (was <40)
        short_ok=(k1<d1) and (k1>50.0) and (slope<0.0)  # slightly looser (was >60)
        ts=d.loc[i,"dt"].to_pydatetime()
        if ts>end_ts: break
        if (side=="LONG" and long_ok) or (side=="SHORT" and short_ok): return ts
    return None

def simulate_trade(df15:pd.DataFrame, side:str, entry:float, tp:float, sl:float, start_ts:dt.datetime):
    f=df15[df15["dt"]>start_ts].reset_index(drop=True)
    for i in range(len(f)):
        hi=float(f.loc[i,"high"]); lo=float(f.loc[i,"low"]); ts=f.loc[i,"dt"].to_pydatetime()
        if side=="LONG":
            if hi>=tp: return "TP",  (tp-entry)/max(1e-12,abs(entry-sl)), ts-start_ts, 0,0
            if lo<=sl: return "SL",  (sl-entry)/max(1e-12,abs(entry-sl)), ts-start_ts, 0,0
        else:
            if lo<=tp: return "TP",  (entry-tp)/max(1e-12,abs(sl-entry)), ts-start_ts, 0,0
            if hi>=sl: return "SL",  (entry-sl)/max(1e-12,abs(sl-entry)), ts-start_ts, 0,0
    if len(f): return "OPEN",0.0,f.loc[len(f)-1,"dt"].to_pydatetime()-start_ts,0,0
    return "OPEN",0.0,dt.timedelta(0),0,0

# ------------- Backtest loop -------------
def run_symbol(symbol:str, start:dt.datetime, end:dt.datetime, params:Dict[str,float])->List[Dict]:
    df1h=load_or_fetch(symbol,"1h",3600,start,end)
    df15=load_or_fetch(symbol,"15m",900,start,end)
    df1d=load_or_fetch(symbol,"1d",86400,start,end)
    df1h=df1h[(df1h["dt"]>=start-dt.timedelta(days=5))&(df1h["dt"]<end+dt.timedelta(days=1))].reset_index(drop=True)
    df15=df15[(df15["dt"]>=start-dt.timedelta(days=5))&(df15["dt"]<end+dt.timedelta(days=1))].reset_index(drop=True)
    df1d=df1d[(df1d["dt"]>=start-dt.timedelta(days=120))&(df1d["dt"]<end+dt.timedelta(days=1))].reset_index(drop=True)
    out=[]
    for idx in range(50, len(df1h)-2):
        ts=pd.to_datetime(df1h.loc[idx,"dt"]).to_pydatetime()
        if ts<start or ts>=end: continue
        if in_weekend_quiet(ts): continue
        sig=structure_signal_1h(df1h.iloc[:idx+1].copy(), df1d, params)
        if not sig: continue
        sig.symbol=symbol
        if USE_CONFIRMATION:
            cts=stoch_confirm_time(df15, sig.side, ts, CONFIRMATION_TIMEOUT_MIN)
            if not cts: continue
        else:
            cts=ts  # count structure entries without confirmation
        outcome, rr, dur, _, _ = simulate_trade(df15, sig.side, sig.entry, sig.tp, sig.sl, cts)
        out.append({
            "symbol":symbol,"side":sig.side,"sig_time":ts.isoformat(),"confirm_time":cts.isoformat(),
            "entry":sig.entry,"tp":sig.tp,"sl":sig.sl,"outcome":outcome,"rr":rr,"duration_min":dur.total_seconds()/60.0,
            "BASE_R_MULT":params["BASE_R_MULT"],"ADX_MIN":params["ADX_MIN"],"BB_PCTL":params["BB_PCTL"]
        })
    return out

def summarize(trades:List[Dict])->pd.DataFrame:
    df=pd.DataFrame(trades)
    if df.empty: return pd.DataFrame()
    df["win"]=(df["outcome"]=="TP").astype(int)
    g=df.groupby(["BASE_R_MULT","ADX_MIN","BB_PCTL"]).agg(
        trades=("symbol","count"), wins=("win","sum"),
        win_rate=("win","mean"), avg_rr=("rr","mean"), median_rr=("rr","median")
    ).reset_index()
    return g.sort_values(["win_rate","avg_rr","trades"], ascending=[False,False,False])

def main():
    start=parse_date(START_DATE); end=parse_date(END_DATE)
    all_trades=[]
    for brm in GRID_BASE_R_MULT:
        for adx_min in GRID_ADX_MIN:
            for pctl in GRID_BB_WIDTH_MIN_PCTL:
                params={"BASE_R_MULT":brm,"ADX_MIN":adx_min,"BB_PCTL":pctl}
                for sym in COINS:
                    try:
                        print(f"Backtesting {sym} {params} (toggles: trend={USE_TREND_BIAS}, stack={USE_EMA_STACKING}, macdZero={REQUIRE_MACD_ZERO_CROSS}, confirm={USE_CONFIRMATION})")
                        all_trades.extend(run_symbol(sym,start,end,params))
                    except Exception as e:
                        print(f"[{sym}] error: {e}")
                    time.sleep(0.1)
    if all_trades:
        df=pd.DataFrame(all_trades); df.to_csv(OUT_TRADES, index=False)
        summ=summarize(all_trades); summ.to_csv(OUT_SUMMARY, index=False)
        print(f"Saved {len(df)} trades -> {OUT_TRADES}")
        print(f"Saved summary -> {OUT_SUMMARY}")
        print(summ.head(20))
    else:
        print("Still no trades. Try: USE_CONFIRMATION=False, USE_TREND_BIAS=False, USE_EMA_STACKING=False, expand date range or coins.")

if __name__=="__main__": main()
