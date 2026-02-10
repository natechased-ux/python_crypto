#!/usr/bin/env python3
"""
Mean-Reversion Telegram Bot (Coinbase data, per-coin training)
Now includes CSV logging and mute option:
- Logs every live alert to mr_live_alerts.csv
- Use --mute to disable Telegram alerts while collecting data.
"""
LOG_PATH = "mr_live_alerts.csv"
#!/usr/bin/env python3
"""
Mean-Reversion Telegram Bot (Coinbase data, per-coin training)
Simplified to run cleanly via:
  python mean_rev.py --mode train --symbols ALL --days 90
  python mean_rev.py --mode live --interval 300
"""
import argparse, json, math, os, sys, time, requests
from datetime import datetime, timedelta, timezone
import numpy as np, pandas as pd
from dataclasses import dataclass, asdict
from typing import Optional, Tuple, List, Dict

CB_BASE = "https://api.exchange.coinbase.com"
PARAMS_PATH = "mr_params.json"
TG_BOT_TOKEN = os.getenv("TG_BOT_TOKEN", "8177096945:AAHTg5nxVTA6hcPkAL4MLAAHPsZ1at7Ywmw")
TG_CHAT_ID   = os.getenv("TG_CHAT_ID", "7967738614")

BLOCKLIST = {"VELO-USD","APT-USD","SYRUP-USD","AXS-USD","SAND-USD","PEPE-USD","WIF-USD","TOSHI-USD","MOODENG-USD","TAO-USD","MORPHO-USD","TIA-USD","FARTCOIN-USD","POPCAT-USD","FET-USD","TURBO-USD","CRV-USD","MAGIC-USD","PNUT-USD"}
DEFAULT_SYMBOLS = ["BTC-USD","ETH-USD","SOL-USD","XRP-USD","ADA-USD","DOGE-USD","AVAX-USD","DOT-USD","LINK-USD","LTC-USD","BCH-USD","MATIC-USD","ATOM-USD","NEAR-USD","ARB-USD","OP-USD","INJ-USD","SUI-USD","SEI-USD","STX-USD","RUNE-USD","AAVE-USD","UNI-USD","MKR-USD","SNX-USD","FIL-USD","ETC-USD","ALGO-USD","HBAR-USD","ICP-USD","LDO-USD","AR-USD","XLM-USD","XTZ-USD","EOS-USD","KAVA-USD","GRT-USD","IMX-USD","APE-USD","JUP-USD","QNT-USD","CHZ-USD","SUSHI-USD","1INCH-USD","LRC-USD","BAT-USD","ZRX-USD","FLOW-USD","ENJ-USD","ANKR-USD"]
DEFAULT_SYMBOLS = [s for s in DEFAULT_SYMBOLS if s not in BLOCKLIST]

@dataclass
class MRParams:
    ma_window: int
    bb_mult: float
    z_entry: float
    hold_max_candles: int
    atr_mult_tp: float
    atr_mult_sl: float
    allow_mean_exit: bool
    adx_max: float  # regime gate
    ma_slope_max: float  # new: require flat-ish MA (reduces trend risk)

    def to_dict(self): return asdict(self)

def list_products() -> set:
    """Fetch available product IDs (e.g., BTC-USD) from Coinbase."""
    try:
        r = requests.get(f"{CB_BASE}/products", timeout=30)
        r.raise_for_status()
        js = r.json()
        return {p.get("id") for p in js if isinstance(p, dict) and p.get("id")}
    except Exception:
        return set()


def fetch_candles(symbol: str, start: datetime, end: datetime, granularity: int=900) -> pd.DataFrame:
    """
    Coinbase limits each call to ~300 candles. Chunk the range and stitch.
    Handle 404 (unsupported product) by returning an empty DataFrame.
    """
    max_span = timedelta(seconds=granularity * 300 - 1)
    frames = []
    cur_start = start.replace(microsecond=0)
    while cur_start < end:
        cur_end = min(cur_start + max_span, end)
        url = f"{CB_BASE}/products/{symbol}/candles"
        params = {
            "start": cur_start.isoformat(),
            "end": cur_end.isoformat(),
            "granularity": granularity,
        }
        r = requests.get(url, params=params, timeout=30)
        if r.status_code == 404:
            # Symbol not supported/delisted on Coinbase
            return pd.DataFrame(columns=["time","low","high","open","close","volume"]) 
        if r.status_code == 429:
            time.sleep(1.0)
            continue
        if not (200 <= r.status_code < 300):
            # Skip this window gracefully
            cur_start = cur_end + timedelta(seconds=granularity)
            time.sleep(0.2)
            continue
        arr = r.json()
        if not isinstance(arr, list):
            arr = []
        df = pd.DataFrame(arr, columns=["time","low","high","open","close","volume"]) if arr else pd.DataFrame(columns=["time","low","high","open","close","volume"]) 
        if not df.empty:
            df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
            frames.append(df)
        # move window forward by one candle to avoid overlap
        cur_start = cur_end + timedelta(seconds=granularity)
        # be polite to rate limits
        time.sleep(0.2)
    if not frames:
        return pd.DataFrame(columns=["time","low","high","open","close","volume"]) 
    out = pd.concat(frames, ignore_index=True)
    return out.sort_values("time").drop_duplicates(subset=["time"]).reset_index(drop=True)

def add_indicators(df: pd.DataFrame, ma_window:int) -> pd.DataFrame:
    d = df.copy()
    # Core MR features
    d["ma"] = d["close"].rolling(ma_window).mean()
    d["std"] = d["close"].rolling(ma_window).std(ddof=0)
    d["z"] = (d["close"] - d["ma"]) / (d["std"] + 1e-9)
    high, low, close = d["high"], d["low"], d["close"]
    prev = close.shift(1)
    tr = pd.concat([(high-low), (high-prev).abs(), (low-prev).abs()], axis=1).max(axis=1)
    d["atr"] = tr.ewm(alpha=1/14, adjust=False).mean()
    # ADX(14) for regime filter
    up = high.diff()
    down = -low.diff()
    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)
    tr_14 = tr.rolling(14).sum()
    plus_di = 100 * pd.Series(plus_dm, index=d.index).rolling(14).sum() / (tr_14 + 1e-9)
    minus_di = 100 * pd.Series(minus_dm, index=d.index).rolling(14).sum() / (tr_14 + 1e-9)
    dx = (100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-9))
    d["adx"] = dx.rolling(14).mean()
    # MA flatness (6-bar slope normalized by level)
    d["ma_slope"] = (d["ma"] - d["ma"].shift(6)).abs() / (d["ma"].abs() + 1e-9)
    return d

def backtest_mean_rev(d:pd.DataFrame,p:MRParams)->Tuple[float,float,int,float,float]:
    d=d.copy()
    d["u"]=d["ma"]+p.bb_mult*d["std"]
    d["l"]=d["ma"]-p.bb_mult*d["std"]
    trades=[]
    pos=None; entry=0; idx=None
    last_trade_day=None
    start_i=max(p.ma_window,20)
    rr_floor = 1.25  # minimum TP/SL ratio for healthier payoffs; entry=0; idx=None
    start_i=max(p.ma_window,20)
    for i in range(start_i,len(d)):
        row=d.iloc[i]
        if pos is None:
            # Re-entry confirmation + regime/flatness gates
            prev = d.iloc[i-1]
            adx_ok = (row.get("adx", np.nan) < p.adx_max) if "adx" in row else True
            flat_ok = (row.get("ma_slope", np.nan) < p.ma_slope_max) if "ma_slope" in row else True
            # one trade per day cap
            day_ok = (last_trade_day is None) or (row["time"].date() != last_trade_day)
            # require decent RR from params
            rr_ok = (p.atr_mult_tp / max(1e-9, p.atr_mult_sl)) >= rr_floor
            long_signal = (prev["z"] <= -p.z_entry) and (row["z"] > -p.z_entry)
            short_signal= (prev["z"] >=  p.z_entry) and (row["z"] <  p.z_entry)
            if day_ok and rr_ok and adx_ok and flat_ok and long_signal:
                pos="long"; entry=row["close"]; idx=i; last_trade_day=row["time"].date()
            elif day_ok and rr_ok and adx_ok and flat_ok and short_signal:
                pos="short"; entry=row["close"]; idx=i; last_trade_day=row["time"].date()
            elif day_ok and adx_ok and short_signal:
                pos="short"; entry=row["close"]; idx=i; last_trade_day=row["time"].date()
        else:
            held=i-idx
            atr=float(d.iloc[i]["atr"])
            if atr<=0: atr=1e-6
            if pos=="long": tp=entry+p.atr_mult_tp*atr; sl=entry-p.atr_mult_sl*atr
            else: tp=entry-p.atr_mult_tp*atr; sl=entry+p.atr_mult_sl*atr
            hit_tp=row["high"]>=tp if pos=="long" else row["low"]<=tp
            hit_sl=row["low"]<=sl if pos=="long" else row["high"]>=sl
            mean_touch=p.allow_mean_exit and ((pos=="long" and row["close"]>=row["ma"]) or (pos=="short" and row["close"]<=row["ma"]))
            if hit_tp or hit_sl or mean_touch or held>=p.hold_max_candles:
                pnl=(tp-entry) if hit_tp else (sl-entry if hit_sl else row["close"]-entry)
                if pos=="short": pnl=-pnl
                trades.append(pnl/(abs(entry-sl)+1e-9))
                pos=None
    if not trades: return (0,0,0)
    arr=np.array(trades)
    total=arr.sum(); sharpe=arr.mean()/(arr.std(ddof=1)+1e-9); n=len(arr)
    winrate=float((arr>0).mean()); avg_r=float(arr.mean())
    return total, sharpe, n, winrate, avg_r

def train_best_params(df:pd.DataFrame)->Tuple[MRParams, Dict]:
    best=None
    for mw in [20,30,40]:
        base=add_indicators(df,mw).dropna()
        for bb in [1.5,2.0]:
            for zt in [1.5,1.75,2.0,2.25]:
                for hold in [24,36]:
                    for tp in [1.0,1.25,1.5,1.75,2.0]:
                        for sl in [0.75,1.0,1.25]:
                            for mean_exit in [True,False]:
                                for adx_cap in [18,20,22]:
                                    for slope_cap in [0.0015, 0.0025, 0.0035]:
                                        p=MRParams(mw,bb,zt,hold,tp,sl,mean_exit, adx_cap, slope_cap)
                                total,sh,n,wr,avg_r=backtest_mean_rev(base,p)
                                # Emphasize better per-trade payoff while keeping consistency and limiting churn
                                score = 0.75*sh + 0.2*avg_r + 0.05*total - 0.0005*n  # push for higher Sharpe
                                if not best or score>best[0]:
                                    best=(score,p)
                                    best_metrics={"total":float(total),"sharpe":float(sh),"trades":int(n),"winrate":float(wr),"avg_r":float(avg_r)}
    return best[1], best_metrics

def load_params()->Dict:
    if os.path.exists(PARAMS_PATH):
        with open(PARAMS_PATH) as f: return json.load(f)
    return {}

def save_params(d:Dict):
    with open(PARAMS_PATH,"w") as f: json.dump(d,f,indent=2)

def tg_send(msg:str):
    try:
        requests.get(f"https://api.telegram.org/bot{TG_BOT_TOKEN}/sendMessage",params={"chat_id":TG_CHAT_ID,"text":msg},timeout=10)
    except Exception as e: print(e)

def cmd_train(symbols:List[str],days:int,granularity:int):
    end=datetime.now(timezone.utc); start=end-timedelta(days=days)
    store=load_params()
    # Filter to only available products
    available = list_products()
    if available:
        missing=[s for s in symbols if s not in available]
        if missing:
            print(f"[warn] Skipping unsupported symbols: {', '.join(missing)}")
        symbols=[s for s in symbols if s in available]
    for sym in symbols:
        print(f"Training {sym}...")
        df=fetch_candles(sym,start,end,granularity)
        if df.empty or len(df)<200:
            print(f"[warn] Not enough data for {sym} (len={len(df)}). Skipping.")
            continue
        best,metrics=train_best_params(df)
        store[sym]={"params":best.to_dict(),"trained":datetime.now(timezone.utc).isoformat(),"granularity":granularity,"lookback_days":days, "metrics":metrics}
    save_params(store)
    print(f"[ok] Saved params to {PARAMS_PATH}")

def check_signal(df:pd.DataFrame,p:MRParams)->Optional[Dict]:
    d=add_indicators(df,p.ma_window).dropna()
    if len(d)<p.ma_window+6:
        return None
    row=d.iloc[-1]
    prev=d.iloc[-2]
    # re-entry + ADX + flat MA gate + RR floor
    adx_ok = (float(row.get("adx", 0.0)) < p.adx_max) if "adx" in d.columns else True
    flat_ok = (float(row.get("ma_slope", 0.0)) < p.ma_slope_max) if "ma_slope" in d.columns else True
    rr_ok = (p.atr_mult_tp / max(1e-9, p.atr_mult_sl)) >= 1.25
    long_sig = (prev["z"] <= -p.z_entry) and (row["z"] > -p.z_entry) and adx_ok and flat_ok and rr_ok
    short_sig= (prev["z"] >=  p.z_entry) and (row["z"] <  p.z_entry) and adx_ok and flat_ok and rr_ok
    if not (long_sig or short_sig):
        return None
    side = "LONG" if long_sig else ("SHORT" if short_sig else None)
    if side is None:
        return None
    atr=float(row["atr"]); entry=float(row["close"])
    tp=entry+(p.atr_mult_tp*atr if side=="LONG" else -p.atr_mult_tp*atr)
    sl=entry-(p.atr_mult_sl*atr if side=="LONG" else -p.atr_mult_sl*atr)
    return {"side":side,"entry":entry,"tp":tp,"sl":sl,"z":row["z"],"ma":row["ma"],"atr":atr}

def cmd_live(symbols:List[str],interval:int,granularity:int,cooldown:int, mute:bool):
    store=load_params(); last={}
    # per-day cap
    used_day = {}
    # Filter to only available products
    available = list_products()
    if available:
        symbols=[s for s in symbols if s in available]
    if not symbols:
        print("[error] No supported symbols to scan. Exiting.")
        return
    while True:
        loop_start = time.time()
        for sym in symbols:
            conf=store.get(sym)
            if not conf:
                continue
            p=MRParams(**conf["params"])
            end=datetime.now(timezone.utc); start=end-timedelta(hours=48)
            df=fetch_candles(sym,start,end,granularity)
            if df.empty:
                continue
            sig=check_signal(df,p)
            # one trade per coin per day
            today = datetime.now(timezone.utc).date()
            day_ok = (used_day.get(sym) != today)
            if sig and day_ok and (sym not in last or time.time()-last[sym]>cooldown*60):
                used_day[sym] = today
                last[sym]=time.time()
                msg = "".join([
                    f"{sym} {sig['side']}",
                    f"Entry:{sig['entry']:.4f}",
                    f"TP:{sig['tp']:.4f}",
                    f"SL:{sig['sl']:.4f}",
                    f"z:{sig['z']:.2f} ma:{sig['ma']:.4f}",
                ])
                print(msg)
                append_live_log(sym, sig)
                if not mute:
                    tg_send(msg)
        # maintain interval pacing even if loop runs fast
        elapsed = time.time() - loop_start
        time.sleep(max(1, interval - int(elapsed)))

def parse_args():
    ap=argparse.ArgumentParser()
    ap.add_argument('--mode',choices=['train','live','report'],required=True)
    ap.add_argument('--symbols',nargs='*',default=None)
    ap.add_argument('--days',type=int,default=90)
    ap.add_argument('--interval',type=int,default=300)
    ap.add_argument('--granularity',type=int,default=900)
    ap.add_argument('--cooldown_min',type=int,default=30)
    ap.add_argument('--mute', action='store_true', help='Mute Telegram alerts (still logs to CSV)')
    ap.add_argument('--top', type=int, default=15, help='report: show top-N by Sharpe')
    return ap.parse_args()

def cmd_report(top:int):
    store=load_params()
    rows=[]
    for sym,conf in store.items():
        m=conf.get('metrics',{})
        rows.append((sym,float(m.get('sharpe',0.0)),int(m.get('trades',0)),float(m.get('winrate',0.0)),float(m.get('avg_r',0.0))))
    rows.sort(key=lambda x:(x[1],x[3],x[2]), reverse=True)
    print("SYMBOL  SHARPE  TRADES  WINRATE  AVG_R")
    for sym,sh,tr,wr,ar in rows[:top]:
        print(f"{sym:7s} {sh:6.2f} {tr:7d} {wr:7.2%} {ar:6.3f}")


def main():
    args=parse_args()
    if not args.symbols or args.symbols==['ALL']:
        syms=DEFAULT_SYMBOLS
    else:
        syms=[s.upper() for s in args.symbols if s.upper() not in BLOCKLIST]
    if args.mode=='train':
        cmd_train(syms,args.days,args.granularity)
    elif args.mode=='live':
        if not os.path.exists(PARAMS_PATH):
            print('Train first!'); sys.exit(1)
        cmd_live(syms,args.interval,args.granularity,args.cooldown_min, args.mute)
    else:
        cmd_report(args.top)

if __name__=='__main__':
    main()
