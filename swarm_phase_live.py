#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, json, math, threading, time, traceback, queue, sys, hmac, hashlib
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta, timezone

import requests
import pandas as pd
import numpy as np
from websocket import WebSocketApp

# -------------- Config / Utils --------------

CB_REST = "https://api.exchange.coinbase.com"
WS_URL  = "wss://ws-feed.exchange.coinbase.com"

def utcnow():
    return datetime.now(timezone.utc)

def pct_rank(arr: np.ndarray, v: float) -> float:
    if arr.size < 8: return 0.0
    return float((arr <= v).sum()) / float(arr.size)

def clamp01(x): return max(0.0, min(1.0, float(x)))

def bb_width_pct(close: pd.Series, period=20, stds=2.0):
    mid = close.rolling(period, min_periods=period).mean()
    sd  = close.rolling(period, min_periods=period).std(ddof=0)
    upper, lower = mid + stds*sd, mid - stds*sd
    return ((upper - lower) / mid.replace(0, np.nan)).replace([np.inf,-np.inf], np.nan)

def atr_from_df(df: pd.DataFrame, period=14):
    h,l,c = df["high"], df["low"], df["close"]
    prev_c = c.shift(1)
    tr = pd.concat([h-l, (h-prev_c).abs(), (l-prev_c).abs()], axis=1).max(axis=1)
    return tr.rolling(period, min_periods=period).mean()

def stoch_rsi(close: pd.Series, rsi_period=14, stoch_period=14, smooth_k=3, smooth_d=3):
    delta = close.diff()
    up = delta.clip(lower=0.0); down = -delta.clip(upper=0.0)
    roll_up = up.ewm(alpha=1/rsi_period, adjust=False).mean()
    roll_dn = down.ewm(alpha=1/rsi_period, adjust=False).mean()
    rs = roll_up / roll_dn.replace(0,np.nan)
    rsi = 100 - (100/(1+rs))
    lo = rsi.rolling(stoch_period, min_periods=stoch_period).min()
    hi = rsi.rolling(stoch_period, min_periods=stoch_period).max()
    st = (rsi - lo) / (hi - lo).replace(0,np.nan) * 100.0
    k = st.rolling(smooth_k, min_periods=smooth_k).mean()
    d = k.rolling(smooth_d, min_periods=smooth_d).mean()
    return k, d

def donchian(df: pd.DataFrame, period=20):
    hi = df["high"].rolling(period, min_periods=period).max()
    lo = df["low"].rolling(period, min_periods=period).min()
    return hi.shift(1), lo.shift(1)  # IMPORTANT: use prior box

# -------------- Strategy knobs --------------

@dataclass
class Strat:
    swarm_thresh: float = 0.60
    min_comp: float    = 0.55
    min_flow: float    = 0.35
    min_cluster: float = 0.30
    breakout_buffer_atr: float = 0.25
    donchian_period: int = 14
    atr_period: int = 14
    stoch_gate: bool = True
    cooldown_min: int = 45
    clv_window_sec: int = 600   # 10 min for flow window
    vbp_lookback_min: int = 72*60 # 72h for cluster VBP
    cluster_min_pct: float = 0.0075 # 0.75%
    cluster_max_pct: float = 0.03   # 3.0%
    bin_width_pct: float = 0.002    # 0.2%

# -------------- Data stores (per symbol) --------------

class LiveState:
    def __init__(self):
        self.trades = []   # list of dicts {t, side, size, price}
        self.l2_bins = {}  # price_bin -> cum_size (rolling)
        self.l2_prices = []  # recent mid prices for proximity calc
        self.candles_1m = pd.DataFrame(columns=["open","high","low","close","volume"])
        self.candles_15m = pd.DataFrame(columns=["open","high","low","close","volume"])
        self.candles_1h = pd.DataFrame(columns=["open","high","low","close","volume"])
        self.last_alert_at = None

# -------------- Coinbase REST candles --------------

def fetch_candles(product: str, start: datetime, end: datetime, gran: int) -> pd.DataFrame:
    frames = []
    span = gran * 300
    t1 = start
    while t1 < end:
        t2 = min(end, t1 + timedelta(seconds=span))
        params = {"start": t1.isoformat(), "end": t2.isoformat(), "granularity": gran}
        r = requests.get(f"{CB_REST}/products/{product}/candles", params=params, timeout=20)
        r.raise_for_status()
        data = r.json()
        if not data:
            t1 = t2 + timedelta(seconds=gran); continue
        df = pd.DataFrame(data, columns=["time","low","high","open","close","volume"])
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
        df = df.sort_values("time").set_index("time")[["open","high","low","close","volume"]].astype(float)
        frames.append(df)
        last = df.index.max()
        t1 = (last + timedelta(seconds=gran)) if pd.notna(last) else t2 + timedelta(seconds=gran)
        time.sleep(0.1)
    if not frames: return pd.DataFrame(columns=["open","high","low","close","volume"])
    out = pd.concat(frames).sort_index()
    return out[~out.index.duplicated(keep="last")]

# -------------- WebSocket consumers --------------
def normalize_products(names: List[str]) -> List[str]:
    out = []
    for n in names:
        n = n.strip()
        if not n: continue
        n = n.replace("_", "-").upper()
        # ensure -USD suffix if user passed just "BTC"
        if "-" not in n:
            n = f"{n}-USD"
        elif not n.endswith("-USD"):
            # if user passed BTC-USD already, fine; otherwise leave as-is
            pass
        out.append(n)
    # dedupe preserving order
    seen = set(); out2 = []
    for x in out:
        if x not in seen:
            out2.append(x); seen.add(x)
    return out2


class WSGroup:
    def __init__(self, products: List[str]):
        self.products = products
        self.q = queue.Queue()
        self.ws = None
        self.thread = None
        self._stop = False

    def start(self):
        subs = {
            "type":"subscribe",
            "product_ids": self.products,
            "channels":[
                {"name":"matches","product_ids":self.products},
                {"name":"level2","product_ids":self.products}
            ]
        }
        def on_open(ws):
            ws.send(json.dumps(subs))

        def on_message(ws, msg):
            try:
                data = json.loads(msg)
                self.q.put(data)
            except Exception:
                pass

        def on_error(ws, err):
            print("[WS error]", err, file=sys.stderr)

        def on_close(ws, status, msg):
            print("[WS closed]", status, msg)

        def run():
            while not self._stop:
                try:
                    self.ws = WebSocketApp(
                        WS_URL, on_open=on_open, on_message=on_message, on_error=on_error, on_close=on_close
                    )
                    self.ws.run_forever(ping_interval=20, ping_timeout=10)
                except Exception as e:
                    print("WS run_forever exception:", e, file=sys.stderr)
                time.sleep(2)

        self.thread = threading.Thread(target=run, daemon=True)
        self.thread.start()

    def stop(self):
        self._stop = True
        try:
            self.ws and self.ws.close()
        except:
            pass

# -------------- Feature builders (live) --------------

def flow_from_trades(trades: List[dict], now_ts: datetime, window_sec: int) -> Tuple[int, float]:
    """Return (dir, strength) using real trades over window."""
    cutoff = now_ts - timedelta(seconds=window_sec)
    buys = sells = 0.0
    for t in trades[::-1]:
        if t["t"] < cutoff: break
        if t["side"] == "buy":  buys  += t["size"]
        else:                   sells += t["size"]
    tot = buys + sells
    if tot <= 0: return 0, 0.0
    p = buys / tot
    # simple strength (robust live): |p-0.5| * 2
    strength = clamp01(abs(p - 0.5) * 2.0)
    direction = 1 if p > 0.5 else -1 if p < 0.5 else 0
    return direction, strength

def update_vbp_bins(state: LiveState, price: float, side: str, size: float, cfg: Strat):
    """Update rolling volume-by-price bin by last trade price."""
    if price <= 0: return
    bin_w = price * cfg.bin_width_pct
    b = round(price/bin_w) * bin_w
    state.l2_bins[b] = state.l2_bins.get(b, 0.0) + size
    state.l2_prices.append(price)
    # prune old bins by time/len of prices
    if len(state.l2_prices) > cfg.vbp_lookback_min:  # ~one per sec if busy
        # approximate pruning: shrink dict by removing light tails
        if len(state.l2_bins) > 5000:
            # keep top bins
            items = sorted(state.l2_bins.items(), key=lambda x: x[1], reverse=True)[:2000]
            state.l2_bins = dict(items)
        state.l2_prices = state.l2_prices[-cfg.vbp_lookback_min:]

def cluster_scores_from_vbp(state: LiveState, price: float, cfg: Strat) -> Tuple[float, float]:
    if price <= 0 or not state.l2_bins: return (0.0, 0.0)
    prices = np.array(list(state.l2_bins.keys()), dtype=float)
    vols   = np.array(list(state.l2_bins.values()), dtype=float)
    dists  = (prices - price) / price
    above = dists > 0; below = dists < 0

    def nearest(mask):
        if not mask.any(): return (np.nan, np.nan)
        cand_d = np.abs(dists[mask])
        cand_v = vols[mask]
        w = (cand_d >= cfg.cluster_min_pct) & (cand_d <= cfg.cluster_max_pct)
        if not w.any(): return (np.nan, np.nan)
        i = np.argmin(cand_d[w])
        return float(cand_v[w][i]), float(cand_d[w][i])

    va, da = nearest(above); vb, db = nearest(below)
    def vol_rank(v):
        if not np.isfinite(v): return 0.0
        arr = vols[np.isfinite(vols)]
        if arr.size < 8: return 0.0
        return pct_rank(arr, v)

    dens_above = vol_rank(va)
    dens_below = vol_rank(vb)
    def prox_w(d):
        a, b = cfg.cluster_min_pct, cfg.cluster_max_pct
        if not np.isfinite(d) or d < a or d > b: return 0.0
        return clamp01(1.0 - (d - a)/(b - a))

    score_long  = clamp01(dens_below * prox_w(db))
    score_short = clamp01(dens_above * prox_w(da))
    return score_long, score_short

def compression_score(df15: pd.DataFrame, df1h: pd.DataFrame) -> pd.Series:
    bbw1h = bb_width_pct(df1h["close"]).reindex(df15.index, method='ffill')
    bbw15 = bb_width_pct(df15["close"])
    atrp15= (atr_from_df(df15)/df15["close"]).replace([np.inf,-np.inf], np.nan)
    def pctr(s, look=90):
        return s.rolling(look, min_periods=look).apply(lambda a: pct_rank(np.asarray(a), a[-1]), raw=True)
    comp = 1.0 - pd.concat([pctr(bbw1h), pctr(bbw15), pctr(atrp15)], axis=1).mean(axis=1)
    return comp.clip(0,1)

def swarm_scores(comp, flow_dir, flow_strength, cl_long, cl_short):
    return (
        clamp01(0.40*comp + 0.35*flow_strength*(1 if flow_dir==1 else 0) + 0.25*cl_long),
        clamp01(0.40*comp + 0.35*flow_strength*(1 if flow_dir==-1 else 0) + 0.25*cl_short)
    )

# -------------- Telegram -------------

def send_telegram(token: str, chat_id: str, text: str):
    try:
        requests.post(f"https://api.telegram.org/bot{token}/sendMessage",
                      json={"chat_id": chat_id, "text": text, "parse_mode":"Markdown"},
                      timeout=10)
    except Exception as e:
        print("[TG ERROR]", e, file=sys.stderr)

# -------------- Main live loop --------------

def live_run(symbols: List[str], strat: Strat, entry_mode: str, use_telegram: bool, token: str, chat_id: str):
    # per-symbol state
    S: Dict[str, LiveState] = {s: LiveState() for s in symbols}
    # prime candles
    end = utcnow().replace(second=0, microsecond=0)
    start_1m  = end - timedelta(days=2)
    start_15m = end - timedelta(days=14)
    start_1h  = end - timedelta(days=60)
    for sym in symbols:
        S[sym].candles_1m  = fetch_candles(sym, start_1m, end, 60)
        S[sym].candles_15m = fetch_candles(sym, start_15m, end, 900)
        S[sym].candles_1h  = fetch_candles(sym, start_1h, end, 3600)
        time.sleep(0.2)

    ws = WSGroup(symbols); ws.start()
    print("[LIVE] Started. Shadow mode ON." + ("" if not use_telegram else " Telegram ON."))

    try:
        last_rest_pull = {s: {"1m":end, "15m":end, "1h":end} for s in symbols}
        while True:
            # drain websocket
            try:
                while True:
                    msg = ws.q.get_nowait()
                    if "type" not in msg: continue
                    tnow = utcnow()
                    if msg["type"] == "match":
                        sym = msg.get("product_id")
                        if sym not in S: continue
                        side = msg.get("side")
                        price = float(msg.get("price", 0))
                        size  = float(msg.get("size", 0))
                        S[sym].trades.append({"t": tnow, "side": side, "price": price, "size": size})
                        update_vbp_bins(S[sym], price, side, size, strat)
                    elif msg["type"] in ("snapshot","l2update"):
                        # optional: you can incorporate book sizes here if desired
                        pass
            except queue.Empty:
                pass

            # periodic REST refresh for candles (once per minute)
            now = utcnow().replace(second=0, microsecond=0)
            for sym in symbols:
                # only refresh at new minute
                if now > last_rest_pull[sym]["1m"]:
                    df1m = fetch_candles(sym, now - timedelta(hours=3), now, 60)  # recent 3h 1m
                    if not df1m.empty:
                        S[sym].candles_1m = pd.concat([S[sym].candles_1m, df1m]).sort_index()
                        S[sym].candles_1m = S[sym].candles_1m[~S[sym].candles_1m.index.duplicated(keep="last")].last("2D")
                    last_rest_pull[sym]["1m"] = now

                if now.minute % 15 == 0 and now > last_rest_pull[sym]["15m"]:
                    df15 = fetch_candles(sym, now - timedelta(days=3), now, 900)
                    if not df15.empty:
                        S[sym].candles_15m = pd.concat([S[sym].candles_15m, df15]).sort_index()
                        S[sym].candles_15m = S[sym].candles_15m[~S[sym].candles_15m.index.duplicated(keep="last")].last("14D")
                    last_rest_pull[sym]["15m"] = now

                if now.hour != last_rest_pull[sym]["1h"].hour:
                    df1h = fetch_candles(sym, now - timedelta(days=10), now, 3600)
                    if not df1h.empty:
                        S[sym].candles_1h = pd.concat([S[sym].candles_1h, df1h]).sort_index()
                        S[sym].candles_1h = S[sym].candles_1h[~S[sym].candles_1h.index.duplicated(keep="last")].last("60D")
                    last_rest_pull[sym]["1h"] = now

            # evaluate signals every 5 seconds (intrabar stop-style possible)
            for sym in symbols:
                st = S[sym]
                if st.candles_15m.empty or st.candles_1h.empty: continue
                price = float(st.candles_1m["close"].iloc[-1]) if not st.candles_1m.empty else None
                if not price: continue

                # Features
                comp_s = compression_score(st.candles_15m, st.candles_1h).iloc[-1]
                fdir, fstr = flow_from_trades(st.trades, utcnow(), strat.clv_window_sec)
                cl_long, cl_short = cluster_scores_from_vbp(st, price, strat)
                long_score, short_score = swarm_scores(comp_s, fdir, fstr, cl_long, cl_short)
                # Box / ATR from 15m
                box_hi, box_lo = donchian(st.candles_15m, strat.donchian_period)
                atr15 = atr_from_df(st.candles_15m, strat.atr_period).iloc[-1]
                boxhi = float(box_hi.iloc[-1]); boxlo = float(box_lo.iloc[-1])
                if not np.isfinite(boxhi) or not np.isfinite(boxlo) or not np.isfinite(atr15): continue

                # Momentum gate
                gate_ok_long = True; gate_ok_short = True
                if strat.stoch_gate:
                    k, d = stoch_rsi(st.candles_15m["close"])
                    kk = float(k.iloc[-1]); dd = float(d.iloc[-1])
                    gate_ok_long  = (kk > dd) and (kk < 40.0)
                    gate_ok_short = (kk < dd) and (kk > 60.0)

                # Cooldown
                cooldown_ok = True
                if st.last_alert_at and (utcnow() - st.last_alert_at) < timedelta(minutes=strat.cooldown_min):
                    cooldown_ok = False

                # Directional pre-checks
                pre_long = (long_score >= strat.swarm_thresh and comp_s >= strat.min_comp and
                            fstr >= strat.min_flow and cl_long >= strat.min_cluster and gate_ok_long)
                pre_short= (short_score>= strat.swarm_thresh and comp_s >= strat.min_comp and
                            fstr >= strat.min_flow and cl_short>= strat.min_cluster and gate_ok_short)

                # Breakout (stop vs close)
                trig_long  = (price > (boxhi + strat.breakout_buffer_atr*atr15)) if entry_mode=="stop"  else False
                trig_short = (price < (boxlo - strat.breakout_buffer_atr*atr15)) if entry_mode=="stop"  else False

                if cooldown_ok and pre_long and trig_long:
                    st.last_alert_at = utcnow()
                    text = (f"SWARM LIVE — LONG {sym}\n"
                            f"Price: {price:.8f}\n"
                            f"BoxHi: {boxhi:.8f}  ATR15: {atr15:.8f}\n"
                            f"Scores: comp {comp_s:.2f}, flow {fstr:.2f}, cl {cl_long:.2f}, swarm {long_score:.2f}\n"
                            f"Time: {utcnow().strftime('%Y-%m-%d %H:%M:%S %Z')}")
                    print(text, flush=True)
                    if use_telegram: send_telegram(TG_TOKEN, TG_CHAT, text)

                if cooldown_ok and pre_short and trig_short:
                    st.last_alert_at = utcnow()
                    text = (f"SWARM LIVE — SHORT {sym}\n"
                            f"Price: {price:.8f}\n"
                            f"BoxLo: {boxlo:.8f}  ATR15: {atr15:.8f}\n"
                            f"Scores: comp {comp_s:.2f}, flow {fstr:.2f}, cl {cl_short:.2f}, swarm {short_score:.2f}\n"
                            f"Time: {utcnow().strftime('%Y-%m-%d %H:%M:%S %Z')}")
                    print(text, flush=True)
                    if use_telegram: send_telegram(TG_TOKEN, TG_CHAT, text)

            time.sleep(5)

    except KeyboardInterrupt:
        print("\n[EXIT] Stopping…")
    except Exception:
        traceback.print_exc()
    finally:
        ws.stop()

# -------------- CLI --------------

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Swarm Phase LIVE (shadow/alert)")
    ap.add_argument("--symbols", nargs="+", help="Symbols (e.g. BTC-USD ETH-USD). Optional if --symbols-file is used.")
    ap.add_argument("--symbols-file", type=str, help="Path to a text file with one symbol per line")
    ap.add_argument("--entry-mode", choices=["stop","close"], default="stop")
    ap.add_argument("--telegram", action="store_true", help="Send Telegram alerts (otherwise shadow/log only)")
    # thresholds
    ap.add_argument("--thresh", type=float, default=0.60)
    ap.add_argument("--min-flow", type=float, default=0.35)
    ap.add_argument("--min-cluster", type=float, default=0.30)
    ap.add_argument("--min-comp", type=float, default=0.55)
    ap.add_argument("--buffer", type=float, default=0.25)
    ap.add_argument("--donchian", type=int, default=14)
    ap.add_argument("--cooldown", type=int, default=45)
    ap.add_argument("--no-stoch-gate", action="store_true")
    args = ap.parse_args()

    # load symbols
    if args.symbols_file:
        with open(args.symbols_file, "r", encoding="utf-8") as f:
            symbols = [line.strip() for line in f if line.strip()]
    elif args.symbols:
        symbols = args.symbols
    else:
        print("Error: must provide either --symbols or --symbols-file", file=sys.stderr)
        sys.exit(1)

    # normalize to Coinbase style (BTC-USD etc.)
    symbols = normalize_products(symbols)

    # Telegram creds
    TG_TOKEN = "8177096945:AAHTg5nxVTA6hcPkAL4MLAAHPsZ1at7Ywmw"
    TG_CHAT  = "7967738614"

    s = Strat(
        swarm_thresh=args.thresh, min_comp=args.min_comp, min_flow=args.min_flow,
        min_cluster=args.min_cluster, breakout_buffer_atr=args.buffer,
        donchian_period=args.donchian, cooldown_min=args.cooldown,
        stoch_gate=(not args.no_stoch_gate)
    )
    print(f"[CFG] {s} entry_mode={args.entry_mode} telegram={args.telegram} symbols={symbols}")
    live_run(symbols, s, args.entry_mode, args.telegram, TG_TOKEN, TG_CHAT)

