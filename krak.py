#!/usr/bin/env python3
# Kraken spot flow tester: ΔCVD (60s) + near-book notional (±0.5%) via WebSocket
# pip install websocket-client
import json, time, threading, collections, math
from websocket import WebSocketApp

PAIRS = ["XBT/USD","ETH/USD","SOL/USD","XRP/USD","MATIC/USD"]  # adjust as you like
NEAR_PCT = 0.005     # ±0.5% window around mid
WINDOW_S = 60        # ΔCVD rolling window
PRINT_EVERY = 5      # seconds

# Rolling trade buffer per pair
Trade = collections.namedtuple("Trade", "ts side price vol notional")
trades = {p: collections.deque(maxlen=20000) for p in PAIRS}
tlock  = {p: threading.Lock() for p in PAIRS}

# Shallow book store per pair
books = {p: {"bids": [], "asks": [], "mid": None} for p in PAIRS}
block  = {p: threading.Lock() for p in PAIRS}

def _to_float(x):
    try: return float(x)
    except: return 0.0

def on_message(ws, msg):
    try:
        data = json.loads(msg)
    except:
        return
    # Heartbeats / events
    if isinstance(data, dict):
        # subscription status, heartbeat, etc.
        return
    if not isinstance(data, list) or len(data) < 2:
        return

    # Kraken WS public messages are: [channelID, payload, channelName, pair]
    # We rely on channelName and pair at the end.
    chan = data[-2] if len(data) >= 3 else ""
    pair = data[-1] if len(data) >= 4 else ""

    if chan == "trade" and isinstance(data[1], list):
        # payload: list of trades: ["price","volume","time","side","ordertype","misc"]
        for t in data[1]:
            px  = _to_float(t[0]); vol = _to_float(t[1])   # base volume
            side= t[3]  # "b"=buy, "s"=sell (taker side)
            tm  = float(t[2])  # epoch seconds float
            notional = px * vol
            with tlock.get(pair, threading.Lock()):
                if pair in trades:
                    trades[pair].append(Trade(tm, side, px, vol, notional))
    elif isinstance(data[1], dict) and (chan.startswith("book")):
        payload = data[1]
        with block.get(pair, threading.Lock()):
            bk = books.get(pair)
            if bk is None: return
            # Snapshot
            if "as" in payload or "bs" in payload:
                asks = payload.get("as", [])
                bids = payload.get("bs", [])
                # asks: [price, volume, timestamp]
                bk["asks"] = [(_to_float(a[0]), _to_float(a[1])) for a in asks][:200]
                bk["bids"] = [(_to_float(b[0]), _to_float(b[1])) for b in bids][:200]
            # Updates
            if "a" in payload:
                for a in payload["a"]:
                    px = _to_float(a[0]); vol = _to_float(a[1])
                    # update asks
                    # remove if vol==0; else set
                    bk["asks"] = [(p,v) for (p,v) in bk["asks"] if p != px]
                    if vol > 0: bk["asks"].append((px, vol))
                # keep sorted
                bk["asks"].sort(key=lambda x: x[0])
                bk["asks"] = bk["asks"][:200]
            if "b" in payload:
                for b in payload["b"]:
                    px = _to_float(b[0]); vol = _to_float(b[1])
                    bk["bids"] = [(p,v) for (p,v) in bk["bids"] if p != px]
                    if vol > 0: bk["bids"].append((px, vol))
                # sort desc
                bk["bids"].sort(key=lambda x: x[0], reverse=True)
                bk["bids"] = bk["bids"][:200]
            # Mid
            if bk["bids"] and bk["asks"]:
                bk["mid"] = 0.5 * (bk["bids"][0][0] + bk["asks"][0][0])

def on_open(ws):
    # Subscribe to trades
    ws.send(json.dumps({
        "event": "subscribe",
        "pair": PAIRS,
        "subscription": {"name": "trade"}
    }))
    # Subscribe to order book (depth 100)
    ws.send(json.dumps({
        "event": "subscribe",
        "pair": PAIRS,
        "subscription": {"name": "book", "depth": 100}
    }))

def on_error(ws, err):
    print("[WS error]", err)

def on_close(ws, *args):
    print("[WS closed]")

def cvd_window(pair, now_s):
    cutoff = now_s - WINDOW_S
    buys = sells = 0.0
    n=0
    with tlock[pair]:
        for t in reversed(trades[pair]):
            if t.ts < cutoff: break
            n += 1
            if t.side == "b": buys += t.notional
            else:             sells += t.notional
    return buys, sells, buys - sells, n

def near_book_notional(pair):
    with block[pair]:
        bk = books[pair]
        mid = bk["mid"]
        if not mid: return 0.0, 0.0, None
        lo, hi = mid*(1-NEAR_PCT), mid*(1+NEAR_PCT)
        def tot(levels):
            s=0.0
            for px, vol in levels:
                if lo <= px <= hi: s += px*vol
            return s
        return tot(bk["bids"]), tot(bk["asks"]), mid

def printer():
    while True:
        rows=[]
        now = time.time()
        for p in PAIRS:
            b, s, d, n = cvd_window(p, now)
            nb, na, mid = near_book_notional(p)
            if mid is None: continue
            rows.append((p, d, nb, na, mid))
        rows.sort(key=lambda r: abs(r[1]), reverse=True)
        print("\n[Kraken Debug candidates]")
        for p, d, nb, na, mid in rows[:5]:
            # compact money print
            def fmt(x):
                x = float(x); a=abs(x)
                if a>=1e9: s=f"{x/1e9:.2f}B"
                elif a>=1e6: s=f"{x/1e6:.2f}M"
                elif a>=1e3: s=f"{x/1e3:.2f}K"
                else: s=f"{x:.0f}"
                return s.rstrip("0").rstrip(".")
            print(f"  {p:8s}: ΔCVD={fmt(d):>6}  nearB={fmt(nb):>6}  nearA={fmt(na):>6}  mid={mid:.5f}")
        time.sleep(PRINT_EVERY)

if __name__ == "__main__":
    ws = WebSocketApp(
        "wss://ws.kraken.com",
        on_open=on_open,
        on_message=on_message,
        on_error=on_error,
        on_close=on_close,
    )
    t = threading.Thread(target=ws.run_forever, kwargs={"ping_interval":20,"ping_timeout":10}, daemon=True)
    t.start()
    # give it a moment to subscribe
    time.sleep(2.0)
    printer()
