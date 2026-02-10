# pip install websocket-client requests numpy
import json, time, threading, math, collections
from datetime import datetime, timezone
import numpy as np
import requests
import websocket

WS_URL = "wss://advanced-trade-ws.coinbase.com"

# ========= USER SETTINGS =========
PRODUCTS = ["SUI-USD", "SEI-USD", "XRP-USD", "crv-usd"]

ROLLING_WINDOW_SEC = 600            # window for dynamic percentile
LARGE_TRADE_PERCENTILE = 95         # size threshold (dynamic)
BURST_COUNT = 3                     # >= N large prints...
BURST_WINDOW_MS = 700               # ...within Y ms

# Liquidity clustering
CLUSTER_BIN_PCT = 0.001             # bin width ~ 0.1% of price
CLUSTER_TOP_PERCENTILE = 97         # consider only top clusters by qty (>= this pct)
CLUSTER_NEARBY_PCT = 0.01           # must be within 1% of price to be "nearby"

# Trade plan
SL_BUFFER_PCT = 0.0015              # push SL ~0.15% beyond cluster
MIN_TP_RR = 1.5                     # if no next cluster found, use fallback RR (e.g., 2.0 = 2R)
TELEGRAM_BOT_TOKEN = "8177096945:AAHTg5nxVTA6hcPkAL4MLAAHPsZ1at7Ywmw"
TELEGRAM_CHAT_ID = "7967738614"

ALERT_COOLDOWN_SEC = 180            # per product cooldown
# =================================

def now_ts(): return time.time()
def fmt_ts(ts=None):
    if ts is None: ts = now_ts()
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

def dyn_decimals(px):
    if px >= 100: return 2
    if px >= 10:  return 3
    if px >= 1:   return 4
    if px >= 0.1: return 5
    return 6

def fmt_price(px):
    d = dyn_decimals(px)
    return f"{px:.{d}f}"

def bin_width(price): return max(1e-10, price * CLUSTER_BIN_PCT)
def bin_price(price):
    bw = bin_width(price)
    return round(price / bw) * bw

# ----------------- STATE -----------------
last_price = {p: None for p in PRODUCTS}
trade_buffer = {p: collections.deque() for p in PRODUCTS}      # (ts, size_base, signed_size)
large_trade_times = {p: collections.deque() for p in PRODUCTS} # ts of large prints for burst
last_alert_time = {p: 0 for p in PRODUCTS}

# side-separated clusters
# clusters[product]["bid"][bin_price] = cumulative_qty
# clusters[product]["ask"][bin_price] = cumulative_qty
clusters = {p: {"bid": collections.defaultdict(float),
                "ask": collections.defaultdict(float)} for p in PRODUCTS}
# -----------------------------------------

def prune_old_trades(product):
    cutoff = now_ts() - ROLLING_WINDOW_SEC
    dq = trade_buffer[product]
    while dq and dq[0][0] < cutoff:
        dq.popleft()

def percentile_threshold(product):
    prune_old_trades(product)
    sizes = [abs(s) for (_, s, _) in trade_buffer[product]]
    if len(sizes) < 20:
        return None
    return float(np.percentile(sizes, LARGE_TRADE_PERCENTILE))

def top_clusters(product, side):
    """Return list of (bin_px, qty) for clusters above the percentile threshold."""
    items = list(clusters[product][side].items())
    if not items:
        return []
    qtys = [q for _, q in items]
    if not qtys:
        return []
    thr = np.percentile(qtys, CLUSTER_TOP_PERCENTILE)
    return [(bp, q) for (bp, q) in items if q >= thr]

def nearest_cluster(product, price, side, within_pct=CLUSTER_NEARBY_PCT):
    """Nearest top cluster on the given side within 'within_pct' of price."""
    tcs = top_clusters(product, side)
    if not tcs or price is None:
        return None
    best = None
    for bp, q in tcs:
        dist = abs(bp - price) / price
        if dist <= within_pct:
            cand = (dist, bp, q)
            if (best is None) or (cand[0] < best[0]):
                best = cand
    if best:
        _, bp, q = best
        return (bp, q)
    return None

def next_tp_cluster(product, price, direction):
    """
    Long -> next ASK cluster ABOVE price.
    Short -> next BID cluster BELOW price.
    Preference: nearest top cluster in that direction; fallback to globally strongest in that direction.
    """
    if price is None:
        return None
    if direction == "long":
        side = "ask"
        candidates = [(bp, q) for (bp, q) in top_clusters(product, side) if bp > price]
    else:
        side = "bid"
        candidates = [(bp, q) for (bp, q) in top_clusters(product, side) if bp < price]

    if candidates:
        # nearest in price
        if direction == "long":
            bp, q = sorted(candidates, key=lambda x: x[0])[0]
        else:
            bp, q = sorted(candidates, key=lambda x: -x[0])[0]
        return (bp, q, side)

    # Fallback: strongest cluster on that side (anywhere)
    tcs = top_clusters(product, side)
    if not tcs:
        return None
    bp, q = sorted(tcs, key=lambda x: x[1], reverse=True)[0]
    return (bp, q, side)

def send_telegram(text):
    try:
        requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
            json={"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": "HTML"},
            timeout=10,
        )
    except Exception as e:
        print("Telegram error:", e)

def propose_trade_plan(product, flow_bias, price, near_bid, near_ask):
    """
    Decide LONG or SHORT based on flow + nearby cluster side.
    Return dict with plan or None.
    """
    if price is None:
        return None

    # LONG: Aggressive BUY and nearby BID cluster below price (support)
    if flow_bias == "Aggressive BUY" and near_bid:
        c_px, c_qty = near_bid
        if c_px <= price:
            entry = price
            # SL just beyond the bid cluster
            sl = c_px * (1 - SL_BUFFER_PCT)
            risk = entry - sl
            if risk <= 0:
                return None
            tp_info = next_tp_cluster(product, entry, "long")
            if tp_info:
                tp_px, tp_qty, _ = tp_info
                if tp_px <= entry:
                    # sanity: TP must be above entry; otherwise fallback
                    tp = entry + MIN_TP_RR * risk
                    tp_note = f"Fallback {MIN_TP_RR:.2f}R (no valid ask cluster above)"
                else:
                    tp = tp_px  # magnet
                    tp_note = f"Next ask cluster @ {fmt_price(tp_px)}"
            else:
                tp = entry + MIN_TP_RR * risk
                tp_note = f"Fallback {MIN_TP_RR:.2f}R (no ask clusters)"
            rr = (tp - entry) / risk
            return {
                "side": "LONG",
                "entry": entry,
                "sl": sl,
                "tp": tp,
                "rr": rr,
                "anchor_cluster": ("bid", c_px, c_qty),
                "tp_note": tp_note
            }

    # SHORT: Aggressive SELL and nearby ASK cluster above price (resistance)
    if flow_bias == "Aggressive SELL" and near_ask:
        c_px, c_qty = near_ask
        if c_px >= price:
            entry = price
            # SL just beyond the ask cluster
            sl = c_px * (1 + SL_BUFFER_PCT)
            risk = sl - entry
            if risk <= 0:
                return None
            tp_info = next_tp_cluster(product, entry, "short")
            if tp_info:
                tp_px, tp_qty, _ = tp_info
                if tp_px >= entry:
                    tp = entry - MIN_TP_RR * risk
                    tp_note = f"Fallback {MIN_TP_RR:.2f}R (no valid bid cluster below)"
                else:
                    tp = tp_px
                    tp_note = f"Next bid cluster @ {fmt_price(tp_px)}"
            else:
                tp = entry - MIN_TP_RR * risk
                tp_note = f"Fallback {MIN_TP_RR:.2f}R (no bid clusters)"
            rr = (entry - tp) / risk
            return {
                "side": "SHORT",
                "entry": entry,
                "sl": sl,
                "tp": tp,
                "rr": rr,
                "anchor_cluster": ("ask", c_px, c_qty),
                "tp_note": tp_note
            }

    return None

def handle_market_trades(product, trades):
    """
    Coinbase 'market_trades' uses maker side semantics:
      maker=SELL -> taker bought (Aggressive BUY)  -> positive delta
      maker=BUY  -> taker sold  (Aggressive SELL)  -> negative delta
    """
    ts_now = now_ts()
    burst_times = large_trade_times[product]

    for t in trades:
        price = float(t["price"])
        size = float(t["size"])
        side = t.get("side", "BUY")   # maker side; 'SELL' means taker bought
        signed = size if side == "SELL" else -size

        last_price[product] = price
        trade_buffer[product].append((ts_now, size, signed))
        prune_old_trades(product)

        th = percentile_threshold(product)
        if th is not None and size >= th:
            burst_times.append(ts_now)

    # prune burst window
    while burst_times and (ts_now - burst_times[0]) * 1000 > BURST_WINDOW_MS:
        burst_times.popleft()

    if len(burst_times) >= BURST_COUNT and (ts_now - last_alert_time[product]) > ALERT_COOLDOWN_SEC:
        entry_px = last_price[product]
        # flow bias from last few prints in the burst window
        recent_signed = [sg for (_, _, sg) in list(trade_buffer[product])[-BURST_COUNT:]]
        flow_bias = "Aggressive BUY" if sum(recent_signed) > 0 else "Aggressive SELL"

        # nearby clusters (top-ranked only)
        near_bid = nearest_cluster(product, entry_px, "bid")
        near_ask = nearest_cluster(product, entry_px, "ask")

        plan = propose_trade_plan(product, flow_bias, entry_px, near_bid, near_ask)

        # Compose alert
        msg = [
            f"<b>Large-Order Burst</b> on <b>{product}</b> @ {fmt_ts()}",
            f"Last price: <b>{fmt_price(entry_px)}</b>",
            f"Burst: <b>{len(burst_times)}</b> large prints in ≤ {BURST_WINDOW_MS} ms",
            f"Flow bias: <b>{flow_bias}</b>"
        ]

        if near_bid:
            dist = abs(near_bid[0] - entry_px) / entry_px * 100
            msg.append(f"Nearby <b>bid cluster</b>: {fmt_price(near_bid[0])} (~{dist:.2f}% away)")
        if near_ask:
            dist = abs(near_ask[0] - entry_px) / entry_px * 100
            msg.append(f"Nearby <b>ask cluster</b>: {fmt_price(near_ask[0])} (~{dist:.2f}% away)")

        if plan:
            side = plan["side"]
            rr = plan["rr"]
            entry = plan["entry"]; sl = plan["sl"]; tp = plan["tp"]
            anch_side, anch_px, anch_qty = plan["anchor_cluster"]
            msg += [
                "",
                f"<b>Trade Plan</b> ({side})",
                f"Entry: <b>{fmt_price(entry)}</b>",
                f"SL: <b>{fmt_price(sl)}</b>  (beyond {anch_side} cluster @ {fmt_price(anch_px)})",
                f"TP: <b>{fmt_price(tp)}</b>  ({plan['tp_note']})",
                f"Est. R:R ≈ <b>{rr:.2f}</b>"
            ]
        else:
            msg += ["", "No valid trade plan (flow/cluster side didn’t align)."]

        send_telegram("\n".join(msg))
        last_alert_time[product] = ts_now
        burst_times.clear()

def handle_level2(product, updates):
    """
    Maintain side-separated binned clusters from absolute quantities.
    If new_quantity = 0 at a price level, decay that bin (so stale clusters fade).
    """
    book = clusters[product]
    for u in updates:
        price = float(u["price_level"])
        qty = float(u["new_quantity"])
        side = u.get("side", "").lower()
        side = "bid" if side.startswith("b") else "ask"
        b = bin_price(price)

        if qty <= 0:
            # light decay on removal so bins can clear
            book[side][b] = max(0.0, book[side].get(b, 0.0) * 0.8)
        else:
            # keep the max resting qty seen for that bin (acts as a magnet)
            book[side][b] = max(book[side].get(b, 0.0), qty)

    # global gentle decay over time
    for s in ("bid", "ask"):
        for bp in list(book[s].keys()):
            book[s][bp] *= 0.999
            if book[s][bp] < 1e-9:
                del book[s][bp]

# ------------- WS WIRING -------------
def on_open(ws):
    ws.send(json.dumps({"type": "subscribe", "channel": "heartbeats"}))
    ws.send(json.dumps({"type": "subscribe", "channel": "market_trades", "product_ids": PRODUCTS}))
    ws.send(json.dumps({"type": "subscribe", "channel": "level2", "product_ids": PRODUCTS}))

def on_message(ws, message):
    try:
        data = json.loads(message)
    except Exception:
        return
    ch = data.get("channel")
    events = data.get("events", [])
    if ch == "market_trades":
        by_product = {}
        for ev in events:
            if ev.get("type") in ("snapshot", "update"):
                for t in ev.get("trades", []):
                    by_product.setdefault(t["product_id"], []).append(t)
        for pid, arr in by_product.items():
            handle_market_trades(pid, arr)
    elif ch in ("level2", "l2_data"):
        for ev in events:
            pid = ev.get("product_id")
            if not pid: continue
            handle_level2(pid, ev.get("updates", []))
    # heartbeats ignored

def on_error(ws, err): print("WS error:", err)
def on_close(ws, code, msg): print("WS closed:", code, msg)

def run_ws():
    while True:
        try:
            ws = websocket.WebSocketApp(
                WS_URL,
                on_open=on_open, on_message=on_message,
                on_error=on_error, on_close=on_close
            )
            ws.run_forever(ping_interval=20, ping_timeout=10)
        except Exception as e:
            print("WS exception, reconnecting in 3s:", e); time.sleep(3)

if __name__ == "__main__":
    print("Starting Vivian-style Coinbase watcher v2…")
    print("Products:", PRODUCTS)
    run_ws()
