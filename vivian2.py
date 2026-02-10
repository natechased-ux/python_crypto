# vivian_coinbase_alerts_v3.py
# pip install websocket-client requests numpy
import json, time, math, collections
from datetime import datetime, timezone
import numpy as np
import requests
import websocket

WS_URL = "wss://advanced-trade-ws.coinbase.com"

# ========= USER SETTINGS =========
PRODUCTS = ["SUI-USD", "SEI-USD", "XRP-USD", "crv-usd","ltc-usd","ada-usd","doge-usd",
    "sol-usd","wif-usd","ondo-usd"]  # add/remove freely

# Burst detection (quality gate)
LARGE_TRADE_PERCENTILE = 95       # dynamic size threshold vs last N minutes
ROLLING_WINDOW_SEC = 600          # window for percentile stats (10m)
BURST_COUNT = 4                   # require >= N large prints in window
BURST_WINDOW_MS = 700             # within Y ms
MIN_BURST_NOTIONAL_USD = 0        # set >0 to demand min burst notional (e.g., 250000)

# Core “bigger swing” constraints
MIN_RR = 1.6                      # skip if projected R:R < MIN_RR
USE_ATR = True                    # use ATR floors to avoid micro moves
ATR_LEN = 120                     # samples for pseudo-ATR on last trade prices
MIN_RISK_ATR = 0.18               # SL >= 0.18 * ATR
MIN_TP_ATR   = 0.36               # TP >= 0.36 * ATR

# Cluster logic (adaptive baseline; will be scaled per-asset via ATR%)
BASE_CLUSTER_BIN_PCT = 0.0018     # ~0.18% of price (merged by ATR profile)
CLUSTER_TOP_PERCENTILE = 97       # consider only top ≥ percentile clusters
CLUSTER_NEARBY_PCT = 0.012        # cluster must be within 1.2% of price to be "nearby"
SL_BUFFER_PCT = 0.0015            # push SL ~0.15% beyond anchor cluster

# Alert verbosity
ONLY_ALERT_ON_VIABLE = True   # if True, send Telegram ONLY when a trade plan passes all filters
SEND_NONVIABLE_SUMMARY = False  # if True and ONLY_ALERT_ON_VIABLE is False, send a short "skipped" note


# TP distance gate (absolute % floor; adaptive per-asset too)
BASE_MIN_TP_PCT = 0.006           # 0.6% min TP distance; will be scaled by ATR%

# Telegram
TELEGRAM_BOT_TOKEN = "8177096945:AAHTg5nxVTA6hcPkAL4MLAAHPsZ1at7Ywmw"
TELEGRAM_CHAT_ID = "7967738614"

# Cooldown per product to avoid spam
ALERT_COOLDOWN_SEC = 240
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

# ----------------- STATE -----------------
last_price = {p: None for p in PRODUCTS}
trade_buffer = {p: collections.deque() for p in PRODUCTS}      # (ts, price, size_base, signed_size, notional)
large_trade_times = {p: collections.deque() for p in PRODUCTS} # timestamps of large prints
burst_notional = {p: collections.deque() for p in PRODUCTS}    # notional per large print in burst window
last_alert_time = {p: 0 for p in PRODUCTS}

# side-separated clusters
clusters = {p: {"bid": collections.defaultdict(float),
                "ask": collections.defaultdict(float)} for p in PRODUCTS}

# price ring for pseudo-ATR
price_ring = {p: collections.deque(maxlen=ATR_LEN) for p in PRODUCTS}
# -----------------------------------------

# ---------- Adaptive per-asset profile ----------
def update_price_ring(product, px):
    if px is None: return
    price_ring[product].append(px)

def get_atr_abs(product):
    dq = price_ring[product]
    if len(dq) < 6:
        return None
    diffs = [abs(dq[i] - dq[i-1]) for i in range(1, len(dq))]
    if not diffs:
        return None
    return float(sum(diffs) / len(diffs))

def get_atr_pct(product):
    px = last_price.get(product)
    atr = get_atr_abs(product)
    if px and atr:
        return atr / px  # e.g., 0.007 = 0.7%
    return None

def asset_profile(product):
    """
    Returns a dict with adaptive parameters scaled by current ATR%.
    Profiles:
      - Low vol (atr% < 0.4%): tighter bin, smaller TP floor
      - Med vol (0.4%–1.2%): baseline
      - High vol (> 1.2%): chunkier bins, larger TP floor, stricter burst
    """
    atrp = get_atr_pct(product) or 0.006  # assume ~0.6% if unknown
    if atrp < 0.004:  # <0.4%
        profile = dict(
            CL_BIN = BASE_CLUSTER_BIN_PCT * 0.8,
            MIN_TP = BASE_MIN_TP_PCT * 0.8,
            BURST_N = max(3, BURST_COUNT - 1),
            MIN_BURST_NOTIONAL = max(0, MIN_BURST_NOTIONAL_USD * 0.75)
        )
    elif atrp > 0.012:  # >1.2%
        profile = dict(
            CL_BIN = BASE_CLUSTER_BIN_PCT * 1.6,
            MIN_TP = BASE_MIN_TP_PCT * 1.5,
            BURST_N = BURST_COUNT + 1,
            MIN_BURST_NOTIONAL = max(MIN_BURST_NOTIONAL_USD, 0)
        )
    else:
        profile = dict(
            CL_BIN = BASE_CLUSTER_BIN_PCT,
            MIN_TP = BASE_MIN_TP_PCT,
            BURST_N = BURST_COUNT,
            MIN_BURST_NOTIONAL = MIN_BURST_NOTIONAL_USD
        )
    profile["ATR_PCT"] = atrp
    return profile

# --------------- Helpers ---------------
def prune_old_trades(product):
    cutoff = now_ts() - ROLLING_WINDOW_SEC
    dq = trade_buffer[product]
    while dq and dq[0][0] < cutoff:
        dq.popleft()

def percentile_threshold(product):
    prune_old_trades(product)
    sizes = [abs(s) for (_, _, _, sgn, _) in trade_buffer[product] for s in [sgn] if True]  # signed_size magnitude
    # (Using signed_size magnitude here because it's size_base; either is fine as absolute size)
    sizes = [abs(sgn) for (_, _, _, sgn, _) in trade_buffer[product]]
    if len(sizes) < 20:
        return None
    return float(np.percentile(sizes, LARGE_TRADE_PERCENTILE))

def bin_width(product, price):
    prof = asset_profile(product)
    return max(1e-10, price * prof["CL_BIN"])

def bin_price(product, price):
    bw = bin_width(product, price)
    return round(price / bw) * bw

def top_clusters(product, side):
    items = list(clusters[product][side].items())
    if not items:
        return []
    qtys = [q for _, q in items]
    if not qtys:
        return []
    thr = np.percentile(qtys, CLUSTER_TOP_PERCENTILE)
    return [(bp, q) for (bp, q) in items if q >= thr]

def nearest_cluster(product, price, side, within_pct):
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

def next_tp_cluster_with_min_distance(product, price, direction, min_pct):
    """Pick nearest opposite-side cluster at least min_pct away; else None."""
    if price is None:
        return None
    if direction == "long":
        side = "ask"
        cands = [(bp, q) for (bp, q) in top_clusters(product, side) if bp > price]
        cands.sort(key=lambda x: x[0])
        for bp, q in cands:
            if (bp - price) / price >= min_pct:
                return (bp, q, side)
    else:
        side = "bid"
        cands = [(bp, q) for (bp, q) in top_clusters(product, side) if bp < price]
        cands.sort(key=lambda x: -x[0])
        for bp, q in cands:
            if (price - bp) / price >= min_pct:
                return (bp, q, side)
    return None

def send_telegram(text):
    try:
        requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
            json={"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": "HTML"},
            timeout=10,
        )
    except Exception as e:
        print("Telegram error:", e)

# --------------- Trade Plan ---------------
def propose_trade_plan(product, flow_bias, price, near_bid, near_ask):
    if price is None:
        return None
    prof = asset_profile(product)
    min_tp_pct = prof["MIN_TP"]

    # LONG plan
    if flow_bias == "Aggressive BUY" and near_bid:
        c_px, c_qty = near_bid
        if c_px <= price:
            entry = price
            sl = c_px * (1 - SL_BUFFER_PCT)
            risk = max(entry - sl, 1e-12)

            # ATR floors
            if USE_ATR:
                atr_abs = get_atr_abs(product)
                if atr_abs:
                    min_risk = max(risk, MIN_RISK_ATR * atr_abs)
                    if min_risk > risk:
                        sl = entry - min_risk
                        risk = max(entry - sl, 1e-12)

            # TP cluster with min distance
            tp_info = next_tp_cluster_with_min_distance(product, entry, "long", min_tp_pct)
            if tp_info:
                tp_px, tp_qty, _ = tp_info
                tp = tp_px
                tp_note = f"Next ask cluster @ {fmt_price(tp_px)} (≥{min_tp_pct*100:.2f}% away)"
            else:
                # Fallback to R multiple and/or ATR TP floor
                tp = entry + max(MIN_RR * risk,
                                 (MIN_TP_ATR * get_atr_abs(product)) if (USE_ATR and get_atr_abs(product)) else 0.0)
                tp_note = f"Fallback: max({MIN_RR:.2f}R, ATR floor)"

            rr = (tp - entry) / risk
            if rr < MIN_RR:
                return None
            return dict(
                side="LONG", entry=entry, sl=sl, tp=tp, rr=rr,
                anchor_cluster=("bid", c_px, c_qty), tp_note=tp_note
            )

    # SHORT plan
    if flow_bias == "Aggressive SELL" and near_ask:
        c_px, c_qty = near_ask
        if c_px >= price:
            entry = price
            sl = c_px * (1 + SL_BUFFER_PCT)
            risk = max(sl - entry, 1e-12)

            if USE_ATR:
                atr_abs = get_atr_abs(product)
                if atr_abs:
                    min_risk = max(risk, MIN_RISK_ATR * atr_abs)
                    if min_risk > risk:
                        sl = entry + min_risk
                        risk = max(sl - entry, 1e-12)

            tp_info = next_tp_cluster_with_min_distance(product, entry, "short", min_tp_pct)
            if tp_info:
                tp_px, tp_qty, _ = tp_info
                tp = tp_px
                tp_note = f"Next bid cluster @ {fmt_price(tp_px)} (≥{min_tp_pct*100:.2f}% away)"
            else:
                tp = entry - max(MIN_RR * risk,
                                 (MIN_TP_ATR * get_atr_abs(product)) if (USE_ATR and get_atr_abs(product)) else 0.0)
                tp_note = f"Fallback: max({MIN_RR:.2f}R, ATR floor)"

            rr = (entry - tp) / risk
            if rr < MIN_RR:
                return None
            return dict(
                side="SHORT", entry=entry, sl=sl, tp=tp, rr=rr,
                anchor_cluster=("ask", c_px, c_qty), tp_note=tp_note
            )
    return None

# ------------- Handlers -------------
def handle_market_trades(product, trades):
    """
    Coinbase 'market_trades' side is maker side:
      maker=SELL -> taker bought (Aggressive BUY)
      maker=BUY  -> taker sold  (Aggressive SELL)
    """
    ts_now = now_ts()
    burst_times = large_trade_times[product]
    burst_vals  = burst_notional[product]

    th = percentile_threshold(product)

    for t in trades:
        price = float(t["price"])
        size = float(t["size"])
        side = t.get("side", "BUY")   # maker side
        # Notional estimate in USD (assuming quote is USD); safe for our symbols
        notional = price * size

        # signed_size as delta hint
        signed = size if side == "SELL" else -size  # SELL maker => taker bought => +delta

        last_price[product] = price
        update_price_ring(product, price)

        trade_buffer[product].append((ts_now, price, size, signed, notional))
        prune_old_trades(product)

        # refresh threshold after appending (adapts quickly)
        th = percentile_threshold(product)
        if th is not None and size >= th:
            burst_times.append(ts_now)
            burst_vals.append(notional)

    # prune burst window
    while burst_times and (ts_now - burst_times[0]) * 1000 > BURST_WINDOW_MS:
        burst_times.popleft()
        if burst_vals: burst_vals.popleft()

    # Check burst gates
    prof = asset_profile(product)
    needed_n = prof["BURST_N"]
    min_notional_ok = True
    if MIN_BURST_NOTIONAL_USD > 0 or prof["MIN_BURST_NOTIONAL"] > 0:
        min_req = max(MIN_BURST_NOTIONAL_USD, prof["MIN_BURST_NOTIONAL"])
        total_notional = sum(list(burst_vals))
        min_notional_ok = (total_notional >= min_req)

    if len(burst_times) >= needed_n and min_notional_ok and (ts_now - last_alert_time[product]) > ALERT_COOLDOWN_SEC:
        entry_px = last_price[product]
        # flow bias from recent large prints in window
        recent_signed = [sg for (_, _, _, sg, _) in list(trade_buffer[product])[-needed_n:]]
        flow_bias = "Aggressive BUY" if sum(recent_signed) > 0 else "Aggressive SELL"

        # nearby clusters (top-ranked only)
        near_bid = nearest_cluster(product, entry_px, "bid", CLUSTER_NEARBY_PCT)
        near_ask = nearest_cluster(product, entry_px, "ask", CLUSTER_NEARBY_PCT)

        plan = propose_trade_plan(product, flow_bias, entry_px, near_bid, near_ask)

# If we only want viable trades, skip sending anything unless we have a plan
        if ONLY_ALERT_ON_VIABLE and not plan:
    # optional: keep internal hygiene, then bail quietly
            last_alert_time[product] = ts_now  # prevent immediate re-evals on same burst
            burst_times.clear()
            burst_vals.clear()
            return

# Compose the message (plan or summary)
        msg = [
            f"<b>Large-Order Burst</b> on <b>{product}</b> @ {fmt_ts()}",
            f"Last price: <b>{fmt_price(entry_px)}</b>",
            f"Burst: <b>{len(burst_times)}</b> prints in ≤ {BURST_WINDOW_MS} ms",
            f"ATR% (est): <b>{(prof['ATR_PCT']*100):.2f}%</b>",
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
            if not ONLY_ALERT_ON_VIABLE and SEND_NONVIABLE_SUMMARY:
                msg += ["", "Skipped: min distance / ATR / R:R filters not met."]
            else:
        # If we don't want non-viable summaries, just stop here
                last_alert_time[product] = ts_now
                burst_times.clear()
                burst_vals.clear()
                return

        send_telegram("\n".join(msg))
        last_alert_time[product] = ts_now
        burst_times.clear()
        burst_vals.clear()


def handle_level2(product, updates):
    """
    Maintain side-separated binned clusters from absolute quantities.
    If new_quantity = 0 at a level, decay that bin so stale clusters fade.
    """
    book = clusters[product]
    for u in updates:
        price = float(u["price_level"])
        qty = float(u["new_quantity"])
        side = u.get("side", "").lower()
        side = "bid" if side.startswith("b") else "ask"
        b = bin_price(product, price)

        if qty <= 0:
            book[side][b] = max(0.0, book[side].get(b, 0.0) * 0.8)
        else:
            book[side][b] = max(book[side].get(b, 0.0), qty)

    # gentle decay
    for s in ("bid", "ask"):
        for bp in list(book[s].keys()):
            book[s][bp] *= 0.999
            if book[s][bp] < 1e-9:
                del book[s][bp]

# ------------- WS wiring -------------
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
    print("Starting Vivian-style Coinbase watcher v3…")
    print("Products:", PRODUCTS)
    run_ws()
