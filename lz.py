"""
Whale Liquidity Zones — Visual Plotter + Executed Flow (Coinbase, interchangeable coins)
"""
from __future__ import annotations
import argparse
import os
import time
import sys
import json
import threading
from collections import deque
from dataclasses import dataclass
from typing import List, Tuple, Dict

import requests
import pandas as pd
import matplotlib.pyplot as plt
import pytz
from datetime import datetime

try:
    from websocket import WebSocketApp  # pip install websocket-client
except Exception:
    WebSocketApp = None

@dataclass
class Config:
    symbol: str = "btc-usd"
    price_range_percent: float = 8.0
    bin_width_percent: float = 0.15
    whale_cluster_percentile: float = 95.0
    top_labels: int = 6
    timeout: int = 10
    save_path: str | None = None
    watch_minutes: float | None = None
    iterations: int | None = None
    save_dir: str | None = None
    near_pct: float = 2.0
    enable_flow: bool = False
    flow_minutes: int = 5

COINBASE_REST = "https://api.exchange.coinbase.com"
COINBASE_WS = "wss://ws-feed.exchange.coinbase.com"

_flow_lock = threading.Lock()
_flow_deque: deque[tuple[float, float, float, str]] = deque()

def get_live_price(symbol: str, timeout: int) -> float:
    r = requests.get(f"{COINBASE_REST}/products/{symbol}/ticker", timeout=timeout)
    r.raise_for_status()
    return float(r.json()["price"])

def get_orderbook(symbol: str, timeout: int) -> Dict:
    r = requests.get(f"{COINBASE_REST}/products/{symbol}/book", params={"level": 2}, timeout=timeout)
    r.raise_for_status()
    return r.json()

def dynamic_price_decimals(price: float) -> int:
    if price >= 1000: return 0
    if price >= 100: return 1
    if price >= 1: return 2
    if price >= 0.1: return 3
    return 4

def bin_orders(orders: List[Tuple[float, float]], spot: float, bin_width_percent: float, window_percent: float) -> List[Tuple[float, float]]:
    binned: Dict[float, float] = {}
    bin_width = spot * (bin_width_percent / 100.0)
    for p, q in orders:
        if abs(p - spot) / spot * 100.0 > window_percent:
            continue
        bin_price = round(p / bin_width) * bin_width
        notional = p * q
        binned[bin_price] = binned.get(bin_price, 0.0) + notional
    return sorted(binned.items(), key=lambda x: x[0])

def top_percentile_cutoff(values: List[float], pct: float) -> float:
    if not values:
        return 0.0
    s = pd.Series(values)
    return float(s.quantile(pct / 100.0))

def totals_within_pct(orders: List[Tuple[float, float]], spot: float, pct: float) -> float:
    lim = pct / 100.0
    total = 0.0
    for p, q in orders:
        if abs(p - spot) / spot <= lim:
            total += p * q
    return total

# ---------------- Flow ----------------

def _ws_on_message(symbol_upper: str, message: str):
    try:
        msg = json.loads(message)
    except Exception:
        return
    if msg.get("type") != "match":
        return
    if msg.get("product_id") != symbol_upper:
        return
    try:
        price = float(msg.get("price"))
        size = float(msg.get("size"))
        side = msg.get("side", "")
        ts = datetime.fromisoformat(msg.get("time").replace("Z", "+00:00")).timestamp()
    except Exception:
        return
    with _flow_lock:
        _flow_deque.append((ts, price, size, side))
        cutoff = time.time() - 2 * 3600
        while _flow_deque and _flow_deque[0][0] < cutoff:
            _flow_deque.popleft()

def _ws_thread(symbol_upper: str):
    if WebSocketApp is None:
        print("websocket-client not installed; executed flow overlay disabled.")
        return
    sub_msg = json.dumps({"type": "subscribe", "product_ids": [symbol_upper], "channels": ["matches"]})
    def on_open(ws): ws.send(sub_msg)
    def on_message(ws, message): _ws_on_message(symbol_upper, message)
    while True:
        try:
            ws = WebSocketApp(COINBASE_WS, on_open=on_open, on_message=on_message)
            ws.run_forever(ping_interval=30, ping_timeout=10)
        except Exception as e:
            print(f"WS reconnect: {e}")
            time.sleep(3)

def start_flow_thread(cfg: Config):
    symbol_upper = cfg.symbol.upper()
    t = threading.Thread(target=_ws_thread, args=(symbol_upper,), daemon=True)
    t.start()

def compute_flow_split(spot: float, near_pct: float, window_minutes: int):
    now = time.time()
    window_sec = int(window_minutes * 60)
    lo = spot * (1 - near_pct / 100.0)
    hi = spot * (1 + near_pct / 100.0)
    buy_total, sell_total = 0.0, 0.0
    with _flow_lock:
        while _flow_deque and _flow_deque[0][0] < now - window_sec:
            _flow_deque.popleft()
        for ts, price, size, side in _flow_deque:
            if now - ts <= window_sec and lo <= price <= hi:
                notional = price * size
                if side == "buy":
                    sell_total += notional
                elif side == "sell":
                    buy_total += notional
    return buy_total, sell_total

# ---------------- Core ----------------

def prepare_data(cfg: Config):
    spot = get_live_price(cfg.symbol, cfg.timeout)
    ob = get_orderbook(cfg.symbol, cfg.timeout)
    bids_raw = [(float(p), float(q)) for p, q, *_ in ob.get("bids", [])]
    asks_raw = [(float(p), float(q)) for p, q, *_ in ob.get("asks", [])]
    bids = bin_orders(bids_raw, spot, cfg.bin_width_percent, cfg.price_range_percent)
    asks = bin_orders(asks_raw, spot, cfg.bin_width_percent, cfg.price_range_percent)
    bid_cut = top_percentile_cutoff([q for _, q in bids], cfg.whale_cluster_percentile)
    ask_cut = top_percentile_cutoff([q for _, q in asks], cfg.whale_cluster_percentile)
    whale_bids = [(p, q) for p, q in bids if q >= bid_cut and q > 0]
    whale_asks = [(p, q) for p, q in asks if q >= ask_cut and q > 0]
    near_bid_total = totals_within_pct(bids_raw, spot, cfg.near_pct)
    near_ask_total = totals_within_pct(asks_raw, spot, cfg.near_pct)
    return spot, whale_bids, whale_asks, near_bid_total, near_ask_total

def build_labels(spot: float, bids: List[Tuple[float, float]], asks: List[Tuple[float, float]], n: int):
    combined = [(p, q, "BID") for p, q in bids] + [(p, q, "ASK") for p, q in asks]
    combined.sort(key=lambda x: x[1], reverse=True)
    return combined[:n]

# ---------------- Plot ----------------

def current_pst_time() -> str:
    tz = pytz.timezone("America/Los_Angeles")
    return datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S %Z")

def _render(ax, cfg: Config, spot: float, bids: List[Tuple[float, float]], asks: List[Tuple[float, float]], near_bid_total: float, near_ask_total: float, flow_split: Tuple[float,float] | None):
    dec = dynamic_price_decimals(spot)
    y_prices = sorted({p for p, _ in bids} | {p for p, _ in asks})
    if not y_prices:
        ax.text(0.5, 0.5, "No data in window", ha='center', va='center', transform=ax.transAxes)
        return
    bid_map = {p: q for p, q in bids}
    ask_map = {p: q for p, q in asks}
    bid_vals = [-bid_map.get(p, 0.0) for p in y_prices]
    ask_vals = [ ask_map.get(p, 0.0) for p in y_prices]
    step = (y_prices[1]-y_prices[0] if len(y_prices) > 1 else spot*cfg.bin_width_percent/100.0)
    ax.barh(y_prices, bid_vals, height=step, align='center', label='Bids (top percentile)')
    ax.barh(y_prices, ask_vals, height=step, align='center', label='Asks (top percentile)')
    ax.axhline(spot, linestyle='--', linewidth=1)
    xmax = ax.get_xlim()[1]
    ax.text(xmax, spot, f" Spot: {spot:.{dec}f}", va='center', ha='right', fontsize=9)
    for p, q, side in build_labels(spot, bids, asks, cfg.top_labels):
        x = -q if side == "BID" else q
        ax.text(x, p, f"{side} ${q:,.0f}", va='center', ha='left' if side=="ASK" else 'right', fontsize=8)
    ax.text(0.01, 0.08, f"±{cfg.near_pct:.1f}% Totals — Bids: ${near_bid_total:,.0f}  |  Asks: ${near_ask_total:,.0f}", transform=ax.transAxes, fontsize=9, ha='left', va='bottom', alpha=0.95)
    if flow_split is not None:
        buy_total, sell_total = flow_split
        ax.text(0.01, 0.04, f"Executed flow (last {cfg.flow_minutes}m ±{cfg.near_pct:.1f}%):", transform=ax.transAxes, fontsize=9, ha='left', va='bottom')
        ax.text(0.25, 0.04, f"Buys ${buy_total:,.0f}", transform=ax.transAxes, fontsize=9, ha='left', va='bottom', color='green')
        ax.text(0.55, 0.04, f"Sells ${sell_total:,.0f}", transform=ax.transAxes, fontsize=9, ha='left', va='bottom', color='red')
    ax.text(0.01, 0.01, f"Updated: {current_pst_time()}", transform=ax.transAxes, fontsize=8, ha='left', va='bottom')
    ax.set_xlabel("Notional Value in USD (negative = bids, positive = asks)")
    ax.set_ylabel("Price")
    ax.set_title(f"{cfg.symbol.upper()} Liquidity Zones (Top {cfg.whale_cluster_percentile:.0f}%, ±{cfg.price_range_percent:.1f}%, bin {cfg.bin_width_percent:.3f}%)")
    ax.legend(loc='lower right')
    ax.grid(True, which='both', axis='both', alpha=0.25)

# ---------------- Runners ----------------

def plot_liquidity(cfg: Config):
    if cfg.enable_flow:
        start_flow_thread(cfg)
        time.sleep(1.0)
    spot, bids, asks, near_bid_total, near_ask_total = prepare_data(cfg)
    flow_split = compute_flow_split(spot, cfg.near_pct, cfg.flow_minutes) if cfg.enable_flow else None
    fig, ax = plt.subplots(figsize=(10, 7))
    _render(ax, cfg, spot, bids, asks, near_bid_total, near_ask_total, flow_split)
    plt.tight_layout()
    if cfg.save_path:
        plt.savefig(cfg.save_path, dpi=150)
        print(f"Saved plot to {cfg.save_path}")
    plt.show()

def watch_liquidity(cfg: Config):
    if cfg.enable_flow:
        start_flow_thread(cfg)
        time.sleep(1.0)
    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 7))
    iteration = 0
    try:
        while True:
            iteration += 1
            ax.clear()
            spot, bids, asks, near_bid_total, near_ask_total = prepare_data(cfg)
            flow_split = compute_flow_split(spot, cfg.near_pct, cfg.flow_minutes) if cfg.enable_flow else None
            _render(ax, cfg, spot, bids, asks, near_bid_total, near_ask_total, flow_split)
            plt.tight_layout()
            if cfg.save_dir:
                os.makedirs(cfg.save_dir, exist_ok=True)
                ts = time.strftime('%Y%m%d_%H%M%S')
                out = os.path.join(cfg.save_dir, f"{cfg.symbol.replace('-', '_')}_{ts}.png")
                plt.savefig(out, dpi=150)
                print(f"Saved snapshot: {out} at {current_pst_time()}")
            fig.canvas.draw()
            fig.canvas.flush_events()
            if cfg.iterations and iteration >= cfg.iterations:
                break
            time.sleep(cfg.watch_minutes * 60)
    except KeyboardInterrupt:
        print("Stopped watch mode.")
    finally:
        plt.ioff()
        plt.show()

# ---------------- CLI ----------------

def parse_args(argv: List[str]) -> Config:
    p = argparse.ArgumentParser(description="Plot heavy liquidity zones from Coinbase order book (level 2)")
    p.add_argument("--symbol", default="eth-usd")
    p.add_argument("--range", type=float, default=8.0, dest="price_range_percent")
    p.add_argument("--bin", type=float, default=0.1, dest="bin_width_percent")
    p.add_argument("--percentile", type=float, default=1.0, dest="whale_cluster_percentile")
    p.add_argument("--labels", type=int, default=6, dest="top_labels")
    p.add_argument("--timeout", type=int, default=10)
    p.add_argument("--save", dest="save_path", default=None)
    p.add_argument("--watch", type=float, dest="watch_minutes", default=None)
    p.add_argument("--iterations", type=int, dest="iterations", default=None)
    p.add_argument("--save_dir", dest="save_dir", default=None)
    p.add_argument("--near", type=float, dest="near_pct", default=2.0)
    p.add_argument("--flow", action="store_true", dest="enable_flow")
    p.add_argument("--flow_minutes", type=int, dest="flow_minutes", default=5)
    args = p.parse_args(argv)
    return Config(
        symbol=args.symbol,
        price_range_percent=args.price_range_percent,
        bin_width_percent=args.bin_width_percent,
        whale_cluster_percentile=args.whale_cluster_percentile,
        top_labels=args.top_labels,
        timeout=args.timeout,
        save_path=args.save_path,
        watch_minutes=args.watch_minutes,
        iterations=args.iterations,
        save_dir=args.save_dir,
        near_pct=args.near_pct,
        enable_flow=args.enable_flow,
        flow_minutes=args.flow_minutes,
    )

if __name__ == "__main__":
    cfg = parse_args(sys.argv[1:])
    if cfg.watch_minutes:
        watch_liquidity(cfg)
    else:
        plot_liquidity(cfg)
