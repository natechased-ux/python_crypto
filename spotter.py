"""
Whale Liquidity Zones — Visual Plotter (Coinbase, interchangeable coins)

Purpose
- Visualize *heavy liquidity zones* (large resting limit orders) around current price.
- No trade signals — purely a plotting/analysis tool.
- Default instrument: BTC-USD; easily switch to others (e.g., ETH-USD, XRP-USD).

Key Features
- Level 2 order book from Coinbase.
- Binning by fixed % of price using: round(price / bin_width) * bin_width.
- Separate bid/ask aggregation with independent percentile cutoffs (e.g., top 95%).
- Focus window: ±N% from current price.
- One clean chart: horizontal bars at each price bin, length ∝ *notional dollar value* (price × size) instead of raw size.
- Current price line + labels for nearest major clusters.
- CLI flags and/or config block for quick tweaks.
- Optional file output (PNG) in addition to interactive view.
- **New:** Auto-refresh watch mode (e.g., update every 5 minutes) with optional snapshot saving per refresh.
- **New:** Timestamp in PST printed and displayed on chart, so you know when data was fetched.
- **New:** Shows **total $ value of bids vs. asks within ±2% of spot** on the chart.

Usage
- One-off plot (recommended):
  python3 liquidity_zones.py --symbol btc-usd --percentile 95 --range 8 --bin 0.15 --save out.png
- Live watch (5-minute refresh, infinite loop):
  python3 liquidity_zones.py --symbol btc-usd --watch 5
- Live watch, save a PNG each refresh to a folder:
  python3 liquidity_zones.py --symbol btc-usd --watch 5 --save_dir snaps/

Dependencies: requests, pandas, matplotlib, pytz

Notes on interpretation
- The X-axis labelled "Size ($)" is now the **notional dollar value of resting orders** at each binned price level (price × size).
- Negative values are plotted for bids so they extend leftward; positive values are for asks extending rightward.

"""
from __future__ import annotations
import argparse
import os
import time
import math
import sys
from dataclasses import dataclass
from typing import List, Tuple, Dict

import requests
import pandas as pd
import matplotlib.pyplot as plt
import pytz
from datetime import datetime

# ------------------------- Config -------------------------
@dataclass
class Config:
    symbol: str = "eth-usd"
    price_range_percent: float = 8.0       # show orders within ±X% of spot
    bin_width_percent: float = 0.15        # bin width as % of spot (e.g., 0.15% of price)
    whale_cluster_percentile: float = 95.0 # keep top X% by size per side
    top_labels: int = 6                    # annotate N biggest clusters (bids+asks combined)
    timeout: int = 10                      # HTTP timeout seconds
    save_path: str | None = None           # if set, saves PNG here (single run)
    watch_minutes: float | None = None     # if set, auto-refresh every N minutes
    iterations: int | None = None          # if set in watch mode, limit number of refreshes
    save_dir: str | None = None            # if set in watch mode, save snapshot per refresh here

# ------------------------- HTTP helpers -------------------------
COINBASE = "https://api.exchange.coinbase.com"

def get_live_price(symbol: str, timeout: int) -> float:
    r = requests.get(f"{COINBASE}/products/{symbol}/ticker", timeout=timeout)
    r.raise_for_status()
    return float(r.json()["price"])  # type: ignore

def get_orderbook(symbol: str, timeout: int) -> Dict:
    # level=2 is aggregated by price; includes many levels.
    r = requests.get(f"{COINBASE}/products/{symbol}/book", params={"level": 2}, timeout=timeout)
    r.raise_for_status()
    return r.json()  # bids/asks: [[price, size, num-orders], ...]

# ------------------------- Math helpers -------------------------

def dynamic_price_decimals(price: float) -> int:
    if price >= 1000: return 0
    if price >= 100: return 1
    if price >= 1: return 2
    if price >= 0.1: return 3
    return 4


def bin_orders(orders: List[Tuple[float, float]], spot: float, bin_width_percent: float,
               window_percent: float) -> List[Tuple[float, float]]:
    """Aggregate order notional values into fixed price bins around spot.
    Returns list of (bin_price, total_notional).
    """
    binned: Dict[float, float] = {}
    bin_width = spot * (bin_width_percent / 100.0)
    if bin_width <= 0:
        raise ValueError("bin_width_percent must be > 0")

    for p, q in orders:
        # limit to focus window
        if abs(p - spot) / spot * 100.0 > window_percent:
            continue
        # price binning using round(price / bin_width) * bin_width
        bin_price = round(p / bin_width) * bin_width
        notional = p * q  # convert to dollar value
        binned[bin_price] = binned.get(bin_price, 0.0) + notional

    return sorted(binned.items(), key=lambda x: x[0])


def top_percentile_cutoff(values: List[float], pct: float) -> float:
    if not values:
        return 0.0
    s = pd.Series(values)
    return float(s.quantile(pct / 100.0))


def totals_within_pct(orders: List[Tuple[float, float]], spot: float, pct: float) -> float:
    """Sum notional (price*size) for orders within ±pct of spot."""
    lim = pct / 100.0
    total = 0.0
    for p, q in orders:
        if abs(p - spot) / spot <= lim:
            total += p * q
    return total

# ------------------------- Core -------------------------

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

    near_bid_total = totals_within_pct(bids_raw, spot, 2.0)
    near_ask_total = totals_within_pct(asks_raw, spot, 2.0)

    return spot, whale_bids, whale_asks, near_bid_total, near_ask_total


def build_labels(spot: float, bids: List[Tuple[float, float]], asks: List[Tuple[float, float]], n: int):
    combined = [(p, q, "BID") for p, q in bids] + [(p, q, "ASK") for p, q in asks]
    combined.sort(key=lambda x: x[1], reverse=True)
    return combined[:n]


# ------------------------- Plot helpers -------------------------

def current_pst_time() -> str:
    tz = pytz.timezone("America/Los_Angeles")
    return datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S %Z")


def _render(ax, cfg: Config, spot: float, bids: List[Tuple[float, float]], asks: List[Tuple[float, float]], near_bid_total: float, near_ask_total: float):
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

    # Spot line
    ax.axhline(spot, linestyle='--', linewidth=1)
    xmax = ax.get_xlim()[1]
    ax.text(xmax, spot, f" Spot: {spot:.{dec}f}", va='center', ha='right', fontsize=9)

    # Labels for biggest clusters (show $ value)
    for p, q, side in build_labels(spot, bids, asks, cfg.top_labels):
        x = -q if side == "BID" else q
        ax.text(x, p, f"{side} ${q:,.0f}", va='center', ha='left' if side=="ASK" else 'right', fontsize=8)

    # Totals within ±2% overlay (bottom-left)
    ax.text(0.01, 0.06, f"±2% Totals — Bids: ${near_bid_total:,.0f}  |  Asks: ${near_ask_total:,.0f}", transform=ax.transAxes, fontsize=9, ha='left', va='bottom', alpha=0.9)

    # Timestamp label in PST
    ax.text(0.01, 0.01, f"Updated: {current_pst_time()}", transform=ax.transAxes, fontsize=8, ha='left', va='bottom')

    ax.set_xlabel("Notional Value in USD (negative = bids, positive = asks)")
    ax.set_ylabel("Price")
    ax.set_title(f"{cfg.symbol.upper()} Liquidity Zones (Top {cfg.whale_cluster_percentile:.0f}%, ±{cfg.price_range_percent:.1f}%, bin {cfg.bin_width_percent:.3f}%)")
    ax.legend(loc='lower right')
    ax.grid(True, which='both', axis='both', alpha=0.25)


def plot_liquidity(cfg: Config):
    spot, bids, asks, near_bid_total, near_ask_total = prepare_data(cfg)
    fig, ax = plt.subplots(figsize=(10, 7))
    _render(ax, cfg, spot, bids, asks, near_bid_total, near_ask_total)
    plt.tight_layout()
    if cfg.save_path:
        plt.savefig(cfg.save_path, dpi=150)
        print(f"Saved plot to {cfg.save_path}")
    plt.show()


def watch_liquidity(cfg: Config):
    assert cfg.watch_minutes and cfg.watch_minutes > 0
    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 7))
    iteration = 0
    try:
        while True:
            iteration += 1
            ax.clear()
            spot, bids, asks, near_bid_total, near_ask_total = prepare_data(cfg)
            _render(ax, cfg, spot, bids, asks, near_bid_total, near_ask_total)
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


# ------------------------- CLI -------------------------

def parse_args(argv: List[str]) -> Config:
    p = argparse.ArgumentParser(description="Plot heavy liquidity zones from Coinbase order book (level 2)")
    p.add_argument("--symbol", default="eth-usd", help="Product symbol, e.g., btc-usd, eth-usd, xrp-usd")
    p.add_argument("--range", type=float, default=30.0, dest="price_range_percent", help="±% of spot to include")
    p.add_argument("--bin", type=float, default=0.01, dest="bin_width_percent", help="bin width as % of spot")
    p.add_argument("--percentile", type=float, default=95.0, dest="whale_cluster_percentile", help="keep top % by size per side")
    p.add_argument("--labels", type=int, default=6, dest="top_labels", help="annotate N largest clusters")
    p.add_argument("--timeout", type=int, default=10, help="HTTP timeout seconds")
    p.add_argument("--save", dest="save_path", default=None, help="optional PNG output path (single run)")
    p.add_argument("--watch", type=float, dest="watch_minutes", default=None, help="auto-refresh every N minutes")
    p.add_argument("--iterations", type=int, dest="iterations", default=None, help="limit refresh cycles in watch mode")
    p.add_argument("--save_dir", dest="save_dir", default=None, help="if set in watch mode, save snapshot per refresh here")
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
    )


if __name__ == "__main__":
    cfg = parse_args(sys.argv[1:])
    if cfg.watch_minutes:
        watch_liquidity(cfg)
    else:
        plot_liquidity(cfg)
