"""
Liquidity Map / Heatmap + Pools-as-Stop-Loss Telegram Alert System

Author: ChatGPT
Date: 2025-10-03

Overview
--------
- Builds a cross-exchange liquidity heatmap (binned order book) around current price
- Finds nearest significant liquidity pools (support below / resistance above)
- Generates LONG when: strong bid pool below price + trend/momentum confirm
- Generates SHORT when: strong ask pool above price + trend/momentum confirm
- Stop-loss is placed *just beyond the liquidity pool* (instead of swing)
- Take-profit can target the opposite-side pool or ATR multiple
- Sends Telegram alerts (non-async) with full context; includes CSV logging

Notes
-----
- Uses CCXT for exchanges. Default: Coinbase (spot) + Kraken (spot). Add others easily.
- For Kraken Futures, you can wire a dedicated fetcher later (ccxt has 'kraken' spot; futures may be 'cryptofacilities' in some ccxt versions). This bot is exchange-agnostic.
- Momentum filter uses 1H Stochastic RSI (K>D rising for longs; K<D falling for shorts) and 200 EMA trend filter from 1H candles.
- Binning uses your preferred method: bin_price = round(price / bin_width) * bin_width.

Setup
-----
1) pip install -U ccxt pandas numpy requests python-dateutil
2) (Optional) export TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID as env vars
3) Edit CONFIG below (symbols, thresholds, schedule)
4) python liquidity_map_bot.py
"""
from __future__ import annotations
import os
import time
import math
import json
import csv
import threading
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import requests
import numpy as np
import html
import pandas as pd
from datetime import datetime, timezone
from dateutil import tz

try:
    import ccxt  # type: ignore
except Exception as e:
    raise SystemExit("Please `pip install ccxt` before running. Error: %s" % e)

# =============================
# ======== CONFIG =============
# =============================
@dataclass
class Config:
    symbols: List[str]
    exchanges: List[str]
    timeframe_candles: str = "1h"
    candles_limit: int = 300
    # Liquidity heatmap
    max_distance_pct: float = 0.03          # consider orders within ±3% of spot
    bin_width_pct: float = 0.0015           # bin width as % of price (0.15%)
    pool_percentile: float = 90.0           # bins >= this percentile considered "significant pools"
    min_quote_in_bin: float = 1000.0        # ignore bins with quote value < this (safety floor)
    # Momentum / trend
    use_trend_filter: bool = True
    ema_period: int = 200                   # 200 EMA on 1H
    stoch_k_period: int = 14
    stoch_d_period: int = 3
    stoch_smooth_k: int = 3
    # Entry confirmation (stricter like your preference)
    long_k_lt: float = 40.0                 # Long: K > D and K < 40
    short_k_gt: float = 60.0                # Short: K < D and K > 60
    # Risk / targets
    sl_offset_pct: float = 0.0008           # place SL 0.08% beyond pool
    tp_mode: str = "opposite_pool"          # "opposite_pool" | "atr_multiple"
    atr_period: int = 14
    tp_multiple: float = 2.0                # if atr_multiple
    sl_multiple: float = 1.3
    tp_inside_offset_pct: float = 0.0002   # nudge TP just inside opposite pool
    min_tp_pct: float = 0.01               # require at least +/-1% from entry                # fallback if no pool found
    # Scanning
    scan_interval_sec: int = 300            # 5 minutes
    cooldown_minutes: int = 90              # per-symbol cooldown (aligns w/ your recent pref)
    # Telegram
    telegram_bot_token: str = os.getenv("TELEGRAM_BOT_TOKEN", "8177096945:AAHTg5nxVTA6hcPkAL4MLAAHPsZ1at7Ywmw")
    telegram_chat_id: str = os.getenv("TELEGRAM_CHAT_ID", "7967738614")
    # Logging
    log_csv: str = "liquidity_alerts.csv"
    timezone_name: str = "America/Los_Angeles"

CONFIG = Config(
    symbols=[
        # Use Coinbase symbols with ccxt unified notation
        # Examples: 'BTC/USD', 'ETH/USD', 'XRP/USD'
        "BTC/USD", "ETH/USD","SOL/USD","XRP/USD","ADA/USD","AVAX/USD","DOGE/USD","DOT/USD",
        "LINK/USD","ATOM/USD","NEAR/USD","ARB/USD","OP/USD","INJ/USD","AAVE/USD","LTC/USD","BCH/USD",
        "ETC/USD","ALGO/USD","FIL/USD","ICP/USD","STX/USD","GRT/USD","SEI/USD","HBAR/USD",
        "TIA/USD","UNI/USD","FET/USD","CRV/USD","SAND/USD","MANA/USD","AXS/USD",
        "PEPE/USD","XLM/USD"
    ],
    exchanges=["coinbase", "kraken"],
)

# =============================
# ===== Utilities =============
# =============================
PT_TZ = tz.gettz(CONFIG.timezone_name)

def now_ts() -> float:
    return time.time()

def fmt_ts(ts: float) -> str:
    dt = datetime.fromtimestamp(ts, tz=PT_TZ)
    return dt.strftime("%Y-%m-%d %H:%M:%S %Z")

def price_decimals(price: float) -> int:
    if price >= 1000: return 0
    if price >= 100: return 1
    if price >= 10: return 2
    if price >= 1: return 3
    if price >= 0.1: return 4
    if price >= 0.01: return 5
    return 6

# =============================
# ===== Indicators ============
# =============================

def ema(series: np.ndarray, period: int) -> np.ndarray:
    alpha = 2 / (period + 1)
    ema_vals = np.empty_like(series, dtype=float)
    ema_vals[:] = np.nan
    if len(series) == 0: return ema_vals
    ema_vals[0] = series[0]
    for i in range(1, len(series)):
        ema_vals[i] = alpha * series[i] + (1 - alpha) * ema_vals[i-1]
    return ema_vals


def atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    trs = []
    prev_close = np.nan
    for h, l, c in zip(high, low, close):
        if np.isnan(prev_close):
            tr = h - l
        else:
            tr = max(h - l, abs(h - prev_close), abs(l - prev_close))
        trs.append(tr)
        prev_close = c
    trs = np.array(trs)
    return ema(trs, period)


def stoch_rsi(close: np.ndarray, period: int = 14, smooth_k: int = 3, d_period: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    # Compute RSI first
    deltas = np.diff(close)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    # Wilder's smoothing
    rsi = np.empty_like(close)
    rsi[:] = np.nan
    if len(close) < period + 1:
        return rsi, rsi
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])
    rs = (avg_gain / avg_loss) if avg_loss != 0 else np.inf
    rsi[period] = 100 - (100 / (1 + rs))
    for i in range(period + 1, len(close)):
        gain = gains[i-1]
        loss = losses[i-1]
        avg_gain = (avg_gain * (period - 1) + gain) / period
        avg_loss = (avg_loss * (period - 1) + loss) / period
        rs = (avg_gain / avg_loss) if avg_loss != 0 else np.inf
        rsi[i] = 100 - (100 / (1 + rs))
    # Stochastic of RSI
    k = np.empty_like(close)
    d = np.empty_like(close)
    k[:] = np.nan
    d[:] = np.nan
    for i in range(len(close)):
        start = max(0, i - period + 1)
        window = rsi[start:i+1]
        if np.sum(~np.isnan(window)) < 2:
            continue
        mn = np.nanmin(window)
        mx = np.nanmax(window)
        denom = (mx - mn) if (mx - mn) != 0 else np.nan
        k[i] = 100 * (rsi[i] - mn) / denom if not np.isnan(denom) else np.nan
    # smooth K
    k_s = pd.Series(k).rolling(window=smooth_k, min_periods=1).mean().to_numpy()
    d_s = pd.Series(k_s).rolling(window=d_period, min_periods=1).mean().to_numpy()
    return k_s, d_s

# =============================
# ===== Exchanges =============
# =============================

def build_exchanges(ids: List[str]):
    exs = []
    for x in ids:
        try:
            ex = getattr(ccxt, x)({"enableRateLimit": True})
            ex.load_markets()
            exs.append(ex)
        except Exception as e:
            print(f"[WARN] Failed to init exchange {x}: {e}")
    if not exs:
        raise SystemExit("No exchanges initialized. Check ccxt ids in CONFIG.exchanges")
    return exs

EXCHANGES = build_exchanges(CONFIG.exchanges)

# Cache which exchange supports which market symbol for OHLCV to avoid 429s
_MARKET_EX_CACHE: Dict[str, ccxt.Exchange] = {}


def resolve_market_exchange(symbol: str) -> Optional[ccxt.Exchange]:
    """Return the first exchange that supports this symbol for OHLCV/candles."""
    if symbol in _MARKET_EX_CACHE:
        return _MARKET_EX_CACHE[symbol]
    for ex in EXCHANGES:
        try:
            if symbol in ex.markets and ex.markets[symbol].get('active', True):
                _MARKET_EX_CACHE[symbol] = ex
                return ex
        except Exception:
            continue
    return None

# =============================
# ===== Data Fetching =========
# =============================

def fetch_orderbook_around_price(ex, symbol: str, max_distance_pct: float, depth: int = 200):
    ob = ex.fetch_order_book(symbol, limit=depth)
    ticker = ex.fetch_ticker(symbol)
    price = float(ticker['last']) if ticker and ticker.get('last') else float(ob['bids'][0][0])
    lower = price * (1 - max_distance_pct)
    upper = price * (1 + max_distance_pct)
    bids = [b for b in ob['bids'] if lower <= b[0] <= upper]
    asks = [a for a in ob['asks'] if lower <= a[0] <= upper]
    return price, bids, asks


def fetch_ohlcv_df(ex, symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
    ohlcv = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=["ts","open","high","low","close","volume"])
    df['ts'] = pd.to_datetime(df['ts'], unit='ms', utc=True).dt.tz_convert(PT_TZ)
    return df


def safe_fetch_ohlcv_any(symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
    """Try preferred exchange for this symbol; fall back to others; backoff on 429."""
    preferred = resolve_market_exchange(symbol)
    ex_list = ([preferred] if preferred else []) + [ex for ex in EXCHANGES if ex is not preferred]
    last_err = None
    for ex in ex_list:
        try:
            return fetch_ohlcv_df(ex, symbol, timeframe, limit)
        except ccxt.RateLimitExceeded as e:
            last_err = e
            time.sleep(2.5)
            continue
        except Exception as e:
            last_err = e
            continue
    raise last_err if last_err else RuntimeError(f"No OHLCV source available for {symbol}")

# =============================
# ===== Heatmap / Pools =======
# =============================

def bin_orders(orders: List[List[float]], price: float, bin_width_pct: float) -> Dict[float, float]:
    """
    Some exchanges (e.g., Kraken via CCXT) may return order rows with >2 fields
    (e.g., [price, amount, timestamp]). To be robust, index the first two fields
    instead of tuple-unpacking to (p, amt).
    """
    bins: Dict[float, float] = {}
    bin_width = price * bin_width_pct
    if bin_width <= 0:
        return bins
    for row in orders:
        if not row or len(row) < 2:
            continue
        p = float(row[0])
        amt = float(row[1])
        bin_price = round(p / bin_width) * bin_width
        quote_val = abs(amt) * p  # normalize to quote currency
        bins[bin_price] = bins.get(bin_price, 0.0) + quote_val
    return bins


def aggregate_cross_exchanges(symbol: str, max_distance_pct: float, bin_width_pct: float) -> Tuple[float, Dict[str, Dict[float, float]]]:
    """Returns (mid price, {'bids':{bin:quote}, 'asks':{bin:quote}}) across exchanges"""
    all_bids: Dict[float, float] = {}
    all_asks: Dict[float, float] = {}
    spot_price: Optional[float] = None
    for ex in EXCHANGES:
        try:
            price, bids, asks = fetch_orderbook_around_price(ex, symbol, max_distance_pct)
            spot_price = spot_price or price
            binned_bids = bin_orders(bids, price, bin_width_pct)
            binned_asks = bin_orders(asks, price, bin_width_pct)
            for k,v in binned_bids.items():
                all_bids[k] = all_bids.get(k, 0.0) + v
            for k,v in binned_asks.items():
                all_asks[k] = all_asks.get(k, 0.0) + v
        except Exception as e:
            print(f"[WARN] Orderbook fetch failed on {ex.id} {symbol}: {e}")
            continue
    if spot_price is None:
        raise RuntimeError("Failed to fetch spot price from any exchange")
    return spot_price, {"bids": all_bids, "asks": all_asks}


def significant_pools(heatmap: Dict[str, Dict[float, float]], percentile: float, min_quote: float) -> Tuple[List[Tuple[float,float]], List[Tuple[float,float]]]:
    # returns sorted lists: bids (desc price), asks (asc price) of (bin_price, quote_val)
    bids_items = list(heatmap["bids"].items())
    asks_items = list(heatmap["asks"].items())
    vals = [v for _,v in bids_items + asks_items]
    if not vals:
        return [], []
    thresh = np.percentile(vals, percentile)
    bids_sig = [(p,v) for p,v in bids_items if v >= max(thresh, min_quote)]
    asks_sig = [(p,v) for p,v in asks_items if v >= max(thresh, min_quote)]
    bids_sig.sort(key=lambda x: x[0], reverse=True)
    asks_sig.sort(key=lambda x: x[0])
    return bids_sig, asks_sig


def nearest_pool(price: float, pools: List[Tuple[float,float]], side: str) -> Optional[Tuple[float,float]]:
    # side='below' -> pool price < spot; side='above' -> pool price > spot
    if side == 'below':
        below = [p for p in pools if p[0] < price]
        if not below: return None
        return min(below, key=lambda x: abs(x[0]-price))
    else:
        above = [p for p in pools if p[0] > price]
        if not above: return None
        return min(above, key=lambda x: abs(x[0]-price))

# =============================
# ===== Signal Logic ==========
# =============================

def compute_context(ex, symbol: str) -> Dict:
    df = safe_fetch_ohlcv_any(symbol, CONFIG.timeframe_candles, CONFIG.candles_limit)
    close = df['close'].to_numpy()
    high = df['high'].to_numpy()
    low = df['low'].to_numpy()
    ema200 = ema(close, CONFIG.ema_period)
    k, d = stoch_rsi(close, CONFIG.stoch_k_period, CONFIG.stoch_smooth_k, CONFIG.stoch_d_period)
    atr_vals = atr(high, low, close, CONFIG.atr_period)
    ctx = {
        "df": df,
        "close": close,
        "ema200": ema200,
        "stoch_k": k,
        "stoch_d": d,
        "atr": atr_vals,
    }
    return ctx


def trend_ok_long(ctx: Dict, px: float) -> bool:
    if not CONFIG.use_trend_filter:
        return True
    ema200 = ctx['ema200']
    return not np.isnan(ema200[-1]) and px > ema200[-1]


def trend_ok_short(ctx: Dict, px: float) -> bool:
    if not CONFIG.use_trend_filter:
        return True
    ema200 = ctx['ema200']
    return not np.isnan(ema200[-1]) and px < ema200[-1]


def momentum_ok_long(ctx: Dict) -> bool:
    k = ctx['stoch_k'][-1]
    d = ctx['stoch_d'][-1]
    return (not np.isnan(k) and not np.isnan(d) and k > d and k < CONFIG.long_k_lt)


def momentum_ok_short(ctx: Dict) -> bool:
    k = ctx['stoch_k'][-1]
    d = ctx['stoch_d'][-1]
    return (not np.isnan(k) and not np.isnan(d) and k < d and k > CONFIG.short_k_gt)


def build_signal(symbol: str) -> Optional[Dict]:
    # Use the first successfully initialized exchange for candles (unified context)
    ex0 = EXCHANGES[0]
    ctx = compute_context(ex0, symbol)

    # Opposite pool must be at least 1% away, else skip
    min_gap = CONFIG.min_tp_pct  # reuse 1% setting


    # Cross-exchange heatmap
    spot, heatmap = aggregate_cross_exchanges(symbol, CONFIG.max_distance_pct, CONFIG.bin_width_pct)
    bids_sig, asks_sig = significant_pools(heatmap, CONFIG.pool_percentile, CONFIG.min_quote_in_bin)

    # nearest pools
    bid_below = nearest_pool(spot, bids_sig, 'below')  # support
    ask_above = nearest_pool(spot, asks_sig, 'above')  # resistance

    # Decide if we have a long or short setup
    long_ok = bid_below is not None and trend_ok_long(ctx, spot) and momentum_ok_long(ctx)
    short_ok = ask_above is not None and trend_ok_short(ctx, spot) and momentum_ok_short(ctx)

    # Opposite pool must be at least 1% away, else skip
    if long_ok and ask_above is not None:
        if (ask_above[0] - spot) / spot < CONFIG.min_tp_pct:
            return None  # skip LONG: resistance too close
    if short_ok and bid_below is not None:
        if (spot - bid_below[0]) / spot < CONFIG.min_tp_pct:
            return None  # skip SHORT: support too close

    if not long_ok and not short_ok:
        return None

    px_dec = price_decimals(spot)
    last_close = ctx['close'][-1]
    last_atr = ctx['atr'][-1]

    if long_ok and (not short_ok or (bid_below and ask_above and (spot - bid_below[0]) <= (ask_above[0] - spot))):
        pool_price = bid_below[0]
        entry = spot  # market-ish entry (could refine to limit near pool)
        sl = pool_price * (1 - CONFIG.sl_offset_pct)
        if CONFIG.tp_mode == "opposite_pool" and ask_above is not None:
            tp_candidate = ask_above[0] * (1 - CONFIG.tp_inside_offset_pct)
            tp = max(entry * (1 + CONFIG.min_tp_pct), tp_candidate)
            tp = min(tp, ask_above[0] * (1 - 1e-6))
        else:
            tp = entry + CONFIG.tp_multiple * last_atr
        side = "LONG"
        context = {
            "pool_side": "BID_SUPPORT",
            "pool_price": round(pool_price, px_dec),
            "pool_size": bid_below[1],
        }
    else:
        pool_price = ask_above[0]
        entry = spot
        sl = pool_price * (1 + CONFIG.sl_offset_pct)
        if CONFIG.tp_mode == "opposite_pool" and bid_below is not None:
            tp_candidate = bid_below[0] * (1 + CONFIG.tp_inside_offset_pct)
            tp = min(entry * (1 - CONFIG.min_tp_pct), tp_candidate)
            tp = max(tp, bid_below[0] * (1 + 1e-6))
        else:
            tp = entry - CONFIG.tp_multiple * last_atr
        side = "SHORT"
        context = {
            "pool_side": "ASK_RESISTANCE",
            "pool_price": round(pool_price, px_dec),
            "pool_size": ask_above[1],
        }

    signal = {
        "symbol": symbol,
        "time": now_ts(),
        "entry": round(entry, px_dec),
        "tp": round(tp, px_dec),
        "sl": round(sl, px_dec),
        "side": side,
        "spot": round(spot, px_dec),
        "ema200": round(ctx['ema200'][-1], px_dec) if not np.isnan(ctx['ema200'][-1]) else None,
        "k": None if math.isnan(ctx['stoch_k'][-1]) else round(float(ctx['stoch_k'][-1]), 2),
        "d": None if math.isnan(ctx['stoch_d'][-1]) else round(float(ctx['stoch_d'][-1]), 2),
        "atr": None if math.isnan(ctx['atr'][-1]) else round(float(ctx['atr'][-1]), px_dec),
        "context": context,
        "bids_pools": [(round(p, px_dec), round(v, 2)) for p,v in bids_sig[:5]],
        "asks_pools": [(round(p, px_dec), round(v, 2)) for p,v in asks_sig[:5]],
    }
    return signal

# =============================
# ===== Telegram + Logging =====
# =============================

def send_telegram(text: str, use_html: bool = True):
    """Send Telegram message. Defaults to HTML parse mode with proper escaping to avoid Markdown entity errors."""
    if not CONFIG.telegram_bot_token or not CONFIG.telegram_chat_id:
        print("[WARN] Missing Telegram credentials.")
        return
    url = f"https://api.telegram.org/bot{CONFIG.telegram_bot_token}/sendMessage"
    payload = {"chat_id": CONFIG.telegram_chat_id, "text": text}
    if use_html:
        payload["parse_mode"] = "HTML"
        payload["disable_web_page_preview"] = True
    try:
        r = requests.post(url, json=payload, timeout=10)
        if r.status_code != 200:
            print("[WARN] Telegram send failed:", r.text)
    except Exception as e:
        print("[WARN] Telegram send exception:", e)


_last_alert_time: Dict[str, float] = {}

def cooled_down(symbol: str) -> bool:
    t = _last_alert_time.get(symbol, 0)
    return (now_ts() - t) >= CONFIG.cooldown_minutes * 60


def mark_alert(symbol: str):
    _last_alert_time[symbol] = now_ts()


def log_signal(sig: Dict):
    hdr = [
        "time","symbol","side","entry","tp","sl","spot","ema200","k","d","atr",
        "pool_side","pool_price","pool_size","bids_pools","asks_pools"
    ]
    new_file = not os.path.exists(CONFIG.log_csv)
    with open(CONFIG.log_csv, "a", newline="") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(hdr)
        w.writerow([
            fmt_ts(sig['time']), sig['symbol'], sig['side'], sig['entry'], sig['tp'], sig['sl'], sig['spot'],
            sig['ema200'], sig['k'], sig['d'], sig['atr'], sig['context']['pool_side'], sig['context']['pool_price'],
            round(sig['context']['pool_size'],2), json.dumps(sig['bids_pools']), json.dumps(sig['asks_pools'])
        ])


def format_alert(sig: Dict) -> str:
    tz_time = fmt_ts(sig['time'])

    # Safe escaper for HTML parse mode
    def esc(x):
        return html.escape(str(x)) if x is not None else ""

    side = esc(sig['side'])
    sym = esc(sig['symbol'])
    pool = sig['context']

    entry = esc(sig['entry'])
    tp = esc(sig['tp'])
    sl = esc(sig['sl'])
    spot = esc(sig['spot'])
    ema200 = esc(sig['ema200'])
    k = esc(sig['k'])
    d = esc(sig['d'])

    pool_side = esc(pool['pool_side'])
    pool_price = esc(pool['pool_price'])
    pool_size = esc(round(pool['pool_size'], 2))
    bids_pools = esc(sig['bids_pools'])
    asks_pools = esc(sig['asks_pools'])
    tz_time_e = esc(tz_time)

    lines = [
        f"<b>{side}</b> — {sym}",
        f"Time: {tz_time_e}",
        f"Entry: <code>{entry}</code>  TP: <code>{tp}</code>  SL (pool-based): <code>{sl}</code>",
        f"Spot: <code>{spot}</code>  EMA200: <code>{ema200}</code>",
        f"StochRSI K/D: <code>{k}</code> / <code>{d}</code>",
        f"Pool: <b>{pool_side}</b> at <code>{pool_price}</code> (size≈{pool_size})",
        f"Top Bids Pools: <code>{bids_pools}</code>",
        f"Top Asks Pools: <code>{asks_pools}</code>",
        f"Mode: pools-as-SL, TP={html.escape(CONFIG.tp_mode)}",
    ]

    text = "".join(lines)
    return text

# =============================
# ===== Runner ================
# =============================

def scan_symbol(symbol: str):
    try:
        if not cooled_down(symbol):
            return
        sig = build_signal(symbol)
        if sig is None:
            return
        # Only alert if entry between pool and opposite side (sanity)
        # (Optional extra guards can be added here)
        send_telegram(format_alert(sig))
        log_signal(sig)
        mark_alert(symbol)
        print(f"Alert sent: {symbol} {sig['side']} at {fmt_ts(sig['time'])}")
    except Exception as e:
        print(f"[ERR] scan_symbol({symbol}): {e}")


def worker_loop():
    # Concurrency guard to avoid hammering Coinbase and hitting 429
    max_threads = 4
    sem = threading.Semaphore(max_threads)
    while True:
        threads = []
        for sym in CONFIG.symbols:
            def _run(s=sym):
                with sem:
                    scan_symbol(s)
                    time.sleep(0.2)
            t = threading.Thread(target=_run, daemon=True)
            t.start()
            threads.append(t)
        for t in threads:
            t.join()
        time.sleep(CONFIG.scan_interval_sec)


if __name__ == "__main__":
    # Filter out symbols that aren't supported anywhere to reduce errors
    supported = []
    for s in CONFIG.symbols:
        ex = resolve_market_exchange(s)
        if ex is None:
            print(f"[WARN] Skipping unsupported symbol {s} on all configured exchanges")
        else:
            supported.append(s)
    CONFIG.symbols = supported
    print("Starting Liquidity Map Telegram Alert Bot (Pools-as-SL)")
    print(f"Symbols: {CONFIG.symbols} | Exchanges: {[ex.id for ex in EXCHANGES]}")
    print(f"Scan interval: {CONFIG.scan_interval_sec}s | Cooldown: {CONFIG.cooldown_minutes}m")
    worker_loop()
