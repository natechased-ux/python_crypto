import time
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

COINBASE_REST = "https://api.exchange.coinbase.com"
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "top-pairs-script/1.1"})

# ---- CONFIG ----
QUOTE = "USD"          # build list of *-USD pairs
TOP_N = 250
MAX_WORKERS = 10
REQUEST_SLEEP = 0.08
TIMEOUT = 15

# Stablecoins (do NOT include fiat USD here)
STABLECOINS = {
    "USDC", "USDT", "DAI", "TUSD", "USDP", "BUSD", "FDUSD",
    "PYUSD", "EURC", "GUSD", "FRAX", "LUSD", "SUSD",
}

FIAT = {"USD", "EUR", "GBP"}  # expand if you want

def get_json(url, params=None):
    r = SESSION.get(url, params=params, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()

def list_products():
    products = get_json(f"{COINBASE_REST}/products")
    out = []
    for p in products:
        if str(p.get("status", "")).lower() != "online":
            continue
        if p.get("trading_disabled") is True:
            continue

        pid = str(p.get("id", "")).upper()
        base = str(p.get("base_currency", "")).upper()
        quote = str(p.get("quote_currency", "")).upper()

        # only USD-quoted (or whatever QUOTE is)
        if quote != QUOTE:
            continue

        # Exclude stablecoin BASES (so you don't get USDC-USD, DAI-USD, etc.)
        if base in STABLECOINS:
            continue

        # Optional: if you ever set QUOTE to a stablecoin, exclude it too
        # (but for USD this is false anyway)
        if quote in STABLECOINS:
            continue

        # If you only want fiat quote currencies, enforce it:
        if QUOTE in FIAT and quote not in FIAT:
            continue

        out.append((pid, base, quote))

    return out

def fetch_stats(pid):
    stats = get_json(f"{COINBASE_REST}/products/{pid}/stats")
    vol_base_24h = float(stats.get("volume", 0.0) or 0.0)  # base units
    last = float(stats.get("last", 0.0) or 0.0)            # quote per base
    usd_notional_24h = vol_base_24h * last                 # approx quote notional
    return pid, vol_base_24h, last, usd_notional_24h

def main():
    pairs = list_products()
    print(f"Found {len(pairs)} {QUOTE}-quote non-stablecoin-base products.")

    results = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(fetch_stats, pid): pid for pid, _, _ in pairs}
        for fut in as_completed(futures):
            pid = futures[fut]
            try:
                results.append(fut.result())
            except Exception:
                pass
            time.sleep(REQUEST_SLEEP)

    results.sort(key=lambda x: x[3], reverse=True)
    top = results[:TOP_N]

    print("\nTop by approx 24h USD notional:")
    for i, (pid, vol_base, last, usd_notional) in enumerate(top, 1):
        print(f"{i:>3}. {pid:<12}  usd_24h≈{usd_notional:,.0f}  (vol={vol_base:,.2f} @ last={last:,.4f})")

    top_list = [pid for pid, _, _, _ in top]
    print("\n\n# Paste into your scanner:\nTOP_PAIRS = " + repr(top_list))

if __name__ == "__main__":
    main()
