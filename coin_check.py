import requests
import time

PRODUCTS_URL = "https://api.exchange.coinbase.com/products"
CANDLES_URL = "https://api.exchange.coinbase.com/products/{pair}/candles"

# Stablecoins to exclude
STABLES = {"USDC", "USDT", "DAI", "PYUSD", "EUR", "GBP"}

# Minimum candles required for your bot
MIN_1H_CANDLES = 200   # adjust as needed
MIN_1D_CANDLES = 200   # ditto


def is_stablecoin_pair(pair: str) -> bool:
    """
    Returns True if the quoted asset is a stablecoin.
    Example: "BTC-USD" -> USD is stable => skip
    """
    try:
        base, quote = pair.split("-")
        return quote in STABLES
    except:
        return True


def fetch_candles(pair: str, granularity: int):
    """
    Fetch candle data for the given pair and granularity.
    Granularity (in seconds): 3600 = 1h, 86400 = 1d
    """
    url = CANDLES_URL.format(pair=pair)
    params = {"granularity": granularity}
    resp = requests.get(url, params=params)

    if resp.status_code != 200:
        return None

    try:
        return resp.json()
    except:
        return None


def has_enough_candles(pair: str) -> bool:
    """
    Ensures a pair has enough historical candles for both
    1H and 1D analysis.
    """

    candles_1h = fetch_candles(pair, 3600)
    time.sleep(0.15)  # avoid API rate limits

    candles_1d = fetch_candles(pair, 86400)
    time.sleep(0.15)

    if not candles_1h or not candles_1d:
        return False

    # Length must meet your bot's minimum requirements
    return (len(candles_1h) >= MIN_1H_CANDLES and
            len(candles_1d) >= MIN_1D_CANDLES)


def discover_valid_pairs():
    """
    Fetch all Coinbase products, remove stablecoins,
    and test each for candle history.
    """
    resp = requests.get(PRODUCTS_URL)

    if resp.status_code != 200:
        print("ERROR: Could not fetch product list.")
        return [], []

    all_products = resp.json()

    all_pairs = [p["id"] for p in all_products if "id" in p]

    print(f"Total raw Coinbase pairs: {len(all_pairs)}")

    # Filter 1: exclude stablecoin-quoted markets
    filtered_pairs = [p for p in all_pairs if not is_stablecoin_pair(p)]

    print(f"After removing stablecoins: {len(filtered_pairs)} remaining.\n")

    valid = []
    invalid = []

    for pair in filtered_pairs:
        print(f"Checking {pair}...")

        if has_enough_candles(pair):
            print(f" ✓ Valid for trading model")
            valid.append(pair)
        else:
            print(f" ✗ Not enough candles")
            invalid.append(pair)

        print("")  # spacing

    return valid, invalid


if __name__ == "__main__":
    valid, invalid = discover_valid_pairs()

    print("\n==========================")
    print("VALID TRADING PAIRS")
    print("==========================")
    for v in valid:
        print(v)

    print("\n==========================")
    print("INVALID PAIRS")
    print("==========================")
    for iv in invalid:
        print(iv)

    print("\nDone.")
