import requests

# 👇 Your list of desired symbols
your_list = [
    "BTCUSDT", "ETHUSDT", "XRPUSDT", "LTCUSDT", "ADAUSDT",
    "DOGEUSDT", "SOLUSDT", "WIFUSDT", "ONDOUSDT", "SEIUSDT",
    "MAGICUSDT", "APEUSDT", "JASMYUSDT", "WLDUSDT", "SYRUPUSDT",
    "AEROUSDT", "LINKUSDT", "HBARUSDT", "AAVEUSDT", "FETUSDT",
    "CRVUSDT", "AVAXUSDT", "XCNUSDT", "UNIUSDT", "MKRUSDT",
    "TOSHIUSDT", "NEARUSDT", "ALGOUSDT", "TRUMPUSDT", "BCHUSDT",
    "INJUSDT", "PEPEUSDT", "XLMUSDT", "MOODENGUSDT", "BONKUSDT",
    "DOTUSDT", "POPCATUSDT", "ARBUSDT", "ICPUSDT", "QNTUSDT",
    "TIAUSDT", "IPUSDT", "PNUTUSDT", "APTUSDT", "VETUSDT",
    "ENAUSDT", "TURBOUSDT", "BERAUSDT", "POLUSDT", "MASKUSDT",
    "ACHUSDT", "PYTHUSDT", "SANDUSDT", "MORPHOUSDT", "MANAUSDT",
    "COTIUSDT", "AXSUSDT"
]

def get_all_bybit_usdt_futures_symbols():
    base_url = "https://api.bybit.com/v5/market/instruments-info"
    params = {"category": "linear"}
    all_symbols = set()

    while True:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        data = response.json()

        if data["retCode"] != 0 or "list" not in data["result"]:
            raise Exception(f"Unexpected response format: {data}")

        all_symbols.update(item["symbol"] for item in data["result"]["list"])

        cursor = data["result"].get("nextPageCursor")
        if not cursor:
            break  # no more pages

        params["cursor"] = cursor

    return all_symbols

def find_missing_symbols(user_list, available_symbols):
    user_symbols = {s.replace("-", "").upper() for s in user_list}
    missing = user_symbols - available_symbols
    return sorted(missing)

def main():
    print("🔍 Fetching all Bybit USDT Perpetual symbols...")
    available = get_all_bybit_usdt_futures_symbols()
    print(f"✅ Retrieved {len(available)} symbols.")

    missing = find_missing_symbols(your_list, available)

    if missing:
        print(f"\n❌ Missing from Bybit Futures ({len(missing)}):")
        for symbol in missing:
            print(f" - {symbol}")
    else:
        print("🎉 All your symbols are available on Bybit Futures!")

if __name__ == "__main__":
    main()
