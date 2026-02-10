import requests

def get_current_price(symbol):
    try:
        url = f"https://api.exchange.coinbase.com/products/{symbol}/ticker"
        response = requests.get(url).json()
        if "price" in response:
            return float(response["price"])
        else:
            return None
    except Exception:
        return None

# List of coins to test (Coinbase symbols: lowercase and dash format)
coins = ["btc-usd", "eth-usd", "xrp-usd", "ltc-usd", "ada-usd",
           "doge-usd", "sol-usd", "wif-usd", "ondo-usd",
           "sei-usd", "magic-usd", "ape-usd",
           "fartcoin-usd", "aero-usd", "link-usd","hbar-usd",
           "aave-usd", "fet-usd", "crv-usd","tao-usd", "avax-usd", "xcn-usd",
           "uni-usd", "mkr-usd", "toshi-usd", "near-usd", "algo-usd", "trump-usd",
           "bch-usd", "inj-usd", "pepe-usd", "xlm-usd", "moodeng-usd", "bonk-usd",
           "dot-usd", "popcat-usd", "arb-usd", "icp-usd",
           "qnt-usd", "tia-usd", "ip-usd" , "pnut-usd", "apt-usd", "vet-usd",
           "ena-usd", "turbo-usd", "bera-usd", "pol-usd", "mask-usd", "ach-usd",
           "pyth-usd", "sand-usd", "morpho-usd", "mana-usd", "coti-usd",
           "axs-usd"]

working = []
not_working = []

for coin in coins:
    price = get_current_price(coin)
    if price is not None:
        print(f"[✓] {coin.upper()} - Current Price: ${price}")
        working.append(coin)
    else:
        print(f"[✗] {coin.upper()} - Not available on Coinbase")
        not_working.append(coin)

# Optional: Print summaries
print("\n--- Summary ---")
print(f"Working coins: {working}")
print(f"Not working coins: {not_working}")
