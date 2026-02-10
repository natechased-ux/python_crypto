import requests

# 1️⃣ Define stablecoins to filter out
STABLE = {"USDC","USDT","DAI","BUSD","FRAX","TUSD","USDP","GUSD","LUSD"}

# 2️⃣ Get all Coinbase products
coinbase_products = requests.get("https://api.exchange.coinbase.com/products").json()

# Extract tradable base currencies
coinbase_assets = {p["base_currency"] for p in coinbase_products if p["status"]=="online"}
coinbase_assets -= STABLE  # remove stablecoins

# 3️⃣ Get top coins by market cap from CoinGecko
cg = requests.get(
    "https://api.coingecko.com/api/v3/coins/markets",
    params={"vs_currency": "usd", "order": "market_cap_desc", "per_page": 200, "page": 1}
).json()

# 4️⃣ Filter to only those on Coinbase
top_coinbase = []
for c in cg:
    symbol = c["symbol"].upper()
    if symbol in coinbase_assets:
        top_coinbase.append({
            "symbol": symbol,
            "coinbase_id": f"{symbol}-USD"
        })
    if len(top_coinbase) >= 50:
        break

# 5️⃣ Output ready-to-use array
import json
print(json.dumps(top_coinbase, indent=2))
