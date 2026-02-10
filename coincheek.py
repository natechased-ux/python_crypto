import requests

pairs = [
    "BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD", "DOGE-USD",
    "ADA-USD", "AVAX-USD", "LINK-USD", "MATIC-USD", "DOT-USD",
    "LTC-USD", "BCH-USD", "XLM-USD", "FIL-USD", "ATOM-USD",
    "APT-USD", "SUI-USD", "ARB-USD", "OP-USD", "HBAR-USD",
    "AAVE-USD", "MKR-USD", "CRV-USD", "UNI-USD", "RPL-USD",
    "ICP-USD", "NEAR-USD", "ALGO-USD", "QNT-USD", "EGLD-USD",
    "IMX-USD", "RNDR-USD", "INJ-USD", "FTM-USD", "GRT-USD",
    "PEPE-USD", "BONK-USD", "TIA-USD", "SEI-USD", "PYTH-USD",
    "JTO-USD", "MINA-USD", "ZEC-USD", "ETC-USD", "CRO-USD",
    "ANKR-USD", "SKL-USD", "ONT-USD", "OCEAN-USD", "NKN-USD"
]

PUBLIC_API = "https://api.exchange.coinbase.com/products"

valid_pairs = []
invalid_pairs = []

for pair in pairs:
    url = f"{PUBLIC_API}/{pair}"
    response = requests.get(url)

    if response.status_code == 200:
        valid_pairs.append(pair)
    else:
        invalid_pairs.append(pair)

print("\nVALID PAIRS:")
print(valid_pairs)

print("\nINVALID PAIRS:")
print(invalid_pairs)
