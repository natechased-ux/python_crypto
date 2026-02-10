import requests

def get_valid_binance_symbols():
    """Fetch all valid symbols from Binance API."""
    url = "https://api.binance.com/api/v3/exchangeInfo"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        valid_symbols = {symbol['symbol'] for symbol in data['symbols']}
        return valid_symbols
    else:
        raise Exception("Failed to fetch Binance symbols")

# Your symbols
symbols_binance = [
    "BTCUSDT", "ETHUSDT", "XRPUSDT", "LTCUSDT", "ADAUSDT",
    "DOGEUSDT", "SOLUSDT", "WIFUSDT", "ONDOUSDT", "SEIUSDT",
    "MAGICUSDT", "APEUSDT", "JASMYUSDT", "WLDUSDT", "SYRUPUSDT",
    "FARTCOINUSDT", "AEROUSDT", "LINKUSDT", "HBARUSDT", "AAVEUSDT",
    "FETUSDT", "CRVUSDT", "TAOUSDT", "AVAXUSDT", "XCNUSDT", "UNIUSDT",
    "MKRUSDT", "TOSHIUSDT", "NEARUSDT", "ALGOUSDT", "TRUMPUSDT",
    "BCHUSDT", "INJUSDT", "PEPEUSDT", "XLMUSDT", "MOODENGUSDT", "BONKUSDT",
    "DOTUSDT", "POPCATUSDT", "ARBUSDT", "ICPUSDT", "QNTUSDT", "TIAUSDT",
    "IPUSDT", "PNUTUSDT", "APTUSDT", "VETUSDT", "ENAUSDT", "TURBOUSDT",
    "BERAUSDT", "POLUSDT", "MASKUSDT", "ACHUSDT", "PYTHUSDT", "SANDUSDT",
    "MORPHOUSDT", "MANAUSDT", "VELOUSDT", "COTIUSDT", "AXSUSDT"
]

# Filter only valid symbols
try:
    valid_symbols = get_valid_binance_symbols()
    symbols_binance = [symbol for symbol in symbols_binance if symbol in valid_symbols]
    print(f"Valid symbols: {symbols_binance}")
except Exception as e:
    print(f"Error: {str(e)}")
