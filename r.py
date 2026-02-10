import ccxt
ex = ccxt.deribit()
markets = ex.load_markets()
print(list(markets.keys())[:20])
