import requests
import pandas as pd
from datetime import datetime

# --- SETTINGS ---
COINS = ["btc-usd", "eth-usd", "xrp-usd", "ltc-usd", "ada-usd",
         "doge-usd", "wif-usd", "ondo-usd", "magic-usd", "ape-usd",
         "jasmy-usd", "wld-usd", "syrup-usd", "fartcoin-usd", "aero-usd",
         "link-usd", "hbar-usd", "aave-usd", "fet-usd", "crv-usd", "tao-usd",
         "avax-usd", "xcn-usd", "uni-usd", "toshi-usd", "near-usd", "algo-usd",
         "trump-usd", "bch-usd", "inj-usd", "xlm-usd", "moodeng-usd", "bonk-usd",
         "dot-usd", "popcat-usd", "arb-usd", "icp-usd", "tia-usd", "ip-usd",
         "pnut-usd", "apt-usd", "vet-usd", "ena-usd", "turbo-usd", "bera-usd",
         "pol-usd", "mask-usd", "ach-usd", "pyth-usd", "sand-usd", "morpho-usd",
         "mana-usd", "velo-usd", "coti-usd", "axs-usd"]
BASE_URL = "https://api.exchange.coinbase.com/products/{}/candles?granularity=21600"

# --- FUNCTIONS ---
def fetch_ohlcv(symbol):
    url = BASE_URL.format(symbol)
    try:
        r = requests.get(url)
        data = r.json()
        df = pd.DataFrame(data, columns=["time", "low", "high", "open", "close", "volume"])
        df["time"] = pd.to_datetime(df["time"], unit="s")
        return df.sort_values("time").tail(300).reset_index(drop=True)
    except Exception as e:
        print(f"Error fetching {symbol}: {e}")
        return None

def apply_indicators(df):
    df['ma'] = df['close'].rolling(20).mean()
    df['std'] = df['close'].rolling(20).std()
    df['upper'] = df['ma'] + 2 * df['std']
    df['lower'] = df['ma'] - 2 * df['std']
    df['bandwidth'] = df['upper'] - df['lower']
    df['H-L'] = df['high'] - df['low']
    df['H-PC'] = abs(df['high'] - df['close'].shift(1))
    df['L-PC'] = abs(df['low'] - df['close'].shift(1))
    df['TR'] = df[['H-L', 'H-PC', 'L-PC']].max(axis=1)
    df['ATR'] = df['TR'].rolling(14).mean()
    return df

def backtest(df):
    in_trade = False
    entry = sl = tp = direction = None
    wins = losses = 0
    profit = 0.0

    for i in range(21, len(df)):
        row = df.iloc[i]
        prev = df.iloc[i - 1]
        min_bw = df['bandwidth'].iloc[i-168:i].min()

        if not in_trade:
            if row['bandwidth'] <= min_bw * 1.05:
                if row['close'] > row['upper']:
                    direction = 'long'
                    entry = row['close']
                    sl = entry - 1.5 * row['ATR']
                    tp = entry + 3 * row['ATR']
                    in_trade = True
                elif row['close'] < row['lower']:
                    direction = 'short'
                    entry = row['close']
                    sl = entry + 1.5 * row['ATR']
                    tp = entry - 3 * row['ATR']
                    in_trade = True
        else:
            price = row['close']
            if direction == 'long':
                if price >= tp:
                    profit += tp - entry
                    wins += 1
                    in_trade = False
                elif price <= sl:
                    profit -= entry - sl
                    losses += 1
                    in_trade = False
            elif direction == 'short':
                if price <= tp:
                    profit += entry - tp
                    wins += 1
                    in_trade = False
                elif price >= sl:
                    profit -= sl - entry
                    losses += 1
                    in_trade = False

    return wins, losses, profit

# --- RUN BACKTEST ---
summary = []
for coin in COINS:
    df = fetch_ohlcv(coin)
    if df is None or len(df) < 200:
        continue
    df = apply_indicators(df)
    wins, losses, profit = backtest(df)
    summary.append({
        "Coin": coin,
        "Wins": wins,
        "Losses": losses,
        "Total Trades": wins + losses,
        "Net Profit ($1 per move)": round(profit, 2),
        "Win Rate": round(wins / (wins + losses) * 100, 2) if wins + losses > 0 else 0.0
    })

results = pd.DataFrame(summary)
results = results.sort_values(by="Net Profit ($1 per move)", ascending=False)
print(results.to_string(index=False))
