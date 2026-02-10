import requests, pandas as pd, numpy as np, time, joblib
from datetime import datetime, timedelta, timezone
from pathlib import Path
from ta.volatility import BollingerBands, AverageTrueRange
from ta.momentum import RSIIndicator
from ta.trend import ADXIndicator, MACD
from lightgbm import LGBMClassifier
from sklearn.utils.class_weight import compute_sample_weight

# === CONFIG ===
TELEGRAM_TOKEN = "8177096945:AAHTg5nxVTA6hcPkAL4MLAAHPsZ1at7Ywmw"
TELEGRAM_CHAT_ID = "7967738614"
COINS = [
    "btc-usd", "eth-usd", "xrp-usd", "ltc-usd", "ada-usd", "doge-usd",
    "sol-usd", "wif-usd", "ondo-usd", "sei-usd", "magic-usd", "ape-usd",
    "jasmy-usd", "wld-usd", "syrup-usd", "fartcoin-usd", "aero-usd",
    "link-usd", "hbar-usd", "aave-usd", "fet-usd", "crv-usd", "tao-usd",
    "avax-usd", "xcn-usd", "uni-usd", "mkr-usd", "toshi-usd", "near-usd",
    "algo-usd", "trump-usd", "bch-usd", "inj-usd", "pepe-usd", "xlm-usd",
    "moodeng-usd", "bonk-usd", "dot-usd", "popcat-usd", "arb-usd", "icp-usd",
    "qnt-usd", "tia-usd", "ip-usd", "pnut-usd", "apt-usd", "ena-usd", "turbo-usd",
    "bera-usd", "pol-usd", "mask-usd", "pyth-usd", "sand-usd", "morpho-usd",
    "mana-usd", "coti-usd", "c98-usd", "axs-usd"
]


LOOKAHEAD_CANDLES = 3
ATR_MULT_TARGET = 1.3
LAST_TRAIN_FILE = Path("last_train.txt")
cooldowns = {}
now = datetime.now(timezone.utc)
# ML model files
MODEL_PATH_BREAKOUT = Path("breakout_model_1hr.pkl")
MODEL_PATH_IMMINENT = Path("imminent_breakout_model_1hr.pkl")
MODEL_PATH_TREND = Path("trend_follow_model_1hr.pkl")
MODEL_PATH_MEANREV = Path("mean_reversion_model_1hr.pkl")

# ML confidence thresholds
TREND_HIST=0.65
BREAKOUT_THRESHOLD = 0.7
IMMINENT_THRESHOLD = 0.6
TREND_THRESHOLD = 0.70
MEANREV_THRESHOLD = 0.6

def format_price(v):
    return f"${v:.2f}" if v >= 100 else f"${v:.4f}" if v >= 1 else f"${v:.6f}"

def fetch_data(symbol, granularity='3600', limit=4000):
    """
    Fetch OHLCV candles from Coinbase with real pagination.
    Works for large history by requesting 300 candles per call.
    
    symbol: "btc-usd"
    granularity: seconds per candle (900=15m)
    limit: total candles desired
    """
    try:
        granularity = int(granularity)
        max_per_request = 300
        all_dfs = []

        # Start from now, work backwards
        end_time = datetime.now(timezone.utc)

        while len(all_dfs) * max_per_request < limit:
            start_time = end_time - timedelta(seconds=granularity * max_per_request)

            url = f"https://api.exchange.coinbase.com/products/{symbol}/candles"
            params = {
                "granularity": granularity,
                "start": start_time.isoformat(),
                "end": end_time.isoformat()
            }

            r = requests.get(url, params=params, timeout=10)
            if r.status_code == 429:
                print(f"⚠ Rate limit hit for {symbol}, retrying in 5s...")
                time.sleep(5)
                continue
            elif r.status_code != 200:
                print(f"⚠ Error fetching {symbol}: {r.status_code}")
                break

            df = pd.DataFrame(r.json(), columns=["time", "low", "high", "open", "close", "volume"])
            if df.empty:
                break

            df["time"] = pd.to_datetime(df["time"], unit="s")
            all_dfs.append(df)

            # Move the window back in time
            end_time = start_time

        if not all_dfs:
            return None

        # Merge and sort oldest → newest
        full_df = pd.concat(all_dfs, ignore_index=True)
        full_df = full_df.sort_values("time").reset_index(drop=True)

        # Trim to exact limit requested
        return full_df.tail(limit)

    except Exception as e:
        print(f"Error in fetch_data for {symbol}: {e}")
        return None

def get_historical_win_rate(current_df, label_func, symbol):
    """
    Find the raw historical win rate for setups similar to the current one.
    Uses ~4 weeks of recent history from the exchange for the SAME symbol.
    """
    try:
        # Extract current feature vector
        features_now = extract_features(current_df.iloc[[-1]])

        # Fetch ~4 weeks of 15m candles: 2000 bars ≈ 28 days
        df = fetch_data(symbol, granularity='3600', limit=4000)  # ~83 days

        if df is None or df.empty:
            return None

        df = compute_indicators(df)
        df = label_func(df)
        if df.empty:
            return None

        X_hist = extract_features(df)
        y_hist = df["label"]

        # Similarity filter — same logic as before
        mask = (
            (abs(X_hist["bb_width"] - features_now["bb_width"].iloc[0]) <= 0.15 * features_now["bb_width"].iloc[0]) &
            (abs(X_hist["rsi"] - features_now["rsi"].iloc[0]) <= 5) &
            (abs(X_hist["adx"] - features_now["adx"].iloc[0]) <= 5)
        )

        similar_cases = y_hist[mask]
        if len(similar_cases) == 0:
            return None

        win_rate = (similar_cases == 1).mean()
        return win_rate

    except Exception as e:
        print("Historical win rate error:", e)
        return None

    

def compute_indicators(df):
    # RSI with shorter lookback for H1
    df["rsi"] = RSIIndicator(df["close"], window=7).rsi()
    
    # ATR with shorter lookback for H1
    df["atr"] = AverageTrueRange(df["high"], df["low"], df["close"], window=7).average_true_range()
    df["atr_norm"] = df["atr"] / df["close"]
    
    # Bollinger Bands with shorter lookback for H1
    bb = BollingerBands(df["close"], window=7, window_dev=2)  
    df["bb_upper"] = bb.bollinger_hband()
    df["bb_lower"] = bb.bollinger_lband()
    df["bb_middle"] = bb.bollinger_mavg()
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["close"]
    
    # ADX, MACD, EMA settings can stay or also be adjusted
    df["adx"] = ADXIndicator(df["high"], df["low"], df["close"], window=7).adx()
    df["macd_diff"] = MACD(df["close"]).macd_diff()
    df["ema20"] = df["close"].ewm(span=7).mean()
    df["ema50"] = df["close"].ewm(span=20).mean()
    
    return df


def multi_tf_confirmation(symbol):
    df_6h = fetch_data(symbol, granularity='21600', limit=60)
    if df_6h is None or df_6h.empty:
        return None
    df_6h = compute_indicators(df_6h)
    latest = df_6h.iloc[-1]
    if latest["adx"] > 20:
        return "Bullish" if latest["close"] > latest["bb_middle"] else "Bearish"
    return None

# Breakout (existing)
def label_breakouts(df):
    labels = []
    for i in range(len(df)):
        if i + LOOKAHEAD_CANDLES >= len(df): labels.append(np.nan); continue
        atr = df["atr"].iloc[i]
        price_now = df["close"].iloc[i]
        future_max = df["high"].iloc[i+1:i+LOOKAHEAD_CANDLES].max()
        future_min = df["low"].iloc[i+1:i+LOOKAHEAD_CANDLES].min()
        labels.append(int((future_max - price_now) > ATR_MULT_TARGET * atr or
                          (price_now - future_min) > ATR_MULT_TARGET * atr))
    df["label"] = labels
    return df.dropna()

# Imminent Breakout (existing)
def label_imminent(df):
    labels = []
    for i in range(len(df)):
        if i + LOOKAHEAD_CANDLES >= len(df): labels.append(np.nan); continue
        if df["bb_width"].iloc[i] < df["bb_width"].rolling(50).mean().iloc[i] * 0.85:
            atr = df["atr"].iloc[i]
            price_now = df["close"].iloc[i]
            future_max = df["high"].iloc[i+1:i+LOOKAHEAD_CANDLES].max()
            future_min = df["low"].iloc[i+1:i+LOOKAHEAD_CANDLES].min()
            labels.append(int((future_max - price_now) > ATR_MULT_TARGET * atr or
                              (price_now - future_min) > ATR_MULT_TARGET * atr))
        else:
            labels.append(0)
    df["label"] = labels
    return df.dropna()

# Trend-Follow (new)
def label_trend_follow(df):
    labels = []
    for i in range(len(df)):
        if i + LOOKAHEAD_CANDLES >= len(df): labels.append(np.nan); continue
        bullish = df["ema20"].iloc[i] > df["ema50"].iloc[i] and df["adx"].iloc[i] > 20
        bearish = df["ema20"].iloc[i] < df["ema50"].iloc[i] and df["adx"].iloc[i] > 20
        atr = df["atr"].iloc[i]
        price_now = df["close"].iloc[i]
        future_max = df["high"].iloc[i+1:i+LOOKAHEAD_CANDLES].max()
        future_min = df["low"].iloc[i+1:i+LOOKAHEAD_CANDLES].min()
        if bullish and (future_max - price_now) > ATR_MULT_TARGET * atr:
            labels.append(1)
        elif bearish and (price_now - future_min) > ATR_MULT_TARGET * atr:
            labels.append(1)
        else:
            labels.append(0)
    df["label"] = labels
    return df.dropna()

# Mean Reversion (new)
def label_mean_reversion(df):
    labels = []
    for i in range(len(df)):
        if i + LOOKAHEAD_CANDLES >= len(df): labels.append(np.nan); continue
        overbought = df["rsi"].iloc[i] > 65 and df["close"].iloc[i] > df["bb_upper"].iloc[i]
        oversold = df["rsi"].iloc[i] < 35 and df["close"].iloc[i] < df["bb_lower"].iloc[i]
        mid = df["bb_middle"].iloc[i]
        if overbought:
            labels.append(1 if mid < df["close"].iloc[i] else 0)
        elif oversold:
            labels.append(1 if mid > df["close"].iloc[i] else 0)
        else:
            labels.append(0)
    df["label"] = labels
    return df.dropna()

def extract_features(df):
    f = df[["atr_norm", "bb_width", "rsi", "macd_diff", "adx",
            "volume", "open", "close", "high", "low"]].copy()
    f["volume_ratio"] = df["volume"] / df["volume"].rolling(20).mean()
    f["price_above_mid"] = df["close"] - df["bb_middle"]
    f["price_body_ratio"] = abs(df["close"] - df["open"]) / (df["high"] - df["low"] + 1e-9)
    return f.fillna(0)

def train_model(label_func, save_path, label_name, historical_data):
    all_data, coins_used, coins_skipped, pos_total, neg_total = [], 0, 0, 0, 0

    for coin, df in historical_data.items():
        df_labeled = label_func(df.copy())
        if df_labeled.empty:
            coins_skipped += 1
            continue

        X, y = extract_features(df_labeled), df_labeled["label"].astype(int)
        pos_total += (y == 1).sum()
        neg_total += (y == 0).sum()
        all_data.append((X, y))
        coins_used += 1

    if not all_data:
        print(f"❌ No data for {label_name} model.")
        return False

    X_all = pd.concat([x for x, _ in all_data], ignore_index=True)
    y_all = pd.concat([y for _, y in all_data], ignore_index=True)
    weights = compute_sample_weight(class_weight="balanced", y=y_all)

    model = LGBMClassifier(n_estimators=500, learning_rate=0.05, max_depth=6,
                           subsample=0.8, colsample_bytree=0.8)
    model.fit(X_all, y_all, sample_weight=weights)
    joblib.dump(model, save_path)

    print(f"\n✅ {label_name} Model trained:")
    print(f"   Coins used: {coins_used} / {len(COINS)}")
    print(f"   Examples: {len(X_all)} | Positive: {pos_total} | Negative: {neg_total}")
    return True


def retrain_if_needed():
    now = datetime.now(timezone.utc)

    # Check last retrain date
    if LAST_TRAIN_FILE.exists():
        last_train = datetime.fromisoformat(LAST_TRAIN_FILE.read_text().strip())
        if (now - last_train).days < 7:
            return

    print("🔄 Downloading historical data for all coins once...")
    historical_data = {}
    for coin in COINS:
        df = fetch_data(coin, granularity='3600', limit=4000)  # datetime.now(timezone.utc) for M15 bot
        if df is not None and not df.empty:
            df = compute_indicators(df)
            historical_data[coin] = df
        time.sleep(0.5)  # Avoid 429 rate-limit

    print("🔄 Training all models with cached data...")
    train_model(label_breakouts, MODEL_PATH_BREAKOUT, "Breakout",historical_data)
    train_model(label_imminent, MODEL_PATH_IMMINENT, "Imminent Breakout",historical_data)
    train_model(label_trend_follow, MODEL_PATH_TREND, "Trend Follow",historical_data)
    train_model(label_mean_reversion, MODEL_PATH_MEANREV, "Mean Reversion",historical_data)
    LAST_TRAIN_FILE.write_text(now.isoformat())

def predict_probability(df, model_path):
    if not model_path.exists():
        return None
    model = joblib.load(model_path)
    return model.predict_proba(extract_features(df.iloc[[-1]]))[0][1]

def send_alert(strategy, symbol, side, live_price, tp, sl, prob, hist_win_rate):
    """
    Sends a color-coded alert to Telegram based on ML confidence & historical win rate.
    """

    # Convert to percentages for easier reading
    conf_pct = prob * 100
    hist_pct = hist_win_rate * 100 if hist_win_rate is not None else None

    # === Determine Color Code ===
    if conf_pct >= 75 and (hist_pct is None or hist_pct >= 70):
        color_icon = "🟢 Strong"
    elif conf_pct >= 65 and (hist_pct is None or hist_pct >= 60):
        color_icon = "🟡 Moderate"
    else:
        color_icon = "⚪ Borderline"

    # Format historical win rate
    hist_str = f"{hist_pct:.1f}%" if hist_pct is not None else "N/A"

    # Build Telegram message
    msg = (
        f"{color_icon} *{strategy}* — {symbol.upper()}\n"
        f"Side: {side}\n"
        f"Price: {format_price(live_price)}\n"
        f"TP: {format_price(tp)} | SL: {format_price(sl)}\n"
        f"Confidence: {conf_pct:.1f}%\n"
        f"Historical win rate for similar setups: {hist_str}"
    )

    # Send to Telegram
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    try:
        requests.post(url, data={"chat_id": TELEGRAM_CHAT_ID, "text": msg, "parse_mode": "Markdown"})
    except Exception as e:
        print(f"Telegram error: {e}")


def check_breakout(symbol, df, live_price):
    latest, vol_mean, atr = df.iloc[-1], df["volume"].rolling(20).mean().iloc[-1], df["atr"].iloc[-1]
    squeeze = latest["bb_width"] < df["bb_width"].rolling(50).mean().iloc[-1] * 0.8
    volume_surge = latest["volume"] > 1.5 * vol_mean
    trend_strong = latest["adx"] > 25
    tf_bias = multi_tf_confirmation(symbol)

    if squeeze and volume_surge and trend_strong:
        if latest["close"] > df["high"].rolling(20).max().iloc[-2] and tf_bias == "Bullish":
            direction = "Long"
        elif latest["close"] < df["low"].rolling(20).min().iloc[-2] and tf_bias == "Bearish":
            direction = "Short"
        else:
            return

        prob = predict_probability(df, MODEL_PATH_BREAKOUT)
        if prob and prob >= BREAKOUT_THRESHOLD:
            hist_win_rate = get_historical_win_rate(df, label_breakouts,symbol)
            tp = live_price + 2 * atr if direction == "Long" else live_price - 2 * atr
            sl = live_price - 1.2 * atr if direction == "Long" else live_price + 1.2 * atr
            send_alert("Breakout Alert", symbol, direction, live_price, tp, sl, prob, hist_win_rate)


def check_breakout_imminent(symbol, df, live_price):
    latest, atr = df.iloc[-1], df["atr"].iloc[-1]
    band_press_upper = latest["close"] > latest["bb_middle"] and latest["close"] > (latest["bb_upper"] - 0.25 * atr)
    band_press_lower = latest["close"] < latest["bb_middle"] and latest["close"] < (latest["bb_lower"] + 0.25 * atr)
    accumulation = df["volume"].rolling(3).mean().iloc[-1] > 0.9 * df["volume"].rolling(10).mean().iloc[-1]
    tf_bias = multi_tf_confirmation(symbol)

    if latest["bb_width"] < df["bb_width"].rolling(50).mean().iloc[-1] * 0.85 and accumulation:
        if band_press_upper and tf_bias == "Bullish":
            direction = "Bullish"
        elif band_press_lower and tf_bias == "Bearish":
            direction = "Bearish"
        else:
            return

        prob = predict_probability(df, MODEL_PATH_IMMINENT)
        if prob and prob >= IMMINENT_THRESHOLD:
            hist_win_rate = get_historical_win_rate(df, label_imminent, symbol)
            tp = live_price + 1.8 * atr if direction == "Bullish" else live_price - 1.8 * atr
            sl = live_price - 1.0 * atr if direction == "Bullish" else live_price + 1.0 * atr
            send_alert("Breakout Imminent", symbol, direction, live_price, tp, sl, prob, hist_win_rate)


def check_trend_follow(symbol, df, live_price):
    hist_win_rate = None
    latest, atr = df.iloc[-1], df["atr"].iloc[-1]
    bullish_pullback = latest["ema20"] > latest["ema50"] and latest["adx"] > 20 and latest["rsi"] < 60
    bearish_pullback = latest["ema20"] < latest["ema50"] and latest["adx"] > 20 and latest["rsi"] > 40

    if bullish_pullback:
        direction = "Long"
    elif bearish_pullback:
        direction = "Short"
    else:
        return

    prob = predict_probability(df, MODEL_PATH_TREND)
    hist_win_rate = get_historical_win_rate(df, label_trend_follow, symbol)
    
    if (
    (prob and hist_win_rate and prob >= TREND_THRESHOLD and hist_win_rate >= TREND_HIST)  # Case 1
    or (hist_win_rate and hist_win_rate > .70 and prob and prob >= .60)  # Case 2
    or (prob and prob >= .90)  # Case 3
    ):

        
        tp = live_price + 1.5 * atr if direction == "Long" else live_price - 1.5 * atr
        sl = live_price - 1.0 * atr if direction == "Long" else live_price + 1.0 * atr
        send_alert("Trend Follow", symbol, direction, live_price, tp, sl, prob, hist_win_rate)



def check_mean_reversion(symbol, df, live_price):
    latest, atr = df.iloc[-1], df["atr"].iloc[-1]
    overbought = latest["rsi"] > 65 and latest["close"] > latest["bb_upper"]
    oversold = latest["rsi"] < 35 and latest["close"] < latest["bb_lower"]

    if overbought:
        direction = "Short"
    elif oversold:
        direction = "Long"
    else:
        return

    prob = predict_probability(df, MODEL_PATH_MEANREV)
    if prob and prob >= MEANREV_THRESHOLD:
        hist_win_rate = get_historical_win_rate(df, label_mean_reversion, symbol)
        tp = live_price - 1.2 * atr if direction == "Short" else live_price + 1.2 * atr
        sl = live_price + 1.0 * atr if direction == "Short" else live_price - 1.0 * atr
        send_alert("Mean Reversion", symbol, direction, live_price, tp, sl, prob, hist_win_rate)


def run_alerts():
    now = datetime.now(timezone.utc)

    for symbol in COINS:
        # Cooldown: avoid duplicate alerts for the same coin in <30 min
        if symbol in cooldowns and (now - cooldowns[symbol]).total_seconds() < 1800:
            continue

        # Fetch recent market data
        df = fetch_data(symbol, granularity='datetime.now(timezone.utc)', limit=4000)
        if df is None or df.empty:
            continue

        # Calculate indicators
        df = compute_indicators(df)
        live_price = df["close"].iloc[-1]

        # Run all 4 strategy checks in parallel
        check_breakout(symbol, df, live_price)
        check_breakout_imminent(symbol, df, live_price)
        check_trend_follow(symbol, df, live_price)
        check_mean_reversion(symbol, df, live_price)

        # Store cooldown time
        cooldowns[symbol] = now


if __name__ == "__main__":
    print("🚀 Starting Multi-Strategy AI Crypto Bot...")
    retrain_if_needed()  # Auto-retrain weekly

    while True:
        run_alerts()
        time.sleep(300)  # Wait 5 min before next scan
