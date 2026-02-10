import requests
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


# --- User settings ---
COINS = ['BTC-USD', 'ETH-USD', 'XRP-USD', 'WIF-USD', 'XCN-USD',
         'ONDO-USD', 'ENA-USD', 'WLD-USD', 'SEI-USD', 'DOGE-USD', 'SUI-USD']  # Add or modify your coins here
LOOKBACK_CANDLES = 300
LOOKAHEAD = 4  # periods ahead for future price / classification
PRICE_COL = 'close'
THRESHOLD = 0.002  # 0.2% return threshold for classification
MIN_PREDICTED_CHANGE = 0.005  # 0.5%
# Telegram
TELEGRAM_TOKEN = '8177096945:AAHTg5nxVTA6hcPkAL4MLAAHPsZ1at7Ywmw'
CHAT_ID = '7967738614'


# Coinbase API URL template for candles
COINBASE_CANDLE_URL = 'https://api.exchange.coinbase.com/products/{}/candles'

def fetch_current_price(coin):
    try:
        url = f'https://api.exchange.coinbase.com/products/{coin}/ticker'
        response = requests.get(url)
        response.raise_for_status()
        return float(response.json()['price'])
    except Exception as e:
        print(f"Error fetching current price for {coin}: {e}")
        return None


def fetch_historical_data(coin, granularity=900, limit=300):
    """
    Fetch historical candle data from Coinbase API.
    granularity: seconds per candle (3600 = 1 hour)
    limit: number of candles to fetch
    Returns DataFrame with columns: time, low, high, open, close, volume
    """
    params = {
        'granularity': granularity,
    }
    end_time = datetime.utcnow()
    start_time = end_time - timedelta(seconds=granularity*limit)
    params['start'] = start_time.isoformat()
    params['end'] = end_time.isoformat()

    url = COINBASE_CANDLE_URL.format(coin)
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        # Coinbase returns list of [time, low, high, open, close, volume]
        df = pd.DataFrame(data, columns=['time', 'low', 'high', 'open', 'close', 'volume'])
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df = df.sort_values('time').reset_index(drop=True)
        return df
    except Exception as e:
        print(f"Error fetching data for {coin}: {e}")
        return pd.DataFrame()

def compute_RSI(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def fetch_combined_data(coins, lookback=LOOKBACK_CANDLES, lookahead=LOOKAHEAD, price_col=PRICE_COL, threshold=THRESHOLD):
    all_data = []
    for coin in coins:
        df = fetch_historical_data(coin, limit=lookback)
        if df.empty:
            print(f"No data for {coin}, skipping.")
            continue

        df['return_1'] = df[price_col].pct_change()
        df['SMA_10'] = df[price_col].rolling(window=10, min_periods=1).mean()
        df['SMA_50'] = df[price_col].rolling(window=50, min_periods=1).mean()
        df['RSI_14'] = compute_RSI(df[price_col])
        df['volume'] = df['volume']

        df['future_price'] = df[price_col].shift(-lookahead)
        df['future_return'] = (df['future_price'] - df[price_col]) / df[price_col]

        def label_return(r):
            if pd.isna(r):
                return np.nan
            if r > threshold:
                return 'bullish'
            elif r < -threshold:
                return 'bearish'
            else:
                return 'neutral'

        df['target_clf'] = df['future_return'].apply(label_return)
        df['target_reg'] = df['future_price']
        df['coin'] = coin

        all_data.append(df)

    combined_df = pd.concat(all_data, ignore_index=True)
    combined_df.dropna(subset=['target_clf', 'target_reg'], inplace=True)
    combined_df.reset_index(drop=True, inplace=True)

    # Encode classification labels
    le = LabelEncoder()
    combined_df['target_clf_code'] = le.fit_transform(combined_df['target_clf'])

    return combined_df, le

def train_models(df):
    features = ['return_1', 'SMA_10', 'SMA_50', 'RSI_14', 'volume']
    X = df[features]
    y_clf = df['target_clf_code']
    y_reg = df['target_reg']

    X_train, X_test, y_clf_train, y_clf_test, y_reg_train, y_reg_test = train_test_split(
        X, y_clf, y_reg, test_size=0.2, random_state=42, stratify=y_clf)

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_clf_train)

    reg = RandomForestRegressor(n_estimators=100, random_state=42)
    reg.fit(X_train, y_reg_train)

    # Optional: print accuracy and regression score
    print(f"Classification Accuracy: {clf.score(X_test, y_clf_test):.3f}")
    print(f"Regression R2 Score: {reg.score(X_test, y_reg_test):.3f}")

    return clf, reg

def predict(clf, reg, le, latest_features):
    """
    latest_features: pd.DataFrame with one row of features (same columns as training)
    Returns:
    - predicted class label (string)
    - predicted class probabilities (dict)
    - predicted future price (float)
    """
    probs = clf.predict_proba(latest_features)[0]
    class_idx = np.argmax(probs)
    class_label = le.inverse_transform([class_idx])[0]
    class_probs = dict(zip(le.classes_, probs))
    price_pred = reg.predict(latest_features)[0]

    return class_label, class_probs, price_pred

# --- Telegram Alerting ---
def send_telegram_message(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {'chat_id': CHAT_ID, 'text': message}
    try:
        requests.post(url, data=payload)
    except Exception as e:
        print(f"Failed to send Telegram message: {e}")

def main_loop():
    print("Starting training and prediction loop...")

    combined_df, le = fetch_combined_data(COINS)
    if combined_df.empty:
        print("No combined data to train on. Exiting.")
        return

    clf, reg = train_models(combined_df)

    # Get the latest features for each coin and predict
    for coin in COINS:
        df = combined_df[combined_df['coin'] == coin].copy()
        if df.empty:
            print(f"No data for {coin} on prediction step.")
            continue

        latest = df.iloc[-1]
        features = ['return_1', 'SMA_10', 'SMA_50', 'RSI_14', 'volume']
        latest_features = pd.DataFrame([latest[features]])

        pred_class, pred_probs, pred_price = predict(clf, reg, le, latest_features)
        current_price = fetch_current_price(coin)

        # Calculate predicted % change
        price_change_pct = abs(pred_price - current_price) / current_price

        if price_change_pct >= 0.01:  # 1% threshold
            msg = (f"{coin} Prediction:\n"
                   f"Class: {pred_class} (Probabilities: {pred_probs})\n"
                   f"Predicted price in {LOOKAHEAD} periods: {pred_price:.4f}\n"
                   f"Current price: {current_price:.4f}\n"
                   f"Predicted change: {price_change_pct*100:.2f}%")
            print(msg)
            send_telegram_message(msg)
        else:
            print(f"{coin}: Predicted price change {price_change_pct*100:.2f}% below threshold; no alert sent.")

        send_telegram_message(msg)

if __name__ == "__main__":
    while True:
        try:
            main_loop()
        except Exception as e:
            print(f"Error in main loop: {e}")
        print("Sleeping for 10 minutes...")
        time.sleep(600)  # Run every 10 minutes
