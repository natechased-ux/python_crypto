import pandas as pd
import numpy as np
import joblib
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# Config
DATA_FILE = "combined_historical_dataset.csv"
MODEL_FILE = "hourly_entry_model.pkl"
TEST_SIZE = 0.2
RANDOM_STATE = 42
def load_and_prepare_data():
    df = pd.read_csv(DATA_FILE)

    # Features we want to keep
    features = [
        "rsi", "bb_high", "bb_low", "atr", "macd", "macd_signal",
        "adx", "volatility_pct", "volume"
    ]

    # Add coin as categorical feature
    if "coin" in df.columns:
        df["coin"] = df["coin"].astype("category").cat.codes
        features.append("coin")

    X = df[features]
    y = df["target"]

    return X, y, features
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score
)
import numpy as np

def train_and_save_model():
    X, y, features = load_and_prepare_data()

    # Train-test split
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    # LightGBM Classifier
    from lightgbm import LGBMClassifier
    model = LGBMClassifier(
        n_estimators=500,
        learning_rate=0.02,
        max_depth=8,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE
    )

    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # === Metrics ===
    print("\n=== CLASSIFICATION REPORT ===")
    print(classification_report(y_test, y_pred))
    print(f"ROC AUC: {roc_auc_score(y_test, y_proba):.3f}")
    print(f"Precision: {precision_score(y_test, y_pred):.3f}")
    print(f"Recall: {recall_score(y_test, y_pred):.3f}")
    print(f"F1 Score: {f1_score(y_test, y_pred):.3f}")

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    print("\n=== CONFUSION MATRIX ===")
    print(cm)
    print(f"True Negatives: {tn}, False Positives: {fp}, False Negatives: {fn}, True Positives: {tp}")

    # === Quick Profit Factor Simulation ===
    # Assume: +1 reward for each correct "good trade" (TP), -1 penalty for each FP
    # Skip cost for TN and FN since we only care about taken trades here
    simulated_profit = tp - fp
    num_trades_taken = tp + fp
    win_rate = tp / num_trades_taken if num_trades_taken > 0 else 0
    profit_factor = (tp / fp) if fp > 0 else float('inf')

    print("\n=== QUICK PF SIMULATION ===")
    print(f"Trades Taken: {num_trades_taken}")
    print(f"Win Rate: {win_rate:.2%}")
    print(f"Profit Factor: {profit_factor:.2f}")
    print(f"Simulated Net: {simulated_profit}")

    # Save model
    import joblib
    joblib.dump((model, features), MODEL_FILE)
    print(f"\n✅ Model saved as {MODEL_FILE}")

if __name__ == "__main__":
    train_and_save_model()
