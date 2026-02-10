
# train_baseline.py
# Time-aware baseline for 1-minute crypto modeling.
# - Fetches OHLCV (Coinbase) for a symbol over N days
# - Builds features/targets via feature_builder.py
# - Trains a simple model with time-based CV
# - Saves artifacts: model.pkl, metrics.json, feat_importance.csv
#
# Usage examples:
#   python train_baseline.py --symbol BTC-USD --days 60 --horizon 10 --thr-bp 3
#   python train_baseline.py --symbol ETH-USD --days 120 --horizon 30 --target regression
#
# Dependencies (recommended):
#   pandas numpy requests scikit-learn joblib (optional: lightgbm or xgboost)

import argparse
import json
import math
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import requests

from pathlib import Path

# Local module built earlier
from feature_builder import build_dataset

# ----------------------
# Data fetch (paged REST)
# ----------------------
DEFAULT_GRAN = 60               # 1-minute
BATCH_POINTS = 300              # Coinbase per-call cap
REQUEST_TIMEOUT_S = 15
PAUSE_BETWEEN_CALLS = 0.18
MAX_RETRIES = 5

def fetch_all_candles(product_id: str, granularity: int = DEFAULT_GRAN, days: int = 30) -> pd.DataFrame:
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days)

    all_rows = []
    t0 = start
    batch_seconds = BATCH_POINTS * granularity

    while t0 < end:
        t1 = min(t0 + timedelta(seconds=batch_seconds), end)
        url = (
            f"https://api.exchange.coinbase.com/products/{product_id}/candles"
            f"?granularity={granularity}&start={t0.isoformat()}&end={t1.isoformat()}"
        )

        attempt = 0
        while True:
            try:
                r = requests.get(url, timeout=REQUEST_TIMEOUT_S)
                if r.status_code == 429:
                    time.sleep(1.0 + attempt * 0.5)
                    attempt += 1
                    if attempt > MAX_RETRIES:
                        raise RuntimeError("Rate limit exceeded repeatedly.")
                    continue
                r.raise_for_status()
                data = r.json()
                break
            except Exception:
                attempt += 1
                if attempt > MAX_RETRIES:
                    raise
                time.sleep(0.6 * attempt)

        if data:
            df_chunk = pd.DataFrame(data, columns=["time","low","high","open","close","volume"])
            all_rows.append(df_chunk)

        t0 = t1
        time.sleep(PAUSE_BETWEEN_CALLS)

    if not all_rows:
        raise RuntimeError("No data returned from Coinbase")

    df = pd.concat(all_rows, ignore_index=True)
    df = df.drop_duplicates(subset=["time"]).sort_values("time").reset_index(drop=True)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    # reorder to common OHLCV
    df = df[["time","open","high","low","close","volume"]]
    return df

# ----------------------
# Modeling utilities
# ----------------------
def time_series_split_indices(n: int, n_splits: int = 5, min_train_frac: float = 0.6):
    """Yields (train_idx, valid_idx) for expanding window CV."""
    min_train = int(n * min_train_frac)
    fold_size = (n - min_train) // n_splits if n_splits > 0 else 0
    for k in range(n_splits):
        train_end = min_train + k * fold_size
        valid_end = train_end + fold_size if k < n_splits - 1 else n
        train_idx = np.arange(0, train_end)
        valid_idx = np.arange(train_end, valid_end)
        if len(valid_idx) == 0:
            break
        yield train_idx, valid_idx

def get_model(kind: str, n_estimators: int = 400, random_state: int = 1337):
    kind = kind.lower()
    if kind == "lightgbm":
        try:
            import lightgbm as lgb
            return lgb.LGBMClassifier(
                n_estimators=n_estimators,
                learning_rate=0.05,
                max_depth=-1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=random_state,
                objective="multiclass"
            )
        except Exception:
            print("[warn] lightgbm not available; falling back to RandomForest")
    if kind == "xgboost":
        try:
            from xgboost import XGBClassifier
            return XGBClassifier(
                n_estimators=n_estimators,
                learning_rate=0.05,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=random_state,
                objective="multi:softprob",
                eval_metric="mlogloss"
            )
        except Exception:
            print("[warn] xgboost not available; falling back to RandomForest")

    # Fallback
    from sklearn.ensemble import RandomForestClassifier
    return RandomForestClassifier(
        n_estimators=400, max_depth=None, min_samples_leaf=2, n_jobs=-1, random_state=random_state
    )

# ----------------------
# Metrics
# ----------------------
def score_classification(y_true, y_pred, y_prob=None):
    from sklearn.metrics import accuracy_score, f1_score, classification_report
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
    }
    try:
        out["report"] = classification_report(y_true, y_pred, digits=4)
    except Exception:
        pass
    return out

# ----------------------
# Main
# ----------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="BTC-USD")
    ap.add_argument("--days", type=int, default=60)
    ap.add_argument("--granularity", type=int, default=60)
    ap.add_argument("--horizon", type=int, default=10)             # minutes ahead to predict
    ap.add_argument("--thr-bp", type=float, default=3.0)           # ±bps threshold for classification
    ap.add_argument("--target", choices=["classification","regression"], default="classification")
    ap.add_argument("--model", choices=["lightgbm","xgboost","rf"], default="lightgbm")
    ap.add_argument("--splits", type=int, default=5)
    args = ap.parse_args()

    outdir = Path("artifacts_" + args.symbol.replace("-", "_"))
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"[fetch] {args.symbol} {args.granularity}s for {args.days} days…")
    df = fetch_all_candles(args.symbol, args.granularity, args.days)
    print(f"[fetch] got {len(df)} rows from {df['time'].iloc[0]} to {df['time'].iloc[-1]}")

    print("[features] building features + target…")
    X, y = build_dataset(
        df,
        use_optional_technicals=True,
        lag_max=5,
        roll_windows=(3,5,15,30),
        horizon=args.horizon,
        target_type=args.target,
        thr_bp=args.thr_bp,
    )

    # Align (drop any leftover NaNs just in case)
    idx = X.index.intersection(y.index)
    X = X.loc[idx].replace([np.inf, -np.inf], np.nan).dropna(how="any")
    y = y.loc[X.index]

    print(f"[dataset] final shapes: X={X.shape}, y={y.shape} (classes: {sorted(pd.unique(y)) if args.target=='classification' else 'regression'})")

    # Time-based CV
    metrics = []
    oof_pred = np.zeros_like(y, dtype=float) if args.target=="regression" else None
    oof_proba = None

    model_importances = None
    fold_id = 0

    for tr_idx, va_idx in time_series_split_indices(len(X), n_splits=args.splits, min_train_frac=0.6):
        fold_id += 1
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

        clf = get_model(args.model)

        if args.target == "classification":
            clf.fit(X_tr, y_tr)
            y_hat = clf.predict(X_va)
            try:
                y_prob = clf.predict_proba(X_va)
            except Exception:
                y_prob = None
            m = score_classification(y_va, y_hat, y_prob)
        else:
            # Regression baseline
            from sklearn.ensemble import RandomForestRegressor
            clf = RandomForestRegressor(n_estimators=400, n_jobs=-1, random_state=1337)
            clf.fit(X_tr, y_tr)
            y_hat = clf.predict(X_va)
            from sklearn.metrics import mean_squared_error, r2_score
            m = {
                "rmse": float(np.sqrt(mean_squared_error(y_va, y_hat))),
                "r2": float(r2_score(y_va, y_hat)),
            }

        m["fold"] = fold_id
        metrics.append(m)
        print(f"[cv] fold {fold_id}: {m}")

        # Collect importances if available
        imp = None
        if hasattr(clf, "feature_importances_"):
            imp = pd.DataFrame({"feature": X.columns, "importance": clf.feature_importances_})
            imp["fold"] = fold_id
            model_importances = imp if model_importances is None else pd.concat([model_importances, imp], axis=0)

    # Aggregate metrics
    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(outdir / "cv_metrics.csv", index=False)

    agg = metrics_df.drop(columns=["fold"]).mean(numeric_only=True).to_dict()
    with open(outdir / "metrics_mean.json", "w") as f:
        json.dump(agg, f, indent=2)

    if model_importances is not None:
        imp_agg = model_importances.groupby("feature")["importance"].mean().sort_values(ascending=False)
        imp_agg.to_csv(outdir / "feature_importance.csv", header=["importance"])

    # Final fit on all data (to export a model)
    print("[fit] training final model on full dataset…")
    final_model = get_model(args.model)
    if args.target == "classification":
        final_model.fit(X, y)
    else:
        from sklearn.ensemble import RandomForestRegressor
        final_model = RandomForestRegressor(n_estimators=400, n_jobs=-1, random_state=1337)
        final_model.fit(X, y)

    # Save model
    try:
        import joblib
        joblib.dump({"model": final_model, "columns": X.columns.tolist()}, outdir / "model.pkl")
        print(f"[save] model saved to {outdir / 'model.pkl'}")
    except Exception:
        print("[warn] joblib not available, skipping model save.")

    # Save a small sample of the dataset for inspection
    X.head(2000).to_parquet(outdir / "X_sample.parquet")
    y.head(2000).to_csv(outdir / "y_sample.csv", index=True)

    print("[done] artifacts in:", outdir.resolve())

if __name__ == "__main__":
    main()
