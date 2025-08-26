# train_baseline.py
import os
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline

from src.load_data import load_timelines
from src.splits import louo_splits
from src.make_windows import make_windows

K = 3  # number of past posts to use as context


def safe_spearman(y_true, y_pred):
    """
    Spearman correlation, but return NaN when undefined
    (fewer than 2 points or either sequence is constant).
    """
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    if len(y_true) < 2:
        return np.nan
    if np.all(y_true == y_true[0]) or np.all(y_pred == y_pred[0]):
        return np.nan
    return spearmanr(y_true, y_pred).statistic


def evaluate_fold(y_true, y_pred):
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    rho = safe_spearman(y_true, y_pred)
    return mae, rmse, rho


def main():
    os.makedirs("reports", exist_ok=True)

    # load data
    df = load_timelines("data/raw")  # change to "data" if your JSONs are directly in data/
    print(f"Loaded {len(df)} posts across {df.timeline_id.nunique()} timelines.")
    if df.empty:
        print("No posts found. Check that your JSONs are in data/raw and have a numeric well-being field.")
        return

    results = []

    # LOUO
    for tl, tr_df, te_df in louo_splits(df):
        # build training windows across all non-held-out users
        X_tr, y_tr = [], []
        for _, g in tr_df.groupby("timeline_id"):
            X, y = make_windows(g, k=K, h=1)
            X_tr += X
            y_tr += y

        # build test windows on held-out user
        X_te, y_te = make_windows(te_df, k=K, h=1)
        if len(y_te) == 0:
            continue

        # persistence baseline: predict last seen wb as next
        pers_pred = []
        rows = te_df.sort_values("post_idx").to_dict("records")
        for t in range(K - 1, len(rows) - 1):
            pers_pred.append(rows[t]["wb"])
        if len(pers_pred) == 0:
            continue
        p_mae, p_rmse, p_rho = evaluate_fold(y_te, pers_pred)

        # TF-IDF + Ridge
        pipe = Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=50000)),
            ("reg", Ridge(alpha=1.0, random_state=42)),
        ])
        if len(X_tr) == 0:
            # no training samples formed (e.g., very short timelines everywhere)
            continue
        pipe.fit(X_tr, y_tr)
        y_hat = pipe.predict(X_te).clip(1, 10)

        b_mae, b_rmse, b_rho = evaluate_fold(y_te, y_hat)

        results.append({
            "heldout_user": tl,
            "N_test": len(y_te),
            "pers_mae": p_mae, "pers_rmse": p_rmse, "pers_rho": p_rho,
            "ridge_mae": b_mae, "ridge_rmse": b_rmse, "ridge_rho": b_rho,
        })

    if not results:
        print("No evaluation windows were created. Try reducing K (e.g., K=2) or check your labels.")
        return

    out = pd.DataFrame(results).sort_values("heldout_user")
    print(out)

    # Macro (simple mean over users)
    print("\nMACRO (mean over users):")
    print(out.drop(columns=["heldout_user", "N_test"]).mean(numeric_only=True))

    # Micro (weighted by N_test)
    w = out["N_test"].values
    micro = {
        "pers_mae_micro":  float(np.average(out["pers_mae"],  weights=w)),
        "ridge_mae_micro": float(np.average(out["ridge_mae"], weights=w)),
        # combine RMSE via MSE weighting
        "pers_rmse_micro":  float(np.sqrt(np.average(out["pers_rmse"] ** 2,  weights=w))),
        "ridge_rmse_micro": float(np.sqrt(np.average(out["ridge_rmse"] ** 2, weights=w))),
    }
    print("\nMICRO (weighted by N_test):")
    print(micro)

    out.to_csv("reports/baseline_louo.csv", index=False)
    print("\nSaved: reports/baseline_louo.csv")


if __name__ == "__main__":
    main()
