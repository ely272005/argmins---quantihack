"""
Phase 13 — Train Breakout Model
=================================
Temporal split:
  train:      1805–1980
  validation: 1981–1995
  test:       1996–2003

Models:
  1. Logistic Regression (baseline)
  2. Random Forest

Outputs:
  data/processed/breakout_model.pkl          (best model)
  data/processed/breakout_predictions.parquet (test-set predictions)
  data/processed/breakout_model_meta.json    (metrics + feature importances)
"""

import json
import os
import pickle
import time

import numpy as np
import polars as pl
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler

# ── Paths ──────────────────────────────────────────────────────────
PROCESSED       = "data/processed"
FEATURES_PATH   = os.path.join(PROCESSED, "prediction_features.parquet")
MODEL_PATH      = os.path.join(PROCESSED, "breakout_model.pkl")
SCALER_PATH     = os.path.join(PROCESSED, "breakout_scaler.pkl")
PRED_PATH       = os.path.join(PROCESSED, "breakout_predictions.parquet")
META_PATH       = os.path.join(PROCESSED, "breakout_model_meta.json")

# ── Temporal split boundaries ──────────────────────────────────────
TRAIN_END  = 1980
VAL_END    = 1995
# Test: 1996–2003 (last usable year with h=5 → 2003)

# ── Feature columns ───────────────────────────────────────────────
FEATURE_COLS = [
    "latent_drift", "curvature", "local_instability", "latent_level",
    "regime_adoption", "regime_decline", "regime_turbulent",
    "recent_changepoints", "years_since_first", "recent_log_change",
    "factor_1", "factor_2", "factor_3", "factor_4", "factor_5",
    "factor_6", "factor_7", "factor_8", "factor_9", "factor_10",
]
TARGET_COL = "breakout_label"


def precision_at_k(y_true, y_prob, k):
    """Among the top-k predicted, how many actually broke out?"""
    top_k_idx = np.argsort(y_prob)[::-1][:k]
    return y_true[top_k_idx].mean()


def evaluate(name, y_true, y_prob, ks=(50, 100, 200, 500)):
    """Print and return evaluation metrics."""
    auc = roc_auc_score(y_true, y_prob)
    y_pred = (y_prob >= 0.5).astype(int)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)

    print(f"  {name:25s}  AUC={auc:.4f}  P={prec:.3f}  R={rec:.3f}", end="")
    pak = {}
    for k in ks:
        if k <= len(y_true):
            p = precision_at_k(y_true, y_prob, k)
            pak[f"p@{k}"] = round(p, 4)
            print(f"  P@{k}={p:.3f}", end="")
    print()
    return {"auc": round(auc, 4), "precision": round(prec, 4),
            "recall": round(rec, 4), **pak}


def main():
    t0 = time.time()
    print("=" * 60)
    print("Phase 13 — Train Breakout Model")
    print("=" * 60)

    # ── Load data ──────────────────────────────────────────────
    print("[1/5] Loading features …")
    df = pl.read_parquet(FEATURES_PATH)
    print(f"  {df.shape[0]:,} rows, {len(FEATURE_COLS)} features")

    # ── Temporal split ─────────────────────────────────────────
    print(f"\n[2/5] Temporal split: train ≤{TRAIN_END}, val ≤{VAL_END}, test >{VAL_END}")
    train = df.filter(pl.col("year") <= TRAIN_END)
    val   = df.filter((pl.col("year") > TRAIN_END) & (pl.col("year") <= VAL_END))
    test  = df.filter(pl.col("year") > VAL_END)

    for name, split in [("train", train), ("val", val), ("test", test)]:
        n = split.shape[0]
        pos = split.filter(pl.col(TARGET_COL) == 1).shape[0]
        print(f"  {name:5s}: {n:>10,} rows, {pos:>8,} breakouts ({100*pos/n:.1f}%)")

    X_train = train.select(FEATURE_COLS).to_numpy().astype(np.float64)
    y_train = train[TARGET_COL].to_numpy()
    X_val   = val.select(FEATURE_COLS).to_numpy().astype(np.float64)
    y_val   = val[TARGET_COL].to_numpy()
    X_test  = test.select(FEATURE_COLS).to_numpy().astype(np.float64)
    y_test  = test[TARGET_COL].to_numpy()

    # Replace remaining NaN/inf with 0
    for X in [X_train, X_val, X_test]:
        X[~np.isfinite(X)] = 0.0

    # ── Scale ──────────────────────────────────────────────────
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s   = scaler.transform(X_val)
    X_test_s  = scaler.transform(X_test)

    # ── Naive baselines ────────────────────────────────────────
    print(f"\n[3/5] Naive baselines (on validation set) …")
    # Baseline 1: predict using current frequency (latent_level)
    level_idx = FEATURE_COLS.index("latent_level")
    baseline_level = evaluate("Baseline: latent_level", y_val, X_val[:, level_idx])

    # Baseline 2: predict using recent growth
    growth_idx = FEATURE_COLS.index("recent_log_change")
    baseline_growth = evaluate("Baseline: recent_growth", y_val, X_val[:, growth_idx])

    # Baseline 3: predict using drift
    drift_idx = FEATURE_COLS.index("latent_drift")
    baseline_drift = evaluate("Baseline: latent_drift", y_val, X_val[:, drift_idx])

    # ── Train models ───────────────────────────────────────────
    print(f"\n[4/5] Training models …")

    # Logistic Regression
    print("  Training Logistic Regression …")
    lr = LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced",
                            solver="lbfgs", n_jobs=-1)
    lr.fit(X_train_s, y_train)
    lr_prob_val  = lr.predict_proba(X_val_s)[:, 1]
    lr_prob_test = lr.predict_proba(X_test_s)[:, 1]
    lr_val_metrics  = evaluate("LogReg (val)", y_val, lr_prob_val)
    lr_test_metrics = evaluate("LogReg (test)", y_test, lr_prob_test)

    # Random Forest
    print("  Training Random Forest …")
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=12, min_samples_leaf=50,
        class_weight="balanced", n_jobs=-1, random_state=42
    )
    rf.fit(X_train, y_train)
    rf_prob_val  = rf.predict_proba(X_val)[:, 1]
    rf_prob_test = rf.predict_proba(X_test)[:, 1]
    rf_val_metrics  = evaluate("RandomForest (val)", y_val, rf_prob_val)
    rf_test_metrics = evaluate("RandomForest (test)", y_test, rf_prob_test)

    # ── Pick best model by validation AUC ──────────────────────
    print(f"\n[5/5] Selecting best model …")
    if rf_val_metrics["auc"] > lr_val_metrics["auc"]:
        best_name = "RandomForest"
        best_model = rf
        best_prob_test = rf_prob_test
        best_test_metrics = rf_test_metrics
        best_needs_scaler = False
    else:
        best_name = "LogisticRegression"
        best_model = lr
        best_prob_test = lr_prob_test
        best_test_metrics = lr_test_metrics
        best_needs_scaler = True

    print(f"  Best: {best_name}  (val AUC={max(lr_val_metrics['auc'], rf_val_metrics['auc']):.4f})")

    # ── Feature importances ────────────────────────────────────
    if best_name == "RandomForest":
        importances = dict(zip(FEATURE_COLS, rf.feature_importances_.tolist()))
    else:
        importances = dict(zip(FEATURE_COLS,
                               np.abs(lr.coef_[0]).tolist()))
    # Sort descending
    importances = dict(sorted(importances.items(), key=lambda x: -x[1]))

    print("  Top features:")
    for feat, imp in list(importances.items())[:8]:
        print(f"    {feat:30s} {imp:.4f}")

    # ── Save model + scaler ────────────────────────────────────
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(best_model, f)
    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)
    print(f"  → {MODEL_PATH}")

    # ── Save test predictions ──────────────────────────────────
    test_preds = test.select("word", "year", TARGET_COL).with_columns(
        pl.Series("breakout_prob", best_prob_test)
    )
    test_preds.write_parquet(PRED_PATH)
    print(f"  → {PRED_PATH}  ({os.path.getsize(PRED_PATH)/1e6:.1f} MB)")

    # ── Save metadata ──────────────────────────────────────────
    meta = {
        "best_model": best_name,
        "needs_scaler": best_needs_scaler,
        "horizon": 5,
        "train_years": f"1805–{TRAIN_END}",
        "val_years": f"{TRAIN_END+1}–{VAL_END}",
        "test_years": f"{VAL_END+1}–2003",
        "feature_columns": FEATURE_COLS,
        "baselines": {
            "latent_level": baseline_level,
            "recent_growth": baseline_growth,
            "latent_drift": baseline_drift,
        },
        "logistic_regression": {
            "val": lr_val_metrics,
            "test": lr_test_metrics,
        },
        "random_forest": {
            "val": rf_val_metrics,
            "test": rf_test_metrics,
        },
        "best_test_metrics": best_test_metrics,
        "feature_importances": {k: round(v, 6) for k, v in importances.items()},
    }
    with open(META_PATH, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  → {META_PATH}")

    print(f"\nPhase 13 complete.  Total time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
