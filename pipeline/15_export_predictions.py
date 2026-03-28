"""
Phase 15 — Export Breakout Predictions for Frontend
=====================================================
Produces compact JSON files for the Emerging Words dashboard.

Outputs (in outputs/):
  predictions.json       — model metadata + per-year top-20 predictions
  prediction_eval.json   — copy of evaluation report
"""

import json
import os
import pickle
import time

import numpy as np
import polars as pl

# ── Paths ──────────────────────────────────────────────────────────
PROCESSED       = "data/processed"
FEATURES_PATH   = os.path.join(PROCESSED, "prediction_features.parquet")
MODEL_PATH      = os.path.join(PROCESSED, "breakout_model.pkl")
SCALER_PATH     = os.path.join(PROCESSED, "breakout_scaler.pkl")
META_PATH       = os.path.join(PROCESSED, "breakout_model_meta.json")
EVAL_PATH       = os.path.join(PROCESSED, "prediction_eval.json")
CLEAN_PANEL     = os.path.join(PROCESSED, "clean_panel.parquet")

OUT_DIR         = "outputs"
OUT_PREDICTIONS = os.path.join(OUT_DIR, "predictions.json")
OUT_EVAL        = os.path.join(OUT_DIR, "prediction_eval.json")

FEATURE_COLS = [
    "latent_drift", "curvature", "local_instability", "latent_level",
    "regime_adoption", "regime_decline", "regime_turbulent",
    "recent_changepoints", "years_since_first", "recent_log_change",
    "factor_1", "factor_2", "factor_3", "factor_4", "factor_5",
    "factor_6", "factor_7", "factor_8", "factor_9", "factor_10",
]

# Years to generate predictions for (the demo slider)
DEMO_YEARS = list(range(1900, 2004))
TOP_K = 20


def main():
    t0 = time.time()
    print("=" * 60)
    print("Phase 15 — Export Breakout Predictions")
    print("=" * 60)

    # ── Load model + data ──────────────────────────────────────
    print("[1/3] Loading model and data …")
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
    meta = json.load(open(META_PATH))
    needs_scaler = meta["needs_scaler"]

    features = pl.read_parquet(FEATURES_PATH)
    panel = pl.read_parquet(CLEAN_PANEL, columns=["word", "year", "frequency"])

    # ── Score all word-years ───────────────────────────────────
    print("[2/3] Scoring all word-years …")
    X = features.select(FEATURE_COLS).to_numpy().astype(np.float64)
    X[~np.isfinite(X)] = 0.0
    if needs_scaler:
        X = scaler.transform(X)
    probs = model.predict_proba(X)[:, 1]

    features = features.with_columns(
        pl.Series("breakout_prob", probs)
    )

    # ── Build per-year top-k predictions ───────────────────────
    print(f"[3/3] Building top-{TOP_K} predictions for {len(DEMO_YEARS)} years …")
    yearly_predictions = {}

    for year in DEMO_YEARS:
        yr_data = features.filter(pl.col("year") == year)
        if yr_data.shape[0] == 0:
            continue

        top = yr_data.sort("breakout_prob", descending=True).head(TOP_K)
        words_list = []
        for row in top.iter_rows(named=True):
            w = row["word"]
            p = row["breakout_prob"]
            actual = row["breakout_label"]

            # Get actual growth if we have future data
            freq_now = panel.filter(
                (pl.col("word") == w) & (pl.col("year") == year)
            )
            freq_fut = panel.filter(
                (pl.col("word") == w) & (pl.col("year") == year + 5)
            )
            fn = freq_now["frequency"].to_list()[0] if freq_now.shape[0] > 0 else 0
            ff = freq_fut["frequency"].to_list()[0] if freq_fut.shape[0] > 0 else 0
            ratio = round(ff / fn, 2) if fn > 0 else None

            # Determine regime
            regime = "stable"
            if row.get("regime_turbulent", 0) == 1:
                regime = "turbulent"
            elif row.get("regime_adoption", 0) == 1:
                regime = "adoption"
            elif row.get("regime_decline", 0) == 1:
                regime = "decline"

            words_list.append({
                "word": w,
                "prob": round(p, 4),
                "actual": int(actual),
                "growth": ratio,
                "regime": regime,
            })

        yearly_predictions[str(year)] = words_list

    # ── Assemble output ────────────────────────────────────────
    output = {
        "model": meta["best_model"],
        "horizon": meta["horizon"],
        "test_auc": meta["best_test_metrics"]["auc"],
        "test_p50": meta["best_test_metrics"].get("p@50"),
        "feature_importances": dict(list(meta["feature_importances"].items())[:10]),
        "years": sorted(yearly_predictions.keys()),
        "predictions": yearly_predictions,
    }

    with open(OUT_PREDICTIONS, "w") as f:
        json.dump(output, f)
    print(f"  → {OUT_PREDICTIONS}  ({os.path.getsize(OUT_PREDICTIONS)/1e3:.1f} KB)")

    # Copy eval report
    eval_data = json.load(open(EVAL_PATH))
    with open(OUT_EVAL, "w") as f:
        json.dump(eval_data, f, indent=2)
    print(f"  → {OUT_EVAL}")

    print(f"\nPhase 15 complete.  Total time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
