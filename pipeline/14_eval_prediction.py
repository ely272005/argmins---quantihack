"""
Phase 14 — Evaluate Breakout Predictions
==========================================
Inspect real examples, compare against baselines, produce a human-readable
evaluation report and machine-readable JSON.

Outputs:
  data/processed/prediction_eval.json
"""

import json
import os
import time

import numpy as np
import polars as pl

# ── Paths ──────────────────────────────────────────────────────────
PROCESSED       = "data/processed"
PRED_PATH       = os.path.join(PROCESSED, "breakout_predictions.parquet")
FEATURES_PATH   = os.path.join(PROCESSED, "prediction_features.parquet")
META_PATH       = os.path.join(PROCESSED, "breakout_model_meta.json")
CLEAN_PANEL     = os.path.join(PROCESSED, "clean_panel.parquet")
EVAL_OUT        = os.path.join(PROCESSED, "prediction_eval.json")

# ── Known breakout words to inspect ────────────────────────────────
# Words that we KNOW surged historically (from c/t/w shards only)
INSPECT_WORDS = [
    "computer", "technology", "wireless", "television",
    "communism", "capitalism", "cybersex", "transgenic",
    "telecare", "cdrom",
]


def precision_at_k(y_true, y_prob, k):
    idx = np.argsort(y_prob)[::-1][:k]
    return float(y_true[idx].mean())


def main():
    t0 = time.time()
    print("=" * 60)
    print("Phase 14 — Evaluate Breakout Predictions")
    print("=" * 60)

    preds = pl.read_parquet(PRED_PATH)
    meta  = json.load(open(META_PATH))
    panel = pl.read_parquet(CLEAN_PANEL, columns=["word", "year", "frequency"])

    y_true = preds["breakout_label"].to_numpy()
    y_prob = preds["breakout_prob"].to_numpy()

    # ── 1. Overall test metrics ─────────────────────────────────
    print("\n[1/4] Test-set metrics (from Phase 13):")
    tm = meta["best_test_metrics"]
    print(f"  Model: {meta['best_model']}")
    print(f"  AUC:   {tm['auc']}")
    print(f"  P@50:  {tm.get('p@50', 'N/A')}  P@100: {tm.get('p@100', 'N/A')}  P@500: {tm.get('p@500', 'N/A')}")

    # ── 2. Per-year precision@k ─────────────────────────────────
    print("\n[2/4] Per-year precision@50 on test set:")
    year_metrics = {}
    for year in sorted(preds["year"].unique().to_list()):
        mask = preds.filter(pl.col("year") == year)
        yt = mask["breakout_label"].to_numpy()
        yp = mask["breakout_prob"].to_numpy()
        if len(yt) < 50:
            continue
        p50 = precision_at_k(yt, yp, 50)
        year_metrics[str(year)] = {"p@50": round(p50, 4), "n_words": len(yt)}
        print(f"  {year}: P@50={p50:.3f}  ({len(yt):,} words)")

    # ── 3. Inspect known words ──────────────────────────────────
    print("\n[3/4] Inspecting known words in test period:")
    word_inspections = {}
    for word in INSPECT_WORDS:
        rows = preds.filter(pl.col("word") == word).sort("year")
        if rows.shape[0] == 0:
            print(f"  {word:20s}  (not in test set)")
            continue

        # Get the latest test-year prediction
        last = rows.row(-1, named=True)
        yr = last["year"]
        prob = last["breakout_prob"]
        actual = last["breakout_label"]

        # Get actual frequency trajectory
        freq_data = panel.filter(pl.col("word") == word).sort("year")
        freq_at_yr = freq_data.filter(pl.col("year") == yr)
        freq_at_future = freq_data.filter(pl.col("year") == yr + 5)

        freq_now = freq_at_yr["frequency"].to_list()[0] if freq_at_yr.shape[0] > 0 else None
        freq_fut = freq_at_future["frequency"].to_list()[0] if freq_at_future.shape[0] > 0 else None
        ratio = freq_fut / freq_now if freq_now and freq_fut and freq_now > 0 else None

        status = "TP" if actual == 1 and prob > 0.5 else \
                 "FN" if actual == 1 and prob <= 0.5 else \
                 "FP" if actual == 0 and prob > 0.5 else "TN"

        print(f"  {word:20s}  year={yr}  P(breakout)={prob:.3f}  actual={actual}  "
              f"growth={ratio:.2f}x  → {status}" if ratio else
              f"  {word:20s}  year={yr}  P(breakout)={prob:.3f}  actual={actual}  → {status}")

        word_inspections[word] = {
            "year": yr,
            "breakout_prob": round(prob, 4),
            "actual_label": int(actual),
            "growth_ratio": round(ratio, 4) if ratio else None,
            "classification": status,
        }

    # ── 4. Top predicted breakouts at specific years ────────────
    print("\n[4/4] Top predicted breakouts by year:")
    demo_years = {}
    for demo_year in [1996, 1998, 2000, 2003]:
        yr_data = preds.filter(pl.col("year") == demo_year)
        if yr_data.shape[0] == 0:
            continue
        top = yr_data.sort("breakout_prob", descending=True).head(20)

        words_list = []
        for row in top.iter_rows(named=True):
            w = row["word"]
            p = row["breakout_prob"]
            a = row["breakout_label"]

            # Get actual growth
            freq_now = panel.filter(
                (pl.col("word") == w) & (pl.col("year") == demo_year)
            )
            freq_fut = panel.filter(
                (pl.col("word") == w) & (pl.col("year") == demo_year + 5)
            )
            fn = freq_now["frequency"].to_list()[0] if freq_now.shape[0] > 0 else 0
            ff = freq_fut["frequency"].to_list()[0] if freq_fut.shape[0] > 0 else 0
            ratio = ff / fn if fn > 0 else None

            words_list.append({
                "word": w,
                "breakout_prob": round(p, 4),
                "actual_label": int(a),
                "growth_ratio": round(ratio, 2) if ratio else None,
            })

        demo_years[str(demo_year)] = words_list
        print(f"\n  Year {demo_year} — top 10 predicted breakouts:")
        for i, w in enumerate(words_list[:10]):
            tag = "✓" if w["actual_label"] == 1 else "✗"
            gr = f"{w['growth_ratio']:.1f}x" if w["growth_ratio"] else "?"
            print(f"    {i+1:2d}. {w['word']:25s}  P={w['breakout_prob']:.3f}  "
                  f"grew={gr}  {tag}")

    # ── Save evaluation ────────────────────────────────────────
    eval_report = {
        "model": meta["best_model"],
        "horizon": meta["horizon"],
        "test_metrics": meta["best_test_metrics"],
        "baselines": meta["baselines"],
        "feature_importances": meta["feature_importances"],
        "per_year_metrics": year_metrics,
        "word_inspections": word_inspections,
        "demo_predictions": demo_years,
    }
    with open(EVAL_OUT, "w") as f:
        json.dump(eval_report, f, indent=2)
    print(f"\n→ {EVAL_OUT}")
    print(f"\nPhase 14 complete.  Total time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
