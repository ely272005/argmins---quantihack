"""
Phase 17 — Cultural Shift Detection
======================================
Two complementary approaches:

  1. Unsupervised ShiftScore = z(LII) + z(CP_density)
     - No labels needed, pure linguistic signal
     - Evaluated by: "how many of 12 known events does it detect?"

  2. Supervised Logistic Regression (secondary)
     - Train: 1800–1950, Test: 1951–2008
     - Strong regularization (only 209 datapoints)

Shift types (rule-based):
  - structural_upheaval:      high LII + high CP  (wars, revolutions)
  - smooth_transformation:    high LII + low CP   (technology, culture)
  - localized_restructuring:  low LII + high CP   (political, sectoral)
  - stable:                   low LII + low CP

Outputs:
  outputs/shift_detection.json  (frontend-ready)
"""

import json
import os
import pickle
import time

import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler

# ── Paths ──────────────────────────────────────────────────────────
PROCESSED = "data/processed"
DATASET_PATH = os.path.join(PROCESSED, "cultural_shift_dataset.parquet")
MODEL_PATH = os.path.join(PROCESSED, "shift_model.pkl")
SCALER_PATH = os.path.join(PROCESSED, "shift_scaler.pkl")
META_PATH = os.path.join(PROCESSED, "shift_model_meta.json")
EVENTS_PATH = "outputs/events.json"
OUT_DIR = "outputs"
OUT_DETECTION = os.path.join(OUT_DIR, "shift_detection.json")

TRAIN_END = 1950

FEATURE_COLS = [
    "lii", "lii_delta", "lii_rolling_mean", "lii_z",
    "cp_density", "cp_delta", "cp_z",
    "cr", "bvn",
    "factor_volatility", "top_eigenvalue",
]


def classify_shift_type(lii_z, cp_z):
    high_lii = lii_z > 1.0
    high_cp = cp_z > 1.0
    if high_lii and high_cp:
        return "structural_upheaval"
    elif high_lii and not high_cp:
        return "smooth_transformation"
    elif not high_lii and high_cp:
        return "localized_restructuring"
    else:
        return "stable"


def main():
    t0 = time.time()
    print("=" * 60)
    print("Phase 17 — Cultural Shift Detection")
    print("=" * 60)

    # ── Load ───────────────────────────────────────────────────
    print("[1/5] Loading …")
    df = pl.read_parquet(DATASET_PATH)
    events = json.load(open(EVENTS_PATH))
    years = [int(y) for y in df["year"].to_list()]
    print(f"  {len(years)} years, {len(FEATURE_COLS)} features, {len(events)} events")

    lii_z_all = df["lii_z"].to_numpy().astype(np.float64)
    cp_z_all = df["cp_z"].to_numpy().astype(np.float64)
    shift_score_all = df["shift_score"].to_numpy().astype(np.float64)
    labels_all = df["label"].to_numpy()

    # ── Unsupervised evaluation ────────────────────────────────
    print("\n[2/5] Unsupervised ShiftScore evaluation …")

    # Build year→score lookup
    year_to_idx = {y: i for i, y in enumerate(years)}

    # For each event, classify detection strength:
    #   strong: peak > 1.5    weak: peak > 0.5    missed: peak ≤ 0.5
    STRONG_THRESHOLD = 1.5
    WEAK_THRESHOLD = 0.5
    detections = []
    for ev in events:
        ev_years = range(max(ev["start_year"] - 2, min(years)),
                         min(ev["end_year"] + 3, max(years) + 1))
        scores_during = []
        for y in ev_years:
            if y in year_to_idx:
                scores_during.append(shift_score_all[year_to_idx[y]])

        if not scores_during:
            detections.append({"event": ev["name"], "strength": "missed",
                               "peak_score": 0, "peak_year": None,
                               "category": ev.get("category", "")})
            continue

        peak_score = max(scores_during)
        peak_idx = scores_during.index(peak_score)
        peak_year = list(ev_years)[peak_idx]

        if peak_score > STRONG_THRESHOLD:
            strength = "strong"
        elif peak_score > WEAK_THRESHOLD:
            strength = "weak"
        else:
            strength = "missed"

        detections.append({
            "event": ev["name"],
            "strength": strength,
            "peak_score": round(float(peak_score), 3),
            "peak_year": int(peak_year),
            "category": ev.get("category", ""),
        })
        print(f"  {strength:8s}  {ev['name']:30s}  peak={peak_score:.2f} at {peak_year}")

    n_strong = sum(1 for d in detections if d["strength"] == "strong")
    n_weak = sum(1 for d in detections if d["strength"] == "weak")
    n_missed = sum(1 for d in detections if d["strength"] == "missed")
    print(f"\n  Strong: {n_strong}  Weak: {n_weak}  Missed: {n_missed}  (out of {len(events)})")

    # ── Supervised model ───────────────────────────────────────
    print(f"\n[3/5] Supervised model (train ≤{TRAIN_END}) …")
    train = df.filter(pl.col("year") <= TRAIN_END)
    test = df.filter(pl.col("year") > TRAIN_END)

    X_train = train.select(FEATURE_COLS).to_numpy().astype(np.float64)
    y_train = train["label"].to_numpy()
    X_test = test.select(FEATURE_COLS).to_numpy().astype(np.float64)
    y_test = test["label"].to_numpy()

    for X in [X_train, X_test]:
        X[~np.isfinite(X)] = 0.0

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    lr = LogisticRegression(max_iter=2000, C=0.1, class_weight="balanced",
                            solver="lbfgs")
    lr.fit(X_train_s, y_train)

    importances = dict(zip(FEATURE_COLS, np.abs(lr.coef_[0]).tolist()))
    importances = dict(sorted(importances.items(), key=lambda x: -x[1]))

    # Score all years
    X_all = df.select(FEATURE_COLS).to_numpy().astype(np.float64)
    X_all[~np.isfinite(X_all)] = 0.0
    X_all_s = scaler.transform(X_all)
    supervised_probs = lr.predict_proba(X_all_s)[:, 1]

    # Test metrics (for reference, not the main story)
    try:
        test_auc = roc_auc_score(y_test, lr.predict_proba(X_test_s)[:, 1])
    except ValueError:
        test_auc = 0.5
    print(f"  Test AUC: {test_auc:.4f}  (note: only {len(y_test)} test points)")

    # ── Classify shift types ───────────────────────────────────
    print(f"\n[4/5] Classifying shift types …")
    shift_types = [classify_shift_type(lii_z_all[i], cp_z_all[i])
                   for i in range(len(years))]
    type_counts = {}
    for t in shift_types:
        type_counts[t] = type_counts.get(t, 0) + 1
    for k, v in sorted(type_counts.items(), key=lambda x: -x[1]):
        print(f"  {k:30s} {v:3d} years")

    # ── Build frontend export ──────────────────────────────────
    print(f"\n[5/5] Exporting …")

    # Normalize shift score to 0–100 for display
    ss_min, ss_max = shift_score_all.min(), shift_score_all.max()
    ss_range = ss_max - ss_min if ss_max > ss_min else 1.0

    timeline = []
    for i, year in enumerate(years):
        timeline.append({
            "year": year,
            "shift_score": round(float(shift_score_all[i]), 4),
            "shift_pct": round(100 * float((shift_score_all[i] - ss_min) / ss_range), 1),
            "supervised_prob": round(float(supervised_probs[i]), 4),
            "lii_z": round(float(lii_z_all[i]), 4),
            "cp_z": round(float(cp_z_all[i]), 4),
            "lii": round(float(df["lii"][i]), 6),
            "cp_density": round(float(df["cp_density"][i]), 6),
            "shift_type": shift_types[i],
            "label": int(labels_all[i]),
        })

    peaks = [t for t in timeline if t["shift_score"] > 1.5]

    event_overlay = [{"name": ev["name"], "start": ev["start_year"],
                      "end": ev["end_year"], "category": ev["category"]}
                     for ev in events]

    output = {
        "n_strong": n_strong,
        "n_weak": n_weak,
        "n_missed": n_missed,
        "n_events": len(events),
        "detections": detections,
        "strong_threshold": STRONG_THRESHOLD,
        "weak_threshold": WEAK_THRESHOLD,
        "supervised_test_auc": round(test_auc, 4),
        "feature_importances": {k: round(v, 6) for k, v in list(importances.items())[:12]},
        "timeline": timeline,
        "detected_peaks": peaks,
        "events": event_overlay,
        "shift_type_legend": {
            "structural_upheaval": "High LII + High CP — wars, revolutions",
            "smooth_transformation": "High LII + Low CP — technology, gradual cultural change",
            "localized_restructuring": "Low LII + High CP — political shifts, sector-specific",
            "stable": "Low LII + Low CP — no significant cultural shift",
        },
        "shift_type_counts": type_counts,
    }

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(lr, f)
    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)

    meta = {
        "n_strong": n_strong, "n_weak": n_weak, "n_missed": n_missed,
        "detections": detections,
        "supervised_test_auc": round(test_auc, 4),
        "feature_importances": {k: round(v, 6) for k, v in importances.items()},
    }
    with open(META_PATH, "w") as f:
        json.dump(meta, f, indent=2)

    with open(OUT_DETECTION, "w") as f:
        json.dump(output, f)
    sz = os.path.getsize(OUT_DETECTION) / 1e3
    print(f"  → {OUT_DETECTION}  ({sz:.1f} KB)")
    print(f"\nPhase 17 complete.  Total time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
