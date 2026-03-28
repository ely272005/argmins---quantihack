"""
Phase 16 — Build Cultural Shift Dataset
=========================================
Constructs a year-level feature matrix for detecting cultural shifts.

Labels: Y_t = 1 if year t falls within ±2 years of a major historical event.

Features (from existing pipeline outputs):
  - LII, ΔLII, rolling mean, z-score
  - Changepoint density, Δdensity, z-score
  - Concentration ratio, broad-vs-narrow
  - Factor scores (f1–f10)
  - Factor volatility (rolling std of factor norms)
  - ShiftScore (unsupervised: z(LII) + z(density))

Output:
  data/processed/cultural_shift_dataset.parquet
"""

import json
import os
import time

import numpy as np
import polars as pl

# ── Paths ──────────────────────────────────────────────────────────
PROCESSED = "data/processed"
LII_PATH = os.path.join(PROCESSED, "language_instability_index.parquet")
CP_PATH = os.path.join(PROCESSED, "changepoint_density.parquet")
FACTORS_PATH = os.path.join(PROCESSED, "factor_trajectories.parquet")
EVENTS_PATH = "outputs/events.json"
OUT_PATH = os.path.join(PROCESSED, "cultural_shift_dataset.parquet")

EVENT_WINDOW = 2  # ± years around event onset


def main():
    t0 = time.time()
    print("=" * 60)
    print("Phase 16 — Build Cultural Shift Dataset")
    print("=" * 60)

    # ── Load data ──────────────────────────────────────────────
    print("[1/4] Loading pipeline outputs …")
    lii = pl.read_parquet(LII_PATH)
    cp = pl.read_parquet(CP_PATH)
    factors = pl.read_parquet(FACTORS_PATH)
    events = json.load(open(EVENTS_PATH))

    print(f"  LII: {lii.shape[0]} years")
    print(f"  Changepoints: {cp.shape[0]} years")
    print(f"  Factors: {factors.shape[0]} years × {factors.shape[1] - 1} factors")
    print(f"  Events: {len(events)} historical events")

    # ── Build labels ───────────────────────────────────────────
    # Use event ONSET (start year ± window) only, not full duration.
    # For very long events (>10 years), also mark the end.
    print("\n[2/4] Building labels from historical events …")
    years = sorted(lii["year"].to_list())
    event_years = set()
    for ev in events:
        # Always mark onset
        for y in range(ev["start_year"] - EVENT_WINDOW,
                       ev["start_year"] + EVENT_WINDOW + 1):
            event_years.add(y)
        # For short events (≤10 years), mark the full span
        duration = ev["end_year"] - ev["start_year"]
        if duration <= 10:
            for y in range(ev["start_year"], ev["end_year"] + 1):
                event_years.add(y)

    labels = [1 if y in event_years else 0 for y in years]
    n_pos = sum(labels)
    print(f"  Positive (shift) years: {n_pos}/{len(years)} ({100*n_pos/len(years):.1f}%)")

    # ── Build features ─────────────────────────────────────────
    print("\n[3/4] Engineering features …")

    # Start with year column
    df = pl.DataFrame({"year": years})

    # Join LII
    lii_renamed = lii.select([
        "year",
        pl.col("lii_value").alias("lii"),
        pl.col("concentration_ratio").alias("cr"),
        pl.col("broad_vs_narrow_score").alias("bvn"),
        pl.col("lii_fallback").alias("lii_fallback"),
    ])
    df = df.join(lii_renamed, on="year", how="left")

    # Join changepoint density
    cp_renamed = cp.select([
        "year",
        pl.col("n_changepoints").alias("cp_count"),
        pl.col("normalized_density").alias("cp_density"),
    ])
    df = df.join(cp_renamed, on="year", how="left")

    # Join factor scores
    df = df.join(factors, on="year", how="left")

    # Fill nulls with 0 for early years
    df = df.fill_null(0.0)

    # ── Derived features ───────────────────────────────────────
    # ΔLII (year-over-year change)
    lii_arr = df["lii"].to_numpy().astype(np.float64)
    delta_lii = np.diff(lii_arr, prepend=lii_arr[0])

    # Rolling mean of LII (10-year window)
    lii_rolling = np.convolve(lii_arr, np.ones(10) / 10, mode="same")

    # Z-scores
    lii_mean, lii_std = np.nanmean(lii_arr), np.nanstd(lii_arr)
    lii_z = (lii_arr - lii_mean) / (lii_std if lii_std > 0 else 1)

    cp_arr = df["cp_density"].to_numpy().astype(np.float64)
    delta_cp = np.diff(cp_arr, prepend=cp_arr[0])
    cp_mean, cp_std = np.nanmean(cp_arr), np.nanstd(cp_arr)
    cp_z = (cp_arr - cp_mean) / (cp_std if cp_std > 0 else 1)

    # Factor volatility: rolling std of L2 norm of factor vector
    factor_cols = [c for c in df.columns if c.startswith("factor_")]
    factor_mat = df.select(factor_cols).to_numpy().astype(np.float64)
    factor_norm = np.linalg.norm(factor_mat, axis=1)

    # Rolling std of factor norm (10-year window)
    factor_vol = np.full_like(factor_norm, np.nan)
    for i in range(len(factor_norm)):
        start = max(0, i - 9)
        window = factor_norm[start:i + 1]
        factor_vol[i] = np.std(window) if len(window) >= 3 else 0.0
    factor_vol = np.nan_to_num(factor_vol, 0.0)

    # Average pairwise factor correlation (rolling 10-year window)
    avg_factor_corr = np.zeros(len(years))
    for i in range(len(years)):
        start = max(0, i - 9)
        window = factor_mat[start:i + 1]
        if window.shape[0] >= 5:
            corr = np.corrcoef(window.T)
            mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
            avg_factor_corr[i] = np.nanmean(corr[mask])

    # ShiftScore (unsupervised)
    shift_score = lii_z + cp_z

    # Top eigenvalue of rolling factor covariance
    top_eigenval = np.zeros(len(years))
    for i in range(len(years)):
        start = max(0, i - 19)
        window = factor_mat[start:i + 1]
        if window.shape[0] >= 5:
            cov = np.cov(window.T)
            eigvals = np.linalg.eigvalsh(cov)
            top_eigenval[i] = eigvals[-1]

    # ── Assemble final dataset ─────────────────────────────────
    df = df.with_columns([
        pl.Series("lii_delta", delta_lii),
        pl.Series("lii_rolling_mean", lii_rolling),
        pl.Series("lii_z", lii_z),
        pl.Series("cp_delta", delta_cp),
        pl.Series("cp_z", cp_z),
        pl.Series("factor_volatility", factor_vol),
        pl.Series("avg_factor_corr", avg_factor_corr),
        pl.Series("top_eigenvalue", top_eigenval),
        pl.Series("shift_score", shift_score),
        pl.Series("label", labels),
    ])

    # Replace any remaining inf/nan
    for col in df.columns:
        if col not in ("year", "label"):
            arr = df[col].to_numpy()
            if np.issubdtype(arr.dtype, np.floating):
                arr = np.where(np.isfinite(arr), arr, 0.0)
                df = df.with_columns(pl.Series(col, arr))

    print(f"\n[4/4] Saving dataset …")
    df.write_parquet(OUT_PATH)
    print(f"  → {OUT_PATH}")
    print(f"  Shape: {df.shape[0]} years × {df.shape[1]} columns")
    print(f"  Features: {df.shape[1] - 2}")  # minus year and label
    print(f"  Columns: {df.columns}")
    print(f"\nPhase 16 complete.  Total time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
