"""
Phase 12 — Build Prediction Targets & Feature Table
=====================================================
Creates breakout labels and feature matrix from existing pipeline outputs.

Target:  Y_{w,t} = 1  if  freq_{w,t+h} / freq_{w,t} > threshold  (top-10% growth)
Features: latent_drift, curvature, local_instability, latent_level,
          regime dummies, recent changepoint count, years since first appearance,
          factor loadings (f1–f10)

Outputs:
  data/processed/prediction_targets.parquet
  data/processed/prediction_features.parquet
"""

import argparse
import os
import sys
import time

import numpy as np
import polars as pl

# ── Paths ──────────────────────────────────────────────────────────
PROCESSED     = "data/processed"
CLEAN_PANEL   = os.path.join(PROCESSED, "clean_panel.parquet")
WORD_FITS     = os.path.join(PROCESSED, "word_level_fits.parquet")
CHANGEPOINTS  = os.path.join(PROCESSED, "changepoints.parquet")
REGIMES       = os.path.join(PROCESSED, "regimes.parquet")
FACTOR_LOAD   = os.path.join(PROCESSED, "factor_loadings.parquet")
VOCAB_META    = os.path.join(PROCESSED, "vocabulary_metadata.parquet")

OUT_TARGETS   = os.path.join(PROCESSED, "prediction_targets.parquet")
OUT_FEATURES  = os.path.join(PROCESSED, "prediction_features.parquet")

# ── Config ─────────────────────────────────────────────────────────
HORIZON       = 5          # predict h years ahead
GROWTH_PCTILE = 90         # top-10% growth = breakout
YEAR_MIN      = 1800
YEAR_MAX      = 2008


def build_targets(horizon: int) -> pl.DataFrame:
    """Build future-growth labels from clean_panel frequencies."""
    print(f"[1/3] Building breakout targets (h={horizon}) …")

    panel = pl.read_parquet(CLEAN_PANEL, columns=["word", "year", "frequency"])

    # Current frequency
    current = panel.rename({"frequency": "freq_now"})

    # Future frequency (shift by horizon)
    future = panel.select(
        pl.col("word"),
        (pl.col("year") - horizon).alias("year"),      # align to "now"
        pl.col("frequency").alias("freq_future"),
    )

    # Join: for each (word, year), get freq_now and freq_{year+h}
    joined = current.join(future, on=["word", "year"], how="inner")

    # Growth ratio
    joined = joined.with_columns(
        (pl.col("freq_future") / pl.col("freq_now")).alias("future_growth_ratio")
    ).filter(
        pl.col("freq_now") > 0  # avoid division issues
    )

    # Breakout label: top GROWTH_PCTILE percentile per year
    threshold_per_year = joined.group_by("year").agg(
        pl.col("future_growth_ratio")
          .quantile(GROWTH_PCTILE / 100.0)
          .alias("threshold")
    )

    joined = joined.join(threshold_per_year, on="year")
    joined = joined.with_columns(
        (pl.col("future_growth_ratio") > pl.col("threshold"))
        .cast(pl.Int8)
        .alias("breakout_label")
    ).drop("threshold")

    # Keep only years where we can compute the label
    max_label_year = YEAR_MAX - horizon
    joined = joined.filter(pl.col("year") <= max_label_year)

    n_pos = joined.filter(pl.col("breakout_label") == 1).shape[0]
    n_tot = joined.shape[0]
    print(f"  {n_tot:,} word-years, {n_pos:,} breakouts ({100*n_pos/n_tot:.1f}%)")
    print(f"  Year range: {joined['year'].min()}–{joined['year'].max()}")

    joined.write_parquet(OUT_TARGETS)
    print(f"  → {OUT_TARGETS}  ({os.path.getsize(OUT_TARGETS)/1e6:.1f} MB)")
    return joined


def build_features(targets: pl.DataFrame) -> pl.DataFrame:
    """Build feature table from pipeline outputs, joined with targets."""
    print(f"\n[2/3] Building feature table …")

    # ── Word-level Kalman fits ──────────────────────────────────
    fits = pl.read_parquet(WORD_FITS, columns=[
        "word", "year", "latent_level", "latent_drift",
        "curvature", "local_instability",
    ])

    # ── Regime dummies ──────────────────────────────────────────
    regimes = pl.read_parquet(REGIMES, columns=["word", "year", "regime_label"])
    regimes = regimes.with_columns([
        (pl.col("regime_label") == "adoption").cast(pl.Int8).alias("regime_adoption"),
        (pl.col("regime_label") == "decline").cast(pl.Int8).alias("regime_decline"),
        (pl.col("regime_label") == "turbulent").cast(pl.Int8).alias("regime_turbulent"),
    ]).drop("regime_label")

    # ── Recent changepoint count (within last 5 years) ──────────
    cp = pl.read_parquet(CHANGEPOINTS, columns=["word", "changepoint_year"])

    # For each (word, year), count changepoints in [year-5, year]
    # Efficient: cross-join targets years with changepoints, then filter
    target_wy = targets.select("word", "year").unique()

    # Explode approach: for each changepoint, it contributes to years [cp_year, cp_year+5]
    cp_expanded = cp.with_columns([
        pl.col("changepoint_year").alias("cp_year"),
    ])
    # Join and filter
    cp_joined = target_wy.join(cp_expanded, on="word", how="left")
    cp_joined = cp_joined.with_columns(
        ((pl.col("changepoint_year") >= (pl.col("year") - 5)) &
         (pl.col("changepoint_year") <= pl.col("year")))
        .alias("recent")
    )
    recent_cp = cp_joined.group_by(["word", "year"]).agg(
        pl.col("recent").sum().alias("recent_changepoints")
    )

    # ── Years since first appearance ────────────────────────────
    vocab = pl.read_parquet(VOCAB_META, columns=["word", "first_year_seen"])
    # Will join on word only

    # ── Factor loadings (f1–f10) ────────────────────────────────
    factor_cols = [f"factor_{i}" for i in range(1, 11)]
    try:
        loadings = pl.read_parquet(FACTOR_LOAD, columns=["word"] + factor_cols)
        has_factors = True
    except Exception:
        has_factors = False
        print("  (factor loadings not available, skipping)")

    # ── Assemble ────────────────────────────────────────────────
    print("  Joining features …")
    feat = targets.select("word", "year", "breakout_label")

    # Join Kalman fits
    feat = feat.join(fits, on=["word", "year"], how="left")

    # Join regime dummies
    feat = feat.join(regimes, on=["word", "year"], how="left")

    # Join recent changepoints
    feat = feat.join(recent_cp, on=["word", "year"], how="left")
    feat = feat.with_columns(
        pl.col("recent_changepoints").fill_null(0)
    )

    # Join years since first appearance
    feat = feat.join(vocab, on="word", how="left")
    feat = feat.with_columns(
        (pl.col("year") - pl.col("first_year_seen")).alias("years_since_first")
    ).drop("first_year_seen")

    # Join factor loadings (static per word)
    if has_factors:
        feat = feat.join(loadings, on="word", how="left")

    # ── Recent log-frequency change (5-year delta) ──────────────
    panel = pl.read_parquet(CLEAN_PANEL, columns=["word", "year", "frequency"])
    panel = panel.with_columns(
        pl.col("frequency").log().alias("log_freq")
    )
    past = panel.select(
        pl.col("word"),
        (pl.col("year") + 5).alias("year"),
        pl.col("log_freq").alias("log_freq_past"),
    )
    log_change = panel.select("word", "year", "log_freq").join(
        past, on=["word", "year"], how="inner"
    ).with_columns(
        (pl.col("log_freq") - pl.col("log_freq_past")).alias("recent_log_change")
    ).select("word", "year", "recent_log_change")

    feat = feat.join(log_change, on=["word", "year"], how="left")

    # ── Drop rows with null features ────────────────────────────
    feature_cols = [
        "latent_drift", "curvature", "local_instability", "latent_level",
        "regime_adoption", "regime_decline", "regime_turbulent",
        "recent_changepoints", "years_since_first", "recent_log_change",
    ]
    if has_factors:
        feature_cols += factor_cols

    before = feat.shape[0]
    feat = feat.drop_nulls(subset=feature_cols)
    after = feat.shape[0]
    print(f"  Dropped {before - after:,} rows with nulls → {after:,} rows")

    feat.write_parquet(OUT_FEATURES)
    print(f"  → {OUT_FEATURES}  ({os.path.getsize(OUT_FEATURES)/1e6:.1f} MB)")

    # Summary
    print(f"\n[3/3] Feature summary")
    print(f"  Rows:     {feat.shape[0]:,}")
    print(f"  Features: {len(feature_cols)}")
    print(f"  Breakout rate: {feat['breakout_label'].mean():.3f}")
    print(f"  Year range: {feat['year'].min()}–{feat['year'].max()}")

    return feat


def main():
    parser = argparse.ArgumentParser(description="Phase 12: Build prediction data")
    parser.add_argument("--horizon", type=int, default=HORIZON)
    args = parser.parse_args()

    t0 = time.time()
    print("=" * 60)
    print(f"Phase 12 — Build Prediction Data  (h={args.horizon})")
    print("=" * 60)

    targets = build_targets(args.horizon)
    features = build_features(targets)

    print(f"\nPhase 12 complete.  Total time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
