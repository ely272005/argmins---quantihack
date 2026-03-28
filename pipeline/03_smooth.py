"""Phase 3 — Diagnostic Smooth Series

For each word in the clean panel, fit a smoothing spline to log-frequency
over time and extract first and second derivatives (drift and curvature).

Usage:
    python pipeline/03_smooth.py          # process all kept words
    python pipeline/03_smooth.py --plot   # also save sanity check plot
"""

import argparse
import os
import time
import warnings

import numpy as np
import pandas as pd
import polars as pl
from joblib import Parallel, delayed
from scipy.interpolate import UnivariateSpline
from tqdm import tqdm

# ── Configuration ────────────────────────────────────────────
PROCESSED_DIR = "data/processed"
OUTPUTS_DIR   = "outputs"

CLEAN_PANEL_PATH = os.path.join(PROCESSED_DIR, "clean_panel.parquet")
VOCAB_META_PATH  = os.path.join(PROCESSED_DIR, "vocabulary_metadata.parquet")
OUTPUT_PATH      = os.path.join(PROCESSED_DIR, "smoothed_word_series.parquet")
PLOT_PATH        = os.path.join(OUTPUTS_DIR, "phase3_sanity_check.png")

LOG_FLOOR = 1e-9       # ε in log(f + ε)
SPLINE_DEGREE = 3       # cubic spline
SMOOTH_FACTOR = 0.3     # s = len(years) * SMOOTH_FACTOR  (lower = less smooth)
MIN_OBS = 5             # skip words with fewer observed years

SAMPLE_WORDS = ["war", "computer", "internet", "pandemic"]

os.makedirs(OUTPUTS_DIR, exist_ok=True)


def smooth_word(word, years, log_freqs):
    """Fit a smoothing spline to one word's log-frequency series.

    Returns a DataFrame with columns:
        word, year, log_freq, smooth_level, smooth_drift, smooth_curvature
    for every integer year in the word's observed range.
    """
    if len(years) < MIN_OBS:
        return None

    year_min, year_max = int(years.min()), int(years.max())
    year_grid = np.arange(year_min, year_max + 1, dtype=float)

    s = max(1.0, len(years) * SMOOTH_FACTOR)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            spline = UnivariateSpline(years, log_freqs, k=SPLINE_DEGREE, s=s)
        except Exception:
            return None

    smooth_level     = spline(year_grid)
    smooth_drift     = spline.derivative(n=1)(year_grid)
    smooth_curvature = spline.derivative(n=2)(year_grid)

    # For observed years, attach the actual log_freq; fill NaN for interpolated years
    obs_map = dict(zip(years.astype(int), log_freqs))
    log_freq_col = np.array([obs_map.get(int(y), np.nan) for y in year_grid])

    return pd.DataFrame({
        "word":             word,
        "year":             year_grid.astype(int),
        "log_freq":         log_freq_col,
        "smooth_level":     smooth_level,
        "smooth_drift":     smooth_drift,
        "smooth_curvature": smooth_curvature,
    })


def process_word_group(word, group_df):
    """Extract arrays and call smooth_word."""
    years     = group_df["year"].values.astype(float)
    freqs     = group_df["frequency"].values.astype(float)
    log_freqs = np.log(freqs + LOG_FLOOR)
    return smooth_word(word, years, log_freqs)


def make_sanity_plot(smoothed_df, words, out_path):
    """Plot smooth_level, smooth_drift, smooth_curvature for a list of words."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, len(words), figsize=(5 * len(words), 9))
    if len(words) == 1:
        axes = axes.reshape(3, 1)

    row_titles = ["Log-freq (smooth level)", "Drift (1st deriv)", "Curvature (2nd deriv)"]
    cols_to_plot = ["smooth_level", "smooth_drift", "smooth_curvature"]

    for col_idx, word in enumerate(words):
        sub = smoothed_df[smoothed_df["word"] == word].sort_values("year")
        if sub.empty:
            for row_idx in range(3):
                axes[row_idx][col_idx].set_title(f"{word}\n(not found)")
            continue
        for row_idx, (col, row_title) in enumerate(zip(cols_to_plot, row_titles)):
            ax = axes[row_idx][col_idx]
            ax.plot(sub["year"], sub[col], lw=1.5, color="steelblue")
            ax.axhline(0, color="grey", lw=0.5, ls="--")
            ax.set_title(f"{word}" if row_idx == 0 else "")
            ax.set_ylabel(row_title if col_idx == 0 else "")
            ax.tick_params(axis="x", rotation=45)

    fig.suptitle("Phase 3 — Smoothed Word Series (sanity check)", y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot saved: {out_path}")


def main(make_plot=False):
    print("=== Phase 3: Smooth & Derivatives ===\n")
    t0 = time.time()

    # 1. Load clean panel and vocabulary metadata
    print("[1/4] Loading clean panel and vocabulary metadata...")
    panel = pl.read_parquet(CLEAN_PANEL_PATH)
    meta  = pl.read_parquet(VOCAB_META_PATH)

    kept_words = set(
        meta.filter(pl.col("is_kept")).select("word").to_series().to_list()
    )
    panel_pd = (
        panel
        .filter(pl.col("word").is_in(list(kept_words)))
        .select(["word", "year", "frequency"])
        .to_pandas()
    )
    print(f"  Words to process: {len(kept_words):,}")
    print(f"  Rows to process:  {len(panel_pd):,}")

    # 2. Group by word
    print("\n[2/4] Fitting splines (parallel)...")
    grouped = list(panel_pd.groupby("word", sort=False))

    results = Parallel(n_jobs=-1)(
        delayed(process_word_group)(word, grp)
        for word, grp in tqdm(grouped, desc="smoothing words")
    )

    # 3. Collect results
    print("\n[3/4] Collecting results...")
    valid = [r for r in results if r is not None]
    skipped = len(results) - len(valid)
    if skipped:
        print(f"  Skipped {skipped} word(s) with fewer than {MIN_OBS} observations")

    smoothed_df = pd.concat(valid, ignore_index=True)
    print(f"  Total rows in smoothed series: {len(smoothed_df):,}")

    # 4. Save
    print("\n[4/4] Saving output...")
    pl.from_pandas(smoothed_df).write_parquet(OUTPUT_PATH)
    print(f"  Saved: {OUTPUT_PATH}  ({os.path.getsize(OUTPUT_PATH)/1e6:.1f} MB)")

    elapsed = time.time() - t0
    print(f"\n  Total time: {elapsed:.1f}s")

    # Optional sanity plot
    if make_plot:
        print("\nGenerating sanity check plot...")
        make_sanity_plot(smoothed_df, SAMPLE_WORDS, PLOT_PATH)

    print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 3: Smooth word series")
    parser.add_argument("--plot", action="store_true", help="Save sanity check plot")
    args = parser.parse_args()
    main(make_plot=args.plot)
