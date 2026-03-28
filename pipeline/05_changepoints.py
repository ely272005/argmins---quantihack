"""Phase 5 — Changepoint Detection

For each word, runs PELT (Pruned Exact Linear Time) on the Kalman-smoothed
latent_level series from Phase 4 to detect structural breaks.

Outputs:
  data/processed/changepoints.parquet      — one row per detected break
  data/processed/changepoint_density.parquet — one row per year 1800-2008

Usage:
    python pipeline/05_changepoints.py            # full run (~30s)
    python pipeline/05_changepoints.py --plot     # also save sanity plot
    python pipeline/05_changepoints.py --penalty 10   # stricter (fewer breaks)
"""

import argparse
import os
import time
import warnings

import numpy as np
import pandas as pd
import polars as pl
import ruptures as rpt
from joblib import Parallel, delayed
from tqdm import tqdm

# ── Configuration ────────────────────────────────────────────
PROCESSED_DIR = "data/processed"
OUTPUTS_DIR   = "outputs"

WORD_FITS_PATH    = os.path.join(PROCESSED_DIR, "word_level_fits.parquet")
CHANGEPOINTS_PATH = os.path.join(PROCESSED_DIR, "changepoints.parquet")
DENSITY_PATH      = os.path.join(PROCESSED_DIR, "changepoint_density.parquet")
PLOT_PATH         = os.path.join(OUTPUTS_DIR,   "phase5_sanity_check.png")

YEAR_MIN    = 1800
YEAR_MAX    = 2008
DEFAULT_PEN = 3        # PELT penalty — lower = more sensitive
SIGNAL_COL  = "latent_level"

SAMPLE_WORDS = ["war", "computer", "typewriter", "telegram"]

os.makedirs(OUTPUTS_DIR, exist_ok=True)


# ── Core detection function ───────────────────────────────────

def detect_changepoints(word: str, years: np.ndarray, signal: np.ndarray, pen: float) -> list:
    """Run PELT on one word's latent_level series.

    Returns a list of changepoint years (may be empty).
    The PELT sentinel (index = len(signal)) is dropped.
    """
    try:
        algo = rpt.Pelt(model="rbf").fit(signal.reshape(-1, 1))
        bkps = algo.predict(pen=pen)
        # bkps[-1] is always len(signal) (sentinel), skip it
        cp_indices = bkps[:-1]
        return [int(years[i - 1]) for i in cp_indices if 0 < i <= len(years)]
    except Exception:
        return []


def process_word(word: str, years: np.ndarray, signal: np.ndarray, pen: float) -> list:
    """Wrapper returning list of dicts, one per detected changepoint."""
    cp_years = detect_changepoints(word, years, signal, pen)
    return [
        {"word": word, "changepoint_year": yr, "signal": SIGNAL_COL, "penalty": float(pen)}
        for yr in cp_years
    ]


# ── Density builder ───────────────────────────────────────────

def build_density(cp_df: pl.DataFrame, n_words: int) -> pl.DataFrame:
    """Aggregate changepoints into a per-year density series."""
    all_years = pl.DataFrame({"year": list(range(YEAR_MIN, YEAR_MAX + 1))})

    if len(cp_df) == 0:
        return all_years.with_columns([
            pl.lit(0).cast(pl.Int64).alias("n_changepoints"),
            pl.lit(0.0).alias("normalized_density"),
        ])

    counts = (
        cp_df
        .group_by("changepoint_year")
        .agg(pl.len().alias("n_changepoints"))
        .rename({"changepoint_year": "year"})
    )
    density = (
        all_years
        .join(counts, on="year", how="left")
        .with_columns(pl.col("n_changepoints").fill_null(0))
        .with_columns(
            (pl.col("n_changepoints").cast(pl.Float64) / n_words)
            .alias("normalized_density")
        )
        .sort("year")
    )
    return density


# ── Sanity plot ───────────────────────────────────────────────

def make_sanity_plot(fits_df: pd.DataFrame, cp_dict: dict, words: list, out_path: str):
    """Plot latent_level with changepoint markers for sample words."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(words)
    fig, axes = plt.subplots(n, 1, figsize=(12, 3 * n), sharex=False)
    if n == 1:
        axes = [axes]

    for ax, word in zip(axes, words):
        sub = fits_df[fits_df["word"] == word].sort_values("year")
        if sub.empty:
            ax.set_title(f"{word} (not found)")
            continue
        ax.plot(sub["year"], sub[SIGNAL_COL], lw=1.5, color="steelblue", label="latent level")
        for yr in cp_dict.get(word, []):
            ax.axvline(yr, color="red", lw=1.0, ls="--", alpha=0.7)
        cp_list = cp_dict.get(word, [])
        ax.set_title(f"{word}  —  {len(cp_list)} changepoint(s): {cp_list}")
        ax.set_ylabel("log-intensity")
        ax.tick_params(axis="x", rotation=30)

    fig.suptitle("Phase 5 — Changepoint Detection (sanity check)", y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot saved: {out_path}")


# ── Main ──────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Phase 5: Changepoint detection")
    parser.add_argument("--penalty", type=float, default=DEFAULT_PEN,
                        help=f"PELT penalty (default={DEFAULT_PEN}). Lower = more breaks.")
    parser.add_argument("--plot", action="store_true",
                        help="Save sanity check plot for sample words")
    args = parser.parse_args()

    print("=== Phase 5: Changepoint Detection ===\n")
    t0 = time.time()

    # [1/5] Load word fits
    print("[1/5] Loading word_level_fits.parquet...")
    fits_pl = pl.read_parquet(WORD_FITS_PATH, columns=["word", "year", SIGNAL_COL])
    n_words = fits_pl["word"].n_unique()
    print(f"  Words loaded: {n_words:,}")
    print(f"  Rows loaded:  {len(fits_pl):,}")
    print(f"  PELT penalty: {args.penalty}")

    # [2/5] Group by word into sorted arrays
    print("\n[2/5] Grouping by word...")
    fits_pd = fits_pl.to_pandas()
    groups = []
    for word, grp in fits_pd.groupby("word", sort=False):
        grp_sorted = grp.sort_values("year")
        groups.append((
            word,
            grp_sorted["year"].to_numpy(),
            grp_sorted[SIGNAL_COL].to_numpy(),
        ))
    print(f"  Groups formed: {len(groups):,}")

    # [3/5] Parallel PELT
    print(f"\n[3/5] Running PELT (parallel, n_jobs=-1)...")
    raw_results = Parallel(n_jobs=-1)(
        delayed(process_word)(w, yrs, sig, args.penalty)
        for w, yrs, sig in tqdm(groups, desc="detecting changepoints")
    )

    # [4/5] Build changepoints table
    print("\n[4/5] Building changepoints table...")
    all_rows = [row for word_rows in raw_results for row in word_rows]
    n_total_cps = len(all_rows)
    n_words_with_cps = sum(1 for r in raw_results if len(r) > 0)

    if all_rows:
        cp_pl = pl.DataFrame(all_rows).with_columns(
            pl.col("changepoint_year").cast(pl.Int64),
            pl.col("penalty").cast(pl.Float64),
        )
    else:
        cp_pl = pl.DataFrame(schema={
            "word": pl.String,
            "changepoint_year": pl.Int64,
            "signal": pl.String,
            "penalty": pl.Float64,
        })

    cp_pl.write_parquet(CHANGEPOINTS_PATH)
    print(f"  Total changepoints detected: {n_total_cps:,}")
    print(f"  Words with ≥1 changepoint:   {n_words_with_cps:,} / {n_words:,} "
          f"({100*n_words_with_cps/n_words:.1f}%)")
    print(f"  Mean changepoints per word:  {n_total_cps/n_words:.2f}")
    print(f"  Saved: {CHANGEPOINTS_PATH}  ({os.path.getsize(CHANGEPOINTS_PATH)/1e6:.1f} MB)")

    # [5/5] Build density
    print("\n[5/5] Building changepoint density...")
    density_pl = build_density(cp_pl, n_words)
    density_pl.write_parquet(DENSITY_PATH)

    peak_row = density_pl.sort("n_changepoints", descending=True).row(0, named=True)
    print(f"  Peak year: {peak_row['year']}  ({peak_row['n_changepoints']:,} changepoints)")
    print(f"  Saved: {DENSITY_PATH}  ({os.path.getsize(DENSITY_PATH)/1e6:.1f} MB)")

    # Show top-10 peak years
    print("\n  Top 10 changepoint-density years:")
    top10 = density_pl.sort("n_changepoints", descending=True).head(10)
    for row in top10.iter_rows(named=True):
        print(f"    {row['year']}: {row['n_changepoints']:,} words  "
              f"({row['normalized_density']*100:.2f}%)")

    # Sanity plot
    if args.plot:
        print("\nGenerating sanity check plot...")
        cp_dict = {}
        for word_rows in raw_results:
            for row in word_rows:
                cp_dict.setdefault(row["word"], []).append(row["changepoint_year"])
        make_sanity_plot(fits_pd, cp_dict, SAMPLE_WORDS, PLOT_PATH)

    print(f"\nTotal elapsed time: {time.time()-t0:.1f}s")
    print("Done.")


if __name__ == "__main__":
    main()
