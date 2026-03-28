"""Phase 6 — Regime Labeling

For each (word, year), assigns an interpretable dynamic regime label based on
global robust z-scores of latent_drift and local_instability from Phase 4.

Labels (in priority order):
  turbulent  — z_instability > threshold
  adoption   — z_drift > threshold
  decline    — z_drift < -threshold
  stable     — otherwise

Also adds a near_cp boolean column marking rows within PROXIMITY_WINDOW years
of any detected changepoint (from Phase 5), for downstream diagnostic use.

Outputs:
  data/processed/regimes.parquet  — one row per (word, year), same shape as word_level_fits

Usage:
    python pipeline/06_regimes.py               # full run (~10s)
    python pipeline/06_regimes.py --plot        # also save sanity plot
    python pipeline/06_regimes.py --threshold 1.5   # stricter (fewer active labels)
"""

import argparse
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import polars as pl

# ── Configuration ─────────────────────────────────────────────
PROCESSED_DIR     = "data/processed"
OUTPUTS_DIR       = "outputs"

WORD_FITS_PATH    = os.path.join(PROCESSED_DIR, "word_level_fits.parquet")
CHANGEPOINTS_PATH = os.path.join(PROCESSED_DIR, "changepoints.parquet")
REGIMES_PATH      = os.path.join(PROCESSED_DIR, "regimes.parquet")
PLOT_PATH         = os.path.join(OUTPUTS_DIR,   "phase6_sanity_check.png")

YEAR_MIN          = 1800
YEAR_MAX          = 2008
THRESHOLD         = 1.0          # one robust sigma; configurable via --threshold
PROXIMITY_WINDOW  = 3            # ±years around a changepoint for the near_cp flag
MAD_SCALE         = 1.4826       # converts MAD to σ-equivalent (consistency factor)

VALID_LABELS      = {"adoption", "decline", "turbulent", "stable"}
SAMPLE_WORDS      = ["war", "computer", "telegraph", "cholera"]

os.makedirs(OUTPUTS_DIR, exist_ok=True)


# ── Core functions ─────────────────────────────────────────────

def compute_global_robust_stats(fits_df: pl.DataFrame) -> tuple:
    """Compute global median and MAD for latent_drift and local_instability.

    Uses the full 14.7M-row DataFrame so that z-scores are comparable across
    all words and all years (enabling cross-era comparisons).

    Returns:
        (drift_median, drift_mad, inst_median, inst_mad)
    """
    drift_vals = fits_df["latent_drift"].drop_nulls()
    inst_vals  = fits_df["local_instability"].drop_nulls()

    drift_median = float(drift_vals.median())
    drift_mad    = float((drift_vals - drift_median).abs().median())

    inst_median  = float(inst_vals.median())
    inst_mad     = float((inst_vals - inst_median).abs().median())

    if drift_mad < 1e-10:
        raise ValueError(
            f"Drift MAD is effectively zero ({drift_mad:.2e}). "
            "This indicates a modeling problem in Phase 4 — all words "
            "converged to the same drift value."
        )
    if inst_mad < 1e-10:
        raise ValueError(
            f"Instability MAD is effectively zero ({inst_mad:.2e}). "
            "This indicates a modeling problem in Phase 4."
        )

    return drift_median, drift_mad, inst_median, inst_mad


def assign_labels(
    fits_df: pl.DataFrame,
    drift_median: float,
    drift_mad: float,
    inst_median: float,
    inst_mad: float,
    threshold: float,
) -> pl.DataFrame:
    """Compute global robust z-scores and assign regime labels.

    local_instability nulls (early years before rolling window fills) are
    filled with 0.0 before z-scoring. This maps to z_I ≈ -1.83, safely below
    the turbulent threshold, so early sparse years are never falsely labeled
    turbulent — they receive labels from drift alone.

    Label priority (first matching rule wins):
      1. turbulent  — z_instability > threshold
      2. adoption   — z_drift > threshold
      3. decline    — z_drift < -threshold
      4. stable     — otherwise

    Returns the input DataFrame augmented with z_drift, z_instability,
    and regime_label columns.
    """
    result = fits_df.with_columns([
        ((pl.col("latent_drift") - drift_median) / (MAD_SCALE * drift_mad))
            .alias("z_drift"),
        ((pl.col("local_instability").fill_null(0.0) - inst_median) / (MAD_SCALE * inst_mad))
            .alias("z_instability"),
    ])

    result = result.with_columns(
        pl.when(pl.col("z_instability") > threshold)
            .then(pl.lit("turbulent"))
        .when(pl.col("z_drift") > threshold)
            .then(pl.lit("adoption"))
        .when(pl.col("z_drift") < -threshold)
            .then(pl.lit("decline"))
        .otherwise(pl.lit("stable"))
        .alias("regime_label")
    )

    return result


def build_proximity_set(cp_df: pl.DataFrame, window: int) -> pl.DataFrame:
    """Return a (word, year) frame marking all rows within window of any changepoint.

    For each detected changepoint (word, year_cp), generates year_cp ± window
    and returns a deduplicated frame suitable for a left join.
    """
    frames = [
        cp_df.select(["word", pl.col("changepoint_year").alias("year")])
             .with_columns((pl.col("year") + offset).alias("year"))
        for offset in range(-window, window + 1)
    ]
    return pl.concat(frames).unique()


def make_sanity_plot(regimes_df: pl.DataFrame, words: list, out_path: str) -> None:
    """Save a multi-panel plot showing the regime timeline for each sample word."""
    label_colors = {
        "adoption":  "#2196F3",   # blue
        "decline":   "#F44336",   # red
        "turbulent": "#FF9800",   # orange
        "stable":    "#9E9E9E",   # grey
    }

    n = len(words)
    fig, axes = plt.subplots(n, 1, figsize=(14, 3 * n), sharex=True)
    if n == 1:
        axes = [axes]

    for ax, word in zip(axes, words):
        sub = regimes_df.filter(pl.col("word") == word).sort("year")
        if len(sub) == 0:
            ax.set_title(f"'{word}' — NOT IN VOCABULARY")
            continue

        years  = sub["year"].to_numpy()
        labels = sub["regime_label"].to_list()
        z_drift = sub["z_drift"].to_numpy()

        # Colour each year according to its regime
        for i, (yr, lbl) in enumerate(zip(years, labels)):
            ax.axvspan(yr - 0.5, yr + 0.5, color=label_colors[lbl], alpha=0.5)

        # Overlay z_drift as a line for reference
        ax2 = ax.twinx()
        ax2.plot(years, z_drift, color="black", linewidth=0.8, alpha=0.7)
        ax2.axhline(0, color="black", linewidth=0.4, linestyle="--")
        ax2.set_ylabel("z_drift", fontsize=8)

        ax.set_title(f"'{word}'", fontsize=10)
        ax.set_ylabel("regime", fontsize=8)
        ax.set_yticks([])

    # Legend
    patches = [
        mpatches.Patch(color=c, label=lbl, alpha=0.7)
        for lbl, c in label_colors.items()
    ]
    axes[-1].set_xlabel("year")
    fig.legend(handles=patches, loc="lower center", ncol=4, fontsize=9,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Phase 6 — Regime Labels (sample words)", fontsize=12)
    fig.tight_layout(rect=[0, 0.04, 1, 0.97])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Sanity plot saved to {out_path}")


# ── Main ───────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 6: Regime Labeling")
    parser.add_argument(
        "--threshold", type=float, default=THRESHOLD,
        help=f"Robust z-score cutoff for active regimes (default: {THRESHOLD})"
    )
    parser.add_argument(
        "--plot", action="store_true",
        help="Save a sanity plot to outputs/phase6_sanity_check.png"
    )
    args = parser.parse_args()

    t0 = time.time()
    print("=" * 60)
    print("Phase 6 — Regime Labeling")
    print(f"  threshold      : {args.threshold}")
    print(f"  proximity_window: ±{PROXIMITY_WINDOW} years")
    print("=" * 60)

    # ── Step 1: Load inputs ───────────────────────────────────
    print("\nStep 1: Loading word_level_fits …")
    fits = pl.read_parquet(
        WORD_FITS_PATH,
        columns=["word", "year", "latent_drift", "local_instability"],
    )
    print(f"  {len(fits):,} rows loaded ({fits['word'].n_unique():,} words)")

    print("Step 1b: Loading changepoints …")
    cp = pl.read_parquet(CHANGEPOINTS_PATH)
    print(f"  {len(cp):,} changepoints detected")

    # ── Step 2: Global robust statistics ─────────────────────
    print("\nStep 2: Computing global robust z-score statistics …")
    drift_median, drift_mad, inst_median, inst_mad = compute_global_robust_stats(fits)
    print(f"  latent_drift   : median={drift_median:.6f}, MAD={drift_mad:.6f} "
          f"  → σ̂={MAD_SCALE * drift_mad:.6f}")
    print(f"  local_instability: median={inst_median:.6f}, MAD={inst_mad:.6f} "
          f"  → σ̂={MAD_SCALE * inst_mad:.6f}")

    # ── Step 3: Assign z-scores and regime labels ─────────────
    print("\nStep 3: Assigning z-scores and regime labels …")
    labeled = assign_labels(
        fits, drift_median, drift_mad, inst_median, inst_mad, args.threshold
    )

    # ── Step 4: Build and join changepoint proximity flag ─────
    print("\nStep 4: Building changepoint proximity flag (±{} years) …".format(
        PROXIMITY_WINDOW))
    cp_set = build_proximity_set(cp, PROXIMITY_WINDOW)
    labeled = labeled.join(
        cp_set.with_columns(pl.lit(True).alias("near_cp")),
        on=["word", "year"],
        how="left",
    ).with_columns(pl.col("near_cp").fill_null(False))

    # ── Step 5: Validate label distribution ──────────────────
    print("\nStep 5: Label distribution:")
    n_total = len(labeled)
    counts = (
        labeled.group_by("regime_label")
               .agg(pl.len().alias("n"))
               .sort("n", descending=True)
    )
    for row in counts.iter_rows(named=True):
        frac = row["n"] / n_total
        print(f"  {row['regime_label']:12s} : {row['n']:>10,}  ({frac:.1%})")
        if frac < 0.02:
            raise ValueError(
                f"Degenerate distribution: '{row['regime_label']}' has only "
                f"{frac:.1%} of rows. Check threshold or input data."
            )
        if frac > 0.80:
            raise ValueError(
                f"Degenerate distribution: '{row['regime_label']}' dominates "
                f"at {frac:.1%}. Check threshold or input data."
            )

    # ── Step 6: Select output columns and write ────────────────
    print("\nStep 6: Writing regimes.parquet …")
    output = labeled.select([
        "word", "year", "regime_label",
        "z_drift", "z_instability",
        "near_cp",
    ])
    output.write_parquet(REGIMES_PATH)

    elapsed = time.time() - t0
    print(f"\n  Done. {len(output):,} rows written to {REGIMES_PATH}")
    print(f"  Total time: {elapsed:.1f}s")

    # ── Step 7: Optional sanity plot ──────────────────────────
    if args.plot:
        print("\nStep 7: Generating sanity plot …")
        vocab = set(labeled["word"].unique().to_list())
        plot_words = [w for w in SAMPLE_WORDS if w in vocab]
        if plot_words:
            make_sanity_plot(labeled, plot_words, PLOT_PATH)
        else:
            print("  Warning: none of the sample words are in vocabulary — skipping plot")

    print("\nPhase 6 complete.")


if __name__ == "__main__":
    main()
