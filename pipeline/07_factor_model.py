"""Phase 7 — System-Wide Latent Factor Model

Fits a PCA-based dynamic factor model over all word latent_drift trajectories
from Phase 4. Decomposes the T×W drift matrix (209 years × ~70,535 words) into
k shared factors representing hidden system-wide cultural forces.

Model (from the roadmap):
    x_t = B * f_t + u_t          (factor decomposition)
    f_{t+1} = A * f_t + ξ_t     (factor dynamics)
    ξ_t ~ N(0, Σ_t)

First pass: PCA on per-word-standardized latent_drift matrix.

Signal choice — latent_drift:
  • No nulls (Phase 4 tests guarantee this)
  • Stationary around zero, unlike latent_level which has secular trends
  • Captures usage velocity: co-movement in drift identifies shared cultural forces
  • Per-word standardization makes PCA sensitive to co-movement, not scale

Outputs:
  data/processed/factor_trajectories.parquet   — shape (209, k+1)
  data/processed/factor_loadings.parquet       — shape (~70535, k+1)
  data/processed/factor_metadata.json

Usage:
    python pipeline/07_factor_model.py                 # default k=10
    python pipeline/07_factor_model.py --n-factors 20
    python pipeline/07_factor_model.py --plot
"""

import argparse
import json
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from sklearn.decomposition import PCA

# ── Configuration ─────────────────────────────────────────────
PROCESSED_DIR         = "data/processed"
OUTPUTS_DIR           = "outputs"

WORD_FITS_PATH        = os.path.join(PROCESSED_DIR, "word_level_fits.parquet")
TRAJECTORIES_PATH     = os.path.join(PROCESSED_DIR, "factor_trajectories.parquet")
LOADINGS_PATH         = os.path.join(PROCESSED_DIR, "factor_loadings.parquet")
METADATA_PATH         = os.path.join(PROCESSED_DIR, "factor_metadata.json")
PLOT_PATH             = os.path.join(OUTPUTS_DIR,   "phase7_sanity_check.png")

YEAR_MIN           = 1800
YEAR_MAX = 2019
SIGNAL_COL         = "latent_drift"
DEFAULT_N_FACTORS  = 10
TOP_LOADING_N      = 10    # words per panel in the loading bar chart

os.makedirs(OUTPUTS_DIR, exist_ok=True)


# ── Core functions ─────────────────────────────────────────────

def load_drift_matrix(path: str) -> tuple:
    """Load word_level_fits and pivot to a (T, W) numpy array.

    Reads only the three columns needed (word, year, latent_drift) to keep
    memory usage low (~130 MB instead of the full 605 MB parquet).

    Returns:
        drift_matrix — np.ndarray shape (T, W) = (209, n_words)
        years        — list[int] length T, sorted ascending
        words        — list[str] length W, in column order
    """
    fits = pl.read_parquet(path, columns=["word", "year", SIGNAL_COL])

    # Pivot: rows = years (T), columns = words (W), values = latent_drift
    wide = (
        fits
        .sort(["year", "word"])
        .pivot(index="year", on="word", values=SIGNAL_COL)
        .sort("year")
    )

    years = wide["year"].to_list()
    words = [c for c in wide.columns if c != "year"]

    # Extract to float64 numpy array: shape (T, W)
    drift_matrix = wide.select(words).to_numpy().astype(np.float64)

    return drift_matrix, years, words


def standardize_matrix(drift_matrix: np.ndarray, words: list) -> tuple:
    """Per-word z-score standardization of the T×W drift matrix.

    Subtracts each word's mean over 209 years and divides by its std (ddof=1).
    This makes PCA equivalent to eigendecomposition of the correlation matrix,
    so factors capture co-movement patterns rather than being dominated by
    high-variance words.

    Degenerate words (std < 1e-10, i.e., constant drift over all years) are
    logged and dropped. In practice this should not occur because Phase 4's
    Kalman smoother always produces some variation in latent_drift.

    Returns:
        X_std       — standardized matrix, shape (T, W_clean)
        means       — shape (W_clean,)
        stds        — shape (W_clean,)
        clean_words — list[str], words retained after dropping degenerate
    """
    means = drift_matrix.mean(axis=0)           # shape (W,)
    stds  = drift_matrix.std(axis=0, ddof=1)    # shape (W,)

    degenerate_mask = stds < 1e-10
    n_degen = int(degenerate_mask.sum())
    if n_degen > 0:
        degen_words = [w for w, bad in zip(words, degenerate_mask) if bad]
        print(f"  WARNING: {n_degen} degenerate words (std < 1e-10) dropped: "
              f"{degen_words[:5]}{'...' if n_degen > 5 else ''}")
        keep_mask  = ~degenerate_mask
        drift_matrix = drift_matrix[:, keep_mask]
        means  = means[keep_mask]
        stds   = stds[keep_mask]
        words  = [w for w, bad in zip(words, degenerate_mask) if not bad]

    X_std = (drift_matrix - means) / stds   # broadcast: (T,W) / (W,)
    return X_std, means, stds, words


def fit_pca(X_std: np.ndarray, n_factors: int) -> PCA:
    """Fit truncated PCA on the standardized T×W matrix.

    Uses svd_solver='randomized' (Halko 2011) for efficiency — with k=10
    factors and a 209×70535 matrix, randomized SVD is much faster than full
    SVD while being numerically identical for the top k components.

    random_state=42 ensures reproducible results.

    Args:
        X_std     — shape (T, W), standardized drift matrix
        n_factors — number of components to retain

    Returns:
        Fitted sklearn PCA object.
    """
    pca = PCA(
        n_components=n_factors,
        svd_solver="randomized",
        random_state=42,
    )
    pca.fit(X_std)
    return pca


def extract_outputs(
    pca: PCA,
    X_std: np.ndarray,
    years: list,
    words: list,
    n_factors: int,
) -> tuple:
    """Extract factor trajectories, loadings, and metadata from a fitted PCA.

    In sklearn's PCA convention:
      - pca.components_  : shape (k, W) — the factor directions in word space
      - pca.transform(X) : shape (T, k) — the factor scores for each year

    Factor trajectories: scores = X_std @ components.T, shape (T, k)
      → Each row is a year; each column is one factor's value that year.
    Factor loadings: components_.T, shape (W, k)
      → Each row is a word; each column is that word's loading on one factor.

    Returns:
        trajectories_df — Polars DataFrame (T, k+1): year, factor_1, …, factor_k
        loadings_df     — Polars DataFrame (W, k+1): word, factor_1, …, factor_k
        metadata        — dict suitable for json.dump
    """
    factor_cols = [f"factor_{i + 1}" for i in range(n_factors)]

    # Factor trajectories: (T, k)
    scores = pca.transform(X_std)   # shape (T, n_factors)
    traj_data = {"year": years}
    for i, col in enumerate(factor_cols):
        traj_data[col] = scores[:, i].tolist()
    trajectories_df = pl.DataFrame(traj_data)

    # Factor loadings: (W, k)   [pca.components_ is (k, W)]
    loadings_arr = pca.components_.T    # shape (W, k)
    load_data = {"word": words}
    for i, col in enumerate(factor_cols):
        load_data[col] = loadings_arr[:, i].tolist()
    loadings_df = pl.DataFrame(load_data)

    # Metadata
    evr  = pca.explained_variance_ratio_.tolist()
    cumv = np.cumsum(pca.explained_variance_ratio_).tolist()
    metadata = {
        "n_factors":                n_factors,
        "signal":                   SIGNAL_COL,
        "n_words":                  len(words),
        "n_years":                  len(years),
        "explained_variance_ratio": evr,
        "cumulative_variance":      cumv,
        "total_variance_explained": float(cumv[-1]),
        "year_min":                 min(years),
        "year_max":                 max(years),
        "threshold_pct":            round(float(cumv[-1]) * 100, 2),
    }

    return trajectories_df, loadings_df, metadata


def write_outputs(
    trajectories_df: pl.DataFrame,
    loadings_df: pl.DataFrame,
    metadata: dict,
) -> None:
    """Write all three Phase 7 outputs to disk."""
    trajectories_df.write_parquet(TRAJECTORIES_PATH)
    loadings_df.write_parquet(LOADINGS_PATH)
    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"  {TRAJECTORIES_PATH}  "
          f"({os.path.getsize(TRAJECTORIES_PATH) / 1e6:.1f} MB)")
    print(f"  {LOADINGS_PATH}  "
          f"({os.path.getsize(LOADINGS_PATH) / 1e6:.1f} MB)")
    print(f"  {METADATA_PATH}")


def make_sanity_plot(
    trajectories_df: pl.DataFrame,
    loadings_df: pl.DataFrame,
    n_factors: int,
    out_path: str,
) -> None:
    """Three-panel sanity plot.

    Top panel    — Factor 1 and Factor 2 trajectories 1800–2008.
    Bottom-left  — Top-10 positive loading words for Factor 1 (blue).
    Bottom-right — Top-10 negative loading words for Factor 1 (red).
    """
    years = trajectories_df["year"].to_numpy()
    f1    = trajectories_df["factor_1"].to_numpy()
    f2    = trajectories_df["factor_2"].to_numpy() if n_factors >= 2 else None

    top_pos = (
        loadings_df.sort("factor_1", descending=True).head(TOP_LOADING_N)
        .select(["word", "factor_1"])
    )
    top_neg = (
        loadings_df.sort("factor_1", descending=False).head(TOP_LOADING_N)
        .select(["word", "factor_1"])
    )

    fig = plt.figure(figsize=(14, 10))
    gs  = fig.add_gridspec(2, 2, height_ratios=[1.4, 1], hspace=0.40, wspace=0.35)

    # ── Top: trajectories ──────────────────────────────────────
    ax_top = fig.add_subplot(gs[0, :])
    ax_top.plot(years, f1, lw=2.0, color="#1565C0", label="Factor 1")
    if f2 is not None:
        ax_top.plot(years, f2, lw=1.5, color="#E65100", ls="--", label="Factor 2")
    ax_top.axhline(0, color="grey", lw=0.5, ls=":")
    for yr, lbl in [(1914, "WWI"), (1939, "WWII"), (1969, "1969")]:
        ax_top.axvline(yr, color="grey", lw=0.8, ls="--", alpha=0.5)
        y_pos = ax_top.get_ylim()[1] * 0.85
        ax_top.text(yr + 1, y_pos, lbl, fontsize=7, color="#555555")
    ax_top.set_title("Factor Trajectories 1800–2008", fontsize=11)
    ax_top.set_ylabel("Factor Score")
    ax_top.set_xlabel("Year")
    ax_top.legend(fontsize=9)
    ax_top.tick_params(axis="x", rotation=20)

    # ── Bottom-left: top positive loadings ────────────────────
    ax_pos = fig.add_subplot(gs[1, 0])
    pos_words    = top_pos["word"].to_list()[::-1]
    pos_loadings = top_pos["factor_1"].to_list()[::-1]
    ax_pos.barh(pos_words, pos_loadings, color="#1565C0", alpha=0.75)
    ax_pos.set_title("Top-10 Positive Loadings\n(Factor 1)", fontsize=10)
    ax_pos.set_xlabel("Loading")
    ax_pos.axvline(0, color="black", lw=0.5)

    # ── Bottom-right: top negative loadings ───────────────────
    ax_neg = fig.add_subplot(gs[1, 1])
    neg_words    = top_neg["word"].to_list()
    neg_loadings = top_neg["factor_1"].to_list()
    ax_neg.barh(neg_words, neg_loadings, color="#C62828", alpha=0.75)
    ax_neg.set_title("Top-10 Negative Loadings\n(Factor 1)", fontsize=10)
    ax_neg.set_xlabel("Loading")
    ax_neg.axvline(0, color="black", lw=0.5)

    fig.suptitle("Phase 7 — Latent Factor Model Sanity Check", fontsize=13)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Sanity plot saved: {out_path}")


# ── Main ───────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 7: System-wide latent factor model (PCA on latent_drift)"
    )
    parser.add_argument(
        "--n-factors", type=int, default=DEFAULT_N_FACTORS,
        help=f"Number of PCA factors to retain (default: {DEFAULT_N_FACTORS})"
    )
    parser.add_argument(
        "--plot", action="store_true",
        help="Save sanity plot to outputs/phase7_sanity_check.png"
    )
    args = parser.parse_args()

    t0 = time.time()
    print("=" * 60)
    print("Phase 7 — System-Wide Latent Factor Model")
    print(f"  signal   : {SIGNAL_COL}")
    print(f"  n_factors: {args.n_factors}")
    print("=" * 60)

    # ── [1/5] Load and pivot ──────────────────────────────────
    print(f"\n[1/5] Loading {WORD_FITS_PATH} and pivoting to T×W matrix …")
    t1 = time.time()
    drift_matrix, years, words = load_drift_matrix(WORD_FITS_PATH)
    print(f"  Shape: ({len(years)} years × {len(words):,} words)  "
          f"[{time.time() - t1:.1f}s]")
    print(f"  Memory: {drift_matrix.nbytes / 1e6:.0f} MB")

    # ── [2/5] Standardize ─────────────────────────────────────
    print("\n[2/5] Per-word standardization …")
    X_std, means, stds, words = standardize_matrix(drift_matrix, words)
    print(f"  Words retained: {len(words):,}")
    print(f"  Post-std mean : {X_std.mean():.2e}  (should be ≈ 0)")
    print(f"  Post-std std  : {X_std.std():.4f}   (should be ≈ 1 per word)")

    # ── [3/5] PCA ─────────────────────────────────────────────
    print(f"\n[3/5] Fitting PCA (n_factors={args.n_factors}, svd_solver=randomized) …")
    t3 = time.time()
    pca = fit_pca(X_std, args.n_factors)
    cumvar = float(np.cumsum(pca.explained_variance_ratio_)[-1])
    print(f"  Factor 1 explains         : {pca.explained_variance_ratio_[0]:.3%}")
    print(f"  Factors 1–{args.n_factors} explain  : {cumvar:.3%} of total variance")
    print(f"  PCA fit time: {time.time() - t3:.1f}s")

    # ── [4/5] Extract outputs ─────────────────────────────────
    print("\n[4/5] Extracting trajectories, loadings, and metadata …")
    trajectories_df, loadings_df, metadata = extract_outputs(
        pca, X_std, years, words, args.n_factors
    )
    print(f"  Trajectories shape: {trajectories_df.shape}")
    print(f"  Loadings shape    : {loadings_df.shape}")

    # ── [5/5] Write outputs ───────────────────────────────────
    print("\n[5/5] Writing outputs …")
    write_outputs(trajectories_df, loadings_df, metadata)

    elapsed = time.time() - t0
    print(f"\nPhase 7 complete.  Total time: {elapsed:.1f}s")

    # ── Optional plot ─────────────────────────────────────────
    if args.plot:
        print("\nGenerating sanity plot …")
        make_sanity_plot(trajectories_df, loadings_df, args.n_factors, PLOT_PATH)


if __name__ == "__main__":
    main()
