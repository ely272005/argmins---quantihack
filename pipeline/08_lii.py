"""Phase 8 — Language Instability Index

Constructs the project's macro-level headline quantity: LII_t = tr(Σ_t),
the total variance of factor innovations at each year.

Mathematical derivation
-----------------------
From the roadmap factor model:
    f_{t+1} = A * f_t + ξ_t,   ξ_t ~ N(0, Σ_t)

We estimate innovations ξ_{k,t} for each factor k by fitting an AR(1):
    f_{k,t+1} = a_k * f_{k,t} + ξ_{k,t}

Then for each year t, Σ_t is estimated as the rolling covariance matrix
of the k-dimensional innovation vector over a window of W=20 years.

LII_t = tr(Σ_t)                   — total latent system instability
CR_t  = λ_max(Σ_t) / tr(Σ_t)     — concentration ratio (0=broad, 1=narrow)
B_t   = 1 - CR_t                  — broad-vs-narrow score

Fallback LII (word-level, no factors required):
    LII_t^fallback = mean_w( I_{w,t}^local )
where I_{w,t}^local is the rolling innovation std from Phase 4.

Outputs:
    data/processed/language_instability_index.parquet

Usage:
    python pipeline/08_lii.py
    python pipeline/08_lii.py --window 30   # wider rolling window
    python pipeline/08_lii.py --plot
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

# ── Configuration ─────────────────────────────────────────────
PROCESSED_DIR   = "data/processed"
OUTPUTS_DIR     = "outputs"

TRAJECTORIES_PATH = os.path.join(PROCESSED_DIR, "factor_trajectories.parquet")
METADATA_PATH     = os.path.join(PROCESSED_DIR, "factor_metadata.json")
WORD_FITS_PATH    = os.path.join(PROCESSED_DIR, "word_level_fits.parquet")
LII_PATH          = os.path.join(PROCESSED_DIR, "language_instability_index.parquet")
PLOT_PATH         = os.path.join(OUTPUTS_DIR,   "phase8_lii.png")

YEAR_MIN        = 1800
YEAR_MAX        = 2008
ALL_YEARS       = list(range(YEAR_MIN, YEAR_MAX + 1))  # 209 years
DEFAULT_WINDOW  = 20    # rolling window for Σ_t estimation
MIN_PERIODS     = 5     # minimum observations before LII is defined

os.makedirs(OUTPUTS_DIR, exist_ok=True)


# ── Core functions ─────────────────────────────────────────────

def load_factor_trajectories(path: str, meta_path: str) -> tuple:
    """Load factor trajectories and metadata.

    Returns:
        F     — np.ndarray (T, k), factor scores ordered by year
        years — list[int], length T
        k     — int, number of factors
    """
    with open(meta_path) as f:
        meta = json.load(f)
    k = meta["n_factors"]

    traj = pl.read_parquet(path).sort("year")
    years = traj["year"].to_list()
    factor_cols = [f"factor_{i + 1}" for i in range(k)]
    F = traj.select(factor_cols).to_numpy().astype(np.float64)  # (T, k)

    return F, years, k


def fit_ar1_residuals(F: np.ndarray) -> tuple:
    """Fit AR(1) to each factor column and return innovations.

    For each factor k:
        f_{k,t+1} = a_k * f_{k,t} + ξ_{k,t}

    OLS estimate: a_k = (x' y) / (x' x)

    Returns:
        residuals  — np.ndarray (T-1, k), ξ_{k,t} for t = 1801…2008
        ar1_coeffs — np.ndarray (k,), one coefficient per factor
    """
    T, k = F.shape
    residuals  = np.zeros((T - 1, k), dtype=np.float64)
    ar1_coeffs = np.zeros(k, dtype=np.float64)

    for j in range(k):
        y = F[1:, j]     # shape (T-1,)
        x = F[:-1, j]    # shape (T-1,)
        xx = float(np.dot(x, x))
        if xx < 1e-12:   # degenerate factor — constant, no dynamics
            residuals[:, j] = y
            ar1_coeffs[j]   = 0.0
        else:
            a_k = float(np.dot(x, y) / xx)
            ar1_coeffs[j]   = a_k
            residuals[:, j] = y - a_k * x

    return residuals, ar1_coeffs


def compute_rolling_lii(
    residuals: np.ndarray,
    resid_years: list,
    window: int,
    min_periods: int,
) -> list:
    """Compute rolling LII metrics for each year with innovations.

    For year t (residual index i), uses the window residuals[i-W+1:i+1].

    Returns:
        list of dicts with keys:
            year, lii_value, concentration_ratio, top_eigenvalue,
            broad_vs_narrow_score
        (None values where window is too small)
    """
    T, k = residuals.shape
    records = []

    for i in range(T):
        year  = resid_years[i]
        start = max(0, i - window + 1)
        win   = residuals[start : i + 1]   # shape (≤W, k)

        if len(win) < min_periods:
            records.append({
                "year":                 year,
                "lii_value":            None,
                "concentration_ratio":  None,
                "top_eigenvalue":       None,
                "broad_vs_narrow_score": None,
            })
            continue

        # Rolling covariance matrix (k, k)
        cov = np.cov(win.T, ddof=1) if k > 1 else np.array([[float(np.var(win[:, 0], ddof=1))]])

        trace = float(np.trace(cov))

        if trace <= 1e-20:
            records.append({
                "year":                 year,
                "lii_value":            0.0,
                "concentration_ratio":  None,
                "top_eigenvalue":       None,
                "broad_vs_narrow_score": None,
            })
            continue

        # Eigenvalues (sorted ascending by eigvalsh — take max)
        eigvals = np.linalg.eigvalsh(cov)
        lam_max = float(eigvals.max())
        cr      = lam_max / trace

        records.append({
            "year":                 year,
            "lii_value":            trace,
            "concentration_ratio":  cr,
            "top_eigenvalue":       lam_max,
            "broad_vs_narrow_score": 1.0 - cr,
        })

    return records


def compute_fallback_lii(word_fits_path: str) -> pl.DataFrame:
    """Per-year mean of local_instability across all words (fallback index).

    This is independent of the factor model and serves as a validation
    check: it should correlate positively with the factor-based LII.
    """
    return (
        pl.read_parquet(word_fits_path, columns=["year", "local_instability"])
        .group_by("year")
        .agg(pl.col("local_instability").mean().alias("lii_fallback"))
        .sort("year")
    )


def build_output(
    lii_records: list,
    first_year: int,
    fallback_df: pl.DataFrame,
) -> pl.DataFrame:
    """Merge rolling LII records with fallback LII into a full 209-row frame.

    Year 1800 (no AR(1) residual) is added as a NaN row for the main index.
    """
    # Main LII table (years 1801-2008 from rolling window)
    main_df = pl.DataFrame(lii_records).with_columns(
        pl.col("year").cast(pl.Int32)
    )

    # Add year 1800 with all-null main metrics (cast floats explicitly)
    row_1800 = pl.DataFrame({
        "year":                  [first_year],
        "lii_value":             [None],
        "concentration_ratio":   [None],
        "top_eigenvalue":        [None],
        "broad_vs_narrow_score": [None],
    }).cast({
        "year":                  pl.Int32,
        "lii_value":             pl.Float64,
        "concentration_ratio":   pl.Float64,
        "top_eigenvalue":        pl.Float64,
        "broad_vs_narrow_score": pl.Float64,
    })

    main_df = pl.concat([row_1800, main_df]).sort("year")

    # Join fallback LII
    fallback_df = fallback_df.with_columns(pl.col("year").cast(pl.Int32))
    result = main_df.join(fallback_df, on="year", how="left")

    return result


def make_sanity_plot(lii_df: pl.DataFrame, out_path: str) -> None:
    """Two-panel plot: LII timeline and concentration ratio timeline."""
    df = lii_df.sort("year").filter(pl.col("lii_value").is_not_null())
    years = df["year"].to_numpy()
    lii   = df["lii_value"].to_numpy()
    cr    = df["concentration_ratio"].to_numpy()
    fb    = df["lii_fallback"].to_numpy()

    fig, axes = plt.subplots(3, 1, figsize=(13, 11), sharex=True)
    fig.suptitle("Phase 8 — Language Instability Index (1800–2008)", fontsize=13)

    # ── Panel 1: LII ──────────────────────────────────────────
    ax = axes[0]
    ax.plot(years, lii, color="#1565C0", lw=1.8, label="LII (factor-based)")
    ax.set_ylabel("LII  [tr(Σ_t)]")
    ax.set_title("Language Instability Index")
    for yr, lbl in [(1914, "WWI"), (1939, "WWII"), (1848, "1848"), (1918, "end\nWWI")]:
        ax.axvline(yr, color="grey", lw=0.8, ls="--", alpha=0.5)
        ax.text(yr + 0.5, ax.get_ylim()[1] * 0.88, lbl, fontsize=7, color="#555")
    ax.legend(fontsize=9)

    # ── Panel 2: Concentration Ratio ──────────────────────────
    ax2 = axes[1]
    ax2.plot(years, cr, color="#E65100", lw=1.5)
    ax2.axhline(1.0, color="grey", lw=0.5, ls=":")
    ax2.set_ylabel("Concentration Ratio\n[λ_max / tr(Σ_t)]")
    ax2.set_title("Instability Concentration (1 = one dominant factor)")
    ax2.set_ylim([0, 1.05])

    # ── Panel 3: Fallback LII ─────────────────────────────────
    ax3 = axes[2]
    ax3.plot(years, fb, color="#2E7D32", lw=1.5, label="LII fallback (word-level mean)")
    ax3.set_ylabel("Mean local instability")
    ax3.set_xlabel("Year")
    ax3.set_title("Fallback LII (mean word-level rolling innovation std)")
    ax3.legend(fontsize=9)

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Sanity plot saved: {out_path}")


# ── Main ───────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 8: Language Instability Index"
    )
    parser.add_argument(
        "--window", type=int, default=DEFAULT_WINDOW,
        help=f"Rolling window for Σ_t estimation (default: {DEFAULT_WINDOW})"
    )
    parser.add_argument(
        "--plot", action="store_true",
        help="Save LII plot to outputs/phase8_lii.png"
    )
    args = parser.parse_args()

    t0 = time.time()
    print("=" * 60)
    print("Phase 8 — Language Instability Index")
    print(f"  rolling window : {args.window} years")
    print(f"  min_periods    : {MIN_PERIODS} years")
    print("=" * 60)

    # ── [1/5] Load factor trajectories ───────────────────────
    print(f"\n[1/5] Loading factor trajectories …")
    F, years, k = load_factor_trajectories(TRAJECTORIES_PATH, METADATA_PATH)
    print(f"  Shape: ({len(years)} years × {k} factors)")

    # ── [2/5] Fit AR(1), extract innovations ─────────────────
    print(f"\n[2/5] Fitting AR(1) to each factor, extracting innovations …")
    residuals, ar1_coeffs = fit_ar1_residuals(F)
    resid_years = years[1:]   # 1801-2008 (208 years)
    print(f"  Innovations shape: {residuals.shape}")
    print(f"  AR(1) coefficients: min={ar1_coeffs.min():.4f}, "
          f"max={ar1_coeffs.max():.4f}, "
          f"mean={ar1_coeffs.mean():.4f}")
    print(f"  Innovation residual std: "
          f"{[round(residuals[:, j].std(), 3) for j in range(min(k, 5))]} …")

    # ── [3/5] Rolling LII metrics ─────────────────────────────
    print(f"\n[3/5] Computing rolling LII (window={args.window}) …")
    lii_records = compute_rolling_lii(
        residuals, resid_years, args.window, MIN_PERIODS
    )
    n_valid = sum(1 for r in lii_records if r["lii_value"] is not None)
    print(f"  Valid LII values: {n_valid} / {len(lii_records)} years")

    # ── [4/5] Fallback LII ────────────────────────────────────
    print(f"\n[4/5] Computing fallback LII from word-level instability …")
    fallback_df = compute_fallback_lii(WORD_FITS_PATH)
    n_fb = fallback_df.filter(pl.col("lii_fallback").is_not_null()).shape[0]
    print(f"  Fallback LII available for {n_fb} / {len(fallback_df)} years")

    # ── [5/5] Combine and write ───────────────────────────────
    print(f"\n[5/5] Building output and writing …")
    result = build_output(lii_records, years[0], fallback_df)

    # Summary stats (non-null years)
    valid = result.filter(pl.col("lii_value").is_not_null())
    lii_arr = valid["lii_value"].to_numpy()
    cr_arr  = valid["concentration_ratio"].to_numpy()
    peak_yr = int(valid.sort("lii_value", descending=True)["year"][0])
    print(f"  Rows: {len(result)} (total), {len(valid)} (with valid LII)")
    print(f"  LII range   : [{lii_arr.min():.3f}, {lii_arr.max():.3f}]")
    print(f"  Peak year   : {peak_yr}")
    print(f"  CR range    : [{cr_arr.min():.3f}, {cr_arr.max():.3f}]")

    result.write_parquet(LII_PATH)
    sz = os.path.getsize(LII_PATH)
    print(f"  Written: {LII_PATH}  ({sz / 1024:.1f} KB)")

    elapsed = time.time() - t0
    print(f"\nPhase 8 complete.  Total time: {elapsed:.1f}s")

    # ── Optional plot ─────────────────────────────────────────
    if args.plot:
        print("\nGenerating LII plot …")
        make_sanity_plot(result, PLOT_PATH)


if __name__ == "__main__":
    main()
