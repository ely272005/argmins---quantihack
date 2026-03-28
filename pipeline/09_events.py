"""Phase 9 — Event Annotation and Historical Alignment

Cross-references the pipeline's model outputs against a curated catalog of 12
historical events (1800–2008) to evaluate whether the Language Instability Index,
changepoint density, and factor trajectories align with documented disruptions.

This phase is ALIGNMENT and EVALUATION, not prediction.

Mathematical approach
---------------------
For each event e with window [s_e, e_e] and pre-event baseline [b_s, b_e]:

    lii_elevation_ratio_e = mean(LII_t, t in [s_e, e_e]) / mean(LII_t, t in [b_s, b_e])
    cp_elevation_ratio_e  = mean(D_t,   t in [s_e, e_e]) / mean(D_t,   t in [b_s, b_e])

where D_t = normalized_density from changepoint_density.parquet.

Baseline window:
    [s_e - pre_window, s_e - 1]  if s_e >= 1825  (local 20-year pre-event window)
    [1805, 1824]                  otherwise        (fallback: first 20 valid LII years)

Alignment flag:
    aligned_e = (lii_elevation_ratio_e > 1.0) OR (cp_elevation_ratio_e > 1.0)

Dominant factor:
    argmax_k std(factor_k_t, t in [s_e, e_e])

Top words:
    Group regimes.parquet by word within event window,
    compute mean(z_drift), take top/bottom n_top words.

Vocabulary note: only words starting with c, t, w are in the vocabulary
(shards c, t, w were downloaded). Top words will reflect this constraint.

Outputs:
    data/processed/historical_events.json
    data/processed/event_alignment.json

Usage:
    python pipeline/09_events.py
    python pipeline/09_events.py --n-top 15 --pre-window 15
    python pipeline/09_events.py --plot
"""

import argparse
import json
import math
import os
import time
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

# ── Configuration ─────────────────────────────────────────────
PROCESSED_DIR = "data/processed"
OUTPUTS_DIR   = "outputs"

LII_PATH       = os.path.join(PROCESSED_DIR, "language_instability_index.parquet")
DENSITY_PATH   = os.path.join(PROCESSED_DIR, "changepoint_density.parquet")
REGIMES_PATH   = os.path.join(PROCESSED_DIR, "regimes.parquet")
TRAJ_PATH      = os.path.join(PROCESSED_DIR, "factor_trajectories.parquet")

EVENTS_JSON    = os.path.join(PROCESSED_DIR, "historical_events.json")
ALIGNMENT_JSON = os.path.join(PROCESSED_DIR, "event_alignment.json")
PLOT_PATH      = os.path.join(OUTPUTS_DIR,   "phase9_event_alignment.png")

YEAR_MIN = 1800
YEAR_MAX = 2008
N_FACTORS = 10

VALID_CATEGORIES = {"conflict", "health", "political", "economic", "technology"}

# Baseline fallback for events with insufficient pre-event LII data
# (LII first valid year is 1805; rolling window MIN_PERIODS = 5)
EARLY_FALLBACK_BASELINE_START = 1805
EARLY_FALLBACK_BASELINE_END   = 1824
EARLY_THRESHOLD               = 1825   # start_year < this → use fallback baseline

DEFAULT_N_TOP      = 10
DEFAULT_PRE_WINDOW = 20

# ── Event catalog ──────────────────────────────────────────────
EVENTS = [
    {"name": "Napoleonic Wars",      "start_year": 1803, "end_year": 1815, "category": "conflict"},
    {"name": "Cholera Epidemics",    "start_year": 1817, "end_year": 1866, "category": "health"},
    {"name": "Revolutions of 1848",  "start_year": 1848, "end_year": 1851, "category": "political"},
    {"name": "US Civil War",         "start_year": 1861, "end_year": 1865, "category": "conflict"},
    {"name": "Great Influenza 1889", "start_year": 1889, "end_year": 1895, "category": "health"},
    {"name": "WWI",                  "start_year": 1914, "end_year": 1918, "category": "conflict"},
    {"name": "Spanish Flu",          "start_year": 1918, "end_year": 1920, "category": "health"},
    {"name": "Great Depression",     "start_year": 1929, "end_year": 1939, "category": "economic"},
    {"name": "WWII",                 "start_year": 1939, "end_year": 1945, "category": "conflict"},
    {"name": "Cold War",             "start_year": 1947, "end_year": 1991, "category": "political"},
    {"name": "Vietnam War",          "start_year": 1955, "end_year": 1975, "category": "conflict"},
    {"name": "Computing Revolution", "start_year": 1970, "end_year": 2008, "category": "technology"},
]

CATEGORY_COLORS = {
    "conflict":   "#E53935",
    "health":     "#43A047",
    "political":  "#1E88E5",
    "economic":   "#FB8C00",
    "technology": "#8E24AA",
}

os.makedirs(OUTPUTS_DIR, exist_ok=True)


# ── Core functions ─────────────────────────────────────────────

def build_events_catalog(path: str) -> list:
    """Write the EVENTS catalog to path as JSON and return it.

    Returns:
        The EVENTS list (12 dicts), each with keys:
            name, start_year, end_year, category
    """
    with open(path, "w") as f:
        json.dump(EVENTS, f, indent=2)
    return EVENTS


def compute_event_alignment(
    event: dict,
    lii_df: pl.DataFrame,
    density_df: pl.DataFrame,
    regimes_df: pl.DataFrame,
    traj_df: pl.DataFrame,
    n_top: int = DEFAULT_N_TOP,
    pre_window: int = DEFAULT_PRE_WINDOW,
) -> dict:
    """Compute all alignment metrics for a single historical event.

    Args:
        event:       dict with keys name, start_year, end_year, category
        lii_df:      DataFrame with columns year, lii_value (nullable Float64)
        density_df:  DataFrame with columns year, normalized_density (Float64)
        regimes_df:  DataFrame with columns word, year, z_drift (Float64)
        traj_df:     DataFrame with columns year, factor_1...factor_{N_FACTORS}
        n_top:       number of top/bottom words to return per event
        pre_window:  years before event start to use as local baseline

    Returns:
        dict with 16 keys suitable for JSON serialization (all Python native types)
    """
    s = event["start_year"]
    e = event["end_year"]

    # ── Baseline window ────────────────────────────────────────
    if s < EARLY_THRESHOLD:
        b_s = EARLY_FALLBACK_BASELINE_START   # 1805
        b_e = EARLY_FALLBACK_BASELINE_END     # 1824
    else:
        b_s = s - pre_window
        b_e = s - 1

    # ── LII metrics (null-aware) ───────────────────────────────
    lii_during_vals = (
        lii_df
        .filter((pl.col("year") >= s) & (pl.col("year") <= e))
        .filter(pl.col("lii_value").is_not_null())
        ["lii_value"].to_list()
    )
    lii_base_vals = (
        lii_df
        .filter((pl.col("year") >= b_s) & (pl.col("year") <= b_e))
        .filter(pl.col("lii_value").is_not_null())
        ["lii_value"].to_list()
    )

    lii_mean_during   = float(np.mean(lii_during_vals)) if lii_during_vals   else None
    lii_mean_baseline = float(np.mean(lii_base_vals))   if lii_base_vals     else None

    if (lii_mean_during is not None
            and lii_mean_baseline is not None
            and lii_mean_baseline > 0):
        lii_elevation_ratio = float(lii_mean_during / lii_mean_baseline)
    else:
        lii_elevation_ratio = None

    # ── CP density metrics (no nulls in normalized_density) ───
    cp_during_vals = (
        density_df
        .filter((pl.col("year") >= s) & (pl.col("year") <= e))
        ["normalized_density"].to_list()
    )
    cp_base_vals = (
        density_df
        .filter((pl.col("year") >= b_s) & (pl.col("year") <= b_e))
        ["normalized_density"].to_list()
    )

    cp_mean_during   = float(np.mean(cp_during_vals)) if cp_during_vals else 0.0
    cp_mean_baseline = float(np.mean(cp_base_vals))   if cp_base_vals   else 0.0

    if cp_mean_baseline > 0:
        cp_elevation_ratio = float(cp_mean_during / cp_mean_baseline)
    else:
        cp_elevation_ratio = None

    # ── Top adoption / decline words ───────────────────────────
    event_reg = regimes_df.filter(
        (pl.col("year") >= s) & (pl.col("year") <= e)
    )
    if len(event_reg) > 0:
        word_agg = (
            event_reg
            .group_by("word")
            .agg(pl.col("z_drift").mean().alias("mean_z_drift"))
        )
        top_adoption_words = (
            word_agg.sort("mean_z_drift", descending=True)
            .head(n_top)["word"].to_list()
        )
        top_decline_words = (
            word_agg.sort("mean_z_drift", descending=False)
            .head(n_top)["word"].to_list()
        )
    else:
        top_adoption_words = []
        top_decline_words  = []

    # ── Dominant factor ────────────────────────────────────────
    factor_cols = [f"factor_{i + 1}" for i in range(N_FACTORS)]
    event_traj = traj_df.filter(
        (pl.col("year") >= s) & (pl.col("year") <= e)
    )

    if len(event_traj) >= 2:
        stds = {col: float(event_traj[col].std(ddof=1)) for col in factor_cols}
        dominant_factor = int(max(stds, key=stds.get).split("_")[1])
    else:
        dominant_factor = 1   # single-year event fallback

    # ── Alignment flags ────────────────────────────────────────
    is_lii_elevated = bool(
        lii_elevation_ratio is not None and lii_elevation_ratio > 1.0
    )
    is_cp_elevated = bool(
        cp_elevation_ratio is not None and cp_elevation_ratio > 1.0
    )
    aligned = is_lii_elevated or is_cp_elevated

    return {
        "name":                      event["name"],
        "start_year":                int(s),
        "end_year":                  int(e),
        "category":                  event["category"],
        "lii_mean_during":           lii_mean_during,
        "lii_mean_baseline":         lii_mean_baseline,
        "lii_elevation_ratio":       lii_elevation_ratio,
        "cp_density_mean_during":    cp_mean_during,
        "cp_density_mean_baseline":  cp_mean_baseline,
        "cp_elevation_ratio":        cp_elevation_ratio,
        "top_adoption_words":        top_adoption_words,
        "top_decline_words":         top_decline_words,
        "dominant_factor":           dominant_factor,
        "is_lii_elevated":           is_lii_elevated,
        "is_cp_elevated":            is_cp_elevated,
        "aligned":                   aligned,
    }


def build_summary_stats(alignment_records: list) -> dict:
    """Aggregate alignment metrics per category (stdout only, not written to disk).

    Args:
        alignment_records: list of dicts from compute_event_alignment

    Returns:
        dict with keys per_category (list) and overall (dict)
    """
    by_category = defaultdict(list)
    for rec in alignment_records:
        by_category[rec["category"]].append(rec)

    per_category = []
    for cat, recs in sorted(by_category.items()):
        n_events  = len(recs)
        n_aligned = sum(1 for r in recs if r["aligned"])
        lii_ratios = [r["lii_elevation_ratio"] for r in recs if r["lii_elevation_ratio"] is not None]
        cp_ratios  = [r["cp_elevation_ratio"]  for r in recs if r["cp_elevation_ratio"]  is not None]
        per_category.append({
            "category":                cat,
            "n_events":                n_events,
            "n_aligned":               n_aligned,
            "frac_aligned":            round(n_aligned / n_events, 3),
            "mean_lii_elevation_ratio": round(float(np.mean(lii_ratios)), 3) if lii_ratios else None,
            "mean_cp_elevation_ratio":  round(float(np.mean(cp_ratios)),  3) if cp_ratios  else None,
        })

    n_total   = len(alignment_records)
    n_aligned = sum(1 for r in alignment_records if r["aligned"])

    return {
        "per_category": per_category,
        "overall": {
            "n_events":    n_total,
            "n_aligned":   n_aligned,
            "frac_aligned": round(n_aligned / n_total, 3) if n_total else 0.0,
        },
    }


def make_sanity_plot(
    lii_df: pl.DataFrame,
    density_df: pl.DataFrame,
    events: list,
    alignment: list,
    out_path: str,
) -> None:
    """Two-panel plot: LII timeline + CP density timeline with event bands.

    Aligned events use their category color (alpha=0.12).
    Non-aligned events use grey (#9E9E9E, alpha=0.12).
    """
    aligned_by_name = {r["name"]: r["aligned"] for r in alignment}

    # ── Prepare series ─────────────────────────────────────────
    lii_valid = lii_df.sort("year").filter(pl.col("lii_value").is_not_null())
    lii_years = lii_valid["year"].to_numpy()
    lii_vals  = lii_valid["lii_value"].to_numpy()

    den = density_df.sort("year")
    den_years = den["year"].to_numpy()
    den_vals  = den["normalized_density"].to_numpy()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
    fig.suptitle("Phase 9 — Historical Event Alignment (1800–2008)", fontsize=13)

    # ── Draw event bands on both panels ───────────────────────
    for ev in events:
        aligned = aligned_by_name.get(ev["name"], False)
        color   = CATEGORY_COLORS[ev["category"]] if aligned else "#9E9E9E"
        alpha   = 0.12
        for ax in (ax1, ax2):
            ax.axvspan(ev["start_year"], ev["end_year"], color=color, alpha=alpha)

    # ── Panel 1: LII ──────────────────────────────────────────
    ax1.plot(lii_years, lii_vals, color="#1565C0", lw=1.6, label="LII (factor-based)")
    ax1.set_ylabel("LII  [tr(Σ_t)]")
    ax1.set_title("Language Instability Index with historical event bands")
    ax1.legend(fontsize=9, loc="upper left")

    # ── Panel 2: CP density ───────────────────────────────────
    ax2.plot(den_years, den_vals, color="#B71C1C", lw=1.4,
             label="Changepoint density (normalized)")
    ax2.set_ylabel("Normalized density")
    ax2.set_xlabel("Year")
    ax2.set_title("Changepoint Density with historical event bands")
    ax2.legend(fontsize=9, loc="upper left")

    # ── Category legend ───────────────────────────────────────
    import matplotlib.patches as mpatches
    patches = [
        mpatches.Patch(color=c, label=cat.capitalize(), alpha=0.5)
        for cat, c in CATEGORY_COLORS.items()
    ] + [mpatches.Patch(color="#9E9E9E", label="Non-aligned", alpha=0.5)]
    fig.legend(handles=patches, loc="lower center", ncol=6, fontsize=8,
               bbox_to_anchor=(0.5, -0.01))

    fig.tight_layout(rect=[0, 0.04, 1, 0.97])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Sanity plot saved: {out_path}")


# ── Main ───────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 9: Event Annotation and Historical Alignment"
    )
    parser.add_argument(
        "--n-top", type=int, default=DEFAULT_N_TOP,
        help=f"Top/bottom words per event (default: {DEFAULT_N_TOP})"
    )
    parser.add_argument(
        "--pre-window", type=int, default=DEFAULT_PRE_WINDOW,
        help=f"Pre-event baseline window in years (default: {DEFAULT_PRE_WINDOW})"
    )
    parser.add_argument(
        "--plot", action="store_true",
        help="Save sanity plot to outputs/phase9_event_alignment.png"
    )
    args = parser.parse_args()

    t0 = time.time()
    print("=" * 60)
    print("Phase 9 — Event Annotation and Historical Alignment")
    print(f"  n_top      : {args.n_top}")
    print(f"  pre_window : {args.pre_window} years")
    print("=" * 60)

    # ── [1/5] Build events catalog ────────────────────────────
    print(f"\n[1/5] Building events catalog → {EVENTS_JSON} …")
    events = build_events_catalog(EVENTS_JSON)
    print(f"  {len(events)} events written")

    # ── [2/5] Load input data ─────────────────────────────────
    print("\n[2/5] Loading input data …")
    lii_df      = pl.read_parquet(LII_PATH)
    density_df  = pl.read_parquet(DENSITY_PATH)
    # Load only needed columns from regimes (~60% memory reduction)
    regimes_df  = pl.read_parquet(REGIMES_PATH, columns=["word", "year", "z_drift"])
    traj_df     = pl.read_parquet(TRAJ_PATH)
    lii_valid   = lii_df.filter(pl.col("lii_value").is_not_null()).shape[0]
    print(f"  LII         : {lii_df.shape} ({lii_valid} valid rows)")
    print(f"  CP density  : {density_df.shape}")
    print(f"  Regimes     : {regimes_df.shape} [word, year, z_drift only]")
    print(f"  Trajectories: {traj_df.shape}")

    # ── [3/5] Compute alignment metrics ──────────────────────
    print(f"\n[3/5] Computing alignment for {len(events)} events …")
    alignment_records = []
    for ev in events:
        rec = compute_event_alignment(
            ev, lii_df, density_df, regimes_df, traj_df,
            n_top=args.n_top, pre_window=args.pre_window,
        )
        alignment_records.append(rec)
        lii_r = f"{rec['lii_elevation_ratio']:.3f}" if rec["lii_elevation_ratio"] is not None else "None"
        cp_r  = f"{rec['cp_elevation_ratio']:.3f}"  if rec["cp_elevation_ratio"]  is not None else "None"
        print(f"  {ev['name']:<25} ({ev['start_year']}–{ev['end_year']})  "
              f"lii_ratio={lii_r:>6}  cp_ratio={cp_r:>6}  "
              f"aligned={'Yes' if rec['aligned'] else 'No '}")

    # ── [4/5] Summary statistics ──────────────────────────────
    print("\n[4/5] Summary statistics by category …")
    summary = build_summary_stats(alignment_records)
    header = f"  {'category':<12}  {'n':>3}  {'aligned':>7}  {'frac':>5}  {'lii_ratio':>9}  {'cp_ratio':>8}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for row in summary["per_category"]:
        lii_r = f"{row['mean_lii_elevation_ratio']:.3f}" if row["mean_lii_elevation_ratio"] else "  N/A"
        cp_r  = f"{row['mean_cp_elevation_ratio']:.3f}"  if row["mean_cp_elevation_ratio"]  else "  N/A"
        print(f"  {row['category']:<12}  {row['n_events']:>3}  "
              f"{row['n_aligned']:>7}  {row['frac_aligned']:>5.1%}  "
              f"{lii_r:>9}  {cp_r:>8}")
    ov = summary["overall"]
    print(f"\n  Overall: {ov['n_aligned']}/{ov['n_events']} events aligned "
          f"({ov['frac_aligned']:.1%})")

    # ── [5/5] Write alignment JSON ────────────────────────────
    print(f"\n[5/5] Writing {ALIGNMENT_JSON} …")
    with open(ALIGNMENT_JSON, "w") as f:
        json.dump(alignment_records, f, indent=2)
    sz = os.path.getsize(ALIGNMENT_JSON)
    print(f"  Written: {ALIGNMENT_JSON}  ({sz / 1024:.1f} KB)")

    elapsed = time.time() - t0
    print(f"\nPhase 9 complete.  Total time: {elapsed:.1f}s")

    # ── Optional plot ─────────────────────────────────────────
    if args.plot:
        print("\nGenerating sanity plot …")
        make_sanity_plot(lii_df, density_df, events, alignment_records, PLOT_PATH)


if __name__ == "__main__":
    main()
