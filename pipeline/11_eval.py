"""Phase 11 — Evaluation and Sanity Checks

Runs structured sanity checks across all pipeline outputs and saves a
machine-readable audit report to outputs/eval_summary.json.

Six evaluation sections:
  1. Model quality       — convergence rate, AIC distribution, failed fits
  2. Word sanity         — anchor word peak years, changepoints, regimes
  3. Changepoint sanity  — top CP years, near-event rate
  4. Factor sanity       — top-loading words per factor, explained variance
  5. LII sanity          — peak year, range, correlation with CP density
  6. Failure modes       — extreme drift / high instability words

Usage:
    python pipeline/11_eval.py
    python pipeline/11_eval.py --output outputs/eval_summary.json
"""

import argparse
import json
import math
import os
from datetime import datetime, timezone

import numpy as np
import polars as pl

# ── Configuration ─────────────────────────────────────────────
PROCESSED_DIR = "data/processed"
OUTPUTS_DIR   = "outputs"

SUMMARY_PATH  = os.path.join(PROCESSED_DIR, "word_summary_metrics.parquet")
CP_PATH       = os.path.join(PROCESSED_DIR, "changepoints.parquet")
DENSITY_PATH  = os.path.join(PROCESSED_DIR, "changepoint_density.parquet")
REGIMES_PATH  = os.path.join(PROCESSED_DIR, "regimes.parquet")
LOADINGS_PATH = os.path.join(PROCESSED_DIR, "factor_loadings.parquet")
META_PATH     = os.path.join(PROCESSED_DIR, "factor_metadata.json")
LII_PATH      = os.path.join(PROCESSED_DIR, "language_instability_index.parquet")
EVENTS_PATH   = os.path.join(PROCESSED_DIR, "historical_events.json")

ANCHOR_WORDS      = ["war", "computer", "technology", "typewriter", "wireless"]
N_TOP_CP_YEARS    = 10
N_TOP_WORDS       = 10   # top words per factor
N_EXTREME         = 20   # max words in extreme lists
NEAR_EVENT_WINDOW = 5    # years either side of event.start_year


# ── Evaluation functions ───────────────────────────────────────

def eval_model_quality(summary_path: str) -> dict:
    """Convergence rate, AIC stats, list of failed words."""
    df = pl.read_parquet(summary_path)
    n_total     = len(df)
    n_converged = int((df["fit_status"] == "converged").sum())
    n_failed    = n_total - n_converged

    conv_df = df.filter(pl.col("fit_status") == "converged")
    aic_vals = conv_df["aic"].drop_nulls()
    aic_mean   = float(aic_vals.mean())   if len(aic_vals) > 0 else None
    aic_median = float(aic_vals.median()) if len(aic_vals) > 0 else None

    failed_words = (
        df.filter(pl.col("fit_status") != "converged")
        .sort("word")["word"]
        .head(50)
        .to_list()
    )

    print(f"      {n_total:,} words  |  "
          f"{100*n_converged/n_total:.1f}% converged  |  "
          f"{n_failed:,} failed")

    return {
        "n_words_total":    n_total,
        "n_converged":      n_converged,
        "n_failed":         n_failed,
        "convergence_rate": round(n_converged / n_total, 6),
        "aic_mean":         round(aic_mean,   4) if aic_mean   is not None else None,
        "aic_median":       round(aic_median, 4) if aic_median is not None else None,
        "failed_words":     failed_words,
    }


def eval_word_sanity(summary_path: str, cp_path: str, regimes_path: str,
                     anchor_words: list) -> dict:
    """Peak year, changepoint count, dominant regime for each anchor word."""
    summary = pl.read_parquet(
        summary_path, columns=["word", "peak_year", "mean_drift", "fit_status"]
    )
    cp_counts = (
        pl.read_parquet(cp_path, columns=["word", "changepoint_year"])
        .group_by("word")
        .len()
        .rename({"len": "ncp"})
    )
    dom_regime = (
        pl.read_parquet(regimes_path, columns=["word", "regime_label"])
        .group_by(["word", "regime_label"])
        .len()
        .sort("len", descending=True)
        .unique(["word"], keep="first")
        .select(["word", "regime_label"])
    )

    result = {}
    for word in anchor_words:
        row    = summary.filter(pl.col("word") == word)
        cp_row = cp_counts.filter(pl.col("word") == word)
        rg_row = dom_regime.filter(pl.col("word") == word)

        peak_year       = int(row["peak_year"][0])    if len(row) > 0 else None
        ncp             = int(cp_row["ncp"][0])       if len(cp_row) > 0 else 0
        dominant_regime = rg_row["regime_label"][0]   if len(rg_row) > 0 else "unknown"

        result[word] = {
            "peak_year":        peak_year,
            "n_changepoints":   ncp,
            "dominant_regime":  dominant_regime,
        }
        print(f"      {word:<12} peak={peak_year}  ncp={ncp:<3}  "
              f"regime={dominant_regime:<12}  ✓")

    return result


def eval_changepoint_sanity(density_path: str, events_json_path: str) -> dict:
    """Top CP years and near-event rate."""
    density = pl.read_parquet(density_path).sort("n_changepoints", descending=True)
    top_rows     = density.head(N_TOP_CP_YEARS)
    top_cp_years = [int(y) for y in top_rows["year"].to_list()]
    peak_year    = top_cp_years[0] if top_cp_years else None

    with open(events_json_path) as f:
        events = json.load(f)
    event_starts = [int(e["start_year"]) for e in events]

    n_near = sum(
        any(abs(cy - es) <= NEAR_EVENT_WINDOW for es in event_starts)
        for cy in top_cp_years
    )
    near_event_rate = round(n_near / N_TOP_CP_YEARS, 4) if top_cp_years else 0.0

    print(f"      Top CP year: {peak_year}  |  "
          f"Near-event rate: {100*near_event_rate:.1f}%")

    return {
        "top_cp_years":         top_cp_years,
        "cp_density_peak_year": peak_year,
        "near_event_rate":      near_event_rate,
    }


def eval_factor_interpretability(loadings_path: str, metadata_path: str) -> dict:
    """Top-loading words and explained variance for each factor."""
    loadings = pl.read_parquet(loadings_path)
    factor_cols = [c for c in loadings.columns if c.startswith("factor_")]

    with open(metadata_path) as f:
        meta = json.load(f)
    evr = meta.get("explained_variance_ratio", [])

    result = {}
    for i, fc in enumerate(factor_cols):
        top_words = (
            loadings.sort(pl.col(fc).abs(), descending=True)
            .head(N_TOP_WORDS)["word"]
            .to_list()
        )
        ev = round(float(evr[i]), 6) if i < len(evr) else None
        result[fc] = {"top_words": top_words, "explained_variance": ev}

        pct = f"{100*ev:.1f}%" if ev is not None else "?"
        print(f"      {fc} ({pct}): {', '.join(top_words[:5])}, ...")

    return result


def eval_lii_sanity(lii_path: str, density_path: str) -> dict:
    """Peak year, value range, and correlation with CP density."""
    lii_df = (
        pl.read_parquet(lii_path, columns=["year", "lii_value"])
        .drop_nulls("lii_value")
    )
    peak_row  = lii_df.sort("lii_value", descending=True).head(1)
    peak_year  = int(peak_row["year"][0])
    peak_value = float(peak_row["lii_value"][0])
    min_value  = float(lii_df["lii_value"].min())
    max_value  = float(lii_df["lii_value"].max())

    density_df = pl.read_parquet(density_path, columns=["year", "normalized_density"])
    joined = lii_df.join(density_df, on="year", how="inner")

    lii_arr     = joined["lii_value"].to_numpy()
    density_arr = joined["normalized_density"].to_numpy()

    if len(lii_arr) > 2 and np.std(lii_arr) > 0 and np.std(density_arr) > 0:
        corr = float(np.corrcoef(lii_arr, density_arr)[0, 1])
        corr = round(corr, 6) if math.isfinite(corr) else None
    else:
        corr = None

    print(f"      Peak: {peak_year} ({peak_value:.2f})  |  "
          f"CP-LII correlation: {corr:.3f}" if corr is not None
          else f"      Peak: {peak_year} ({peak_value:.2f})  |  "
               f"CP-LII correlation: N/A")

    return {
        "peak_year":          peak_year,
        "peak_value":         round(peak_value, 4),
        "min_value":          round(min_value,  4),
        "max_value":          round(max_value,  4),
        "cp_lii_correlation": corr,
    }


def eval_failures(summary_path: str) -> dict:
    """Failed fits and extreme outlier words."""
    df = pl.read_parquet(
        summary_path,
        columns=["word", "fit_status", "mean_drift", "mean_instability"]
    )
    n_failed = int((df["fit_status"] != "converged").sum())

    conv = df.filter(pl.col("fit_status") == "converged").drop_nulls()

    drift_std = float(conv["mean_drift"].std(ddof=1))
    extreme_drift = (
        conv.filter(pl.col("mean_drift").abs() > 3 * drift_std)
        .with_columns(pl.col("mean_drift").abs().alias("abs_drift"))
        .sort("abs_drift", descending=True)
        .head(N_EXTREME)["word"]
        .to_list()
    )

    instab_std = float(conv["mean_instability"].std(ddof=1))
    instab_mean = float(conv["mean_instability"].mean())
    high_instab = (
        conv.filter(pl.col("mean_instability") > instab_mean + 3 * instab_std)
        .sort("mean_instability", descending=True)
        .head(N_EXTREME)["word"]
        .to_list()
    )

    print(f"      {n_failed:,} failed fits  |  "
          f"{len(extreme_drift)} extreme-drift  |  "
          f"{len(high_instab)} high-instability")

    return {
        "n_failed_fits":         n_failed,
        "extreme_drift_words":   extreme_drift,
        "high_instability_words": high_instab,
    }


# ── Main ───────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Phase 11 — Evaluation")
    parser.add_argument("--output", default=os.path.join(OUTPUTS_DIR, "eval_summary.json"))
    args = parser.parse_args()

    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    sep = "=" * 60
    print(sep)
    print("Phase 11 — Evaluation and Sanity Checks")
    print(sep)

    print("[1/6] Model quality ...")
    model_quality = eval_model_quality(SUMMARY_PATH)

    print("[2/6] Word sanity (5 anchor words) ...")
    word_sanity = eval_word_sanity(SUMMARY_PATH, CP_PATH, REGIMES_PATH, ANCHOR_WORDS)

    print("[3/6] Changepoint sanity ...")
    changepoint_sanity = eval_changepoint_sanity(DENSITY_PATH, EVENTS_PATH)

    print("[4/6] Factor interpretability ...")
    factor_interpretability = eval_factor_interpretability(LOADINGS_PATH, META_PATH)

    print("[5/6] LII sanity ...")
    lii_sanity = eval_lii_sanity(LII_PATH, DENSITY_PATH)

    print("[6/6] Failure modes ...")
    failures = eval_failures(SUMMARY_PATH)

    summary = {
        "generated_at":          datetime.now(timezone.utc).isoformat(),
        "model_quality":          model_quality,
        "word_sanity":            word_sanity,
        "changepoint_sanity":     changepoint_sanity,
        "factor_interpretability": factor_interpretability,
        "lii_sanity":             lii_sanity,
        "failures":               failures,
    }

    with open(args.output, "w") as f:
        json.dump(summary, f, indent=2)

    size_kb = os.path.getsize(args.output) / 1024
    print(sep)
    print(f"Eval complete → {args.output}  ({size_kb:.1f} KB)")


if __name__ == "__main__":
    main()
