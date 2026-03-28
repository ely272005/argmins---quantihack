"""Phase 4 — Word-Level State-Space Model

For each word in the clean panel, fits a local linear trend state-space model
using statsmodels UnobservedComponents + Kalman smoother. Extracts:
  - latent level x_{w,t} and drift v_{w,t} with 95% CI
  - curvature c_{w,t} = v_{w,t} - v_{w,t-1}
  - local instability = rolling std of one-step-ahead innovations

Usage:
    python pipeline/04_word_model.py                  # all 70k words
    python pipeline/04_word_model.py --subset 20      # top 20 by total_count (smoke test)
    python pipeline/04_word_model.py --max-words 1000 # first 1000 alphabetically
"""

import argparse
import logging
import os
import time
import warnings

import numpy as np
import pandas as pd
import polars as pl
from joblib import Parallel, delayed
from statsmodels.tsa.statespace.structural import UnobservedComponents
from tqdm import tqdm

# ── Configuration ────────────────────────────────────────────
PROCESSED_DIR = "data/processed"
OUTPUTS_DIR   = "outputs"

CLEAN_PANEL_PATH  = os.path.join(PROCESSED_DIR, "clean_panel.parquet")
VOCAB_META_PATH   = os.path.join(PROCESSED_DIR, "vocabulary_metadata.parquet")
WORD_FITS_PATH    = os.path.join(PROCESSED_DIR, "word_level_fits.parquet")
WORD_SUMMARY_PATH = os.path.join(PROCESSED_DIR, "word_summary_metrics.parquet")
FAILED_LOG_PATH   = os.path.join(OUTPUTS_DIR,   "phase4_failed_words.log")

YEAR_MIN = 1800
YEAR_MAX = 2019
ALL_YEARS = np.arange(YEAR_MIN, YEAR_MAX + 1, dtype=int)   # 209 years

INSTABILITY_WINDOW      = 20
INSTABILITY_MIN_PERIODS = 5
LOG_FLOOR = 0.5   # Laplace smoothing: log(count + 0.5) - log(N_t)

os.makedirs(OUTPUTS_DIR, exist_ok=True)


# ── Logging setup ────────────────────────────────────────────

def _configure_logging():
    """Write warnings/errors from worker functions to a persistent log file."""
    logger = logging.getLogger("phase4")
    logger.setLevel(logging.DEBUG)
    if not logger.handlers:
        fh = logging.FileHandler(FAILED_LOG_PATH, mode="w")
        fh.setFormatter(logging.Formatter("%(levelname)-5s  %(asctime)s  %(message)s",
                                          datefmt="%H:%M:%S"))
        logger.addHandler(fh)
    return logger

logger = logging.getLogger("phase4")


# ── Core fitting function ─────────────────────────────────────

def fit_word(word: str, word_df: pd.DataFrame):
    """Fit local linear trend state-space model for one word.

    Returns
    -------
    (fits_df, summary_dict)   on success (soft or hard convergence)
    (None, failure_dict)      on hard exception
    """
    T = len(ALL_YEARS)

    # Build full-grid observation arrays (NaN for missing years)
    obs       = np.full(T, np.nan)
    obs_count = np.full(T, np.nan)
    obs_freq  = np.full(T, np.nan)

    for row in word_df.itertuples(index=False):
        idx = int(row.year) - YEAR_MIN
        if 0 <= idx < T:
            obs[idx]       = np.log(row.count + LOG_FLOOR) - np.log(row.total_tokens)
            obs_count[idx] = float(row.count)
            obs_freq[idx]  = float(row.frequency)

    # Fit model
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model  = UnobservedComponents(obs, "local linear trend")
            result = model.fit(disp=False)
    except Exception as exc:
        logger.error("fit failed  word=%s  error=%s", word, str(exc))
        return None, {"word": word, "fit_status": "failed", "error": str(exc)}

    # Convergence status
    warnflag   = result.mle_retvals.get("warnflag", -1)
    fit_status = "converged" if warnflag == 0 else f"warn_{warnflag}"
    if warnflag != 0:
        logger.warning("convergence warn  word=%s  warnflag=%d  aic=%.2f",
                       word, warnflag, result.aic)

    # Extract smoothed states
    ss = result.smoother_results
    fr = result.filter_results

    latent_level = ss.smoothed_state[0, :]        # x_{w,t}
    latent_drift  = ss.smoothed_state[1, :]        # v_{w,t}

    # 95% CI from smoother covariance (2, 2, T) — take [0,0,:] for level var
    level_var  = ss.smoothed_state_cov[0, 0, :]
    half_width = 1.96 * np.sqrt(np.maximum(level_var, 0.0))
    lower_ci   = latent_level - half_width
    upper_ci   = latent_level + half_width

    # Curvature: first-difference of drift; NaN at t=0
    curvature = np.concatenate([[np.nan], np.diff(latent_drift)])

    # Local instability: rolling std of one-step-ahead innovations
    innovations = fr.forecasts_error[0, :]        # shape (T,)
    local_instability = (
        pd.Series(innovations)
        .rolling(window=INSTABILITY_WINDOW, min_periods=INSTABILITY_MIN_PERIODS)
        .std()
        .to_numpy()
    )

    # Per-year DataFrame
    fits_df = pd.DataFrame({
        "word":              word,
        "year":              ALL_YEARS,
        "observed_count":    obs_count,
        "frequency":         obs_freq,
        "latent_level":      latent_level,
        "latent_drift":      latent_drift,
        "curvature":         curvature,
        "lower_ci":          lower_ci,
        "upper_ci":          upper_ci,
        "local_instability": local_instability,
    })

    # MLE parameter estimates
    pnames       = result.param_names
    params       = result.params
    sigma2_obs   = float(params[pnames.index("sigma2.irregular")])
    sigma2_level = float(params[pnames.index("sigma2.level")])
    sigma2_drift = float(params[pnames.index("sigma2.trend")])

    summary = {
        "word":              word,
        "mean_drift":        float(latent_drift.mean()),
        "current_drift":     float(latent_drift[-1]),
        "mean_curvature":    float(np.nanmean(curvature)),
        "current_curvature": float(curvature[-1]),
        "mean_instability":  float(np.nanmean(local_instability)),
        "peak_year":         int(ALL_YEARS[int(np.argmax(latent_level))]),
        "sigma2_obs":        sigma2_obs,
        "sigma2_level":      sigma2_level,
        "sigma2_drift":      sigma2_drift,
        "aic":               float(result.aic),
        "fit_status":        fit_status,
    }

    return fits_df, summary


# ── Word selection ────────────────────────────────────────────

def select_words(meta: pl.DataFrame, args) -> list:
    """Apply --subset / --max-words flags and return ordered word list."""
    kept = meta.filter(pl.col("is_kept")).sort("total_count", descending=True)

    if args.subset is not None:
        kept = kept.head(args.subset)
        print(f"  Mode: --subset {args.subset}  (top {args.subset} words by total_count)")
    elif args.max_words is not None:
        kept = kept.sort("word").head(args.max_words)
        print(f"  Mode: --max-words {args.max_words}")
    else:
        print(f"  Mode: all words")

    return kept["word"].to_list()


# ── Result logging ────────────────────────────────────────────

def _log_and_report_failures(results):
    """Count outcomes and return (n_converged, n_warned, n_failed)."""
    n_converged = n_warned = n_failed = 0
    for _, summary in results:
        if summary is None:
            continue
        status = summary.get("fit_status", "failed")
        if status == "converged":
            n_converged += 1
        elif status == "failed":
            n_failed += 1
        else:
            n_warned += 1
    return n_converged, n_warned, n_failed


# ── Summary print ─────────────────────────────────────────────

def _print_summary(fits_pl, summary_pl, results, elapsed):
    n_converged, n_warned, n_failed = _log_and_report_failures(results)
    total = n_converged + n_warned + n_failed

    print(f"\n{'='*50}")
    print("Phase 4 Summary")
    print(f"{'='*50}")
    print(f"\nOutput 1: {WORD_FITS_PATH}")
    print(f"  Rows:   {len(fits_pl):,}")
    print(f"  Size:   {os.path.getsize(WORD_FITS_PATH)/1e6:.1f} MB")
    print(f"  Words:  {fits_pl['word'].n_unique():,}")

    print(f"\nOutput 2: {WORD_SUMMARY_PATH}")
    print(f"  Rows:   {len(summary_pl):,}")
    print(f"  Size:   {os.path.getsize(WORD_SUMMARY_PATH)/1e6:.1f} MB")

    print(f"\nConvergence ({total} words attempted):")
    print(f"  Converged (warnflag=0):  {n_converged:,}  ({100*n_converged/max(total,1):.1f}%)")
    print(f"  Converged with warnings: {n_warned:,}  ({100*n_warned/max(total,1):.1f}%)")
    print(f"  Failed (exception):      {n_failed:,}  ({100*n_failed/max(total,1):.1f}%)")
    if n_failed > 0:
        print(f"  Failed words logged to: {FAILED_LOG_PATH}")

    if len(summary_pl) > 0:
        converged = summary_pl.filter(pl.col("fit_status") == "converged")
        if len(converged) > 0:
            aics = converged["aic"].sort()
            print(f"\nAIC statistics (converged words):")
            print(f"  min:    {aics[0]:.1f}")
            print(f"  median: {aics[len(aics)//2]:.1f}")
            print(f"  max:    {aics[-1]:.1f}")

            low_drift = converged.filter(pl.col("sigma2_drift") < 1e-8).shape[0]
            high_drift = converged.filter(pl.col("sigma2_drift") > 1e-4).shape[0]
            print(f"\nsigma2_drift summary:")
            print(f"  Words with sigma2_drift < 1e-8 (stable trend):   {low_drift:,}")
            print(f"  Words with sigma2_drift > 1e-4 (changing trend): {high_drift:,}")

            peak_dist = summary_pl["peak_year"]
            print(f"\nPeak year distribution:")
            print(f"  Pre-1900:  {summary_pl.filter(pl.col('peak_year') < 1900).shape[0]:,}")
            print(f"  1900-1950: {summary_pl.filter((pl.col('peak_year') >= 1900) & (pl.col('peak_year') < 1950)).shape[0]:,}")
            print(f"  1950-2008: {summary_pl.filter(pl.col('peak_year') >= 1950).shape[0]:,}")

    print(f"\nTotal elapsed time: {elapsed:.1f}s")
    print("Done.")


# ── Main ──────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Phase 4: Word-level state-space model")
    parser.add_argument("--subset",    type=int, default=None,
                        help="Top N words by total_count (dev shortcut)")
    parser.add_argument("--max-words", type=int, default=None,
                        help="First N words alphabetically")
    args = parser.parse_args()

    _configure_logging()
    print("=== Phase 4: Word-Level State-Space Model ===\n")
    t0 = time.time()

    # [1/5] Load data
    print("[1/5] Loading clean panel and vocabulary metadata...")
    panel = pl.read_parquet(CLEAN_PANEL_PATH)
    meta  = pl.read_parquet(VOCAB_META_PATH)
    print(f"  Panel rows:  {len(panel):,}")
    print(f"  Vocab words: {meta.filter(pl.col('is_kept')).shape[0]:,}")

    # [2/5] Select words
    print("\n[2/5] Selecting word subset...")
    word_list = select_words(meta, args)
    print(f"  Words queued: {len(word_list):,}")

    # [3/5] Group panel
    print("\n[3/5] Grouping panel by word...")
    t_group = time.time()
    panel_pd = (
        panel
        .filter(pl.col("word").is_in(word_list))
        .select(["word", "year", "count", "total_tokens", "frequency"])
        .to_pandas()
    )
    grouped = {w: grp.reset_index(drop=True)
               for w, grp in panel_pd.groupby("word", sort=False)}
    print(f"  Groups formed in {time.time()-t_group:.1f}s")

    # [4/5] Parallel fit
    print(f"\n[4/5] Fitting UnobservedComponents (parallel, n_jobs=-1)...")
    pairs   = [(w, grouped[w]) for w in word_list if w in grouped]
    results = Parallel(n_jobs=-1)(
        delayed(fit_word)(w, df)
        for w, df in tqdm(pairs, desc="fitting words")
    )

    # [5/5] Aggregate
    print("\n[5/5] Aggregating results...")
    fits_list    = [r[0] for r in results if r[0] is not None]
    summary_list = [r[1] for r in results
                    if r[1] is not None and r[1].get("fit_status") != "failed"]

    n_fits    = len(fits_list)
    n_failed  = len(results) - n_fits
    print(f"  Fits collected: {n_fits:,}")
    if n_failed:
        print(f"  Words failed:   {n_failed:,}  (see {FAILED_LOG_PATH})")

    fits_pl    = pl.from_pandas(pd.concat(fits_list, ignore_index=True))
    summary_pl = pl.from_pandas(pd.DataFrame(summary_list))

    fits_pl.write_parquet(WORD_FITS_PATH)
    summary_pl.write_parquet(WORD_SUMMARY_PATH)

    _print_summary(fits_pl, summary_pl, results, time.time() - t0)


if __name__ == "__main__":
    main()
