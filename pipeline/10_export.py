"""Phase 10 — Frontend-Ready Export Files

Converts all pipeline parquet/JSON outputs into compact, frontend-friendly JSON
files that the existing "Lexical Explorer" website can consume directly.

Outputs (to outputs/):
  outputs/lii.json                  — LII timeline (209 rows, nulls preserved)
  outputs/changepoint_density.json  — CP density (209 rows)
  outputs/factor_trajectories.json  — Factor scores (209 rows × 10 factors)
  outputs/events.json               — Historical events catalog (copy)
  outputs/event_alignment.json      — Event alignment metrics (copy)
  outputs/word_index.json           — Word summary index (~70k words)
  outputs/words/{word}.json         — Per-word detail (top N_WORDS by frequency)
  outputs/word_map.json             — Top N_MAP words for 2D factor scatter map

JSON formats:
  lii.json         → [{year, lii, cr, bvn, fallback}, ...]   (null for missing years)
  density.json     → [{year, n, density}, ...]
  factors.json     → [{year, f1, f2, ..., f10}, ...]
  word_index.json  → {word: {peak, drift, regime, ncp}, ...}
  words/{w}.json   → {word, years[], level[], drift[], instab[], regime[], changepoints[], summary}
                     (parallel arrays for compactness — ~5x smaller than array-of-objects)

Usage:
    python pipeline/10_export.py
    python pipeline/10_export.py --n-words 1000
    python pipeline/10_export.py --skip-words    # system files only (fast)
"""

import argparse
import json
import os
import shutil
import time
from collections import defaultdict

import polars as pl

# ── Configuration ─────────────────────────────────────────────
PROCESSED_DIR = "data/processed"
OUTPUTS_DIR   = "outputs"
WORDS_DIR     = os.path.join(OUTPUTS_DIR, "words")

# Inputs
LII_PATH        = os.path.join(PROCESSED_DIR, "language_instability_index.parquet")
DENSITY_PATH    = os.path.join(PROCESSED_DIR, "changepoint_density.parquet")
TRAJ_PATH       = os.path.join(PROCESSED_DIR, "factor_trajectories.parquet")
EVENTS_JSON_SRC = os.path.join(PROCESSED_DIR, "historical_events.json")
ALIGN_JSON_SRC  = os.path.join(PROCESSED_DIR, "event_alignment.json")
SUMMARY_PATH    = os.path.join(PROCESSED_DIR, "word_summary_metrics.parquet")
FITS_PATH       = os.path.join(PROCESSED_DIR, "word_level_fits.parquet")
REGIMES_PATH    = os.path.join(PROCESSED_DIR, "regimes.parquet")
CP_PATH         = os.path.join(PROCESSED_DIR, "changepoints.parquet")
LOADINGS_PATH   = os.path.join(PROCESSED_DIR, "factor_loadings.parquet")

# Outputs
LII_OUT       = os.path.join(OUTPUTS_DIR, "lii.json")
DENSITY_OUT   = os.path.join(OUTPUTS_DIR, "changepoint_density.json")
FACTORS_OUT   = os.path.join(OUTPUTS_DIR, "factor_trajectories.json")
EVENTS_OUT    = os.path.join(OUTPUTS_DIR, "events.json")
ALIGN_OUT     = os.path.join(OUTPUTS_DIR, "event_alignment.json")
INDEX_OUT     = os.path.join(OUTPUTS_DIR, "word_index.json")
WORD_MAP_OUT  = os.path.join(OUTPUTS_DIR, "word_map.json")

N_WORDS   = 500    # per-word JSON files to generate (top by mean_instability)
N_MAP     = 500    # words for 2D factor scatter map (top by Euclidean factor distance)
N_FACTORS = 10
YEAR_MIN  = 1800
YEAR_MAX  = 2008

# Anchor words always included in per-word export regardless of ranking
ANCHOR_WORDS = {"war", "computer", "technology", "wireless", "television",
                "capitalism", "communism", "warfare", "crisis", "trade"}

os.makedirs(OUTPUTS_DIR, exist_ok=True)
os.makedirs(WORDS_DIR,   exist_ok=True)


# ── Export functions ───────────────────────────────────────────

def export_lii(lii_path: str, out_path: str) -> int:
    """Export LII timeline as compact JSON array.

    Short keys: lii, cr, bvn (broad_vs_narrow), fallback.
    Null values preserved for years 1800–1804.
    """
    df = (
        pl.read_parquet(lii_path)
        .sort("year")
        .select([
            pl.col("year"),
            pl.col("lii_value").alias("lii"),
            pl.col("concentration_ratio").alias("cr"),
            pl.col("broad_vs_narrow_score").alias("bvn"),
            pl.col("lii_fallback").alias("fallback"),
        ])
    )
    records = []
    for row in df.iter_rows(named=True):
        records.append({
            "year":    int(row["year"]),
            "lii":     float(row["lii"])     if row["lii"]     is not None else None,
            "cr":      float(row["cr"])      if row["cr"]      is not None else None,
            "bvn":     float(row["bvn"])     if row["bvn"]     is not None else None,
            "fallback":float(row["fallback"])if row["fallback"]is not None else None,
        })
    with open(out_path, "w") as f:
        json.dump(records, f, separators=(",", ":"))
    return len(records)


def export_density(density_path: str, out_path: str) -> int:
    """Export changepoint density as compact JSON array."""
    df = (
        pl.read_parquet(density_path)
        .sort("year")
        .select([
            pl.col("year"),
            pl.col("n_changepoints").alias("n"),
            pl.col("normalized_density").alias("density"),
        ])
    )
    records = [
        {"year": int(r["year"]), "n": int(r["n"]), "density": float(r["density"])}
        for r in df.iter_rows(named=True)
    ]
    with open(out_path, "w") as f:
        json.dump(records, f, separators=(",", ":"))
    return len(records)


def export_factors(traj_path: str, out_path: str) -> int:
    """Export factor trajectories with short key names f1..f10."""
    df = pl.read_parquet(traj_path).sort("year")
    factor_cols = [f"factor_{i + 1}" for i in range(N_FACTORS)]
    records = []
    for row in df.iter_rows(named=True):
        rec = {"year": int(row["year"])}
        for i, col in enumerate(factor_cols):
            rec[f"f{i + 1}"] = float(row[col])
        records.append(rec)
    with open(out_path, "w") as f:
        json.dump(records, f, separators=(",", ":"))
    return len(records)


def copy_event_files(src_events: str, src_align: str,
                     out_events: str, out_align: str) -> None:
    """Copy the event JSON files from data/processed/ to outputs/."""
    shutil.copy2(src_events, out_events)
    shutil.copy2(src_align,  out_align)


def export_word_index(
    summary_path: str,
    regimes_path: str,
    cp_path: str,
    out_path: str,
) -> int:
    """Build and export the word summary index dict.

    Returns number of words exported.
    """
    # Load word summary (mean_drift, peak_year)
    summary = pl.read_parquet(
        summary_path,
        columns=["word", "mean_drift", "peak_year"],
    )

    # Dominant regime per word (most frequent regime_label)
    regimes_df = pl.read_parquet(regimes_path, columns=["word", "regime_label"])
    dom_regime = (
        regimes_df
        .group_by(["word", "regime_label"])
        .agg(pl.len().alias("n"))
        .sort("n", descending=True)
        .unique(["word"], keep="first")
        .select(["word", "regime_label"])
        .rename({"regime_label": "regime"})
    )

    # Changepoint count per word
    cp_counts = (
        pl.read_parquet(cp_path, columns=["word"])
        .group_by("word")
        .agg(pl.len().alias("ncp"))
    )

    # Join all
    index_df = (
        summary
        .join(dom_regime,  on="word", how="left")
        .join(cp_counts,   on="word", how="left")
        .with_columns(pl.col("ncp").fill_null(0).cast(pl.Int32))
    )

    # Build output dict
    word_index = {}
    for row in index_df.iter_rows(named=True):
        word_index[row["word"]] = {
            "peak":   int(row["peak_year"]) if row["peak_year"] is not None else None,
            "drift":  round(float(row["mean_drift"]), 6) if row["mean_drift"] is not None else 0.0,
            "regime": row["regime"] if row["regime"] else "stable",
            "ncp":    int(row["ncp"]),
        }

    with open(out_path, "w") as f:
        json.dump(word_index, f, separators=(",", ":"))
    return len(word_index)


def export_per_word(
    summary_path: str,
    fits_path: str,
    regimes_path: str,
    cp_path: str,
    words_dir: str,
    n_words: int,
) -> int:
    """Export per-word detail JSON files for the top n_words by mean_frequency.

    Uses parallel arrays format for compact file size.
    Loads word_level_fits and regimes once (batch), groups by word.

    Returns number of files written.
    """
    # Select top words by mean_instability + always include anchor words
    all_summary = pl.read_parquet(summary_path, columns=["word", "mean_instability"])
    top_words_df = all_summary.sort("mean_instability", descending=True).head(n_words)
    top_words = set(top_words_df["word"].to_list())
    # Always include anchor words that exist in the vocabulary
    vocab = set(all_summary["word"].to_list())
    top_words |= (ANCHOR_WORDS & vocab)

    print(f"  Loading word_level_fits …")
    fits_df = (
        pl.read_parquet(
            fits_path,
            columns=["word", "year", "latent_level", "latent_drift",
                     "local_instability"],
        )
        .filter(pl.col("word").is_in(top_words))
        .sort(["word", "year"])
    )

    print(f"  Loading regimes …")
    reg_df = (
        pl.read_parquet(regimes_path, columns=["word", "year", "regime_label"])
        .filter(pl.col("word").is_in(top_words))
        .sort(["word", "year"])
    )

    print(f"  Loading changepoints …")
    cp_df = (
        pl.read_parquet(cp_path, columns=["word", "changepoint_year"])
        .filter(pl.col("word").is_in(top_words))
    )
    cp_by_word = defaultdict(list)
    for row in cp_df.iter_rows(named=True):
        cp_by_word[row["word"]].append(int(row["changepoint_year"]))

    # Word summary for per-word summary block
    summary_df = (
        pl.read_parquet(summary_path, columns=["word", "peak_year", "mean_drift"])
        .filter(pl.col("word").is_in(top_words))
    )
    summary_by_word = {
        row["word"]: row
        for row in summary_df.iter_rows(named=True)
    }

    # Group fits and regimes by word
    print(f"  Grouping by word …")
    fits_grouped  = {w[0]: g for w, g in fits_df.group_by("word")}
    reg_grouped   = {w[0]: g for w, g in reg_df.group_by("word")}

    print(f"  Writing per-word JSON files …")
    written = 0
    for word in sorted(top_words):
        if word not in fits_grouped:
            continue

        fits_w = fits_grouped[word].sort("year")
        reg_w  = reg_grouped.get(word, pl.DataFrame()).sort("year") if word in reg_grouped else None

        years      = fits_w["year"].to_list()
        levels     = [round(v, 6) if v is not None else None
                      for v in fits_w["latent_level"].to_list()]
        drifts     = [round(v, 8) if v is not None else None
                      for v in fits_w["latent_drift"].to_list()]
        instabs    = [round(v, 6) if v is not None else None
                      for v in fits_w["local_instability"].to_list()]

        if reg_w is not None and len(reg_w) > 0:
            regime_map = dict(zip(reg_w["year"].to_list(), reg_w["regime_label"].to_list()))
            regimes = [regime_map.get(y, "stable") for y in years]
        else:
            regimes = ["stable"] * len(years)

        changepoints = sorted(cp_by_word.get(word, []))
        smry = summary_by_word.get(word, {})

        record = {
            "word":        word,
            "years":       [int(y) for y in years],
            "level":       levels,
            "drift":       drifts,
            "instab":      instabs,
            "regime":      regimes,
            "changepoints": changepoints,
            "summary": {
                "peak":  int(smry["peak_year"]) if smry.get("peak_year") else None,
                "drift": round(float(smry["mean_drift"]), 6) if smry.get("mean_drift") else 0.0,
                "ncp":   len(changepoints),
            },
        }

        out_path = os.path.join(words_dir, f"{word}.json")
        with open(out_path, "w") as f:
            json.dump(record, f, separators=(",", ":"))
        written += 1

    return written


def export_word_map(
    loadings_path: str,
    summary_path: str,
    regimes_path: str,
    cp_path: str,
    out_path: str,
    n_map: int = N_MAP,
) -> int:
    """Export top n_map words by factor spread for the 2D word map scatter.

    Selects words by Euclidean distance from origin in (factor_1, factor_2)
    space — words with high |f1| or |f2| are the most interpretable on the map.
    Always includes ANCHOR_WORDS regardless of ranking.

    Returns number of records written.
    """
    loadings = pl.read_parquet(loadings_path, columns=["word", "factor_1", "factor_2"])

    # Score = sqrt(f1^2 + f2^2)
    loadings = loadings.with_columns(
        (pl.col("factor_1").pow(2) + pl.col("factor_2").pow(2)).sqrt().alias("_score")
    )
    top_df   = loadings.sort("_score", descending=True).head(n_map)
    vocab    = set(loadings["word"].to_list())
    top_words = set(top_df["word"].to_list()) | (ANCHOR_WORDS & vocab)

    # Dominant regime (only load the words we need)
    dom_regime = (
        pl.read_parquet(regimes_path, columns=["word", "regime_label"])
        .filter(pl.col("word").is_in(top_words))
        .group_by(["word", "regime_label"])
        .len()
        .sort("len", descending=True)
        .unique(["word"], keep="first")
        .select(["word", "regime_label"])
        .rename({"regime_label": "regime"})
    )

    # CP count
    cp_counts = (
        pl.read_parquet(cp_path, columns=["word", "changepoint_year"])
        .filter(pl.col("word").is_in(top_words))
        .group_by("word")
        .len()
        .rename({"len": "ncp"})
    )

    # Peak year
    summary = pl.read_parquet(summary_path, columns=["word", "peak_year"])

    joined = (
        loadings.filter(pl.col("word").is_in(top_words))
        .select(["word", "factor_1", "factor_2"])
        .join(summary, on="word", how="left")
        .join(dom_regime, on="word", how="left")
        .join(cp_counts, on="word", how="left")
        .with_columns([
            pl.col("ncp").fill_null(0),
            pl.col("regime").fill_null("stable"),
        ])
    )

    records = []
    for row in joined.iter_rows(named=True):
        records.append({
            "word":   row["word"],
            "f1":     round(float(row["factor_1"]), 6),
            "f2":     round(float(row["factor_2"]), 6),
            "regime": row["regime"],
            "peak":   int(row["peak_year"]) if row["peak_year"] is not None else None,
            "ncp":    int(row["ncp"]),
        })

    with open(out_path, "w") as f:
        json.dump(records, f, separators=(",", ":"))

    return len(records)


# ── Main ───────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 10: Frontend-ready export files"
    )
    parser.add_argument(
        "--n-words", type=int, default=N_WORDS,
        help=f"Per-word JSON files to generate (default: {N_WORDS})"
    )
    parser.add_argument(
        "--skip-words", action="store_true",
        help="Skip per-word export (system-level files only)"
    )
    args = parser.parse_args()

    t0 = time.time()
    print("=" * 60)
    print("Phase 10 — Frontend-Ready Export Files")
    print(f"  n_words    : {args.n_words}")
    print(f"  skip_words : {args.skip_words}")
    print("=" * 60)

    # ── [1/6] LII timeline ─────────────────────────────────────
    print(f"\n[1/6] Exporting LII timeline → {LII_OUT} …")
    n = export_lii(LII_PATH, LII_OUT)
    sz = os.path.getsize(LII_OUT)
    print(f"  {n} rows written  ({sz / 1024:.1f} KB)")

    # ── [2/6] Changepoint density ─────────────────────────────
    print(f"\n[2/6] Exporting changepoint density → {DENSITY_OUT} …")
    n = export_density(DENSITY_PATH, DENSITY_OUT)
    sz = os.path.getsize(DENSITY_OUT)
    print(f"  {n} rows written  ({sz / 1024:.1f} KB)")

    # ── [3/6] Factor trajectories ─────────────────────────────
    print(f"\n[3/6] Exporting factor trajectories → {FACTORS_OUT} …")
    n = export_factors(TRAJ_PATH, FACTORS_OUT)
    sz = os.path.getsize(FACTORS_OUT)
    print(f"  {n} rows written  ({sz / 1024:.1f} KB)")

    # ── [4/6] Copy event files ────────────────────────────────
    print(f"\n[4/6] Copying event files …")
    copy_event_files(EVENTS_JSON_SRC, ALIGN_JSON_SRC, EVENTS_OUT, ALIGN_OUT)
    print(f"  {EVENTS_OUT}  ({os.path.getsize(EVENTS_OUT) / 1024:.1f} KB)")
    print(f"  {ALIGN_OUT}   ({os.path.getsize(ALIGN_OUT) / 1024:.1f} KB)")

    # ── [5/6] Word index ──────────────────────────────────────
    print(f"\n[5/6] Exporting word index → {INDEX_OUT} …")
    n = export_word_index(SUMMARY_PATH, REGIMES_PATH, CP_PATH, INDEX_OUT)
    sz = os.path.getsize(INDEX_OUT)
    print(f"  {n:,} words written  ({sz / 1024 / 1024:.1f} MB)")

    # ── [6/7] Per-word JSON files ─────────────────────────────
    if not args.skip_words:
        print(f"\n[6/7] Exporting per-word JSON files (top {args.n_words}) …")
        t6 = time.time()
        written = export_per_word(
            SUMMARY_PATH, FITS_PATH, REGIMES_PATH, CP_PATH,
            WORDS_DIR, args.n_words,
        )
        files = [f for f in os.listdir(WORDS_DIR) if f.endswith(".json")]
        total_kb = sum(os.path.getsize(os.path.join(WORDS_DIR, f)) for f in files) / 1024
        print(f"  {written} files written  (total: {total_kb:.0f} KB)  [{time.time() - t6:.1f}s]")
    else:
        print(f"\n[6/7] Skipping per-word export (--skip-words)")

    # ── [7/7] Word map for 2D factor scatter ─────────────────
    print(f"\n[7/7] Exporting word map → {WORD_MAP_OUT} …")
    n = export_word_map(LOADINGS_PATH, SUMMARY_PATH, REGIMES_PATH, CP_PATH, WORD_MAP_OUT)
    sz = os.path.getsize(WORD_MAP_OUT)
    print(f"  {n} words written  ({sz / 1024:.1f} KB)")

    elapsed = time.time() - t0
    print(f"\nPhase 10 complete.  Total time: {elapsed:.1f}s")
    print(f"\nOutputs in {OUTPUTS_DIR}/:")
    for fname in sorted(os.listdir(OUTPUTS_DIR)):
        fpath = os.path.join(OUTPUTS_DIR, fname)
        if os.path.isfile(fpath):
            print(f"  {fname:<40} {os.path.getsize(fpath) / 1024:>8.1f} KB")
    words_files = [f for f in os.listdir(WORDS_DIR) if f.endswith(".json")]
    print(f"  words/  ({len(words_files)} files)")


if __name__ == "__main__":
    main()
