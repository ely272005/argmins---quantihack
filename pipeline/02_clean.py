import os
import re
import glob
import polars as pl
import pandas as pd

# ── Configuration ────────────────────────────────────────────
INTERMEDIATE_DIR = "data/intermediate"
PROCESSED_DIR    = "data/processed"
TOTAL_COUNTS_FILE = "data/raw/googlebooks-eng-all-totalcounts-20120701.txt"

YEAR_MIN = 1800
YEAR_MAX = 2008  # total counts file only covers up to 2008

MIN_YEARS_PRESENT = 20   # word must appear in at least this many distinct years
MIN_TOTAL_COUNT   = 1000  # word must have at least this many total occurrences

SAMPLE_WORDS = ["war", "computer", "internet", "typewriter", "telegram", "pandemic"]

os.makedirs(PROCESSED_DIR, exist_ok=True)


def load_total_counts(filepath):
    """Parse the Google Books total counts file.

    Format: tab-separated entries, each entry is year,match_count,page_count,volume_count.
    Returns a dict {year: total_tokens}.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        raw = f.read()

    totals = {}
    for entry in raw.strip().split("\t"):
        entry = entry.strip()
        if not entry:
            continue
        parts = entry.split(",")
        if len(parts) != 4:
            continue
        try:
            year = int(parts[0])
            match_count = int(parts[1])
        except ValueError:
            continue
        if YEAR_MIN <= year <= YEAR_MAX:
            totals[year] = match_count

    print(f"  Total counts loaded: {len(totals)} years ({min(totals)} – {max(totals)})")
    return totals


def load_intermediate_panel(intermediate_dir):
    """Load and concatenate all intermediate parquet shards into one polars DataFrame."""
    paths = sorted(glob.glob(os.path.join(intermediate_dir, "*.parquet")))
    if not paths:
        raise FileNotFoundError(f"No parquet files found in {intermediate_dir}")

    print(f"  Loading {len(paths)} shard(s): {[os.path.basename(p) for p in paths]}")
    dfs = [pl.read_parquet(p) for p in paths]
    combined = pl.concat(dfs)
    print(f"  Raw rows loaded: {len(combined):,}")
    return combined


def aggregate_panel(df):
    """Group by (word, year), sum match_count. Drop volume_count and source_file."""
    agg = (
        df
        .group_by(["word", "year"])
        .agg(pl.col("match_count").sum().alias("count"))
        .sort(["word", "year"])
    )
    print(f"  Rows after group-by aggregation: {len(agg):,}")
    return agg


def join_total_counts(df, totals):
    """Join yearly total tokens and compute normalized frequency."""
    totals_df = pl.DataFrame({
        "year": list(totals.keys()),
        "total_tokens": list(totals.values()),
    })
    joined = df.join(totals_df, on="year", how="inner")
    print(f"  Rows after joining total counts: {len(joined):,}")
    joined = joined.with_columns(
        (pl.col("count").cast(pl.Float64) / pl.col("total_tokens").cast(pl.Float64))
        .alias("frequency")
    )
    return joined


def build_vocabulary_metadata(df):
    """Compute per-word stats and assign is_kept flag."""
    meta = (
        df
        .group_by("word")
        .agg([
            pl.col("year").min().alias("first_year_seen"),
            pl.col("year").max().alias("last_year_seen"),
            pl.col("count").sum().alias("total_count"),
            pl.col("year").n_unique().alias("total_years_present"),
            pl.col("frequency").mean().alias("mean_frequency"),
        ])
        .with_columns(
            # sparsity = fraction of years in range with NO observation
            (1.0 - pl.col("total_years_present") / (YEAR_MAX - YEAR_MIN + 1))
            .alias("sparsity")
        )
        .with_columns(
            (
                (pl.col("total_years_present") >= MIN_YEARS_PRESENT) &
                (pl.col("total_count") >= MIN_TOTAL_COUNT)
            ).alias("is_kept")
        )
        .sort("total_count", descending=True)
    )
    return meta


def filter_panel(df, meta):
    """Keep only rows for words that pass vocabulary filter."""
    kept_words = meta.filter(pl.col("is_kept")).select("word")
    filtered = df.join(kept_words, on="word", how="inner")
    print(f"  Rows after vocabulary filter: {len(filtered):,}")
    return filtered


def print_sample_words(df, words):
    """Print frequency at a few years for each sample word."""
    print("\n  Sample word check:")
    for w in words:
        subset = (
            df
            .filter(pl.col("word") == w)
            .sort("year")
        )
        if len(subset) == 0:
            print(f"    {w:15s}  NOT FOUND in vocabulary")
            continue
        # show roughly 5 evenly spaced years
        rows = subset.to_pandas()
        step = max(1, len(rows) // 5)
        samples = rows.iloc[::step][["year", "frequency"]].head(6)
        freq_str = "  ".join(f"{int(r.year)}:{r.frequency:.2e}" for _, r in samples.iterrows())
        print(f"    {w:15s}  {freq_str}")


def main():
    print("=== Phase 2: Clean & Normalize ===\n")

    # 1. Load total counts
    print("[1/6] Loading total counts file...")
    totals = load_total_counts(TOTAL_COUNTS_FILE)

    # 2. Load intermediate panel
    print("\n[2/6] Loading intermediate parquets...")
    raw_df = load_intermediate_panel(INTERMEDIATE_DIR)

    # 3. Filter to year range (total counts years)
    print("\n[3/6] Filtering to year range...")
    raw_df = raw_df.filter(
        (pl.col("year") >= YEAR_MIN) & (pl.col("year") <= YEAR_MAX)
    )
    print(f"  Rows after year filter: {len(raw_df):,}")

    # 4. Aggregate by (word, year)
    print("\n[4/6] Aggregating by (word, year)...")
    panel = aggregate_panel(raw_df)

    # 5. Join total counts and compute frequency
    print("\n[5/6] Joining total counts and computing frequency...")
    panel = join_total_counts(panel, totals)

    # 6. Build vocabulary metadata and filter
    print("\n[6/6] Building vocabulary and applying filters...")
    meta = build_vocabulary_metadata(panel)

    n_total = len(meta)
    n_kept  = meta.filter(pl.col("is_kept")).shape[0]
    print(f"  Vocabulary before filter: {n_total:,} words")
    print(f"  Vocabulary after filter:  {n_kept:,} words")
    print(f"  (min_years_present={MIN_YEARS_PRESENT}, min_total_count={MIN_TOTAL_COUNT})")

    panel_filtered = filter_panel(panel, meta)

    # Sample word check
    print_sample_words(panel_filtered, SAMPLE_WORDS)

    # Save outputs
    clean_path = os.path.join(PROCESSED_DIR, "clean_panel.parquet")
    meta_path  = os.path.join(PROCESSED_DIR, "vocabulary_metadata.parquet")

    panel_filtered.write_parquet(clean_path)
    meta.write_parquet(meta_path)

    print(f"\n  Saved: {clean_path}  ({os.path.getsize(clean_path)/1e6:.1f} MB)")
    print(f"  Saved: {meta_path}  ({os.path.getsize(meta_path)/1e6:.1f} MB)")
    print("\nDone.")


if __name__ == "__main__":
    main()
