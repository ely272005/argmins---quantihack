import gzip
import os
import re
import pandas as pd
from tqdm import tqdm

# ── Configuration ────────────────────────────────────────────
RAW_DIR = "data/raw"
OUT_DIR = "data/intermediate"
YEAR_MIN = 1800
YEAR_MAX = 2019

os.makedirs(OUT_DIR, exist_ok=True)

def is_clean_word(token):
    """Keep only lowercase alphabetic words."""
    return bool(re.fullmatch(r'[a-z]+', token))

def open_file(filepath):
    """Open either gzip or plain text file."""
    try:
        f = gzip.open(filepath, 'rt', encoding='utf-8', errors='replace')
        f.read(1)
        f.seek(0)
        return f
    except Exception:
        return open(filepath, 'r', encoding='utf-8', errors='replace')

def process_shard(filepath):
    filename = os.path.basename(filepath)
    rows = []

    rows_read = 0
    rows_kept = 0
    rows_discarded = 0
    years_seen = set()

    print(f"\nProcessing: {filename}")

    with open_file(filepath) as f:
        for line in tqdm(f, desc=filename):
            rows_read += 1
            parts = line.strip().split('\t')

            if len(parts) != 4:
                rows_discarded += 1
                continue

            token, year_str, match_count_str, volume_count_str = parts

            try:
                year = int(year_str)
            except ValueError:
                rows_discarded += 1
                continue

            if year < YEAR_MIN or year > YEAR_MAX:
                rows_discarded += 1
                continue

            if not is_clean_word(token):
                rows_discarded += 1
                continue

            try:
                match_count = int(match_count_str)
                volume_count = int(volume_count_str)
            except ValueError:
                rows_discarded += 1
                continue

            rows.append({
                'word': token,
                'year': year,
                'match_count': match_count,
                'volume_count': volume_count,
                'source_file': filename
            })
            rows_kept += 1
            years_seen.add(year)

    if rows:
        df = pd.DataFrame(rows)
        out_name = re.sub(r'\.(gz|txt)$', '.parquet', filename)
        out_path = os.path.join(OUT_DIR, out_name)
        df.to_parquet(out_path, index=False)
        print(f"  Saved to: {out_path}")
    else:
        print(f"  No rows kept for {filename}")

    print(f"  Rows read:      {rows_read:,}")
    print(f"  Rows kept:      {rows_kept:,}")
    print(f"  Rows discarded: {rows_discarded:,}")
    if years_seen:
        print(f"  Year range:     {min(years_seen)} – {max(years_seen)}")

# ── Run on all shard files in data/raw ─────────────────────────
all_files = [
    os.path.join(RAW_DIR, f)
    for f in os.listdir(RAW_DIR)
    if '1gram' in f and not f.endswith('.txt') == False or f.endswith('.gz')
]

shard_files = [
    os.path.join(RAW_DIR, f)
    for f in os.listdir(RAW_DIR)
    if '1gram' in f
]

if not shard_files:
    print("No shard files found in data/raw/")
else:
    for filepath in sorted(shard_files):
        process_shard(filepath)

print("\nDone. Filtered files saved to data/intermediate/")