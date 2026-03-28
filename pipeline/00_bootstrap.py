"""Bootstrap — fetch word data from Google Ngrams API and build clean_panel.parquet.

Bypasses 01_ingest.py (which needs the giant .gz shard files).
Produces data/processed/clean_panel.parquet and vocabulary_metadata.parquet
in the exact format expected by phases 03–10.
"""

import os
import time
import urllib.request
import urllib.parse
import json

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

PROCESSED_DIR = "data/processed"
TOTAL_COUNTS_FILE = "data/raw/googlebooks-eng-all-totalcounts-20120701.txt"
YEAR_MIN = 1800
YEAR_MAX = 2008

os.makedirs(PROCESSED_DIR, exist_ok=True)

WORDS = [
    # Technology
    "computer", "internet", "telephone", "television", "radio",
    "electricity", "automobile", "aircraft", "nuclear", "satellite",
    "wireless", "telegraph", "typewriter", "photography", "cinema",
    # Conflict / geopolitics
    "war", "battle", "revolution", "democracy", "empire",
    "freedom", "nationalism", "capitalism", "communism", "crisis",
    "peace", "army", "weapon", "soldier", "terror",
    # Science / medicine
    "pandemic", "disease", "vaccine", "evolution", "theory",
    "chemistry", "biology", "physics", "medicine", "surgery",
    # Society / culture
    "education", "religion", "marriage", "poverty", "trade",
    "labour", "industry", "factory", "newspaper", "literature",
    "science", "philosophy", "culture", "society", "economy",
    # Common dynamics words
    "king", "government", "parliament", "election", "justice",
]

# ── Load total counts ─────────────────────────────────────────
def load_total_counts():
    with open(TOTAL_COUNTS_FILE, "r", encoding="utf-8") as f:
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
    print(f"Total counts: {len(totals)} years ({min(totals)}–{max(totals)})")
    return totals

# ── Fetch one word from Google Ngrams API ─────────────────────
def fetch_ngram(word, retries=3):
    params = urllib.parse.urlencode({
        "content": word,
        "year_start": YEAR_MIN,
        "year_end": YEAR_MAX,
        "corpus": "en-US",
        "smoothing": 0,
    })
    url = f"https://books.google.com/ngrams/json?{params}"
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read())
            if not data:
                print(f"  {word}: no data returned")
                return None
            ts = data[0]["timeseries"]
            years = list(range(YEAR_MIN, YEAR_MIN + len(ts)))
            return dict(zip(years, ts))
        except Exception as e:
            print(f"  {word}: attempt {attempt+1} failed — {e}")
            time.sleep(2)
    return None

# ── Main ──────────────────────────────────────────────────────
def main():
    totals = load_total_counts()

    all_years = list(range(YEAR_MIN, YEAR_MAX + 1))
    rows = []

    print(f"\nFetching {len(WORDS)} words from Google Ngrams API...")
    for word in WORDS:
        print(f"  fetching: {word}", end=" ", flush=True)
        freq_by_year = fetch_ngram(word)
        if freq_by_year is None:
            print("SKIP")
            continue
        for year in all_years:
            if year not in totals:
                continue
            freq = freq_by_year.get(year, 0.0)
            total_tokens = totals[year]
            count = round(freq * total_tokens)  # back-compute approximate count
            rows.append({
                "word": word,
                "year": year,
                "count": count,
                "total_tokens": total_tokens,
                "frequency": freq,
            })
        print("OK")
        time.sleep(0.3)  # be polite to the API

    df = pd.DataFrame(rows)
    print(f"\nTotal rows: {len(df)}")

    # ── Vocabulary metadata ───────────────────────────────────
    meta_rows = []
    for word, grp in df.groupby("word"):
        present = grp[grp["count"] > 0]
        years_present = present["year"].nunique()
        total_count = grp["count"].sum()
        sparsity = 1.0 - years_present / len(all_years)
        meta_rows.append({
            "word": word,
            "first_year_seen": int(present["year"].min()) if len(present) > 0 else YEAR_MIN,
            "last_year_seen": int(present["year"].max()) if len(present) > 0 else YEAR_MAX,
            "total_count": int(total_count),
            "total_years_present": int(years_present),
            "mean_frequency": float(grp["frequency"].mean()),
            "sparsity": float(sparsity),
            "is_kept": years_present >= 20 and total_count >= 1000,
        })
    meta_df = pd.DataFrame(meta_rows)

    # ── Save ─────────────────────────────────────────────────
    panel_path = os.path.join(PROCESSED_DIR, "clean_panel.parquet")
    vocab_path = os.path.join(PROCESSED_DIR, "vocabulary_metadata.parquet")

    df.to_parquet(panel_path, index=False)
    meta_df.to_parquet(vocab_path, index=False)

    print(f"\nSaved: {panel_path}")
    print(f"Saved: {vocab_path}")
    print(f"Words fetched: {df['word'].nunique()}")
    kept = meta_df[meta_df["is_kept"]]["word"].tolist()
    print(f"Words passing filter: {len(kept)} — {kept}")

if __name__ == "__main__":
    main()
