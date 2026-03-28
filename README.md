# CS2 — Language Instability Pipeline

## What this repo does
This is the CS2 data pipeline for the Google Books Ngram language instability project.
It ingests raw Ngram files, cleans and normalises them, fits word-level models,
detects changepoints, builds a Language Instability Index, and exports frontend-ready files.

The pipeline is built around three mathematical layers:
- **Layer 1** — Poisson/NegBin observation model on word counts
- **Layer 2** — Word-level local linear trend state-space model
- **Layer 3** — Dynamic factor model + Language Instability Index `LII_t = tr(Σ_t)`

See [docs/model.md](docs/model.md) for the full mathematical specification.

## Folder structure
```
data/raw/            — original downloaded Ngram .gz files + total counts file
data/intermediate/   — filtered shard-level parquet files (Phase 1 output)
data/processed/      — clean, model-ready datasets (Phase 2+ output)
outputs/             — frontend-ready files for CS1
pipeline/            — all Python scripts (run in order)
notebooks/           — scratch notebooks for exploration only
docs/                — model spec, data dictionary, modeling decisions
```

## Pipeline scripts

| Script | Phase | Status | What it does |
|--------|-------|--------|--------------|
| `01_ingest.py` | 1 | Done | Streams raw `.gz` Ngram shards, filters to lowercase alpha words 1800–2008, saves parquet to `data/intermediate/` |
| `02_clean.py` | 2 | Done | Aggregates by word+year, joins yearly total token counts, computes normalised frequency, applies vocabulary filter, saves `clean_panel.parquet` and `vocabulary_metadata.parquet` |
| `03_smooth.py` | 3 | Done | Fits cubic smoothing splines to log-frequency per word, extracts drift (1st deriv) and curvature (2nd deriv), saves `smoothed_word_series.parquet` |
| `04_word_model.py` | 4 | Pending | Word-level Poisson state-space model (latent level + drift) |
| `05_changepoints.py` | 5 | Pending | Structural break detection per word |
| `06_regimes.py` | 6 | Pending | Regime labelling (adoption / decline / turbulent / stable) |
| `07_factor_model.py` | 7 | Pending | System-wide dynamic factor model |
| `08_lii.py` | 8 | Pending | Language Instability Index construction |
| `09_events.py` | 9 | Pending | Historical event alignment and annotation |
| `10_export.py` | 10 | Pending | Frontend-ready export files |
| `11_eval.py` | 11 | Pending | Evaluation and sanity checks |

## Key processed outputs

| File | Description |
|------|-------------|
| `data/processed/clean_panel.parquet` | 10.4M rows — word, year, count, total_tokens, frequency |
| `data/processed/vocabulary_metadata.parquet` | 447k words with is_kept flag (70,535 pass filter) |
| `data/processed/smoothed_word_series.parquet` | 12.6M rows — log_freq, smooth_level, smooth_drift, smooth_curvature per word-year |
| `outputs/phase3_sanity_check.png` | Sanity check plot of smooth series for key words |

## How to run

```bash
# 1. Activate the virtual environment
source venv/bin/activate

# 2. Download the total counts file (required for Phase 2)
curl -o data/raw/googlebooks-eng-all-totalcounts-20120701.txt \
  "http://storage.googleapis.com/books/ngrams/books/googlebooks-eng-all-totalcounts-20120701.txt"

# 3. Run scripts in order
python pipeline/01_ingest.py
python pipeline/02_clean.py
python pipeline/03_smooth.py --plot   # --plot saves outputs/phase3_sanity_check.png

# 4. Run sanity tests
pytest pipeline/test_phases.py -v
```

## Vocabulary filter (Phase 2)
Words are kept if they:
- Are lowercase alphabetic only (`[a-z]+`)
- Appear in **≥ 20 distinct years**
- Have **≥ 1000 total occurrences** across all years

This reduces 447k unique tokens to **70,535 stable words**.

## Data source
Google Books English 1-gram dataset (2012 release), years 1800–2008.
Total token counts file: `googlebooks-eng-all-totalcounts-20120701.txt`

## Dependencies
```bash
pip install -r requirements.txt
```