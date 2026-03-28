# CS2 — Language Instability Pipeline

## What this repo does
This is the CS2 data pipeline for the Google Books Ngram language instability project.
It ingests raw Ngram files, cleans and normalises them, fits word-level models,
detects changepoints, builds a Language Instability Index, and exports frontend-ready files.

## Folder structure
- data/raw/         — original downloaded Ngram files (.gz)
- data/intermediate/ — filtered shard-level parquet files
- data/processed/   — clean, model-ready datasets
- outputs/          — frontend-ready files for CS1
- pipeline/         — all Python scripts (run these in order)
- notebooks/        — scratch notebooks for exploration only
- docs/             — notes, data dictionary, modeling decisions

## How to run
1. Activate the virtual environment: source venv/bin/activate
2. Run scripts in order: python pipeline/01_ingest.py, then 02, 03, etc.

## Output location
All final frontend files appear in outputs/

## Dependencies
pip install -r requirements.txt