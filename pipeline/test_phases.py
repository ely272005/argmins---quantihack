"""
Sanity tests for Phase 2 (clean), Phase 3 (smooth), and Phase 4 (word model) outputs.

Run with:
    source venv/bin/activate
    pytest pipeline/test_phases.py -v
"""

import math
import os

import numpy as np
import polars as pl
import pytest

# ── Paths ─────────────────────────────────────────────────────
CLEAN_PANEL  = "data/processed/clean_panel.parquet"
VOCAB_META   = "data/processed/vocabulary_metadata.parquet"
SMOOTHED     = "data/processed/smoothed_word_series.parquet"
WORD_FITS    = "data/processed/word_level_fits.parquet"
WORD_SUMMARY = "data/processed/word_summary_metrics.parquet"

YEAR_MIN = 1800
YEAR_MAX = 2008
MIN_YEARS_PRESENT = 20
MIN_TOTAL_COUNT   = 1000

# Words that MUST be in the vocabulary (shards c, t, w were downloaded)
REQUIRED_WORDS = ["war", "computer", "the", "that", "with"]


# ── Fixtures: load once, share across tests ───────────────────

@pytest.fixture(scope="session")
def clean():
    return pl.read_parquet(CLEAN_PANEL)

@pytest.fixture(scope="session")
def meta():
    return pl.read_parquet(VOCAB_META)

@pytest.fixture(scope="session")
def smoothed():
    return pl.read_parquet(SMOOTHED)


# ══════════════════════════════════════════════════════════════
# Phase 2 — clean_panel.parquet
# ══════════════════════════════════════════════════════════════

class TestPhase2CleanPanel:

    def test_file_exists(self):
        assert os.path.isfile(CLEAN_PANEL), f"Missing: {CLEAN_PANEL}"

    def test_schema(self, clean):
        expected = {"word", "year", "count", "total_tokens", "frequency"}
        assert set(clean.columns) == expected

    def test_no_nulls(self, clean):
        nulls = clean.null_count().row(0)
        assert all(n == 0 for n in nulls), f"Nulls found: {dict(zip(clean.columns, nulls))}"

    def test_row_count_reasonable(self, clean):
        # We have 70k+ words × up to 209 years — expect millions of rows
        assert len(clean) > 1_000_000, f"Too few rows: {len(clean)}"

    def test_year_range(self, clean):
        yr_min = clean["year"].min()
        yr_max = clean["year"].max()
        assert yr_min >= YEAR_MIN, f"Year too early: {yr_min}"
        assert yr_max <= YEAR_MAX, f"Year too late: {yr_max}"

    def test_counts_positive(self, clean):
        assert clean["count"].min() > 0, "count has zero or negative values"

    def test_total_tokens_positive(self, clean):
        assert clean["total_tokens"].min() > 0, "total_tokens has zero or negative values"

    def test_frequency_between_zero_and_one(self, clean):
        assert clean["frequency"].min() > 0,  "frequency has zero or negative values"
        assert clean["frequency"].max() < 1.0, "frequency >= 1 — normalization error"

    def test_frequency_equals_count_over_total(self, clean):
        # Spot-check: frequency == count / total_tokens within float tolerance
        sample = clean.head(1000)
        recomputed = sample["count"].cast(pl.Float64) / sample["total_tokens"].cast(pl.Float64)
        diff = (sample["frequency"] - recomputed).abs().max()
        assert diff < 1e-12, f"frequency != count/total_tokens, max diff: {diff}"

    def test_required_words_present(self, clean):
        vocab = set(clean["word"].unique().to_list())
        for w in REQUIRED_WORDS:
            assert w in vocab, f"Expected word '{w}' not found in clean_panel"

    def test_no_non_alpha_words(self, clean):
        # All words must be lowercase alphabetic
        bad = clean.filter(~pl.col("word").str.contains(r"^[a-z]+$"))
        assert len(bad) == 0, f"Non-alphabetic words found: {bad['word'].unique().head(5)}"


# ══════════════════════════════════════════════════════════════
# Phase 2 — vocabulary_metadata.parquet
# ══════════════════════════════════════════════════════════════

class TestPhase2VocabMeta:

    def test_file_exists(self):
        assert os.path.isfile(VOCAB_META), f"Missing: {VOCAB_META}"

    def test_schema(self, meta):
        expected = {
            "word", "first_year_seen", "last_year_seen",
            "total_count", "total_years_present",
            "mean_frequency", "sparsity", "is_kept"
        }
        assert set(meta.columns) == expected

    def test_no_nulls(self, meta):
        nulls = meta.null_count().row(0)
        assert all(n == 0 for n in nulls), f"Nulls found: {dict(zip(meta.columns, nulls))}"

    def test_kept_words_meet_thresholds(self, meta):
        kept = meta.filter(pl.col("is_kept"))
        assert kept["total_years_present"].min() >= MIN_YEARS_PRESENT, \
            "A kept word has too few years present"
        assert kept["total_count"].min() >= MIN_TOTAL_COUNT, \
            "A kept word has too few total counts"

    def test_kept_count_matches_clean_panel(self, clean, meta):
        # Number of unique words in clean_panel == number of is_kept words in meta
        n_clean  = clean["word"].n_unique()
        n_kept   = meta.filter(pl.col("is_kept")).shape[0]
        assert n_clean == n_kept, \
            f"clean_panel has {n_clean} unique words but metadata has {n_kept} kept words"

    def test_sparsity_in_range(self, meta):
        assert meta["sparsity"].min() >= 0.0, "Sparsity < 0"
        assert meta["sparsity"].max() <= 1.0, "Sparsity > 1"

    def test_year_bounds(self, meta):
        assert meta["first_year_seen"].min() >= YEAR_MIN
        assert meta["last_year_seen"].max()  <= YEAR_MAX


# ══════════════════════════════════════════════════════════════
# Phase 3 — smoothed_word_series.parquet
# ══════════════════════════════════════════════════════════════

class TestPhase3Smoothed:

    def test_file_exists(self):
        assert os.path.isfile(SMOOTHED), f"Missing: {SMOOTHED}"

    def test_schema(self, smoothed):
        expected = {
            "word", "year",
            "log_freq", "smooth_level", "smooth_drift", "smooth_curvature"
        }
        assert set(smoothed.columns) == expected

    def test_smooth_columns_no_nulls(self, smoothed):
        # smooth_level / drift / curvature are defined for every interpolated year
        for col in ["smooth_level", "smooth_drift", "smooth_curvature"]:
            n = smoothed[col].null_count()
            assert n == 0, f"{col} has {n} nulls"

    def test_log_freq_nulls_are_interpolated_years(self, smoothed):
        # log_freq is NaN for years between observations — that's expected.
        # But it must NOT be null for every row; there must be actual observed values.
        total_rows  = len(smoothed)
        null_count  = smoothed["log_freq"].null_count()
        assert null_count < total_rows, "log_freq is null for every row"
        # Observed fraction should be reasonable (> 50%)
        frac_observed = 1 - null_count / total_rows
        assert frac_observed > 0.5, \
            f"Too many null log_freq rows: {null_count}/{total_rows} ({frac_observed:.1%} observed)"

    def test_year_range(self, smoothed):
        assert smoothed["year"].min() >= YEAR_MIN
        assert smoothed["year"].max() <= YEAR_MAX

    def test_words_subset_of_clean_panel(self, clean, smoothed):
        # Every word in smoothed must exist in clean_panel
        smooth_words = set(smoothed["word"].unique().to_list())
        clean_words  = set(clean["word"].unique().to_list())
        extra = smooth_words - clean_words
        assert len(extra) == 0, f"Words in smoothed but not in clean_panel: {list(extra)[:5]}"

    def test_row_count_reasonable(self, smoothed):
        # 70k words × ~100–209 years each = several million rows
        assert len(smoothed) > 5_000_000, f"Too few rows: {len(smoothed)}"

    def test_log_freq_matches_clean_panel(self, clean, smoothed):
        # For observed years, log_freq should equal log(frequency + 1e-9) from clean_panel
        word = "war"
        clean_war = clean.filter(pl.col("word") == word).sort("year")
        smooth_war = (
            smoothed
            .filter((pl.col("word") == word) & pl.col("log_freq").is_not_null())
            .sort("year")
        )
        # Join on year
        joined = clean_war.join(smooth_war.select(["year", "log_freq"]), on="year", how="inner")
        expected = (joined["frequency"] + 1e-9).log()
        actual   = joined["log_freq"]
        diff = (expected - actual).abs().max()
        assert diff < 1e-6, f"log_freq mismatch for '{word}': max diff {diff}"

    def test_computer_drift_positive_post_1960(self, smoothed):
        # 'computer' rose sharply after 1960 — drift should be positive in that era
        sub = smoothed.filter(
            (pl.col("word") == "computer") &
            (pl.col("year") >= 1960) &
            (pl.col("year") <= 1990)
        )
        assert len(sub) > 0, "'computer' not found in smoothed series"
        mean_drift = sub["smooth_drift"].mean()
        assert mean_drift > 0, \
            f"Expected positive drift for 'computer' 1960-1990, got {mean_drift:.4f}"

    def test_smooth_level_is_finite(self, smoothed):
        has_inf = smoothed.filter(pl.col("smooth_level").is_infinite()).shape[0]
        assert has_inf == 0, f"smooth_level has {has_inf} infinite values"

    def test_smooth_drift_is_finite(self, smoothed):
        has_inf = smoothed.filter(pl.col("smooth_drift").is_infinite()).shape[0]
        assert has_inf == 0, f"smooth_drift has {has_inf} infinite values"

    def test_years_are_integers_no_gaps_per_word(self, smoothed):
        # For a sample of words, check years form a contiguous integer range
        sample_words = ["war", "computer", "the", "that"]
        for w in sample_words:
            years = (
                smoothed
                .filter(pl.col("word") == w)
                .sort("year")["year"]
                .to_list()
            )
            if not years:
                continue
            expected = list(range(years[0], years[-1] + 1))
            assert years == expected, \
                f"Years for '{w}' are not a contiguous range: {years[:10]}..."


# ══════════════════════════════════════════════════════════════
# Phase 4 — word_level_fits.parquet + word_summary_metrics.parquet
# ══════════════════════════════════════════════════════════════

@pytest.fixture(scope="session")
def word_fits():
    return pl.read_parquet(WORD_FITS)

@pytest.fixture(scope="session")
def word_summary():
    return pl.read_parquet(WORD_SUMMARY)


class TestPhase4WordFits:

    def test_file_exists(self):
        assert os.path.isfile(WORD_FITS), f"Missing: {WORD_FITS}"

    def test_schema(self, word_fits):
        expected = {
            "word", "year", "observed_count", "frequency",
            "latent_level", "latent_drift", "curvature",
            "lower_ci", "upper_ci", "local_instability"
        }
        assert set(word_fits.columns) == expected

    def test_every_word_has_209_rows(self, word_fits):
        rows_per_word = word_fits.group_by("word").agg(pl.len().alias("n"))
        bad = rows_per_word.filter(pl.col("n") != 209)
        assert len(bad) == 0, \
            f"{len(bad)} words do not have exactly 209 rows: {bad.head(5)}"

    def test_year_range(self, word_fits):
        assert word_fits["year"].min() == YEAR_MIN
        assert word_fits["year"].max() == YEAR_MAX

    def test_kalman_states_no_nulls(self, word_fits):
        # latent_level, latent_drift, lower_ci, upper_ci must always be non-null
        for col in ["latent_level", "latent_drift", "lower_ci", "upper_ci"]:
            n = word_fits[col].null_count()
            assert n == 0, f"{col} has {n} null values"

    def test_kalman_states_finite(self, word_fits):
        for col in ["latent_level", "latent_drift"]:
            n_inf = word_fits.filter(pl.col(col).is_infinite()).shape[0]
            assert n_inf == 0, f"{col} has {n_inf} infinite values"

    def test_ci_ordering(self, word_fits):
        # lower_ci must always be <= upper_ci
        bad = word_fits.filter(pl.col("lower_ci") > pl.col("upper_ci"))
        assert len(bad) == 0, f"{len(bad)} rows where lower_ci > upper_ci"

    def test_curvature_nan_only_at_first_year(self, word_fits):
        # curvature is NaN only at year 1800 (first year per word)
        nan_rows = word_fits.filter(pl.col("curvature").is_null())
        assert nan_rows["year"].unique().to_list() == [YEAR_MIN], \
            "curvature is null at years other than 1800"

    def test_observed_count_nan_for_missing_years(self, word_fits):
        # observed_count is NaN for unobserved years — must have SOME non-null values
        n_null  = word_fits["observed_count"].null_count()
        n_total = len(word_fits)
        frac_observed = 1 - n_null / n_total
        assert frac_observed > 0.5, \
            f"Too many null observed_count rows: {n_null}/{n_total}"

    def test_word_count_matches_clean_panel(self, clean, word_fits):
        n_clean = clean["word"].n_unique()
        n_fits  = word_fits["word"].n_unique()
        assert n_fits == n_clean, \
            f"word_level_fits has {n_fits} words but clean_panel has {n_clean}"

    def test_computer_latent_drift_positive_post_1960(self, word_fits):
        sub = word_fits.filter(
            (pl.col("word") == "computer") &
            (pl.col("year") >= 1960) &
            (pl.col("year") <= 1990)
        )
        assert len(sub) > 0, "'computer' not found in word_level_fits"
        assert sub["latent_drift"].mean() > 0, \
            f"Expected positive latent_drift for 'computer' 1960-1990"

    def test_row_count_reasonable(self, word_fits):
        # 70k words × 209 years
        assert len(word_fits) > 14_000_000, f"Too few rows: {len(word_fits)}"


class TestPhase4WordSummary:

    def test_file_exists(self):
        assert os.path.isfile(WORD_SUMMARY), f"Missing: {WORD_SUMMARY}"

    def test_schema(self, word_summary):
        expected = {
            "word", "mean_drift", "current_drift", "mean_curvature",
            "current_curvature", "mean_instability", "peak_year",
            "sigma2_obs", "sigma2_level", "sigma2_drift", "aic", "fit_status"
        }
        assert set(word_summary.columns) == expected

    def test_one_row_per_word(self, word_fits, word_summary):
        n_fits    = word_fits["word"].n_unique()
        n_summary = len(word_summary)
        assert n_fits == n_summary, \
            f"word_level_fits has {n_fits} unique words but summary has {n_summary} rows"

    def test_no_nulls(self, word_summary):
        nulls = word_summary.null_count().row(0)
        assert all(n == 0 for n in nulls), \
            f"Nulls found: {dict(zip(word_summary.columns, nulls))}"

    def test_peak_year_in_range(self, word_summary):
        assert word_summary["peak_year"].min() >= YEAR_MIN
        assert word_summary["peak_year"].max() <= YEAR_MAX

    def test_sigma2_values_non_negative(self, word_summary):
        for col in ["sigma2_obs", "sigma2_level", "sigma2_drift"]:
            assert word_summary[col].min() >= 0, f"{col} has negative values"

    def test_fit_status_values(self, word_summary):
        valid_statuses = {"converged", "warn_1", "warn_2", "failed"}
        actual = set(word_summary["fit_status"].unique().to_list())
        unexpected = actual - valid_statuses
        assert len(unexpected) == 0, f"Unexpected fit_status values: {unexpected}"

    def test_convergence_rate(self, word_summary):
        # At least 90% of words should converge cleanly
        n_converged = word_summary.filter(pl.col("fit_status") == "converged").shape[0]
        rate = n_converged / len(word_summary)
        assert rate >= 0.90, f"Convergence rate too low: {rate:.1%}"

    def test_no_hard_failures(self, word_summary):
        n_failed = word_summary.filter(pl.col("fit_status") == "failed").shape[0]
        assert n_failed == 0, f"{n_failed} words had hard fit failures"

    def test_aic_is_finite(self, word_summary):
        n_inf = word_summary.filter(pl.col("aic").is_infinite()).shape[0]
        assert n_inf == 0, f"aic has {n_inf} infinite values"

    def test_sigma2_drift_distribution(self, word_summary):
        # Most words should have near-zero drift noise (stable series)
        # but some should have meaningful drift noise (dynamic words)
        n_stable  = word_summary.filter(pl.col("sigma2_drift") < 1e-6).shape[0]
        n_dynamic = word_summary.filter(pl.col("sigma2_drift") > 1e-6).shape[0]
        assert n_stable  > 1000, "Too few stable words (sigma2_drift < 1e-6)"
        assert n_dynamic > 1000, "Too few dynamic words (sigma2_drift > 1e-6)"
