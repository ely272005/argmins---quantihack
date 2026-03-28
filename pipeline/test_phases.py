"""
Sanity tests for Phase 2 (clean), Phase 3 (smooth), Phase 4 (word model),
Phase 5 (changepoints), and Phase 6 (regime labeling) outputs.

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
CLEAN_PANEL   = "data/processed/clean_panel.parquet"
VOCAB_META    = "data/processed/vocabulary_metadata.parquet"
SMOOTHED      = "data/processed/smoothed_word_series.parquet"
WORD_FITS     = "data/processed/word_level_fits.parquet"
WORD_SUMMARY  = "data/processed/word_summary_metrics.parquet"
CHANGEPOINTS         = "data/processed/changepoints.parquet"
DENSITY              = "data/processed/changepoint_density.parquet"
REGIMES              = "data/processed/regimes.parquet"
FACTOR_TRAJECTORIES  = "data/processed/factor_trajectories.parquet"
FACTOR_LOADINGS      = "data/processed/factor_loadings.parquet"
FACTOR_METADATA_JSON = "data/processed/factor_metadata.json"
LII_INDEX            = "data/processed/language_instability_index.parquet"
HISTORICAL_EVENTS_JSON = "data/processed/historical_events.json"
EVENT_ALIGNMENT_JSON   = "data/processed/event_alignment.json"

OUTPUTS_LII_JSON     = "outputs/lii.json"
OUTPUTS_DENSITY_JSON = "outputs/changepoint_density.json"
OUTPUTS_FACTORS_JSON = "outputs/factor_trajectories.json"
OUTPUTS_EVENTS_JSON  = "outputs/events.json"
OUTPUTS_WORD_INDEX   = "outputs/word_index.json"
OUTPUTS_WORDS_DIR    = "outputs/words"
EVAL_SUMMARY         = "outputs/eval_summary.json"
WORD_MAP_JSON        = "outputs/word_map.json"
INDEX_HTML           = "index.html"
SERVER_JS            = "server.js"

YEAR_MIN = 1800
YEAR_MAX = 2019
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


# ══════════════════════════════════════════════════════════════
# Phase 5 — changepoints.parquet + changepoint_density.parquet
# ══════════════════════════════════════════════════════════════

@pytest.fixture(scope="session")
def changepoints():
    return pl.read_parquet("data/processed/changepoints.parquet")

@pytest.fixture(scope="session")
def density():
    return pl.read_parquet("data/processed/changepoint_density.parquet")


class TestPhase5Changepoints:

    def test_file_exists(self):
        assert os.path.isfile("data/processed/changepoints.parquet")

    def test_schema(self, changepoints):
        expected = {"word", "changepoint_year", "signal", "penalty"}
        assert set(changepoints.columns) == expected

    def test_no_nulls(self, changepoints):
        nulls = changepoints.null_count().row(0)
        assert all(n == 0 for n in nulls), \
            f"Nulls found: {dict(zip(changepoints.columns, nulls))}"

    def test_changepoint_years_in_range(self, changepoints):
        assert changepoints["changepoint_year"].min() >= YEAR_MIN
        assert changepoints["changepoint_year"].max() <= YEAR_MAX

    def test_signal_column_value(self, changepoints):
        vals = changepoints["signal"].unique().to_list()
        assert vals == ["latent_level"], f"Unexpected signal values: {vals}"

    def test_not_empty(self, changepoints):
        assert len(changepoints) > 0, "changepoints table is empty"

    def test_war_has_changepoints(self, changepoints):
        war_cps = changepoints.filter(pl.col("word") == "war")
        assert len(war_cps) >= 1, "'war' has no detected changepoints"

    def test_computer_has_changepoints(self, changepoints):
        comp_cps = changepoints.filter(pl.col("word") == "computer")
        assert len(comp_cps) >= 1, "'computer' has no detected changepoints"

    def test_war_captures_wwi_or_wwii(self, changepoints):
        # war must have a changepoint in the 1910–1950 window
        war_cps = changepoints.filter(
            (pl.col("word") == "war") &
            (pl.col("changepoint_year") >= 1910) &
            (pl.col("changepoint_year") <= 1950)
        )
        assert len(war_cps) >= 1, \
            "No changepoint for 'war' in WWI/WWII window (1910-1950)"

    def test_no_duplicate_word_year(self, changepoints):
        dupes = changepoints.group_by(["word", "changepoint_year"]).agg(
            pl.len().alias("n")
        ).filter(pl.col("n") > 1)
        assert len(dupes) == 0, \
            f"{len(dupes)} duplicate (word, changepoint_year) pairs found"

    def test_words_in_vocabulary(self, clean, changepoints):
        cp_words  = set(changepoints["word"].unique().to_list())
        vocab     = set(clean["word"].unique().to_list())
        extra = cp_words - vocab
        assert len(extra) == 0, \
            f"Words in changepoints not in vocabulary: {list(extra)[:5]}"

    def test_penalty_column_consistent(self, changepoints):
        # All rows should have the same penalty value (one run = one penalty)
        n_unique_penalties = changepoints["penalty"].n_unique()
        assert n_unique_penalties == 1, \
            f"Expected 1 unique penalty, found {n_unique_penalties}"


class TestPhase5Density:

    def test_file_exists(self):
        assert os.path.isfile("data/processed/changepoint_density.parquet")

    def test_schema(self, density):
        expected = {"year", "n_changepoints", "normalized_density"}
        assert set(density.columns) == expected

    def test_exactly_209_rows(self, density):
        assert len(density) == 209, \
            f"Expected 209 rows (1800-2008), got {len(density)}"

    def test_no_nulls(self, density):
        nulls = density.null_count().row(0)
        assert all(n == 0 for n in nulls), \
            f"Nulls found: {dict(zip(density.columns, nulls))}"

    def test_year_range(self, density):
        assert density["year"].min() == YEAR_MIN
        assert density["year"].max() == YEAR_MAX

    def test_n_changepoints_non_negative(self, density):
        assert density["n_changepoints"].min() >= 0

    def test_normalized_density_in_range(self, density):
        assert density["normalized_density"].min() >= 0.0
        assert density["normalized_density"].max() <= 1.0

    def test_density_sums_to_total_changepoints(self, changepoints, density):
        total_from_density     = density["n_changepoints"].sum()
        total_from_changepoints = len(changepoints)
        assert total_from_density == total_from_changepoints, (
            f"Density sum ({total_from_density}) != "
            f"changepoints table length ({total_from_changepoints})"
        )

    def test_peak_year_not_at_boundary(self, density):
        peak_year = density.sort("n_changepoints", descending=True)["year"][0]
        assert peak_year not in (YEAR_MIN, YEAR_MAX), \
            f"Peak changepoint density is at boundary year {peak_year}"

    def test_historical_years_above_median(self, density):
        # WWI onset (1914) and WWII onset (1939) should be above the median.
        # PELT detects regime changes at conflict onset, not necessarily at end.
        median_density = density["n_changepoints"].median()
        for yr in [1914, 1939]:
            row = density.filter(pl.col("year") == yr)
            assert len(row) == 1
            val = row["n_changepoints"][0]
            assert val > median_density, \
                f"Year {yr} has n_changepoints={val} <= median={median_density}"


# ── Phase 6: Regime Labeling ──────────────────────────────────────────────────

class TestPhase6Regimes:
    """Validates data/processed/regimes.parquet produced by pipeline/06_regimes.py."""

    VALID_LABELS = {"adoption", "decline", "turbulent", "stable"}

    @pytest.fixture(scope="class")
    def regimes(self):
        return pl.read_parquet(REGIMES)

    @pytest.fixture(scope="class")
    def word_fits(self):
        return pl.read_parquet(WORD_FITS)

    @pytest.fixture(scope="class")
    def changepoints(self):
        return pl.read_parquet(CHANGEPOINTS)

    # ── Structural tests ──────────────────────────────────────

    def test_file_exists(self):
        assert os.path.isfile(REGIMES), f"Missing: {REGIMES}"

    def test_schema(self, regimes):
        required = {"word", "year", "regime_label", "z_drift", "z_instability"}
        missing = required - set(regimes.columns)
        assert len(missing) == 0, f"Missing columns: {missing}"

    def test_no_nulls_in_core_columns(self, regimes):
        for col in ["word", "year", "regime_label", "z_drift", "z_instability"]:
            n = regimes[col].null_count()
            assert n == 0, f"Column '{col}' has {n} null values"

    def test_labels_are_valid(self, regimes):
        actual = set(regimes["regime_label"].unique().to_list())
        unexpected = actual - self.VALID_LABELS
        assert len(unexpected) == 0, f"Unexpected regime labels: {unexpected}"

    # ── Shape / uniqueness tests ──────────────────────────────

    def test_shape_matches_word_level_fits(self, regimes, word_fits):
        assert len(regimes) == len(word_fits), (
            f"regimes has {len(regimes):,} rows, "
            f"word_level_fits has {len(word_fits):,}"
        )
        assert regimes["word"].n_unique() == word_fits["word"].n_unique(), (
            "Word count mismatch between regimes and word_level_fits"
        )

    def test_no_duplicate_word_year(self, regimes):
        dupes = (
            regimes.group_by(["word", "year"])
                   .agg(pl.len().alias("n"))
                   .filter(pl.col("n") > 1)
        )
        assert len(dupes) == 0, f"{len(dupes)} duplicate (word, year) pairs found"

    def test_all_words_present(self, regimes, word_fits):
        regime_words = set(regimes["word"].unique().to_list())
        fits_words   = set(word_fits["word"].unique().to_list())
        missing = fits_words - regime_words
        assert len(missing) == 0, (
            f"{len(missing)} words in word_level_fits missing from regimes: "
            f"{sorted(missing)[:5]}"
        )

    def test_year_range(self, regimes):
        assert regimes["year"].min() == YEAR_MIN
        assert regimes["year"].max() == YEAR_MAX

    # ── Distribution tests ────────────────────────────────────

    def test_all_four_regimes_present_and_non_degenerate(self, regimes):
        """All four labels must be present; none < 3% or > 80%."""
        n = len(regimes)
        counts = (
            regimes.group_by("regime_label")
                   .agg(pl.len().alias("n"))
        )
        labels_found = set(counts["regime_label"].to_list())
        assert labels_found == self.VALID_LABELS, (
            f"Not all four regimes present: found {labels_found}"
        )
        for row in counts.iter_rows(named=True):
            frac = row["n"] / n
            assert frac >= 0.03, (
                f"Regime '{row['regime_label']}' is degenerate: {frac:.1%}"
            )
            assert frac <= 0.80, (
                f"Regime '{row['regime_label']}' dominates: {frac:.1%}"
            )

    # ── Numerical quality tests ───────────────────────────────

    def test_z_scores_are_finite(self, regimes):
        """z_drift and z_instability must not contain Inf or IEEE NaN."""
        for col in ["z_drift", "z_instability"]:
            n_inf = regimes.filter(pl.col(col).is_infinite()).shape[0]
            assert n_inf == 0, f"Column '{col}' has {n_inf} infinite values"
        for col in ["z_drift", "z_instability"]:
            n_nan = regimes.filter(pl.col(col).is_nan()).shape[0]
            assert n_nan == 0, \
                f"Column '{col}' has {n_nan} IEEE NaN values (distinct from null)"

    # ── Historical sanity tests ───────────────────────────────

    def test_computer_adoption_1940_1980(self, regimes):
        """'computer' should be adoption for the majority of years 1940–1980
        (z_drift for computer is 7–15 in this era, comfortably above threshold)."""
        sub = regimes.filter(
            (pl.col("word") == "computer") &
            (pl.col("year") >= 1940) &
            (pl.col("year") <= 1980)
        )
        assert len(sub) > 0, "'computer' not found in regimes"
        n_adoption = sub.filter(pl.col("regime_label") == "adoption").shape[0]
        frac = n_adoption / len(sub)
        assert frac >= 0.50, (
            f"Expected ≥50% adoption for 'computer' 1940-1980, got {frac:.1%}"
        )

    def test_telegraph_decline_1941_1980(self, regimes):
        """'telegraph' should be decline for the majority of years 1941–1980
        (technology obsolescence after telephone/radio displaced it)."""
        sub = regimes.filter(
            (pl.col("word") == "telegraph") &
            (pl.col("year") >= 1941) &
            (pl.col("year") <= 1980)
        )
        assert len(sub) > 0, "'telegraph' not found in regimes"
        n_decline = sub.filter(pl.col("regime_label") == "decline").shape[0]
        frac = n_decline / len(sub)
        assert frac >= 0.50, (
            f"Expected ≥50% decline for 'telegraph' 1941-1980, got {frac:.1%}"
        )

    def test_cholera_adoption_1820_1870(self, regimes):
        """'cholera' should be strongly adoption 1820–1870, matching 19th-century
        epidemic waves that drove the word into widespread use."""
        sub = regimes.filter(
            (pl.col("word") == "cholera") &
            (pl.col("year") >= 1820) &
            (pl.col("year") <= 1870)
        )
        assert len(sub) > 0, "'cholera' not found in regimes"
        n_adoption = sub.filter(pl.col("regime_label") == "adoption").shape[0]
        frac = n_adoption / len(sub)
        assert frac >= 0.80, (
            f"Expected ≥80% adoption for 'cholera' 1820-1870, got {frac:.1%}"
        )

    def test_changepoint_proximity_raises_turbulent_rate(self, regimes, changepoints):
        """Rows within ±3 years of any changepoint should have a higher turbulent
        fraction than rows far from changepoints."""
        PROX = 3
        frames = [
            changepoints.select(["word", pl.col("changepoint_year").alias("year")])
                        .with_columns((pl.col("year") + offset).alias("year"))
            for offset in range(-PROX, PROX + 1)
        ]
        cp_set = pl.concat(frames).unique()

        near = regimes.join(cp_set, on=["word", "year"], how="semi")
        far  = regimes.join(cp_set, on=["word", "year"], how="anti")

        turb_near = near.filter(pl.col("regime_label") == "turbulent").shape[0] / len(near)
        turb_far  = far.filter(pl.col("regime_label") == "turbulent").shape[0] / len(far)

        assert turb_near > turb_far, (
            f"Turbulent rate near changepoints ({turb_near:.3f}) is not greater "
            f"than baseline ({turb_far:.3f})"
        )


# ── Phase 7: System-Wide Latent Factor Model ──────────────────────────────────

class TestPhase7FactorModel:
    """Validates the three outputs of pipeline/07_factor_model.py."""

    @pytest.fixture(scope="class")
    def trajectories(self):
        return pl.read_parquet(FACTOR_TRAJECTORIES)

    @pytest.fixture(scope="class")
    def loadings(self):
        return pl.read_parquet(FACTOR_LOADINGS)

    @pytest.fixture(scope="class")
    def meta(self):
        import json as _json
        with open(FACTOR_METADATA_JSON) as f:
            return _json.load(f)

    @pytest.fixture(scope="class")
    def word_fits(self):
        return pl.read_parquet(WORD_FITS, columns=["word"]).unique()

    # ── a. Structural ─────────────────────────────────────────

    def test_files_exist(self):
        assert os.path.isfile(FACTOR_TRAJECTORIES),  f"Missing: {FACTOR_TRAJECTORIES}"
        assert os.path.isfile(FACTOR_LOADINGS),      f"Missing: {FACTOR_LOADINGS}"
        assert os.path.isfile(FACTOR_METADATA_JSON), f"Missing: {FACTOR_METADATA_JSON}"

    def test_trajectories_schema(self, trajectories, meta):
        n = meta["n_factors"]
        expected = {"year"} | {f"factor_{i + 1}" for i in range(n)}
        actual = set(trajectories.columns)
        assert actual == expected, (
            f"Trajectory columns {actual} != expected {expected}"
        )

    def test_loadings_schema(self, loadings, meta):
        n = meta["n_factors"]
        expected = {"word"} | {f"factor_{i + 1}" for i in range(n)}
        actual = set(loadings.columns)
        assert actual == expected, (
            f"Loading columns {actual} != expected {expected}"
        )

    def test_no_nulls(self, trajectories, loadings):
        for col in trajectories.columns:
            n = trajectories[col].null_count()
            assert n == 0, f"factor_trajectories['{col}'] has {n} null values"
        for col in loadings.columns:
            n = loadings[col].null_count()
            assert n == 0, f"factor_loadings['{col}'] has {n} null values"

    # ── b. Shape ──────────────────────────────────────────────

    def test_trajectory_row_count(self, trajectories):
        assert len(trajectories) == 209, (
            f"Expected 209 rows in factor_trajectories, got {len(trajectories)}"
        )

    def test_trajectory_year_range(self, trajectories):
        years = sorted(trajectories["year"].to_list())
        assert years[0]  == YEAR_MIN, f"First year = {years[0]}, expected {YEAR_MIN}"
        assert years[-1] == YEAR_MAX, f"Last year = {years[-1]}, expected {YEAR_MAX}"
        assert years == list(range(YEAR_MIN, YEAR_MAX + 1)), \
            "Year sequence in factor_trajectories has gaps"

    def test_loadings_row_count(self, loadings, word_fits, meta):
        # Loadings must equal meta n_words (degenerate words may have been dropped)
        assert len(loadings) == meta["n_words"], (
            f"factor_loadings has {len(loadings)} rows but meta.n_words = {meta['n_words']}"
        )
        # And it must be close to the Phase 4 word count (degenerate drops allowed)
        n_fits = len(word_fits)
        assert len(loadings) >= n_fits - 250, (
            f"Too many words dropped: loadings={len(loadings)}, "
            f"word_fits={n_fits} (allowed ≤250 degenerate drops)"
        )

    # ── c. Mathematical properties ────────────────────────────

    def test_explained_variance_ordering(self, meta):
        evr = meta["explained_variance_ratio"]
        for i in range(len(evr) - 1):
            assert evr[i] >= evr[i + 1] - 1e-9, (
                f"EVR not descending: factor_{i+1}={evr[i]:.5f} < "
                f"factor_{i+2}={evr[i+1]:.5f}"
            )
        total = sum(evr)
        assert total <= 1.0 + 1e-6, f"EVR sum {total:.6f} > 1.0"
        assert total > 0.0, "EVR sum is zero"

    def test_trajectory_orthogonality(self, trajectories, meta):
        """PCA scores are orthogonal: off-diagonal elements of F^T F should be
        negligible relative to diagonal elements."""
        n = meta["n_factors"]
        factor_cols = [f"factor_{i + 1}" for i in range(n)]
        F = trajectories.select(factor_cols).to_numpy()   # (209, k)
        gram = F.T @ F                                      # (k, k)
        diag = np.diag(gram)
        off  = gram - np.diag(diag)
        ratio = np.abs(off).max() / np.abs(diag).mean()
        assert ratio < 0.01, (
            f"Factor trajectories not orthogonal: "
            f"max_off_diag / mean_diag = {ratio:.4f} (threshold: 0.01)"
        )

    def test_loading_unit_norms(self, loadings, meta):
        """Each factor's loading vector should be unit-norm (PCA convention)."""
        n = meta["n_factors"]
        for i in range(n):
            col  = f"factor_{i + 1}"
            vals = loadings[col].to_numpy()
            norm = float(np.sqrt((vals ** 2).sum()))
            assert abs(norm - 1.0) < 1e-3, (
                f"factor_{i+1} loading norm = {norm:.6f}, expected 1.0. "
                "Loadings may have been scaled by singular values."
            )

    def test_factor1_dominates(self, meta):
        evr = meta["explained_variance_ratio"]
        assert evr[0] > evr[1], (
            f"factor_1 EVR ({evr[0]:.4f}) ≤ factor_2 EVR ({evr[1]:.4f})"
        )
        assert evr[0] >= 0.02, (
            f"factor_1 explains only {evr[0]:.2%} — suspiciously low. "
            "Check that per-word standardization ran correctly."
        )

    def test_metadata_cumulative_variance_consistent(self, meta):
        evr  = np.array(meta["explained_variance_ratio"])
        cumv = np.array(meta["cumulative_variance"])
        expected = np.cumsum(evr)
        max_diff = np.abs(cumv - expected).max()
        assert max_diff < 1e-7, (
            f"cumulative_variance does not match cumsum(EVR): max diff = {max_diff:.2e}"
        )
        assert abs(meta["total_variance_explained"] - float(cumv[-1])) < 1e-7

    # ── d. Historical / domain sanity ─────────────────────────

    def test_war_warfare_correlated_loadings(self, loadings):
        """'war' and 'warfare' have highly similar drift dynamics — they should
        load on the same side of the dominant factor.
        Note: vocabulary is limited to c/t/w shards, so 'battle' is unavailable."""
        war_rows     = loadings.filter(pl.col("word") == "war")
        warfare_rows = loadings.filter(pl.col("word") == "warfare")
        assert len(war_rows) > 0,     "'war' not found in factor_loadings"
        assert len(warfare_rows) > 0, "'warfare' not found in factor_loadings"
        war_f1     = float(war_rows["factor_1"][0])
        warfare_f1 = float(warfare_rows["factor_1"][0])
        assert war_f1 * warfare_f1 > 0, (
            f"'war' (f1={war_f1:.4f}) and 'warfare' (f1={warfare_f1:.4f}) "
            "have opposite signs on factor_1 — military words should co-move"
        )

    def test_computer_era_cluster(self, loadings):
        """Computer-era words should share the same directional loading on factor_1
        — they all grew in frequency during the same historical period.
        Note: vocabulary is limited to c/t/w shards, so candidates are
        computer, computing, technology, wireless, television."""
        tech_words = ["computer", "computing", "technology", "wireless", "television"]
        present = [
            w for w in tech_words
            if loadings.filter(pl.col("word") == w).shape[0] > 0
        ]
        assert len(present) >= 3, (
            f"Only {len(present)} tech words found in vocabulary: {present}. "
            "Expected ≥3 of: computer, computing, technology, wireless, television"
        )
        signs = [
            np.sign(float(loadings.filter(pl.col("word") == w)["factor_1"][0]))
            for w in present
        ]
        dominant = max(
            sum(1 for s in signs if s > 0),
            sum(1 for s in signs if s < 0),
        )
        assert dominant >= 3, (
            f"Computer-era words do not cluster on factor_1: "
            f"signs = {list(zip(present, signs))}"
        )

    def test_factor1_captures_20th_century_variance(self, trajectories):
        """Factor 1 drift should be more volatile during the WWI/WWII era
        (1914–1945) than during the quieter late-Victorian era (1860–1895).
        This test directly detects the bug of using latent_level instead of
        latent_drift as the PCA signal."""
        f1_all = trajectories.sort("year")
        pre = f1_all.filter(
            (pl.col("year") >= 1860) & (pl.col("year") <= 1895)
        )["factor_1"].to_numpy()
        war = f1_all.filter(
            (pl.col("year") >= 1914) & (pl.col("year") <= 1945)
        )["factor_1"].to_numpy()
        var_pre = float(np.std(pre))
        var_war = float(np.std(war))
        assert var_war > var_pre, (
            f"factor_1 std in 1914–1945 ({var_war:.4f}) ≤ 1860–1895 ({var_pre:.4f}). "
            "This suggests latent_level (not latent_drift) was used as signal — "
            "level has smoother dynamics in the war era than pre-war."
        )

    def test_metadata_signal_is_drift(self, meta):
        assert meta["signal"] == "latent_drift", (
            f"Expected signal='latent_drift', got '{meta['signal']}'"
        )

    def test_metadata_n_words_consistent(self, loadings, meta):
        assert len(loadings) == meta["n_words"], (
            f"factor_loadings has {len(loadings)} rows but meta['n_words']={meta['n_words']}"
        )


# ══════════════════════════════════════════════════════════════
# Phase 8 — language_instability_index.parquet
# ══════════════════════════════════════════════════════════════

class TestPhase8LII:
    """15 tests for the Language Instability Index output from Phase 8.

    Verified data properties (from actual run):
      - 209 rows (1800–2008), 204 valid (5 null: years 1800–1804)
      - LII range: [1.763, 78.889], peak year: 1987
      - CR range: [0.581, 0.980]
      - Mean LII 1860–1895 ≈ 4.26, mean LII 1914–1945 ≈ 10.40,
        mean LII 1970–2008 ≈ 66.83
    """

    @pytest.fixture(scope="class")
    def lii(self):
        return pl.read_parquet(LII_INDEX)

    # ── (a) Structural ─────────────────────────────────────────

    def test_file_exists(self):
        assert os.path.isfile(LII_INDEX), f"Missing: {LII_INDEX}"

    def test_schema(self, lii):
        expected = {
            "year", "lii_value", "concentration_ratio",
            "top_eigenvalue", "broad_vs_narrow_score", "lii_fallback",
        }
        assert set(lii.columns) == expected, (
            f"Unexpected columns: {set(lii.columns) ^ expected}"
        )

    # ── (b) Shape ──────────────────────────────────────────────

    def test_exactly_209_rows(self, lii):
        assert len(lii) == 209, f"Expected 209 rows, got {len(lii)}"

    def test_year_range(self, lii):
        assert int(lii["year"].min()) == 1800
        assert int(lii["year"].max()) == 2008

    def test_no_duplicate_years(self, lii):
        assert lii["year"].n_unique() == 209, "Duplicate years found"

    # ── (c) Null / valid-value pattern ────────────────────────

    def test_null_pattern(self, lii):
        """Exactly the first MIN_PERIODS (5) years should have null lii_value."""
        null_years = (
            lii.filter(pl.col("lii_value").is_null())["year"].sort().to_list()
        )
        assert null_years == list(range(1800, 1805)), (
            f"Unexpected null years: {null_years}"
        )

    def test_sufficient_valid_rows(self, lii):
        n_valid = lii.filter(pl.col("lii_value").is_not_null()).shape[0]
        assert n_valid >= 200, f"Too few valid LII rows: {n_valid}"

    # ── (d) Mathematical properties ───────────────────────────

    def test_lii_value_nonnegative(self, lii):
        valid = lii.filter(pl.col("lii_value").is_not_null())
        neg = valid.filter(pl.col("lii_value") < 0)
        assert len(neg) == 0, f"{len(neg)} rows have negative lii_value"

    def test_concentration_ratio_in_range(self, lii):
        valid = lii.filter(pl.col("concentration_ratio").is_not_null())
        out = valid.filter(
            (pl.col("concentration_ratio") <= 0) | (pl.col("concentration_ratio") > 1.0)
        )
        assert len(out) == 0, (
            f"{len(out)} rows have CR outside (0, 1]: "
            f"{valid['concentration_ratio'].min():.4f} – {valid['concentration_ratio'].max():.4f}"
        )

    def test_top_eigenvalue_lte_lii(self, lii):
        """Top eigenvalue must not exceed trace(Σ) — it is one of its summands."""
        valid = lii.filter(
            pl.col("lii_value").is_not_null() & pl.col("top_eigenvalue").is_not_null()
        )
        violations = valid.filter(
            pl.col("top_eigenvalue") > pl.col("lii_value") + 1e-9
        )
        assert len(violations) == 0, (
            f"{len(violations)} rows where top_eigenvalue > lii_value"
        )

    def test_broad_vs_narrow_consistency(self, lii):
        """broad_vs_narrow_score must equal 1 - concentration_ratio (within 1e-9)."""
        valid = lii.filter(
            pl.col("broad_vs_narrow_score").is_not_null()
            & pl.col("concentration_ratio").is_not_null()
        )
        max_err = float(
            valid.select(
                ((pl.col("broad_vs_narrow_score") - (1.0 - pl.col("concentration_ratio"))).abs())
                .alias("err")
            )["err"].max()
        )
        assert max_err < 1e-9, f"broad_vs_narrow ≠ 1-CR: max_err={max_err:.2e}"

    def test_lii_fallback_nonnegative(self, lii):
        valid = lii.filter(pl.col("lii_fallback").is_not_null())
        neg = valid.filter(pl.col("lii_fallback") < 0)
        assert len(neg) == 0, f"{len(neg)} rows have negative lii_fallback"

    # ── (e) Historical / domain sanity ────────────────────────

    def test_lii_peak_not_at_boundary(self, lii):
        """LII peak should not be at the extreme boundary years."""
        valid = lii.filter(pl.col("lii_value").is_not_null())
        peak_year = int(valid.sort("lii_value", descending=True)["year"][0])
        assert peak_year not in (1800, 2008), (
            f"LII peak at boundary year {peak_year} — suggests edge artefact"
        )

    def test_war_era_more_turbulent_than_pre_war(self, lii):
        """Mean LII 1914–1945 should exceed mean LII 1860–1895.

        Even though the absolute peak is post-1945 (PC era), the WWI/WWII
        period has measurably higher instability than the quieter Victorian era.
        Verified values: 1914–1945 mean ≈ 10.4, 1860–1895 mean ≈ 4.3.
        """
        valid = lii.filter(pl.col("lii_value").is_not_null())
        pre_war = float(
            valid.filter((pl.col("year") >= 1860) & (pl.col("year") <= 1895))
            ["lii_value"].mean()
        )
        war_era = float(
            valid.filter((pl.col("year") >= 1914) & (pl.col("year") <= 1945))
            ["lii_value"].mean()
        )
        assert war_era > pre_war, (
            f"War era LII mean ({war_era:.3f}) ≤ pre-war mean ({pre_war:.3f})"
        )

    def test_late_modern_era_highest(self, lii):
        """Mean LII 1970–2008 should far exceed mean LII 1860–1895.

        The PC/internet era drives massive vocabulary co-movement.
        Verified values: 1970–2008 mean ≈ 66.8, 1860–1895 mean ≈ 4.3.
        """
        valid = lii.filter(pl.col("lii_value").is_not_null())
        pre_war = float(
            valid.filter((pl.col("year") >= 1860) & (pl.col("year") <= 1895))
            ["lii_value"].mean()
        )
        modern = float(
            valid.filter((pl.col("year") >= 1970) & (pl.col("year") <= 2008))
            ["lii_value"].mean()
        )
        assert modern > pre_war * 5, (
            f"Late modern LII mean ({modern:.3f}) is not >> pre-war mean ({pre_war:.3f})"
        )


# ══════════════════════════════════════════════════════════════
# Phase 9 — historical_events.json + event_alignment.json
# ══════════════════════════════════════════════════════════════

class TestPhase9Events:
    """15 tests for Phase 9 event annotation and historical alignment outputs.

    Event catalog: 12 hard-coded events spanning 1800–2008.
    Alignment: per-event LII and CP density elevation ratios vs local pre-event
    baseline (20-year window before event; fallback 1805–1824 for early events).
    """

    VALID_CATEGORIES     = {"conflict", "health", "political", "economic", "technology"}
    REQUIRED_EVENT_NAMES = {"WWI", "WWII", "Computing Revolution"}
    ALIGNMENT_SCHEMA     = {
        "name", "start_year", "end_year", "category",
        "lii_mean_during", "lii_mean_baseline", "lii_elevation_ratio",
        "cp_density_mean_during", "cp_density_mean_baseline", "cp_elevation_ratio",
        "top_adoption_words", "top_decline_words",
        "dominant_factor", "is_lii_elevated", "is_cp_elevated", "aligned",
    }

    @pytest.fixture(scope="class")
    def events(self):
        import json
        with open(HISTORICAL_EVENTS_JSON) as f:
            return json.load(f)

    @pytest.fixture(scope="class")
    def alignment(self):
        import json
        with open(EVENT_ALIGNMENT_JSON) as f:
            return json.load(f)

    # ── (a) Structural ─────────────────────────────────────────

    def test_files_exist(self):
        assert os.path.isfile(HISTORICAL_EVENTS_JSON), f"Missing: {HISTORICAL_EVENTS_JSON}"
        assert os.path.isfile(EVENT_ALIGNMENT_JSON),   f"Missing: {EVENT_ALIGNMENT_JSON}"

    def test_events_is_list_of_dicts(self, events):
        assert isinstance(events, list), "historical_events.json must be a list"
        assert len(events) > 0, "historical_events.json is empty"
        required_keys = {"name", "start_year", "end_year", "category"}
        for ev in events:
            assert isinstance(ev, dict), f"Non-dict entry: {ev}"
            missing = required_keys - ev.keys()
            assert not missing, f"Event missing required keys {missing}: {ev}"

    def test_alignment_schema(self, alignment):
        assert isinstance(alignment, list), "event_alignment.json must be a list"
        assert len(alignment) > 0, "event_alignment.json is empty"
        for rec in alignment:
            missing = self.ALIGNMENT_SCHEMA - rec.keys()
            assert not missing, (
                f"Alignment record for '{rec.get('name', '?')}' missing keys: {missing}"
            )

    # ── (b) Events catalog ─────────────────────────────────────

    def test_event_year_range_valid(self, events):
        for ev in events:
            assert 1800 <= ev["start_year"] <= 2008, (
                f"Event '{ev['name']}': start_year={ev['start_year']} out of [1800, 2008]"
            )
            assert 1800 <= ev["end_year"] <= 2008, (
                f"Event '{ev['name']}': end_year={ev['end_year']} out of [1800, 2008]"
            )

    def test_event_start_before_end(self, events):
        for ev in events:
            assert ev["start_year"] < ev["end_year"], (
                f"Event '{ev['name']}': start_year={ev['start_year']} >= end_year={ev['end_year']}"
            )

    def test_required_events_present(self, events):
        names = {ev["name"] for ev in events}
        missing = self.REQUIRED_EVENT_NAMES - names
        assert not missing, (
            f"Required events missing from catalog: {missing}. Found: {names}"
        )

    def test_valid_categories(self, events):
        for ev in events:
            assert ev["category"] in self.VALID_CATEGORIES, (
                f"Event '{ev['name']}': invalid category '{ev['category']}'. "
                f"Must be one of: {self.VALID_CATEGORIES}"
            )

    # ── (c) Alignment metrics validity ─────────────────────────

    def test_lii_metrics_finite(self, alignment):
        for rec in alignment:
            for field in ("lii_mean_during", "lii_mean_baseline", "lii_elevation_ratio"):
                val = rec[field]
                if val is not None:
                    assert math.isfinite(val), (
                        f"Event '{rec['name']}': {field}={val} is not finite"
                    )

    def test_cp_metrics_finite(self, alignment):
        for rec in alignment:
            for field in ("cp_density_mean_during", "cp_density_mean_baseline",
                          "cp_elevation_ratio"):
                val = rec[field]
                if val is not None:
                    assert math.isfinite(val), (
                        f"Event '{rec['name']}': {field}={val} is not finite"
                    )

    def test_top_words_nonempty_strings(self, alignment):
        for rec in alignment:
            for field in ("top_adoption_words", "top_decline_words"):
                words = rec[field]
                assert isinstance(words, list), (
                    f"Event '{rec['name']}': {field} must be list, got {type(words)}"
                )
                assert len(words) > 0, (
                    f"Event '{rec['name']}': {field} is empty"
                )
                for w in words:
                    assert isinstance(w, str), (
                        f"Event '{rec['name']}': {field} contains non-string: {w!r}"
                    )

    def test_dominant_factor_in_range(self, alignment):
        for rec in alignment:
            df = rec["dominant_factor"]
            assert isinstance(df, int), (
                f"Event '{rec['name']}': dominant_factor must be int, got {type(df)}"
            )
            assert 1 <= df <= 10, (
                f"Event '{rec['name']}': dominant_factor={df} not in {{1..10}}"
            )

    # ── (d) Historical / domain sanity ─────────────────────────

    def test_wwi_lii_elevated(self, alignment):
        """WWI LII elevation vs pre-war baseline (1894–1913) must exceed 1.0.

        War era mean ≈ 10.4 >> Victorian pre-war mean ≈ 4–6."""
        wwi = next((r for r in alignment if r["name"] == "WWI"), None)
        assert wwi is not None, "WWI record not found in alignment"
        ratio = wwi["lii_elevation_ratio"]
        assert ratio is not None, "WWI lii_elevation_ratio is None"
        assert ratio > 1.0, (
            f"WWI lii_elevation_ratio={ratio:.4f} ≤ 1.0 — "
            "expected LII elevated vs pre-war baseline"
        )

    def test_wwii_lii_elevated(self, alignment):
        """WWII LII elevation vs inter-war baseline (1919–1938) must exceed 1.0."""
        wwii = next((r for r in alignment if r["name"] == "WWII"), None)
        assert wwii is not None, "WWII record not found in alignment"
        ratio = wwii["lii_elevation_ratio"]
        assert ratio is not None, "WWII lii_elevation_ratio is None"
        assert ratio > 1.0, (
            f"WWII lii_elevation_ratio={ratio:.4f} ≤ 1.0 — "
            "expected LII elevated vs inter-war baseline"
        )

    def test_computing_revolution_lii_exceeds_wwi(self, alignment):
        """Computing Revolution absolute LII must far exceed WWI absolute LII.

        PC-era mean LII ≈ 66.8 >> WWI mean LII ≈ 10.4.
        Factor 1 (58.3% variance) is dominated by technology vocabulary."""
        wwi = next((r for r in alignment if r["name"] == "WWI"), None)
        cr  = next((r for r in alignment if r["name"] == "Computing Revolution"), None)
        assert wwi is not None, "WWI record not found in alignment"
        assert cr  is not None, "Computing Revolution record not found in alignment"
        assert cr["lii_mean_during"] > wwi["lii_mean_during"], (
            f"Computing Revolution lii_mean_during ({cr['lii_mean_during']:.2f}) "
            f"should exceed WWI ({wwi['lii_mean_during']:.2f})"
        )

    def test_majority_events_aligned(self, alignment):
        """At least 6 of 12 events must have aligned=True (LII or CP elevated)."""
        n_aligned = sum(1 for r in alignment if r["aligned"])
        n_total   = len(alignment)
        assert n_aligned >= 6, (
            f"Only {n_aligned}/{n_total} events aligned — expected ≥ 6. "
            "Check baseline computation or elevation ratio logic."
        )


# ── Phase 10: Frontend-Ready Export ───────────────────────────
class TestPhase10Export:
    """15 tests verifying outputs/ JSON files produced by 10_export.py."""

    # ── Fixtures ──────────────────────────────────────────────

    @pytest.fixture(scope="class")
    def lii_json(self):
        import json
        with open(OUTPUTS_LII_JSON) as f:
            return json.load(f)

    @pytest.fixture(scope="class")
    def density_json(self):
        import json
        with open(OUTPUTS_DENSITY_JSON) as f:
            return json.load(f)

    @pytest.fixture(scope="class")
    def factors_json(self):
        import json
        with open(OUTPUTS_FACTORS_JSON) as f:
            return json.load(f)

    @pytest.fixture(scope="class")
    def events_json(self):
        import json
        with open(OUTPUTS_EVENTS_JSON) as f:
            return json.load(f)

    @pytest.fixture(scope="class")
    def word_index(self):
        import json
        with open(OUTPUTS_WORD_INDEX) as f:
            return json.load(f)

    @pytest.fixture(scope="class")
    def war_word(self):
        import json
        fpath = os.path.join(OUTPUTS_WORDS_DIR, "war.json")
        with open(fpath) as f:
            return json.load(f)

    # ── a. System-level JSON files (7 tests) ──────────────────

    def test_system_files_exist(self):
        """All 5 core output JSON files must exist."""
        for fpath in [OUTPUTS_LII_JSON, OUTPUTS_DENSITY_JSON, OUTPUTS_FACTORS_JSON,
                      OUTPUTS_EVENTS_JSON, OUTPUTS_WORD_INDEX]:
            assert os.path.exists(fpath), f"Missing output file: {fpath}"

    def test_lii_json_row_count(self, lii_json):
        """lii.json must have exactly 209 rows (years 1800–2008 inclusive)."""
        assert len(lii_json) == 209, (
            f"lii.json has {len(lii_json)} rows — expected 209 (1800–2008)"
        )

    def test_lii_json_schema(self, lii_json):
        """Every entry in lii.json must have keys: year, lii, cr, bvn, fallback."""
        required = {"year", "lii", "cr", "bvn", "fallback"}
        for row in lii_json:
            missing = required - set(row.keys())
            assert not missing, f"lii.json row missing keys {missing}: {row}"

    def test_lii_null_pattern(self, lii_json):
        """Years 1800–1804 must have lii=null (not enough history for rolling window)."""
        early = [r for r in lii_json if r["year"] <= 1804]
        assert len(early) == 5, f"Expected 5 early rows, got {len(early)}"
        for row in early:
            assert row["lii"] is None, (
                f"Year {row['year']}: lii={row['lii']!r} — expected null"
            )

    def test_density_json_row_count(self, density_json):
        """changepoint_density.json must have exactly 209 rows."""
        assert len(density_json) == 209, (
            f"density.json has {len(density_json)} rows — expected 209"
        )

    def test_factor_traj_json_schema(self, factors_json):
        """Every entry in factor_trajectories.json must have year + f1..f10."""
        required = {"year"} | {f"f{i}" for i in range(1, 11)}
        for row in factors_json:
            missing = required - set(row.keys())
            assert not missing, f"factor_trajectories.json row missing keys {missing}"

    def test_events_json_consistent(self, events_json):
        """outputs/events.json must contain the same 12 event names as the source file."""
        import json
        with open(HISTORICAL_EVENTS_JSON) as f:
            source = json.load(f)
        src_names = {e["name"] for e in source}
        out_names = {e["name"] for e in events_json}
        assert src_names == out_names, (
            f"Event name mismatch.\n  Source only: {src_names - out_names}\n"
            f"  Output only: {out_names - src_names}"
        )

    # ── b. Word index (4 tests) ────────────────────────────────

    def test_word_index_exists(self, word_index):
        """outputs/word_index.json must exist and be a dict."""
        assert isinstance(word_index, dict), (
            f"word_index.json is type {type(word_index).__name__} — expected dict"
        )

    def test_word_index_min_coverage(self, word_index):
        """Word index must contain at least 50,000 entries."""
        assert len(word_index) >= 50_000, (
            f"word_index has {len(word_index)} entries — expected ≥ 50,000"
        )

    def test_word_index_schema(self, word_index):
        """Entry for 'war' must have keys: peak, drift, regime, ncp."""
        assert "war" in word_index, "'war' not found in word_index"
        entry = word_index["war"]
        required = {"peak", "drift", "regime", "ncp"}
        missing = required - set(entry.keys())
        assert not missing, f"word_index['war'] missing keys {missing}: {entry}"

    def test_word_index_regime_valid(self, word_index):
        """'war' regime must be one of the four valid labels."""
        valid = {"adoption", "decline", "turbulent", "stable"}
        regime = word_index["war"]["regime"]
        assert regime in valid, (
            f"word_index['war']['regime'] = {regime!r} — not in {valid}"
        )

    # ── c. Per-word JSON files (3 tests) ──────────────────────

    def test_words_dir_has_files(self):
        """outputs/words/ must contain at least 100 JSON files."""
        assert os.path.isdir(OUTPUTS_WORDS_DIR), f"Missing directory: {OUTPUTS_WORDS_DIR}"
        files = [f for f in os.listdir(OUTPUTS_WORDS_DIR) if f.endswith(".json")]
        assert len(files) >= 100, (
            f"outputs/words/ has {len(files)} JSON files — expected ≥ 100"
        )

    def test_war_word_file_schema(self, war_word):
        """outputs/words/war.json must have all required top-level keys."""
        required = {"word", "years", "level", "drift", "instab", "regime",
                    "changepoints", "summary"}
        missing = required - set(war_word.keys())
        assert not missing, f"war.json missing keys {missing}"

    def test_war_word_year_coverage(self, war_word):
        """war.json 'years' array must have 209 entries (1800–2008)."""
        n = len(war_word["years"])
        assert n == 209, f"war.json['years'] has {n} entries — expected 209"

    # ── d. Data integrity (1 test) ────────────────────────────

    def test_war_changepoints_match_pipeline(self, war_word):
        """Changepoints in war.json must be a subset of changepoints.parquet for 'war'."""
        cp_df = pl.read_parquet(CHANGEPOINTS).filter(pl.col("word") == "war")
        pipeline_cps = set(cp_df["changepoint_year"].to_list())
        export_cps   = set(war_word["changepoints"])
        extra = export_cps - pipeline_cps
        assert not extra, (
            f"war.json has changepoints {extra} not in changepoints.parquet: "
            f"pipeline={sorted(pipeline_cps)}"
        )


# ── Phase 11: Evaluation and Sanity Checks ────────────────────
class TestPhase11Eval:
    """15 tests verifying outputs/eval_summary.json produced by 11_eval.py."""

    ANCHOR_WORDS = ["war", "computer", "technology", "typewriter", "wireless"]

    # ── Fixture ───────────────────────────────────────────────

    @pytest.fixture(scope="class")
    def summary(self):
        import json
        with open(EVAL_SUMMARY) as f:
            return json.load(f)

    # ── a. Eval summary structure (2 tests) ───────────────────

    def test_eval_summary_exists(self):
        """outputs/eval_summary.json must exist."""
        assert os.path.exists(EVAL_SUMMARY), f"Missing: {EVAL_SUMMARY}"

    def test_eval_summary_top_level_keys(self, summary):
        """Must have all 6 top-level section keys."""
        required = {
            "model_quality", "word_sanity", "changepoint_sanity",
            "factor_interpretability", "lii_sanity", "failures",
        }
        missing = required - set(summary.keys())
        assert not missing, f"eval_summary.json missing top-level keys: {missing}"

    # ── b. Model quality (3 tests) ────────────────────────────

    def test_convergence_rate_high(self, summary):
        """At least 90% of words must have converged fits."""
        rate = summary["model_quality"]["convergence_rate"]
        assert rate >= 0.90, (
            f"convergence_rate={rate:.4f} < 0.90 — "
            "too many failed fits"
        )

    def test_failed_words_is_list(self, summary):
        """model_quality['failed_words'] must be a list."""
        assert isinstance(summary["model_quality"]["failed_words"], list), (
            "failed_words should be a list"
        )

    def test_aic_median_finite(self, summary):
        """AIC median must be a finite number (not None or NaN)."""
        import math
        aic_median = summary["model_quality"]["aic_median"]
        assert aic_median is not None, "aic_median is null"
        assert isinstance(aic_median, (int, float)), (
            f"aic_median type {type(aic_median).__name__} — expected numeric"
        )
        assert math.isfinite(aic_median), f"aic_median={aic_median} is not finite"

    # ── c. Word sanity (4 tests) ──────────────────────────────

    def test_anchor_words_present(self, summary):
        """All 5 anchor words must appear in word_sanity."""
        ws = summary["word_sanity"]
        missing = [w for w in self.ANCHOR_WORDS if w not in ws]
        assert not missing, f"word_sanity missing anchor words: {missing}"

    def test_war_peak_plausible(self, summary):
        """'war' peak year must be in [1905, 1960] (WWI/WWII era)."""
        peak = summary["word_sanity"]["war"]["peak_year"]
        assert 1905 <= peak <= 1960, (
            f"war peak_year={peak} outside [1905, 1960]"
        )

    def test_computer_peak_plausible(self, summary):
        """'computer' peak year must be >= 1960 (computing era)."""
        peak = summary["word_sanity"]["computer"]["peak_year"]
        assert peak >= 1960, (
            f"computer peak_year={peak} < 1960 — expected in computing era"
        )

    def test_anchor_changepoints_nonzero(self, summary):
        """Every anchor word must have at least 1 detected changepoint."""
        ws = summary["word_sanity"]
        for word in self.ANCHOR_WORDS:
            if word not in ws:
                continue
            ncp = ws[word]["n_changepoints"]
            assert ncp > 0, (
                f"'{word}' has n_changepoints={ncp} — expected > 0"
            )

    # ── d. Changepoint sanity (2 tests) ───────────────────────

    def test_top_cp_years_plausible(self, summary):
        """At least 1 of the top 10 CP years must fall in a historically turbulent era."""
        years = summary["changepoint_sanity"]["top_cp_years"]
        plausible = any(
            (1905 <= y <= 1955) or (1965 <= y <= 2005)
            for y in years
        )
        assert plausible, (
            f"None of the top CP years {years} fall in [1905,1955] or [1965,2005]"
        )

    def test_near_event_rate_positive(self, summary):
        """At least one of the top 10 CP years must be near a historical event start."""
        rate = summary["changepoint_sanity"]["near_event_rate"]
        assert rate > 0.0, (
            f"near_event_rate={rate} — no top CP year is within ±5 years of any event start"
        )

    # ── e. Factor sanity (2 tests) ────────────────────────────

    def test_factor1_top_words_nonempty(self, summary):
        """Factor 1 must have a non-empty list of top-loading words."""
        top = summary["factor_interpretability"]["factor_1"]["top_words"]
        assert isinstance(top, list) and len(top) > 0, (
            "factor_1 top_words is empty or not a list"
        )

    def test_factor1_variance_dominant(self, summary):
        """Factor 1 must explain more than 50% of variance."""
        ev = summary["factor_interpretability"]["factor_1"]["explained_variance"]
        assert ev is not None and ev > 0.50, (
            f"factor_1 explained_variance={ev} <= 0.50 — Factor 1 should dominate"
        )

    # ── f. LII sanity (2 tests) ───────────────────────────────

    def test_lii_peak_plausible(self, summary):
        """LII peak year must be in [1960, 2005] (computing/digital era)."""
        peak = summary["lii_sanity"]["peak_year"]
        assert 1960 <= peak <= 2005, (
            f"lii peak_year={peak} outside [1960, 2005]"
        )

    def test_lii_cp_correlation_finite(self, summary):
        """cp_lii_correlation must be a finite float (not NaN or None)."""
        import math
        corr = summary["lii_sanity"]["cp_lii_correlation"]
        assert corr is not None, "cp_lii_correlation is None"
        assert isinstance(corr, float), f"cp_lii_correlation is not a float: {corr!r}"
        assert math.isfinite(corr), f"cp_lii_correlation={corr} is not finite"


# ── Phase 12: Demo and Presentation Layer ─────────────────────
class TestPhase12Demo:
    """12 tests verifying word_map.json, index.html tabs, and server.js endpoint."""

    ANCHOR_WORDS_12 = ["war", "computer", "technology", "wireless"]

    # ── Fixture ───────────────────────────────────────────────

    @pytest.fixture(scope="class")
    def word_map(self):
        import json
        with open(WORD_MAP_JSON) as f:
            return json.load(f)

    # ── a. word_map.json schema (5 tests) ─────────────────────

    def test_word_map_exists(self):
        """outputs/word_map.json must exist."""
        assert os.path.exists(WORD_MAP_JSON), f"Missing: {WORD_MAP_JSON}"

    def test_word_map_min_entries(self, word_map):
        """word_map.json must have at least 100 entries."""
        assert len(word_map) >= 100, (
            f"word_map has {len(word_map)} entries — expected ≥ 100"
        )

    def test_word_map_schema(self, word_map):
        """Every entry must have keys: word, f1, f2, regime, peak, ncp."""
        required = {"word", "f1", "f2", "regime", "peak", "ncp"}
        for entry in word_map[:20]:   # spot-check first 20
            missing = required - set(entry.keys())
            assert not missing, f"word_map entry missing keys {missing}: {entry}"

    def test_word_map_f1_spread(self, word_map):
        """std(f1 values) must be > 0 — entries must have genuine spread (not all identical)."""
        import statistics
        f1_vals = [e["f1"] for e in word_map if e["f1"] is not None]
        assert len(f1_vals) >= 10, "too few finite f1 values"
        std = statistics.stdev(f1_vals)
        assert std > 0, f"f1 std={std:.8f} — word_map has no spread on Factor 1 axis"

    def test_word_map_regime_valid(self, word_map):
        """Every regime value must be one of the four valid labels."""
        valid = {"adoption", "decline", "turbulent", "stable"}
        bad = [e["regime"] for e in word_map if e["regime"] not in valid]
        assert not bad, f"Invalid regimes in word_map: {set(bad)}"

    # ── b. Anchor word coverage (2 tests) ─────────────────────

    def test_anchor_words_in_word_map(self, word_map):
        """All anchor words must appear in word_map (always-included guarantee)."""
        words_in_map = {e["word"] for e in word_map}
        missing = [w for w in self.ANCHOR_WORDS_12 if w not in words_in_map]
        assert not missing, f"Anchor words missing from word_map: {missing}"

    def test_word_map_f1_finite(self, word_map):
        """All f1 values must be finite floats."""
        import math
        bad = [e for e in word_map if e["f1"] is None or not math.isfinite(e["f1"])]
        assert not bad, f"{len(bad)} entries have non-finite f1 values"

    # ── c. index.html structure (3 tests) ─────────────────────

    def test_factor_tab_in_html(self):
        """index.html must include the factor explorer tab."""
        with open(INDEX_HTML) as f:
            html = f.read()
        assert "showTab('factors')" in html, (
            "showTab('factors') not found in index.html — factor tab missing"
        )

    def test_wordmap_tab_in_html(self):
        """index.html must include the word map tab."""
        with open(INDEX_HTML) as f:
            html = f.read()
        assert "showTab('wordmap')" in html, (
            "showTab('wordmap') not found in index.html — word map tab missing"
        )

    def test_chart_canvases_in_html(self):
        """index.html must have both factorChart and wordMapChart canvas elements."""
        with open(INDEX_HTML) as f:
            html = f.read()
        assert 'id="factorChart"' in html,  "factorChart canvas not found in index.html"
        assert 'id="wordMapChart"' in html, "wordMapChart canvas not found in index.html"

    # ── d. server.js (2 tests) ────────────────────────────────

    def test_word_map_api_endpoint(self):
        """server.js must define the /api/word-map endpoint."""
        with open(SERVER_JS) as f:
            js = f.read()
        assert "'/api/word-map'" in js or '"/api/word-map"' in js, (
            "/api/word-map endpoint not found in server.js"
        )

    def test_word_map_json_loadable(self):
        """server.js must reference word_map.json in its file loading logic."""
        with open(SERVER_JS) as f:
            js = f.read()
        assert "word_map.json" in js, (
            "'word_map.json' not found in server.js"
        )
