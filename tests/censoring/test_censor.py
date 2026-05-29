"""Unit tests for simace.censor functions."""

import numpy as np
import pandas as pd
import pytest

from simace.censoring.censor import age_censor, death_censor, run_censor

# ---------------------------------------------------------------------------
# age_censor
# ---------------------------------------------------------------------------


class TestAgeCensor:
    def test_no_censoring_within_window(self):
        t = np.array([30.0, 40.0, 50.0])
        left = np.array([20.0, 20.0, 20.0])
        right = np.array([60.0, 60.0, 60.0])
        t_out, censored = age_censor(t, left, right)
        np.testing.assert_array_equal(t_out, t)
        assert not censored.any()

    def test_left_censoring(self):
        t = np.array([10.0, 5.0, 30.0])
        left = np.array([20.0, 20.0, 20.0])
        right = np.array([60.0, 60.0, 60.0])
        t_out, censored = age_censor(t, left, right)
        assert t_out[0] == 20.0
        assert t_out[1] == 20.0
        assert t_out[2] == 30.0
        assert censored[0]
        assert censored[1]
        assert not censored[2]

    def test_right_censoring(self):
        t = np.array([70.0, 80.0, 50.0])
        left = np.array([20.0, 20.0, 20.0])
        right = np.array([60.0, 60.0, 60.0])
        t_out, censored = age_censor(t, left, right)
        assert t_out[0] == 60.0
        assert t_out[1] == 60.0
        assert t_out[2] == 50.0
        assert censored[0]
        assert censored[1]
        assert not censored[2]

    def test_per_individual_windows(self):
        t = np.array([15.0, 55.0, 90.0])
        left = np.array([10.0, 20.0, 30.0])
        right = np.array([50.0, 60.0, 70.0])
        t_out, censored = age_censor(t, left, right)
        assert t_out[0] == 15.0
        assert not censored[0]
        assert t_out[1] == 55.0
        assert not censored[1]
        assert t_out[2] == 70.0
        assert censored[2]

    def test_output_shapes(self):
        n = 100
        t = np.random.default_rng(0).uniform(0, 100, n)
        left = np.full(n, 20.0)
        right = np.full(n, 80.0)
        t_out, censored = age_censor(t, left, right)
        assert t_out.shape == (n,)
        assert censored.shape == (n,)
        assert censored.dtype == bool


# ---------------------------------------------------------------------------
# death_censor
# ---------------------------------------------------------------------------


class TestDeathCensor:
    def test_output_shapes(self):
        t = np.random.default_rng(0).uniform(10, 100, 200)
        t_out, censored = death_censor(t.copy(), seed=42)
        assert t_out.shape == (200,)
        assert censored.shape == (200,)
        assert censored.dtype == bool

    def test_deterministic_with_same_seed(self):
        t = np.array([50.0, 60.0, 70.0, 80.0])
        t1, c1 = death_censor(t.copy(), seed=42)
        t2, c2 = death_censor(t.copy(), seed=42)
        np.testing.assert_array_equal(t1, t2)
        np.testing.assert_array_equal(c1, c2)

    def test_censored_times_are_death_ages(self):
        """For censored individuals, observed time should be <= original time."""
        rng = np.random.default_rng(0)
        t_original = rng.uniform(10, 100, 1000)
        t_copy = t_original.copy()
        t_out, censored = death_censor(t_copy, seed=42, scale=100.0, rho=5)
        # Censored individuals: observed time should be less than original
        assert np.all(t_out[censored] <= t_original[censored])

    def test_uncensored_times_unchanged(self):
        """For uncensored individuals, time should remain the same."""
        rng = np.random.default_rng(0)
        t_original = rng.uniform(10, 100, 1000)
        t_copy = t_original.copy()
        t_out, censored = death_censor(t_copy, seed=42)
        np.testing.assert_array_equal(t_out[~censored], t_original[~censored])


# ---------------------------------------------------------------------------
# run_censor integration tests
# ---------------------------------------------------------------------------


class TestRunCensor:
    @pytest.fixture
    def raw_phenotype(self):
        """Create a minimal raw phenotype DataFrame matching run_phenotype output."""
        rng = np.random.default_rng(42)
        n = 200
        return pd.DataFrame(
            {
                "id": np.arange(n),
                "generation": np.repeat([0, 1, 2, 3], n // 4),
                "sex": rng.integers(0, 2, n),
                "household_id": np.arange(n),
                "mother": np.full(n, -1),
                "father": np.full(n, -1),
                "twin": np.full(n, -1),
                "A1": rng.standard_normal(n),
                "C1": rng.standard_normal(n),
                "E1": rng.standard_normal(n),
                "liability1": rng.standard_normal(n),
                "A2": rng.standard_normal(n),
                "C2": rng.standard_normal(n),
                "E2": rng.standard_normal(n),
                "liability2": rng.standard_normal(n),
                "t1": rng.uniform(10, 200, n),
                "t2": rng.uniform(10, 200, n),
            }
        )

    @pytest.fixture
    def censor_params(self):
        return {
            "censor_age": 80,
            "seed": 42,
            "gen_censoring": {0: [40, 80], 1: [0, 80], 2: [0, 80], 3: [0, 45]},
            "death_scale": 163.265,
            "death_rho": 2.73,
        }

    def test_output_columns(self, raw_phenotype, censor_params):
        result = run_censor(raw_phenotype, **censor_params)
        expected_new = {
            "death_age",
            "age_censored1",
            "t_observed1",
            "death_censored1",
            "affected1",
            "age_censored2",
            "t_observed2",
            "death_censored2",
            "affected2",
        }
        assert expected_new.issubset(set(result.columns))

    def test_preserves_input_columns(self, raw_phenotype, censor_params):
        result = run_censor(raw_phenotype, **censor_params)
        for col in raw_phenotype.columns:
            assert col in result.columns
            np.testing.assert_array_equal(result[col].values, raw_phenotype[col].values)

    def test_affected_consistency(self, raw_phenotype, censor_params):
        """affected should be True only when neither age- nor death-censored."""
        result = run_censor(raw_phenotype, **censor_params)
        for trait in ["1", "2"]:
            affected = result[f"affected{trait}"].values
            age_cens = result[f"age_censored{trait}"].values
            death_cens = result[f"death_censored{trait}"].values
            np.testing.assert_array_equal(affected, ~age_cens & ~death_cens)

    def test_deterministic_with_seed(self, raw_phenotype, censor_params):
        r1 = run_censor(raw_phenotype, **censor_params)
        r2 = run_censor(raw_phenotype, **censor_params)
        np.testing.assert_array_equal(r1["t_observed1"].values, r2["t_observed1"].values)
        np.testing.assert_array_equal(r1["t_observed2"].values, r2["t_observed2"].values)
        np.testing.assert_array_equal(r1["death_age"].values, r2["death_age"].values)

    def test_row_count_unchanged(self, raw_phenotype, censor_params):
        result = run_censor(raw_phenotype, **censor_params)
        assert len(result) == len(raw_phenotype)


# ---------------------------------------------------------------------------
# Stage edges
# ---------------------------------------------------------------------------


class TestRunCensorGenDefaults:
    """Per-generation window defaulting and edge cases."""

    @pytest.fixture
    def phenotype_3gen(self):
        n_per_gen = 50
        n = 3 * n_per_gen
        rng = np.random.default_rng(0)
        return pd.DataFrame(
            {
                "id": np.arange(n),
                "generation": np.repeat([0, 1, 2], n_per_gen),
                "sex": rng.integers(0, 2, n),
                "household_id": np.arange(n),
                "mother": np.full(n, -1),
                "father": np.full(n, -1),
                "twin": np.full(n, -1),
                "A1": rng.standard_normal(n),
                "C1": rng.standard_normal(n),
                "E1": rng.standard_normal(n),
                "liability1": rng.standard_normal(n),
                "A2": rng.standard_normal(n),
                "C2": rng.standard_normal(n),
                "E2": rng.standard_normal(n),
                "liability2": rng.standard_normal(n),
                # Onset times in (30, 75) — within the default [0, censor_age=80] window
                # for generations 1 and 2, but outside [40, 80] for left-censorable gen 0.
                "t1": rng.uniform(30.0, 75.0, n),
                "t2": rng.uniform(30.0, 75.0, n),
            }
        )

    def test_missing_generation_uses_default_window(self, phenotype_3gen):
        """Generations absent from gen_censoring must fall back to [0, censor_age]."""
        result = run_censor(
            phenotype_3gen,
            censor_age=80.0,
            seed=42,
            gen_censoring={0: [40.0, 80.0]},  # only gen 0 specified
            death_scale=1e9,  # disable death censoring so we isolate age-censoring
            death_rho=10.0,
        )
        # All gen-1 and gen-2 onsets are between 30 and 75 — strictly within
        # [0, 80] — so the age-censoring mask must be all-False there.
        for gen in (1, 2):
            mask = result["generation"] == gen
            assert not result.loc[mask, "age_censored1"].any()
            assert not result.loc[mask, "age_censored2"].any()
        # Gen-0 onsets between 30 and 40 are left-censored under [40, 80] → some censored.
        gen0 = result["generation"] == 0
        assert result.loc[gen0, "age_censored1"].any()

    def test_zero_width_window_fully_censors(self, phenotype_3gen):
        """A zero-width window [a, a] flags every member of the generation."""
        result = run_censor(
            phenotype_3gen,
            censor_age=80.0,
            seed=42,
            gen_censoring={0: [80.0, 80.0]},
            death_scale=1e9,
            death_rho=10.0,
        )
        gen0 = result["generation"] == 0
        assert result.loc[gen0, "age_censored1"].all()
        assert result.loc[gen0, "age_censored2"].all()


class TestDeathCensorNoMutation:
    """``death_censor`` must not mutate the caller's input array."""

    def test_input_unchanged(self):
        t_in = np.array([50.0, 60.0, 70.0, 80.0, 200.0, 250.0])
        snapshot = t_in.copy()
        t_out, censored = death_censor(t_in, seed=42, scale=50.0, rho=10)
        np.testing.assert_array_equal(t_in, snapshot)
        assert t_out is not t_in
        # Vacuous otherwise: confirm the test setup actually exercises the mutation site.
        assert censored.any()


class TestRunCensorCLI:
    """End-to-end CLI for ``simace.censoring.censor:cli``."""

    @staticmethod
    def _run_cli(monkeypatch, argv):
        import sys

        from simace.censoring.censor import cli as censor_cli

        monkeypatch.setattr(sys, "argv", ["censor", *argv])
        censor_cli()

    def test_cli_gen_censoring_json_round_trip(self, tmp_path, monkeypatch, tiny_phenotype_parquet):
        out_path = tmp_path / "censored.parquet"
        self._run_cli(
            monkeypatch,
            [
                "--phenotype",
                str(tiny_phenotype_parquet),
                "--output",
                str(out_path),
                "--seed",
                "42",
                "--censor-age",
                "80",
                "--gen-censoring",
                '{"0": [40, 80], "1": [0, 45]}',
            ],
        )
        assert out_path.exists()
        result = pd.read_parquet(out_path)
        # JSON keys parsed as ints → gen-1 must use the [0, 45] window.
        # Anything in gen 1 with onset > 45 should be age-censored.
        gen1 = result["generation"] == 1
        if gen1.any():
            high_onset = result.loc[gen1, "t1"] > 45
            # All gen-1 rows with t1 > 45 are age-censored.
            assert (result.loc[gen1 & high_onset, "age_censored1"]).all()
