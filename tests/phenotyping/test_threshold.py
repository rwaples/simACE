"""Unit tests for simace.threshold.apply_threshold and sex-specific prevalence."""

import sys

import numpy as np
import pandas as pd
import pytest

from simace.phenotyping.threshold import (
    _apply_threshold_sex_aware,
    _parse_prevalence_arg,
    apply_threshold,
    run_threshold,
)
from simace.phenotyping.threshold import (
    cli as threshold_cli,
)


class TestApplyThreshold:
    def test_output_is_boolean(self):
        liability = np.random.default_rng(0).standard_normal(1000)
        generation = np.zeros(1000, dtype=int)
        affected = apply_threshold(liability, generation, prevalence=0.1)
        assert affected.dtype == bool

    def test_output_shape(self):
        liability = np.random.default_rng(0).standard_normal(500)
        generation = np.zeros(500, dtype=int)
        affected = apply_threshold(liability, generation, prevalence=0.2)
        assert affected.shape == (500,)

    def test_prevalence_fraction_single_generation(self):
        """Affected fraction should match prevalence within rounding."""
        rng = np.random.default_rng(42)
        n = 10000
        liability = rng.standard_normal(n)
        generation = np.zeros(n, dtype=int)
        prev = 0.15
        affected = apply_threshold(liability, generation, prevalence=prev)
        observed = affected.mean()
        # With percentile-based cutoff, should be very close
        assert abs(observed - prev) < 0.01

    def test_prevalence_per_generation(self):
        """Each generation should independently have ~prevalence fraction affected
        when standardize='per_generation'."""
        rng = np.random.default_rng(42)
        n_per_gen = 5000
        liability = rng.standard_normal(3 * n_per_gen)
        generation = np.repeat([0, 1, 2], n_per_gen)
        prev = 0.10
        affected = apply_threshold(liability, generation, prevalence=prev, standardize="per_generation")
        for gen in [0, 1, 2]:
            mask = generation == gen
            gen_prev = affected[mask].mean()
            assert abs(gen_prev - prev) < 0.01

    def test_higher_liability_more_likely_affected(self):
        """Individuals with higher liability should be affected more often."""
        rng = np.random.default_rng(0)
        n = 10000
        liability = rng.standard_normal(n)
        generation = np.zeros(n, dtype=int)
        affected = apply_threshold(liability, generation, prevalence=0.2)
        # Top 20% by liability should be affected
        high = liability > np.percentile(liability, 80)
        low = liability < np.percentile(liability, 20)
        assert affected[high].mean() > affected[low].mean()

    def test_zero_prevalence_raises(self):
        with pytest.raises(ValueError, match="prevalence"):
            apply_threshold(np.array([1.0, 2.0]), np.array([0, 0]), prevalence=0.0)

    def test_one_prevalence_raises(self):
        with pytest.raises(ValueError, match="prevalence"):
            apply_threshold(np.array([1.0, 2.0]), np.array([0, 0]), prevalence=1.0)

    def test_negative_prevalence_raises(self):
        with pytest.raises(ValueError, match="prevalence"):
            apply_threshold(np.array([1.0, 2.0]), np.array([0, 0]), prevalence=-0.1)

    def test_constant_liability_within_generation(self):
        """If all liabilities are equal, standardized values are 0; threshold is
        ndtri(0.5) = 0, so 0 >= 0 → all affected.  Shouldn't crash."""
        liability = np.full(1000, 5.0)
        generation = np.zeros(1000, dtype=int)
        affected = apply_threshold(liability, generation, prevalence=0.5)
        assert affected.dtype == bool

    def test_standardize_none_prevalence_drifts(self):
        """With standardize='none' and non-unit variance, prevalence drifts from K."""
        rng = np.random.default_rng(42)
        n = 20000
        # Wide distribution: std=2, so threshold ndtri(0.9) ≈ 1.28 is easier to exceed
        liability = rng.normal(0, 2.0, n)
        generation = np.zeros(n, dtype=int)
        affected = apply_threshold(liability, generation, prevalence=0.1, standardize="none")
        observed = affected.mean()
        # With std=2, more people cross the N(0,1) threshold → prevalence > 0.1
        assert observed > 0.15, f"Expected prevalence > 0.15 with wide distribution, got {observed:.3f}"

    def test_standardize_false_aliases_none(self):
        """Legacy bool ``standardize=False`` resolves to ``'none'``."""
        rng = np.random.default_rng(42)
        n = 20000
        liability = rng.normal(0, 2.0, n)
        generation = np.zeros(n, dtype=int)
        a_false = apply_threshold(liability, generation, prevalence=0.1, standardize=False)
        a_none = apply_threshold(liability, generation, prevalence=0.1, standardize="none")
        np.testing.assert_array_equal(a_false, a_none)

    def test_standardize_global_preserves_prevalence_single_cohort(self):
        """With standardize='global' and a single cohort, prevalence matches K."""
        rng = np.random.default_rng(42)
        n = 20000
        liability = rng.normal(0, 2.0, n)
        generation = np.zeros(n, dtype=int)
        affected = apply_threshold(liability, generation, prevalence=0.1, standardize="global")
        observed = affected.mean()
        assert abs(observed - 0.1) < 0.02

    def test_standardize_true_aliases_global(self):
        """Legacy bool ``standardize=True`` resolves to ``'global'``."""
        rng = np.random.default_rng(42)
        n = 20000
        liability = rng.normal(0, 2.0, n)
        generation = np.zeros(n, dtype=int)
        a_true = apply_threshold(liability, generation, prevalence=0.1, standardize=True)
        a_global = apply_threshold(liability, generation, prevalence=0.1, standardize="global")
        np.testing.assert_array_equal(a_true, a_global)

    def test_standardize_per_generation_preserves_per_gen_prevalence(self):
        """With heteroscedastic per-gen liability, only 'per_generation' preserves K per gen."""
        rng = np.random.default_rng(7)
        n_per = 20_000
        # gen 0 std=1, gen 1 std=3
        gen0 = rng.normal(0.0, 1.0, n_per)
        gen1 = rng.normal(0.0, 3.0, n_per)
        liability = np.concatenate([gen0, gen1])
        generation = np.repeat([0, 1], n_per)
        affected = apply_threshold(liability, generation, prevalence=0.1, standardize="per_generation")
        for gen in (0, 1):
            mask = generation == gen
            assert abs(affected[mask].mean() - 0.1) < 0.01

    def test_standardize_global_multi_gen_drifts_when_var_changes(self):
        """Same heteroscedastic input under 'global' yields drifted per-gen prevalence."""
        rng = np.random.default_rng(7)
        n_per = 20_000
        gen0 = rng.normal(0.0, 1.0, n_per)
        gen1 = rng.normal(0.0, 3.0, n_per)
        liability = np.concatenate([gen0, gen1])
        generation = np.repeat([0, 1], n_per)
        affected = apply_threshold(liability, generation, prevalence=0.1, standardize="global")
        gen0_obs = affected[generation == 0].mean()
        gen1_obs = affected[generation == 1].mean()
        # The narrow gen sees fewer cases (its tails are inside the global threshold);
        # the wide gen sees more.
        assert gen0_obs < 0.05
        assert gen1_obs > 0.15

    def test_standardize_invalid_string_raises(self):
        liability = np.array([0.0, 1.0])
        generation = np.array([0, 0])
        with pytest.raises(ValueError, match="standardize must be one of"):
            apply_threshold(liability, generation, prevalence=0.1, standardize="per_gen")


class TestApplyThresholdDictPrevalence:
    def test_dict_prevalence_per_gen_rates(self):
        """Each generation gets its own prevalence from the dict."""
        rng = np.random.default_rng(42)
        n_per_gen = 10000
        liability = rng.standard_normal(3 * n_per_gen)
        generation = np.repeat([0, 1, 2], n_per_gen)
        prev_dict = {0: 0.05, 1: 0.10, 2: 0.20}
        affected = apply_threshold(liability, generation, prevalence=prev_dict)
        for gen, expected_prev in prev_dict.items():
            mask = generation == gen
            observed = affected[mask].mean()
            assert abs(observed - expected_prev) < 0.01, f"gen {gen}: expected ~{expected_prev}, got {observed}"

    def test_dict_prevalence_output_shape_and_dtype(self):
        rng = np.random.default_rng(0)
        n = 500
        liability = rng.standard_normal(2 * n)
        generation = np.repeat([0, 1], n)
        affected = apply_threshold(liability, generation, prevalence={0: 0.1, 1: 0.2})
        assert affected.shape == (2 * n,)
        assert affected.dtype == bool

    def test_dict_missing_generation_raises(self):
        """Dict missing a generation key should raise ValueError."""
        liability = np.array([1.0, 2.0, 3.0, 4.0])
        generation = np.array([0, 0, 1, 1])
        with pytest.raises(ValueError, match="missing entries for generations"):
            apply_threshold(liability, generation, prevalence={0: 0.5})

    def test_dict_prevalence_out_of_range_raises(self):
        """Dict values outside (0,1) should raise ValueError."""
        liability = np.array([1.0, 2.0])
        generation = np.array([0, 0])
        with pytest.raises(ValueError, match="prevalence must be between 0 and 1"):
            apply_threshold(liability, generation, prevalence={0: 0.0})
        with pytest.raises(ValueError, match="prevalence must be between 0 and 1"):
            apply_threshold(liability, generation, prevalence={0: 1.0})
        with pytest.raises(ValueError, match="prevalence must be between 0 and 1"):
            apply_threshold(liability, generation, prevalence={0: -0.1})

    def test_scalar_per_generation_exact(self):
        """Scalar prevalence with standardize='per_generation' hits K exactly per gen."""
        rng = np.random.default_rng(99)
        n = 5000
        liability = rng.standard_normal(n)
        generation = np.repeat([0, 1], n // 2)
        affected = apply_threshold(liability, generation, prevalence=0.15, standardize="per_generation")
        for gen in [0, 1]:
            mask = generation == gen
            observed = affected[mask].mean()
            assert abs(observed - 0.15) < 0.01


# ---------------------------------------------------------------------------
# Sex-specific prevalence via _apply_threshold_sex_aware
# ---------------------------------------------------------------------------


class TestThresholdSexPrevalence:
    def test_sex_specific_case_rates(self):
        """Male and female affected rates should match their respective prevalences."""
        rng = np.random.default_rng(42)
        n = 20000
        liability = rng.standard_normal(n)
        generation = np.zeros(n, dtype=int)
        sex = np.array([0] * (n // 2) + [1] * (n // 2))
        prev_f, prev_m = 0.08, 0.15
        affected = _apply_threshold_sex_aware(
            liability,
            generation,
            sex,
            prevalence={"female": prev_f, "male": prev_m},
        )
        female_rate = affected[: n // 2].mean()
        male_rate = affected[n // 2 :].mean()
        assert abs(female_rate - prev_f) < 0.02
        assert abs(male_rate - prev_m) < 0.02

    def test_scalar_prevalence_unchanged(self):
        """Without sex-specific keys, behaviour is identical to apply_threshold."""
        rng = np.random.default_rng(0)
        n = 5000
        liability = rng.standard_normal(n)
        generation = np.repeat([0, 1], n // 2)
        sex = rng.integers(0, 2, size=n)
        affected_sex_aware = _apply_threshold_sex_aware(
            liability,
            generation,
            sex,
            prevalence=0.15,
        )
        affected_direct = apply_threshold(liability, generation, prevalence=0.15)
        np.testing.assert_array_equal(affected_sex_aware, affected_direct)

    def test_sex_specific_with_per_gen_dict(self):
        """Sex-specific + per-generation dict prevalence should compose under
        ``standardize='per_generation'``."""
        rng = np.random.default_rng(42)
        n_per_gen = 10000
        n = 2 * n_per_gen
        liability = rng.standard_normal(n)
        generation = np.repeat([0, 1], n_per_gen)
        sex = np.tile([0, 1], n // 2)  # alternating male/female

        prev_f = {0: 0.05, 1: 0.10}
        prev_m = {0: 0.10, 1: 0.20}
        affected = _apply_threshold_sex_aware(
            liability,
            generation,
            sex,
            prevalence={"female": prev_f, "male": prev_m},
            standardize="per_generation",
        )

        for gen, exp_f, exp_m in [(0, 0.05, 0.10), (1, 0.10, 0.20)]:
            gen_mask = generation == gen
            female_mask = gen_mask & (sex == 0)
            male_mask = gen_mask & (sex == 1)
            assert abs(affected[female_mask].mean() - exp_f) < 0.02, (
                f"gen {gen} female: expected ~{exp_f}, got {affected[female_mask].mean()}"
            )
            assert abs(affected[male_mask].mean() - exp_m) < 0.02, (
                f"gen {gen} male: expected ~{exp_m}, got {affected[male_mask].mean()}"
            )

    def test_sex_aware_per_gen_pools_across_sexes(self):
        """Under standardize='per_generation', L is z-scored once per gen across
        both sexes — sex-shifted means therefore yield sex-specific realised rates."""
        rng = np.random.default_rng(13)
        n_per_sex = 20_000
        # Single generation. Female liability shifted +1.0, male shifted -1.0.
        liab_f = rng.normal(1.0, 1.0, n_per_sex)
        liab_m = rng.normal(-1.0, 1.0, n_per_sex)
        liability = np.concatenate([liab_f, liab_m])
        sex = np.array([0] * n_per_sex + [1] * n_per_sex)
        generation = np.zeros(2 * n_per_sex, dtype=int)
        affected = _apply_threshold_sex_aware(
            liability,
            generation,
            sex,
            prevalence={"female": 0.10, "male": 0.10},
            standardize="per_generation",
        )
        female_rate = affected[:n_per_sex].mean()
        male_rate = affected[n_per_sex:].mean()
        # Pooled z-score puts female mean above 0 and male mean below 0; with a
        # K=0.10 threshold (~ +1 SD pooled), females cross it far more often.
        assert female_rate > 0.20
        assert male_rate < 0.05

    def test_sex_aware_global_unchanged(self):
        """Under standardize='global', sex-aware path matches today's pooled-global behavior."""
        rng = np.random.default_rng(13)
        n_per_sex = 20_000
        liab_f = rng.normal(1.0, 1.0, n_per_sex)
        liab_m = rng.normal(-1.0, 1.0, n_per_sex)
        liability = np.concatenate([liab_f, liab_m])
        sex = np.array([0] * n_per_sex + [1] * n_per_sex)
        generation = np.zeros(2 * n_per_sex, dtype=int)
        affected = _apply_threshold_sex_aware(
            liability,
            generation,
            sex,
            prevalence={"female": 0.10, "male": 0.10},
            standardize="global",
        )
        # Same shape: pooled z-score → female mean > 0, male mean < 0.
        female_rate = affected[:n_per_sex].mean()
        male_rate = affected[n_per_sex:].mean()
        assert female_rate > 0.20
        assert male_rate < 0.05


# ---------------------------------------------------------------------------
# run_threshold orchestrator
# ---------------------------------------------------------------------------


def _tiny_pedigree(n_per_gen: int = 1000, n_gens: int = 2, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = n_per_gen * n_gens
    return pd.DataFrame(
        {
            "id": np.arange(n, dtype=np.int64),
            "mother": np.full(n, -1, dtype=np.int64),
            "father": np.full(n, -1, dtype=np.int64),
            "twin": np.full(n, -1, dtype=np.int64),
            "sex": rng.integers(0, 2, size=n).astype(np.int8),
            "generation": np.repeat(np.arange(n_gens), n_per_gen).astype(np.int32),
            "household_id": np.arange(n, dtype=np.int64),
            "A1": rng.standard_normal(n).astype(np.float32),
            "C1": rng.standard_normal(n).astype(np.float32),
            "E1": rng.standard_normal(n).astype(np.float32),
            "liability1": rng.standard_normal(n).astype(np.float32),
            "A2": rng.standard_normal(n).astype(np.float32),
            "C2": rng.standard_normal(n).astype(np.float32),
            "E2": rng.standard_normal(n).astype(np.float32),
            "liability2": rng.standard_normal(n).astype(np.float32),
        }
    )


class TestRunThreshold:
    def test_emits_expected_columns(self):
        df = _tiny_pedigree(n_per_gen=500, n_gens=2)
        out = run_threshold(
            df,
            G_pheno=2,
            phenotype_params1={"prevalence": 0.1},
            phenotype_params2={"prevalence": 0.2},
        )
        for col in ("id", "sex", "generation", "affected1", "affected2", "liability1", "liability2"):
            assert col in out.columns
        assert out["affected1"].dtype == bool
        assert out["affected2"].dtype == bool

    def test_filters_to_last_g_pheno_generations(self):
        df = _tiny_pedigree(n_per_gen=500, n_gens=3)
        # Keep only the last 2 generations: gens 1 and 2
        out = run_threshold(
            df,
            G_pheno=2,
            phenotype_params1={"prevalence": 0.1},
            phenotype_params2={"prevalence": 0.1},
        )
        assert set(out["generation"].unique()) == {1, 2}
        assert len(out) == 2 * 500

    def test_observed_prevalence_matches(self):
        df = _tiny_pedigree(n_per_gen=20_000, n_gens=1, seed=42)
        out = run_threshold(
            df,
            G_pheno=1,
            phenotype_params1={"prevalence": 0.1},
            phenotype_params2={"prevalence": 0.2},
        )
        assert abs(out["affected1"].mean() - 0.1) < 0.02
        assert abs(out["affected2"].mean() - 0.2) < 0.02

    def test_dict_prevalence_logging_path(self, caplog):
        df = _tiny_pedigree(n_per_gen=500, n_gens=2)
        with caplog.at_level("INFO"):
            run_threshold(
                df,
                G_pheno=2,
                phenotype_params1={"prevalence": {0: 0.05, 1: 0.10}},
                phenotype_params2={"prevalence": 0.15},
            )
        # Ensures the dict-aware log branch (vs %.3f format) was taken
        assert any("Applying threshold model" in m for m in caplog.messages)

    def test_g_pheno_too_large_raises(self):
        df = _tiny_pedigree(n_per_gen=500, n_gens=2)
        with pytest.raises(AssertionError, match="G_pheno"):
            run_threshold(
                df,
                G_pheno=5,
                phenotype_params1={"prevalence": 0.1},
                phenotype_params2={"prevalence": 0.1},
            )


class TestParsePrevalenceArg:
    def test_scalar_passthrough(self):
        assert _parse_prevalence_arg(0.1, None) == 0.1

    def test_none_passthrough(self):
        assert _parse_prevalence_arg(None, None) is None

    def test_by_gen_json_parses_to_int_keyed_dict(self):
        result = _parse_prevalence_arg(None, '{"0": 0.05, "1": 0.10}')
        assert result == {0: 0.05, 1: 0.10}
        assert all(isinstance(k, int) for k in result)
        assert all(isinstance(v, float) for v in result.values())

    def test_by_gen_takes_precedence_over_scalar(self):
        # If the JSON arg is set, the scalar is ignored.
        result = _parse_prevalence_arg(0.5, '{"0": 0.1}')
        assert result == {0: 0.1}


class TestThresholdCli:
    def test_writes_phenotype_parquet(self, tmp_path, monkeypatch):
        df = _tiny_pedigree(n_per_gen=200, n_gens=2)
        ped_path = tmp_path / "pedigree.parquet"
        df.to_parquet(ped_path)

        out_path = tmp_path / "phenotype.parquet"
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "threshold",
                "--pedigree",
                str(ped_path),
                "--output",
                str(out_path),
                "--G-pheno",
                "2",
                "--prevalence1",
                "0.1",
                "--prevalence2",
                "0.2",
            ],
        )
        threshold_cli()
        assert out_path.exists()
        out = pd.read_parquet(out_path)
        assert {"affected1", "affected2", "id"}.issubset(out.columns)

    def test_per_gen_json_path(self, tmp_path, monkeypatch):
        df = _tiny_pedigree(n_per_gen=200, n_gens=2)
        ped_path = tmp_path / "pedigree.parquet"
        df.to_parquet(ped_path)

        out_path = tmp_path / "phenotype.parquet"
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "threshold",
                "--pedigree",
                str(ped_path),
                "--output",
                str(out_path),
                "--G-pheno",
                "2",
                "--prevalence1-by-gen",
                '{"0": 0.05, "1": 0.15}',
                "--prevalence2",
                "0.1",
            ],
        )
        threshold_cli()
        assert out_path.exists()
