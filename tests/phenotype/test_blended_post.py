"""Smoke tests for simace.phenotype.blended_post.blended_diagnosis."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from simace.phenotype.blended_post import (
    MAX_AGE,
    blended_diagnosis,
)


def _make_phenotype(
    rng: np.random.Generator,
    n_per_gen: int = 5000,
    gens: tuple[int, ...] = (2, 3, 4),
    death_age: float | np.ndarray = 1e6,
) -> pd.DataFrame:
    """Synthesize a minimal phenotype DataFrame with A1/C1/E1 + L1/L2 columns.

    ``death_age`` may be a scalar (broadcast to every row) or an array of
    length ``n_per_gen * len(gens)`` to drive competing-risk scenarios.
    """
    n_total = n_per_gen * len(gens)
    if np.isscalar(death_age):
        death_ages = np.full(n_total, float(death_age))
    else:
        death_ages = np.asarray(death_age, dtype=np.float64)
        if death_ages.shape != (n_total,):
            raise ValueError(f"death_age array must have shape ({n_total},), got {death_ages.shape}")
    rows = []
    for g in gens:
        A1 = rng.normal(0.0, np.sqrt(0.5), n_per_gen)
        E1 = rng.normal(0.0, np.sqrt(0.5), n_per_gen)
        A2 = rng.normal(0.0, np.sqrt(0.5), n_per_gen)
        E2 = rng.normal(0.0, np.sqrt(0.5), n_per_gen)
        L1 = A1 + E1
        L2 = A2 + E2
        for i in range(n_per_gen):
            idx = len(rows)
            rows.append(
                {
                    "id": idx,
                    "generation": g,
                    "sex": rng.integers(0, 2),
                    "A1": A1[i],
                    "C1": 0.0,
                    "E1": E1[i],
                    "liability1": L1[i],
                    "A2": A2[i],
                    "C2": 0.0,
                    "E2": E2[i],
                    "liability2": L2[i],
                    "death_age": death_ages[idx],
                    "affected1": False,
                    "t_observed1": MAX_AGE,
                    "age_censored1": True,
                    "death_censored1": False,
                }
            )
    return pd.DataFrame(rows)


class TestBlendedDiagnosis:
    def test_alpha_zero_recovers_pure_trait1(self):
        """With α=0 everywhere, case fraction should track K_by_gen on L1 alone."""
        rng = np.random.default_rng(0)
        pheno = _make_phenotype(rng)
        out = blended_diagnosis(
            pheno,
            alpha_by_gen={2: 0.0, 3: 0.0, 4: 0.0},
            K_by_gen={2: 0.05, 3: 0.05, 4: 0.05},
        )
        # Per-gen K should be ~5% within sampling tolerance.
        rates = out.groupby("generation")["affected1"].mean()
        for g, rate in rates.items():
            assert 0.035 < rate < 0.065, f"gen {g} case rate {rate:.3f} too far from 5%"

    def test_per_gen_K_targets(self):
        """Per-gen case fractions should match per-gen K_by_gen."""
        rng = np.random.default_rng(1)
        pheno = _make_phenotype(rng, n_per_gen=10000)
        alpha = {2: 0.0, 3: 0.3, 4: 0.55}
        K = {2: 0.03, 3: 0.05, 4: 0.10}
        out = blended_diagnosis(pheno, alpha_by_gen=alpha, K_by_gen=K)
        rates = out.groupby("generation")["affected1"].mean()
        for g in (2, 3, 4):
            target = K[g]
            assert abs(rates[g] - target) < 0.01, f"gen {g}: realized K={rates[g]:.3f} vs target {target}"

    def test_late_onset_cases_right_censored(self):
        """Onset > MAX_AGE should be right-censored: affected1=False, t_observed1=MAX_AGE."""
        rng = np.random.default_rng(2)
        pheno = _make_phenotype(rng, n_per_gen=2000)
        out = blended_diagnosis(
            pheno,
            alpha_by_gen={2: 0.0, 3: 0.0, 4: 0.0},
            K_by_gen={2: 0.5, 3: 0.5, 4: 0.5},  # Very high K → some onsets exceed MAX_AGE
        )
        assert (out["t_observed1"] <= MAX_AGE).all(), "no t_observed1 > MAX_AGE expected"
        # No NaNs or impossibly young onsets
        assert (out["t_observed1"] > 0).all()
        # age_censored1 is a boolean flag
        assert out["age_censored1"].dtype == bool

    def test_audit_columns_present(self):
        """Output should expose A_blend, C_blend, E_blend, liability_blend."""
        rng = np.random.default_rng(3)
        pheno = _make_phenotype(rng, n_per_gen=200)
        out = blended_diagnosis(
            pheno,
            alpha_by_gen={2: 0.0, 3: 0.5, 4: 0.5},
            K_by_gen={2: 0.05, 3: 0.05, 4: 0.05},
        )
        for col in ("A_blend", "C_blend", "E_blend", "liability_blend"):
            assert col in out.columns

    def test_blend_components_match_formula(self):
        """A_blend = (1-α)·A1 + α·A2 per row, similar for others."""
        rng = np.random.default_rng(4)
        pheno = _make_phenotype(rng, n_per_gen=100)
        alpha = {2: 0.2, 3: 0.5, 4: 0.8}
        K = {2: 0.05, 3: 0.05, 4: 0.05}
        out = blended_diagnosis(pheno, alpha_by_gen=alpha, K_by_gen=K)
        for g, a in alpha.items():
            sub = out[out["generation"] == g]
            expected_A = (1 - a) * sub["A1"] + a * sub["A2"]
            np.testing.assert_allclose(sub["A_blend"], expected_A, atol=1e-5)

    def test_missing_gen_in_alpha_raises(self):
        """alpha_by_gen lacking an observed generation should raise ValueError.

        The error originates from the shared prevalence_to_array helper, so
        the message says "prevalence dict missing generation ...".
        """
        rng = np.random.default_rng(5)
        pheno = _make_phenotype(rng, n_per_gen=100, gens=(2, 3, 4))
        with pytest.raises(ValueError, match="missing generation"):
            blended_diagnosis(
                pheno,
                alpha_by_gen={2: 0.0, 3: 0.0},  # gen 4 missing
                K_by_gen={2: 0.05, 3: 0.05, 4: 0.05},
            )

    def test_missing_required_columns_raises(self):
        """phenotype lacking A1 or liability1 should raise."""
        df = pd.DataFrame({"generation": [2, 2], "sex": [0, 1]})
        with pytest.raises(ValueError, match="missing required columns"):
            blended_diagnosis(df, alpha_by_gen={2: 0.0}, K_by_gen={2: 0.05})

    def test_non_case_with_early_death_is_death_censored(self):
        """Non-cases whose death_age < MAX_AGE must be death-censored."""
        rng = np.random.default_rng(6)
        pheno = _make_phenotype(rng, n_per_gen=500, gens=(2,), death_age=40.0)
        out = blended_diagnosis(
            pheno,
            alpha_by_gen={2: 0.0},
            K_by_gen={2: 1e-6},  # ~no cases, so all rows are non-cases
        )
        assert (~out["affected1"]).all(), "no cases expected at K=1e-6"
        assert out["death_censored1"].all(), "every non-case with death<MAX should be death-censored"
        np.testing.assert_allclose(out["t_observed1"], 40.0)
        assert (~out["age_censored1"]).all()

    def test_death_censoring_invariants_with_late_onset_population(self):
        """t_observed1 must never exceed death_age, and late-onset / non-case
        rows with death_age < MAX_AGE must be death-censored.

        Uses K=0.5 to generate a population mix of strong cases (onset near 0),
        marginal cases (onset > MAX_AGE), and non-cases.  With death_age=70
        every row whose latent follow-up exceeds 70 must be death-censored.
        """
        rng = np.random.default_rng(7)
        pheno = _make_phenotype(rng, n_per_gen=2000, gens=(2,), death_age=70.0)
        out = blended_diagnosis(
            pheno,
            alpha_by_gen={2: 0.0},
            K_by_gen={2: 0.5},
        )
        # Core invariant: nobody is observed past their death.
        assert (out["t_observed1"] <= out["death_age"]).all()
        # Status flags are mutually exclusive.
        flags = out[["affected1", "age_censored1", "death_censored1"]].to_numpy()
        assert (flags.sum(axis=1) == 1).all()
        # Every row that is not affected and has death_age < MAX_AGE must be
        # death-censored at death_age (covers non-cases and late-onset cases).
        unobserved = ~out["affected1"]
        assert out.loc[unobserved, "death_censored1"].all()
        np.testing.assert_allclose(out.loc[unobserved, "t_observed1"], 70.0)
        # The population should still contain affected cases (strong cases
        # whose onset precedes death).
        assert out["affected1"].any()

    def test_no_early_death_preserves_full_followup(self):
        """With death_age=1e6 (effectively never), no row is death-censored."""
        rng = np.random.default_rng(8)
        pheno = _make_phenotype(rng, n_per_gen=1000, gens=(2,), death_age=1e6)
        out = blended_diagnosis(
            pheno,
            alpha_by_gen={2: 0.3},
            K_by_gen={2: 0.05},
        )
        assert (~out["death_censored1"]).all()
        assert (out["t_observed1"] <= MAX_AGE).all()


class TestDualFramePolars:
    def test_polars_result_matches_pandas_exactly(self):
        """Same-type dual-frame API (ADR 0015): identical values via either library."""
        import polars as pl
        import polars.testing

        rng = np.random.default_rng(11)
        pheno = _make_phenotype(rng, n_per_gen=2000, gens=(2, 3))
        alpha = {2: 0.2, 3: 0.6}
        K = {2: 0.05, 3: 0.08}

        out_pd = blended_diagnosis(pheno, alpha_by_gen=alpha, K_by_gen=K)
        out_pl = blended_diagnosis(pl.from_pandas(pheno), alpha_by_gen=alpha, K_by_gen=K)

        assert isinstance(out_pl, pl.DataFrame)
        polars.testing.assert_frame_equal(out_pl, pl.from_pandas(out_pd))

    def test_polars_input_not_mutated(self):
        import polars as pl

        rng = np.random.default_rng(12)
        pheno = pl.from_pandas(_make_phenotype(rng, n_per_gen=500, gens=(2,)))
        before = pheno.clone()
        blended_diagnosis(pheno, alpha_by_gen={2: 0.5}, K_by_gen={2: 0.05})
        import polars.testing

        polars.testing.assert_frame_equal(pheno, before)
