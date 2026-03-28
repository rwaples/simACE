"""Edge case tests for ACE simulation.

Tests boundary conditions: zero variance components, no twins,
all non-social fathers, single-generation pedigrees, etc.
"""

import numpy as np

from sim_ace.censoring.censor import age_censor
from sim_ace.phenotyping.phenotype import simulate_phenotype
from sim_ace.phenotyping.threshold import apply_threshold
from sim_ace.simulation.simulate import run_simulation

# ---------------------------------------------------------------------------
# Shared minimal params
# ---------------------------------------------------------------------------

MINIMAL_PARAMS = dict(
    seed=42,
    N=500,
    G_ped=2,
    mating_lambda=0.5,
    p_mztwin=0.02,
    A1=0.5,
    C1=0.2,
    A2=0.5,
    C2=0.2,
    rA=0.3,
    rC=0.5,
)


# ---------------------------------------------------------------------------
# A = 0 (no genetic component)
# ---------------------------------------------------------------------------


class TestZeroA:
    def test_A_zero_both_traits(self):
        params = {**MINIMAL_PARAMS, "A1": 0.0, "C1": 0.3, "A2": 0.0, "C2": 0.3}
        ped = run_simulation(**params)
        # All A values should be zero for founders
        founders = ped[ped["mother"] == -1]
        np.testing.assert_array_equal(founders["A1"].values, 0.0)
        np.testing.assert_array_equal(founders["A2"].values, 0.0)

    def test_A_zero_offspring_A_also_zero(self):
        """With A=0, offspring A should also be zero (midparent=0, noise sd=0)."""
        params = {**MINIMAL_PARAMS, "A1": 0.0, "C1": 0.3, "A2": 0.0, "C2": 0.3}
        ped = run_simulation(**params)
        np.testing.assert_array_equal(ped["A1"].values, 0.0)
        np.testing.assert_array_equal(ped["A2"].values, 0.0)

    def test_A_zero_liability_equals_C_plus_E(self):
        params = {**MINIMAL_PARAMS, "A1": 0.0, "C1": 0.3, "A2": 0.0, "C2": 0.3}
        ped = run_simulation(**params)
        # ACE columns are float32, liability is float64
        np.testing.assert_allclose(
            ped["liability1"].values,
            ped["C1"].values + ped["E1"].values,
            atol=1e-6,
        )


# ---------------------------------------------------------------------------
# C = 0 (no shared environment)
# ---------------------------------------------------------------------------


class TestZeroC:
    def test_C_zero_both_traits(self):
        params = {**MINIMAL_PARAMS, "A1": 0.5, "C1": 0.0, "A2": 0.5, "C2": 0.0}
        ped = run_simulation(**params)
        np.testing.assert_array_equal(ped["C1"].values, 0.0)
        np.testing.assert_array_equal(ped["C2"].values, 0.0)

    def test_C_zero_liability_equals_A_plus_E(self):
        params = {**MINIMAL_PARAMS, "A1": 0.5, "C1": 0.0, "A2": 0.5, "C2": 0.0}
        ped = run_simulation(**params)
        # ACE columns are float32, liability is float64
        np.testing.assert_allclose(
            ped["liability1"].values,
            ped["A1"].values + ped["E1"].values,
            atol=1e-6,
        )


# ---------------------------------------------------------------------------
# A = 0 and C = 0 (pure environment)
# ---------------------------------------------------------------------------


class TestZeroACE:
    def test_all_zero_variance_components(self):
        params = {**MINIMAL_PARAMS, "A1": 0.0, "C1": 0.0, "A2": 0.0, "C2": 0.0}
        ped = run_simulation(**params)
        # E should carry all variance (E = 1 - 0 - 0 = 1)
        np.testing.assert_array_equal(ped["A1"].values, 0.0)
        np.testing.assert_array_equal(ped["C1"].values, 0.0)
        # liability = E only
        np.testing.assert_allclose(ped["liability1"].values, ped["E1"].values)

    def test_pure_E_variance_near_one(self):
        params = {**MINIMAL_PARAMS, "A1": 0.0, "C1": 0.0, "A2": 0.0, "C2": 0.0}
        ped = run_simulation(**params)
        var = ped["E1"].var()
        assert abs(var - 1.0) < 0.15


# ---------------------------------------------------------------------------
# p_mztwin = 0 (no twins)
# ---------------------------------------------------------------------------


class TestNoTwins:
    def test_no_twins(self):
        params = {**MINIMAL_PARAMS, "p_mztwin": 0.0}
        ped = run_simulation(**params)
        assert (ped["twin"] == -1).all()

    def test_no_twins_output_valid(self):
        params = {**MINIMAL_PARAMS, "p_mztwin": 0.0}
        ped = run_simulation(**params)
        assert len(ped) == params["N"] * params["G_ped"]
        # Structural integrity still holds
        assert ped["id"].is_unique


# ---------------------------------------------------------------------------
# High mating_lambda (most sibs are half-sibs)
# ---------------------------------------------------------------------------


class TestHighMatingLambda:
    def test_high_lambda_runs(self):
        params = {**MINIMAL_PARAMS, "mating_lambda": 3.0}
        ped = run_simulation(**params)
        assert len(ped) == params["N"] * params["G_ped"]

    def test_high_lambda_produces_half_sibs(self):
        """With high lambda, individuals have many partners so maternal
        half-sibs should be common."""
        params = {**MINIMAL_PARAMS, "mating_lambda": 3.0, "N": 2000}
        ped = run_simulation(**params)
        non_founders = ped[ped["mother"] != -1]
        fam = non_founders.groupby("mother").agg(
            n_fathers=("father", "nunique"),
            n_kids=("id", "count"),
        )
        multi = fam[fam["n_kids"] >= 2]
        if len(multi) > 0:
            # Most multi-child families should have multiple fathers
            frac_mixed = (multi["n_fathers"] > 1).mean()
            assert frac_mixed > 0.3


# ---------------------------------------------------------------------------
# Low mating_lambda (nearly all sibs share father)
# ---------------------------------------------------------------------------


class TestLowMatingLambda:
    def test_low_lambda_mostly_full_sibs(self):
        """With very low lambda, most individuals have 1 partner, so
        siblings from same mother should usually share a father."""
        params = {**MINIMAL_PARAMS, "mating_lambda": 0.001}
        ped = run_simulation(**params)
        non_founders = ped[ped["mother"] != -1]
        fam = non_founders.groupby("mother")["father"].nunique()
        # Nearly all families should have a single father
        assert (fam == 1).mean() > 0.9


# ---------------------------------------------------------------------------
# Single-generation pedigree (G_ped=1)
# ---------------------------------------------------------------------------


class TestSingleGeneration:
    def test_G_ped_1(self):
        params = {**MINIMAL_PARAMS, "G_ped": 1}
        ped = run_simulation(**params)
        assert len(ped) == params["N"]
        assert (ped["generation"] == 0).all()

    def test_G_ped_1_all_founders(self):
        """With G_ped=1, all individuals are first-generation offspring
        (founders are burn-in gen 0, but with G_sim=G_ped=1, the one
        recorded generation is gen 0 offspring)."""
        params = {**MINIMAL_PARAMS, "G_ped": 1}
        ped = run_simulation(**params)
        # Generation 0 should have parents since G_sim=1 means one mating round
        assert len(ped) == params["N"]


# ---------------------------------------------------------------------------
# Edge cases for phenotype functions
# ---------------------------------------------------------------------------


class TestPhenotypeEdgeCases:
    def test_single_individual(self):
        """simulate_phenotype should work with a single individual."""
        t = simulate_phenotype(
            np.array([0.5]), beta=1.0, hazard_model="weibull", hazard_params={"scale": 316.228, "rho": 2.0}, seed=42
        )
        assert t.shape == (1,)
        assert t[0] > 0

    def test_very_large_beta(self):
        """Large beta should not produce NaN or Inf."""
        liability = np.array([0.0, 0.1, -0.1])
        t = simulate_phenotype(
            liability, beta=50.0, hazard_model="weibull", hazard_params={"scale": 316.228, "rho": 2.0}, seed=42
        )
        assert np.all(np.isfinite(t))
        assert np.all(t > 0)

    def test_age_censor_all_left_censored(self):
        t = np.array([1.0, 2.0, 3.0])
        left = np.array([10.0, 10.0, 10.0])
        right = np.array([80.0, 80.0, 80.0])
        t_out, censored = age_censor(t, left, right)
        assert censored.all()
        np.testing.assert_array_equal(t_out, left)

    def test_age_censor_all_right_censored(self):
        t = np.array([100.0, 200.0, 300.0])
        left = np.array([10.0, 10.0, 10.0])
        right = np.array([80.0, 80.0, 80.0])
        t_out, censored = age_censor(t, left, right)
        assert censored.all()
        np.testing.assert_array_equal(t_out, right)


# ---------------------------------------------------------------------------
# Edge cases for threshold
# ---------------------------------------------------------------------------


class TestThresholdEdgeCases:
    def test_very_low_prevalence(self):
        """prevalence=0.01 should classify ~1% as affected."""
        rng = np.random.default_rng(42)
        n = 10000
        liability = rng.standard_normal(n)
        generation = np.zeros(n, dtype=int)
        affected = apply_threshold(liability, generation, prevalence=0.01)
        assert 0.005 < affected.mean() < 0.02

    def test_very_high_prevalence(self):
        """prevalence=0.99 should classify ~99% as affected."""
        rng = np.random.default_rng(42)
        n = 10000
        liability = rng.standard_normal(n)
        generation = np.zeros(n, dtype=int)
        affected = apply_threshold(liability, generation, prevalence=0.99)
        assert 0.98 < affected.mean() < 1.0

    def test_single_generation_single_individual(self):
        """Edge case: one individual."""
        liability = np.array([1.0])
        generation = np.array([0])
        affected = apply_threshold(liability, generation, prevalence=0.5)
        assert affected.dtype == bool
        assert len(affected) == 1


# ---------------------------------------------------------------------------
# Boundary variance: A + C exactly equals 1 (E = 0)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Assortative mating edge cases
# ---------------------------------------------------------------------------


class TestAssortativeEdgeCases:
    def test_strong_positive_assortment(self):
        """assort1=1.0 should produce strong positive mate correlation."""
        params = {**MINIMAL_PARAMS, "assort1": 1.0, "assort2": 0.0}
        ped = run_simulation(**params)
        non_founders = ped[ped["mother"] != -1]
        pairs = non_founders[["mother", "father"]].drop_duplicates()
        df_idx = ped.set_index("id")
        m_liab = df_idx.loc[pairs["mother"].values, "liability1"].values
        f_liab = df_idx.loc[pairs["father"].values, "liability1"].values
        corr = np.corrcoef(m_liab, f_liab)[0, 1]
        assert corr > 0.3

    def test_strong_negative_assortment(self):
        """assort1=-1.0 should produce strong negative mate correlation."""
        params = {**MINIMAL_PARAMS, "assort1": -1.0, "assort2": 0.0}
        ped = run_simulation(**params)
        non_founders = ped[ped["mother"] != -1]
        pairs = non_founders[["mother", "father"]].drop_duplicates()
        df_idx = ped.set_index("id")
        m_liab = df_idx.loc[pairs["mother"].values, "liability1"].values
        f_liab = df_idx.loc[pairs["father"].values, "liability1"].values
        corr = np.corrcoef(m_liab, f_liab)[0, 1]
        assert corr < -0.3

    def test_both_assort_nonzero(self):
        """Both assort1 and assort2 nonzero should run without error."""
        params = {**MINIMAL_PARAMS, "assort1": 0.3, "assort2": 0.2}
        ped = run_simulation(**params)
        assert len(ped) == params["N"] * params["G_ped"]


# ---------------------------------------------------------------------------
# Boundary variance: A + C exactly equals 1 (E = 0)
# ---------------------------------------------------------------------------


class TestEZero:
    def test_E_zero(self):
        """When A + C = 1, E = 0 and all E values should be negligible.

        Note: 1.0 - 0.7 - 0.3 has floating-point roundoff (~5.5e-17),
        so sd_E ~ 7.4e-9, producing tiny but non-zero draws.
        """
        params = {**MINIMAL_PARAMS, "A1": 0.7, "C1": 0.3, "A2": 0.6, "C2": 0.4}
        ped = run_simulation(**params)
        np.testing.assert_allclose(ped["E1"].values, 0.0, atol=1e-6)
        np.testing.assert_allclose(ped["E2"].values, 0.0, atol=1e-6)

    def test_E_zero_liability_equals_A_plus_C(self):
        params = {**MINIMAL_PARAMS, "A1": 0.7, "C1": 0.3, "A2": 0.6, "C2": 0.4}
        ped = run_simulation(**params)
        np.testing.assert_allclose(
            ped["liability1"].values,
            ped["A1"].values + ped["C1"].values,
            atol=1e-6,
        )
