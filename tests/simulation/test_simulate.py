"""Unit tests for simace.simulate functions."""

import builtins
import importlib

import numpy as np
import polars as pl
import pytest
from polars.testing import assert_frame_equal

import simace.simulation.simulate as simulate_mod
from simace.simulation.simulate import (
    _assortative_pair_partners,
    _find_duplicate_pairs,
    _mating_wf,
    _metropolis_full_python,
    _metropolis_sweep_python,
    add_to_pedigree,
    allocate_offspring,
    assign_twins,
    balance_mating_slots,
    draw_mating_counts,
    generate_correlated_components,
    generate_mendelian_noise,
    mating,
    pair_partners,
    resolve_per_gen_param,
    run_simulation,
)

# ---------------------------------------------------------------------------
# optional numba fallback
# ---------------------------------------------------------------------------


class TestOptionalNumbaFallback:
    def test_simulate_module_loads_without_numba(self, monkeypatch):
        real_import = builtins.__import__

        def blocked_import(name, global_vars=None, local_vars=None, fromlist=(), level=0):
            if name == "numba":
                raise ImportError("numba blocked for fallback test")
            return real_import(name, global_vars, local_vars, fromlist, level)

        try:
            with monkeypatch.context() as m:
                m.setattr(builtins, "__import__", blocked_import)
                reloaded = importlib.reload(simulate_mod)
                assert reloaded.njit is None
                assert reloaded._quantile_normal_nb is reloaded._quantile_normal_nb_python
                assert reloaded._midparent is reloaded._midparent_python
                assert reloaded._metropolis_sweep is reloaded._metropolis_sweep_python
                assert reloaded._metropolis_full is reloaded._metropolis_full_python
        finally:
            importlib.reload(simulate_mod)


# ---------------------------------------------------------------------------
# generate_correlated_components
# ---------------------------------------------------------------------------


class TestGenerateCorrelatedComponents:
    def test_output_shapes(self, rng):
        c1, c2 = generate_correlated_components(rng, 500, 1.0, 1.0, 0.5)
        assert c1.shape == (500,)
        assert c2.shape == (500,)

    def test_zero_sd_gives_zeros(self, rng):
        c1, c2 = generate_correlated_components(rng, 100, 0.0, 0.0, 0.0)
        np.testing.assert_array_equal(c1, 0.0)
        np.testing.assert_array_equal(c2, 0.0)

    def test_negative_sd_raises(self, rng):
        with pytest.raises(ValueError, match="non-negative"):
            generate_correlated_components(rng, 100, -1.0, 1.0, 0.0)

    def test_correlation_out_of_range_raises(self, rng):
        with pytest.raises(ValueError, match="Correlation"):
            generate_correlated_components(rng, 100, 1.0, 1.0, 1.5)

    def test_mean_near_zero(self, rng):
        c1, c2 = generate_correlated_components(rng, 10000, 1.0, 1.0, 0.5)
        assert abs(c1.mean()) < 0.1
        assert abs(c2.mean()) < 0.1

    def test_sd_matches_input(self, rng):
        sd1, sd2 = 0.7, 1.3
        c1, c2 = generate_correlated_components(rng, 50000, sd1, sd2, 0.0)
        assert abs(c1.std() - sd1) < 0.05
        assert abs(c2.std() - sd2) < 0.05

    def test_correlation_matches_input(self, rng):
        c1, c2 = generate_correlated_components(rng, 50000, 1.0, 1.0, 0.6)
        observed = np.corrcoef(c1, c2)[0, 1]
        assert abs(observed - 0.6) < 0.05

    def test_perfect_correlation(self, rng):
        c1, c2 = generate_correlated_components(rng, 1000, 1.0, 1.0, 1.0)
        np.testing.assert_allclose(c1, c2, atol=1e-10)

    def test_perfect_negative_correlation(self, rng):
        c1, c2 = generate_correlated_components(rng, 1000, 1.0, 1.0, -1.0)
        np.testing.assert_allclose(c1, -c2, atol=1e-10)


# ---------------------------------------------------------------------------
# generate_mendelian_noise
# ---------------------------------------------------------------------------


class TestGenerateMendelianNoise:
    def test_output_shapes(self, rng):
        n1, n2 = generate_mendelian_noise(rng, 500, 1.0, 1.0, 0.5)
        assert n1.shape == (500,)
        assert n2.shape == (500,)

    def test_variance_is_half_parental(self, rng):
        """Mendelian noise sd = sd_A / sqrt(2), so var = A/2."""
        sd_A = 0.8
        n1, n2 = generate_mendelian_noise(rng, 50000, sd_A, sd_A, 0.0)
        expected_var = sd_A**2 / 2
        assert abs(n1.var() - expected_var) < 0.02
        assert abs(n2.var() - expected_var) < 0.02


# ---------------------------------------------------------------------------
# draw_mating_counts
# ---------------------------------------------------------------------------


class TestDrawMatingCounts:
    def test_all_positive(self, rng):
        counts = draw_mating_counts(rng, 500, 0.5)
        assert np.all(counts >= 1)

    def test_shape(self, rng):
        counts = draw_mating_counts(rng, 300, 0.5)
        assert counts.shape == (300,)

    def test_mean_near_expected(self, rng):
        """ZTP(0.5) mean = lambda / (1 - e^{-lambda}) = 0.5 / (1 - e^{-0.5}) ~ 1.27."""
        counts = draw_mating_counts(rng, 50000, 0.5)
        expected = 0.5 / (1 - np.exp(-0.5))
        assert abs(counts.mean() - expected) < 0.05

    def test_high_lambda(self, rng):
        counts = draw_mating_counts(rng, 1000, 3.0)
        assert np.all(counts >= 1)
        assert counts.mean() > 2.5


# ---------------------------------------------------------------------------
# balance_mating_slots
# ---------------------------------------------------------------------------


class TestBalanceMatingSlots:
    def test_totals_match(self, rng):
        mc = np.array([2, 3, 1])
        fc = np.array([1, 1, 1, 1])
        bm, bf = balance_mating_slots(rng, mc, fc)
        assert bm.sum() == bf.sum()

    def test_no_trim_needed(self, rng):
        mc = np.array([2, 2])
        fc = np.array([1, 1, 1, 1])
        bm, bf = balance_mating_slots(rng, mc, fc)
        assert bm.sum() == bf.sum() == 4

    def test_all_counts_nonnegative(self, rng):
        mc = np.array([5, 3, 2])
        fc = np.array([1, 1])
        bm, bf = balance_mating_slots(rng, mc, fc)
        assert np.all(bm >= 0)
        assert np.all(bf >= 0)


# ---------------------------------------------------------------------------
# pair_partners
# ---------------------------------------------------------------------------


class TestPairPartners:
    def test_shape(self, rng):
        males = np.array([0, 1, 2])
        females = np.array([3, 4, 5])
        mc = np.array([1, 1, 1])
        fc = np.array([1, 1, 1])
        pairs = pair_partners(rng, males, mc, females, fc)
        assert pairs.shape == (3, 2)

    def test_mothers_from_females(self, rng):
        males = np.array([10, 11])
        females = np.array([20, 21])
        mc = np.array([2, 1])
        fc = np.array([1, 2])
        pairs = pair_partners(rng, males, mc, females, fc)
        assert np.all(np.isin(pairs[:, 0], females))

    def test_fathers_from_males(self, rng):
        males = np.array([10, 11])
        females = np.array([20, 21])
        mc = np.array([2, 1])
        fc = np.array([1, 2])
        pairs = pair_partners(rng, males, mc, females, fc)
        assert np.all(np.isin(pairs[:, 1], males))

    def test_dedup_breaks_when_no_non_duplicates(self, rng, monkeypatch):
        """Defensive branch: if all pairs are flagged duplicate, stop swapping."""
        monkeypatch.setattr(simulate_mod, "_find_duplicate_pairs", lambda matings: np.ones(len(matings), dtype=bool))
        males = np.array([0, 1])
        females = np.array([10, 11])
        counts = np.array([1, 1])
        pairs = pair_partners(rng, males, counts, females, counts)
        assert pairs.shape == (2, 2)


class TestFindDuplicatePairs:
    def test_empty_input_returns_empty_mask(self):
        mask = _find_duplicate_pairs(np.empty((0, 2), dtype=int))
        assert mask.shape == (0,)
        assert mask.dtype == bool


class TestMetropolisHelpers:
    """Directly exercise accept/reject branches in the Python fallback helpers."""

    @staticmethod
    def _sweep_inputs():
        return dict(
            f1_z=np.array([1.0, 0.0]),
            f2_z=np.zeros(2),
            m1_z=np.array([0.0, 1.0]),
            m2_z=np.zeros(2),
            male_perm=np.array([0, 1]),
            idx_i=np.array([0]),
            idx_j=np.array([1]),
            S1=0.0,
            S2=0.0,
            S12=0.0,
            S21=0.0,
            T2=0.0,
            T12=0.0,
            T21=0.0,
            batch=1,
        )

    def test_metropolis_sweep_accepts_improving_swap(self):
        values = self._sweep_inputs()
        result = _metropolis_sweep_python(**values, T1=10.0)
        assert result[0] == 1.0
        np.testing.assert_array_equal(values["male_perm"], np.array([1, 0]))

    def test_metropolis_sweep_rejects_worsening_swap(self):
        values = self._sweep_inputs()
        result = _metropolis_sweep_python(**values, T1=0.0)
        assert result[0] == 0.0
        np.testing.assert_array_equal(values["male_perm"], np.array([0, 1]))

    @staticmethod
    def _full_inputs():
        return dict(
            fz=np.array([[1.0, 0.0], [0.0, 0.0]]),
            mz=np.array([[0.0, 0.0], [1.0, 0.0]]),
            male_perm=np.array([0, 1]),
            S1=0.0,
            S2=0.0,
            S12=0.0,
            S21=0.0,
            T2=0.0,
            T12=0.0,
            T21=0.0,
            M=2,
            max_proposals=1,
            seed=123,
        )

    def test_metropolis_full_breaks_when_already_within_tolerance(self):
        result = _metropolis_full_python(**self._full_inputs(), T1=0.0, tol=1.0)
        assert result[-1] == 0

    def test_metropolis_full_accepts_improving_swap(self):
        values = self._full_inputs()
        result = _metropolis_full_python(**values, T1=10.0, tol=0.0)
        assert result[0] == 1.0
        assert result[-1] == 1

    def test_metropolis_full_rejects_worsening_swap(self):
        values = self._full_inputs()
        result = _metropolis_full_python(**values, T1=0.1, tol=0.0)
        assert result[0] == 0.0
        assert result[-1] == 1


# ---------------------------------------------------------------------------
# allocate_offspring
# ---------------------------------------------------------------------------


class TestAllocateOffspring:
    def test_sum_equals_N(self, rng):
        counts = allocate_offspring(rng, 50, 1000)
        assert counts.sum() == 1000

    def test_shape(self, rng):
        counts = allocate_offspring(rng, 30, 500)
        assert counts.shape == (30,)

    def test_all_nonneg(self, rng):
        counts = allocate_offspring(rng, 100, 200)
        assert np.all(counts >= 0)


# ---------------------------------------------------------------------------
# assign_twins
# ---------------------------------------------------------------------------


class TestAssignTwins:
    def test_only_eligible(self, rng):
        counts = np.array([0, 1, 2, 3, 0, 1])
        mask = assign_twins(rng, counts, 1.0)
        # Only indices 2, 3 are eligible (counts >= 2); with p=1.0 all should be True
        assert mask[2]
        assert mask[3]
        assert not mask[0]
        assert not mask[1]
        assert not mask[4]
        assert not mask[5]

    def test_no_twins_p_zero(self, rng):
        counts = np.array([3, 4, 5])
        mask = assign_twins(rng, counts, 0.0)
        assert not mask.any()

    def test_no_eligible_matings(self, rng):
        counts = np.array([0, 1, 1])
        mask = assign_twins(rng, counts, 1.0)
        assert not mask.any()

    def test_shape(self, rng):
        counts = np.array([2, 0, 3])
        mask = assign_twins(rng, counts, 0.5)
        assert mask.shape == (3,)
        assert mask.dtype == bool


# ---------------------------------------------------------------------------
# mating (orchestrator)
# ---------------------------------------------------------------------------


class TestMating:
    def test_output_shapes(self, rng):
        sex = rng.binomial(n=1, p=0.5, size=1000)
        parents, twins, household_ids = mating(rng, sex, 0.5, 0.02)
        assert parents.shape == (1000, 2)
        assert household_ids.shape == (1000,)
        assert twins.ndim == 2
        if len(twins) > 0:
            assert twins.shape[1] == 2

    def test_mothers_are_female(self, rng):
        sex = rng.binomial(n=1, p=0.5, size=1000)
        parents, _, _ = mating(rng, sex, 0.5, 0.02)
        mother_sexes = sex[parents[:, 0]]
        assert np.all(mother_sexes == 0)

    def test_fathers_are_male(self, rng):
        sex = rng.binomial(n=1, p=0.5, size=1000)
        parents, _, _ = mating(rng, sex, 0.5, 0.02)
        father_sexes = sex[parents[:, 1]]
        assert np.all(father_sexes == 1)

    def test_twin_pairs_share_mother(self, rng):
        sex = rng.binomial(n=1, p=0.5, size=2000)
        parents, twins, _ = mating(rng, sex, 0.5, 0.05)
        if len(twins) > 0:
            for t1, t2 in twins:
                assert parents[t1, 0] == parents[t2, 0]  # same mother

    def test_twin_pairs_share_father(self, rng):
        sex = rng.binomial(n=1, p=0.5, size=2000)
        parents, twins, _ = mating(rng, sex, 0.5, 0.05)
        if len(twins) > 0:
            for t1, t2 in twins:
                assert parents[t1, 1] == parents[t2, 1]  # same bio father

    def test_no_twins_when_p_zero(self, rng):
        sex = rng.binomial(n=1, p=0.5, size=1000)
        _, twins, _ = mating(rng, sex, 0.5, 0.0)
        assert len(twins) == 0

    def test_household_ids_nonnegative(self, rng):
        sex = rng.binomial(n=1, p=0.5, size=1000)
        _, _, hh = mating(rng, sex, 0.5, 0.02)
        assert np.all(hh >= 0)

    def test_siblings_share_household(self, rng):
        sex = rng.binomial(n=1, p=0.5, size=1000)
        parents, _, hh = mating(rng, sex, 0.5, 0.02)
        # Siblings with same mother should have same household
        for mother_idx in np.unique(parents[:, 0]):
            sib_mask = parents[:, 0] == mother_idx
            sib_hh = hh[sib_mask]
            assert len(np.unique(sib_hh)) == 1


# ---------------------------------------------------------------------------
# reproduce
# ---------------------------------------------------------------------------


class TestReproduce:
    def test_output_shapes(self, founders_and_offspring):
        d = founders_and_offspring
        assert d["offspring"].shape[1] == 6  # A1, C1, E1, A2, C2, E2
        assert d["offspring"].shape[0] == len(d["parents"])
        assert d["sex_offspring"].shape[0] == len(d["parents"])

    def test_sex_binary(self, founders_and_offspring):
        sex = founders_and_offspring["sex_offspring"]
        assert set(np.unique(sex)).issubset({0, 1})

    def test_mz_twins_share_A(self, founders_and_offspring):
        d = founders_and_offspring
        twins = d["twins"]
        if len(twins) > 0:
            for t1, t2 in twins:
                np.testing.assert_equal(d["offspring"][t1, 0], d["offspring"][t2, 0])  # A1
                np.testing.assert_equal(d["offspring"][t1, 3], d["offspring"][t2, 3])  # A2

    def test_mz_twins_share_sex(self, founders_and_offspring):
        d = founders_and_offspring
        twins = d["twins"]
        if len(twins) > 0:
            for t1, t2 in twins:
                assert d["sex_offspring"][t1] == d["sex_offspring"][t2]

    def test_siblings_share_C(self, founders_and_offspring):
        d = founders_and_offspring
        hh = d["household_ids"]
        offspring = d["offspring"]
        for hh_id in np.unique(hh)[:50]:  # check first 50 households
            mask = hh == hh_id
            c1_vals = offspring[mask, 1]
            c2_vals = offspring[mask, 4]
            assert np.all(c1_vals == c1_vals[0])
            assert np.all(c2_vals == c2_vals[0])

    def test_E_differs_between_siblings(self, founders_and_offspring):
        """E should be independently drawn — siblings should NOT share E values."""
        d = founders_and_offspring
        hh = d["household_ids"]
        offspring = d["offspring"]
        found_diff = False
        for hh_id in np.unique(hh):
            mask = hh == hh_id
            if mask.sum() >= 2:
                e1_vals = offspring[mask, 2]
                if e1_vals[0] != e1_vals[1]:
                    found_diff = True
                    break
        assert found_diff, "Expected at least some siblings with different E values"


# ---------------------------------------------------------------------------
# add_to_pedigree
# ---------------------------------------------------------------------------


class TestAddToPedigree:
    def test_founder_generation(self, rng):
        N = 100
        pheno = rng.standard_normal((N, 6))
        sex = rng.binomial(n=1, p=0.5, size=N)
        parents = np.column_stack([np.arange(N), np.arange(N)])
        twins = np.array([], dtype=int).reshape(0, 2)
        hh = np.arange(N)

        ped = add_to_pedigree(pheno, sex, parents, twins, hh, generation=0, pedigree=None)
        assert len(ped) == N
        assert (ped["mother"] == -1).all()
        assert (ped["father"] == -1).all()
        assert (ped["generation"] == 0).all()

    def test_ids_are_contiguous(self, rng):
        N = 100
        pheno = rng.standard_normal((N, 6))
        sex = rng.binomial(n=1, p=0.5, size=N)
        parents = np.column_stack([np.zeros(N, dtype=int), np.ones(N, dtype=int)])
        twins = np.array([], dtype=int).reshape(0, 2)
        hh = np.arange(N)

        ped = add_to_pedigree(pheno, sex, parents, twins, hh, generation=0, pedigree=None)
        np.testing.assert_array_equal(ped["id"].to_numpy(), np.arange(N))

    def test_appending_generation(self, rng):
        N = 50
        pheno = rng.standard_normal((N, 6))
        sex = rng.binomial(n=1, p=0.5, size=N)
        parents = np.column_stack([np.arange(N), np.arange(N)])
        twins = np.array([], dtype=int).reshape(0, 2)
        hh = np.arange(N)

        ped = add_to_pedigree(pheno, sex, parents, twins, hh, generation=0, pedigree=None)
        ped = add_to_pedigree(pheno, sex, parents, twins, hh, generation=1, pedigree=ped)
        assert len(ped) == 2 * N
        np.testing.assert_array_equal(ped["id"].to_numpy(), np.arange(2 * N))

    def test_liability_equals_sum(self, rng):
        N = 100
        pheno = rng.standard_normal((N, 6))
        sex = rng.binomial(n=1, p=0.5, size=N)
        parents = np.column_stack([np.arange(N), np.arange(N)])
        twins = np.array([], dtype=int).reshape(0, 2)
        hh = np.arange(N)

        ped = add_to_pedigree(pheno, sex, parents, twins, hh, generation=0, pedigree=None)
        # ACE columns are float32, liability is float64 — allow float32 precision loss
        np.testing.assert_allclose(
            ped["liability1"].to_numpy(),
            ped["A1"].to_numpy() + ped["C1"].to_numpy() + ped["E1"].to_numpy(),
            atol=1e-6,
        )
        np.testing.assert_allclose(
            ped["liability2"].to_numpy(),
            ped["A2"].to_numpy() + ped["C2"].to_numpy() + ped["E2"].to_numpy(),
            atol=1e-6,
        )

    def test_twin_column_bidirectional(self, rng):
        N = 100
        pheno = rng.standard_normal((N, 6))
        sex = rng.binomial(n=1, p=0.5, size=N)
        parents = np.column_stack([np.zeros(N, dtype=int), np.ones(N, dtype=int)])
        twins = np.array([[0, 1], [10, 11]])
        hh = np.arange(N)

        ped = add_to_pedigree(pheno, sex, parents, twins, hh, generation=0, pedigree=None)
        # ids are contiguous from 0, so row position == id.
        for t1, t2 in twins:
            assert ped["twin"][int(t1)] == t2
            assert ped["twin"][int(t2)] == t1


# ---------------------------------------------------------------------------
# run_simulation
# ---------------------------------------------------------------------------


class TestRunSimulation:
    def test_output_is_dataframe(self, default_params):
        ped = run_simulation(**default_params)
        assert isinstance(ped, pl.DataFrame)

    def test_output_size(self, default_params):
        ped = run_simulation(**default_params)
        expected = default_params["N"] * default_params["G_ped"]
        assert len(ped) == expected

    def test_required_columns_present(self, default_params):
        ped = run_simulation(**default_params)
        expected_cols = {
            "id",
            "sex",
            "mother",
            "father",
            "twin",
            "generation",
            "household_id",
            "A1",
            "C1",
            "E1",
            "liability1",
            "A2",
            "C2",
            "E2",
            "liability2",
        }
        assert expected_cols.issubset(set(ped.columns))

    def test_deterministic_with_same_seed(self, default_params):
        ped1 = run_simulation(**default_params)
        ped2 = run_simulation(**default_params)
        assert_frame_equal(ped1, ped2)

    def test_different_seeds_differ(self, default_params):
        p1 = {**default_params, "seed": 1}
        p2 = {**default_params, "seed": 2}
        ped1 = run_simulation(**p1)
        ped2 = run_simulation(**p2)
        assert not ped1["A1"].equals(ped2["A1"])

    def test_burnin_generations(self, default_params):
        params = {**default_params, "G_sim": 5, "G_ped": 3}
        ped = run_simulation(**params)
        assert len(ped) == params["N"] * params["G_ped"]

    # --- Validation error tests ---

    def test_negative_A_raises(self, default_params):
        with pytest.raises(ValueError, match="must be a non-negative scalar"):
            run_simulation(**{**default_params, "A1": -0.1})

    def test_negative_E_raises(self, default_params):
        with pytest.raises(ValueError, match="E1 must be >= 0"):
            run_simulation(**{**default_params, "E1": -0.1})

    def test_negative_N_raises(self, default_params):
        with pytest.raises(ValueError, match="N must be a positive integer"):
            run_simulation(**{**default_params, "N": -10})

    def test_zero_mating_lambda_raises(self, default_params):
        with pytest.raises(ValueError, match="mating_lambda must be > 0"):
            run_simulation(**{**default_params, "mating_lambda": 0})

    def test_G_sim_less_than_G_ped_raises(self, default_params):
        with pytest.raises(ValueError, match=r"G_sim .* must be >= G_ped"):
            run_simulation(**{**default_params, "G_sim": 1, "G_ped": 3})

    def test_total_pedigree_size_int32_limit_raises(self, default_params, monkeypatch):
        class TinyIntInfo:
            max = 10

        monkeypatch.setattr(simulate_mod.np, "iinfo", lambda _dtype: TinyIntInfo())
        with pytest.raises(ValueError, match="exceeds int32 max"):
            run_simulation(**{**default_params, "N": 4, "G_ped": 3, "G_sim": 3})

    def test_rA_out_of_range_raises(self, default_params):
        with pytest.raises(ValueError, match=r"rA must be in \[-1, 1\]"):
            run_simulation(**{**default_params, "rA": 1.5})

    def test_rE_out_of_range_raises(self, default_params):
        with pytest.raises(ValueError, match=r"rE must be in \[-1, 1\]"):
            run_simulation(**{**default_params, "rE": 1.5})

    def test_p_mztwin_equals_one_raises(self, default_params):
        with pytest.raises(ValueError, match=r"p_mztwin must be in \[0, 1\)"):
            run_simulation(**{**default_params, "p_mztwin": 1.0})

    def test_assort_out_of_range_raises(self, default_params):
        with pytest.raises(ValueError, match=r"assort1 must be in \[-1, 1\]"):
            run_simulation(**{**default_params, "assort1": 1.5})
        with pytest.raises(ValueError, match=r"assort2 must be in \[-1, 1\]"):
            run_simulation(**{**default_params, "assort2": -1.5})

    def test_rho_w_one_raises(self, default_params):
        """Both-trait assort with perfectly correlated traits should raise."""
        with pytest.raises(ValueError, match="rho_w"):
            run_simulation(
                **{
                    **default_params,
                    "A1": 0.5,
                    "C1": 0.5,
                    "A2": 0.5,
                    "C2": 0.5,
                    "rA": 1.0,
                    "rC": 1.0,
                    "assort1": 0.3,
                    "assort2": 0.3,
                }
            )

    def test_assort_zero_preserves_rng(self, default_params):
        """assort=0 should produce identical output to no assort params."""
        ped1 = run_simulation(**{**default_params, "assort1": 0.0, "assort2": 0.0})
        ped2 = run_simulation(**default_params)
        assert_frame_equal(ped1, ped2)


# ---------------------------------------------------------------------------
# _assortative_pair_partners
# ---------------------------------------------------------------------------


class TestAssortativePairPartners:
    def _make_pop(self, rng, n=2000):
        """Create a test population with known pheno."""
        sex = rng.binomial(n=1, p=0.5, size=n)
        pheno = rng.standard_normal((n, 6))
        male_idxs = np.where(sex == 1)[0]
        female_idxs = np.where(sex == 0)[0]
        male_counts = np.ones(len(male_idxs), dtype=int)
        female_counts = np.ones(len(female_idxs), dtype=int)
        # Balance
        M = min(male_counts.sum(), female_counts.sum())
        male_counts = male_counts[:M]
        male_idxs = male_idxs[:M]
        female_counts = female_counts[:M]
        female_idxs = female_idxs[:M]
        return male_idxs, male_counts, female_idxs, female_counts, pheno

    def test_shape(self, rng):
        mi, mc, fi, fc, pheno = self._make_pop(rng)
        pairs = _assortative_pair_partners(rng, mi, mc, fi, fc, pheno, 0.3, 0.0, rho_w=0.0)
        assert pairs.shape == (mc.sum(), 2)

    def test_positive_correlation(self, rng):
        mi, mc, fi, fc, pheno = self._make_pop(rng, 5000)
        pairs = _assortative_pair_partners(rng, mi, mc, fi, fc, pheno, 0.5, 0.0, rho_w=0.0)
        liab1_m = pheno[pairs[:, 0], :3].sum(axis=1)
        liab1_f = pheno[pairs[:, 1], :3].sum(axis=1)
        corr = np.corrcoef(liab1_m, liab1_f)[0, 1]
        assert corr > 0.15

    def test_negative_correlation(self, rng):
        mi, mc, fi, fc, pheno = self._make_pop(rng, 5000)
        pairs = _assortative_pair_partners(rng, mi, mc, fi, fc, pheno, -0.5, 0.0, rho_w=0.0)
        liab1_m = pheno[pairs[:, 0], :3].sum(axis=1)
        liab1_f = pheno[pairs[:, 1], :3].sum(axis=1)
        corr = np.corrcoef(liab1_m, liab1_f)[0, 1]
        assert corr < -0.15

    def test_negative_trait2_single_trait_branch(self, rng):
        mi, mc, fi, fc, pheno = self._make_pop(rng, 2000)
        pairs = _assortative_pair_partners(rng, mi, mc, fi, fc, pheno, 0.0, -0.5, rho_w=0.0)
        assert pairs.shape == (mc.sum(), 2)

    def test_dedup_breaks_when_no_non_duplicates(self, rng, monkeypatch):
        """Defensive branch: if all assortative pairs are flagged duplicate, stop swapping."""
        monkeypatch.setattr(simulate_mod, "_find_duplicate_pairs", lambda matings: np.ones(len(matings), dtype=bool))
        male_idxs = np.array([0, 1])
        female_idxs = np.array([2, 3])
        counts = np.ones(2, dtype=int)
        pheno = rng.standard_normal((4, 6))
        pairs = _assortative_pair_partners(rng, male_idxs, counts, female_idxs, counts, pheno, 0.3, 0.0)
        assert pairs.shape == (2, 2)

    def test_both_traits_positive(self, rng):
        mi, mc, fi, fc, pheno = self._make_pop(rng, 10000)
        pairs = _assortative_pair_partners(
            rng,
            mi,
            mc,
            fi,
            fc,
            pheno,
            0.4,
            0.3,
            rho_w=0.25,
        )
        liab1_m = pheno[pairs[:, 0], :3].sum(axis=1)
        liab1_f = pheno[pairs[:, 1], :3].sum(axis=1)
        liab2_m = pheno[pairs[:, 0], 3:].sum(axis=1)
        liab2_f = pheno[pairs[:, 1], 3:].sum(axis=1)
        corr1 = np.corrcoef(liab1_m, liab1_f)[0, 1]
        corr2 = np.corrcoef(liab2_m, liab2_f)[0, 1]
        assert abs(corr1 - 0.4) < 0.05
        assert abs(corr2 - 0.3) < 0.05

    def test_mixed_sign(self, rng):
        mi, mc, fi, fc, pheno = self._make_pop(rng, 10000)
        pairs = _assortative_pair_partners(
            rng,
            mi,
            mc,
            fi,
            fc,
            pheno,
            0.4,
            -0.3,
            rho_w=0.25,
        )
        liab1_m = pheno[pairs[:, 0], :3].sum(axis=1)
        liab1_f = pheno[pairs[:, 1], :3].sum(axis=1)
        liab2_m = pheno[pairs[:, 0], 3:].sum(axis=1)
        liab2_f = pheno[pairs[:, 1], 3:].sum(axis=1)
        corr1 = np.corrcoef(liab1_m, liab1_f)[0, 1]
        corr2 = np.corrcoef(liab2_m, liab2_f)[0, 1]
        assert abs(corr1 - 0.4) < 0.05
        assert abs(corr2 - (-0.3)) < 0.05

    def test_cross_trait_symmetric(self, rng):
        """Cross-trait mate correlations should be approximately symmetric."""
        mi, mc, fi, fc, pheno = self._make_pop(rng, 20000)
        pairs = _assortative_pair_partners(
            rng,
            mi,
            mc,
            fi,
            fc,
            pheno,
            0.4,
            0.3,
            rho_w=0.25,
        )
        liab1_m = pheno[pairs[:, 0], :3].sum(axis=1)
        liab1_f = pheno[pairs[:, 1], :3].sum(axis=1)
        liab2_m = pheno[pairs[:, 0], 3:].sum(axis=1)
        liab2_f = pheno[pairs[:, 1], 3:].sum(axis=1)
        corr12 = np.corrcoef(liab1_m, liab2_f)[0, 1]  # F1 x M2
        corr21 = np.corrcoef(liab2_m, liab1_f)[0, 1]  # F2 x M1
        assert abs(corr12 - corr21) < 0.05


# ---------------------------------------------------------------------------
# mating: assortative mating integration
# ---------------------------------------------------------------------------


class TestMatingAssortative:
    def test_assort_without_pheno_raises(self, rng):
        sex = rng.binomial(n=1, p=0.5, size=500)
        with pytest.raises(ValueError, match="pheno must be provided"):
            mating(rng, sex, 0.5, 0.02, pheno=None, assort1=0.3)


# ---------------------------------------------------------------------------
# resolve_per_gen_param
# ---------------------------------------------------------------------------


class TestResolvePerGenParam:
    def test_scalar_returns_constant_list(self):
        result = resolve_per_gen_param(0.3, 5, name="E1")
        assert result == [0.3, 0.3, 0.3, 0.3, 0.3]

    def test_dict_forward_fill(self):
        result = resolve_per_gen_param({0: 0.2, 2: 0.5}, 4, name="E1")
        assert result == [0.2, 0.2, 0.5, 0.5]

    def test_dict_all_keys(self):
        result = resolve_per_gen_param({0: 0.1, 1: 0.2, 2: 0.3}, 3, name="E1")
        assert result == [0.1, 0.2, 0.3]

    def test_negative_scalar_raises(self):
        with pytest.raises(ValueError, match="must be >= 0"):
            resolve_per_gen_param(-0.1, 3, name="E1")

    def test_negative_dict_value_raises(self):
        with pytest.raises(ValueError, match="must be >= 0"):
            resolve_per_gen_param({0: -0.1}, 3, name="E1")

    def test_no_key_le_zero_raises(self):
        with pytest.raises(ValueError, match="must have a key <= 0"):
            resolve_per_gen_param({2: 0.5}, 3, name="E1")

    def test_non_scalar_non_dict_raises(self):
        with pytest.raises(TypeError, match="must be a scalar or dict"):
            resolve_per_gen_param([0.1], 3, name="E1")

    def test_empty_dict_raises(self):
        with pytest.raises(ValueError, match="must not be empty"):
            resolve_per_gen_param({}, 3, name="E1")


# ---------------------------------------------------------------------------
# Per-generation variance
# ---------------------------------------------------------------------------


class TestPerGenerationVariance:
    def test_scalar_E_matches_old_behavior(self, default_params):
        """Scalar E1/E2 should produce identical results to omitting them."""
        ped_no_e = run_simulation(**default_params)
        ped_with_e = run_simulation(**{**default_params, "E1": 0.3, "E2": 0.3})
        assert_frame_equal(ped_no_e, ped_with_e)

    def test_per_gen_E_changes_variance(self):
        """Per-gen E should produce different variance in different generations."""
        params = dict(
            seed=42,
            N=5000,
            G_ped=3,
            G_sim=3,
            mating_lambda=0.5,
            p_mztwin=0.02,
            A1=0.5,
            C1=0.0,
            A2=0.5,
            C2=0.0,
            E1={0: 0.3, 2: 0.8},
            E2={0: 0.3, 2: 0.8},
            rA=0.0,
            rC=0.0,
        )
        ped = run_simulation(**params)

        # Check that E variance differs across generations
        gen0 = ped.filter(pl.col("generation") == 0)
        gen2 = ped.filter(pl.col("generation") == 2)
        var_E1_gen0 = gen0["E1"].var()
        var_E1_gen2 = gen2["E1"].var()

        # Gen 0: E1 ~ N(0, sqrt(0.3)), Var ≈ 0.3
        # Gen 2: E1 ~ N(0, sqrt(0.8)), Var ≈ 0.8
        assert var_E1_gen0 < 0.5, f"Gen 0 Var(E1) should be ~0.3, got {var_E1_gen0:.3f}"
        assert var_E1_gen2 > 0.5, f"Gen 2 Var(E1) should be ~0.8, got {var_E1_gen2:.3f}"

    def test_per_gen_E_total_variance(self):
        """Total liability variance should match A + C + E per generation."""
        params = dict(
            seed=99,
            N=10000,
            G_ped=3,
            G_sim=3,
            mating_lambda=0.5,
            p_mztwin=0.02,
            A1=0.5,
            C1=0.0,
            A2=0.5,
            C2=0.0,
            E1={0: 0.2, 1: 0.5, 2: 1.0},
            E2=0.3,
            rA=0.0,
            rC=0.0,
        )
        ped = run_simulation(**params)

        for gen_idx, expected_E1 in [(0, 0.2), (1, 0.5), (2, 1.0)]:
            gen = ped.filter(pl.col("generation") == gen_idx)
            liab_var = gen["liability1"].var()
            expected_total = 0.5 + 0.0 + expected_E1
            assert abs(liab_var - expected_total) < 0.15, (
                f"Gen {gen_idx}: Var(L1) = {liab_var:.3f}, expected ~{expected_total:.1f}"
            )


# ---------------------------------------------------------------------------
# Cross-trait unique environment correlation (rE)
# ---------------------------------------------------------------------------


class TestCrossTraitRE:
    def test_rE_zero_preserves_rng(self, default_params):
        """rE=0 (default) should produce bit-identical output to omitting rE."""
        ped1 = run_simulation(**default_params)
        ped2 = run_simulation(**{**default_params, "rE": 0.0})
        assert_frame_equal(ped1, ped2)

    def test_rE_positive_correlates_E(self):
        """With rE=0.5, E1 and E2 should be positively correlated within individuals."""
        params = dict(
            seed=42,
            N=5000,
            G_ped=2,
            G_sim=2,
            mating_lambda=0.5,
            p_mztwin=0.02,
            A1=0.5,
            C1=0.2,
            E1=0.3,
            A2=0.5,
            C2=0.2,
            E2=0.3,
            rA=0.0,
            rC=0.0,
            rE=0.5,
        )
        ped = run_simulation(**params)
        founders = ped.filter(pl.col("generation") == 0)
        corr = np.corrcoef(founders["E1"].to_numpy(), founders["E2"].to_numpy())[0, 1]
        assert abs(corr - 0.5) < 0.1, f"Expected rE ≈ 0.5, got {corr:.3f}"

    def test_rE_does_not_share_E_between_siblings(self):
        """rE correlates E across traits, not across individuals."""
        params = dict(
            seed=42,
            N=2000,
            G_ped=2,
            G_sim=2,
            mating_lambda=0.5,
            p_mztwin=0.02,
            A1=0.5,
            C1=0.2,
            E1=0.3,
            A2=0.5,
            C2=0.2,
            E2=0.3,
            rA=0.0,
            rC=0.0,
            rE=0.8,
        )
        ped = run_simulation(**params)
        non_founders = ped.filter(pl.col("mother") != -1)
        # Siblings should NOT share E values (E is unique per person)
        per_mother = non_founders.group_by("mother").agg(
            pl.col("E1").n_unique().alias("n_unique_E1"),
            pl.len().alias("n_sibs"),
        )
        multi_sib = per_mother.filter(pl.col("n_sibs") > 1)
        if len(multi_sib) > 0:
            assert (multi_sib["n_unique_E1"] > 1).all(), "Siblings should have different E1 values"


# ---------------------------------------------------------------------------
# Wright-Fisher mating model
# ---------------------------------------------------------------------------


class TestMatingWF:
    """Unit tests for the ``_mating_wf`` sex-structured WF sampler."""

    def test_shapes(self):
        rng = np.random.default_rng(42)
        sex = rng.binomial(size=200, n=1, p=0.5)
        parents, twins, hh = _mating_wf(rng, sex, N=200, generation=0)
        assert parents.shape == (200, 2)
        assert twins.shape == (0, 2)
        assert hh.shape == (200,)

    def test_household_grouped_by_mother(self):
        rng = np.random.default_rng(7)
        sex = rng.binomial(size=300, n=1, p=0.5)
        parents, _, hh = _mating_wf(rng, sex, N=300, generation=0)
        # Every offspring sharing a mother shares a household.
        df = pl.DataFrame({"mother": parents[:, 0], "household": hh})
        assert (df.group_by("mother").agg(pl.col("household").n_unique())["household"] == 1).all()

    def test_household_ids_contiguous_from_zero(self):
        rng = np.random.default_rng(11)
        sex = rng.binomial(size=200, n=1, p=0.5)
        _, _, hh = _mating_wf(rng, sex, N=200, generation=0)
        n_hh = int(hh.max()) + 1
        assert set(hh.tolist()) == set(range(n_hh))

    def test_no_females_raises(self):
        rng = np.random.default_rng(3)
        sex = np.ones(50, dtype=int)  # all male
        with pytest.raises(ValueError, match=r"generation 5.*female"):
            _mating_wf(rng, sex, N=50, generation=5)

    def test_no_males_raises(self):
        rng = np.random.default_rng(4)
        sex = np.zeros(50, dtype=int)  # all female
        with pytest.raises(ValueError, match=r"generation 9.*male"):
            _mating_wf(rng, sex, N=50, generation=9)


class TestMatingDispatcher:
    """The public ``mating()`` routes by ``mating_model``."""

    def test_default_routes_to_standard(self):
        rng = np.random.default_rng(5)
        sex = rng.binomial(size=200, n=1, p=0.5)
        # Default mating_model="standard" — should use ZTP path and may produce twins.
        parents, twins, _ = mating(rng, sex, mating_lambda=0.5, p_mztwin=0.1)
        assert parents.shape == (200, 2)
        # With p_mztwin=0.1 we expect some twins on average; just check the call works.
        assert twins.shape[1] == 2

    def test_wf_routes_to_wf(self):
        rng = np.random.default_rng(5)
        sex = rng.binomial(size=200, n=1, p=0.5)
        _, twins, _ = mating(rng, sex, mating_model="wright_fisher", generation=0)
        assert twins.shape == (0, 2)

    def test_unknown_model_raises(self):
        rng = np.random.default_rng(5)
        sex = rng.binomial(size=200, n=1, p=0.5)
        with pytest.raises(ValueError, match="Unknown mating_model"):
            mating(rng, sex, mating_model="foo")


class TestRunSimulationWF:
    """End-to-end sanity for ``run_simulation(mating_model='wright_fisher')``."""

    @staticmethod
    def _wf_params(**overrides):
        # Inherited defaults of mating_lambda / p_mztwin must flow through
        # silently — WF's gating in run_simulation skips the standard-only
        # validation that would otherwise reject p_mztwin=0.02 etc.
        base = dict(
            seed=42,
            N=500,
            G_ped=3,
            mating_lambda=0.5,
            p_mztwin=0.02,
            A1=0.5,
            C1=0.0,
            E1=0.5,
            A2=0.4,
            C2=0.0,
            E2=0.6,
            rA=0.0,
            rC=0.0,
            rE=0.0,
            mating_model="wright_fisher",
        )
        base.update(overrides)
        return base

    def test_runs_with_inherited_standard_defaults(self):
        ped = run_simulation(**self._wf_params())
        # All WF offspring have twin == -1 (twins disabled).
        assert (ped["twin"] == -1).all()

    def test_household_grouped_by_mother(self):
        ped = run_simulation(**self._wf_params(seed=99))
        non_founders = ped.filter(pl.col("mother") != -1)
        # All offspring of the same mother share a household.
        hh_per_mother = non_founders.group_by("generation", "mother").agg(pl.col("household_id").n_unique())
        assert (hh_per_mother["household_id"] == 1).all()

    def test_C_still_works_under_wf(self):
        ped = run_simulation(**self._wf_params(C1=0.2, E1=0.3))
        # Household-grouped C: variance across households should be ≈ 0.2; total variance > 0.
        assert ped["C1"].var() > 0.05
        # Within a household, C is a single value.
        non_founders = ped.filter(pl.col("mother") != -1)
        c1_per_hh = non_founders.group_by("household_id").agg(pl.col("C1").n_unique())
        assert (c1_per_hh["C1"] == 1).all()

    def test_invalid_mating_model_raises(self):
        with pytest.raises(ValueError, match="mating_model must be"):
            run_simulation(**{**self._wf_params(), "mating_model": "foo"})

    def test_standard_model_unchanged_when_omitted(self):
        # Existing call sites that don't pass mating_model get standard behavior.
        params = {**self._wf_params()}
        del params["mating_model"]
        ped = run_simulation(**params)
        # Standard model produces some twins at p_mztwin=0.02.
        # With small N=500 / G_ped=3 this isn't guaranteed every seed but should
        # at least not be all -1 in expectation; just check the shape.
        assert "twin" in ped.columns

    def test_output_independent_of_mating_lambda(self):
        """Under WF, mating_lambda is a no-op — changing it must not change output."""
        ped_a = run_simulation(**self._wf_params(mating_lambda=0.5))
        ped_b = run_simulation(**self._wf_params(mating_lambda=10.0))
        assert_frame_equal(ped_a, ped_b)


class TestAssortMatrixValidation:
    """Validation branches for the explicit ``assort_matrix`` argument."""

    def test_wrong_shape_raises(self, default_params):
        with pytest.raises(ValueError, match="must be 2x2"):
            run_simulation(**{**default_params, "assort_matrix": [[0.1]]})

    def test_not_symmetric_raises(self, default_params):
        with pytest.raises(ValueError, match="must be symmetric"):
            run_simulation(**{**default_params, "assort_matrix": [[0.2, 0.05], [0.07, 0.2]]})

    def test_diagonal_out_of_range_raises(self, default_params):
        with pytest.raises(ValueError, match=r"assort_matrix\[0,0\]"):
            run_simulation(**{**default_params, "assort_matrix": [[1.5, 0.0], [0.0, 0.2]]})

    def test_with_per_gen_dict_assort_raises(self, default_params):
        """Per-gen dict assort* + assort_matrix is rejected (cross-AM ambiguity)."""
        with pytest.raises(ValueError, match="incompatible"):
            run_simulation(
                **{
                    **default_params,
                    "assort1": {0: 0.1},
                    "assort_matrix": [[0.2, 0.05], [0.05, 0.2]],
                }
            )

    def test_happy_path(self, default_params):
        """Valid assort_matrix runs end-to-end."""
        ped = run_simulation(**{**default_params, "assort_matrix": [[0.2, 0.05], [0.05, 0.15]]})
        assert len(ped) == default_params["N"] * default_params["G_ped"]
        assert "twin" in ped.columns


class TestPSDFailure:
    """The 4x4 mate-correlation matrix must be PSD across all generations."""

    def test_non_psd_off_diagonal_raises(self, default_params):
        # Force rho_w=0 by zeroing the cross-trait correlations, then push the
        # assort_matrix off-diagonal large enough that Sigma_4 = [[I, R_mf],
        # [R_mf, I]] has min-eigenvalue 1 - λ_max(R_mf) < 0.
        with pytest.raises(ValueError, match="not PSD"):
            run_simulation(
                **{
                    **default_params,
                    "rA": 0.0,
                    "rC": 0.0,
                    "rE": 0.0,
                    "assort_matrix": [[0.8, 0.95], [0.95, 0.8]],
                }
            )


class TestSimulateCLI:
    """End-to-end CLI for ``simace.simulation.simulate:cli``."""

    @staticmethod
    def _run_cli(monkeypatch, argv):
        import sys

        from simace.simulation.simulate import cli as simulate_cli

        monkeypatch.setattr(sys, "argv", ["simulate", *argv])
        simulate_cli()

    def test_writes_both_outputs(self, tmp_path, monkeypatch):
        import yaml

        out_pedigree = tmp_path / "pedigree.parquet"
        out_params = tmp_path / "params.yaml"

        self._run_cli(
            monkeypatch,
            [
                "--seed",
                "42",
                "--N",
                "100",
                "--G-ped",
                "2",
                "--G-sim",
                "2",
                "--E1",
                "0.3",
                "--E2",
                "0.3",
                "--A1",
                "0.5",
                "--A2",
                "0.5",
                "--C1",
                "0.2",
                "--C2",
                "0.2",
                "--output-pedigree",
                str(out_pedigree),
                "--output-params",
                str(out_params),
            ],
        )

        assert out_pedigree.exists()
        assert out_params.exists()

        ped = pl.read_parquet(out_pedigree)
        assert len(ped) == 100 * 2
        assert {"id", "mother", "father", "twin", "A1", "C1", "E1"}.issubset(ped.columns)

        params = yaml.safe_load(out_params.read_text())
        assert params["N"] == 100
        assert params["G_ped"] == 2
        assert params["mating_model"] == "standard"

    def test_assort_matrix_json_round_trip(self, tmp_path, monkeypatch):
        """--assort-matrix takes a JSON string and recovers the same values in params.yaml."""
        import yaml

        out_pedigree = tmp_path / "pedigree.parquet"
        out_params = tmp_path / "params.yaml"

        self._run_cli(
            monkeypatch,
            [
                "--seed",
                "7",
                "--N",
                "100",
                "--G-ped",
                "2",
                "--G-sim",
                "2",
                "--E1",
                "0.3",
                "--E2",
                "0.3",
                "--assort-matrix",
                "[[0.2, 0.05], [0.05, 0.15]]",
                "--output-pedigree",
                str(out_pedigree),
                "--output-params",
                str(out_params),
            ],
        )

        params = yaml.safe_load(out_params.read_text())
        assert params["assort_matrix"] == [[0.2, 0.05], [0.05, 0.15]]
