"""Unit tests for sim_ace.simulate functions."""

import numpy as np
import pandas as pd
import pytest

from sim_ace.simulate import (
    add_to_pedigree,
    allocate_offspring,
    assign_twins,
    balance_mating_slots,
    draw_mating_counts,
    generate_correlated_components,
    generate_mendelian_noise,
    mating,
    pair_partners,
    run_simulation,
)

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
        assert mask[2] and mask[3]
        assert not mask[0] and not mask[1] and not mask[4] and not mask[5]

    def test_no_twins_p_zero(self, rng):
        counts = np.array([3, 4, 5])
        mask = assign_twins(rng, counts, 0.0)
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
        np.testing.assert_array_equal(ped["id"].values, np.arange(N))

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
        np.testing.assert_array_equal(ped["id"].values, np.arange(2 * N))

    def test_liability_equals_sum(self, rng):
        N = 100
        pheno = rng.standard_normal((N, 6))
        sex = rng.binomial(n=1, p=0.5, size=N)
        parents = np.column_stack([np.arange(N), np.arange(N)])
        twins = np.array([], dtype=int).reshape(0, 2)
        hh = np.arange(N)

        ped = add_to_pedigree(pheno, sex, parents, twins, hh, generation=0, pedigree=None)
        np.testing.assert_allclose(
            ped["liability1"].values,
            ped["A1"].values + ped["C1"].values + ped["E1"].values,
        )
        np.testing.assert_allclose(
            ped["liability2"].values,
            ped["A2"].values + ped["C2"].values + ped["E2"].values,
        )

    def test_twin_column_bidirectional(self, rng):
        N = 100
        pheno = rng.standard_normal((N, 6))
        sex = rng.binomial(n=1, p=0.5, size=N)
        parents = np.column_stack([np.zeros(N, dtype=int), np.ones(N, dtype=int)])
        twins = np.array([[0, 1], [10, 11]])
        hh = np.arange(N)

        ped = add_to_pedigree(pheno, sex, parents, twins, hh, generation=0, pedigree=None)
        for t1, t2 in twins:
            assert ped.loc[t1, "twin"] == t2
            assert ped.loc[t2, "twin"] == t1


# ---------------------------------------------------------------------------
# run_simulation
# ---------------------------------------------------------------------------


class TestRunSimulation:
    def test_output_is_dataframe(self, default_params):
        ped = run_simulation(**default_params)
        assert isinstance(ped, pd.DataFrame)

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
        pd.testing.assert_frame_equal(ped1, ped2)

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
        with pytest.raises(ValueError):
            run_simulation(**{**default_params, "A1": -0.1})

    def test_A_plus_C_exceeds_one_raises(self, default_params):
        with pytest.raises(ValueError):
            run_simulation(**{**default_params, "A1": 0.6, "C1": 0.5})

    def test_negative_N_raises(self, default_params):
        with pytest.raises(ValueError):
            run_simulation(**{**default_params, "N": -10})

    def test_zero_mating_lambda_raises(self, default_params):
        with pytest.raises(ValueError):
            run_simulation(**{**default_params, "mating_lambda": 0})

    def test_G_sim_less_than_G_ped_raises(self, default_params):
        with pytest.raises(ValueError):
            run_simulation(**{**default_params, "G_sim": 1, "G_ped": 3})

    def test_rA_out_of_range_raises(self, default_params):
        with pytest.raises(ValueError):
            run_simulation(**{**default_params, "rA": 1.5})

    def test_p_mztwin_equals_one_raises(self, default_params):
        with pytest.raises(ValueError):
            run_simulation(**{**default_params, "p_mztwin": 1.0})
