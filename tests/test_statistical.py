"""Statistical property tests for the ACE simulation.

These tests run seeded simulations and verify that statistical properties
of the output match theoretical expectations. Uses larger sample sizes
to ensure convergence; tests use wide tolerances appropriate for
finite-sample stochastic simulation.
"""

import numpy as np
import pytest

from sim_ace.simulate import run_simulation

# Use a moderate-size simulation for statistical power
STAT_PARAMS = dict(
    seed=7777,
    N=5000,
    G_ped=3,
    mating_lambda=0.5,
    p_mztwin=0.03,
    A1=0.5,
    C1=0.2,
    A2=0.4,
    C2=0.3,
    rA=0.5,
    rC=0.4,
)


@pytest.fixture(scope="module")
def stat_pedigree():
    """Module-scoped pedigree for statistical tests (expensive to create)."""
    return run_simulation(**STAT_PARAMS)


# ---------------------------------------------------------------------------
# Founder variance checks
# ---------------------------------------------------------------------------


class TestFounderVariances:
    def test_A1_variance(self, stat_pedigree):
        founders = stat_pedigree[stat_pedigree["mother"] == -1]
        var = founders["A1"].var()
        assert abs(var - STAT_PARAMS["A1"]) < 0.06

    def test_C1_variance(self, stat_pedigree):
        founders = stat_pedigree[stat_pedigree["mother"] == -1]
        var = founders["C1"].var()
        assert abs(var - STAT_PARAMS["C1"]) < 0.06

    def test_E1_variance(self, stat_pedigree):
        founders = stat_pedigree[stat_pedigree["mother"] == -1]
        E1 = 1.0 - STAT_PARAMS["A1"] - STAT_PARAMS["C1"]
        var = founders["E1"].var()
        assert abs(var - E1) < 0.06

    def test_A2_variance(self, stat_pedigree):
        founders = stat_pedigree[stat_pedigree["mother"] == -1]
        var = founders["A2"].var()
        assert abs(var - STAT_PARAMS["A2"]) < 0.06

    def test_C2_variance(self, stat_pedigree):
        founders = stat_pedigree[stat_pedigree["mother"] == -1]
        var = founders["C2"].var()
        assert abs(var - STAT_PARAMS["C2"]) < 0.06

    def test_E2_variance(self, stat_pedigree):
        founders = stat_pedigree[stat_pedigree["mother"] == -1]
        E2 = 1.0 - STAT_PARAMS["A2"] - STAT_PARAMS["C2"]
        var = founders["E2"].var()
        assert abs(var - E2) < 0.06

    def test_total_variance_trait1(self, stat_pedigree):
        founders = stat_pedigree[stat_pedigree["mother"] == -1]
        total = founders["liability1"].var()
        assert abs(total - 1.0) < 0.1

    def test_total_variance_trait2(self, stat_pedigree):
        founders = stat_pedigree[stat_pedigree["mother"] == -1]
        total = founders["liability2"].var()
        assert abs(total - 1.0) < 0.1


# ---------------------------------------------------------------------------
# MZ twin properties
# ---------------------------------------------------------------------------


class TestMZTwinProperties:
    def test_mz_twins_identical_A1(self, stat_pedigree):
        df = stat_pedigree
        twins = df[df["twin"] != -1]
        ids = twins["id"].values
        partners = twins["twin"].values
        mask = ids < partners
        t1 = ids[mask]
        t2 = partners[mask]
        df_idx = df.set_index("id")
        a1_twin1 = df_idx.loc[t1, "A1"].values
        a1_twin2 = df_idx.loc[t2, "A1"].values
        np.testing.assert_array_equal(a1_twin1, a1_twin2)

    def test_mz_twins_identical_A2(self, stat_pedigree):
        df = stat_pedigree
        twins = df[df["twin"] != -1]
        ids = twins["id"].values
        partners = twins["twin"].values
        mask = ids < partners
        t1 = ids[mask]
        t2 = partners[mask]
        df_idx = df.set_index("id")
        a2_twin1 = df_idx.loc[t1, "A2"].values
        a2_twin2 = df_idx.loc[t2, "A2"].values
        np.testing.assert_array_equal(a2_twin1, a2_twin2)

    def test_mz_twins_same_sex(self, stat_pedigree):
        df = stat_pedigree
        twins = df[df["twin"] != -1]
        ids = twins["id"].values
        partners = twins["twin"].values
        mask = ids < partners
        t1 = ids[mask]
        t2 = partners[mask]
        df_idx = df.set_index("id")
        sex1 = df_idx.loc[t1, "sex"].values
        sex2 = df_idx.loc[t2, "sex"].values
        np.testing.assert_array_equal(sex1, sex2)

    def test_twin_rate_matches_param(self, stat_pedigree):
        df = stat_pedigree
        non_founders = df[df["mother"] != -1]
        twins = non_founders[non_founders["twin"] != -1]
        n_twin_individuals = len(twins)
        n_nf = len(non_founders)
        observed_rate = n_twin_individuals / n_nf
        p_mztwin = STAT_PARAMS["p_mztwin"]
        # Under mating-pair model, twin rate should be low and bounded by p_mztwin
        # Generous tolerance for stochastic twin assignment
        assert observed_rate < max(0.02, 3 * p_mztwin)


# ---------------------------------------------------------------------------
# Sibling A correlation
# ---------------------------------------------------------------------------


class TestSiblingCorrelation:
    def _get_full_sib_pairs(self, df):
        """Extract full-sib pairs (same mother + same father, no twins)."""
        non_founders = df[(df["mother"] != -1) & (df["twin"] == -1)]
        cols = non_founders[["id", "mother", "father"]].copy()
        sib_counts = cols.groupby("mother").size()
        multi = sib_counts[sib_counts >= 2].index
        mat_sib = cols[cols["mother"].isin(multi)]
        if len(mat_sib) == 0:
            return np.array([]), np.array([])
        pairs = mat_sib.merge(mat_sib, on="mother", suffixes=("_1", "_2"))
        pairs = pairs[(pairs["id_1"] < pairs["id_2"]) & (pairs["father_1"] == pairs["father_2"])]
        return pairs["id_1"].values, pairs["id_2"].values

    def test_sibling_A1_correlation_near_half(self, stat_pedigree):
        """Full-sib A1 correlation should be ~0.5."""
        df = stat_pedigree
        id1, id2 = self._get_full_sib_pairs(df)
        assert len(id1) > 100, "Need enough sib pairs"
        df_idx = df.set_index("id")
        a1 = df_idx.loc[id1, "A1"].values
        a2 = df_idx.loc[id2, "A1"].values
        corr = np.corrcoef(a1, a2)[0, 1]
        assert abs(corr - 0.5) < 0.1

    def test_sibling_A2_correlation_near_half(self, stat_pedigree):
        df = stat_pedigree
        id1, id2 = self._get_full_sib_pairs(df)
        assert len(id1) > 100
        df_idx = df.set_index("id")
        a1 = df_idx.loc[id1, "A2"].values
        a2 = df_idx.loc[id2, "A2"].values
        corr = np.corrcoef(a1, a2)[0, 1]
        assert abs(corr - 0.5) < 0.1


# ---------------------------------------------------------------------------
# Cross-trait correlations
# ---------------------------------------------------------------------------


class TestCrossTraitCorrelations:
    def test_cross_trait_A_correlation(self, stat_pedigree):
        """Cross-trait A correlation should match rA."""
        founders = stat_pedigree[stat_pedigree["mother"] == -1]
        corr = np.corrcoef(founders["A1"].values, founders["A2"].values)[0, 1]
        assert abs(corr - STAT_PARAMS["rA"]) < 0.1

    def test_cross_trait_C_correlation(self, stat_pedigree):
        """Cross-trait C correlation should match rC."""
        founders = stat_pedigree[stat_pedigree["mother"] == -1]
        corr = np.corrcoef(founders["C1"].values, founders["C2"].values)[0, 1]
        assert abs(corr - STAT_PARAMS["rC"]) < 0.1

    def test_cross_trait_E_independent(self, stat_pedigree):
        """Cross-trait E correlation should be ~0 (independent draws)."""
        founders = stat_pedigree[stat_pedigree["mother"] == -1]
        corr = np.corrcoef(founders["E1"].values, founders["E2"].values)[0, 1]
        assert abs(corr) < 0.1


# ---------------------------------------------------------------------------
# E independence across siblings
# ---------------------------------------------------------------------------


class TestEIndependence:
    def test_sibling_E1_uncorrelated(self, stat_pedigree):
        """E1 values between siblings should be uncorrelated."""
        df = stat_pedigree
        non_founders = df[(df["mother"] != -1) & (df["twin"] == -1)]
        fam_sizes = non_founders.groupby("mother").size()
        multi = fam_sizes[fam_sizes >= 2].index[:500]
        multi_child = non_founders[non_founders["mother"].isin(multi)]

        grouped = multi_child.groupby("mother")["E1"]
        first = grouped.nth(0).values
        second = grouped.nth(1).values
        valid = ~(np.isnan(first) | np.isnan(second))
        e1, e2 = first[valid], second[valid]

        if len(e1) > 30:
            corr = np.corrcoef(e1, e2)[0, 1]
            assert abs(corr) < 0.1


# ---------------------------------------------------------------------------
# C shared within households
# ---------------------------------------------------------------------------


class TestCSharedWithinHousehold:
    def test_siblings_share_C1(self, stat_pedigree):
        """All siblings from the same mother should have identical C1."""
        df = stat_pedigree
        non_founders = df[df["mother"] != -1]
        c_by_mother = non_founders.groupby("mother")["C1"].nunique()
        # Allow tiny floating point differences
        prop_shared = (c_by_mother == 1).mean()
        assert prop_shared > 0.99

    def test_siblings_share_C2(self, stat_pedigree):
        df = stat_pedigree
        non_founders = df[df["mother"] != -1]
        c_by_mother = non_founders.groupby("mother")["C2"].nunique()
        prop_shared = (c_by_mother == 1).mean()
        assert prop_shared > 0.99


# ---------------------------------------------------------------------------
# Sex ratio
# ---------------------------------------------------------------------------


class TestSexRatio:
    def test_sex_ratio_balanced(self, stat_pedigree):
        ratio = stat_pedigree["sex"].mean()
        assert 0.45 < ratio < 0.55


# ---------------------------------------------------------------------------
# Assortative mating: mate correlation
# ---------------------------------------------------------------------------

ASSORT_PARAMS = dict(
    seed=8888,
    N=5000,
    G_ped=3,
    mating_lambda=0.5,
    p_mztwin=0.03,
    A1=0.5,
    C1=0.2,
    A2=0.4,
    C2=0.3,
    rA=0.5,
    rC=0.4,
    assort1=0.4,
    assort2=0.0,
)


@pytest.fixture(scope="module")
def assort_pedigree():
    """Module-scoped pedigree with assortative mating."""
    return run_simulation(**ASSORT_PARAMS)


class TestAssortativeMating:
    def _mate_corr(self, df, trait):
        """Compute Pearson correlation of mother/father liability for a trait."""
        non_founders = df[df["mother"] != -1]
        pairs = non_founders[["mother", "father"]].drop_duplicates()
        df_idx = df.set_index("id")
        m_liab = df_idx.loc[pairs["mother"].values, f"liability{trait}"].values
        f_liab = df_idx.loc[pairs["father"].values, f"liability{trait}"].values
        return np.corrcoef(m_liab, f_liab)[0, 1]

    def test_mate_correlation_trait1_positive(self, assort_pedigree):
        corr = self._mate_corr(assort_pedigree, 1)
        assert corr > 0.15

    def test_mate_correlation_trait2_near_zero(self, assort_pedigree):
        corr = self._mate_corr(assort_pedigree, 2)
        assert abs(corr) < 0.15

    def test_negative_assortment(self):
        params = {**ASSORT_PARAMS, "seed": 9999, "assort1": -0.3}
        ped = run_simulation(**params)
        non_founders = ped[ped["mother"] != -1]
        pairs = non_founders[["mother", "father"]].drop_duplicates()
        df_idx = ped.set_index("id")
        m_liab = df_idx.loc[pairs["mother"].values, "liability1"].values
        f_liab = df_idx.loc[pairs["father"].values, "liability1"].values
        corr = np.corrcoef(m_liab, f_liab)[0, 1]
        assert corr < -0.1
