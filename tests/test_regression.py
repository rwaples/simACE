"""Regression tests: golden-seed scenario checked for bit-level reproducibility.

These tests ensure that for a fixed seed, the simulation produces
bit-identical output. Any change to the simulation logic or RNG usage
will cause these tests to fail, which is intentional — it forces
explicit acknowledgment when outputs change.
"""

import numpy as np
import pytest

from sim_ace.simulate import run_simulation

# Golden-seed scenario: N=500, G_ped=2, seed=99999
GOLDEN_PARAMS = dict(
    seed=99999,
    N=500,
    G_ped=2,
    fam_size=2.3,
    p_mztwin=0.02,
    p_nonsocial_father=0.05,
    A1=0.5, C1=0.2,
    A2=0.4, C2=0.3,
    rA=0.3, rC=0.5,
)


@pytest.fixture(scope="module")
def golden_pedigree():
    return run_simulation(**GOLDEN_PARAMS)


class TestGoldenSeedReproducibility:

    def test_shape(self, golden_pedigree):
        assert golden_pedigree.shape == (1000, 15)

    def test_generation_counts(self, golden_pedigree):
        for gen in range(2):
            assert (golden_pedigree["generation"] == gen).sum() == 500

    def test_sex_sum(self, golden_pedigree):
        assert golden_pedigree["sex"].sum() == 507

    def test_twin_count(self, golden_pedigree):
        assert (golden_pedigree["twin"] != -1).sum() == 28

    def test_A1_sum(self, golden_pedigree):
        assert golden_pedigree["A1"].values.sum() == pytest.approx(
            -16.620849382638458, abs=1e-10
        )

    def test_C1_sum(self, golden_pedigree):
        assert golden_pedigree["C1"].values.sum() == pytest.approx(
            -22.588583966904718, abs=1e-10
        )

    def test_E1_sum(self, golden_pedigree):
        assert golden_pedigree["E1"].values.sum() == pytest.approx(
            13.01105936616101, abs=1e-10
        )

    def test_liability1_sum(self, golden_pedigree):
        assert golden_pedigree["liability1"].values.sum() == pytest.approx(
            -26.198373983382158, abs=1e-10
        )

    def test_A2_sum(self, golden_pedigree):
        assert golden_pedigree["A2"].values.sum() == pytest.approx(
            13.583600440392479, abs=1e-10
        )

    def test_liability2_sum(self, golden_pedigree):
        assert golden_pedigree["liability2"].values.sum() == pytest.approx(
            -10.486373147888656, abs=1e-10
        )

    def test_row_0_values(self, golden_pedigree):
        row = golden_pedigree.iloc[0]
        assert row["A1"] == pytest.approx(-0.5824133490860317, abs=1e-14)
        assert row["C1"] == pytest.approx(0.04131489172699709, abs=1e-14)
        assert row["E1"] == pytest.approx(0.05415663482066094, abs=1e-14)

    def test_row_499_A1(self, golden_pedigree):
        assert golden_pedigree.iloc[499]["A1"] == pytest.approx(
            -0.2667646519008917, abs=1e-14
        )

    def test_row_999_A1(self, golden_pedigree):
        assert golden_pedigree.iloc[999]["A1"] == pytest.approx(
            0.5483731218910545, abs=1e-14
        )

    def test_A1_first5(self, golden_pedigree):
        expected = [
            -0.5824133490860317,
            0.681931458108332,
            0.22623076719782464,
            -0.21699347733277374,
            -0.5782277114326064,
        ]
        np.testing.assert_allclose(
            golden_pedigree["A1"].values[:5], expected, atol=1e-14
        )

    def test_A1_last5(self, golden_pedigree):
        expected = [
            0.053102412591784054,
            -1.1623134575111753,
            0.016029075118654312,
            0.6025076458945845,
            0.5483731218910545,
        ]
        np.testing.assert_allclose(
            golden_pedigree["A1"].values[-5:], expected, atol=1e-14
        )

    def test_full_reproducibility(self, golden_pedigree):
        """Re-run with same params and verify bit-identical output."""
        ped2 = run_simulation(**GOLDEN_PARAMS)
        np.testing.assert_array_equal(
            golden_pedigree["A1"].values, ped2["A1"].values
        )
        np.testing.assert_array_equal(
            golden_pedigree["liability2"].values, ped2["liability2"].values
        )
        assert golden_pedigree["sex"].equals(ped2["sex"])
        assert golden_pedigree["twin"].equals(ped2["twin"])
