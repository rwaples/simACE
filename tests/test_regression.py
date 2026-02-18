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
        assert golden_pedigree["sex"].sum() == 502

    def test_twin_count(self, golden_pedigree):
        assert (golden_pedigree["twin"] != -1).sum() == 16

    def test_A1_sum(self, golden_pedigree):
        assert golden_pedigree["A1"].values.sum() == pytest.approx(
            4.665457579645139, abs=1e-10
        )

    def test_C1_sum(self, golden_pedigree):
        assert golden_pedigree["C1"].values.sum() == pytest.approx(
            -0.7544175309886896, abs=1e-10
        )

    def test_E1_sum(self, golden_pedigree):
        assert golden_pedigree["E1"].values.sum() == pytest.approx(
            16.40091026097675, abs=1e-10
        )

    def test_liability1_sum(self, golden_pedigree):
        assert golden_pedigree["liability1"].values.sum() == pytest.approx(
            20.311950309633204, abs=1e-10
        )

    def test_A2_sum(self, golden_pedigree):
        assert golden_pedigree["A2"].values.sum() == pytest.approx(
            23.049678479828202, abs=1e-10
        )

    def test_liability2_sum(self, golden_pedigree):
        assert golden_pedigree["liability2"].values.sum() == pytest.approx(
            26.310218211948317, abs=1e-10
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
            0.2906551843340913, abs=1e-14
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
            1.3671976430106019,
            -0.9694012428067426,
            0.05568605810687821,
            -0.2921934333707687,
            0.2906551843340913,
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
