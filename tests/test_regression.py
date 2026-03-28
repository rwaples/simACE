"""Regression tests: golden-seed scenario checked for bit-level reproducibility.

These tests ensure that for a fixed seed, the simulation produces
bit-identical output. Any change to the simulation logic or RNG usage
will cause these tests to fail, which is intentional — it forces
explicit acknowledgment when outputs change.

ACE components are stored as float32; liabilities as float64.
Golden values use float32-appropriate tolerances for ACE columns
and tight tolerances for liabilities.
"""

import numpy as np
import pytest

from sim_ace.simulation.simulate import run_simulation

# Golden-seed scenario: N=500, G_ped=2, seed=99999
GOLDEN_PARAMS = dict(
    seed=99999,
    N=500,
    G_ped=2,
    mating_lambda=0.5,
    p_mztwin=0.02,
    A1=0.5,
    C1=0.2,
    A2=0.4,
    C2=0.3,
    rA=0.3,
    rC=0.5,
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
        assert golden_pedigree["sex"].sum() == 500

    def test_twin_count(self, golden_pedigree):
        assert (golden_pedigree["twin"] != -1).sum() == 16

    # ACE columns are float32 — use float32-appropriate tolerance
    def test_A1_sum(self, golden_pedigree):
        assert golden_pedigree["A1"].values.sum() == pytest.approx(-50.37713, abs=1e-3)

    def test_C1_sum(self, golden_pedigree):
        assert golden_pedigree["C1"].values.sum() == pytest.approx(15.701193, abs=1e-3)

    def test_E1_sum(self, golden_pedigree):
        assert golden_pedigree["E1"].values.sum() == pytest.approx(-8.570223, abs=1e-3)

    # Liabilities are float64 — tight tolerance
    def test_liability1_sum(self, golden_pedigree):
        assert golden_pedigree["liability1"].values.sum() == pytest.approx(-43.24615936305929, abs=1e-10)

    def test_A2_sum(self, golden_pedigree):
        assert golden_pedigree["A2"].values.sum() == pytest.approx(-36.61953, abs=1e-3)

    def test_liability2_sum(self, golden_pedigree):
        assert golden_pedigree["liability2"].values.sum() == pytest.approx(-67.88303437335138, abs=1e-10)

    def test_row_0_values(self, golden_pedigree):
        row = golden_pedigree.iloc[0]
        assert row["A1"] == pytest.approx(0.517821729183197, abs=1e-6)
        assert row["C1"] == pytest.approx(-0.3015773296356201, abs=1e-6)
        assert row["E1"] == pytest.approx(-0.4798222780227661, abs=1e-6)

    def test_row_499_A1(self, golden_pedigree):
        assert golden_pedigree.iloc[499]["A1"] == pytest.approx(1.1855268478393555, abs=1e-6)

    def test_row_999_A1(self, golden_pedigree):
        assert golden_pedigree.iloc[999]["A1"] == pytest.approx(1.2123000621795654, abs=1e-6)

    def test_A1_first5(self, golden_pedigree):
        expected = [
            0.517821729183197,
            0.008222561329603195,
            -0.8315518498420715,
            -1.3413809537887573,
            -0.22008804976940155,
        ]
        np.testing.assert_allclose(golden_pedigree["A1"].values[:5], expected, atol=1e-6)

    def test_A1_last5(self, golden_pedigree):
        expected = [
            0.46030372381210327,
            1.2395710945129395,
            0.6966517567634583,
            1.003187656402588,
            1.2123000621795654,
        ]
        np.testing.assert_allclose(golden_pedigree["A1"].values[-5:], expected, atol=1e-6)

    def test_full_reproducibility(self, golden_pedigree):
        """Re-run with same params and verify bit-identical output."""
        ped2 = run_simulation(**GOLDEN_PARAMS)
        np.testing.assert_array_equal(golden_pedigree["A1"].values, ped2["A1"].values)
        np.testing.assert_array_equal(golden_pedigree["liability2"].values, ped2["liability2"].values)
        assert golden_pedigree["sex"].equals(ped2["sex"])
        assert golden_pedigree["twin"].equals(ped2["twin"])
