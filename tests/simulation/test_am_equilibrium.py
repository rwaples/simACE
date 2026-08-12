"""Tests for the assortative-mating additive-variance equilibrium.

Covers the theory module (:mod:`simace.simulation.am_equilibrium`) — including
the algebraic identity with the Herzig et al. (2026) closed form — and the
validation check (:mod:`simace.analysis.validate.am_equilibrium`).
"""

import numpy as np
import pytest

from simace.analysis.validate import validate_am_equilibrium
from simace.simulation.am_equilibrium import am_equilibrium_variance, am_variance_trajectory
from simace.simulation.simulate import run_simulation


def _paper_a2(A_base: float, V_env: float, r_ho: float) -> float:
    """Herzig et al. (2026) AM-only a², with their g0²=A_base/2, e²=V_env.

    a² = [ sqrt((e²+2g0²)² − 8·r·e²·g0²) − (e²−2g0²) ] / (2(1−r)).
    Independent reimplementation used to verify the simACE-derived root.
    """
    g0_sq = A_base / 2.0
    disc = (V_env + 2 * g0_sq) ** 2 - 8 * r_ho * V_env * g0_sq
    return (np.sqrt(disc) - (V_env - 2 * g0_sq)) / (2 * (1 - r_ho))


class TestEquilibriumTheory:
    @pytest.mark.parametrize(
        ("A_base", "V_env", "r_ho"),
        [(0.6, 0.4, 0.5), (0.5, 0.5, 0.6), (0.3, 0.7, 0.2), (0.8, 0.2, 0.8), (0.5, 0.5, 0.3)],
    )
    def test_matches_herzig_closed_form(self, A_base, V_env, r_ho):
        """simACE recursion fixed point equals Herzig et al. (2026) a²."""
        assert am_equilibrium_variance(A_base, V_env, r_ho) == pytest.approx(_paper_a2(A_base, V_env, r_ho))

    def test_no_am_no_inflation(self):
        """r_ho = 0 -> equilibrium equals the founder variance; trajectory flat."""
        assert am_equilibrium_variance(0.6, 0.4, 0.0) == pytest.approx(0.6)
        traj = am_variance_trajectory(0.6, 0.4, 0.0, 20)
        np.testing.assert_allclose(traj, 0.6)

    def test_trajectory_converges_to_equilibrium(self):
        """The recursion converges to the closed-form equilibrium."""
        A_base, V_env, r_ho = 0.6, 0.4, 0.5
        traj = am_variance_trajectory(A_base, V_env, r_ho, 200)
        assert traj[0] == pytest.approx(A_base)  # founders
        assert traj[-1] == pytest.approx(am_equilibrium_variance(A_base, V_env, r_ho), abs=1e-9)

    def test_am_inflates_variance(self):
        """Positive assortment raises the equilibrium above the base variance."""
        assert am_equilibrium_variance(0.6, 0.4, 0.5) > 0.6

    def test_degenerate_inputs(self):
        assert am_equilibrium_variance(0.0, 0.4, 0.5) == 0.0
        assert am_equilibrium_variance(0.6, 0.4, 1.0) == float("inf")


def _params(**over):
    base = dict(
        seed=7,
        N=3000,
        G_ped=8,
        G_sim=8,
        mating_lambda=0.5,
        p_mztwin=0.0,
        A1=0.6,
        C1=0.0,
        E1=0.4,
        A2=0.6,
        C2=0.0,
        E2=0.4,
        rA=0.0,
        rC=0.0,
        rE=0.0,
        assort1=0.5,
        assort2=0.0,
        mating_model="standard",
    )
    base.update(over)
    return base


class TestValidation:
    def test_single_trait_am_passes(self):
        p = _params()
        df = run_simulation(**p)
        res = validate_am_equilibrium(df, p)
        assert "am_equilibrium_A1" in res
        assert "am_equilibrium_A2" not in res  # trait 2 does not assort
        check = res["am_equilibrium_A1"]
        assert check["passed"], check["details"]
        # Observed final-gen Var(A1) is inflated above the base A1=0.6.
        assert check["observed"] > 0.6
        assert check["equilibrium"] > 0.6

    def test_no_am_emits_no_checks(self):
        p = _params(assort1=0.0, assort2=0.0)
        df = run_simulation(**p)
        assert validate_am_equilibrium(df, p) == {}

    def test_wright_fisher_emits_no_checks(self):
        p = _params(mating_model="wright_fisher")
        df = run_simulation(**p)
        assert validate_am_equilibrium(df, p) == {}

    def test_bivariate_am_skips_with_reason(self):
        p = _params(assort2=0.5)
        df = run_simulation(**p)
        res = validate_am_equilibrium(df, p)
        assert res["am_equilibrium_A1"]["passed"]
        assert "cross-trait" in res["am_equilibrium_A1"]["details"]
