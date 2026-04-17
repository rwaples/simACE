"""Mathematical postcondition checks on a single iter_reml fit.

These assertions are properties the REML solver must satisfy
at convergence regardless of exact σ² values.  One fit is reused
across all checks via ``fit_result``.
"""

from __future__ import annotations

import numpy as np
import pytest

from fit_ace.iter_reml.fit import fit_iter_reml
from tests.fit_ace.iter_reml.conftest import needs_bin


@pytest.fixture(scope="module")
def fit_result(tiny_ace_inputs):
    """Run one fit at module scope; every test below inspects it."""
    inp = tiny_ace_inputs
    return fit_iter_reml(
        y=inp["y"],
        kinship=inp["K"],
        household_id=inp["household_id"],
        iids=inp["iids"],
        phase1_probes=30,
        phase1_blocks=5,
        phase2_probes=30,
        max_iter=30,
        tol=1e-3,
        threads=1,
        log_level="warn",
    )


@needs_bin
class TestNumericalInvariants:
    def test_vc_positive_at_convergence(self, fit_result):
        vc = fit_result.vc.set_index("vc_name")["estimate"]
        for name in ("V(A)", "V(C)", "Ve"):
            assert float(vc[name]) > 0, f"{name} non-positive"

    def test_vp_equals_sum_of_components(self, fit_result):
        vc = fit_result.vc.set_index("vc_name")["estimate"]
        vp = float(vc["Vp"])
        vsum = float(vc["V(A)"]) + float(vc["V(C)"]) + float(vc["Ve"])
        assert vp == pytest.approx(vsum, abs=1e-6)

    def test_h2_c2_consistent_with_vc(self, fit_result):
        """h2 = V(A)/Vp, c2 = V(C)/Vp.  Tolerance 1e-5 accommodates the
        6-decimal rounding in the binary's TSV writer — h2 is computed
        and written at higher precision internally, then round-trips
        through ``%.6f`` formatting."""
        vc = fit_result.vc.set_index("vc_name")["estimate"]
        vp = float(vc["Vp"])
        assert float(vc["h2"]) == pytest.approx(float(vc["V(A)"]) / vp, abs=1e-5)
        assert float(vc["c2"]) == pytest.approx(float(vc["V(C)"]) / vp, abs=1e-5)

    def test_cov_matrix_symmetric(self, fit_result):
        cov = fit_result.cov.values
        assert cov.shape == (3, 3)
        np.testing.assert_allclose(cov, cov.T, atol=1e-8)

    def test_cov_diagonal_matches_vc_se(self, fit_result):
        """sqrt(diag(AI⁻¹)) must equal the SE column in vc.tsv for the
        three fitted components.  Derived rows (Vp, h2, c2) use a
        different formula (delta method) so are not checked here."""
        cov_diag = np.sqrt(np.diag(fit_result.cov.values))
        cov_index = list(fit_result.cov.index)
        vc = fit_result.vc.set_index("vc_name")
        for name in ("V(A)", "V(C)", "Ve"):
            pos = cov_index.index(name)
            assert float(vc.loc[name, "se"]) == pytest.approx(float(cov_diag[pos]), rel=1e-4)

    def test_grad_norm_decreases(self, fit_result):
        g = fit_result.iter_log["grad_norm"].to_numpy()
        if len(g) < 2:
            pytest.skip("need ≥2 iterations to check gradient decrease")
        assert g[-1] < g[0], f"grad_norm did not decrease: {g[0]:.3e} → {g[-1]:.3e}"

    def test_pcg_iters_never_hit_cap(self, fit_result):
        """If PCG hits pcg_max_iter on any outer iteration, the solver
        bailed out with an unconverged inner solve — σ²/logLik are
        unreliable."""
        pcg_max_iter = 500  # matches fast_kwargs default
        assert (fit_result.iter_log["pcg_iters_avg"] < pcg_max_iter).all()

    def test_vc_matches_final_iter_log_row(self, fit_result):
        """The last row of iter_log carries the σ² that get written to
        vc.tsv.  Verify the handshake is consistent."""
        last = fit_result.iter_log.iloc[-1]
        vc = fit_result.vc.set_index("vc_name")["estimate"]
        assert float(vc["V(A)"]) == pytest.approx(float(last["VC_A"]), rel=1e-6)
        assert float(vc["V(C)"]) == pytest.approx(float(last["VC_C"]), rel=1e-6)
        assert float(vc["Ve"]) == pytest.approx(float(last["VC_E"]), rel=1e-6)

    def test_wall_s_positive_finite(self, fit_result):
        assert np.isfinite(fit_result.wall_s)
        assert fit_result.wall_s > 0
