"""Property-based algebraic invariants for the tetrachoric MLE.

Tetrachoric correlation is symmetric and equivariant under affection relabeling:
``r(a,b)==r(b,a)``, ``r(~a,~b)==r(a,b)``, ``r(~a,b)==-r(a,b)``, with
``-1<=r<=1``. The estimator canonicalizes equivalent contingency tables before
optimization, so these identities must remain exact even when the likelihood is
flat near the correlation bracket.
"""

import numpy as np
from hypothesis import assume, given
from hypothesis import strategies as st
from scipy.special import ndtri

from simace.analysis.stats.tetrachoric import tetrachoric_corr


@st.composite
def _binary_pair(draw):
    """Correlated binary arrays with controlled, interior marginal prevalences.

    Built by thresholding a latent bivariate normal so the two variables carry a
    real (sign-varying) association — exercising both internal MLE branches —
    while keeping prevalences in [0.2, 0.8] without rejection-heavy filtering.
    """
    n = draw(st.integers(min_value=80, max_value=300))
    seed = draw(st.integers(min_value=0, max_value=2**31 - 1))
    p_a = draw(st.floats(0.2, 0.8))
    p_b = draw(st.floats(0.2, 0.8))
    rho = draw(st.floats(-0.8, 0.8))

    rng = np.random.default_rng(seed)
    z1 = rng.standard_normal(n)
    z2 = rho * z1 + np.sqrt(1.0 - rho**2) * rng.standard_normal(n)
    a = z1 > ndtri(1.0 - p_a)
    b = z2 > ndtri(1.0 - p_b)
    # both marginals must stay strictly interior (rarely fails for n >= 80)
    assume(0 < a.mean() < 1 and 0 < b.mean() < 1)
    return a, b


@given(_binary_pair())
def test_tetrachoric_symmetry_relabel_bounds(ab):
    a, b = ab
    r = tetrachoric_corr(a, b)
    assume(not np.isnan(r))

    assert -1.0 - 1e-9 <= r <= 1.0 + 1e-9
    assert tetrachoric_corr(b, a) == r
    assert tetrachoric_corr(~a, ~b) == r
    assert tetrachoric_corr(~a, b) == -r
