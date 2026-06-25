"""Property-based tests for the simulation core.

Two properties:
  * ``generate_correlated_components`` produces exact collinearity at |r|=1 for
    arbitrary standard deviations — the off-diagonal ``r*sd1*sd2`` covariance
    term is invisible when sd1==sd2==1 (where every existing test lives), so a
    mis-scaled off-diagonal would pass today and fail here.
  * ``run_simulation`` returns a structurally sound pedigree across the
    parameter space: N rows per generation, founders parentless, non-founders'
    parents in the previous generation with correct sex, and
    ``liability == A + C + E``. Offset/burn-in arithmetic is the class of bug
    that passes at one default config but breaks at edge N/G_ped.
"""

import warnings

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from simace.simulation.simulate import generate_correlated_components, run_simulation


@settings(deadline=None, max_examples=50)
@given(
    seed=st.integers(min_value=0, max_value=2**31 - 1),
    sd1=st.floats(min_value=0.1, max_value=5.0),
    sd2=st.floats(min_value=0.1, max_value=5.0),
    sign=st.sampled_from([1.0, -1.0]),
)
def test_correlated_components_collinear_at_unit_correlation(seed, sd1, sd2, sign):
    rng = np.random.default_rng(seed)
    with warnings.catch_warnings():
        # a rank-1 covariance at |r|=1 is PSD; ignore numpy's roundoff warning
        warnings.simplefilter("ignore", category=RuntimeWarning)
        comp1, comp2 = generate_correlated_components(rng, 300, sd1, sd2, sign)
    # samples lie exactly on the line sd1*comp2 == sign*sd2*comp1 through origin
    assert np.allclose(sd1 * comp2, sign * sd2 * comp1, rtol=1e-6, atol=1e-6)


# Fixed, valid variance decomposition (A + C + E == 1 per trait); the structural
# invariants do not depend on the variance values.
_VARIANCES = dict(A1=0.3, C1=0.2, A2=0.3, C2=0.2, E1=0.5, E2=0.5, rA=0.0, rC=0.0)


@pytest.mark.slow
@settings(deadline=None, max_examples=15, suppress_health_check=[HealthCheck.too_slow])
@given(
    seed=st.integers(min_value=0, max_value=2**31 - 1),
    N=st.integers(min_value=20, max_value=120),
    G_ped=st.integers(min_value=1, max_value=3),
    mating_lambda=st.floats(min_value=0.3, max_value=2.0),
    p_mztwin=st.floats(min_value=0.0, max_value=0.05),
)
def test_run_simulation_structural_integrity(seed, N, G_ped, mating_lambda, p_mztwin):
    ped = run_simulation(
        seed=seed,
        N=N,
        G_ped=G_ped,
        mating_lambda=mating_lambda,
        p_mztwin=p_mztwin,
        **_VARIANCES,
    )

    # exactly N individuals in each of G_ped consecutive generations
    assert len(ped) == N * G_ped
    uniq, counts = np.unique(ped["generation"].to_numpy(), return_counts=True)
    assert len(uniq) == G_ped
    assert np.all(counts == N)

    gen = ped["generation"].to_numpy()
    mother = ped["mother"].to_numpy()
    father = ped["father"].to_numpy()
    by_id = ped.set_index("id")
    gen_by_id = by_id["generation"]
    sex_by_id = by_id["sex"]

    founders = gen == uniq.min()
    assert np.all(mother[founders] == -1)
    assert np.all(father[founders] == -1)

    nf = ~founders
    if nf.any():
        mom, dad, child_gen = mother[nf], father[nf], gen[nf]
        assert np.all(np.isin(mom, by_id.index))
        assert np.all(np.isin(dad, by_id.index))
        # parents live exactly one generation back
        assert np.all(gen_by_id.loc[mom].to_numpy() == child_gen - 1)
        assert np.all(gen_by_id.loc[dad].to_numpy() == child_gen - 1)
        # mother is female (0), father is male (1)
        assert np.all(sex_by_id.loc[mom].to_numpy() == 0)
        assert np.all(sex_by_id.loc[dad].to_numpy() == 1)

    # liability is the sum of its variance components (float32 inputs)
    for s in ("1", "2"):
        components = ped[f"A{s}"].to_numpy() + ped[f"C{s}"].to_numpy() + ped[f"E{s}"].to_numpy()
        np.testing.assert_allclose(ped[f"liability{s}"].to_numpy(), components, atol=1e-5)
