"""Property-based censoring invariants (strengthened CLAUDE.md identity #6).

Beyond the boolean identity ``affected == ~(age_censored | death_censored)``,
two stronger relations cross-check the three separately-computed output arrays
over arbitrary onset times and observation windows:
  * every recorded observation time precedes death (``t_observed <= death_age``);
  * an affected individual's observed onset equals its raw onset — i.e. it was a
    real in-window event before death (``affected => t_observed == t_raw``).
A flipped clip direction, a wrong source array in the ``np.where`` death
substitution, or a desynced flag would break these where the boolean identity
alone (De Morgan of one assignment line) would not.
"""

import numpy as np
import polars as pl
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp

from simace.censoring.censor import run_censor

_onset = st.floats(min_value=0.0, max_value=300.0, allow_nan=False, allow_infinity=False)


@st.composite
def _censor_case(draw):
    n = draw(st.integers(min_value=1, max_value=60))
    gens = draw(hnp.arrays(np.int64, n, elements=st.integers(0, 3)))
    t1 = draw(hnp.arrays(np.float64, n, elements=_onset))
    t2 = draw(hnp.arrays(np.float64, n, elements=_onset))
    censor_age = draw(st.floats(min_value=1.0, max_value=300.0, allow_nan=False, allow_infinity=False))
    seed = draw(st.integers(min_value=0, max_value=2**31 - 1))

    gen_censoring: dict[int, list[float]] = {}
    present = sorted({int(g) for g in gens})
    if present and draw(st.booleans()):
        for g in present:
            if draw(st.booleans()):
                lo = draw(st.floats(0.0, 100.0, allow_nan=False, allow_infinity=False))
                hi = lo + draw(st.floats(0.0, 100.0, allow_nan=False, allow_infinity=False))
                gen_censoring[g] = [lo, hi]
    return n, gens, t1, t2, censor_age, seed, gen_censoring


def _make_frames(n, gens, t1, t2):
    ids = np.arange(n)
    zeros = np.zeros(n)
    pedigree = pl.DataFrame(
        {
            "id": ids,
            "generation": gens,
            "sex": np.zeros(n, dtype=np.int64),
            "mother": np.full(n, -1),
            "father": np.full(n, -1),
            "twin": np.full(n, -1),
            "household_id": ids,
            "A1": zeros,
            "C1": zeros,
            "E1": zeros,
            "liability1": zeros,
            "A2": zeros,
            "C2": zeros,
            "E2": zeros,
            "liability2": zeros,
        }
    )
    phenotype = pl.DataFrame({"id": ids, "t1": t1, "t2": t2})
    return phenotype, pedigree


@settings(deadline=None, max_examples=60)
@given(_censor_case())
def test_censoring_identity_and_time_bounds(case):
    n, gens, t1, t2, censor_age, seed, gen_censoring = case
    phenotype, pedigree = _make_frames(n, gens, t1, t2)

    result = run_censor(
        phenotype,
        pedigree,
        censor_age=censor_age,
        seed=seed,
        gen_censoring=gen_censoring,
        death_scale=79.433,
        death_rho=10.0,
    )
    death_age = result["death_age"].to_numpy()

    for trait, t_raw in (("1", t1), ("2", t2)):
        affected = result[f"affected{trait}"].to_numpy()
        age_c = result[f"age_censored{trait}"].to_numpy()
        death_c = result[f"death_censored{trait}"].to_numpy()
        t_obs = result[f"t_observed{trait}"].to_numpy()

        # the load-bearing boolean identity
        np.testing.assert_array_equal(affected, ~age_c & ~death_c)
        # no observation can be recorded after death
        assert np.all(t_obs <= death_age + 1e-9)
        # affected => observed onset is the (unclipped, pre-death) raw onset
        if affected.any():
            np.testing.assert_array_equal(t_obs[affected], t_raw[affected])
