"""Property-based censoring invariants (strengthened CLAUDE.md identity #6).

Beyond the boolean identity ``affected == ~(age_censored | death_censored)``,
two stronger relations cross-check the three separately-computed output arrays
over arbitrary onset times and observation windows:
  * every recorded observation time precedes death (``t_observed <= death_age``);
  * an affected individual's observed onset equals its raw onset — i.e. it was a
    real in-window event before death (``affected => t_observed == t_raw``).
  * a null raw onset has the same derived censoring outcome as a never-onset
    sentinel beyond every applicable right boundary.
A flipped clip direction, a wrong source array in the ``np.where`` death
substitution, or a desynced flag would break these where the boolean identity
alone (De Morgan of one assignment line) would not.
"""

import numpy as np
import polars as pl
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp

from simace.censoring.censor import run_censor

_onset = st.floats(min_value=0.0, max_value=300.0, allow_nan=False, allow_infinity=False)
_DERIVED_COLUMNS = (
    "age_censored1",
    "t_observed1",
    "death_censored1",
    "affected1",
    "age_censored2",
    "t_observed2",
    "death_censored2",
    "affected2",
)


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


def _with_selected_nulls(name: str, values: np.ndarray, selected: np.ndarray) -> pl.Series:
    return pl.Series(
        name,
        [None if replace else value for value, replace in zip(values, selected, strict=True)],
        dtype=pl.Float64,
    )


@given(case=_censor_case(), data=st.data())
def test_null_onset_matches_out_of_window_sentinel(case, data) -> None:
    """Null onsets behave like a sentinel strictly beyond every right boundary."""
    n, gens, t1, t2, censor_age, seed, gen_censoring = case
    phenotype, pedigree = _make_frames(n, gens, t1, t2)

    replace = data.draw(hnp.arrays(np.bool_, (2, n), elements=st.booleans())).copy()
    replace.flat[data.draw(st.integers(min_value=0, max_value=replace.size - 1))] = True

    effective_right = np.full(n, censor_age)
    for gen, (_, hi) in gen_censoring.items():
        effective_right[gens == gen] = hi
    assert np.all(effective_right < 1e6)

    null_phenotype = phenotype.with_columns(
        _with_selected_nulls("t1", t1, replace[0]),
        _with_selected_nulls("t2", t2, replace[1]),
    )
    sentinel_phenotype = phenotype.with_columns(
        pl.Series("t1", np.where(replace[0], 1e6, t1)),
        pl.Series("t2", np.where(replace[1], 1e6, t2)),
    )
    kwargs = {
        "censor_age": censor_age,
        "seed": seed,
        "gen_censoring": gen_censoring,
        "death_scale": 79.433,
        "death_rho": 10.0,
    }

    null_result = run_censor(null_phenotype, pedigree, **kwargs)
    sentinel_result = run_censor(sentinel_phenotype, pedigree, **kwargs)
    assert null_result.select(_DERIVED_COLUMNS).equals(sentinel_result.select(_DERIVED_COLUMNS))

    for trait, selected in (("1", replace[0]), ("2", replace[1])):
        affected = null_result[f"affected{trait}"].to_numpy()
        assert null_result[f"t{trait}"].null_count() == selected.sum()
        assert not affected[selected].any()
        assert np.isfinite(null_result[f"t_observed{trait}"].to_numpy()).all()
