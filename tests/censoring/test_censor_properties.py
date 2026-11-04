"""Property-based censoring invariants (strengthened CLAUDE.md identity #6).

Over arbitrary onsets (nulls included), observation windows and seeds:
  * every output column equals its closed form, ``c = clip(t_fill, left,
    right)`` with a null onset filled as +inf: ``t_observed == min(c,
    death_age)``, ``death_censored == c > death_age``, ``age_censored == (t <
    left) | (t > right)`` and ``affected == ~(age_censored | death_censored)``,
    with onsets tied exactly to death ages to separate ``>`` from ``>=``;
  * changing one trait's onsets leaves the other trait and the death draw
    unchanged;
  * scaling every time by a power of two scales the times and keeps the flags.
"""

import numpy as np
import polars as pl
import pytest
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp
from polars.testing import assert_frame_equal

from simace.censoring.censor import run_censor
from simace.core.trait_schema import TRAIT_CENSORED_COLUMNS

_KWARGS = {"death_scale": 79.433, "death_rho": 10.0}
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


def _with_selected_nulls(name: str, values: np.ndarray, selected: np.ndarray) -> pl.Series:
    return pl.Series(
        name,
        [None if replace else value for value, replace in zip(values, selected, strict=True)],
        dtype=pl.Float64,
    )


@pytest.mark.parametrize("changed_trait", ["1", "2"])
@given(case=_censor_case(), data=st.data())
def test_changing_one_raw_onset_leaves_the_other_trait_and_death_unchanged(case, data, changed_trait) -> None:
    """Replacing one trait's raw onsets, nulls included, leaves every other censoring output identical.

    Windows, seed, and row order are held. Rejects reading the wrong onset
    array for a trait and death draws that depend on onsets.
    """
    n, gens, t1, t2, censor_age, seed, gen_censoring = case
    phenotype, pedigree = _make_frames(n, gens, t1, t2)
    replacement = data.draw(hnp.arrays(np.float64, n, elements=_onset))
    nulls = data.draw(hnp.arrays(np.bool_, n, elements=st.booleans()))
    changed = phenotype.with_columns(_with_selected_nulls(f"t{changed_trait}", replacement, nulls))
    kwargs = {"censor_age": censor_age, "seed": seed, "gen_censoring": gen_censoring, **_KWARGS}

    held = [column for column in TRAIT_CENSORED_COLUMNS if not column.endswith(changed_trait)]
    assert_frame_equal(
        run_censor(changed, pedigree, **kwargs).select(held),
        run_censor(phenotype, pedigree, **kwargs).select(held),
        check_exact=True,
    )


def _windows(n, gens, censor_age, gen_censoring):
    left = np.zeros(n)
    right = np.full(n, float(censor_age))
    for gen, (lo, hi) in gen_censoring.items():
        left[gens == gen] = lo
        right[gens == gen] = hi
    return left, right


@given(case=_censor_case(), data=st.data())
def test_censoring_matches_closed_form(case, data) -> None:
    """Every censoring column equals its closed form, including null onsets and onsets tied with death.

    A null onset is filled as +inf and passes through unchanged in ``t{trait}``.
    With ``c = clip(t_fill, left, right)``: ``t_observed == min(c, death_age)``,
    ``death_censored == c > death_age``, ``age_censored == (t < left) | (t > right)``
    and ``affected == ~(age_censored | death_censored)``. The death draw depends only on seed and row count, so a first run supplies
    death ages that the second run's onsets can tie exactly; that tie separates
    ``>`` from ``>=`` in the death flag.
    """
    n, gens, t1, t2, censor_age, seed, gen_censoring = case
    kwargs = {"censor_age": censor_age, "seed": seed, "gen_censoring": gen_censoring, **_KWARGS}
    phenotype, pedigree = _make_frames(n, gens, t1, t2)
    death_age = run_censor(phenotype, pedigree, **kwargs)["death_age"].to_numpy()

    tie = data.draw(hnp.arrays(np.bool_, (2, n), elements=st.booleans()))
    nulls = data.draw(hnp.arrays(np.bool_, (2, n), elements=st.booleans())) & ~tie
    raw = [np.where(tie[i], death_age, t) for i, t in enumerate((t1, t2))]
    phenotype = phenotype.with_columns(
        _with_selected_nulls("t1", raw[0], nulls[0]),
        _with_selected_nulls("t2", raw[1], nulls[1]),
    )
    result = run_censor(phenotype, pedigree, **kwargs)
    np.testing.assert_array_equal(result["death_age"].to_numpy(), death_age)

    left, right = _windows(n, gens, censor_age, gen_censoring)
    for i, trait in enumerate(("1", "2")):
        t_fill = np.where(nulls[i], np.inf, raw[i])
        c = np.clip(t_fill, left, right)
        np.testing.assert_array_equal(result[f"t_observed{trait}"].to_numpy(), np.minimum(c, death_age))
        np.testing.assert_array_equal(result[f"death_censored{trait}"].to_numpy(), c > death_age)
        age_censored = (t_fill < left) | (t_fill > right)
        np.testing.assert_array_equal(result[f"age_censored{trait}"].to_numpy(), age_censored)
        np.testing.assert_array_equal(result[f"affected{trait}"].to_numpy(), ~age_censored & ~(c > death_age))
        assert result[f"t{trait}"].equals(phenotype[f"t{trait}"])


@given(case=_censor_case(), k=st.integers(min_value=1, max_value=8), data=st.data())
def test_censoring_is_equivariant_under_power_of_two_time_scaling(case, k, data) -> None:
    """Scaling onsets, windows, censor_age and death_scale by 2**k scales times by 2**k and keeps every flag.

    Multiplying by a positive power of two is exact (no overflow at these
    magnitudes), so the comparison is bit-exact: a censoring rule that mixed
    in a fixed age constant would break it.
    """
    n, gens, t1, t2, censor_age, seed, gen_censoring = case
    nulls = data.draw(hnp.arrays(np.bool_, (2, n), elements=st.booleans()))
    s = 2.0**k
    phenotype, pedigree = _make_frames(n, gens, t1, t2)

    def run(scale):
        frame = phenotype.with_columns(
            _with_selected_nulls("t1", t1 * scale, nulls[0]),
            _with_selected_nulls("t2", t2 * scale, nulls[1]),
        )
        return run_censor(
            frame,
            pedigree,
            censor_age=censor_age * scale,
            seed=seed,
            gen_censoring={g: [lo * scale, hi * scale] for g, (lo, hi) in gen_censoring.items()},
            death_scale=_KWARGS["death_scale"] * scale,
            death_rho=_KWARGS["death_rho"],
        )

    base, scaled = run(1.0), run(s)
    times = ["death_age", "t_observed1", "t_observed2"]
    flags = [c for c in _DERIVED_COLUMNS if not c.startswith("t_observed")]
    assert_frame_equal(scaled.select(flags), base.select(flags), check_exact=True)
    assert_frame_equal(scaled.select(times), base.select(pl.col(times) * s), check_exact=True)
