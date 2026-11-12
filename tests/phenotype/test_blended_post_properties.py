"""Property-based invariants of ``blended_diagnosis``.

Example tests in ``test_blended_post.py`` pin rates and edge cases; these
check, over arbitrary generations, liabilities, death ages, weights and
prevalences:
  * the censoring identity ``affected1 == ~(age_censored1 | death_censored1)``
    and ``t_observed1 <= min(death_age, MAX_AGE)`` (past bug 8e939ae left
    non-cases at ``MAX_AGE`` past their death);
  * raising one generation's prevalence only adds affected rows in that
    generation and leaves every other generation's rows unchanged;
  * a generation with ``alpha == 0`` never reads the trait-2 columns;
  * the input frame is not mutated (past bug c60cce5, in ``death_censor``).
"""

from __future__ import annotations

import numpy as np
import polars as pl
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp
from polars.testing import assert_frame_equal

from simace.phenotype.blended_post import blended_diagnosis

MAX_AGE = 80.0

_VALUES = st.floats(min_value=-5.0, max_value=5.0, allow_nan=False, allow_infinity=False)
_DEATH = st.one_of(st.just(MAX_AGE), st.floats(min_value=0.0, max_value=150.0, allow_nan=False))
_PREVALENCE = st.floats(min_value=0.01, max_value=0.99)
_ALPHA = st.one_of(st.just(0.0), st.just(1.0), st.floats(min_value=0.0, max_value=1.0))
_TRAIT2 = ("A2", "C2", "E2", "liability2")
_OUTPUTS = ("affected1", "t_observed1", "age_censored1", "death_censored1")


@st.composite
def blended_case(draw) -> tuple[pl.DataFrame, dict[int, float], dict[int, float]]:
    labels = draw(st.lists(st.integers(min_value=0, max_value=20), min_size=1, max_size=3, unique=True))
    gen = np.concatenate([np.full(draw(st.integers(min_value=1, max_value=40)), g) for g in labels])
    n = len(gen)
    columns = {c: draw(hnp.arrays(np.float64, n, elements=_VALUES)) for c in ("A1", "C1", "E1", "A2", "C2", "E2")}
    frame = pl.DataFrame(
        {
            "id": np.arange(n),
            "generation": gen,
            **columns,
            "liability1": columns["A1"] + columns["C1"] + columns["E1"],
            "liability2": columns["A2"] + columns["C2"] + columns["E2"],
            "death_age": draw(hnp.arrays(np.float64, n, elements=_DEATH)),
            "affected1": np.zeros(n, dtype=bool),
            "t_observed1": np.full(n, MAX_AGE),
            "age_censored1": np.ones(n, dtype=bool),
            "death_censored1": np.zeros(n, dtype=bool),
        }
    )
    alpha = {g: draw(_ALPHA) for g in labels}
    K = {g: draw(_PREVALENCE) for g in labels}
    return frame, alpha, K


@given(case=blended_case())
def test_censoring_identity_and_observation_bound(case) -> None:
    frame, alpha, K = case
    out = blended_diagnosis(frame, censor_age=MAX_AGE, gen_censoring={}, alpha_by_gen=alpha, K_by_gen=K)
    affected = out["affected1"].to_numpy()
    age_c = out["age_censored1"].to_numpy()
    death_c = out["death_censored1"].to_numpy()
    np.testing.assert_array_equal(affected, ~(age_c | death_c))
    bound = np.minimum(frame["death_age"].to_numpy(), MAX_AGE)
    assert np.all(out["t_observed1"].to_numpy() <= bound)


@given(case=blended_case(), data=st.data())
def test_raising_one_generations_prevalence_only_adds_its_cases(case, data) -> None:
    """Affected rows in the raised generation are a superset; other generations are unchanged.

    The threshold falls and every case's onset moves earlier as K rises, so no
    case is lost to right-censoring or death. Each row reads only its own
    generation's K, so the other generations do not see the change.
    """
    frame, alpha, K = case
    g = data.draw(st.sampled_from(sorted(K)))
    raised = K | {g: data.draw(st.floats(min_value=K[g], max_value=0.99))}
    before = blended_diagnosis(frame, censor_age=MAX_AGE, gen_censoring={}, alpha_by_gen=alpha, K_by_gen=K)
    after = blended_diagnosis(frame, censor_age=MAX_AGE, gen_censoring={}, alpha_by_gen=alpha, K_by_gen=raised)

    in_g = (frame["generation"] == g).to_numpy()
    lost = before["affected1"].to_numpy() & ~after["affected1"].to_numpy()
    assert not lost.any(), frame.filter(pl.Series(lost))
    assert_frame_equal(before.filter(pl.Series(~in_g)), after.filter(pl.Series(~in_g)), check_exact=True)


@given(case=blended_case(), data=st.data())
def test_zero_alpha_ignores_trait_two(case, data) -> None:
    """Replacing trait-2 columns on rows whose generation has ``alpha == 0`` changes no output column."""
    frame, alpha, K = case
    zeroed = data.draw(st.sets(st.sampled_from(sorted(alpha)), min_size=1))
    alpha = alpha | dict.fromkeys(zeroed, 0.0)
    rows = frame["generation"].is_in(sorted(zeroed)).to_numpy()
    replaced = frame.with_columns(
        pl.Series(c, np.where(rows, data.draw(hnp.arrays(np.float64, len(frame), elements=_VALUES)), frame[c]))
        for c in _TRAIT2
    )
    assert_frame_equal(
        blended_diagnosis(replaced, censor_age=MAX_AGE, gen_censoring={}, alpha_by_gen=alpha, K_by_gen=K).drop(_TRAIT2),
        blended_diagnosis(frame, censor_age=MAX_AGE, gen_censoring={}, alpha_by_gen=alpha, K_by_gen=K).drop(_TRAIT2),
        check_exact=True,
    )


@given(case=blended_case())
def test_input_is_not_mutated(case) -> None:
    frame, alpha, K = case
    snapshot = frame.clone()
    out = blended_diagnosis(frame, censor_age=MAX_AGE, gen_censoring={}, alpha_by_gen=alpha, K_by_gen=K)
    assert_frame_equal(frame, snapshot, check_exact=True)
    assert set(_OUTPUTS) <= set(out.columns)
