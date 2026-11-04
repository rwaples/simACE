"""Censoring statistics agree with the censor stage's own flags.

``compute_censoring_cascade``, ``compute_censoring_windows`` and
``compute_cumulative_incidence`` re-derive censoring fates from raw onsets,
windows and death ages instead of reading the flags ``run_censor`` wrote. Over
arbitrary onsets (nulls included) and windows, run through the real censor
stage:
  * the cascade's fates partition the true cases (onset before ``censor_age``),
    and its observed and death-censored counts equal the censor flags summed
    over those cases;
  * the observed cumulative incidence never exceeds the true one at any age.
"""

import numpy as np
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp

from simace.analysis.stats.censoring import compute_censoring_cascade, compute_censoring_windows
from simace.analysis.stats.incidence import compute_cumulative_incidence
from simace.censoring.censor import run_censor
from tests.censoring.test_censor_properties import _KWARGS, _censor_case, _make_frames, _with_selected_nulls


def _censored(case, data):
    """Run the censor stage on ``case`` with some raw onsets replaced by null, as never-onset rows arrive."""
    n, gens, t1, t2, censor_age, seed, gen_censoring = case
    phenotype, pedigree = _make_frames(n, gens, t1, t2)
    nulls = data.draw(hnp.arrays(np.bool_, (2, n), elements=st.booleans()))
    phenotype = phenotype.with_columns(
        _with_selected_nulls("t1", t1, nulls[0]), _with_selected_nulls("t2", t2, nulls[1])
    )
    out = run_censor(phenotype, pedigree, censor_age=censor_age, seed=seed, gen_censoring=gen_censoring, **_KWARGS)
    return out.with_columns(generation=gens), censor_age, gen_censoring


@given(case=_censor_case(), data=st.data())
def test_cascade_matches_the_censor_flags(case, data) -> None:
    df, censor_age, gen_censoring = _censored(case, data)
    cascade = compute_censoring_cascade(df, censor_age, gen_censoring)
    gens = df["generation"].to_numpy()
    for trait in ("1", "2"):
        t = df[f"t{trait}"].to_numpy()
        affected = df[f"affected{trait}"].to_numpy()
        death_censored = df[f"death_censored{trait}"].to_numpy()
        for g, (lo, hi) in gen_censoring.items():
            if hi <= lo:
                continue
            fates = cascade[f"trait{trait}"][f"gen{g}"]
            true_case = (gens == g) & (t < censor_age)
            assert fates["true_affected"] == true_case.sum()
            parts = ("observed", "death_censored", "right_censored", "left_truncated")
            assert sum(fates[p] for p in parts) == fates["true_affected"]
            assert fates["observed"] == (affected & true_case).sum()
            in_window = true_case & (t >= lo) & (t <= hi)
            assert fates["death_censored"] == (death_censored & in_window).sum()


@given(case=_censor_case(), data=st.data())
def test_observed_cumulative_incidence_never_exceeds_true(case, data) -> None:
    df, censor_age, gen_censoring = _censored(case, data)
    incidence = compute_cumulative_incidence(df, censor_age)
    for trait in ("trait1", "trait2"):
        curves = incidence[trait]
        assert np.all(np.asarray(curves["observed_values"]) <= np.asarray(curves["true_values"]))
    windows = compute_censoring_windows(df, censor_age, gen_censoring)
    for per_gen in windows["generations"].values():
        for trait in ("trait1", "trait2"):
            if trait in per_gen:
                curves = per_gen[trait]
                assert np.all(np.asarray(curves["observed_incidence"]) <= np.asarray(curves["true_incidence"]))
