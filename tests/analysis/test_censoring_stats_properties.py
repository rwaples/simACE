"""Censoring statistics agree with the censor stage's own flags.

Over arbitrary onsets (nulls included) and windows, run through the real censor
stage:
  * the cascade's fates partition the true cases (affected, or onset before
    ``censor_age``), and its observed, death-censored, left-truncated and
    right-censored counts equal the censor flags summed over those cases, and
    the window shares partition each generation by the same flags, both on the
    stage's float64 output and after the float32 narrowing the stats stage
    reads it through;
  * the observed cumulative incidence never exceeds the true one at any age.
"""

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp

from simace.analysis.stats.censoring import compute_censoring_cascade, compute_censoring_windows
from simace.analysis.stats.incidence import compute_cumulative_incidence
from simace.censoring.censor import run_censor
from simace.core.parquet import normalize_for_parquet
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


def _assert_cascade_matches_flags(df, raw_t, censor_age, gen_censoring, min_split_width=0.0) -> None:
    """Check the cascade and window shares on ``df`` against its flags; ``raw_t`` holds the float64 onsets the flags came from.

    The left/right split of age-censored cases is checked only for windows wider
    than ``min_split_width``: a narrowed onset cannot be placed on either side of
    a window narrower than its float32 resolution.
    """
    cascade = compute_censoring_cascade(df, censor_age, gen_censoring)
    gens = df["generation"].to_numpy()
    for trait in ("1", "2"):
        t = df[f"t{trait}"].to_numpy()
        affected = df[f"affected{trait}"].to_numpy()
        age_censored = df[f"age_censored{trait}"].to_numpy()
        death_censored = df[f"death_censored{trait}"].to_numpy()
        for g, (lo, hi) in gen_censoring.items():
            if hi <= lo:
                continue
            fates = cascade[f"trait{trait}"][f"gen{g}"]
            true_case = (gens == g) & ((t < censor_age) | affected)
            assert fates["true_affected"] == true_case.sum()
            parts = ("observed", "death_censored", "right_censored", "left_truncated")
            assert sum(fates[p] for p in parts) == fates["true_affected"]
            assert fates["observed"] == (affected & (gens == g)).sum()
            assert fates["death_censored"] == (death_censored & ~age_censored & true_case).sum()
            if hi - lo > min_split_width:
                assert fates["left_truncated"] == (age_censored & true_case & (raw_t[trait] < lo)).sum()

    shares = compute_censoring_windows(df, censor_age, gen_censoring)["generations"]
    for g, (lo, hi) in gen_censoring.items():
        in_gen = gens == g
        if not in_gen.any():
            continue
        for trait in ("1", "2"):
            share = shares[f"gen{g}"][f"trait{trait}"]
            age_censored = df[f"age_censored{trait}"].to_numpy()[in_gen]
            death_censored = df[f"death_censored{trait}"].to_numpy()[in_gen]
            parts = ("pct_affected", "left_censored", "right_censored", "death_censored")
            assert sum(share[p] for p in parts) == pytest.approx(1.0)
            assert share["death_censored"] == pytest.approx((death_censored & ~age_censored).mean())
            if hi - lo > min_split_width:
                left = age_censored & (raw_t[trait][in_gen] < lo)
                assert share["left_censored"] == pytest.approx(left.mean())


@pytest.mark.parametrize("narrow", [False, True], ids=["float64", "parquet"])
@given(case=_censor_case(), data=st.data())
def test_cascade_matches_the_censor_flags(narrow, case, data) -> None:
    df, censor_age, gen_censoring = _censored(case, data)
    raw_t = {trait: df[f"t{trait}"].to_numpy() for trait in ("1", "2")}
    if narrow:
        _assert_cascade_matches_flags(normalize_for_parquet(df), raw_t, censor_age, gen_censoring, min_split_width=1e-3)
    else:
        _assert_cascade_matches_flags(df, raw_t, censor_age, gen_censoring)


def test_cascade_matches_the_flags_when_onsets_tie_death_and_window_edges_in_float32() -> None:
    """Onsets within float32 resolution of the death age or a window edge keep the fates the flags give them.

    Regression for #85: the cascade compared the narrowed onset with the
    narrowed death age and window edges, so ties and swapped order moved cases
    between observed, death-censored, left-truncated and right-censored.
    """
    n = 6000
    gens = np.zeros(n, dtype=np.int64)
    lo, hi = 50.0, 90.0
    kwargs = {"censor_age": 200.0, "seed": 3, "gen_censoring": {0: [lo, hi]}, **_KWARGS}
    phenotype, pedigree = _make_frames(n, gens, np.zeros(n), np.zeros(n))
    death = run_censor(phenotype, pedigree, **kwargs)["death_age"].to_numpy()
    jitter = 1 + np.random.default_rng(1).uniform(-3e-8, 3e-8, (2, n))
    edges = np.where(np.arange(n) % 2 == 0, lo, hi)
    t1 = death * jitter[0]
    t2 = edges * jitter[1]
    phenotype, pedigree = _make_frames(n, gens, t1, t2)
    df = run_censor(phenotype, pedigree, **kwargs).with_columns(generation=gens)
    narrowed = normalize_for_parquet(df)
    assert (narrowed["t1"].to_numpy() == narrowed["death_age"].to_numpy()).any()
    assert np.isin(narrowed["t2"].to_numpy(), [lo, hi]).any()
    _assert_cascade_matches_flags(narrowed, {"1": t1, "2": t2}, kwargs["censor_age"], kwargs["gen_censoring"])


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
