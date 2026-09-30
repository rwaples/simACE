"""Property-based tests for the simulation core.

Properties:
  * ``generate_correlated_components`` produces exact collinearity at |r|=1 for
    arbitrary standard deviations — the off-diagonal ``r*sd1*sd2`` covariance
    term is invisible when sd1==sd2==1 (where every existing test lives), so a
    mis-scaled off-diagonal would pass today and fail here.
  * ``mating`` assigns one household per mother under both mating models.
  * ``run_simulation`` returns a structurally and scientifically sound
    pedigree across mating models, variance components (exact zeros included),
    E schedules, and burn-in: N rows per generation, parent offsets and sexes,
    reciprocal twin links, household C, maternal households, scheduled zero
    components, and ``liability == A + C + E`` within float32 storage error.
    Offset/burn-in arithmetic is the class of bug that passes at one default
    config but breaks at edge N/G_ped.
  * The recording window only selects rows: two runs differing only in G_ped
    agree exactly on their overlapping generations.
"""

import warnings

import numpy as np
import polars as pl
import pytest
from hypothesis import HealthCheck, event, given, settings
from hypothesis import strategies as st
from polars.testing import assert_frame_equal

from simace.simulation.simulate import generate_correlated_components, mating, run_simulation


# Explicit budget, not the conftest profile: every example calls run_simulation,
# so HYPOTHESIS_PROFILE=thorough would dominate suite wall-clock here.
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
    # Samples lie on the line sd1*comp2 == sign*sd2*comp1 through the origin.
    #
    # Compare against the vector scale, not elementwise. At |r|=1 the covariance
    # is singular (rank-1), so the decomposition's accuracy degrades to roughly
    # sqrt(float64 eps) and the error tracks the magnitude of the whole draw
    # rather than of each sample. An elementwise rtol/atol therefore fails on
    # whichever sample lands nearest zero, since its own magnitude gives it no
    # budget -- a property of the assertion, not of the generator.
    #
    # The deviation grows with the sd ratio, which sets how ill-conditioned the
    # rank-1 covariance is: measured worst-of-300-seeds is 0 at sd1 == sd2 and
    # 2.0e-7 at the (0.1, 4.9) corner of the strategy's range. 1e-5 keeps ~50x
    # margin over that while staying ~5 orders of magnitude below a real break
    # in collinearity, which would be O(1).
    lhs = sd1 * comp2
    rhs = sign * sd2 * comp1
    scale = max(float(np.abs(rhs).max()), 1.0)
    assert np.abs(lhs - rhs).max() <= 1e-5 * scale


_MATING_MODELS = ["standard", "wright_fisher"]
_E_TRANSITIONS = ["none", "before_first_recorded", "at_first_recorded", "after_first_recorded"]
_SEEDS = st.integers(min_value=0, max_value=2**31 - 1)
_VARIANCES = st.one_of(st.just(0.0), st.floats(min_value=0.05, max_value=1.0))
_CORRELATIONS = st.one_of(st.sampled_from([-1.0, 0.0, 1.0]), st.floats(min_value=-1.0, max_value=1.0))


def _canonical_labels(labels: np.ndarray) -> np.ndarray:
    """Relabel groups 0, 1, ... by first row, so equal partitions compare equal."""
    _, first_row, inverse = np.unique(labels, return_index=True, return_inverse=True)
    return np.argsort(np.argsort(first_row))[inverse]


def _is_bijection(x: np.ndarray, y: np.ndarray) -> bool:
    n_pairs = len(np.unique(np.column_stack([x, y]), axis=0))
    return n_pairs == len(np.unique(x)) == len(np.unique(y))


def _constant_within(values: np.ndarray, groups: np.ndarray) -> bool:
    representative = np.empty(groups.max() + 1, dtype=values.dtype)
    representative[groups] = values
    return np.array_equal(values, representative[groups])


def _scheduled(value: float | dict[int, float], iteration: int) -> float:
    if isinstance(value, dict):
        return value[max(k for k in value if k <= iteration)]
    return value


@pytest.mark.parametrize("mating_model", _MATING_MODELS)
@given(
    seed=_SEEDS,
    n_female=st.integers(min_value=1, max_value=30),
    n_male=st.integers(min_value=1, max_value=30),
    mating_lambda=st.floats(min_value=0.3, max_value=2.0),
    p_mztwin=st.floats(min_value=0.0, max_value=0.9),
    data=st.data(),
)
def test_mating_households_are_maternal(mating_model, seed, n_female, n_male, mating_lambda, p_mztwin, data):
    """Children share a household exactly when they share a mother, with households labelled 0..k-1.

    Holds for any parent sex layout with at least one parent of each sex.
    Rejects grouping households by mating or by father.
    """
    sex = np.array(data.draw(st.permutations([0] * n_female + [1] * n_male)))
    parents, _, households = mating(
        np.random.default_rng(seed), sex, mating_lambda, p_mztwin, mating_model=mating_model
    )
    assert np.array_equal(np.unique(households), np.arange(households.max() + 1))
    assert _is_bijection(parents[:, 0], households)


def _draw_shared_kwargs(draw, mating_model: str) -> dict:
    return dict(
        seed=draw(_SEEDS),
        N=draw(st.integers(min_value=20, max_value=120)),
        mating_model=mating_model,
        mating_lambda=draw(st.floats(min_value=0.3, max_value=2.0)),
        p_mztwin=draw(st.floats(min_value=0.0, max_value=0.9)),
        A1=draw(_VARIANCES),
        C1=draw(_VARIANCES),
        A2=draw(_VARIANCES),
        C2=draw(_VARIANCES),
        rA=draw(_CORRELATIONS),
        rC=draw(_CORRELATIONS),
        rE=draw(_CORRELATIONS),
    )


@st.composite
def simulation_kwargs(draw, mating_model: str, e_transition: str) -> dict:
    """``run_simulation`` inputs with zero assortment and G_sim >= G_ped.

    Each E schedule is a scalar (``e_transition == "none"``) or has one
    transition before, at, or after the first recorded raw iteration.
    """
    G_ped = draw(st.integers(min_value=2 if e_transition == "after_first_recorded" else 1, max_value=4))
    min_burnin = {"before_first_recorded": 2, "at_first_recorded": 1}.get(e_transition, 0)
    burnin = draw(st.integers(min_value=min_burnin, max_value=min_burnin + 3))
    G_sim = G_ped + burnin

    def e_schedule() -> float | dict[int, float]:
        if e_transition == "none":
            return draw(_VARIANCES)
        if e_transition == "before_first_recorded":
            key = draw(st.integers(min_value=1, max_value=burnin - 1))
        elif e_transition == "at_first_recorded":
            key = burnin
        else:
            key = draw(st.integers(min_value=burnin + 1, max_value=G_sim - 1))
        return {0: draw(_VARIANCES), key: draw(_VARIANCES)}

    return _draw_shared_kwargs(draw, mating_model) | dict(G_ped=G_ped, G_sim=G_sim, E1=e_schedule(), E2=e_schedule())


def _check_recorded_pedigree(ped: pl.DataFrame, kw: dict) -> None:
    N, G_ped = kw["N"], kw["G_ped"]
    burnin = kw["G_sim"] - G_ped
    col = {c: ped[c].to_numpy() for c in ped.columns}
    gen, mother, father, sex, twin, household = (
        col[c] for c in ("generation", "mother", "father", "sex", "twin", "household_id")
    )

    # Ids equal row positions, so ids index the columns directly below.
    assert np.array_equal(col["id"], np.arange(N * G_ped))
    assert np.array_equal(gen, np.repeat(np.arange(G_ped), N))

    founders = gen == 0
    assert np.all(mother[founders] == -1)
    assert np.all(father[founders] == -1)
    nf = ~founders
    mom, dad = mother[nf], father[nf]
    assert np.all((mom >= 0) & (dad >= 0))
    assert np.all(gen[mom] == gen[nf] - 1)
    assert np.all(gen[dad] == gen[nf] - 1)
    assert np.all(sex[mom] == 0)
    assert np.all(sex[dad] == 1)
    # Founders' mothers are unrecorded (-1), so maternal grouping is checked only below them.
    assert _is_bijection(mom, household[nf])
    for c in ("C1", "C2"):
        assert _constant_within(col[c], household), c

    paired = np.flatnonzero(twin >= 0)
    if kw["mating_model"] == "wright_fisher":
        assert paired.size == 0
    partner = twin[paired]
    assert np.all(partner != paired)
    assert np.array_equal(twin[partner], paired)
    for c in ("mother", "father", "sex", "generation", "A1", "C1", "A2", "C2"):
        assert np.array_equal(col[c][partner], col[c][paired]), c

    for c in ("A1", "C1", "A2", "C2"):
        if kw[c] == 0:
            assert np.all(col[c] == 0), c
    for c in ("E1", "E2"):
        zero_variance = np.array([_scheduled(kw[c], g + burnin) == 0 for g in range(G_ped)])
        assert np.all(col[c][zero_variance[gen]] == 0), c

    u32, u64 = 2.0**-24, 2.0**-53
    gamma2 = 2 * u64 / (1 - 2 * u64)
    for t in ("1", "2"):
        parts = [col[f"{k}{t}"].astype(np.float64) for k in "ACE"]
        total = parts[0] + parts[1] + parts[2]
        # liability is fl64(fl64(a + c) + e) over the float64 draws; the stored components are
        # their float32 roundings, |x - fl32(x)| <= u32 |x| <= u32 |fl32(x)| / (1 - u32). Each side's
        # two float64 additions err by at most gamma2 * S, with S the sum of stored magnitudes, so
        # |liability - total| <= (u32 + gamma2) / (1 - u32) * S + gamma2 * S.
        bound = ((u32 + gamma2) / (1 - u32) + gamma2) * (np.abs(parts[0]) + np.abs(parts[1]) + np.abs(parts[2]))
        assert np.all(np.abs(col[f"liability{t}"] - total) <= bound), t


@pytest.mark.parametrize("mating_model", _MATING_MODELS)
@pytest.mark.parametrize("e_transition", _E_TRANSITIONS)
# Explicit budget, not the conftest profile: run_simulation per example (see above).
# suppress_health_check is independent of the example budget and is preserved.
@settings(deadline=None, max_examples=15, suppress_health_check=[HealthCheck.too_slow])
@given(data=st.data())
def test_run_simulation_structural_integrity(mating_model, e_transition, data):
    """Every recorded pedigree satisfies the structural and ACE contracts in ``_check_recorded_pedigree``.

    Zero assortment, scalar A and C (exact zeros included), and scalar or
    scheduled E. Rejects parent or twin offset errors, households grouped by
    mating or father, per-child C, E schedules read in recorded rather than
    raw coordinates, and noise in a zero-variance component.
    """
    kw = data.draw(simulation_kwargs(mating_model, e_transition))
    ped = run_simulation(**kw)
    event("twins recorded" if (ped["twin"] >= 0).any() else "no twins recorded")
    _check_recorded_pedigree(ped, kw)


def test_run_simulation_twin_links_example():
    """A twin-rich standard run, twins guaranteed present, passes the recorded-pedigree checks.

    Generated examples reach twins only by chance; this pins the twin-link
    assertions to real twins, including in the parentless first generation.
    """
    kw = dict(
        seed=11,
        N=100,
        G_ped=3,
        G_sim=4,
        mating_model="standard",
        mating_lambda=1.0,
        p_mztwin=0.5,
        A1=0.4,
        C1=0.2,
        A2=0.3,
        C2=0.0,
        E1={0: 0.4, 2: 0.0},
        E2=0.5,
        rA=0.5,
        rC=1.0,
        rE=0.0,
    )
    ped = run_simulation(**kw)
    twin_generations = ped.filter(pl.col("twin") >= 0)["generation"].unique().sort().to_list()
    assert twin_generations == [0, 1, 2]
    _check_recorded_pedigree(ped, kw)


@st.composite
def recording_windows(draw, mating_model: str) -> tuple[dict, int, int]:
    """Shared ``run_simulation`` inputs (no G_ped) plus a shorter and a longer G_ped <= G_sim."""
    G_sim = draw(st.integers(min_value=2, max_value=6))
    G_long = draw(st.integers(min_value=2, max_value=G_sim))
    G_short = draw(st.integers(min_value=1, max_value=G_long - 1))

    def e_schedule() -> float | dict[int, float]:
        if draw(st.booleans()):
            return draw(_VARIANCES)
        later = draw(st.dictionaries(st.integers(min_value=1, max_value=G_sim - 1), _VARIANCES, max_size=3))
        return {0: draw(_VARIANCES)} | later

    kw = _draw_shared_kwargs(draw, mating_model) | dict(G_sim=G_sim, E1=e_schedule(), E2=e_schedule())
    return kw, G_short, G_long


def _window_view(ped: pl.DataFrame, first_gen: int, N: int) -> pl.DataFrame:
    """Rows from ``first_gen`` on, with ids, links, and generations rebased to start at 0.

    Links into earlier generations become -1; households are relabelled by first row.
    """
    shift = first_gen * N

    def rebased(c: str) -> pl.Expr:
        return pl.when(pl.col(c) >= shift).then(pl.col(c) - shift).otherwise(-1).cast(pl.Int32).alias(c)

    view = ped.filter(pl.col("generation") >= first_gen).with_columns(
        *(rebased(c) for c in ("id", "mother", "father", "twin")),
        (pl.col("generation") - first_gen).cast(pl.Int32),
    )
    return view.with_columns(
        pl.Series("household_id", _canonical_labels(view["household_id"].to_numpy()), dtype=pl.Int32)
    )


@pytest.mark.parametrize("mating_model", _MATING_MODELS)
# Two run_simulation calls per example; 50 examples measured at 1-3 s per mating model.
@settings(deadline=None, max_examples=50, suppress_health_check=[HealthCheck.too_slow])
@given(data=st.data())
def test_recording_window_only_selects_rows(mating_model, data):
    """Two runs differing only in G_ped agree exactly on the generations both record.

    Same seed and G_sim, zero assortment; compared after rebasing ids,
    generations, and links and relabelling households. Rejects random draws
    that depend on the recording window, E schedules read in recorded
    coordinates, and parent, twin, or household offsets that depend on the
    window; a constant offset error is left to ``_check_recorded_pedigree``.
    """
    kw, G_short, G_long = data.draw(recording_windows(mating_model))
    short = run_simulation(**kw, G_ped=G_short)
    long = run_simulation(**kw, G_ped=G_long)
    assert_frame_equal(
        _window_view(short, 0, kw["N"]),
        _window_view(long, G_long - G_short, kw["N"]),
        check_exact=True,
    )
