"""Property-based tests for :mod:`simace.simulation.assortment` and its oracle.

``AssortmentPlan`` had no direct test anywhere before this file.  The two
modules under test carry the same formulas written twice —
``_cross_am_matrix`` / ``rho_w`` in ``assortment.py`` and the both-trait branch
of ``expected_mate_corr_matrix`` in ``mate_correlation.py``, whose docstring
says "Change one, change the other" with nothing enforcing it.  The first
property is that enforcement.

Every assertion is exact or floating-point tight; no tolerance here absorbs
sampling uncertainty.
"""

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from simace.simulation.assortment import AssortmentPlan, _cross_am_matrix, _parent_ce_gen
from simace.simulation.mate_correlation import expected_mate_corr_matrix

_CORRELATIONS = st.sampled_from([-1.0, -0.7, -0.3, 0.0, 0.3, 0.7, 1.0])
_ASSORT_VALUES = [-1.0, -0.99, -0.6, -0.2, 0.0, 0.2, 0.6, 0.99, 1.0]


def _variance(draw, *, budget: float) -> float:
    return draw(st.floats(min_value=0.0, max_value=budget, allow_nan=False, allow_infinity=False))


@st.composite
def _assort_schedule(draw, g_sim: int, *, nonzero: bool):
    """Draw a scalar or per-iteration dict of mate correlations."""
    values = [v for v in _ASSORT_VALUES if v != 0.0] if nonzero else _ASSORT_VALUES
    if draw(st.booleans()):
        return draw(st.sampled_from(values))
    return {i: draw(st.sampled_from(values)) for i in range(g_sim)}


@st.composite
def _assortment_inputs(draw, *, constant_ce=False, both_trait=False, user_matrix=False):
    """Draw a complete ``AssortmentPlan.build`` argument bundle.

    Variance components follow the model's own budget (``A + C + E == 1`` per
    trait per generation), and the correlations include exactly ``+-1`` so that
    ``|rho_w| >= 1`` — the guarded corner — is genuinely reachable rather than
    vanishingly improbable.
    """
    g_sim = draw(st.integers(min_value=1, max_value=4))
    a1 = _variance(draw, budget=1.0)
    a2 = _variance(draw, budget=1.0)

    n_ce = 1 if constant_ce else g_sim
    c1 = [_variance(draw, budget=1.0 - a1) for _ in range(n_ce)]
    c2 = [_variance(draw, budget=1.0 - a2) for _ in range(n_ce)]
    e1 = [1.0 - a1 - c for c in c1]
    e2 = [1.0 - a2 - c for c in c2]
    if constant_ce:
        c1, c2, e1, e2 = (values * g_sim for values in (c1, c2, e1, e2))

    matrix = None
    if user_matrix:
        entries = [draw(st.sampled_from(_ASSORT_VALUES)) for _ in range(4)]
        matrix = np.array(entries, dtype=np.float64).reshape(2, 2)

    return dict(
        assort1=draw(_assort_schedule(g_sim, nonzero=both_trait)),
        assort2=draw(_assort_schedule(g_sim, nonzero=both_trait)),
        R_mf_user=matrix,
        rA=draw(_CORRELATIONS),
        rC=draw(_CORRELATIONS),
        rE=draw(_CORRELATIONS),
        A1=a1,
        A2=a2,
        C1_per_gen=c1,
        C2_per_gen=c2,
        E1_per_gen=e1,
        E2_per_gen=e2,
        G_sim=g_sim,
    )


def _rho_w_schedule(kwargs: dict) -> list[float]:
    """Independently recompute ``rho_w`` per C/E generation from the ACE inputs."""
    return [
        kwargs["rA"] * np.sqrt(kwargs["A1"] * kwargs["A2"])
        + kwargs["rC"] * np.sqrt(kwargs["C1_per_gen"][g] * kwargs["C2_per_gen"][g])
        + kwargs["rE"] * np.sqrt(kwargs["E1_per_gen"][g] * kwargs["E2_per_gen"][g])
        for g in range(kwargs["G_sim"])
    ]


def _per_iteration(value, g_sim: int) -> list[float]:
    """Expand a scalar or fully-specified dict schedule to one value per iteration."""
    return [float(value)] * g_sim if isinstance(value, (int, float)) else [float(value[i]) for i in range(g_sim)]


class TestPlanMatchesOracle:
    """The duplicated formulas in the two modules must not drift apart."""

    @given(kwargs=_assortment_inputs(constant_ce=True, both_trait=True))
    def test_plan_matches_expected_mate_corr_matrix(self, kwargs):
        """``for_generation(i)[3]`` equals the analytic oracle under constant C/E.

        Constant C/E removes the schedule from the comparison, isolating the
        shared ``rho_w`` and cross-AM expressions.  Both modules build the same
        product in the same association order, so the agreement is exact.
        """
        try:
            plan = AssortmentPlan.build(**kwargs)
        except ValueError:
            return  # guarded corner; covered exhaustively by TestRuntimeScheduledGuards
        a1_schedule = _per_iteration(kwargs["assort1"], kwargs["G_sim"])
        a2_schedule = _per_iteration(kwargs["assort2"], kwargs["G_sim"])

        for i in range(kwargs["G_sim"]):
            _, _, _, r_mf = plan.for_generation(i)
            want = expected_mate_corr_matrix(
                a1_schedule[i],
                a2_schedule[i],
                kwargs["rA"],
                kwargs["rC"],
                kwargs["A1"],
                kwargs["C1_per_gen"][0],
                kwargs["A2"],
                kwargs["C2_per_gen"][0],
                None,
                kwargs["rE"],
                kwargs["E1_per_gen"][0],
                kwargs["E2_per_gen"][0],
            )
            assert np.array_equal(r_mf, want), f"iteration {i}: plan and oracle disagree"


class TestGenerationSchedule:
    """``rho_w`` is indexed by the parental generation, not the offspring's."""

    @given(kwargs=_assortment_inputs())
    def test_rho_w_uses_the_parental_generation(self, kwargs):
        """``for_generation(i)[2] == rho_w_per_ce[max(0, i - 1)]``.

        This is the mechanism behind CLAUDE.md gotcha #5: under
        generation-dependent C/E the schedule is shifted by one, so an oracle
        that reads ``rho_w_per_ce[i]`` diverges from the simulator for
        ``i >= 2``.
        """
        try:
            plan = AssortmentPlan.build(**kwargs)
        except ValueError:
            return
        expected = _rho_w_schedule(kwargs)
        for i in range(kwargs["G_sim"]):
            assert plan.for_generation(i)[2] == pytest.approx(expected[max(0, i - 1)], abs=0.0, rel=0.0)

    @given(g_sim=st.integers(min_value=1, max_value=8))
    def test_parent_schedule_never_reads_the_final_offspring_generation(self, g_sim):
        """The mated populations span C/E generations ``0 .. G_sim - 2`` (and 0)."""
        used = {_parent_ce_gen(i) for i in range(g_sim)}
        assert used == set(range(max(1, g_sim - 1)))
        assert _parent_ce_gen(0) == 0


class TestCrossAmMatrix:
    """Shape invariants of the auto-computed cross-AM matrix."""

    @given(
        assort1=st.sampled_from(_ASSORT_VALUES),
        assort2=st.sampled_from(_ASSORT_VALUES),
        rho_w=st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    )
    def test_symmetric_and_bounded(self, assort1, assort2, rho_w):
        """``R_mf`` is symmetric with ``|c| <= sqrt(|a1 * a2|)`` for ``|rho_w| <= 1``."""
        r_mf = _cross_am_matrix(assort1, assort2, rho_w)
        assert np.array_equal(r_mf, r_mf.T)
        assert r_mf[0, 0] == assort1
        assert r_mf[1, 1] == assort2
        assert abs(r_mf[0, 1]) <= np.sqrt(abs(assort1 * assort2)) + 1e-15


class TestRuntimeScheduledGuards:
    """``build()`` raises exactly on the schedule the simulator actually runs."""

    @given(kwargs=st.one_of(_assortment_inputs(), _assortment_inputs(user_matrix=True)))
    def test_guards_are_exact_in_both_directions(self, kwargs):
        """Raises iff some *mating iteration* violates the rho_w or PSD guard.

        The oracle reconstructs each ``Sigma_4`` from the runtime pairing —
        iteration ``i``'s assortment against ``rho_w_per_ce[max(0, i - 1)]``,
        the contract ``for_generation`` documents — so a false accept and a
        false reject are both caught.  Before the schedule fix, ``build()``
        paired iteration ``g`` with ``rho_w_per_ce[g]`` and also validated the
        final offspring C/E generation, which is never mated.
        """
        g_sim = kwargs["G_sim"]
        rho_w = _rho_w_schedule(kwargs)
        a1_schedule = _per_iteration(kwargs["assort1"], g_sim)
        a2_schedule = _per_iteration(kwargs["assort2"], g_sim)
        user_matrix = kwargs["R_mf_user"]

        should_raise = False
        for i in range(g_sim):
            rw = rho_w[max(0, i - 1)]
            a1_i, a2_i = a1_schedule[i], a2_schedule[i]
            both_trait = a1_i != 0 and a2_i != 0
            if both_trait and abs(rw) >= 1.0 - 1e-10:
                should_raise = True
                break
            if user_matrix is None and not both_trait:
                continue
            r_mf = user_matrix if user_matrix is not None else _cross_am_matrix(a1_i, a2_i, rw)
            r_ff = np.array([[1.0, rw], [rw, 1.0]])
            sigma_4 = np.block([[r_ff, r_mf.T], [r_mf, r_ff]])
            if np.linalg.eigvalsh(sigma_4)[0] < -1e-8:
                should_raise = True
                break

        if should_raise:
            with pytest.raises(ValueError, match=r"requires \|rho_w\| < 1|is not PSD"):
                AssortmentPlan.build(**kwargs)
        else:
            plan = AssortmentPlan.build(**kwargs)
            for i in range(g_sim):
                assert plan.for_generation(i)[2] == rho_w[max(0, i - 1)]


class TestExpectedMateCorrMatrix:
    """Metamorphic and pass-through properties of the analytic oracle."""

    @given(
        assort1=st.sampled_from(_ASSORT_VALUES),
        assort2=st.sampled_from(_ASSORT_VALUES),
        rA=_CORRELATIONS,
        rC=_CORRELATIONS,
        rE=_CORRELATIONS,
        A1=st.floats(min_value=0.0, max_value=1.0),
        A2=st.floats(min_value=0.0, max_value=1.0),
        C1=st.floats(min_value=0.0, max_value=1.0),
        C2=st.floats(min_value=0.0, max_value=1.0),
        E1=st.floats(min_value=0.0, max_value=1.0),
        E2=st.floats(min_value=0.0, max_value=1.0),
    )
    def test_trait_relabelling_reverses_both_axes(self, assort1, assort2, rA, rC, rE, A1, A2, C1, C2, E1, E2):
        """Swapping trait 1 and trait 2 everywhere swaps both matrix axes.

        Not merely a transpose — these matrices are already symmetric — so this
        pins the two *asymmetric* single-trait branches against each other:
        the trait-1 branch relabelled must reproduce the trait-2 branch.
        """
        original = expected_mate_corr_matrix(assort1, assort2, rA, rC, A1, C1, A2, C2, None, rE, E1, E2)
        swapped = expected_mate_corr_matrix(assort2, assort1, rA, rC, A2, C2, A1, C1, None, rE, E2, E1)
        assert np.array_equal(swapped, original[::-1, ::-1])

    @given(
        entries=st.lists(st.floats(min_value=-1.0, max_value=1.0), min_size=4, max_size=4),
        kwargs=_assortment_inputs(),
    )
    def test_user_matrix_is_returned_verbatim_and_suppresses_auto_computation(self, entries, kwargs):
        """An explicit ``assort_matrix`` wins in both the oracle and the plan."""
        matrix = np.array(entries, dtype=np.float64).reshape(2, 2)
        assert np.array_equal(
            expected_mate_corr_matrix(0.5, 0.5, 0.3, 0.3, 0.5, 0.2, 0.5, 0.2, matrix),
            matrix,
        )

        kwargs = {**kwargs, "R_mf_user": matrix}
        try:
            plan = AssortmentPlan.build(**kwargs)
        except ValueError:
            return
        for i in range(kwargs["G_sim"]):
            assert np.array_equal(plan.for_generation(i)[3], matrix)
