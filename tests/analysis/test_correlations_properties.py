"""Property-based tests for :mod:`simace.analysis.stats.correlations`.

These functions take a *pre-extracted* ``pairs`` dict, so the properties hand
them synthetic index arrays directly: no pedigree is simulated, no relationship
extraction runs, and nothing recovers a planted parameter from a sample.  Each
example activates a single relationship type and leaves the rest empty, which
exercises the full result shape without paying for a dozen tetrachoric
optimizations per example.
"""

import numpy as np
import polars as pl
from hypothesis import given
from hypothesis import strategies as st

from simace.analysis.stats.correlations import (
    compute_affected_correlations,
    compute_liability_correlations,
    compute_mate_correlation,
    compute_observed_h2_estimators,
    compute_tetrachoric,
)
from simace.core._numba_utils import _ndtri_approx, _norm_cdf, _tetrachoric_nll
from simace.core.relationships import RELATIONSHIP_TYPES
from tests.conftest import pedigree_frame

_MIN_PAIRS = 10  # the documented gate in correlations.py


def _well_conditioned(values: np.ndarray) -> bool:
    """True when a sample's spread is well above the floating-point rounding floor.

    ``_pearsonr_core`` gates on an *exactly* zero sum of squares and returns
    ``0.0``; ``np.corrcoef`` has no such gate and divides by whatever rounding
    residue its own mean leaves behind, which can be a full-magnitude garbage
    value (``np.corrcoef`` returns ``1.0`` for a repeated constant whose
    computed std is 2.2e-16).  This condition excludes only the region where
    the *oracle* is meaningless; the kernel's behaviour there is asserted
    separately by ``test_pearson_paths_never_return_nan``.
    """
    scale = max(1.0, float(np.max(np.abs(values))))
    return float(values.max() - values.min()) > 1e-8 * scale


def _tetrachoric_nll_at(left: np.ndarray, right: np.ndarray, r: float) -> float:
    """Negative log-likelihood of ``r`` for the 2x2 table formed by two binary arrays."""
    t_a = float(_ndtri_approx(1.0 - left.mean()))
    t_b = float(_ndtri_approx(1.0 - right.mean()))
    return float(
        _tetrachoric_nll(
            r,
            t_a,
            t_b,
            _norm_cdf(t_a),
            _norm_cdf(t_b),
            t_a > 1e-15 and t_b > 1e-15,
            float((left & right).sum()),
            float((left & ~right).sum()),
            float((~left & right).sum()),
            float((~left & ~right).sum()),
        )
    )


def _empty_pairs() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    empty = np.empty(0, dtype=np.int64)
    return dict.fromkeys(RELATIONSHIP_TYPES, (empty, empty))


@st.composite
def _frame_and_pairs(draw, *, min_pairs=0, max_pairs=24):
    """Draw ``(frame, pairs, active_type)`` for the pair-correlation functions.

    The frame carries only what these functions read — two liabilities and two
    affected flags — and the pair indices are drawn independently of it, so
    constant columns, constant pair sides, and repeated indices all occur.
    """
    n = draw(st.integers(min_value=2, max_value=30))
    columns = {}
    for trait in (1, 2):
        columns[f"liability{trait}"] = np.asarray(
            [draw(st.floats(min_value=-4.0, max_value=4.0, allow_nan=False, allow_infinity=False)) for _ in range(n)],
            dtype=np.float64,
        )
        columns[f"affected{trait}"] = np.asarray([draw(st.booleans()) for _ in range(n)], dtype=bool)
    frame = pl.DataFrame(columns)

    n_pairs = draw(st.integers(min_value=min_pairs, max_value=max_pairs))
    row = st.integers(min_value=0, max_value=n - 1)
    idx1 = np.asarray([draw(row) for _ in range(n_pairs)], dtype=np.int64)
    idx2 = np.asarray([draw(row) for _ in range(n_pairs)], dtype=np.int64)

    active = draw(st.sampled_from(RELATIONSHIP_TYPES))
    pairs = _empty_pairs()
    pairs[active] = (idx1, idx2)
    return frame, pairs, active


class TestPairCorrelations:
    """Oracle, symmetry, invariance, and the documented gates."""

    @given(case=_frame_and_pairs(min_pairs=_MIN_PAIRS))
    def test_liability_and_affected_match_numpy_corrcoef(self, case):
        """Both Pearson paths agree with ``np.corrcoef`` on the same index arrays.

        Tolerance is floating point between two summation orders — the numba
        ``_pearsonr_core`` kernel and NumPy's — measured worst 8.88e-16
        absolute over this strategy domain (seed 20260825, 391_483
        comparisons).  Tetrachoric correlation is deliberately excluded: it is
        not a Pearson correlation and has no NumPy oracle.
        """
        frame, pairs, active = case
        liability = compute_liability_correlations(frame, pairs=pairs)
        affected = compute_affected_correlations(frame, pairs=pairs)
        idx1, idx2 = pairs[active]

        for trait in (1, 2):
            for got, values in (
                (liability[f"trait{trait}"][active], frame[f"liability{trait}"].to_numpy()),
                (affected[f"trait{trait}"][active], frame[f"affected{trait}"].to_numpy().astype(np.float64)),
            ):
                left, right = values[idx1], values[idx2]
                if not (_well_conditioned(left) and _well_conditioned(right)):
                    continue
                want = float(np.corrcoef(left, right)[0, 1])
                assert got is not None
                assert abs(got - want) < 1e-12

    @given(case=_frame_and_pairs(min_pairs=_MIN_PAIRS))
    def test_pearson_paths_never_return_nan(self, case):
        """Both Pearson paths return ``None`` or a value in ``[-1, 1]`` — never NaN.

        Covers the near-degenerate region the oracle comparison above cannot
        reach: ``_pearsonr_core`` returns ``0.0`` on an exactly-zero sum of
        squares and ``compute_affected_correlations`` maps NaN to ``None``, so
        no NaN escapes into the stats output either way.
        """
        frame, pairs, active = case
        liability = compute_liability_correlations(frame, pairs=pairs)
        affected = compute_affected_correlations(frame, pairs=pairs)
        for trait in (1, 2):
            for got in (liability[f"trait{trait}"][active], affected[f"trait{trait}"][active]):
                assert got is None or (not np.isnan(got) and -1.0 <= got <= 1.0)

    @given(case=_frame_and_pairs(min_pairs=_MIN_PAIRS))
    def test_pair_swap_symmetry(self, case):
        """Swapping ``(idx1, idx2)`` leaves both Pearson correlations unchanged.

        Relationship pairs are unordered, so the direction the extractor
        happened to emit must not reach the result.  Bit-exact for the Pearson
        paths, which are symmetric expressions.
        """
        frame, pairs, active = case
        idx1, idx2 = pairs[active]
        swapped = {**pairs, active: (idx2, idx1)}

        for fn in (compute_liability_correlations, compute_affected_correlations):
            assert fn(frame, pairs=pairs) == fn(frame, pairs=swapped)

    @given(case=_frame_and_pairs(min_pairs=_MIN_PAIRS))
    def test_tetrachoric_pair_swap_reaches_the_same_likelihood(self, case):
        """Swapping the pair sides finds an equally optimal ``r``, not the same ``r``.

        The tetrachoric likelihood *is* symmetric under transposing the 2x2
        table (measured agreement 2.0e-09 over a 20_001-point ``r`` grid), but
        near ``|r| = 1`` it is flat out to the ``+-0.999`` bracket edge, and
        Brent stops at different points on that plateau — measured worst
        ``|dr| = 1.48e-02``.  Asserting equal ``r`` would be asserting which
        arbitrary point an optimizer lands on.  What is actually invariant, and
        what this asserts, is that both answers attain the same likelihood:
        measured worst 1.24e-13 absolute over this domain (seed 20260825,
        28_757 swapped pairs).  Structure — the pair count and the
        None-vs-value pattern — is exact.
        """
        frame, pairs, active = case
        idx1, idx2 = pairs[active]
        forward = compute_tetrachoric(frame, pairs=pairs)
        reverse = compute_tetrachoric(frame, pairs={**pairs, active: (idx2, idx1)})

        for trait in (1, 2):
            values = frame[f"affected{trait}"].to_numpy().astype(bool)
            got, swapped_got = forward[f"trait{trait}"][active], reverse[f"trait{trait}"][active]
            assert got["n_pairs"] == swapped_got["n_pairs"]
            assert (got["r"] is None) == (swapped_got["r"] is None)
            if got["r"] is None:
                continue
            left, right = values[idx1], values[idx2]
            assert (
                abs(_tetrachoric_nll_at(left, right, got["r"]) - _tetrachoric_nll_at(left, right, swapped_got["r"]))
                < 1e-10
            )

    @given(case=_frame_and_pairs(min_pairs=_MIN_PAIRS), data=st.data())
    def test_row_permutation_invariance(self, case, data):
        """Permuting frame rows and remapping the indices changes nothing.

        These are row-position indices, not ids, so a permutation is a pure
        relabelling.  The results are bit-identical, not merely close.
        """
        frame, pairs, active = case
        n = len(frame)
        permutation = np.asarray(data.draw(st.permutations(range(n))), dtype=np.int64)
        inverse = np.empty(n, dtype=np.int64)
        inverse[permutation] = np.arange(n, dtype=np.int64)

        permuted_frame = frame[permutation]
        idx1, idx2 = pairs[active]
        permuted_pairs = {**pairs, active: (inverse[idx1], inverse[idx2])}

        for fn in (compute_liability_correlations, compute_affected_correlations, compute_tetrachoric):
            assert fn(frame, pairs=pairs) == fn(permuted_frame, pairs=permuted_pairs)

    @given(case=_frame_and_pairs(max_pairs=_MIN_PAIRS - 1))
    def test_below_ten_pairs_returns_none(self, case):
        """Fewer than ten pairs yields ``None`` rather than a noisy estimate."""
        frame, pairs, active = case
        liability = compute_liability_correlations(frame, pairs=pairs)
        affected = compute_affected_correlations(frame, pairs=pairs)
        tetrachoric = compute_tetrachoric(frame, pairs=pairs)

        for trait in (1, 2):
            key = f"trait{trait}"
            for ptype in RELATIONSHIP_TYPES:
                assert liability[key][ptype] is None
                assert affected[key][ptype] is None
                assert tetrachoric[key][ptype]["r"] is None
                assert tetrachoric[key][ptype]["se"] is None
            assert tetrachoric[key][active]["n_pairs"] == len(pairs[active][0])

    @given(case=_frame_and_pairs(min_pairs=_MIN_PAIRS))
    def test_phi_is_bounded_and_gated_on_constant_sides(self, case):
        """Phi lies in ``[-1, 1]``, and is ``None`` exactly when a side is constant."""
        frame, pairs, active = case
        result = compute_affected_correlations(frame, pairs=pairs)
        idx1, idx2 = pairs[active]

        for trait in (1, 2):
            values = frame[f"affected{trait}"].to_numpy().astype(np.float64)
            got = result[f"trait{trait}"][active]
            constant = np.std(values[idx1]) < 1e-10 or np.std(values[idx2]) < 1e-10
            if constant:
                assert got is None
            else:
                assert got is not None
                assert -1.0 <= got <= 1.0


_ESTIMATOR_INPUT = st.one_of(st.none(), st.floats(min_value=-1.0, max_value=1.0, allow_nan=False))


class TestObservedH2Estimators:
    """Pure arithmetic and None-propagation in the five closed-form estimators."""

    @given(
        mz=_ESTIMATOR_INPUT,
        fs=_ESTIMATOR_INPUT,
        mhs=_ESTIMATOR_INPUT,
        phs=_ESTIMATOR_INPUT,
        cousins=_ESTIMATOR_INPUT,
        slope=_ESTIMATOR_INPUT,
    )
    def test_closed_forms_and_none_propagation(self, mz, fs, mhs, phs, cousins, slope):
        """``falconer/sibs/hs/cousins/po`` are exact, and None propagates per rule.

        Assertions recompute the identical expression rather than an
        algebraically equivalent one, so the comparison is bit-exact: ``hs``
        returns ``0.6000000000000001`` for an exact-``0.6`` input, which any
        rearranged form would miss.
        """
        correlations = {"MZ": mz, "FS": fs, "MHS": mhs, "PHS": phs, "1C": cousins}
        result = compute_observed_h2_estimators(
            {"trait1": correlations, "trait2": correlations},
            {"trait1": {"slope": slope}, "trait2": {"slope": slope}},
        )

        hs_values = [float(v) for v in (mhs, phs) if v is not None]
        for trait in (1, 2):
            got = result[f"trait{trait}"]
            assert got["falconer"] == (None if mz is None or fs is None else 2.0 * (float(mz) - float(fs)))
            assert got["sibs"] == (None if fs is None else 2.0 * float(fs))
            assert got["cousins"] == (None if cousins is None else 8.0 * float(cousins))
            assert got["po"] == (None if slope is None else float(slope))
            assert got["hs"] == (None if not hs_values else 4.0 * (sum(hs_values) / len(hs_values)))

    @given(missing=st.sampled_from(["MZ", "FS", "1C"]), value=st.floats(min_value=-1.0, max_value=1.0))
    def test_any_missing_input_nulls_its_estimator(self, missing, value):
        """Every estimator except ``hs`` is None as soon as one input is missing."""
        correlations = dict.fromkeys(("MZ", "FS", "MHS", "PHS", "1C"), value)
        correlations[missing] = None
        result = compute_observed_h2_estimators({"trait1": correlations}, {"trait1": {"slope": value}})["trait1"]

        affected_by = {"MZ": ["falconer"], "FS": ["falconer", "sibs"], "1C": ["cousins"]}
        for name in affected_by[missing]:
            assert result[name] is None
        assert result["hs"] is not None  # both half-sib inputs still present


class TestMateCorrelation:
    """Distinct-mating counting and the directed mating-key encoding."""

    @given(pedigree=pedigree_frame(liabilities=True), data=st.data())
    def test_n_pairs_counts_distinct_valid_matings(self, pedigree, data):
        """``n_pairs`` is the number of distinct ``(mother, father)`` tuples present.

        Counts of zero and one are included: the below-two branch returns a NaN
        matrix because a correlation needs two points, but it used to hard-code
        ``n_pairs=0``, mis-reporting a single valid mating as none.

        The ``mother * base + father`` key here is *directed* — mother and
        father are distinct roles — and is not the canonical unordered
        relationship-pair encoding of CLAUDE.md gotcha #8.
        """
        # Drop a random subset so some parent references dangle, exercising the
        # ``ped.contains`` filter the way ascertainment output does.
        keep = np.asarray(
            [data.draw(st.booleans()) for _ in range(len(pedigree))],
            dtype=bool,
        )
        if not keep.any():
            keep[0] = True
        frame = pedigree.filter(pl.Series(keep))

        present = set(frame["id"].to_list())
        expected = {
            (mother, father)
            for mother, father in zip(frame["mother"].to_list(), frame["father"].to_list(), strict=True)
            if mother != -1 and father != -1 and mother in present and father in present
        }

        result = compute_mate_correlation(frame)
        assert result["n_pairs"] == len(expected)
        if len(expected) < 2:
            assert all(np.isnan(value) for row in result["matrix"] for value in row)

    @given(pedigree=pedigree_frame(liabilities=True))
    def test_mating_key_round_trips(self, pedigree):
        """``mother * base + father`` decodes back to the original pair."""
        mothers = pedigree["mother"].to_numpy().astype(np.int64)
        fathers = pedigree["father"].to_numpy().astype(np.int64)
        child = (mothers != -1) & (fathers != -1)
        if not child.any():
            return
        m_child, f_child = mothers[child], fathers[child]

        base = np.int64(max(int(m_child.max()), int(f_child.max())) + 1)
        keys = m_child * base + f_child
        assert np.array_equal(keys // base, m_child)
        assert np.array_equal(keys % base, f_child)
        assert len(np.unique(keys)) == len({(m, f) for m, f in zip(m_child.tolist(), f_child.tolist(), strict=True)})
