"""Property-based tests for :mod:`simace.ascertainment.runner`.

Structural invariants of the dropout → case-weighted draw → ancestor-closure
pipeline.  Every assertion here is exact: sizes, set membership, row order, and
RNG-state identity.  Nothing depends on a statistical tolerance, and no
planted parameter is recovered.
"""

from typing import NamedTuple

import numpy as np
import polars as pl
import polars.testing
import pytest
from hypothesis import event, given
from hypothesis import strategies as st

from simace.ascertainment.runner import _apply_dropout, _sample_trait_ids, run_ascertainment
from simace.core.schema import PEDIGREE
from simace.core.trait_schema import CENSORED_TRAIT
from tests.conftest import pedigree_frame, relabel_ids, schema_pad
from tests.downstream_strategies import ascertainment_inputs


def _draw_affected(draw, n: int) -> np.ndarray:
    """Draw an affected-status column, reaching all-case and all-control pools often.

    Independent per-row booleans make a degenerate pool vanishingly rare —
    measured 1 all-case pool in 800 examples — and the degenerate pool is
    exactly where the ``ratio == 0`` branch ordering matters.  Drawing the
    pattern first puts each degenerate case at roughly one example in three.
    """
    pattern = draw(st.sampled_from(["all_control", "all_case", "mixed"]))
    if pattern == "all_control":
        return np.zeros(n, dtype=bool)
    if pattern == "all_case":
        return np.ones(n, dtype=bool)
    return np.asarray([draw(st.booleans()) for _ in range(n)], dtype=bool)


@st.composite
def _pedigree_and_trait(draw, *, twins=True, relabel=False):
    """Draw a valid pedigree plus the outcomes-only censored trait frame over it.

    The trait covers the last ``G_pheno`` generations, matching the pipeline:
    phenotyping runs on the youngest generations only.  Only ``id`` and the
    ``affected*`` flags carry information; the remaining ``CENSORED_TRAIT``
    columns are padded, with ``age_censored = ~affected`` so the frame respects
    the ``affected = NOT (age_censored OR death_censored)`` identity
    (CLAUDE.md gotcha #6) rather than merely satisfying the dtype contract.
    """
    pedigree = draw(pedigree_frame(twins=twins))
    if relabel:
        pedigree = relabel_ids(pedigree, draw(st.data()))

    generations = sorted(set(pedigree["generation"].to_list()))
    g_pheno = draw(st.integers(min_value=1, max_value=len(generations)))
    phenotyped = pedigree.filter(pl.col("generation").is_in(generations[-g_pheno:]))

    n = len(phenotyped)
    affected1 = _draw_affected(draw, n)
    affected2 = _draw_affected(draw, n)
    trait = pl.DataFrame(
        {
            "id": phenotyped["id"].to_numpy(),
            "affected1": affected1,
            "age_censored1": ~affected1,
            "affected2": affected2,
            "age_censored2": ~affected2,
        }
    )
    return pedigree, schema_pad(trait, CENSORED_TRAIT)


class _Case(NamedTuple):
    """One complete argument set for ``run_ascertainment``, and the calls that use it."""

    pedigree: pl.DataFrame
    trait: pl.DataFrame
    dropout_rate: float
    ratio: float
    n_sample: int
    seed: int

    def run(self, *, dropout_rate: float | None = None) -> tuple[pl.DataFrame, pl.DataFrame]:
        """Run full ascertainment on this case, optionally overriding the drawn rate."""
        return run_ascertainment(
            self.pedigree,
            self.trait,
            dropout_rate=self.dropout_rate if dropout_rate is None else dropout_rate,
            case_ascertainment_ratio=self.ratio,
            N_sample=self.n_sample,
            seed=self.seed,
        )

    def sample_ids(self, *, ratio: float | None = None) -> tuple[np.ndarray, str]:
        """Draw this case's trait id set, optionally overriding the drawn ratio."""
        return _sample_trait_ids(
            self.trait,
            case_ascertainment_ratio=self.ratio if ratio is None else ratio,
            N_sample=self.n_sample,
            rng=np.random.default_rng(self.seed),
        )


@st.composite
def _ascertainment_case(draw, *, relabel=False) -> _Case:
    """Draw a complete ``run_ascertainment`` argument set.

    ``dropout_rate`` is drawn as an exact drop *count* in ``[0, n-1]`` and
    converted back to a rate, so the public-API cases always stay inside the
    branch where ``0 <= round(n * rate) < n``.  The ``n_drop >= n`` boundary,
    which raises, is covered directly against ``_apply_dropout``.
    """
    pedigree, trait = draw(_pedigree_and_trait(relabel=relabel))
    n_drop = draw(st.integers(min_value=0, max_value=len(pedigree) - 1))
    return _Case(
        pedigree=pedigree,
        trait=trait,
        dropout_rate=n_drop / len(pedigree),
        ratio=draw(st.sampled_from([0.0, 0.25, 1.0, 4.0])),
        n_sample=draw(st.integers(min_value=-2, max_value=len(trait) + 2)),
        seed=draw(st.integers(min_value=0, max_value=2**31 - 1)),
    )


def _refuses(trait: pl.DataFrame, *, ratio: float, n_sample: int) -> bool:
    """True when ``_sample_trait_ids`` refuses this pool outright.

    A zero case weight over an all-case pool leaves no eligible individual, so
    the draw raises rather than returning an empty cohort — the same stance
    ``_apply_dropout`` takes on a rate that would remove everyone.  Only the
    weighted-draw branch can refuse; the ``N_sample`` pass-through paths are
    ratio-independent and return the whole pool.
    """
    if not (0 < n_sample < len(trait)):
        return False
    return ratio == 0 and bool(trait["affected1"].to_numpy().all())


def _run_or_none(case: "_Case", **kwargs) -> tuple[pl.DataFrame, pl.DataFrame] | None:
    """Run the case, returning ``None`` when the zero-case-weight refusal fires.

    ``run`` applies dropout first, so whether the surviving pool is all-case is
    not a function of ``case.trait`` alone — catching the refusal is exact where
    predicting it would not be.  Anything else propagates.  The refusal itself
    is asserted directly in :class:`TestSampleTraitIds`; the invariants here
    only bind on a cohort that exists.
    """
    try:
        return case.run(**kwargs)
    except ValueError as exc:
        if "would select nobody" not in str(exc):
            raise
        return None


def _row_positions(haystack: np.ndarray, needles: np.ndarray) -> np.ndarray:
    """Row positions of ``needles`` within ``haystack``; both must be id arrays."""
    order = np.argsort(haystack)
    return order[np.searchsorted(haystack[order], needles)]


class TestRunAscertainment:
    """Structural invariants of the public ``run_ascertainment`` API."""

    @given(case=_ascertainment_case())
    def test_parent_pointers_intact_without_dropout(self, case):
        """At ``dropout_rate=0`` the closure follows every parent, so none is severed.

        The docstring at ``runner.py`` claims parent links are safe by
        construction *only* here; at rate > 0 an ancestor can be removed before
        the closure is built and severing is legitimate, so the negation is not
        asserted there.
        """
        result = _run_or_none(case, dropout_rate=0.0)
        if result is None:
            return
        ped_out, _ = result
        if len(ped_out) == 0:
            return
        original = dict(
            zip(
                case.pedigree["id"].to_list(),
                zip(case.pedigree["mother"].to_list(), case.pedigree["father"].to_list(), strict=True),
                strict=True,
            )
        )
        for row_id, mother, father in zip(
            ped_out["id"].to_list(), ped_out["mother"].to_list(), ped_out["father"].to_list(), strict=True
        ):
            assert (mother, father) == original[row_id]


class TestApplyDropout:
    """``_apply_dropout`` count, ordering, identity, and raising boundaries."""

    @given(pedigree=pedigree_frame(), seed=st.integers(min_value=0, max_value=2**31 - 1))
    def test_zero_rate_is_identity_and_leaves_rng_untouched(self, pedigree, seed):
        """Rate 0 returns the input and never draws — downstream seeds stay stable."""
        rng = np.random.default_rng(seed)
        state_before = rng.bit_generator.state
        result = _apply_dropout(pedigree, 0.0, rng)
        assert result.equals(pedigree)
        assert rng.bit_generator.state == state_before

    @given(data=st.data(), pedigree=pedigree_frame(), seed=st.integers(min_value=0, max_value=2**31 - 1))
    def test_drop_count_and_order(self, data, pedigree, seed):
        """``n - round(n * rate)`` rows survive, as an order-preserving subsequence."""
        n = len(pedigree)
        n_drop = data.draw(st.integers(min_value=0, max_value=n - 1))
        result = _apply_dropout(pedigree, n_drop / n, np.random.default_rng(seed))
        assert len(result) == n - n_drop

        kept = result["id"].to_numpy()
        positions = _row_positions(pedigree["id"].to_numpy(), kept)
        assert np.all(np.diff(positions) > 0), "kept rows are not an order-preserving subsequence"

    @given(pedigree=pedigree_frame(), seed=st.integers(min_value=0, max_value=2**31 - 1))
    def test_full_dropout_raises(self, pedigree, seed):
        """``n_drop >= n`` raises rather than returning an empty pedigree."""
        with pytest.raises(ValueError, match="would remove all"):
            _apply_dropout(pedigree, 1.0, np.random.default_rng(seed))


class TestSampleTraitIds:
    """Bounds and branch selection in the case-weighted draw."""

    @given(case=_ascertainment_case(relabel=True))
    def test_sampled_ids_are_a_bounded_ordered_subset(self, case):
        """Sampled ids are unique, drawn from the pool, and keep pool row order."""
        pool_ids = case.trait["id"].to_numpy()
        if _refuses(case.trait, ratio=case.ratio, n_sample=case.n_sample):
            with pytest.raises(ValueError, match="would select nobody"):
                case.sample_ids()
            return
        sampled, _ = case.sample_ids()
        assert set(sampled.tolist()) <= set(pool_ids.tolist())
        assert len(set(sampled.tolist())) == len(sampled)
        if len(sampled):
            positions = _row_positions(pool_ids, sampled)
            assert np.all(np.diff(positions) > 0)

    @given(case=_ascertainment_case())
    def test_sample_size_follows_the_documented_branches(self, case):
        """Size is 0 / whole pool / ``N_sample`` / controls-clamped, per branch — or a refusal."""
        n_pool = len(case.trait)
        is_case = case.trait["affected1"].to_numpy()
        n_controls = int((~is_case).sum())
        if _refuses(case.trait, ratio=case.ratio, n_sample=case.n_sample):
            with pytest.raises(ValueError, match="would select nobody"):
                case.sample_ids()
            return
        sampled, _ = case.sample_ids()
        if n_pool == 0:
            assert len(sampled) == 0
        elif case.n_sample <= 0 or case.n_sample >= n_pool:
            assert np.array_equal(sampled, case.trait["id"].to_numpy())
        elif case.ratio == 0:
            assert len(sampled) == min(case.n_sample, n_controls)
        else:
            assert len(sampled) == case.n_sample

    @given(case=_ascertainment_case())
    def test_zero_ratio_draws_only_controls(self, case):
        """A zero case weight selects no cases whenever a weighted draw occurs.

        The precondition is exactly ``0 < N_sample < n_pool`` — the branch where
        a draw happens at all.  An all-case pool has no eligible individual at
        all and is refused rather than silently emptied; the ``N_sample``
        pass-through path stays ratio-independent and is excluded here.
        """
        n_pool = len(case.trait)
        if not (0 < case.n_sample < n_pool):
            return
        is_case = case.trait["affected1"].to_numpy()
        n_controls = int((~is_case).sum())
        if n_controls == 0:
            with pytest.raises(ValueError, match="would select nobody"):
                case.sample_ids(ratio=0.0)
            return
        sampled, _ = case.sample_ids(ratio=0.0)
        selected = case.trait.filter(pl.Series(np.isin(case.trait["id"].to_numpy(), sampled)))
        assert not selected["affected1"].any()
        assert len(sampled) == min(case.n_sample, n_controls)


_LINKS = ("mother", "father", "twin")


def _assert_same(got: pl.DataFrame, want: pl.DataFrame) -> None:
    polars.testing.assert_frame_equal(got, want, check_row_order=True, check_dtypes=True, check_exact=True)


def _available_ancestry(available: pl.DataFrame, seeds) -> set[int]:
    """Seeds plus every ancestor reached through parents present in ``available``.

    A plain graph walk over a dict, independent of ``filter_pedigree_to_observed``.
    """
    parents = dict(
        zip(
            available["id"].to_list(),
            zip(available["mother"].to_list(), available["father"].to_list(), strict=True),
            strict=True,
        )
    )
    reached: set[int] = set()
    stack = [int(i) for i in seeds]
    while stack:
        person = stack.pop()
        if person in reached:
            continue
        reached.add(person)
        stack.extend(p for p in parents[person] if p in parents)
    return reached


def _relabel(frame: pl.DataFrame, mapping: dict[int, int], columns) -> pl.DataFrame:
    """Rewrite id-valued ``columns`` through ``mapping``, keeping -1 and each column's dtype."""
    return frame.with_columns(
        pl.Series(col, [-1 if v < 0 else mapping[v] for v in frame[col].to_list()], dtype=frame.schema[col])
        for col in columns
    )


@pytest.mark.parametrize("zero_weight", [False, True], ids=["positive-weight", "zero-weight"])
class TestPreservation:
    """Selection copies input rows; only links to people outside the output change, and only to -1.

    Inputs come from ``ascertainment_inputs``: gapped ids, twins, informative
    trait columns with null raw onsets, and both zero and positive case weights.
    Zero-weight cases guarantee a control survives positive dropout.
    """

    @given(data=st.data())
    def test_sample_rows_are_the_input_rows(self, zero_weight, data):
        """Each sample row equals its input trait row, in input order, with dtypes and nulls intact.

        Rejects outcome recomputation or overwriting during selection, null
        onsets filled with a value, and column or row misalignment.
        """
        inp = data.draw(ascertainment_inputs(zero_weight=zero_weight))
        _, trait_out = inp.run()
        if zero_weight:
            assert inp.kwargs["dropout_rate"] > 0
            assert not trait_out.is_empty()
            assert not trait_out["affected1"].any()
        event(f"sample rows: {'none' if trait_out.is_empty() else 'some'}")
        event(f"null raw onset in sample: {trait_out['t1'].null_count() + trait_out['t2'].null_count() > 0}")
        _assert_same(trait_out, inp.trait.filter(pl.col("id").is_in(trait_out["id"].implode())))

    @given(data=st.data())
    def test_pedigree_rows_change_only_by_severing_unavailable_links(self, zero_weight, data):
        """Each output pedigree row equals its input row, except links to people outside the output, which are -1.

        Rejects modifying a retained parent or twin id, severing a link whose
        referent is present, and pointing a severed link anywhere but -1.
        """
        inp = data.draw(ascertainment_inputs(zero_weight=zero_weight))
        ped_out, _ = inp.run()
        kept = ped_out["id"].implode()
        original = inp.pedigree.filter(pl.col("id").is_in(kept))
        expected = original.with_columns(
            pl.when(pl.col(col).is_in(kept)).then(pl.col(col)).otherwise(-1).cast(original.schema[col]).alias(col)
            for col in _LINKS
        )
        event(f"severed links: {not original.select(_LINKS).equals(expected.select(_LINKS))}")
        _assert_same(ped_out, expected)

    @given(data=st.data())
    def test_pedigree_is_the_sample_ancestry_among_dropout_survivors(self, zero_weight, data):
        """The output pedigree is the sample plus its ancestors reachable through surviving people only.

        The survivors are recomputed from the seed with ``_apply_dropout``;
        the expected closure is an independent walk over their parent links.
        Rejects reconnecting ancestry across a dropped person and keeping a
        dropped person.
        """
        inp = data.draw(ascertainment_inputs(zero_weight=zero_weight))
        ped_out, trait_out = inp.run()
        survivors = _apply_dropout(inp.pedigree, inp.kwargs["dropout_rate"], np.random.default_rng(inp.kwargs["seed"]))
        expected = _available_ancestry(survivors, trait_out["id"].to_list())
        through_dropped = _available_ancestry(inp.pedigree, trait_out["id"].to_list()) & set(survivors["id"].to_list())
        event(f"dropout cut a path to a surviving ancestor: {through_dropped != expected}")
        assert set(ped_out["id"].to_list()) == expected

    @given(data=st.data())
    def test_selection_commutes_with_an_id_bijection(self, zero_weight, data):
        """Relabelling every id through a gapped order-preserving bijection relabels the outputs the same way.

        Row order and seed are unchanged, so the row-position draws match.
        Rejects using ids as row positions or letting id values steer the draw
        or the closure.
        """
        inp = data.draw(ascertainment_inputs(zero_weight=zero_weight))
        relabelled = relabel_ids(inp.pedigree, data)
        mapping = dict(zip(inp.pedigree["id"].to_list(), relabelled["id"].to_list(), strict=True))
        ped_out, trait_out = inp.run()
        ped_mapped, trait_mapped = inp._replace(pedigree=relabelled, trait=_relabel(inp.trait, mapping, ["id"])).run()
        _assert_same(ped_mapped, _relabel(ped_out, mapping, ["id", *_LINKS]))
        _assert_same(trait_mapped, _relabel(trait_out, mapping, ["id"]))


def _interrupted_pedigree() -> pl.DataFrame:
    """Three generations with gapped ids: ``10 x 13`` had 21 and 24, ``15 x 16`` had 22, ``21 x 22`` had 30."""
    rows = [
        # id, generation, sex, mother, father
        (10, 0, 0, -1, -1),
        (13, 0, 1, -1, -1),
        (15, 0, 0, -1, -1),
        (16, 0, 1, -1, -1),
        (21, 1, 0, 10, 13),
        (22, 1, 1, 15, 16),
        (24, 1, 0, 10, 13),
        (30, 2, 0, 21, 22),
    ]
    frame = pl.DataFrame(rows, schema=["id", "generation", "sex", "mother", "father"], orient="row")
    frame = frame.with_columns(
        pl.all().cast(pl.Int32), twin=pl.lit(-1, pl.Int32), household_id=pl.col("mother").rank("dense").cast(pl.Int32)
    )
    return schema_pad(frame, PEDIGREE)


@pytest.mark.parametrize(
    ("sampled", "expected"),
    [
        ((30,), {30, 22, 15, 16}),
        ((24, 30), {30, 22, 15, 16, 24, 10, 13}),
    ],
    ids=["grandparents-unreached", "grandparents-reached-via-aunt"],
)
def test_closure_does_not_cross_a_dropped_parent(sampled, expected):
    """Closure stops at a dropped intermediate parent and never links past it.

    The seed is the first whose one-person dropout removes exactly 30's mother
    21, found by scanning rather than pinned so a changed RNG stream cannot
    silently remove someone else.  Her surviving parents 10 and 13 then join
    the pedigree only through another sampled descendant (24), and 30's mother
    becomes -1 rather than 10.  Rejects building the closure before dropout and
    reconnecting ancestry across a removed person.
    """
    pedigree = _interrupted_pedigree()
    rate = 1 / len(pedigree)
    seed = next(s for s in range(1000) if 21 not in _apply_dropout(pedigree, rate, np.random.default_rng(s))["id"])
    survivors = pedigree.filter(pl.col("id") != 21)
    assert _apply_dropout(pedigree, rate, np.random.default_rng(seed)).equals(survivors)
    assert _available_ancestry(survivors, sampled) == expected

    trait = schema_pad(pl.DataFrame({"id": pl.Series(sampled, dtype=pl.Int32)}), CENSORED_TRAIT)
    ped_out, _ = run_ascertainment(pedigree, trait, dropout_rate=rate, N_sample=0, seed=seed)

    assert set(ped_out["id"].to_list()) == expected
    assert ped_out.filter(pl.col("id") == 30).select("mother", "father").row(0) == (-1, 22)
