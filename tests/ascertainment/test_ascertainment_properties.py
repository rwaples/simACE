"""Property-based tests for :mod:`simace.ascertainment.runner`.

Structural invariants of the dropout → case-weighted draw → ancestor-closure
pipeline.  Every assertion here is exact: sizes, set membership, row order, and
RNG-state identity.  Nothing depends on a statistical tolerance, and no
planted parameter is recovered.
"""

from typing import NamedTuple

import numpy as np
import polars as pl
import pytest
from hypothesis import given
from hypothesis import strategies as st

from simace.ascertainment.runner import _apply_dropout, _sample_trait_ids, run_ascertainment
from simace.core.trait_schema import CENSORED_TRAIT
from tests.conftest import pedigree_frame, relabel_ids, schema_pad


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


def _row_positions(haystack: np.ndarray, needles: np.ndarray) -> np.ndarray:
    """Row positions of ``needles`` within ``haystack``; both must be id arrays."""
    order = np.argsort(haystack)
    return order[np.searchsorted(haystack[order], needles)]


class TestRunAscertainment:
    """Structural invariants of the public ``run_ascertainment`` API."""

    @given(case=_ascertainment_case(relabel=True))
    def test_no_dangling_references(self, case):
        """Every mother/father/twin in the output resolves to -1 or an output id."""
        ped_out, _ = case.run()
        out_ids = ped_out["id"].to_numpy()
        for col in ("mother", "father", "twin"):
            values = ped_out[col].to_numpy()
            resolved = (values == -1) | np.isin(values, out_ids)
            assert resolved.all(), f"{col} has {int((~resolved).sum())} dangling references"

    @given(case=_ascertainment_case(relabel=True))
    def test_never_invents_individuals(self, case):
        """Outputs are subsets: ped ⊆ ped_in, trait ⊆ trait_in, and trait ⊆ ped_out."""
        ped_out, trait_out = case.run()
        ped_in_ids = set(case.pedigree["id"].to_list())
        trait_in_ids = set(case.trait["id"].to_list())
        ped_out_ids = set(ped_out["id"].to_list())
        trait_out_ids = set(trait_out["id"].to_list())

        assert ped_out_ids <= ped_in_ids
        assert trait_out_ids <= trait_in_ids
        assert trait_out_ids <= ped_out_ids
        assert len(ped_out_ids) == len(ped_out)
        assert len(trait_out_ids) == len(trait_out)

    @given(case=_ascertainment_case())
    def test_parent_pointers_intact_without_dropout(self, case):
        """At ``dropout_rate=0`` the closure follows every parent, so none is severed.

        The docstring at ``runner.py`` claims parent links are safe by
        construction *only* here; at rate > 0 an ancestor can be removed before
        the closure is built and severing is legitimate, so the negation is not
        asserted there.
        """
        ped_out, _ = case.run(dropout_rate=0.0)
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
        sampled, _ = case.sample_ids()
        assert set(sampled.tolist()) <= set(pool_ids.tolist())
        assert len(set(sampled.tolist())) == len(sampled)
        if len(sampled):
            positions = _row_positions(pool_ids, sampled)
            assert np.all(np.diff(positions) > 0)

    @given(case=_ascertainment_case())
    def test_sample_size_follows_the_documented_branches(self, case):
        """Size is 0 / whole pool / ``N_sample`` / controls-clamped, per branch."""
        n_pool = len(case.trait)
        is_case = case.trait["affected1"].to_numpy()
        n_controls = int((~is_case).sum())
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
        a draw happens at all.  An all-case pool therefore yields an empty
        selection; the ``N_sample`` pass-through path stays ratio-independent
        and is excluded here.
        """
        n_pool = len(case.trait)
        if not (0 < case.n_sample < n_pool):
            return
        is_case = case.trait["affected1"].to_numpy()
        n_controls = int((~is_case).sum())
        sampled, _ = case.sample_ids(ratio=0.0)
        selected = case.trait.filter(pl.Series(np.isin(case.trait["id"].to_numpy(), sampled)))
        assert not selected["affected1"].any()
        assert len(sampled) == min(case.n_sample, n_controls)
