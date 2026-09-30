"""Generation labelling of the relationship moments (ADR 0020, plan D9.4).

A deterministic chain pedigree gives every generation four mated couples with
two offspring each, so the pair counts per generation are known in closed
form and the report selections can be checked against a pair-list oracle.
"""

from __future__ import annotations

import time

import numpy as np
import polars as pl
import pytest
from pedigree_graph import PedigreeGraph

from simace.analysis.stats.correlations import (
    compute_liability_correlations,
    compute_tetrachoric,
    compute_tetrachoric_by_generation,
    compute_tetrachoric_by_sex,
)
from simace.analysis.stats.moments import Stratum, relationship_moments_for
from simace.analysis.stats.tetrachoric import tetrachoric_corr_se
from simace.core.relationships import RELATIONSHIP_TYPES

PER_GENERATION = 8  # four couples, two offspring each


def chain_pedigree(generation_numbers: list[int], *, seed: int = 0, with_generation: bool = True) -> pl.DataFrame:
    """Eight individuals per generation: couple ``c`` of generation ``g-1`` has two offspring in ``g``.

    Family ``c`` of a generation is the daughter (``2c``) and son (``2c+1``)
    of couple ``c``; couple ``c`` of the next generation is the daughter of
    family ``c`` with the son of family ``c+1``, so no mating is between
    siblings and every generation has both sexes and every affection level.
    Generation numbers are the caller's, so they can be noncontiguous.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for level, gen in enumerate(generation_numbers):
        for j in range(PER_GENERATION):
            ident = level * PER_GENERATION + j
            if level == 0:
                mother = father = -1
            else:
                couple = j // 2
                previous = (level - 1) * PER_GENERATION
                mother = previous + 2 * couple
                father = previous + 2 * ((couple + 1) % (PER_GENERATION // 2)) + 1
            rows.append((ident, mother, father, j % 2, gen))
    ids = np.array([r[0] for r in rows], dtype=np.int32)
    n = len(ids)
    frame = {
        "id": ids,
        "mother": np.array([r[1] for r in rows], dtype=np.int32),
        "father": np.array([r[2] for r in rows], dtype=np.int32),
        "twin": np.full(n, -1, dtype=np.int32),
        "sex": np.array([r[3] for r in rows], dtype=np.int8),
        "affected1": ids % 3 == 0,
        "affected2": ids % 5 == 1,
        "liability1": rng.normal(size=n),
        "liability2": rng.normal(size=n) * 2.0 + 0.5,
    }
    if with_generation:
        frame["generation"] = np.array([r[4] for r in rows], dtype=np.int32)
    return pl.DataFrame(frame)


def _moments(frame: pl.DataFrame):
    return relationship_moments_for(frame, PedigreeGraph.from_frame(frame))


def _pairs(frame: pl.DataFrame):
    return PedigreeGraph.from_frame(frame).relationship_pairs(categories=RELATIONSHIP_TYPES)


def _first_generation_levels(moments) -> list[int]:
    return [int(v) for v in moments.axis("first_generation").levels]


class TestGenerationLevels:
    @pytest.mark.parametrize("numbers", [[1, 2], [1, 2, 3, 4, 5]])
    def test_every_generation_is_its_own_level(self, numbers):
        """Fewer and more than three generations: founders and older ones are kept apart."""
        moments = _moments(chain_pedigree(numbers))
        assert _first_generation_levels(moments) == numbers
        assert moments.shape == (len(RELATIONSHIP_TYPES), len(numbers), 2, 2, 2, 2, 2, 2)

    def test_noncontiguous_generation_numbers(self):
        numbers = [2, 5, 9]
        moments = _moments(chain_pedigree(numbers))
        assert _first_generation_levels(moments) == numbers
        for gen in numbers:
            selected = moments.select(first_generation=gen)
            assert selected.axis("first_generation").levels.tolist() == [gen]
        with pytest.raises(ValueError, match="not a level"):
            moments.select(first_generation=3)

    def test_absent_generation_is_one_level(self):
        frame = chain_pedigree([1, 2, 3], with_generation=False)
        moments = _moments(frame)
        assert _first_generation_levels(moments) == [0]
        assert compute_tetrachoric_by_generation(frame, moments=moments) == {}
        stratum = Stratum(moments)
        pairs = _pairs(frame)
        for code in RELATIONSHIP_TYPES:
            assert int(stratum.n_pairs[stratum.index(code)]) == len(pairs[code])

    def test_pairs_spanning_generations_sit_at_the_first_member(self):
        """MO/FO pairs belong to the offspring's generation; siblings to their shared one."""
        numbers = [1, 2, 3, 4]
        moments = _moments(chain_pedigree(numbers))
        for gen in numbers:
            per_gen = Stratum(moments.select(first_generation=gen))
            n = {code: int(per_gen.n_pairs[per_gen.index(code)]) for code in RELATIONSHIP_TYPES}
            if gen == numbers[0]:
                assert n["MO"] == n["FO"] == n["FS"] == 0
            else:
                assert n["MO"] == n["FO"] == PER_GENERATION
                assert n["FS"] == PER_GENERATION // 2

    def test_overall_totals_keep_every_pair(self):
        frame = chain_pedigree([1, 2, 3, 4, 5])
        moments = _moments(frame)
        pairs = _pairs(frame)
        overall = Stratum(moments)
        for code in RELATIONSHIP_TYPES:
            total = int(overall.n_pairs[overall.index(code)])
            assert total == len(pairs[code])
            by_gen = [
                int(Stratum(moments.select(first_generation=gen)).n_pairs[overall.index(code)])
                for gen in _first_generation_levels(moments)
            ]
            assert sum(by_gen) == total


@pytest.fixture(scope="module")
def six_generations():
    return chain_pedigree([1, 2, 3, 4, 5, 6], seed=3)


@pytest.fixture(scope="module")
def thirty_six_generations():
    return chain_pedigree(list(range(1, 37)), seed=36)


class TestReportSelectionsAgainstPairs:
    """The report's strata reproduce a pair-list computation on the same engine."""

    def _oracle(self, frame, pairs, mask_of):
        affected = {t: frame[f"affected{t}"].to_numpy() for t in (1, 2)}
        liability = {t: frame[f"liability{t}"].to_numpy() for t in (1, 2)}
        out = {}
        for t in (1, 2):
            out[f"trait{t}"] = {}
            for code in RELATIONSHIP_TYPES:
                i1, i2 = pairs[code].first_rows, pairs[code].second_rows
                keep = mask_of(i1, i2)
                i1, i2 = i1[keep], i2[keep]
                entry = {"r": None, "se": None, "n_pairs": len(i1), "liability_r": None}
                if len(i1) >= 10:
                    r, se = tetrachoric_corr_se(affected[t][i1], affected[t][i2])
                    entry["r"], entry["se"] = (None if np.isnan(r) else float(r)), (None if np.isnan(se) else float(se))
                    entry["liability_r"] = float(np.corrcoef(liability[t][i1], liability[t][i2])[0, 1])
                out[f"trait{t}"][code] = entry
        return out

    @staticmethod
    def _assert_close(got, want):
        assert got.keys() == want.keys()
        for trait in got:
            for code in RELATIONSHIP_TYPES:
                g, w = got[trait][code], want[trait][code]
                assert g["n_pairs"] == w["n_pairs"], (trait, code)
                for key in ("r", "se", "liability_r"):
                    if w[key] is None:
                        assert g[key] is None, (trait, code, key)
                    else:
                        assert g[key] == pytest.approx(w[key], abs=1e-10), (trait, code, key)

    def test_by_generation_uses_the_last_three(self, six_generations):
        frame = six_generations
        moments = _moments(frame)
        pairs = _pairs(frame)
        gen = frame["generation"].to_numpy()
        got = compute_tetrachoric_by_generation(frame, moments=moments)
        assert list(got) == ["gen4", "gen5", "gen6"]
        for g in (4, 5, 6):
            self._assert_close(got[f"gen{g}"], self._oracle(frame, pairs, lambda i1, i2, g=g: gen[i1] == g))

    def test_by_sex_reads_both_members(self, six_generations):
        frame = six_generations
        moments = _moments(frame)
        pairs = _pairs(frame)
        sex = frame["sex"].to_numpy()
        got = compute_tetrachoric_by_sex(moments=moments)
        for value, label in ((0, "female"), (1, "male")):
            self._assert_close(
                got[label], self._oracle(frame, pairs, lambda i1, i2, v=value: (sex[i1] == v) & (sex[i2] == v))
            )

    def test_overall(self, six_generations):
        frame = six_generations
        moments = _moments(frame)
        pairs = _pairs(frame)
        want = self._oracle(frame, pairs, lambda i1, i2: np.ones(len(i1), dtype=bool))
        got_t = compute_tetrachoric(moments=moments)
        got_l = compute_liability_correlations(moments=moments)
        for trait in want:
            for code in RELATIONSHIP_TYPES:
                assert got_t[trait][code]["n_pairs"] == want[trait][code]["n_pairs"]
                w = want[trait][code]["liability_r"]
                assert (got_l[trait][code] is None) == (w is None)
                if w is not None:
                    assert got_l[trait][code] == pytest.approx(w, abs=1e-10)


class TestThirtySixGenerations:
    """The ADHD composite's 36 phenotyped generations: 288 first-member cells."""

    def test_layout_and_totals(self, thirty_six_generations):
        frame = thirty_six_generations
        t0 = time.perf_counter()
        moments = _moments(frame)
        elapsed = time.perf_counter() - t0
        assert moments.shape == (len(RELATIONSHIP_TYPES), 36, 2, 2, 2, 2, 2, 2)
        assert int(np.prod(moments.shape[1:5])) == 288
        pairs = _pairs(frame)
        overall = Stratum(moments)
        for code in RELATIONSHIP_TYPES:
            assert int(overall.n_pairs[overall.index(code)]) == len(pairs[code])
        assert elapsed < 30.0

    def test_each_generation_selectable(self, thirty_six_generations):
        frame = thirty_six_generations
        moments = _moments(frame)
        gen = frame["generation"].to_numpy()
        pairs = _pairs(frame)
        for g in (1, 2, 18, 36):
            per_gen = Stratum(moments.select(first_generation=g))
            for code in ("MO", "FS", "1C"):
                want = int((gen[pairs[code].first_rows] == g).sum())
                assert int(per_gen.n_pairs[per_gen.index(code)]) == want

    def test_report_selection_is_the_last_three(self, thirty_six_generations):
        frame = thirty_six_generations
        got = compute_tetrachoric_by_generation(frame, moments=_moments(frame))
        assert list(got) == ["gen34", "gen35", "gen36"]
