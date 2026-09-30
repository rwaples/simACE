"""Household-derived sibling counts (plan D6.3): O(N) from mother group sizes."""

from __future__ import annotations

import numpy as np
import polars as pl
from pedigree_graph import PedigreeGraph

from simace.analysis.validate.half_sibs import household_sibling_counts
from simace.simulation.simulate import run_simulation


def _frame(rows: list[tuple[int, int, int, int]]) -> pl.DataFrame:
    """Rows of ``(id, mother, father, twin)``; parents referenced but not listed are added as founders."""
    ids = {r[0] for r in rows}
    parents = {p for r in rows for p in (r[1], r[2]) if p != -1 and p not in ids}
    founders = [(p, -1, -1, -1) for p in sorted(parents)]
    all_rows = founders + rows
    return pl.DataFrame(
        {
            "id": np.array([r[0] for r in all_rows], dtype=np.int32),
            "mother": np.array([r[1] for r in all_rows], dtype=np.int32),
            "father": np.array([r[2] for r in all_rows], dtype=np.int32),
            "twin": np.array([r[3] for r in all_rows], dtype=np.int32),
        }
    )


class TestHouseholdSiblingCounts:
    def test_two_twin_pairs_in_one_household(self):
        """Each twin has a sibling besides its co-twin; nobody has a different father."""
        frame = _frame([(10, 1, 2, 11), (11, 1, 2, 10), (12, 1, 2, 13), (13, 1, 2, 12)])
        assert household_sibling_counts(frame) == {
            "n_offspring_with_sibs": 4,
            "n_offspring_with_maternal_half_sib": 0,
        }

    def test_twin_whose_only_maternal_sibling_is_the_co_twin(self):
        frame = _frame([(10, 1, 2, 11), (11, 1, 2, 10), (12, 3, 4, -1)])
        assert household_sibling_counts(frame) == {
            "n_offspring_with_sibs": 0,
            "n_offspring_with_maternal_half_sib": 0,
        }

    def test_unknown_father_differs_from_every_father(self):
        """One known and one unknown father, and two unknown fathers, are both maternal half-sib households."""
        frame = _frame(
            [(10, 1, 2, -1), (11, 1, -1, -1), (12, 3, -1, -1), (13, 3, -1, -1), (14, 5, 6, -1), (15, 5, 6, -1)]
        )
        assert household_sibling_counts(frame) == {
            "n_offspring_with_sibs": 6,
            "n_offspring_with_maternal_half_sib": 4,
        }

    def test_co_twins_with_an_unknown_father_are_not_half_sibs(self):
        """The engine files the pair as MZ, so neither co-twin has a maternal half-sib."""
        frame = _frame([(10, 1, -1, 11), (11, 1, -1, 10)])
        pairs = PedigreeGraph.from_frame(frame).relationship_pairs(categories=["MZ", "MHS"])
        assert (len(pairs["MZ"]), len(pairs["MHS"])) == (1, 0)
        assert household_sibling_counts(frame) == {
            "n_offspring_with_sibs": 0,
            "n_offspring_with_maternal_half_sib": 0,
        }

    def test_unknown_mother_takes_part_in_neither_count(self):
        frame = _frame([(10, -1, 2, -1), (11, -1, 2, -1), (12, -1, -1, -1), (13, 1, 2, -1), (14, 1, 3, -1)])
        assert household_sibling_counts(frame) == {
            "n_offspring_with_sibs": 2,
            "n_offspring_with_maternal_half_sib": 2,
        }

    def test_matches_the_engine_on_a_simulated_pedigree(self):
        """Distinct members of FS ∪ MHS and of MHS pairs, twins included (pedigree-graph #29)."""
        pedigree = run_simulation(
            seed=11, N=600, G_ped=3, G_sim=3, mating_lambda=0.8, p_mztwin=0.05,
            A1=0.5, C1=0.2, E1=0.3, A2=0.5, C2=0.2, E2=0.3, rA=0.3, rC=0.5, assort1=0.0, assort2=0.0,
        )  # fmt: skip
        assert (pedigree["twin"] != -1).sum() > 0
        pairs = PedigreeGraph.from_frame(pedigree).relationship_pairs(categories=("FS", "MHS"))
        n = len(pedigree)

        def members(*codes: str) -> int:
            seen = np.zeros(n, dtype=bool)
            for code in codes:
                seen[pairs[code].first_rows] = True
                seen[pairs[code].second_rows] = True
            return int(seen.sum())

        assert household_sibling_counts(pedigree) == {
            "n_offspring_with_sibs": members("FS", "MHS"),
            "n_offspring_with_maternal_half_sib": members("MHS"),
        }
