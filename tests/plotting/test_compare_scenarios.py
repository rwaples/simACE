"""Tests for the cross-scenario comparison loaders."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import polars as pl
import pytest

from simace.core.parquet import save_parquet
from simace.plotting.compare_scenarios import load_pedigree_estimates, load_sib_pair_liabilities

if TYPE_CHECKING:
    from pathlib import Path


def _write_pedigree(path: Path, rows: list[tuple[int, int, int, int, int, float]]) -> Path:
    """Write a hand-built pedigree parquet from ``(id, mother, father, twin, gen, liab)``."""
    df = pl.DataFrame(
        rows,
        schema={
            "id": pl.Int64,
            "mother": pl.Int64,
            "father": pl.Int64,
            "twin": pl.Int64,
            "generation": pl.Int64,
            "liability1": pl.Float64,
        },
        orient="row",
    )
    save_parquet(df, path)
    return path


# Four founders, then one mating with three full sibs, a second mating with a
# single child, and an MZ twin pair sharing the first mating's parents.
_BASIC = [
    (0, -1, -1, -1, 0, 0.0),
    (1, -1, -1, -1, 0, 0.0),
    (2, -1, -1, -1, 0, 0.0),
    (3, -1, -1, -1, 0, 0.0),
    (4, 0, 1, -1, 1, 1.0),
    (5, 0, 1, -1, 1, 2.0),
    (6, 0, 1, -1, 1, 3.0),
    (7, 2, 3, -1, 1, 4.0),
    (8, 0, 1, 9, 1, 5.0),
    (9, 0, 1, 8, 1, 6.0),
]


@pytest.fixture
def basic_pedigree(tmp_path: Path) -> Path:
    return _write_pedigree(tmp_path / "basic.parquet", _BASIC)


def test_picks_the_first_two_full_sibs_per_mating(basic_pedigree: Path):
    liab_a, liab_b = load_sib_pair_liabilities([basic_pedigree], trait=1)

    # Only the (0, 1) mating yields a pair, and it is ids 4 and 5 — id 6 is a
    # third sib, id 7 is an only child, ids 8/9 are MZ twins.
    assert liab_a.tolist() == [1.0]
    assert liab_b.tolist() == [2.0]


def test_concatenates_across_replicates(basic_pedigree: Path, tmp_path: Path):
    second = _write_pedigree(tmp_path / "second.parquet", _BASIC)
    liab_a, liab_b = load_sib_pair_liabilities([basic_pedigree, second], trait=1)

    assert liab_a.tolist() == [1.0, 1.0]
    assert liab_b.tolist() == [2.0, 2.0]


def test_severed_father_keys_separately_from_a_full_mating(tmp_path: Path):
    """A ``father == -1`` mating groups on its own, not with the mother's other one.

    Mother 2 has an only child by father 3 and two children by a severed
    father. The severed pair is the only pair; if the two matings were keyed
    together the group would be ids 7, 10, 11 and the pair would be 7 with 10.
    """
    path = _write_pedigree(
        tmp_path / "severed.parquet",
        [
            (0, -1, -1, -1, 0, 0.0),
            (1, -1, -1, -1, 0, 0.0),
            (2, -1, -1, -1, 0, 0.0),
            (3, -1, -1, -1, 0, 0.0),
            (7, 2, 3, -1, 1, 4.0),
            (10, 2, -1, -1, 1, 7.0),
            (11, 2, -1, -1, 1, 8.0),
        ],
    )
    liab_a, liab_b = load_sib_pair_liabilities([path], trait=1)

    assert liab_a.tolist() == [7.0]
    assert liab_b.tolist() == [8.0]


def test_returns_empty_arrays_when_no_pedigree_has_a_pair(basic_pedigree: Path):
    """min_generation can leave every pedigree empty — np.concatenate rejects []."""
    liab_a, liab_b = load_sib_pair_liabilities([basic_pedigree], trait=1, min_generation=2)

    assert liab_a.size == 0
    assert liab_b.size == 0
    assert liab_a.dtype == np.float64


def test_returns_empty_arrays_for_no_paths():
    liab_a, liab_b = load_sib_pair_liabilities([], trait=1)

    assert liab_a.size == 0
    assert liab_b.size == 0


def _three_generations(path: Path, seed: int = 3, n: int = 120) -> pl.DataFrame:
    """Write *n* founders per sex, then two generations of *n* children each; return the frame."""
    rng = np.random.default_rng(seed)
    ids = list(range(2 * n))
    mothers, fathers, gens = [-1] * (2 * n), [-1] * (2 * n), [1] * (2 * n)
    sexes = [i % 2 for i in ids]
    females, males = np.arange(0, 2 * n, 2), np.arange(1, 2 * n, 2)
    for gen in (2, 3):
        start = len(ids)
        for k in range(2 * n):
            ids.append(start + k)
            mothers.append(int(rng.choice(females)))
            fathers.append(int(rng.choice(males)))
            gens.append(gen)
            sexes.append(k % 2)
        females, males = np.arange(start, start + 2 * n, 2), np.arange(start + 1, start + 2 * n, 2)
    size = len(ids)
    a, c, e = rng.normal(size=size), rng.normal(size=size), rng.normal(size=size)
    df = pl.DataFrame(
        {
            "id": ids,
            "sex": sexes,
            "mother": mothers,
            "father": fathers,
            "twin": [-1] * size,
            "generation": gens,
            "A1": a,
            "C1": c,
            "E1": e,
            "liability1": a + c + e + 0.4 * np.arange(size) / size,
        }
    )
    save_parquet(df, path)
    return df


def _oracle_po_slope(df_full: pl.DataFrame, min_generation: int | None) -> float:
    """OLS slope of the filtered rows' liability on their parents' mean liability, parents read from every row."""
    df = df_full if min_generation is None else df_full.filter(pl.col("generation") >= min_generation)
    liab = dict(zip(df_full["id"].to_list(), df_full["liability1"].to_list(), strict=True))
    rows = [
        (liab[m], liab[f], y)
        for m, f, y in df.select("mother", "father", "liability1").iter_rows()
        if m in liab and f in liab
    ]
    mid = np.array([(m + f) / 2 for m, f, _ in rows])
    y = np.array([y for _, _, y in rows])
    return float(np.polyfit(mid, y, 1)[0])


@pytest.mark.parametrize("min_generation", [None, 3])
def test_po_slope_reads_parents_from_the_full_pedigree(tmp_path: Path, min_generation: int | None):
    """With generation 3 kept, every parent is a filtered-out generation-2 row."""
    df_full = _three_generations(tmp_path / "ped.parquet")
    estimates = load_pedigree_estimates(tmp_path / "ped.parquet", trait=1, min_generation=min_generation)
    assert estimates["po_slope"] == pytest.approx(_oracle_po_slope(df_full, min_generation), rel=1e-10)


def test_po_slope_is_nan_below_ten_children(tmp_path: Path):
    df = _three_generations(tmp_path / "ped.parquet", n=4)
    save_parquet(df.filter(pl.col("generation") < 3), tmp_path / "small.parquet")
    assert np.isnan(load_pedigree_estimates(tmp_path / "small.parquet", trait=1, min_generation=2)["po_slope"])
