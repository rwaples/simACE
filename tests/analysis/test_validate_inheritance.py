"""The sibling inheritance calibrated tests (ADR 0029): Mendelian residual and household draw cells."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import polars as pl
import pytest

from simace.analysis.validate.inheritance import (
    MIN_PAIRS,
    linked_household_share,
    parent_rows,
    run_household_draw,
    run_mendelian_residual,
)
from simace.analysis.validate.on_demand import ON_DEMAND_CHECKS
from simace.simulation.simulate import run_simulation

PARAMS = dict(
    seed=7,
    N=2000,
    G_ped=3,
    G_sim=3,
    mating_lambda=0.5,
    p_mztwin=0.02,
    A1=0.5,
    C1=0.2,
    E1=0.3,
    A2=0.4,
    C2=0.3,
    E2=0.3,
    rA=0.3,
    rC=0.0,
)


@pytest.fixture(scope="module")
def pedigree() -> pl.DataFrame:
    return run_simulation(**PARAMS)


def _inputs(pedigree: pl.DataFrame, **overrides) -> SimpleNamespace:
    return SimpleNamespace(pedigree=pedigree, params={**PARAMS, **overrides})


def _midparent_children(pedigree: pl.DataFrame, column: str, b_m: float, b_f: float) -> pl.DataFrame:
    """The pedigree with each child's ``column`` replaced by its transmitted part, generation by generation."""
    mother, father = parent_rows(pedigree)
    x = pedigree[column].to_numpy().astype(np.float64)
    for g in sorted(pedigree["generation"].unique().to_list()):
        rows = np.flatnonzero((pedigree["generation"].to_numpy() == g) & (mother >= 0) & (father >= 0))
        x[rows] = b_m * x[mother[rows]] + b_f * x[father[rows]]
    return pedigree.with_columns(pl.Series(column, x, dtype=pedigree.schema[column]))


def test_a_correct_simulation_scores_every_cell_near_zero(pedigree) -> None:
    section = run_mendelian_residual(_inputs(pedigree))
    assert set(section["cells"]) == {f"trait{t}_{c}" for t in (1, 2) for c in ("FS", "MHS", "PHS", "variance")}
    assert all(cell["status"] == "scored" for cell in section["cells"].values())
    assert all(abs(cell["z"]) < 4 for cell in section["cells"].values())
    assert section["cells"]["trait1_variance"]["expected"] == pytest.approx(0.25)
    assert section["summary"]["n_scored"] == 8
    household = run_household_draw(_inputs(pedigree))
    assert set(household["cells"]) == {"trait1_PHS", "trait2_PHS"}
    assert all(cell["status"] == "scored" and abs(cell["z"]) < 4 for cell in household["cells"].values())
    assert 0.0 <= household["cells"]["trait1_PHS"]["linked_household_share"] < 0.05


def test_a_trait_configured_without_the_component_has_no_cell(pedigree) -> None:
    assert set(run_mendelian_residual(_inputs(pedigree, A2=0.0))["cells"]) == {
        f"trait1_{c}" for c in ("FS", "MHS", "PHS", "variance")
    }
    assert set(run_household_draw(_inputs(pedigree, C1=0.0))["cells"]) == {"trait2_PHS"}


def test_a_child_equal_to_its_midparent_is_not_applicable(pedigree) -> None:
    section = run_mendelian_residual(_inputs(_midparent_children(pedigree, "A1", 0.5, 0.5)))
    trait1 = {name: cell for name, cell in section["cells"].items() if name.startswith("trait1_")}
    assert {cell["status"] for cell in trait1.values()} == {"not_applicable"}
    assert section["cells"]["trait2_FS"]["status"] == "scored"
    assert section["summary"]["n_scored"] == 4


def test_a_fully_transmitted_c_is_not_applicable(pedigree) -> None:
    transmitted = _midparent_children(pedigree, "C1", 0.6, 0.4)
    cell = run_household_draw(_inputs(transmitted, Ctm_maternal1=0.6, Ctm_paternal1=0.4))["cells"]["trait1_PHS"]
    assert cell["status"] == "not_applicable"
    assert "linked_household_share" in cell


def test_a_wrong_transmission_coefficient_is_seen_as_correlated_draws(pedigree) -> None:
    """Scoring C with b_f = −0.5 when the simulation used 0 adds half the father's C, which PHS share."""
    cell = run_household_draw(_inputs(pedigree, Ctm_paternal1=-0.5))["cells"]["trait1_PHS"]
    assert cell["status"] == "scored"
    assert cell["z"] > 4


def test_too_few_pairs_is_information(pedigree) -> None:
    small = pedigree.filter(pl.col("generation") <= pedigree["generation"].min() + 1).head(300)
    cells = run_mendelian_residual(_inputs(small))["cells"]
    assert cells["trait1_PHS"]["status"] == "information"
    assert cells["trait1_PHS"]["reason"].endswith(f"< {MIN_PAIRS}")


def test_the_variance_counts_one_member_of_each_mz_pair_and_skips_unknown_parents(pedigree) -> None:
    mother, father = parent_rows(pedigree)
    ids, twin = pedigree["id"].to_numpy(), pedigree["twin"].to_numpy()
    with_parents = (mother >= 0) & (father >= 0)
    assert (twin >= 0).any()
    expected = int((with_parents & ((twin < 0) | (ids < twin))).sum())
    cell = run_mendelian_residual(_inputs(pedigree))["cells"]["trait1_variance"]
    assert cell["n_residuals"] == expected
    # A mother id that is not a row leaves its children without a residual, never a lookup of another row.
    orphaned = pedigree.with_columns(
        pl.when(pl.col("mother") >= 0).then(pl.col("mother") + 10**9).otherwise(pl.col("mother")).alias("mother")
    )
    assert (parent_rows(orphaned)[0] == -1).all()
    cells = run_mendelian_residual(_inputs(orphaned))["cells"]
    assert cells["trait1_variance"]["status"] == "information"


def test_linked_households_count_pairs_of_households_sharing_two_fathers() -> None:
    # Mothers 1 and 2 each have a child by fathers 10 and 11: households 0 and 1 are linked twice.
    pedigree = pl.DataFrame(
        {
            "mother": [1, 1, 2, 2, 3],
            "father": [10, 11, 10, 11, 10],
            "household_id": [0, 0, 1, 1, 2],
        }
    )
    # PHS household-pair products: (0,1) via 10 and 11 → 2 pairs, linked; (0,2) and (1,2) via 10 → 1 each.
    assert linked_household_share(pedigree) == pytest.approx(2 / 4)


def test_both_tests_are_registered_with_their_cells() -> None:
    assert ON_DEMAND_CHECKS["mendelian_residual"].cell_names[:4] == (
        "trait1_FS",
        "trait1_MHS",
        "trait1_PHS",
        "trait1_variance",
    )
    assert ON_DEMAND_CHECKS["household_draw"].cell_names == ("trait1_PHS", "trait2_PHS")
    assert ON_DEMAND_CHECKS["household_draw"].min_reps == 20
