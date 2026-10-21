"""Per-generation mate correlations as a calibrated test (ADR 0031)."""

from __future__ import annotations

import itertools
import math
from typing import TYPE_CHECKING

import numpy as np
import polars as pl
import pytest
import yaml

from simace.analysis.mate_correlation import mate_correlation_matrix, mate_pedigree
from simace.analysis.validate import mate_correlation as mc
from simace.analysis.validate.calibrated import RepInputs
from simace.analysis.validate.mate_correlation import CELL_NAMES, MIN_PAIRS, n_couples, run_mate_correlation
from simace.analysis.validate.on_demand import ON_DEMAND_CHECKS
from simace.analysis.validate.verdict import decide, scenario_family
from simace.core.cohort import write_pedigree
from simace.simulation.simulate import run_simulation

if TYPE_CHECKING:
    from pathlib import Path

BASE = dict(
    seed=21,
    N=1000,
    G_ped=6,
    G_sim=8,
    mating_lambda=0.5,
    p_mztwin=0.02,
    A1=0.5,
    C1=0.2,
    E1=0.3,
    A2=0.4,
    C2=0.2,
    E2=0.4,
    rA=0.3,
    rC=0.0,
)
#: Copula on trait 1 in parents' generations 0–1, Metropolis on both in 2–4 (``inh_am_mixed``).
MIXED = {**BASE, "assort1": 0.3, "assort2": {0: 0.0, 5: 0.3}}


@pytest.fixture(scope="module")
def mixed() -> pl.DataFrame:
    return run_simulation(**MIXED)


_written = itertools.count()


def _inputs(tmp_path: Path, pedigree: pl.DataFrame, params: dict) -> RepInputs:
    k = next(_written)
    pedigree_path, report_path = tmp_path / f"pedigree{k}.parquet", tmp_path / f"report{k}.yaml"
    write_pedigree(pedigree, pedigree_path)
    report_path.write_text(yaml.safe_dump({"inputs": {"parameters": params}}))
    return RepInputs(pedigree_path, report_path, seed=params["seed"], phenotype={})


def _run(tmp_path: Path, pedigree: pl.DataFrame, params: dict) -> dict:
    return run_mate_correlation(_inputs(tmp_path, pedigree, params))


def test_registered_with_its_own_page_and_twelve_generations_of_cells() -> None:
    test = ON_DEMAND_CHECKS["mate_correlation"]
    assert test.cell_names == CELL_NAMES
    assert len(CELL_NAMES) == 48
    assert CELL_NAMES[:4] == ("g0_liability1", "g0_liability2", "g0_cross_12", "g0_cross_21")
    assert CELL_NAMES[-1] == "g11_cross_21"
    assert test.own_page
    assert test.min_reps == 20


def test_cells_follow_the_pairing_mode(tmp_path, mixed) -> None:
    section = _run(tmp_path, mixed, MIXED)
    cells = section["cells"]
    assert "not_applicable" not in section
    for g in (0, 1):
        assert cells[f"g{g}_liability1"]["status"] == "scored"
        induced = cells[f"g{g}_liability2"]
        assert induced["status"] == "not_applicable"
        assert induced["reason"] == "induced by assortment on trait 1"
        assert {"observed", "expected", "n_pairs"} <= set(induced)
        assert f"g{g}_cross_12" not in cells
    for g in (2, 3, 4):
        for label in ("liability1", "liability2", "cross_12", "cross_21"):
            cell = cells[f"g{g}_{label}"]
            assert cell["status"] == "scored"
            assert abs(cell["z"]) < 4
    assert section["summary"]["n_scored"] == 2 + 12


def test_the_finite_population_factor_applies_only_where_both_traits_assort(tmp_path, mixed) -> None:
    cells = _run(tmp_path, mixed, MIXED)["cells"]
    copula = cells["g0_liability1"]
    assert "p_hat" not in copula
    for label in ("liability1", "cross_12"):
        cell = cells[f"g3_{label}"]
        assert 0 < cell["p_hat"] < 1
        assert cell["p_hat"] == pytest.approx(cell["n_pairs"] / cell["M_hat"])
        assert cell["se"] == pytest.approx(cell["se_iid"] * math.sqrt(1 - cell["p_hat"]))


@pytest.mark.parametrize("g", [0, 3])
def test_r_matches_pg_phenotype(tmp_path, mixed, g) -> None:
    cells = _run(tmp_path, mixed, MIXED)["cells"]
    rows = mixed.filter(pl.col("generation").is_in([g, g + 1]))
    matrix, n_pairs = mate_correlation_matrix(mate_pedigree(rows), rows, ["liability1", "liability2"])
    assert cells[f"g{g}_liability1"]["n_pairs"] == n_pairs
    assert cells[f"g{g}_liability1"]["observed"] == pytest.approx(matrix[0, 0], abs=1e-6)
    assert cells[f"g{g}_liability2"]["observed"] == pytest.approx(matrix[1, 1], abs=1e-6)
    if g == 3:
        assert cells["g3_cross_12"]["observed"] == pytest.approx(matrix[0, 1], abs=1e-6)
        assert cells["g3_cross_21"]["observed"] == pytest.approx(matrix[1, 0], abs=1e-6)


def test_assort_matrix_scores_all_four_cells_against_it(tmp_path) -> None:
    params = {**BASE, "seed": 22, "assort_matrix": [[0.3, 0.05], [0.05, 0.15]]}
    cells = _run(tmp_path, run_simulation(**params), params)["cells"]
    assert len(cells) == 4 * 5
    assert all(cell["status"] == "scored" for cell in cells.values())
    assert cells["g4_cross_21"]["expected"] == 0.05
    assert cells["g4_liability2"]["expected"] == 0.15


@pytest.mark.parametrize("overrides", [{}, {"mating_model": "wright_fisher", "assort1": 0.3}])
def test_no_assortment_leaves_the_section_not_applicable(tmp_path, overrides) -> None:
    params = {**BASE, "seed": 23, **overrides}
    section = _run(tmp_path, run_simulation(**params), params)
    assert section["cells"] == {}
    assert section["not_applicable"] == "no assortative mating in a recorded mating iteration"


def test_a_random_iteration_inside_an_assortative_scenario_is_scored_at_zero(tmp_path) -> None:
    params = {**BASE, "seed": 24, "assort1": {0: 0.0, 5: 0.3}}
    cells = _run(tmp_path, run_simulation(**params), params)["cells"]
    for label in ("liability1", "liability2"):
        assert cells[f"g1_{label}"]["status"] == "scored"
        assert cells[f"g1_{label}"]["expected"] == 0
    assert cells["g4_liability2"]["status"] == "not_applicable"
    assert "g1_cross_12" not in cells


def test_a_wrong_target_is_caught(tmp_path, mixed) -> None:
    cells = _run(tmp_path, mixed, {**MIXED, "assort1": 0.5, "assort2": {0: 0.0, 5: 0.5}})["cells"]
    assert cells["g0_liability1"]["z"] < -3
    assert cells["g3_liability2"]["z"] < -3


def test_the_metropolis_tolerance_drops_both_trait_cells_whose_se_it_rivals(tmp_path, mixed, monkeypatch) -> None:
    se = _run(tmp_path, mixed, MIXED)["cells"]["g3_liability1"]["se"]
    monkeypatch.setattr(mc, "METROPOLIS_TOL", se)
    cells = _run(tmp_path, mixed, MIXED)["cells"]
    assert cells["g3_liability1"]["status"] == "not_applicable"
    assert "Metropolis tolerance" in cells["g3_liability1"]["reason"]
    assert cells["g0_liability1"]["status"] == "scored"


def test_a_configured_zero_trait_is_not_applicable(tmp_path) -> None:
    params = {**BASE, "seed": 25, "A2": 0.0, "C2": 0.0, "E2": 0.0, "rA": 0.0, "assort1": {0: 0.0, 5: 0.3}}
    cells = _run(tmp_path, run_simulation(**params), params)["cells"]
    assert cells["g1_liability2"]["status"] == "not_applicable"
    assert cells["g1_liability2"]["reason"] == "A, C and E configured 0"
    assert cells["g1_liability1"]["status"] == "scored"


def test_a_constant_side_the_configuration_does_not_explain_is_information(tmp_path, mixed) -> None:
    flat = mixed.with_columns(pl.lit(0.0, dtype=mixed.schema["liability2"]).alias("liability2"))
    cell = _run(tmp_path, flat, MIXED)["cells"]["g3_liability2"]
    assert cell["status"] == "information"
    assert cell["reason"] == "one side's liability is constant"


def test_severed_parents_leave_information_not_a_crash(tmp_path, mixed) -> None:
    severed = mixed.with_columns(
        pl.when(pl.col("generation") == 4)
        .then(-1)
        .otherwise(pl.col("father"))
        .cast(mixed.schema["father"])
        .alias("father")
    )
    cell = _run(tmp_path, severed, MIXED)["cells"]["g3_liability1"]
    assert cell["status"] == "information"
    assert cell["n_pairs"] == 0


def test_small_strata_are_not_applicable_and_the_verdict_still_passes(tmp_path) -> None:
    sections = []
    for k in range(20):
        params = {**BASE, "seed": 300 + k, "N": 400, "G_ped": 3, "G_sim": 4, "assort1": 0.3}
        sections.append(_run(tmp_path, run_simulation(**params), params))
    cell = sections[0]["cells"]["g0_liability1"]
    assert cell["status"] == "not_applicable"
    assert cell["n_pairs"] < MIN_PAIRS
    assert all(s["summary"]["n_scored"] == 0 for s in sections)
    decided = decide(scenario_family({"mate_correlation": sections}, ON_DEMAND_CHECKS))
    assert decided["verdict"] == "pass"


def test_n_couples_inverts_the_expected_occupancy() -> None:
    for m, n_children in ((640, 1000), (2000, 1000), (50, 5000)):
        n = m * (1 - (1 - 1 / m) ** n_children)
        assert n_couples(round(n), n_children) == pytest.approx(m, rel=0.02)
    exact = 1000 * (1 - (1 - 1 / 1000) ** 600)
    m_hat = n_couples(int(exact), 600)
    assert m_hat * -math.expm1(600 * math.log1p(-1 / m_hat)) == pytest.approx(int(exact), rel=1e-9)
    assert n_couples(0, 100) == 0
    assert n_couples(100, 100) == math.inf
    assert n_couples(1, 100) == pytest.approx(1.0)


def test_later_generations_are_scored_without_a_column(tmp_path) -> None:
    params = {**BASE, "seed": 26, "N": 600, "G_ped": 14, "G_sim": 14, "assort1": 0.3}
    cells = _run(tmp_path, run_simulation(**params), params)["cells"]
    assert "g12_liability1" in cells
    assert "g12_liability1" not in CELL_NAMES
    assert np.isfinite(cells["g12_liability1"]["observed"])


def test_a_non_finite_liability_is_information(tmp_path, mixed) -> None:
    broken = mixed.with_columns(
        pl.when(pl.col("generation") == 3)
        .then(float("nan"))
        .otherwise(pl.col("liability1"))
        .cast(mixed.schema["liability1"])
        .alias("liability1")
    )
    cell = _run(tmp_path, broken, MIXED)["cells"]["g3_liability1"]
    assert cell["status"] == "information"
    assert cell["reason"] == "a liability is not finite"
