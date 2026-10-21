"""Recorded generation 0's calibrated tests (ADR 0030): total variance and cross-trait rC."""

from __future__ import annotations

import itertools
from typing import TYPE_CHECKING

import numpy as np
import polars as pl
import pytest
import yaml

from simace.analysis.validate.calibrated import RepInputs
from simace.analysis.validate.generation0 import (
    CROSS_TRAIT_RC_CELLS,
    MIN_HOUSEHOLDS,
    TOTAL_VARIANCE_CELLS,
    run_cross_trait_rc,
    run_total_variance,
)
from simace.analysis.validate.on_demand import ON_DEMAND_CHECKS
from simace.core.cohort import write_pedigree
from simace.simulation.simulate import run_simulation

if TYPE_CHECKING:
    from pathlib import Path

PARAMS = dict(
    seed=11,
    N=5000,
    G_ped=3,
    G_sim=5,
    mating_lambda=0.5,
    p_mztwin=0.02,
    A1=0.5,
    C1=0.3,
    E1=0.2,
    A2=0.4,
    C2=0.3,
    E2=0.3,
    rA=0.3,
    rC=0.5,
)


@pytest.fixture(scope="module")
def pedigree() -> pl.DataFrame:
    return run_simulation(**PARAMS)


_written = itertools.count()


def _inputs(tmp_path: Path, pedigree: pl.DataFrame, **overrides) -> RepInputs:
    """A rep's inputs: the pedigree and a report.yaml whose parameters are PARAMS with the overrides."""
    k = next(_written)
    pedigree_path, report_path = tmp_path / f"pedigree{k}.parquet", tmp_path / f"report{k}.yaml"
    write_pedigree(pedigree, pedigree_path)
    report_path.write_text(yaml.safe_dump({"inputs": {"parameters": {**PARAMS, **overrides}}}))
    return RepInputs(pedigree_path, report_path, seed=11, phenotype={})


def _both(inputs: RepInputs) -> tuple[dict, dict]:
    return run_total_variance(inputs), run_cross_trait_rc(inputs)


def test_both_tests_are_registered_with_their_cells() -> None:
    assert ON_DEMAND_CHECKS["total_variance"].cell_names == TOTAL_VARIANCE_CELLS == ("trait1_total", "trait2_total")
    assert ON_DEMAND_CHECKS["cross_trait_rC"].cell_names == CROSS_TRAIT_RC_CELLS == ("rC",)
    assert not ON_DEMAND_CHECKS["total_variance"].own_page


def test_a_correct_simulation_scores_near_zero(tmp_path, pedigree) -> None:
    total, rc = _both(_inputs(tmp_path, pedigree))
    assert set(total["cells"]) == set(TOTAL_VARIANCE_CELLS)
    for cell in (*total["cells"].values(), rc["cells"]["rC"]):
        assert cell["status"] == "scored"
        assert abs(cell["z"]) < 4
        assert cell["n_households"] >= MIN_HOUSEHOLDS
    assert total["cells"]["trait1_total"]["expected"] == pytest.approx(1.0)
    assert rc["cells"]["rC"]["expected"] == 0.5
    assert "not_applicable" not in total
    assert "not_applicable" not in rc
    assert total["summary"]["n_scored"] == 2


def test_total_variance_is_scored_on_the_relative_scale(tmp_path, pedigree) -> None:
    """z = (m₂/V₀ − 1) / (SE_obs/m₂): the cell's SE is the clustered SE times V₀/m₂ (ADR 0030 §1)."""
    cell = run_total_variance(_inputs(tmp_path, pedigree))["cells"]["trait2_total"]
    m2, v0 = cell["observed"], cell["expected"]
    assert cell["se"] == pytest.approx(cell["se_observed"] * v0 / m2)
    assert cell["z"] == pytest.approx((m2 / v0 - 1) / (cell["se_observed"] / m2))
    gen0 = pedigree.filter(pl.col("generation") == 0)
    x = sum(gen0[f"{k}2"].to_numpy().astype(np.float64) for k in "ACE")
    assert m2 == pytest.approx(float(x.var()))


@pytest.mark.parametrize(
    "assortment",
    [
        {"assort1": 0.3, "assort2": 0.3},
        {"assort1": 0.3},  # trait 1 only: rA carries it into trait 2
        {"assort2": 0.3},
        {"assort_matrix": [[0.3, 0.0], [0.0, 0.0]]},
        {"assort1": {0: 0.0, 2: 0.3}},  # starts at burn-in's iteration, which mates generation 0's parents
    ],
)
def test_assortment_up_to_burn_in_leaves_total_variance_only(tmp_path, pedigree, assortment) -> None:
    total, rc = _both(_inputs(tmp_path, pedigree, **assortment))
    assert total["cells"] == {}
    assert "assortative mating" in total["not_applicable"]
    assert "not_applicable" not in rc
    assert rc["cells"]["rC"]["status"] == "scored"


def test_rc_stays_calibrated_under_assortment(tmp_path) -> None:
    """Without transmission each household's C is a fresh draw, so assortment leaves rC alone (ADR 0030 §4)."""
    params = {**PARAMS, "assort1": 0.3, "assort2": 0.3, "seed": 13}
    ped = run_simulation(**params)
    cell = run_cross_trait_rc(_inputs(tmp_path, ped, assort1=0.3, assort2=0.3, seed=13))["cells"]["rC"]
    assert cell["status"] == "scored"
    assert abs(cell["z"]) < 4


def test_assortment_after_burn_in_stays_applicable(tmp_path, pedigree) -> None:
    total, rc = _both(_inputs(tmp_path, pedigree, assort1={0: 0.0, 3: 0.3}))
    assert "not_applicable" not in total
    assert "not_applicable" not in rc
    assert rc["cells"]["rC"]["status"] == "scored"


def test_wright_fisher_ignores_its_assortment(tmp_path, pedigree) -> None:
    total, rc = _both(_inputs(tmp_path, pedigree, mating_model="wright_fisher", assort1=0.3))
    assert "not_applicable" not in total
    assert "not_applicable" not in rc


@pytest.mark.parametrize("coefficient", ["Ctm_maternal1", "Ctm_paternal2"])
def test_c_transmission_leaves_both_tests(tmp_path, pedigree, coefficient) -> None:
    for section in _both(_inputs(tmp_path, pedigree, **{coefficient: 0.3})):
        assert section["cells"] == {}
        assert "C transmission" in section["not_applicable"]


@pytest.mark.parametrize(("overrides", "reason"), [({"C1": 0.0}, "configured 0"), ({"rC": -1.0}, "|rC| = 1")])
def test_rc_does_not_apply_without_two_c_components_or_at_rc_one(tmp_path, pedigree, overrides, reason) -> None:
    total, rc = _both(_inputs(tmp_path, pedigree, **overrides))
    assert reason in rc["not_applicable"]
    assert rc["cells"] == {}
    assert "not_applicable" not in total


def test_fewer_than_min_households_is_information(tmp_path, pedigree) -> None:
    households = pedigree.filter(pl.col("generation") == 0)["household_id"].unique().sort()[: MIN_HOUSEHOLDS - 1]
    small = pedigree.filter((pl.col("generation") != 0) | pl.col("household_id").is_in(households.implode()))
    total, rc = _both(_inputs(tmp_path, small))
    for cell in (*total["cells"].values(), rc["cells"]["rC"]):
        assert cell["status"] == "information"
        assert f"{MIN_HOUSEHOLDS - 1} households" in cell["reason"]


def test_a_wrong_total_is_caught(tmp_path, pedigree) -> None:
    cell = run_total_variance(_inputs(tmp_path, pedigree, A1=0.7))["cells"]["trait1_total"]
    assert cell["expected"] == pytest.approx(1.2)
    assert cell["z"] < -3


def test_a_wrong_rc_is_caught(tmp_path, pedigree) -> None:
    assert run_cross_trait_rc(_inputs(tmp_path, pedigree, rC=0.0))["cells"]["rC"]["z"] > 3


def test_a_broken_household_share_moves_rc(tmp_path, pedigree) -> None:
    """Permuting C2 over generation 0's householdmates after each household's first row breaks the share."""
    gen0 = pedigree["generation"].to_numpy() == 0
    household = pedigree["household_id"].to_numpy()
    rows = np.flatnonzero(gen0)
    _, first = np.unique(household[rows], return_index=True)
    rest = np.setdiff1d(rows, rows[first])
    c2 = pedigree["C2"].to_numpy().copy()
    c2[rest] = c2[np.random.default_rng(0).permutation(rest)]
    broken = pedigree.with_columns(pl.Series("C2", c2, dtype=pedigree.schema["C2"]))
    assert run_cross_trait_rc(_inputs(tmp_path, broken))["cells"]["rC"]["z"] < -3


def test_per_generation_e_resolves_at_burn_in(tmp_path) -> None:
    params = {**PARAMS, "E1": {0: 0.2, 2: 0.6}, "seed": 12}
    ped = run_simulation(**params)
    cell = run_total_variance(_inputs(tmp_path, ped, E1=params["E1"], seed=12))["cells"]["trait1_total"]
    assert cell["expected"] == pytest.approx(0.5 + 0.3 + 0.6)
    assert abs(cell["z"]) < 4


def test_an_observed_zero_variance_is_information_not_not_applicable(tmp_path, pedigree) -> None:
    gen0 = pl.col("generation") == 0
    flat = pedigree.with_columns(
        pl.when(gen0).then(0.0).otherwise(pl.col(c)).cast(pedigree.schema[c]).alias(c) for c in ("A1", "C1", "E1")
    )
    cell = run_total_variance(_inputs(tmp_path, flat))["cells"]["trait1_total"]
    assert cell["status"] == "information"
    assert cell["reason"] == "observed variance is zero"


def test_a_configured_zero_total_is_not_applicable(tmp_path, pedigree) -> None:
    cell = run_total_variance(_inputs(tmp_path, pedigree, A2=0.0, C2=0.0, E2=0.0))["cells"]["trait2_total"]
    assert cell["status"] == "not_applicable"
