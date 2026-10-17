"""Scoring of the onset-τ cells (ADR 0027)."""

from __future__ import annotations

from statistics import NormalDist

import pytest
import yaml

from simace.analysis.gather import _on_demand_columns
from simace.analysis.validate.onset_tau import validate_onset_tau


def _cell(observed=0.07, expected=0.068, se=0.002, **overrides):
    cell = {
        "n_pairs": 50_000,
        "sampled_fraction": 1.0,
        "n_entries": 100_000,
        "n_entries_population": 100_000,
        "n_people": 60_000,
        "n_components": 20_000,
        "n_eff": 15_000.0,
        "max_component_share": 1e-4,
        "max_component_people": 9,
        "shared_workload": 2_000_000,
        "grid": 256,
        "numerical_bound": 1e-6,
        "tie_share": 1e-7,
        "observed": observed,
        "expected": expected,
        "se": se,
        "r_eta": 0.25,
        "s1": 1.0,
        "s2": 1.0,
        "tau_oracle": 0.067,
    }
    cell.update(overrides)
    return cell


def _summary(**cells):
    return {"cells": {"trait1": cells}, "skipped_traits": {}, "entry_cap": 100_000}


def test_bonferroni_threshold_uses_the_scored_count():
    results = validate_onset_tau(_summary(FS=_cell(), MO=_cell(), MZ=_cell(n_eff=40.0)))
    z = NormalDist().inv_cdf(1 - 0.05 / 4)
    for name in ("trait1_FS", "trait1_MO"):
        assert results[name]["status"] == "scored"
        assert results[name]["z_crit"] == pytest.approx(z)
        assert results[name]["n_scored"] == 2
        assert results[name]["passed"]
    assert results["trait1_MZ"]["status"] == "information"
    assert "effective components" in results["trait1_MZ"]["reason"]


def test_thirteen_cells_give_z_of_two_point_eight_nine():
    cells = {f"C{i}": _cell() for i in range(13)}
    results = validate_onset_tau(_summary(**cells))
    assert results["trait1_C0"]["z_crit"] == pytest.approx(2.8905, abs=1e-4)


def test_a_shifted_cell_fails():
    results = validate_onset_tau(_summary(FS=_cell(observed=0.068 + 0.002 * 3.0)))
    assert results["trait1_FS"]["passed"] is False


@pytest.mark.parametrize(
    ("override", "reason"),
    [
        ({"expected": None, "se": None, "note": "shared-person workload 6e8 > 5e+08"}, "workload"),
        ({"tie_share": 2e-4}, "ties plus numerical bound"),
        ({"numerical_bound": 2e-4}, "ties plus numerical bound"),
        ({"se": 0.0}, "SE is zero"),
    ],
)
def test_unscorable_cells_are_information_with_a_reason(override, reason):
    results = validate_onset_tau(_summary(FS=_cell(**override)))
    assert results["trait1_FS"]["status"] == "information"
    assert reason in results["trait1_FS"]["reason"]
    assert "passed" not in results["trait1_FS"]


def test_skipped_traits_are_reported():
    summary = _summary(FS=_cell())
    summary["skipped_traits"] = {"trait2": "model adult is not frailty"}
    results = validate_onset_tau(summary)
    assert results["trait2_skipped"]["reason"] == "model adult is not frailty"


def test_gather_columns_are_present_and_empty_without_a_validation_file(tmp_path):
    empty = _on_demand_columns(tmp_path / "validation.yaml")
    assert empty["validation_passed"] is None
    assert empty["onset_tau_z_trait1_FS"] is None
    assert len([k for k in empty if k.startswith("onset_tau_z_")]) == 13


def test_gather_columns_take_z_from_scored_cells_only(tmp_path):
    checks = validate_onset_tau(_summary(FS=_cell(observed=0.074, expected=0.068, se=0.002), MZ=_cell(n_eff=10.0)))
    content = {
        "summary": {"passed": True, "n_failed": 0},
        "checks": {"onset_tau": {"checks": checks, "summary": {"passed": True, "n_scored": 1, "n_failed": 0}}},
    }
    path = tmp_path / "validation.yaml"
    path.write_text(yaml.safe_dump(content))
    row = _on_demand_columns(path)
    assert row["onset_tau_z_trait1_FS"] == pytest.approx(3.0)
    assert row["onset_tau_z_trait1_MZ"] is None
    assert row["onset_tau_n_scored"] == 1
