"""Scoring of the onset-τ cells (ADR 0027) and their ``validation.yaml`` shape (ADR 0028)."""

from __future__ import annotations

from statistics import NormalDist

import pytest
import yaml

from simace.analysis.gather import _on_demand_columns
from simace.analysis.validate.on_demand import SCHEMA, load_validation, one_warning_for_refused
from simace.analysis.validate.onset_tau import score_onset_tau


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


def test_scored_cells_carry_z_and_the_rep_records_its_largest():
    scored = score_onset_tau(_summary(FS=_cell(observed=0.074), MO=_cell(observed=0.066), MZ=_cell(n_eff=40.0)))
    cells, summary = scored["cells"], scored["summary"]
    assert cells["trait1_FS"]["status"] == "scored"
    assert cells["trait1_FS"]["z"] == pytest.approx(3.0)
    assert cells["trait1_MO"]["z"] == pytest.approx(-1.0)
    assert cells["trait1_MZ"]["status"] == "information"
    assert "effective components" in cells["trait1_MZ"]["reason"]
    assert summary["n_scored"] == 2
    assert summary["max_abs_z"] == pytest.approx(3.0)
    assert summary["z_bound"] == pytest.approx(NormalDist().inv_cdf(1 - 0.05 / 4))


def test_no_cell_or_summary_carries_a_per_rep_pass():
    """A calibrated test's result is one draw; the verdict is across reps (ADR 0028)."""
    scored = score_onset_tau(_summary(FS=_cell(observed=0.068 + 0.002 * 5.0), MZ=_cell(n_eff=40.0)))
    assert scored["summary"]["max_abs_z"] > scored["summary"]["z_bound"]
    for cell in (*scored["cells"].values(), scored["summary"]):
        assert not {"passed", "n_failed", "tolerance"} & set(cell)


def test_thirteen_cells_give_a_bound_of_two_point_eight_nine():
    scored = score_onset_tau(_summary(**{f"C{i}": _cell() for i in range(13)}))
    assert scored["summary"]["z_bound"] == pytest.approx(2.8905, abs=1e-4)


def test_nothing_scored_leaves_the_largest_z_and_bound_empty():
    scored = score_onset_tau(_summary(MZ=_cell(n_eff=40.0)))
    assert scored["summary"] == {
        "n_scored": 0,
        "max_abs_z": None,
        "z_bound": None,
        "skipped_traits": {},
        "entry_cap": 100_000,
    }


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
    cell = score_onset_tau(_summary(FS=_cell(**override)))["cells"]["trait1_FS"]
    assert cell["status"] == "information"
    assert reason in cell["reason"]
    assert "z" not in cell


def test_skipped_traits_are_reported():
    summary = _summary(FS=_cell())
    summary["skipped_traits"] = {"trait2": "model adult is not frailty"}
    assert score_onset_tau(summary)["summary"]["skipped_traits"] == {"trait2": "model adult is not frailty"}


def _write(path, content):
    path.write_text(yaml.safe_dump(content))
    return path


def test_gather_columns_are_present_and_empty_without_a_validation_file(tmp_path):
    empty = _on_demand_columns(tmp_path / "validation.yaml")
    assert "validation_passed" not in empty
    assert "onset_tau_n_failed" not in empty
    assert empty["onset_tau_max_abs_z"] is None
    assert empty["onset_tau_z_trait1_FS"] is None
    assert len([k for k in empty if k.startswith("onset_tau_z_")]) == 13


def test_gather_columns_take_z_from_scored_cells_only(tmp_path):
    scored = score_onset_tau(_summary(FS=_cell(observed=0.074, expected=0.068, se=0.002), MZ=_cell(n_eff=10.0)))
    path = _write(tmp_path / "validation.yaml", {"schema": dict(SCHEMA), "checks": {"onset_tau": scored}})
    row = _on_demand_columns(path)
    assert row["onset_tau_z_trait1_FS"] == pytest.approx(3.0)
    assert row["onset_tau_z_trait1_MZ"] is None
    assert row["onset_tau_n_scored"] == 1
    assert row["onset_tau_max_abs_z"] == pytest.approx(3.0)
    assert row["onset_tau_rep_bound"] == pytest.approx(NormalDist().inv_cdf(1 - 0.025))


def test_a_pre_schema_validation_yaml_is_refused_by_name(tmp_path, caplog):
    """A file from before ADR 0028 has a top-level ``summary.passed`` and no schema: no reader takes it."""
    old = {
        "summary": {"passed": False, "n_failed": 3},
        "checks": {"onset_tau": {"summary": {"passed": False, "n_scored": 13, "n_failed": 3}}},
    }
    path = _write(tmp_path / "validation.yaml", old)
    with caplog.at_level("WARNING"):
        row = _on_demand_columns(path)
    assert all(value is None for value in row.values())
    assert str(path) in caplog.text
    assert "schema" in caplog.text


@pytest.mark.parametrize("schema", [None, {"kind": "on_demand_checks", "version": 2}, {"kind": "other", "version": 1}])
def test_only_the_current_schema_is_read(tmp_path, schema):
    path = _write(tmp_path / "validation.yaml", {"schema": schema, "checks": {}})
    assert load_validation(path) is None
    _write(path, {"schema": dict(SCHEMA), "checks": {}})
    assert load_validation(path) == {"schema": SCHEMA, "checks": {}}


def test_the_onset_tau_page_draws_the_new_section(tmp_path):
    from simace.plotting.plot_onset_tau import plot_onset_tau

    section = score_onset_tau(_summary(FS=_cell(observed=0.068 + 0.002 * 5.0), MO=_cell(), MZ=_cell(n_eff=40.0)))
    out = tmp_path / "onset_tau.png"
    plot_onset_tau([{"onset_tau": section}, {}], out)
    assert out.stat().st_size > 0


def test_a_reader_run_logs_one_warning_for_every_refused_file(tmp_path, caplog):
    paths = [_write(tmp_path / f"rep{i}.yaml", {"summary": {"passed": True}}) for i in range(3)]
    with caplog.at_level("WARNING"), one_warning_for_refused():
        assert [load_validation(path) for path in paths] == [None, None, None]
    assert len(caplog.records) == 1
    assert "Ignoring 3 validation.yaml file(s)" in caplog.text
    assert str(paths[0]) in caplog.text
