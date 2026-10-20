"""The shared pieces of every calibrated test (ADR 0028, 0029): rep inputs, cells, the rep summary, gather columns."""

from __future__ import annotations

from statistics import NormalDist

import pytest
import yaml

from simace.analysis.gather import _on_demand_columns
from simace.analysis.validate import on_demand
from simace.analysis.validate.calibrated import (
    CalibratedTest,
    RepInputs,
    information_cell,
    not_applicable_cell,
    scored_cell,
    summarize,
)
from simace.analysis.validate.verdict import ZCalibration


def test_params_come_from_the_reports_inputs(tmp_path) -> None:
    report = tmp_path / "report.yaml"
    report.write_text(yaml.safe_dump({"inputs": {"parameters": {"A1": 0.5, "C1": 0.2}}}))
    inputs = RepInputs(tmp_path / "pedigree.parquet", report, seed=1, phenotype={})
    assert inputs.params == {"A1": 0.5, "C1": 0.2}


def test_summary_bounds_only_the_scored_cells() -> None:
    cells = {
        "a": scored_cell(0.3, 0.0, 0.1),
        "b": scored_cell(-0.1, 0.0, 0.1),
        "c": information_cell("few pairs"),
        "d": not_applicable_cell("zero variance"),
    }
    summary = summarize(cells)
    assert summary["n_scored"] == 2
    assert summary["max_abs_z"] == pytest.approx(3.0)
    assert summary["z_bound"] == pytest.approx(NormalDist().inv_cdf(1 - 0.05 / 4))
    assert summarize({"c": information_cell("few pairs")}) == {"n_scored": 0, "max_abs_z": None, "z_bound": None}


def test_gather_gives_every_registered_test_its_columns(tmp_path, monkeypatch) -> None:
    extra = CalibratedTest("extra", "a test", lambda _: {}, ZCalibration(rep_rate=0.05), 20, ("trait1_FS", "trait2_FS"))
    monkeypatch.setitem(on_demand.ON_DEMAND_CHECKS, "extra", extra)
    section = {
        "cells": {"trait1_FS": scored_cell(0.2, 0.0, 0.1)},
        "summary": summarize({"x": scored_cell(0.2, 0.0, 0.1)}),
    }
    path = tmp_path / "validation.yaml"
    path.write_text(yaml.safe_dump({"schema": dict(on_demand.SCHEMA), "checks": {"extra": section}}))
    row = _on_demand_columns(path)
    assert row["extra_z_trait1_FS"] == pytest.approx(2.0)
    assert row["extra_z_trait2_FS"] is None
    assert row["extra_n_scored"] == 1
    assert row["onset_tau_n_scored"] is None
    assert set(_on_demand_columns(tmp_path / "missing.yaml")) == set(row)
