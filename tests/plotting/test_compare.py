"""Tests for the cross-scenario comparison figures and their inputs."""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest
import yaml

from simace.analysis.stats.pedigree_estimates import POOLED_RELATIONSHIP_CLASSES
from simace.plotting.compare import PrevalenceSeries, ScenarioReps, am, group_reps
from simace.plotting.compare._draw import bar_offsets, rep_envelope
from simace.plotting.compare.increasing_e import compare_prevalence_drift
from tests.conftest import SIM_DEFAULTS, sim_params


def _reference_lines(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, params: dict) -> dict[str, float]:
    """The dashed reference value per pooled class that compare_correlations_by_relclass draws."""
    figures = []
    monkeypatch.setattr(am, "save", lambda fig, *a, **k: figures.append(fig))
    names = [name for name, _, _ in POOLED_RELATIONSHIP_CLASSES]
    monkeypatch.setattr(am, "load_pooled_liability_correlations", lambda *a, **k: {name: [0.3] for name in names})
    am.compare_correlations_by_relclass(
        [ScenarioReps("s", (tmp_path,))], tmp_path / "p.png", reference=sim_params(params)
    )
    lines = {}
    dashed = [c for c in figures[0].axes[0].collections if c.get_linestyle()[0][1] is not None]
    for collection in dashed:
        for (x0, y), (x1, _) in collection.get_segments():
            lines[names[round((x0 + x1) / 2)]] = float(y)
    plt.close(figures[0])
    return lines


_REF = {"A1": 0.5, "C1": 0.2, "E1": 0.5, "assort1": 0.0, "assort2": 0.0}


def test_relclass_reference_lines_divide_by_vp(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Without C transmission the HS members disagree (MHS shares C, PHS does not), so HS has no line."""
    lines = _reference_lines(monkeypatch, tmp_path, _REF)
    assert lines == pytest.approx({"FS": (0.25 + 0.2) / 1.2, "PO": 0.25 / 1.2, "1C": 0.0625 / 1.2})


def test_relclass_reference_lines_follow_c_transmission(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """b_m = b_f = 0.5: MO and FO agree on (0.5A + 0.5C)/V_P; 1C has no closed form and no truth here."""
    params = {**_REF, "Ctm_maternal1": 0.5, "Ctm_paternal1": 0.5}
    lines = _reference_lines(monkeypatch, tmp_path, params)
    assert lines == pytest.approx({"FS": (0.25 + 0.2) / 1.2, "PO": (0.25 + 0.1) / 1.2})


def test_parameters_read_the_first_reps_report(tmp_path: Path):
    params = {**SIM_DEFAULTS, "A1": 0.5}
    for rep in ("rep1", "rep2"):
        (tmp_path / rep).mkdir()
    (tmp_path / "rep1" / "report.yaml").write_text(yaml.safe_dump({"inputs": {"parameters": params}}))
    scenario = ScenarioReps("s", (tmp_path / "rep1", tmp_path / "rep2"))
    assert scenario.parameters() == sim_params(params)
    assert scenario.pedigrees == [tmp_path / "rep1" / "pedigree.parquet", tmp_path / "rep2" / "pedigree.parquet"]


def test_scenario_needs_a_replicate():
    with pytest.raises(ValueError, match="no replicate"):
        ScenarioReps("s", ())


def test_group_reps_splits_scenario_major():
    scenarios = group_reps(["a", "b"], ["a1", "a2", "b1"], [2, 1])
    assert scenarios == [ScenarioReps("a", (Path("a1"), Path("a2"))), ScenarioReps("b", (Path("b1"),))]


@pytest.mark.parametrize(
    ("labels", "dirs", "counts", "message"),
    [
        (["a", "b"], ["a1", "b1"], [2], "2 labels but 1 rep counts"),
        (["a", "b"], ["a1", "b1"], [3, -1], "at least 1"),
        (["a", "b"], ["a1", "b1"], [1, 0], "at least 1"),
        (["a", "b"], ["a1", "b1", "b2"], [1, 1], "add up to 2 but 3"),
    ],
)
def test_group_reps_rejects_a_grouping_that_does_not_fit(labels, dirs, counts, message):
    with pytest.raises(ValueError, match=message):
        group_reps(labels, dirs, counts)


def test_prevalence_drift_rejects_series_over_different_scenarios(tmp_path: Path):
    series = [
        PrevalenceSeries("std", (ScenarioReps("flat", (tmp_path,)), ScenarioReps("rise", (tmp_path,)))),
        PrevalenceSeries("nostd", (ScenarioReps("rise", (tmp_path,)), ScenarioReps("flat", (tmp_path,)))),
    ]
    with pytest.raises(ValueError, match="'nostd' runs"):
        compare_prevalence_drift(series, tmp_path / "p.png")


def test_rep_envelope_keeps_finite_values_only():
    env = rep_envelope([0.2, float("nan"), 0.4, float("inf")])
    assert (env.mean, env.low, env.high) == pytest.approx((0.3, 0.2, 0.4))


def test_rep_envelope_of_no_finite_value_is_nan():
    env = rep_envelope([float("nan")])
    assert all(math.isnan(v) for v in (env.mean, env.low, env.high))


def test_bar_offsets_center_each_group_on_x():
    offsets = [bar_offsets(np.array([0.0, 1.0]), 0.8, 2, i) for i in range(2)]
    assert np.allclose(offsets, [[-0.2, 0.8], [0.2, 1.2]])
