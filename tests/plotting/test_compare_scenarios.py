"""Tests for the cross-scenario comparison figures."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pytest
import yaml

from simace.analysis.stats.pedigree_estimates import POOLED_RELATIONSHIP_CLASSES
from simace.plotting import compare_scenarios
from tests.conftest import SIM_DEFAULTS, sim_params

if TYPE_CHECKING:
    from pathlib import Path


def _reference_lines(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, params: dict) -> dict[str, float]:
    """The dashed reference value per pooled class that compare_correlations_by_relclass draws."""
    figures = []
    monkeypatch.setattr(compare_scenarios, "_save", lambda fig, *a, **k: figures.append(fig))
    names = [name for name, _, _ in POOLED_RELATIONSHIP_CLASSES]
    monkeypatch.setattr(
        compare_scenarios, "load_pooled_liability_correlations", lambda *a, **k: {name: [0.3] for name in names}
    )
    compare_scenarios.compare_correlations_by_relclass(
        [[tmp_path]], ["s"], tmp_path / "p.png", reference=sim_params(params)
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


def test_scenario_parameters_reads_the_report_beside_the_pedigree(tmp_path: Path):
    params = {**SIM_DEFAULTS, "A1": 0.5}
    (tmp_path / "report.yaml").write_text(yaml.safe_dump({"inputs": {"parameters": params}}))
    assert compare_scenarios.scenario_parameters(tmp_path / "pedigree.parquet") == sim_params(params)


def test_rep_envelope_keeps_finite_values_only():
    env = compare_scenarios.rep_envelope([0.2, float("nan"), 0.4, float("inf")])
    assert (env.mean, env.low, env.high) == pytest.approx((0.3, 0.2, 0.4))


def test_rep_envelope_of_no_finite_value_is_nan():
    env = compare_scenarios.rep_envelope([float("nan")])
    assert all(math.isnan(v) for v in (env.mean, env.low, env.high))


def test_bar_offsets_center_each_group_on_x():
    offsets = [compare_scenarios.bar_offsets(np.array([0.0, 1.0]), 0.8, 2, i) for i in range(2)]
    assert np.allclose(offsets, [[-0.2, 0.8], [0.2, 1.2]])
