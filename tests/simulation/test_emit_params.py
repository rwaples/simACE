"""Tests for ``simace.simulation.emit_params``."""

from __future__ import annotations

import pytest

from simace.simulation.emit_params import emit_params


@pytest.fixture
def baseline_kwargs() -> dict:
    return {
        "seed": 42,
        "rep": 1,
        "A1": 0.5,
        "C1": 0.2,
        "E1": 0.3,
        "A2": 0.4,
        "C2": 0.1,
        "E2": 0.5,
        "rA": 0.5,
        "rC": 0.3,
        "rE": 0.0,
        "N": 1000,
        "G_ped": 3,
        "G_sim": 5,
        "mating_lambda": 0.5,
        "p_mztwin": 0.02,
        "assort1": 0.0,
        "assort2": 0.0,
    }


class TestEmitParamsShape:
    def test_returns_dict_with_expected_keys(self, baseline_kwargs):
        out = emit_params(**baseline_kwargs)
        expected = {
            "seed", "rep",
            "A1", "C1", "E1", "A2", "C2", "E2",
            "rA", "rC", "rE",
            "N", "G_ped", "G_sim",
            "mating_lambda", "p_mztwin",
            "assort1", "assort2",
        }
        assert set(out.keys()) == expected

    def test_assort_matrix_omitted_when_none(self, baseline_kwargs):
        out = emit_params(**baseline_kwargs, assort_matrix=None)
        assert "assort_matrix" not in out

    def test_assort_matrix_included_when_set(self, baseline_kwargs):
        matrix = [[0.5, 0.1], [0.1, 0.4]]
        out = emit_params(**baseline_kwargs, assort_matrix=matrix)
        assert out["assort_matrix"] == matrix


class TestEchoSemantics:
    def test_g_sim_none_echoed(self, baseline_kwargs):
        kwargs = {**baseline_kwargs, "G_sim": None}
        out = emit_params(**kwargs)
        assert out["G_sim"] is None

    def test_values_round_trip_unchanged(self, baseline_kwargs):
        out = emit_params(**baseline_kwargs)
        for k, v in baseline_kwargs.items():
            assert out[k] == v
