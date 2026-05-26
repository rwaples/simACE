"""CLI tests for ``simace.simulation.emit_params``."""

import sys

import pytest
import yaml

from simace.simulation.emit_params import cli as emit_params_cli

MINIMAL_CFG = {
    "seed": 100,
    "A1": 0.5,
    "C1": 0.2,
    "E1": 0.3,
    "A2": 0.5,
    "C2": 0.2,
    "E2": 0.3,
    "rA": 0.3,
    "rC": 0.5,
    "N": 1000,
    "G_ped": 3,
    "G_sim": 5,
    "mating_model": "standard",
    "mating_lambda": 0.5,
    "p_mztwin": 0.02,
    "assort1": 0.1,
    "assort2": 0.0,
}


def _run_cli(monkeypatch, argv):
    monkeypatch.setattr(sys, "argv", ["emit-params", *argv])
    emit_params_cli()


def _write_cfg(tmp_path, overrides=None):
    cfg = {**MINIMAL_CFG, **(overrides or {})}
    cfg_path = tmp_path / "scenario.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    return cfg_path


def test_cli_round_trip(tmp_path, monkeypatch):
    cfg_path = _write_cfg(tmp_path)
    out_path = tmp_path / "params.yaml"

    _run_cli(monkeypatch, ["--config", str(cfg_path), "--rep", "1", "--output", str(out_path)])

    assert out_path.exists()
    params = yaml.safe_load(out_path.read_text())

    required = {
        "seed",
        "rep",
        "A1",
        "C1",
        "E1",
        "A2",
        "C2",
        "E2",
        "rA",
        "rC",
        "rE",
        "N",
        "G_ped",
        "G_sim",
        "mating_model",
        "mating_lambda",
        "p_mztwin",
        "assort1",
        "assort2",
    }
    assert required.issubset(params.keys())
    assert params["rep"] == 1
    assert params["seed"] == 100  # seed + rep - 1 = 100 + 1 - 1
    assert params["N"] == MINIMAL_CFG["N"]
    assert params["mating_model"] == "standard"
    assert params["assort1"] == pytest.approx(0.1)
    # assort_matrix omitted when absent from config
    assert "assort_matrix" not in params


def test_cli_rep_offsets_seed(tmp_path, monkeypatch):
    cfg_path = _write_cfg(tmp_path)
    out_path = tmp_path / "params_rep3.yaml"

    _run_cli(monkeypatch, ["--config", str(cfg_path), "--rep", "3", "--output", str(out_path)])

    params = yaml.safe_load(out_path.read_text())
    assert params["rep"] == 3
    assert params["seed"] == 102  # 100 + 3 - 1


def test_cli_wright_fisher_echoes_inherited_knobs(tmp_path, monkeypatch):
    """WF ignores mating_lambda/p_mztwin/assort1 at runtime, but params.yaml
    must record the values the user actually set — no silent rewriting."""
    cfg_path = _write_cfg(
        tmp_path,
        overrides={
            "mating_model": "wright_fisher",
            "mating_lambda": 999.0,
            "p_mztwin": 0.5,
            "assort1": 0.7,
        },
    )
    out_path = tmp_path / "params_wf.yaml"

    _run_cli(monkeypatch, ["--config", str(cfg_path), "--rep", "1", "--output", str(out_path)])

    params = yaml.safe_load(out_path.read_text())
    assert params["mating_model"] == "wright_fisher"
    assert params["mating_lambda"] == 999.0
    assert params["p_mztwin"] == 0.5
    assert params["assort1"] == 0.7


def test_cli_assort_matrix_included_when_present(tmp_path, monkeypatch):
    cfg_path = _write_cfg(
        tmp_path,
        overrides={"assort_matrix": [[0.2, 0.05], [0.05, 0.1]]},
    )
    out_path = tmp_path / "params_am.yaml"

    _run_cli(monkeypatch, ["--config", str(cfg_path), "--rep", "1", "--output", str(out_path)])

    params = yaml.safe_load(out_path.read_text())
    assert params["assort_matrix"] == [[0.2, 0.05], [0.05, 0.1]]
