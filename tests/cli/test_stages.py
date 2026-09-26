"""Every scenario's parameters reach the domain functions unchanged through stage argv.

Each stage CLI runs on the argv ``simace run`` builds, with its domain function
replaced by a recorder. The recorded keyword arguments must equal what the
config holds, which is what the Snakemake wrappers passed.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import simace.analysis.analyze as analyze_mod
import simace.ascertainment.runner as ascertain_mod
import simace.censoring.censor as censor_mod
import simace.phenotype.runner as phenotype_mod
import simace.simulation.simulate as simulate_mod
from simace.cli.layout import Layout
from simace.cli.run import ScenarioError, check_runnable, resolve_all
from simace.cli.stages import REP_PARAM_KEYS, STAGES, ResolvedRep

REPO_CONFIG = Path(__file__).resolve().parents[2] / "config"


def _runnable() -> list[tuple[str, dict]]:
    out = []
    for name, params in sorted(resolve_all(REPO_CONFIG).items()):
        try:
            check_runnable(name, params)
        except ScenarioError:
            continue
        out.append((name, params))
    return out


SCENARIOS = _runnable()


class _Recorder:
    def __init__(self, result=None) -> None:
        self.kwargs: dict = {}
        self.result = result

    def __call__(self, *args, **kwargs):
        self.kwargs = kwargs
        return self.result


def _touch(_obj, path) -> None:
    Path(path).write_text("")


def _argv(stage_name: str, rep: ResolvedRep, layout: Layout) -> list[str]:
    stage = next(s for s in STAGES if s.name == stage_name)
    return stage.argv(rep, layout)


@pytest.fixture
def rep_and_layout(request, tmp_path):
    name, params = request.param
    return ResolvedRep(params["folder"], name, 2, params), Layout(root=tmp_path / "r", logs=tmp_path / "l")


def _subset(recorded: dict, keys: list[str]) -> dict:
    return {key: recorded[key] for key in keys}


@pytest.mark.parametrize("rep_and_layout", SCENARIOS, ids=[n for n, _ in SCENARIOS], indirect=True)
def test_simulate_argv_round_trips(monkeypatch, rep_and_layout) -> None:
    rep, layout = rep_and_layout
    recorder = _Recorder()
    monkeypatch.setattr(simulate_mod, "run_simulation", recorder)
    monkeypatch.setattr(simulate_mod, "save_parquet", _touch)
    simulate_mod.cli(_argv("simulate", rep, layout))

    keys = ["N", "G_ped", "G_sim", "mating_model", "mating_lambda", "p_mztwin"]
    keys += ["A1", "C1", "E1", "A2", "C2", "E2", "rA", "rC", "rE", "assort1", "assort2", "assort_matrix"]
    assert recorder.kwargs["seed"] == rep.seed
    assert _subset(recorder.kwargs, keys) == {key: rep.params[key] for key in keys}


@pytest.mark.parametrize("rep_and_layout", SCENARIOS, ids=[n for n, _ in SCENARIOS], indirect=True)
def test_phenotype_argv_round_trips(monkeypatch, rep_and_layout) -> None:
    rep, layout = rep_and_layout
    recorder = _Recorder()
    monkeypatch.setattr(phenotype_mod, "load_parquet", lambda _path: None)
    monkeypatch.setattr(phenotype_mod, "run_phenotype", recorder)
    monkeypatch.setattr(phenotype_mod, "save_parquet", _touch)
    phenotype_mod.cli(_argv("phenotype", rep, layout))

    keys = ["G_pheno", "standardize"]
    keys += [f"{k}{t}" for t in (1, 2) for k in ("phenotype_model", "phenotype_params", "beta", "beta_sex")]
    assert recorder.kwargs["seed"] == rep.seed
    assert _subset(recorder.kwargs, keys) == {key: rep.params[key] for key in keys}


@pytest.mark.parametrize("rep_and_layout", SCENARIOS, ids=[n for n, _ in SCENARIOS], indirect=True)
def test_censor_argv_round_trips(monkeypatch, rep_and_layout) -> None:
    rep, layout = rep_and_layout
    recorder = _Recorder()
    monkeypatch.setattr(censor_mod, "load_parquet", lambda _path: None)
    monkeypatch.setattr(censor_mod, "run_censor", recorder)
    monkeypatch.setattr(censor_mod, "save_parquet", _touch)
    censor_mod.cli(_argv("censor", rep, layout))

    p = rep.params
    assert recorder.kwargs == {
        "censor_age": p["censor_age"],
        "seed": rep.seed,
        "gen_censoring": p["gen_censoring"] or {},
        "death_scale": p["death_scale"],
        "death_rho": p["death_rho"],
    }


@pytest.mark.parametrize("rep_and_layout", SCENARIOS, ids=[n for n, _ in SCENARIOS], indirect=True)
def test_ascertain_argv_round_trips(monkeypatch, rep_and_layout) -> None:
    rep, layout = rep_and_layout
    recorder = _Recorder(result=(None, None))
    monkeypatch.setattr(ascertain_mod, "copy_passthrough_if_possible", lambda *a, **k: False)
    monkeypatch.setattr(ascertain_mod, "load_parquet", lambda _path: None)
    monkeypatch.setattr(ascertain_mod, "run_ascertainment", recorder)
    monkeypatch.setattr(ascertain_mod, "save_parquet", _touch)
    ascertain_mod.cli(_argv("ascertain", rep, layout))

    p = rep.params
    assert recorder.kwargs == {
        "dropout_rate": p["dropout_rate"],
        "case_ascertainment_ratio": p["case_ascertainment_ratio"],
        "N_sample": p["N_sample"],
        "seed": rep.seed,
    }


@pytest.mark.parametrize("rep_and_layout", SCENARIOS, ids=[n for n, _ in SCENARIOS], indirect=True)
def test_analyze_argv_round_trips(monkeypatch, rep_and_layout) -> None:
    rep, layout = rep_and_layout
    recorded: dict = {}

    def fake_run_analysis(**kwargs):
        recorded.update(kwargs)
        for key in ("report_output", "plot_payload_output", "samples_output"):
            Path(kwargs[key]).write_text("")

    monkeypatch.setattr(analyze_mod, "run_analysis", fake_run_analysis)
    analyze_mod.cli(_argv("analyze", rep, layout))

    p = rep.params
    identity = {"folder": rep.folder, "scenario": rep.scenario, "rep": rep.rep, "seed": rep.seed}
    assert _subset(recorded, [*identity]) == identity
    assert _subset(recorded, ["censor_age", "gen_censoring", "max_degree", "case_ascertainment_ratio"]) == {
        "censor_age": p["censor_age"],
        "gen_censoring": p["gen_censoring"] or None,
        "max_degree": p["max_degree"],
        "case_ascertainment_ratio": p["case_ascertainment_ratio"],
    }


def test_rep_param_keys_are_config_keys() -> None:
    from simace.config import resolve_defaults

    assert resolve_defaults(REPO_CONFIG).keys() >= REP_PARAM_KEYS
    assert {"seed", "N", "phenotype_params1", "max_degree", "skip_ne_coancestry"} <= REP_PARAM_KEYS
    assert REP_PARAM_KEYS.isdisjoint({"replicates", "plot_format", "blended_diagnosis", "folder"})
