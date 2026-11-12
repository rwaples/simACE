"""Every scenario's parameters reach the domain functions unchanged through stage argv.

Each stage CLI runs on the argv ``simace run`` builds, with its domain function
replaced by a recorder. The recorded keyword arguments must equal what the
config holds, which is what the Snakemake wrappers passed.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import simace.analysis.analyze as analyze_mod
import simace.cli.cohort_stage as cohort_mod
import simace.simulation.simulate as simulate_mod
from simace.cli.layout import Layout
from simace.cli.stages import REP_PARAM_KEYS, STAGES, ResolvedRep
from simace.cli.status import resolve_all

REPO_CONFIG = Path(__file__).resolve().parents[2] / "config"


_SIMULATE_KEYS = ["N", "G_ped", "G_sim", "mating_model", "mating_lambda", "p_mztwin"]
_SIMULATE_KEYS += ["A1", "C1", "E1", "A2", "C2", "E2", "rA", "rC", "rE", "assort1", "assort2", "assort_matrix"]
_SIMULATE_KEYS += ["inbreeding_mode", "inbreeding_depth"]
_PHENOTYPE_KEYS = ["G_pheno", "standardize"]
_PHENOTYPE_KEYS += [f"{k}{t}" for t in (1, 2) for k in ("phenotype_model", "phenotype_params", "beta", "beta_sex")]
_CENSOR_KEYS = ["censor_age", "gen_censoring", "death_scale", "death_rho"]
_ASCERTAIN_KEYS = ["dropout_rate", "case_ascertainment_ratio", "N_sample"]
_ANALYZE_KEYS = ["censor_age", "gen_censoring", "max_degree", "case_ascertainment_ratio"]
_READ_KEYS = sorted({*_SIMULATE_KEYS, *_PHENOTYPE_KEYS, *_CENSOR_KEYS, *_ASCERTAIN_KEYS, *_ANALYZE_KEYS})


def _shape(key: str, value: object) -> object:
    """The encoding-relevant shape of one param value (model-selecting strings keep their value)."""
    if value is None:
        return "none"
    if isinstance(value, dict):
        items = {("gen" if isinstance(k, int) else k): _shape(str(k), v) for k, v in value.items()}
        return ("dict", tuple(sorted(items.items(), key=lambda kv: str(kv[0]))))
    if isinstance(value, list):
        return ("list", _shape(key, value[0]) if value else "empty")
    if isinstance(value, str) and key.rstrip("12").endswith("_model"):
        return value
    return type(value).__name__


def _representatives(scenarios: list[tuple[str, dict]]) -> list[tuple[str, dict]]:
    """The first scenario (by name) for each distinct shape signature over the keys the round-trip tests read.

    Shape, not value, decides how a param is encoded into argv, so one scenario
    per signature covers every encoding path; a config with a new shape joins
    the grid automatically.
    """
    seen: set[object] = set()
    out = []
    for name, params in scenarios:
        signature = tuple(_shape(key, params.get(key)) for key in _READ_KEYS)
        if signature not in seen:
            seen.add(signature)
            out.append((name, params))
    return out


ALL = sorted(resolve_all(REPO_CONFIG).items())
SCENARIOS = _representatives(ALL)


class _Recorder:
    def __init__(self, result=None) -> None:
        self.kwargs: dict = {}
        self.result = result

    def __call__(self, *args, **kwargs):
        self.kwargs = kwargs
        return self.result


def _touch(_obj, path, **_kwargs) -> None:
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
    monkeypatch.setattr(simulate_mod, "write_pedigree", _touch)
    simulate_mod.cli(_argv("simulate", rep, layout))

    assert recorder.kwargs["seed"] == rep.seed
    assert _subset(recorder.kwargs, _SIMULATE_KEYS) == {key: rep.params[key] for key in _SIMULATE_KEYS}


@pytest.mark.parametrize("rep_and_layout", SCENARIOS, ids=[n for n, _ in SCENARIOS], indirect=True)
def test_cohort_argv_round_trips(monkeypatch, rep_and_layout) -> None:
    rep, layout = rep_and_layout
    recorder = _Recorder(result=(None, {}))
    monkeypatch.setattr(cohort_mod, "read_pedigree", lambda _path: None)
    monkeypatch.setattr(cohort_mod, "run_cohort", recorder)
    monkeypatch.setattr(cohort_mod, "write_cohort", _touch)
    cohort_mod.cli(_argv("cohort", rep, layout))

    p = rep.params
    assert recorder.kwargs["seed"] == rep.seed
    assert recorder.kwargs["phenotype"] == {key: p[key] for key in _PHENOTYPE_KEYS}
    assert recorder.kwargs["censor"] == {
        "censor_age": p["censor_age"],
        "gen_censoring": p["gen_censoring"] or {},
        "death_scale": p["death_scale"],
        "death_rho": p["death_rho"],
    }
    assert recorder.kwargs["ascertain"] == {
        "dropout_rate": p["dropout_rate"],
        "case_ascertainment_ratio": p["case_ascertainment_ratio"],
        "N_sample": p["N_sample"],
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


def test_plot_argv_hands_the_plot_stage_each_validation_file_and_the_scenario_verdict(monkeypatch, tmp_path) -> None:
    import simace.plotting.plot_phenotype as plot_phenotype_mod
    from simace.cli.layout import RepArtifact
    from simace.cli.stages import plot_argv

    params = dict(ALL)["onset_tau_weibull"]
    reps = [ResolvedRep(params["folder"], "onset_tau_weibull", r, params) for r in (1, 2)]
    layout = Layout(root=tmp_path / "r", logs=tmp_path / "l")
    recorder = _Recorder()
    monkeypatch.setattr(plot_phenotype_mod, "main", recorder)
    plot_phenotype_mod.cli(plot_argv(reps, layout))

    assert recorder.kwargs["validation_paths"] == [str(rep.path(layout, RepArtifact.VALIDATION)) for rep in reps]
    assert recorder.kwargs["verdict_path"] == str(layout.scenario_verdict(params["folder"], "onset_tau_weibull"))


def test_flags_refuse_null_outside_the_nullable_keys() -> None:
    """A dropped flag would hand the stage its own default instead of the config's (#94)."""
    from simace.cli.stages import _flags

    assert _flags({"assort_matrix": None, "G_sim": None}) == []
    with pytest.raises(ValueError, match="rA"):
        _flags({"rA": None})
