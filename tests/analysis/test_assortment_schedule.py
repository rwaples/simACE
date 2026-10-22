"""The analysis-side assortment schedule matches the simulator's for every configured scenario (#75)."""

from pathlib import Path

import pytest

from simace.analysis.quality.assortative_mating import assortment_schedule, schedule_span
from simace.config import resolve_defaults, resolve_scenarios
from simace.simulation.assortment import AssortmentPlan
from simace.simulation.params import CTransmission, SimulationParams
from simace.simulation.simulate import resolve_per_gen_param

CONFIG_DIR = Path(__file__).resolve().parents[2] / "config"
_DEFAULTS = resolve_defaults(CONFIG_DIR)
_SCENARIOS = {name: {**_DEFAULTS, **params} for name, params in resolve_scenarios(CONFIG_DIR, _DEFAULTS).items()}

_CREATE_KEYS = (
    "seed",
    "N",
    "G_ped",
    "mating_lambda",
    "p_mztwin",
    "A1",
    "C1",
    "A2",
    "C2",
    "rA",
    "rC",
    "rE",
    "E1",
    "E2",
    "G_sim",
    "assort1",
    "assort2",
    "assort_matrix",
    "mating_model",
)


def _simulator_schedule(params: dict) -> tuple[list[float], list[float]]:
    """The per-iteration assortment ``run_simulation`` mates on, built the way it builds it."""
    sim = SimulationParams.create(
        **{k: params.get(k) for k in _CREATE_KEYS if params.get(k) is not None},
        c_transmission=CTransmission.from_params(params),
    )
    if sim.mating_model != "standard":  # run_simulation builds no plan: Wright-Fisher pairs at random
        return [0.0] * sim.G_sim, [0.0] * sim.G_sim
    per_gen = {k: resolve_per_gen_param(params.get(k, 0.0), sim.G_sim, name=k) for k in ("C1", "C2", "E1", "E2")}
    plan = AssortmentPlan.build(
        assort1=sim.assort1,
        assort2=sim.assort2,
        R_mf_user=sim.R_mf,
        rA=params["rA"],
        rC=params["rC"],
        rE=params.get("rE", 0.0),
        A1=params["A1"],
        A2=params["A2"],
        C1_per_gen=per_gen["C1"],
        C2_per_gen=per_gen["C2"],
        E1_per_gen=per_gen["E1"],
        E2_per_gen=per_gen["E2"],
        G_sim=sim.G_sim,
    )
    schedule = [plan.for_generation(i) for i in range(sim.G_sim)]
    return [s[0] for s in schedule], [s[1] for s in schedule]


_ASSORTING = sorted(name for name, p in _SCENARIOS.items() if any(v != 0 for s in assortment_schedule(p) for v in s))


def test_some_configured_scenarios_assort_and_one_is_wright_fisher():
    assert len(_ASSORTING) >= 3
    assert any(p.get("mating_model") == "wright_fisher" for p in _SCENARIOS.values())


@pytest.mark.parametrize("name", _ASSORTING)
def test_assortment_schedule_matches_the_simulator(name):
    params = _SCENARIOS[name]
    assert assortment_schedule(params) == _simulator_schedule(params)
    assert schedule_span(params)[0] == len(assortment_schedule(params)[0])


def test_wright_fisher_ignores_the_assortment_it_inherits():
    params = {**_DEFAULTS, "mating_model": "wright_fisher", "assort1": 0.3, "G_ped": 3, "G_sim": 5}
    assert assortment_schedule(params) == ([0.0] * 5, [0.0] * 5)
    assert assortment_schedule(params) == _simulator_schedule({**params, "assort1": 0.0})
