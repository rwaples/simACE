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


def _names_assortment(params: dict) -> bool:
    """Whether the raw parameters ask for assortment or a non-standard mating model, whatever either side resolves."""
    raw = [params.get("assort1", 0.0), params.get("assort2", 0.0)]
    values = [v for a in raw for v in (a.values() if isinstance(a, dict) else [a])]
    return (
        any(float(v) != 0 for v in values)
        or params.get("assort_matrix") is not None
        or params.get("mating_model", "standard") != "standard"
    )


# Chosen from the raw parameters, not from the function under test, so a scenario
# whose analysis schedule wrongly comes out all zero is still compared.
_SELECTED = sorted(name for name, p in _SCENARIOS.items() if _names_assortment(p))


def test_the_selection_covers_assortment_matrices_schedules_and_wright_fisher():
    picked = [_SCENARIOS[n] for n in _SELECTED]
    assert sum(p.get("assort_matrix") is not None for p in picked) >= 1
    assert sum(isinstance(p.get("assort1"), dict) or isinstance(p.get("assort2"), dict) for p in picked) >= 1
    assert sum(p.get("mating_model") == "wright_fisher" for p in picked) >= 1


@pytest.mark.parametrize("name", _SELECTED)
def test_assortment_schedule_matches_the_simulator(name):
    params = _SCENARIOS[name]
    assert assortment_schedule(params) == _simulator_schedule(params)
    assert schedule_span(params)[0] == len(assortment_schedule(params)[0])


def test_wright_fisher_ignores_the_assortment_it_inherits():
    """Through ``run_simulation`` itself: under Wright-Fisher, assortment leaves the pedigree unchanged."""
    from polars.testing import assert_frame_equal

    from simace.simulation.simulate import run_simulation

    base = dict(
        seed=5,
        N=400,
        G_ped=3,
        G_sim=4,
        mating_lambda=0.5,
        p_mztwin=0.0,
        A1=0.5,
        C1=0.2,
        A2=0.4,
        C2=0.2,
        rA=0.3,
        rC=0.0,
        E1=0.3,
        E2=0.4,
        mating_model="wright_fisher",
    )
    assert_frame_equal(run_simulation(**base, assort1=0.4, assort2=0.3), run_simulation(**base))
    assert assortment_schedule({**base, "assort1": 0.4, "assort2": 0.3}) == ([0.0] * 4, [0.0] * 4)
