"""Every non-standard scenario mates at random in the simulator, as its ``Assortment`` says (#75, #86)."""

from pathlib import Path

import pytest
from polars.testing import assert_frame_equal

from simace.config import load_scenarios
from simace.simulation.params import SimulationParams
from simace.simulation.simulate import run_simulation

CONFIG_DIR = Path(__file__).resolve().parents[2] / "config"
_SCENARIOS = {name: scenario.flat for name, scenario in load_scenarios(CONFIG_DIR).items()}

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
    "Ctm_maternal1",
    "Ctm_paternal1",
    "Ctm_maternal2",
    "Ctm_paternal2",
)
_NON_STANDARD = sorted(name for name, p in _SCENARIOS.items() if p["mating_model"] != "standard")


def test_a_non_standard_scenario_is_configured():
    assert _NON_STANDARD


@pytest.mark.parametrize("name", _NON_STANDARD)
def test_a_configured_non_standard_scenario_mates_at_random_in_the_simulator(name):
    """Through ``run_simulation`` itself: the scenario as configured (N reduced) and with added assortment both
    give the pedigree its assortment-free version gives, and its ``Assortment`` is inactive."""
    params = _SCENARIOS[name]
    configured = {k: params[k] for k in _CREATE_KEYS if params.get(k) is not None} | {"N": 300}
    stripped = {k: v for k, v in configured.items() if k not in ("assort1", "assort2", "assort_matrix")}
    reference = run_simulation(**stripped).pedigree
    assert_frame_equal(run_simulation(**configured).pedigree, reference)
    assert_frame_equal(run_simulation(**stripped, assort1=0.4, assort2=0.3).pedigree, reference)
    assortment = SimulationParams.from_mapping(params).assortment
    assert not assortment.any_active
    assert len(assortment.schedule[0]) == SimulationParams.from_mapping(params).span.G_sim
