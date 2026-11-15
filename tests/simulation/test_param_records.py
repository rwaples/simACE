"""The parameter records read their keys one way, and ``Assortment`` says what the simulator mates on."""

import pytest
from polars.testing import assert_frame_equal

from simace.simulation.inbreeding_plan import DEFAULT_DEPTH
from simace.simulation.params import Assortment, CTransmission, SimulationParams, Span, VarianceComponents
from simace.simulation.simulate import run_simulation

_PARAMS = dict(
    seed=7,
    N=300,
    G_ped=3,
    G_sim=5,
    mating_lambda=0.5,
    p_mztwin=0.0,
    A1=0.5,
    C1=0.2,
    A2=0.4,
    C2=0.2,
    E1=0.3,
    E2=0.4,
    rA=0.3,
    rC=0.0,
    rE=0.0,
)


def test_a_params_yaml_predating_mating_model_c_transmission_and_inbreeding_takes_their_defaults():
    old = {**_PARAMS, "assort1": 0.0, "assort2": 0.0}
    parsed = SimulationParams.from_mapping(old)
    assert parsed.mating_model == "standard"
    assert parsed.c_transmission == CTransmission()
    assert parsed.inbreeding.mode == "off"
    current = {
        **old,
        "mating_model": "standard",
        "inbreeding_mode": "off",
        "inbreeding_depth": DEFAULT_DEPTH,
        **dict.fromkeys(("Ctm_maternal1", "Ctm_paternal1", "Ctm_maternal2", "Ctm_paternal2"), 0.0),
    }
    assert SimulationParams.from_mapping(current) == parsed


def test_a_key_outside_the_legacy_table_is_required():
    with pytest.raises(KeyError, match="rE"):
        VarianceComponents.from_mapping({k: v for k, v in _PARAMS.items() if k != "rE"})


def test_a_null_g_sim_is_no_burn_in():
    assert Span.from_mapping({"G_ped": 4, "G_sim": None}) == Span(4, 4)
    assert Span.from_mapping({"G_ped": 4, "G_sim": 6}).burnin == 2


def test_wright_fisher_has_no_schedule_and_no_matrix():
    wf = Assortment.create(Span(3, 4), 0.4, 0.3, [[0.4, 0.1], [0.1, 0.3]], "wright_fisher")
    assert wf.schedule == ((0.0,) * 4, (0.0,) * 4)
    assert wf.mated_matrix is None
    assert not wf.any_active


# Each case is (assort1, assort2, assort_matrix, mating_model); G_sim is 5.
_CASES = {
    "none": (0.0, 0.0, None, "standard"),
    "trait1": (0.3, 0.0, None, "standard"),
    "both": (0.3, 0.2, None, "standard"),
    "matrix": (0.0, 0.0, [[0.3, 0.05], [0.05, 0.2]], "standard"),
    "zero_diagonal_matrix": (0.0, 0.0, [[0.0, 0.1], [0.1, 0.0]], "standard"),
    "map_from_iteration_2": ({0: 0.0, 2: 0.3}, 0.0, None, "standard"),
    "map_past_g_sim": ({0: 0.0, 5: 0.3}, 0.0, None, "standard"),
    "wright_fisher": (0.3, 0.2, None, "wright_fisher"),
}


@pytest.mark.parametrize("case", sorted(_CASES))
def test_assortment_is_active_exactly_when_the_simulator_mates_on_it(case):
    """Differential: the pedigree differs from its assortment-free twin iff ``Assortment.any_active``."""
    assort1, assort2, matrix, model = _CASES[case]
    configured = {**_PARAMS, "assort1": assort1, "assort2": assort2, "assort_matrix": matrix, "mating_model": model}
    random_mating = {**_PARAMS, "mating_model": model}
    assortment = Assortment.create(Span(3, 5), assort1, assort2, matrix, model)
    pedigree = run_simulation(**configured).pedigree
    reference = run_simulation(**random_mating).pedigree
    if assortment.any_active:
        assert not pedigree.equals(reference)
    else:
        assert_frame_equal(pedigree, reference)


def test_reading_a_rep_never_runs_the_inbreeding_cost_refusal():
    """A finished rep must still parse if the cost model would now refuse it."""
    from simace.simulation.inbreeding_plan import InbreedingRefused

    refused = {
        **_PARAMS,
        "N": 5_000_000,
        "G_sim": 60,
        "assort1": 0.0,
        "assort2": 0.0,
        "inbreeding_mode": "exact",
    }
    assert SimulationParams.from_mapping(refused).inbreeding.mode == "exact"
    with pytest.raises(InbreedingRefused):
        SimulationParams.create(**{k: v for k, v in refused.items() if k != "inbreeding_mode"}, inbreeding_mode="exact")
