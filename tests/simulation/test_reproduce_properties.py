"""Exact properties of ``reproduce`` over constructed family topologies.

The generator builds reproduction inputs directly (parents, mother-grouped
households, twin pairs) and guarantees each family topology through
parametrization. Expected values never come from ``reproduce`` itself: a
property either checks an identity the model fixes exactly (household C, twin
A, zero-SD components, midparent A) or compares two production calls made
with identical seeds after one intervention on their inputs.
"""

import enum
import warnings
from typing import NamedTuple

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from simace.simulation.simulate import reproduce


class Topology(enum.StrEnum):
    FULL_SIBS = "full_sibs"
    MATERNAL_HALF_SIBS = "maternal_half_sibs"
    PATERNAL_HALF_SIBS = "paternal_half_sibs"
    MZ_TWINS = "mz_twins"
    UNRELATED_HOUSEHOLDS = "unrelated_households"


class Mating(NamedTuple):
    """One mating as ranks into the female and male parent lists."""

    mother: int
    father: int
    n_children: int
    twins: bool


class ReproInputs(NamedTuple):
    """Parent generation and offspring structure accepted by ``reproduce``.

    ``pheno`` is ``(n_par, 6)`` ``[A1, C1, E1, A2, C2, E2]``; ``parent_sex`` is
    ``(n_par,)`` with 0 female, 1 male; ``parents`` is ``(n, 2)``
    ``[mother, father]``; ``household_ids`` is ``(n,)``, contiguous from 0 with
    one household per mother; ``twins`` is ``(k, 2)`` disjoint offspring pairs
    sharing both parents.
    """

    pheno: np.ndarray
    parent_sex: np.ndarray
    parents: np.ndarray
    household_ids: np.ndarray
    twins: np.ndarray


class ReproParams(NamedTuple):
    sd_A1: float
    sd_E1: float
    sd_C1: float
    sd_A2: float
    sd_E2: float
    sd_C2: float
    rA: float
    rC: float
    rE: float


# Each topology's guaranteed block, as (mother rank, father rank, min children, twins).
_REQUIRED_MATINGS: dict[Topology, tuple[tuple[int, int, int, bool], ...]] = {
    Topology.FULL_SIBS: ((0, 0, 2, False),),
    Topology.MATERNAL_HALF_SIBS: ((0, 0, 1, False), (0, 1, 1, False)),
    Topology.PATERNAL_HALF_SIBS: ((0, 0, 1, False), (1, 0, 1, False)),
    Topology.MZ_TWINS: ((0, 0, 2, True),),
    Topology.UNRELATED_HOUSEHOLDS: ((0, 0, 1, False), (1, 1, 1, False)),
}

_COMPONENT_COLUMN = {"sd_A1": 0, "sd_C1": 1, "sd_E1": 2, "sd_A2": 3, "sd_C2": 4, "sd_E2": 5}
_A_COLUMNS = (0, 3)
_C_COLUMNS = (1, 4)
_CE_COLUMNS = (1, 2, 4, 5)
_TRAIT_COLUMNS = {1: (0, 1, 2), 2: (3, 4, 5)}

_SEEDS = st.integers(min_value=0, max_value=2**31 - 1)
_VALUES = st.floats(min_value=-5.0, max_value=5.0, allow_nan=False, allow_infinity=False)
_SDS = st.one_of(st.just(0.0), st.floats(min_value=0.01, max_value=3.0))
_CORRELATIONS = st.one_of(st.sampled_from([-1.0, 0.0, 1.0]), st.floats(min_value=-1.0, max_value=1.0))
# For properties whose defect does not depend on family shape; topology-sensitive
# properties parametrize over every Topology instead.
_ANY_TOPOLOGY = st.sampled_from(list(Topology))


@st.composite
def repro_params(draw) -> ReproParams:
    return ReproParams(*(draw(_SDS) for _ in range(6)), *(draw(_CORRELATIONS) for _ in range(3)))


def _first_appearance_labels(keys: np.ndarray) -> np.ndarray:
    """Contiguous labels numbered by each key's first row."""
    _, first_row, inverse = np.unique(keys, return_index=True, return_inverse=True)
    return np.argsort(np.argsort(first_row))[inverse]


@st.composite
def repro_inputs(draw, topology: Topology) -> ReproInputs:
    n_female = draw(st.integers(min_value=2, max_value=5))
    n_male = draw(st.integers(min_value=2, max_value=5))
    layout = np.array(draw(st.permutations(range(n_female + n_male))))
    females, males = layout[:n_female], layout[n_female:]
    parent_sex = np.zeros(n_female + n_male, dtype=np.int64)
    parent_sex[males] = 1

    required = [
        Mating(mother, father, draw(st.integers(min_value=min_children, max_value=3)), twins)
        for mother, father, min_children, twins in _REQUIRED_MATINGS[topology]
    ]
    extra = draw(
        st.lists(
            st.builds(
                Mating,
                st.integers(min_value=0, max_value=n_female - 1),
                st.integers(min_value=0, max_value=n_male - 1),
                st.integers(min_value=1, max_value=3),
                st.booleans(),
            ),
            max_size=5,
        )
    )

    parent_rows: list[tuple[int, int]] = []
    twin_rows: list[tuple[int, int]] = []
    for m in required + extra:
        if m.twins and m.n_children >= 2:
            twin_rows.append((len(parent_rows), len(parent_rows) + 1))
        parent_rows.extend([(females[m.mother], males[m.father])] * m.n_children)

    # Shuffle offspring rows so neither households nor matings occupy contiguous rows.
    order = np.array(draw(st.permutations(range(len(parent_rows)))))
    new_position = np.argsort(order)
    parents = np.array(parent_rows, dtype=np.int64)[order]
    twins = new_position[np.array(twin_rows, dtype=np.int64).reshape(-1, 2)]

    pheno = draw(arrays(np.float64, (n_female + n_male, 6), elements=_VALUES))
    return ReproInputs(pheno, parent_sex, parents, _first_appearance_labels(parents[:, 0]), twins)


def _reproduce(seed: int, inputs: ReproInputs, params: ReproParams) -> tuple[np.ndarray, np.ndarray]:
    with warnings.catch_warnings():
        # |r| = 1 gives a rank-1 covariance; numpy's PSD roundoff check warns on it.
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return reproduce(
            np.random.default_rng(seed),
            inputs.pheno,
            inputs.parents,
            inputs.twins,
            inputs.household_ids,
            **params._asdict(),
        )


def _witnessed_topologies(inputs: ReproInputs) -> set[Topology]:
    mothers, fathers = inputs.parents[:, 0], inputs.parents[:, 1]
    witnessed = set()
    if len(np.unique(inputs.parents, axis=0)) < len(inputs.parents):
        witnessed.add(Topology.FULL_SIBS)
    if any(len(np.unique(fathers[mothers == m])) > 1 for m in np.unique(mothers)):
        witnessed.add(Topology.MATERNAL_HALF_SIBS)
    if any(len(np.unique(mothers[fathers == f])) > 1 for f in np.unique(fathers)):
        witnessed.add(Topology.PATERNAL_HALF_SIBS)
    if len(inputs.twins):
        witnessed.add(Topology.MZ_TWINS)
    unrelated = (mothers[:, None] != mothers[None, :]) & (fathers[:, None] != fathers[None, :])
    if unrelated.any():
        witnessed.add(Topology.UNRELATED_HOUSEHOLDS)
    return witnessed


def _constant_within(values: np.ndarray, groups: np.ndarray) -> bool:
    representative = np.empty(groups.max() + 1, dtype=values.dtype)
    representative[groups] = values
    return np.array_equal(values, representative[groups])


@pytest.mark.parametrize("topology", list(Topology))
@given(data=st.data())
def test_generator_contract(topology, data):
    """The generator yields the requested topology and valid reproduction inputs.

    Rejects a generator that silently stops producing a parametrized topology,
    breaks the maternal-household grouping, or pairs twins across matings.
    """
    inputs = data.draw(repro_inputs(topology))
    mothers, fathers = inputs.parents[:, 0], inputs.parents[:, 1]

    assert topology in _witnessed_topologies(inputs)
    assert np.all(inputs.parent_sex[mothers] == 0)
    assert np.all(inputs.parent_sex[fathers] == 1)
    assert np.array_equal(np.unique(inputs.household_ids), np.arange(inputs.household_ids.max() + 1))
    assert _constant_within(mothers, inputs.household_ids)
    assert len(np.unique(inputs.household_ids)) == len(np.unique(mothers))
    assert len(np.unique(inputs.twins)) == inputs.twins.size
    assert np.array_equal(inputs.parents[inputs.twins[:, 0]], inputs.parents[inputs.twins[:, 1]])


@pytest.mark.parametrize("topology", list(Topology))
@given(data=st.data(), params=repro_params(), seed=_SEEDS)
def test_household_members_share_c(topology, data, params, seed):
    """Offspring in one household carry identical C for each trait, for any SDs and correlations.

    Rejects drawing C per child or per mating; the maternal half-sib case
    separates per-mating from per-household draws.
    """
    inputs = data.draw(repro_inputs(topology))
    offspring, _ = _reproduce(seed, inputs, params)
    for col in _C_COLUMNS:
        assert _constant_within(offspring[:, col], inputs.household_ids)


@given(data=st.data(), params=repro_params(), seed=_SEEDS)
def test_mz_twins_share_a_c_and_sex(data, params, seed):
    """MZ twins (same parents, hence same household) share A, C, and sex for both traits.

    Rejects twins that do not copy A or sex. E and liability are not required to differ.
    """
    inputs = data.draw(repro_inputs(Topology.MZ_TWINS))
    offspring, sex = _reproduce(seed, inputs, params)
    first, second = inputs.twins[:, 0], inputs.twins[:, 1]
    shared = [*_A_COLUMNS, *_C_COLUMNS]
    assert np.array_equal(offspring[first][:, shared], offspring[second][:, shared])
    assert np.array_equal(sex[first], sex[second])


@given(data=st.data(), params=repro_params(), seed=_SEEDS)
def test_twin_pairing_touches_only_twin_a_and_sex(data, params, seed):
    """Declaring twins changes nothing but the pairs' A and sex, given the same seed.

    Each twin keeps the E it would draw as a singleton, which rejects copying
    one twin's E to the other; unchanged non-twin rows reject twin handling
    that shifts the random stream.
    """
    inputs = data.draw(repro_inputs(Topology.MZ_TWINS))
    with_twins, sex_with = _reproduce(seed, inputs, params)
    singletons, sex_without = _reproduce(seed, inputs._replace(twins=np.empty((0, 2), dtype=np.int64)), params)

    assert np.array_equal(with_twins[:, _CE_COLUMNS], singletons[:, _CE_COLUMNS])
    untouched = np.setdiff1d(np.arange(len(sex_with)), inputs.twins)
    assert np.array_equal(with_twins[untouched], singletons[untouched])
    assert np.array_equal(sex_with[untouched], sex_without[untouched])


@given(data=st.data(), topology=_ANY_TOPOLOGY, params=repro_params(), seed=_SEEDS)
def test_parental_c_and_e_do_not_reach_offspring(data, topology, params, seed):
    """Replacing the parents' C and E columns leaves every offspring value and sex unchanged.

    Holds parents, topology, parameters, and seed fixed. Rejects reading
    parental C as inherited C, or letting parental E enter offspring values.
    """
    inputs = data.draw(repro_inputs(topology))
    replacement = data.draw(arrays(np.float64, (len(inputs.pheno), len(_CE_COLUMNS)), elements=_VALUES))
    altered = inputs.pheno.copy()
    altered[:, _CE_COLUMNS] = replacement

    offspring, sex = _reproduce(seed, inputs, params)
    offspring_alt, sex_alt = _reproduce(seed, inputs._replace(pheno=altered), params)
    assert np.array_equal(offspring, offspring_alt)
    assert np.array_equal(sex, sex_alt)


@given(
    data=st.data(),
    topology=_ANY_TOPOLOGY,
    params=repro_params(),
    zeroed=st.sets(st.sampled_from(sorted(_COMPONENT_COLUMN)), min_size=1),
    seed=_SEEDS,
)
def test_zero_sd_adds_no_noise(data, topology, params, zeroed, seed):
    """A zero C or E SD gives exactly zero offspring C or E; a zero A SD leaves the midparent A.

    Twins share both parents, so twin A copying keeps the midparent. Rejects
    noise leaking into a zeroed component, e.g. through a correlated partner draw.
    """
    inputs = data.draw(repro_inputs(topology))
    offspring, _ = _reproduce(seed, inputs, params._replace(**dict.fromkeys(zeroed, 0.0)))
    mothers, fathers = inputs.parents[:, 0], inputs.parents[:, 1]
    for name in zeroed:
        col = _COMPONENT_COLUMN[name]
        if col in _A_COLUMNS:
            expected = (inputs.pheno[mothers, col] + inputs.pheno[fathers, col]) / 2
        else:
            expected = np.zeros(len(mothers))
        assert np.array_equal(offspring[:, col], expected), name


@given(data=st.data(), topology=_ANY_TOPOLOGY, params=repro_params(), seed=_SEEDS)
def test_parental_a_stays_in_its_trait(data, topology, params, seed):
    """Changing one trait's parental A leaves the other trait's offspring components and sex unchanged.

    Holds cross-trait correlations, noise SDs, and seed fixed, and checks both
    directions. Rejects reading the wrong trait's parental A column.
    """
    inputs = data.draw(repro_inputs(topology))
    offspring, sex = _reproduce(seed, inputs, params)
    for changed, kept in ((1, 2), (2, 1)):
        altered = inputs.pheno.copy()
        altered[:, _TRAIT_COLUMNS[changed][0]] = data.draw(arrays(np.float64, len(inputs.pheno), elements=_VALUES))
        offspring_alt, sex_alt = _reproduce(seed, inputs._replace(pheno=altered), params)
        kept_cols = list(_TRAIT_COLUMNS[kept])
        assert np.array_equal(offspring[:, kept_cols], offspring_alt[:, kept_cols]), f"trait {changed} A leaked"
        assert np.array_equal(sex, sex_alt)
