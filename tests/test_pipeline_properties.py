"""Cross-stage properties for the phenotype → censor pipeline."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypedDict

import numpy as np
import polars as pl
import pytest
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.strategies import SearchStrategy

from simace.censoring.censor import run_censor
from simace.core.schema import PEDIGREE
from simace.core.trait_schema import TRAIT_CENSORED_COLUMNS, TRAIT_RAW_COLUMNS
from simace.phenotype import run_phenotype
from simace.phenotype.hazards import STANDARDIZE_CHOICES
from tests.conftest import HAZARD_PARAM_BOUNDS, hazard_params, pedigree_frame, schema_pad

_MODELS = ("frailty", "cure_frailty", "adult", "first_passage", "simple_ltm")
_DEATH_ANCHOR_SEED = 0

type Draw = Callable[[SearchStrategy[Any]], Any]
type CensorWindows = dict[int, list[float]]


class ChainCase(TypedDict):
    """Drawn inputs for one phenotype → censor chain."""

    pedigree: pl.DataFrame
    phenotype_kwargs: dict[str, Any]
    censor_kwargs: dict[str, Any]
    wide_censoring: CensorWindows


def _finite_float(lo: float, hi: float) -> SearchStrategy[float]:
    return st.floats(min_value=lo, max_value=hi, allow_nan=False, allow_infinity=False)


def _draw_model_params(draw: Draw, model: str) -> dict[str, Any]:
    if model in ("frailty", "cure_frailty"):
        distribution = draw(st.sampled_from(sorted(HAZARD_PARAM_BOUNDS)))
        params = {"distribution": distribution, **draw(hazard_params(distribution))}
        if model == "cure_frailty":
            params["prevalence"] = draw(_finite_float(0.05, 0.95))
        return params
    if model == "adult":
        return {
            "method": draw(st.sampled_from(("ltm", "cox"))),
            "prevalence": draw(_finite_float(0.05, 0.95)),
            "cip_x0": 50.0,
            "cip_k": 0.2,
        }
    if model == "first_passage":
        return {
            "drift": draw(st.one_of(_finite_float(-2.0, -0.02), _finite_float(0.02, 2.0))),
            "shape": draw(_finite_float(0.05, 20.0)),
        }

    onset = draw(st.sampled_from(("fixed", "normal")))
    if onset == "fixed":
        onset_params = {"kind": "fixed", "age": draw(_finite_float(0.01, 120.0))}
    else:
        onset_params = {
            "kind": "normal",
            "mean": draw(_finite_float(0.0, 120.0)),
            "sd": draw(_finite_float(0.1, 30.0)),
        }
    return {"prevalence": draw(_finite_float(0.05, 0.95)), "onset": onset_params}


def _trailing_pedigree(pedigree: pl.DataFrame, g_pheno: int) -> pl.DataFrame:
    min_gen = int(pedigree["generation"].max()) - g_pheno + 1
    return pedigree.filter(pl.col("generation") >= min_gen)


@st.composite
def _chain_case(draw: Draw) -> ChainCase:
    pedigree = draw(pedigree_frame(liabilities=True))
    max_gen = int(pedigree["generation"].max())
    n_generations = max_gen + 1
    g_pheno = draw(st.integers(min_value=1, max_value=n_generations))

    model1 = draw(st.sampled_from(_MODELS))
    model2 = draw(st.sampled_from(_MODELS))
    phenotype_kwargs = {
        "G_pheno": g_pheno,
        "seed": draw(st.integers(min_value=0, max_value=2**31 - 1)),
        "standardize": draw(st.sampled_from(STANDARDIZE_CHOICES)),
        "phenotype_model1": model1,
        "phenotype_params1": _draw_model_params(draw, model1),
        "beta1": draw(_finite_float(-2.0, 2.0)),
        "beta_sex1": draw(_finite_float(-2.0, 2.0)),
        "phenotype_model2": model2,
        "phenotype_params2": _draw_model_params(draw, model2),
        "beta2": draw(_finite_float(-2.0, 2.0)),
        "beta_sex2": draw(_finite_float(-2.0, 2.0)),
    }

    censor_age = draw(_finite_float(1.0, 150.0))
    generations = sorted(int(value) for value in _trailing_pedigree(pedigree, g_pheno)["generation"].unique())
    gen_censoring: CensorWindows = {}
    for generation in generations:
        if draw(st.booleans()):
            lo = draw(_finite_float(0.0, 150.0))
            width = draw(_finite_float(0.0, 150.0))
            gen_censoring[generation] = [lo, lo + width]

    wide_censoring: CensorWindows = {}
    for generation in generations:
        lo, hi = gen_censoring.get(generation, [0.0, censor_age])
        wide_censoring[generation] = [
            draw(_finite_float(0.0, lo)),
            hi + draw(_finite_float(0.0, 150.0)),
        ]

    return {
        "pedigree": pedigree,
        "phenotype_kwargs": phenotype_kwargs,
        "censor_kwargs": {
            "censor_age": censor_age,
            "seed": draw(st.integers(min_value=0, max_value=2**31 - 1)),
            "gen_censoring": gen_censoring,
            "death_scale": draw(_finite_float(1.0, 500.0)),
            "death_rho": draw(_finite_float(0.5, 20.0)),
        },
        "wide_censoring": wide_censoring,
    }


def _phenotyped_pedigree(case: ChainCase) -> pl.DataFrame:
    return _trailing_pedigree(case["pedigree"], case["phenotype_kwargs"]["G_pheno"])


def _run_phenotype(case: ChainCase) -> pl.DataFrame:
    return run_phenotype(case["pedigree"], **case["phenotype_kwargs"])


def _run_censor(case: ChainCase, phenotype: pl.DataFrame, **overrides: Any) -> pl.DataFrame:
    censor_kwargs = {**case["censor_kwargs"], **overrides}
    return run_censor(phenotype, case["pedigree"], **censor_kwargs)


def _effective_right(case: ChainCase) -> np.ndarray:
    generations = _phenotyped_pedigree(case)["generation"].to_numpy()
    right = np.full(len(generations), case["censor_kwargs"]["censor_age"])
    for generation, (_, hi) in case["censor_kwargs"]["gen_censoring"].items():
        right[generations == generation] = hi
    return right


def _full_windows(case: ChainCase) -> CensorWindows:
    generations = _phenotyped_pedigree(case)["generation"].unique()
    return {int(generation): [0.0, 1e6] for generation in generations}


def _assert_raw_onset_contract(phenotype: pl.DataFrame) -> None:
    for trait in ("t1", "t2"):
        assert phenotype[trait].null_count() == 0
        onset = phenotype[trait].to_numpy()
        assert np.isfinite(onset).all()
        assert np.all((onset >= 1e-10) & (onset <= 1e6))


@given(_chain_case())
def test_drawn_models_share_the_raw_trait_contract(case: ChainCase) -> None:
    phenotype = _run_phenotype(case)
    expected_pedigree = _phenotyped_pedigree(case)

    assert tuple(phenotype.columns) == TRAIT_RAW_COLUMNS
    assert phenotype["id"].equals(expected_pedigree["id"])
    _assert_raw_onset_contract(phenotype)

    too_many_generations = int(case["pedigree"]["generation"].max()) + 2
    invalid = {**case["phenotype_kwargs"], "G_pheno": too_many_generations}
    with pytest.raises(ValueError, match="exceeds available generations"):
        run_phenotype(case["pedigree"], **invalid)


@given(_chain_case())
def test_real_chain_preserves_censor_contracts(case: ChainCase) -> None:
    phenotype = _run_phenotype(case)
    censored = _run_censor(case, phenotype)
    death_age = censored["death_age"].to_numpy()
    right = _effective_right(case)

    assert tuple(censored.columns) == TRAIT_CENSORED_COLUMNS
    assert censored["id"].equals(phenotype["id"])
    for trait in ("1", "2"):
        affected = censored[f"affected{trait}"].to_numpy()
        age_censored = censored[f"age_censored{trait}"].to_numpy()
        death_censored = censored[f"death_censored{trait}"].to_numpy()
        observed = censored[f"t_observed{trait}"].to_numpy()
        raw = phenotype[f"t{trait}"].to_numpy()

        np.testing.assert_array_equal(affected, ~age_censored & ~death_censored)
        assert np.all(observed <= death_age)
        assert np.all(observed <= right)
        np.testing.assert_array_equal(observed[affected], raw[affected])


@given(_chain_case())
def test_both_stages_are_seed_deterministic_in_values_and_dtypes(case: ChainCase) -> None:
    first_raw = _run_phenotype(case)
    second_raw = _run_phenotype(case)
    assert first_raw.schema == second_raw.schema
    assert first_raw.equals(second_raw)

    first_censored = _run_censor(case, first_raw)
    second_censored = _run_censor(case, second_raw)
    assert first_censored.schema == second_censored.schema
    assert first_censored.equals(second_censored)


_MATRIX_PARAMS = {
    "frailty": {"distribution": "weibull", "scale": 100.0, "rho": 2.0},
    "cure_frailty": {
        "distribution": "weibull",
        "scale": 100.0,
        "rho": 2.0,
        "prevalence": 0.5,
    },
    "adult": {"method": "ltm", "prevalence": 0.5, "cip_x0": 50.0, "cip_k": 0.2},
    "first_passage": {"drift": -0.5, "shape": 2.0},
    "simple_ltm": {"prevalence": 0.5, "onset": {"kind": "fixed", "age": 40.0}},
}


def _matrix_pedigree() -> pl.DataFrame:
    frame = pl.DataFrame(
        {
            "id": np.arange(6, dtype=np.int32),
            "generation": np.repeat([0, 1], 3).astype(np.int32),
            "sex": np.tile([0, 1, 0], 2).astype(np.int32),
            "mother": np.array([-1, -1, -1, 0, 0, 2], dtype=np.int32),
            "father": np.array([-1, -1, -1, 1, 1, 1], dtype=np.int32),
            "twin": np.full(6, -1, dtype=np.int32),
            "household_id": np.array([0, 1, 2, 3, 3, 4], dtype=np.int32),
            "liability1": np.linspace(-2.0, 2.0, 6),
            "liability2": np.linspace(2.0, -2.0, 6),
        }
    )
    return schema_pad(frame, PEDIGREE)


@pytest.mark.parametrize("model", _MODELS)
@pytest.mark.parametrize("standardize", STANDARDIZE_CHOICES)
def test_explicit_model_by_standardize_matrix_reaches_the_full_chain(model: str, standardize: str) -> None:
    pedigree = _matrix_pedigree()
    params = _MATRIX_PARAMS[model]
    phenotype = run_phenotype(
        pedigree,
        G_pheno=2,
        seed=123,
        standardize=standardize,
        phenotype_model1=model,
        phenotype_params1=params,
        beta1=0.75,
        beta_sex1=0.25,
        phenotype_model2=model,
        phenotype_params2=params,
        beta2=-0.75,
        beta_sex2=-0.25,
    )
    censored = run_censor(
        phenotype,
        pedigree,
        censor_age=120.0,
        seed=123,
        gen_censoring={},
        death_scale=200.0,
        death_rho=3.0,
    )
    assert tuple(phenotype.columns) == TRAIT_RAW_COLUMNS
    assert tuple(censored.columns) == TRAIT_CENSORED_COLUMNS
    assert censored["id"].equals(phenotype["id"])
    _assert_raw_onset_contract(phenotype)
    for trait in ("1", "2"):
        np.testing.assert_array_equal(
            censored[f"affected{trait}"].to_numpy(),
            ~censored[f"age_censored{trait}"].to_numpy() & ~censored[f"death_censored{trait}"].to_numpy(),
        )


@given(_chain_case())
def test_widening_every_window_can_only_add_affected_rows(case: ChainCase) -> None:
    phenotype = _run_phenotype(case)
    narrow = _run_censor(case, phenotype)
    wide = _run_censor(case, phenotype, gen_censoring=case["wide_censoring"])

    for trait in ("1", "2"):
        narrow_affected = narrow[f"affected{trait}"].to_numpy()
        wide_affected = wide[f"affected{trait}"].to_numpy()
        assert np.all(~narrow_affected | wide_affected)


@given(_chain_case())
def test_full_model_range_has_no_age_censoring(case: ChainCase) -> None:
    phenotype = _run_phenotype(case)
    censored = _run_censor(case, phenotype, censor_age=1e6, gen_censoring=_full_windows(case))

    assert not censored["age_censored1"].any()
    assert not censored["age_censored2"].any()


@given(_chain_case())
def test_scaled_death_anchor_eliminates_death_censoring_exactly(case: ChainCase) -> None:
    phenotype = _run_phenotype(case)
    n = len(phenotype)
    rho = case["censor_kwargs"]["death_rho"]
    rng = np.random.default_rng(_DEATH_ANCHOR_SEED + 1000)
    unit_age = (-np.log(1.0 - rng.uniform(size=n))) ** (1.0 / rho)
    assert np.all(unit_age > 0.0)
    death_scale = 2e6 / float(unit_age.min())

    censored = _run_censor(
        case,
        phenotype,
        censor_age=1e6,
        gen_censoring=_full_windows(case),
        seed=_DEATH_ANCHOR_SEED,
        death_scale=death_scale,
    )

    assert np.all(censored["death_age"].to_numpy() > 1e6)
    assert not censored["death_censored1"].any()
    assert not censored["death_censored2"].any()
