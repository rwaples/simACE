"""Per-trait standardize_hazard override + per-model self-consistency tests.

Covers behaviors that the per-model tests don't exercise:

* Each phenotype model behaves consistently when ``standardize_hazard`` is
  set on the model and the global flag is the legacy bool / new string.
* The per-trait override decouples the hazard step from the global flag.
* Threshold-only models (``threshold``, ``adult.ltm``) reject the field.
* ``cure_frailty`` honors both knobs independently.
"""

from __future__ import annotations

import numpy as np
import pytest

from simace.phenotyping.models.adult import AdultModel
from simace.phenotyping.models.cure_frailty import CureFrailtyModel
from simace.phenotyping.models.first_passage import FirstPassageModel
from simace.phenotyping.models.frailty import FrailtyModel

# ---------------------------------------------------------------------------
# Hazard-bearing models accept the override and inherit by default
# ---------------------------------------------------------------------------


def _frailty_kwargs(standardize_hazard=None):
    return dict(
        distribution="weibull",
        hazard_params={"scale": 100.0, "rho": 2.0},
        beta=1.0,
        beta_sex=0.0,
        standardize_hazard=standardize_hazard,
    )


def _fpt_kwargs(standardize_hazard=None):
    return dict(
        drift=-0.05,
        shape=4.0,
        beta=1.0,
        beta_sex=0.0,
        standardize_hazard=standardize_hazard,
    )


def _cure_frailty_kwargs(standardize_hazard=None):
    return dict(
        distribution="weibull",
        hazard_params={"scale": 100.0, "rho": 2.0},
        prevalence=0.1,
        beta=1.0,
        beta_sex=0.0,
        standardize_hazard=standardize_hazard,
    )


def _adult_cox_kwargs(standardize_hazard=None):
    return dict(
        method="cox",
        prevalence=0.1,
        cip_x0=50.0,
        cip_k=0.2,
        beta=1.0,
        beta_sex=0.0,
        standardize_hazard=standardize_hazard,
    )


def _heteroscedastic_two_gen(n_per=4000, seed=0):
    rng = np.random.default_rng(seed)
    # gen 0 std=1, gen 1 std=3 → very different per-gen variances
    L = np.concatenate([rng.normal(0.0, 1.0, n_per), rng.normal(0.0, 3.0, n_per)])
    g = np.repeat([0, 1], n_per).astype(np.int64)
    return L, g


# ---------------------------------------------------------------------------
# Per-trait override decouples from the global flag
# ---------------------------------------------------------------------------


def test_frailty_hazard_override_decouples_from_global():
    """standardize='global', standardize_hazard='per_generation' → output differs
    from the all-global run."""
    L, g = _heteroscedastic_two_gen()
    sex = np.zeros(len(L))

    t_inherit = FrailtyModel(**_frailty_kwargs(standardize_hazard=None)).simulate(
        liability=L, seed=42, standardize="global", sex=sex, generation=g
    )
    t_override = FrailtyModel(**_frailty_kwargs(standardize_hazard="per_generation")).simulate(
        liability=L, seed=42, standardize="global", sex=sex, generation=g
    )
    # Same RNG seed, but different scaled_beta per individual → outputs diverge.
    assert not np.allclose(t_inherit, t_override)


def test_frailty_hazard_override_inherits_when_unset():
    """standardize_hazard=None should match passing standardize_hazard=standardize."""
    L, g = _heteroscedastic_two_gen()
    sex = np.zeros(len(L))

    t_inherit = FrailtyModel(**_frailty_kwargs(standardize_hazard=None)).simulate(
        liability=L, seed=99, standardize="per_generation", sex=sex, generation=g
    )
    t_explicit = FrailtyModel(**_frailty_kwargs(standardize_hazard="per_generation")).simulate(
        liability=L, seed=99, standardize="per_generation", sex=sex, generation=g
    )
    np.testing.assert_array_equal(t_inherit, t_explicit)


def test_first_passage_hazard_override_decouples():
    L, g = _heteroscedastic_two_gen()
    sex = np.zeros(len(L))
    t_inherit = FirstPassageModel(**_fpt_kwargs(standardize_hazard=None)).simulate(
        liability=L, seed=7, standardize="global", sex=sex, generation=g
    )
    t_override = FirstPassageModel(**_fpt_kwargs(standardize_hazard="per_generation")).simulate(
        liability=L, seed=7, standardize="global", sex=sex, generation=g
    )
    assert not np.allclose(t_inherit, t_override)


def test_adult_cox_hazard_override_decouples():
    L, g = _heteroscedastic_two_gen()
    sex = np.zeros(len(L))
    t_inherit = AdultModel(**_adult_cox_kwargs(standardize_hazard=None)).simulate(
        liability=L, seed=11, standardize="global", sex=sex, generation=g
    )
    t_override = AdultModel(**_adult_cox_kwargs(standardize_hazard="per_generation")).simulate(
        liability=L, seed=11, standardize="global", sex=sex, generation=g
    )
    assert not np.allclose(t_inherit, t_override)


# ---------------------------------------------------------------------------
# cure_frailty's two knobs work independently
# ---------------------------------------------------------------------------


def test_cure_frailty_threshold_per_gen_hazard_global():
    """standardize='per_generation' (threshold) + standardize_hazard='global'
    (hazard) preserves per-gen prevalence AND uses one global hazard slope."""
    L, g = _heteroscedastic_two_gen(n_per=20_000)
    sex = np.zeros(len(L))
    t = CureFrailtyModel(**_cure_frailty_kwargs(standardize_hazard="global")).simulate(
        liability=L, seed=21, standardize="per_generation", sex=sex, generation=g
    )
    # Non-cases are flagged with t=1e6; per-gen case fraction ≈ K=0.10.
    is_case = t < 1e6
    for gi in (0, 1):
        gen_rate = is_case[g == gi].mean()
        assert abs(gen_rate - 0.10) < 0.02, f"gen {gi}: {gen_rate}"


def test_cure_frailty_both_per_generation_matches_inherit():
    """standardize='per_generation' with explicit standardize_hazard='per_generation'
    produces the same output as letting it inherit."""
    L, g = _heteroscedastic_two_gen()
    sex = np.zeros(len(L))
    t_inherit = CureFrailtyModel(**_cure_frailty_kwargs(standardize_hazard=None)).simulate(
        liability=L, seed=33, standardize="per_generation", sex=sex, generation=g
    )
    t_explicit = CureFrailtyModel(**_cure_frailty_kwargs(standardize_hazard="per_generation")).simulate(
        liability=L, seed=33, standardize="per_generation", sex=sex, generation=g
    )
    np.testing.assert_array_equal(t_inherit, t_explicit)


# ---------------------------------------------------------------------------
# Threshold-only models reject standardize_hazard
# ---------------------------------------------------------------------------


def test_adult_ltm_rejects_standardize_hazard():
    with pytest.raises(ValueError, match="standardize_hazard is only valid"):
        AdultModel(
            method="ltm",
            prevalence=0.1,
            cip_x0=50.0,
            cip_k=0.2,
            beta=1.0,
            beta_sex=0.0,
            standardize_hazard="per_generation",
        )


def test_adult_ltm_accepts_none_standardize_hazard():
    """The default None is allowed for LTM — it just means 'no override'."""
    m = AdultModel(
        method="ltm",
        prevalence=0.1,
        cip_x0=50.0,
        cip_k=0.2,
        beta=1.0,
        beta_sex=0.0,
        standardize_hazard=None,
    )
    assert m.standardize_hazard is None


# ---------------------------------------------------------------------------
# Legacy bool passthrough end-to-end
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("factory", "kwargs_fn"),
    [
        (FrailtyModel, _frailty_kwargs),
        (FirstPassageModel, _fpt_kwargs),
        (CureFrailtyModel, _cure_frailty_kwargs),
        (AdultModel, _adult_cox_kwargs),
    ],
)
def test_hazard_override_legacy_bool_passthrough(factory, kwargs_fn):
    """standardize_hazard=True/False on the model resolves to 'global'/'none'."""
    L, g = _heteroscedastic_two_gen(n_per=2000)
    sex = np.zeros(len(L))
    t_bool = factory(**kwargs_fn(standardize_hazard=True)).simulate(
        liability=L, seed=5, standardize="none", sex=sex, generation=g
    )
    t_str = factory(**kwargs_fn(standardize_hazard="global")).simulate(
        liability=L, seed=5, standardize="none", sex=sex, generation=g
    )
    np.testing.assert_array_equal(t_bool, t_str)


# ---------------------------------------------------------------------------
# Per-model self-consistency under heteroscedastic per-gen liability
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("factory", "kwargs_fn"),
    [
        (FrailtyModel, _frailty_kwargs),
        (FirstPassageModel, _fpt_kwargs),
        (AdultModel, _adult_cox_kwargs),
    ],
)
def test_per_generation_decouples_generations(factory, kwargs_fn):
    """Under per_generation, gen-1's onset distribution should not drag with
    gen-0's variance the way it does under global."""
    L, g = _heteroscedastic_two_gen(n_per=10_000)
    sex = np.zeros(len(L))
    model = factory(**kwargs_fn(standardize_hazard=None))
    t_global = model.simulate(liability=L, seed=1, standardize="global", sex=sex, generation=g)
    t_pergen = model.simulate(liability=L, seed=1, standardize="per_generation", sex=sex, generation=g)
    # Gen 1 (high-variance) sees a substantially different mean onset under
    # per_gen vs global, because per_gen rescales beta by std_g (≈ 3) instead
    # of the pooled std (≈ 2.2). The difference should be noticeable.
    finite = (t_global < 1e6) & (t_pergen < 1e6) & (g == 1)
    if finite.sum() > 100:
        assert not np.isclose(t_global[finite].mean(), t_pergen[finite].mean(), rtol=0.05)


def test_adult_ltm_per_generation_preserves_prevalence():
    """adult.ltm under standardize='per_generation' hits K per gen."""
    L, g = _heteroscedastic_two_gen(n_per=20_000)
    sex = np.zeros(len(L))
    m = AdultModel(method="ltm", prevalence=0.1, cip_x0=50.0, cip_k=0.2, beta=1.0)
    t = m.simulate(liability=L, seed=2, standardize="per_generation", sex=sex, generation=g)
    is_case = t < 1e6
    for gi in (0, 1):
        rate = is_case[g == gi].mean()
        assert abs(rate - 0.10) < 0.02, f"gen {gi}: {rate}"
