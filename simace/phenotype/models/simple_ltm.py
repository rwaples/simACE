"""Simple liability-threshold phenotype model.

Liability above a probit threshold sets case status (WHO has the disease) at a
target prevalence ``K``; a small ``onset`` sub-model assigns an age-of-onset
(WHEN) to cases.  Unlike ``cure_frailty`` the onset is independent of liability:

* ``onset.kind = "fixed"`` — every case onsets at the same ``age``.
* ``onset.kind = "normal"`` — case onset ~ ``Normal(mean, sd)``.

Controls are censored at ``1e6``.  Onset times flow through the standard censor
stage like every other model, so the *observed* affected rate after age-window
and death censoring is below ``K``.

Case onsets are clipped to ``[0.01, 1e6]``.  With realistic ``onset`` parameters
they never reach the ``1e6`` control sentinel, but a pathological ``normal``
onset (e.g. ``sd`` orders of magnitude beyond a human lifespan) can clip a case
onto ``1e6`` and so be miscounted as a never-affected control.  Keep ``normal``
onset on a human-lifespan scale to preserve the case/control distinction.

This is a threshold-on-liability model: it consumes the global ``standardize``
flag directly (like ``adult.ltm``) and does not accept ``standardize_hazard``.
``beta`` / ``beta_sex`` are parsed for uniformity with the other models but are
unused — neither case status nor onset depends on them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar, Self

import numpy as np

from simace.core.phenotype_keys import ONSET_KINDS
from simace.phenotype.hazards import coerce_standardize_mode
from simace.phenotype.models._base import (
    PhenotypeModel,
    check_finite_beta,
)
from simace.phenotype.models._prevalence import case_status_from_liability, validate_prevalence_spec

if TYPE_CHECKING:
    from simace.phenotype.hazards import StandardizeMode

__all__ = ["SimpleLtmModel"]


def _validate_onset(onset: dict[str, Any]) -> None:
    """Validate the ``onset`` sub-model dict (discriminated on ``kind``)."""
    kind = onset.get("kind")
    if kind not in ONSET_KINDS:
        raise ValueError(f"simple_ltm onset.kind must be one of {sorted(ONSET_KINDS)}; got {kind!r}")
    if kind == "fixed":
        if "age" not in onset:
            raise ValueError("simple_ltm onset.kind='fixed' requires 'age'")
    else:  # normal
        missing = {"mean", "sd"} - set(onset)
        if missing:
            raise ValueError(f"simple_ltm onset.kind='normal' requires {sorted(missing)}")
        if float(onset["sd"]) <= 0.0:
            raise ValueError(f"simple_ltm onset.sd must be > 0, got {onset['sd']}")


def _draw_onset(onset: dict[str, Any], n_cases: int, rng: np.random.Generator) -> np.ndarray:
    """Draw ``n_cases`` onset ages from the configured onset sub-model."""
    if onset["kind"] == "fixed":
        return np.full(n_cases, float(onset["age"]))
    return rng.normal(float(onset["mean"]), float(onset["sd"]), size=n_cases)


@dataclass(frozen=True)
class SimpleLtmModel(PhenotypeModel):
    """Simple liability-threshold model with a fixed or normal age-of-onset.

    Parameters:
        prevalence: case fraction ``K`` — scalar, per-generation dict, or
            sex-specific ``{"female": ..., "male": ...}`` dict.
        onset:      onset sub-model dict. ``{"kind": "fixed", "age": A}`` or
            ``{"kind": "normal", "mean": M, "sd": S}``.
        beta:       parsed for uniformity; **unused** (onset is independent of L).
        beta_sex:   parsed for uniformity; **unused**.
    """

    prevalence: Any
    onset: dict[str, Any] = field(default_factory=dict)
    beta: float = 1.0
    beta_sex: float = 0.0

    name: ClassVar[str] = "simple_ltm"
    config_keys: ClassVar[frozenset[str]] = frozenset({"prevalence", "onset"})

    def __post_init__(self) -> None:
        check_finite_beta(self.beta)
        validate_prevalence_spec(self.prevalence)
        _validate_onset(self.onset)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def _from_config(cls, phenotype_params: dict[str, Any], trait_num: int, *, beta: float, beta_sex: float) -> Self:
        if "prevalence" not in phenotype_params:
            raise ValueError(f"phenotype_params{trait_num} for model 'simple_ltm' must include 'prevalence' key")
        if "onset" not in phenotype_params:
            raise ValueError(
                f"phenotype_params{trait_num} for model 'simple_ltm' must include 'onset' key "
                f"(e.g. {{'kind': 'fixed', 'age': 30}} or {{'kind': 'normal', 'mean': 30, 'sd': 8}})"
            )
        onset = phenotype_params["onset"]
        if not isinstance(onset, dict):
            raise ValueError(
                f"phenotype_params{trait_num} for model 'simple_ltm': 'onset' must be a dict, got {onset!r} "
                f"(e.g. {{'kind': 'fixed', 'age': 30}} or {{'kind': 'normal', 'mean': 30, 'sd': 8}})"
            )
        return cls(
            prevalence=phenotype_params["prevalence"],
            onset=dict(onset),
            beta=beta,
            beta_sex=beta_sex,
        )

    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------

    def simulate(
        self,
        liability: np.ndarray,
        *,
        seed: int,
        standardize: StandardizeMode | bool,
        sex: np.ndarray,
        generation: np.ndarray,
    ) -> np.ndarray:
        mode = coerce_standardize_mode(standardize)
        is_case = case_status_from_liability(liability, self.prevalence, sex, generation, mode)

        t = np.full(len(liability), 1e6)
        n_cases = int(is_case.sum())
        if n_cases > 0:
            rng = np.random.default_rng(seed)
            t[is_case] = _draw_onset(self.onset, n_cases, rng)

        np.clip(t, 0.01, 1e6, out=t)
        return t
