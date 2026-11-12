"""Mixture cure-frailty phenotype model.

Liability above a threshold sets case status (WHO has the disease); a
proportional hazards frailty model with a configurable baseline hazard
determines onset time among cases (WHEN). Controls are censored at 1e6.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar, Self

import numpy as np

from simace.phenotype.hazards import (
    BASELINE_HAZARDS,
    coerce_standardize_mode,
    compute_event_times,
    hazard_config_keys,
    iter_generation_groups,
    resolve_hazard_mode,
    standardize_beta,
    validate_hazard_params,
)
from simace.phenotype.models._base import (
    PhenotypeModel,
    check_finite_beta,
    validate_standardize_hazard,
)
from simace.phenotype.models._prevalence import case_status_from_liability, validate_prevalence_spec

if TYPE_CHECKING:
    from simace.phenotype.hazards import StandardizeMode

__all__ = ["CureFrailtyModel"]


@dataclass(frozen=True)
class CureFrailtyModel(PhenotypeModel):
    """Mixture cure-frailty model.

    Parameters:
        distribution:       baseline hazard for cases (see ``simace.phenotype.hazards``).
        hazard_params:      dict of distribution-specific parameter values.
        prevalence:         case fraction; scalar, per-generation dict, or sex-specific dict.
        beta:               coefficient on liability in the log-hazard among cases.
        beta_sex:           coefficient on sex in the log-hazard (0 = no effect).
        standardize_hazard: per-trait override for the *case-onset hazard* step.
            ``None`` inherits from the global ``standardize`` flag passed to
            ``simulate``.  This is the only model that honors both knobs:
            ``standardize`` controls the threshold step (case status), while
            ``standardize_hazard`` controls the hazard step (case onset). Setting
            them to different modes (e.g. ``standardize='per_generation'`` +
            ``standardize_hazard='global'``) preserves per-gen prevalence while
            keeping a constant hazard slope across generations.
    """

    distribution: str
    prevalence: Any
    hazard_params: dict[str, float] = field(default_factory=dict)
    beta: float = 1.0
    beta_sex: float = 0.0
    standardize_hazard: StandardizeMode | None = None

    name: ClassVar[str] = "cure_frailty"
    config_keys: ClassVar[frozenset[str]] = frozenset({"distribution", "standardize_hazard", "prevalence"})

    @classmethod
    def _accepted_keys(cls, phenotype_params: dict[str, Any]) -> frozenset[str]:
        return cls.config_keys | hazard_config_keys(phenotype_params.get("distribution"))

    def __post_init__(self) -> None:
        validate_hazard_params(self.distribution, self.hazard_params, model_name="cure_frailty")
        validate_prevalence_spec(self.prevalence)
        check_finite_beta(self.beta)
        validate_standardize_hazard(self.standardize_hazard)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def _from_config(cls, phenotype_params: dict[str, Any], trait_num: int, *, beta: float, beta_sex: float) -> Self:
        distribution = phenotype_params.pop("distribution", None)
        if distribution is None:
            raise ValueError(
                f"phenotype_params{trait_num} for model 'cure_frailty' must include "
                f"'distribution' key (one of {sorted(BASELINE_HAZARDS)})"
            )
        if "prevalence" not in phenotype_params:
            raise ValueError(f"phenotype_params{trait_num} for model 'cure_frailty' must include 'prevalence' key")
        prevalence = phenotype_params.pop("prevalence")
        standardize_hazard = phenotype_params.pop("standardize_hazard", None)
        return cls(
            distribution=distribution,
            hazard_params=phenotype_params,
            prevalence=prevalence,
            beta=beta,
            beta_sex=beta_sex,
            standardize_hazard=standardize_hazard,
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
        # Threshold step: uses the global ``standardize`` mode for liability.
        mode_thr = coerce_standardize_mode(standardize)
        is_case = case_status_from_liability(liability, self.prevalence, sex, generation, mode_thr)

        n = len(liability)
        t = np.full(n, 1e6)
        n_cases = int(is_case.sum())
        if n_cases == 0:
            return t

        # Hazard step: uses the per-trait override (or inherits from ``standardize``).
        mode_haz = resolve_hazard_mode(standardize, self.standardize_hazard)
        mean_arr, beta_arr = standardize_beta(liability, self.beta, mode_haz, generation)

        rng = np.random.default_rng(seed)
        neg_log_u_full = np.zeros(n)
        neg_log_u_full[is_case] = rng.exponential(size=n_cases)
        if self.beta_sex != 0.0:
            neg_log_u_full[is_case] = neg_log_u_full[is_case] / np.exp(self.beta_sex * sex[is_case])

        for mask in iter_generation_groups(mode_haz, generation):
            cell = mask & is_case
            if not cell.any():
                continue
            m = float(mean_arr[cell][0])
            b = float(beta_arr[cell][0])
            # Pass raw ``liability`` here, not the standardized liability the
            # threshold step uses: the kernel does ``exp(scaled_beta * (L - mean))``
            # and double-shifting silently biases the log-hazard under
            # non-unit-variance liability.
            t[cell] = compute_event_times(
                neg_log_u_full[cell],
                liability[cell],
                m,
                b,
                self.distribution,
                self.hazard_params,
            )
        return t
