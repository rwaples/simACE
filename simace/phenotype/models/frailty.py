"""Proportional-hazards frailty phenotype model.

Per-individual onset time is drawn from a parametric baseline hazard
modulated by an exp(beta * L) frailty multiplier. Liability translates
into earlier onset for higher beta * L.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar, Self

import numpy as np

from simace.phenotype.hazards import (
    BASELINE_HAZARDS,
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

if TYPE_CHECKING:
    from simace.phenotype.hazards import StandardizeMode

__all__ = ["FrailtyModel"]


@dataclass(frozen=True)
class FrailtyModel(PhenotypeModel):
    """Frailty phenotype model.

    Parameters:
        distribution:       baseline hazard name (see ``simace.phenotype.hazards``).
        hazard_params:      dict of distribution-specific parameter values.
        beta:               coefficient on liability in the log-hazard.
        beta_sex:           coefficient on sex in the log-hazard (0 = no effect).
        standardize_hazard: per-trait override for the hazard-step standardization
            mode (``"none" | "global" | "per_generation"``).  ``None`` inherits
            from the global ``standardize`` flag passed to ``simulate``.
    """

    distribution: str
    hazard_params: dict[str, float] = field(default_factory=dict)
    beta: float = 1.0
    beta_sex: float = 0.0
    standardize_hazard: StandardizeMode | None = None

    name: ClassVar[str] = "frailty"
    config_keys: ClassVar[frozenset[str]] = frozenset({"distribution", "standardize_hazard"})

    @classmethod
    def _accepted_keys(cls, phenotype_params: dict[str, Any]) -> frozenset[str]:
        return cls.config_keys | hazard_config_keys(phenotype_params.get("distribution"))

    def __post_init__(self) -> None:
        validate_hazard_params(self.distribution, self.hazard_params, model_name="frailty")
        check_finite_beta(self.beta)
        validate_standardize_hazard(self.standardize_hazard)

    # ------------------------------------------------------------------
    # Construction from config dict / CLI args
    # ------------------------------------------------------------------

    @classmethod
    def _from_config(cls, phenotype_params: dict[str, Any], trait_num: int, *, beta: float, beta_sex: float) -> Self:
        distribution = phenotype_params.pop("distribution", None)
        if distribution is None:
            raise ValueError(
                f"phenotype_params{trait_num} for model 'frailty' must include "
                f"'distribution' key (one of {sorted(BASELINE_HAZARDS)})"
            )
        standardize_hazard = phenotype_params.pop("standardize_hazard", None)
        return cls(
            distribution=distribution,
            hazard_params=phenotype_params,
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
        mode_haz = resolve_hazard_mode(standardize, self.standardize_hazard)
        rng = np.random.default_rng(seed)
        n = len(liability)
        neg_log_u = rng.exponential(size=n)
        if self.beta_sex != 0.0:
            neg_log_u = neg_log_u / np.exp(self.beta_sex * sex)
        mean_arr, beta_arr = standardize_beta(liability, self.beta, mode_haz, generation)
        t = np.empty(n)
        for mask in iter_generation_groups(mode_haz, generation):
            m = float(mean_arr[mask][0])
            b = float(beta_arr[mask][0])
            t[mask] = compute_event_times(neg_log_u[mask], liability[mask], m, b, self.distribution, self.hazard_params)
        return t
