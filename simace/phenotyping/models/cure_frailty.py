"""Mixture cure-frailty phenotype model.

Liability above a threshold sets case status (WHO has the disease); a
proportional hazards frailty model with a configurable baseline hazard
determines onset time among cases (WHEN). Controls are censored at 1e6.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar, Self

import numpy as np
from scipy.special import ndtri

from simace.phenotyping.hazards import (
    BASELINE_HAZARDS,
    add_hazard_cli_args,
    compute_event_times,
    hazard_cli_flag_attrs,
    parse_hazard_cli,
    standardize_beta,
    standardize_liability,
    validate_hazard_params,
)
from simace.phenotyping.models._base import (
    PhenotypeModel,
    check_finite_beta,
    check_no_foreign_flags,
    wrap_trait_error,
)
from simace.phenotyping.models._prevalence import resolve_prevalence

if TYPE_CHECKING:
    import argparse

__all__ = ["CureFrailtyModel"]


@dataclass(frozen=True)
class CureFrailtyModel(PhenotypeModel):
    """Mixture cure-frailty model.

    Parameters:
        distribution:  baseline hazard for cases (see ``simace.phenotyping.hazards``).
        hazard_params: dict of distribution-specific parameter values.
        prevalence:    case fraction; scalar, per-generation dict, or sex-specific dict.
        beta:          coefficient on liability in the log-hazard among cases.
        beta_sex:      coefficient on sex in the log-hazard (0 = no effect).
    """

    distribution: str
    prevalence: Any
    hazard_params: dict[str, float] = field(default_factory=dict)
    beta: float = 1.0
    beta_sex: float = 0.0

    name: ClassVar[str] = "cure_frailty"

    def __post_init__(self) -> None:
        validate_hazard_params(self.distribution, self.hazard_params, model_name="cure_frailty")
        check_finite_beta(self.beta)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, params: dict[str, Any], trait_num: int) -> Self:
        with wrap_trait_error(trait_num):
            phenotype_params = dict(params.get(f"phenotype_params{trait_num}", {}))
            distribution = phenotype_params.pop("distribution", None)
            if distribution is None:
                raise ValueError(
                    f"phenotype_params{trait_num} for model 'cure_frailty' must include "
                    f"'distribution' key (one of {sorted(BASELINE_HAZARDS)})"
                )
            if "prevalence" not in phenotype_params:
                raise ValueError(
                    f"phenotype_params{trait_num} for model 'cure_frailty' must include 'prevalence' key"
                )
            prevalence = phenotype_params.pop("prevalence")
            return cls(
                distribution=distribution,
                hazard_params=phenotype_params,
                prevalence=prevalence,
                beta=params[f"beta{trait_num}"],
                beta_sex=params.get(f"beta_sex{trait_num}", 0.0),
            )

    @classmethod
    def add_cli_args(cls, parser: argparse.ArgumentParser, trait: int) -> None:
        group = add_hazard_cli_args(parser, trait, kebab_prefix="cure-frailty", model_label="cure_frailty")
        group.add_argument(f"--cure-frailty-prevalence{trait}", type=float, default=None)

    @classmethod
    def from_cli(cls, args: argparse.Namespace, trait: int) -> Self:
        check_no_foreign_flags(cls, args, trait)
        with wrap_trait_error(trait):
            distribution, hazard_params = parse_hazard_cli(
                args, trait, attr_prefix="cure_frailty", kebab_prefix="cure-frailty"
            )
            prevalence = getattr(args, f"cure_frailty_prevalence{trait}")
            if prevalence is None:
                raise ValueError(
                    f"--cure-frailty-prevalence{trait} is required when --phenotype-model{trait}=cure_frailty"
                )
            return cls(
                distribution=distribution,
                hazard_params=hazard_params,
                prevalence=prevalence,
                beta=getattr(args, f"beta{trait}"),
                beta_sex=getattr(args, f"beta_sex{trait}", 0.0),
            )

    @classmethod
    def cli_flag_attrs(cls, trait: int) -> set[str]:
        return hazard_cli_flag_attrs(trait, attr_prefix="cure_frailty") | {f"cure_frailty_prevalence{trait}"}

    def to_params_dict(self) -> dict[str, Any]:
        return {"distribution": self.distribution, "prevalence": self.prevalence, **self.hazard_params}

    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------

    def simulate(
        self,
        liability: np.ndarray,
        *,
        seed: int,
        standardize: bool,
        sex: np.ndarray | None,
        generation: np.ndarray,
    ) -> np.ndarray:
        n = len(liability)
        prevalence = resolve_prevalence(self.prevalence, sex, generation)

        mean, scaled_beta = standardize_beta(liability, self.beta, standardize)

        L = standardize_liability(liability) if standardize else liability

        threshold = ndtri(1.0 - np.asarray(prevalence))
        is_case = threshold < L

        t = np.full(n, 1e6)
        n_cases = is_case.sum()
        if n_cases > 0:
            rng = np.random.default_rng(seed)
            neg_log_u = rng.exponential(size=n_cases)
            if self.beta_sex != 0.0 and sex is not None:
                neg_log_u = neg_log_u / np.exp(self.beta_sex * sex[is_case])
            t[is_case] = compute_event_times(
                neg_log_u,
                L[is_case],
                mean,
                scaled_beta,
                self.distribution,
                self.hazard_params,
            )
        return t
