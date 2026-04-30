"""Proportional-hazards frailty phenotype model.

Per-individual onset time is drawn from a parametric baseline hazard
modulated by an exp(beta * L) frailty multiplier. Liability translates
into earlier onset for higher beta * L.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar, Self

import numpy as np

from simace.phenotyping.hazards import (
    BASELINE_HAZARDS,
    add_hazard_cli_args,
    compute_event_times,
    hazard_cli_flag_attrs,
    parse_hazard_cli,
    standardize_beta,
    validate_hazard_params,
)
from simace.phenotyping.models._base import (
    PhenotypeModel,
    check_finite_beta,
    check_no_foreign_flags,
    wrap_trait_error,
)

if TYPE_CHECKING:
    import argparse

__all__ = ["FrailtyModel"]


@dataclass(frozen=True)
class FrailtyModel(PhenotypeModel):
    """Frailty phenotype model.

    Parameters:
        distribution:  baseline hazard name (see ``simace.phenotyping.hazards``).
        hazard_params: dict of distribution-specific parameter values.
        beta:          coefficient on liability in the log-hazard.
        beta_sex:      coefficient on sex in the log-hazard (0 = no effect).
    """

    distribution: str
    hazard_params: dict[str, float] = field(default_factory=dict)
    beta: float = 1.0
    beta_sex: float = 0.0

    name: ClassVar[str] = "frailty"

    def __post_init__(self) -> None:
        validate_hazard_params(self.distribution, self.hazard_params, model_name="frailty")
        check_finite_beta(self.beta)

    # ------------------------------------------------------------------
    # Construction from config dict / CLI args
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, params: dict[str, Any], trait_num: int) -> Self:
        with wrap_trait_error(trait_num):
            phenotype_params = dict(params.get(f"phenotype_params{trait_num}", {}))
            distribution = phenotype_params.pop("distribution", None)
            if distribution is None:
                raise ValueError(
                    f"phenotype_params{trait_num} for model 'frailty' must include "
                    f"'distribution' key (one of {sorted(BASELINE_HAZARDS)})"
                )
            return cls(
                distribution=distribution,
                hazard_params=phenotype_params,
                beta=params[f"beta{trait_num}"],
                beta_sex=params.get(f"beta_sex{trait_num}", 0.0),
            )

    @classmethod
    def add_cli_args(cls, parser: argparse.ArgumentParser, trait: int) -> None:
        add_hazard_cli_args(parser, trait, kebab_prefix="frailty", model_label="frailty")

    @classmethod
    def from_cli(cls, args: argparse.Namespace, trait: int) -> Self:
        check_no_foreign_flags(cls, args, trait)
        with wrap_trait_error(trait):
            distribution, hazard_params = parse_hazard_cli(
                args, trait, attr_prefix="frailty", kebab_prefix="frailty"
            )
            return cls(
                distribution=distribution,
                hazard_params=hazard_params,
                beta=getattr(args, f"beta{trait}"),
                beta_sex=getattr(args, f"beta_sex{trait}", 0.0),
            )

    @classmethod
    def cli_flag_attrs(cls, trait: int) -> set[str]:
        return hazard_cli_flag_attrs(trait, attr_prefix="frailty")

    def to_params_dict(self) -> dict[str, Any]:
        return {"distribution": self.distribution, **self.hazard_params}

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
        rng = np.random.default_rng(seed)
        mean, scaled_beta = standardize_beta(liability, self.beta, standardize)
        neg_log_u = rng.exponential(size=len(liability))
        if self.beta_sex != 0.0 and sex is not None:
            neg_log_u = neg_log_u / np.exp(self.beta_sex * sex)
        return compute_event_times(neg_log_u, liability, mean, scaled_beta, self.distribution, self.hazard_params)
