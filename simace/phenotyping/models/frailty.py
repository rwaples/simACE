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
    StandardizeMode,
    add_hazard_cli_args,
    add_standardize_hazard_cli_arg,
    coerce_standardize_mode,
    compute_event_times,
    hazard_cli_flag_attrs,
    iter_generation_groups,
    parse_hazard_cli,
    resolve_hazard_mode,
    standardize_beta,
    standardize_hazard_cli_attr,
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
        distribution:       baseline hazard name (see ``simace.phenotyping.hazards``).
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

    def __post_init__(self) -> None:
        validate_hazard_params(self.distribution, self.hazard_params, model_name="frailty")
        check_finite_beta(self.beta)
        if self.standardize_hazard is not None:
            coerce_standardize_mode(self.standardize_hazard)  # validates value

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
            standardize_hazard = phenotype_params.pop("standardize_hazard", None)
            return cls(
                distribution=distribution,
                hazard_params=phenotype_params,
                beta=params[f"beta{trait_num}"],
                beta_sex=params.get(f"beta_sex{trait_num}", 0.0),
                standardize_hazard=standardize_hazard,
            )

    @classmethod
    def add_cli_args(cls, parser: argparse.ArgumentParser, trait: int) -> None:
        group = add_hazard_cli_args(parser, trait, name="frailty")
        add_standardize_hazard_cli_arg(group, trait, name="frailty")

    @classmethod
    def from_cli(cls, args: argparse.Namespace, trait: int) -> Self:
        check_no_foreign_flags(cls, args, trait)
        with wrap_trait_error(trait):
            distribution, hazard_params = parse_hazard_cli(args, trait, name="frailty")
            return cls(
                distribution=distribution,
                hazard_params=hazard_params,
                beta=getattr(args, f"beta{trait}"),
                beta_sex=getattr(args, f"beta_sex{trait}", 0.0),
                standardize_hazard=getattr(args, standardize_hazard_cli_attr(trait, name="frailty")),
            )

    @classmethod
    def cli_flag_attrs(cls, trait: int) -> set[str]:
        return hazard_cli_flag_attrs(trait, name="frailty") | {standardize_hazard_cli_attr(trait, name="frailty")}

    def to_params_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"distribution": self.distribution, **self.hazard_params}
        if self.standardize_hazard is not None:
            out["standardize_hazard"] = self.standardize_hazard
        return out

    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------

    def simulate(
        self,
        liability: np.ndarray,
        *,
        seed: int,
        standardize: StandardizeMode | bool,
        sex: np.ndarray | None,
        generation: np.ndarray,
    ) -> np.ndarray:
        mode_haz = resolve_hazard_mode(standardize, self.standardize_hazard)
        rng = np.random.default_rng(seed)
        n = len(liability)
        neg_log_u = rng.exponential(size=n)
        if self.beta_sex != 0.0 and sex is not None:
            neg_log_u = neg_log_u / np.exp(self.beta_sex * sex)
        mean_arr, beta_arr = standardize_beta(liability, self.beta, mode_haz, generation)
        t = np.empty(n)
        for mask in iter_generation_groups(mode_haz, generation):
            m = float(mean_arr[mask][0])
            b = float(beta_arr[mask][0])
            t[mask] = compute_event_times(neg_log_u[mask], liability[mask], m, b, self.distribution, self.hazard_params)
        return t
