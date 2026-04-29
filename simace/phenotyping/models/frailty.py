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
    BASELINE_PARAMS,
    compute_event_times,
    standardize_beta,
)
from simace.phenotyping.models._base import (
    PhenotypeModel,
    check_no_foreign_flags,
    wrap_trait_error,
)

if TYPE_CHECKING:
    import argparse

__all__ = ["HAZARD_FLAG_ROOTS", "FrailtyModel"]


# Hazard-parameter flag roots, shared by FrailtyModel and CureFrailtyModel.
# Covers every key that any baseline distribution may require (see
# ``simace.phenotyping.hazards.BASELINE_PARAMS``); plus exponential's
# alternate ``scale``.
HAZARD_FLAG_ROOTS: tuple[str, ...] = ("scale", "rho", "rate", "gamma", "mu", "sigma", "shape")


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
        if self.distribution not in BASELINE_HAZARDS:
            raise ValueError(f"unknown frailty distribution {self.distribution!r}; valid: {sorted(BASELINE_HAZARDS)}")
        required = set(BASELINE_PARAMS[self.distribution])
        # Exponential accepts either "rate" or "scale".
        if self.distribution == "exponential" and "scale" in self.hazard_params:
            required = (required - {"rate"}) | {"scale"}
        missing = required - set(self.hazard_params)
        if missing:
            raise ValueError(
                f"frailty distribution {self.distribution!r} missing required hazard params: {sorted(missing)}"
            )
        if not np.isfinite(self.beta):
            raise ValueError(f"beta must be finite, got {self.beta}")

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
        group = parser.add_argument_group(f"Trait {trait} — frailty")
        group.add_argument(
            f"--frailty-distribution{trait}",
            default=None,
            choices=sorted(BASELINE_HAZARDS),
            help=f"Baseline hazard for trait {trait} when phenotype-model{trait}=frailty",
        )
        for flag_root in HAZARD_FLAG_ROOTS:
            group.add_argument(f"--frailty-{flag_root}{trait}", type=float, default=None)

    @classmethod
    def from_cli(cls, args: argparse.Namespace, trait: int) -> Self:
        check_no_foreign_flags(cls, args, trait)
        with wrap_trait_error(trait):
            distribution = getattr(args, f"frailty_distribution{trait}")
            if distribution is None:
                raise ValueError(f"--frailty-distribution{trait} is required when --phenotype-model{trait}=frailty")
            required = list(BASELINE_PARAMS[distribution])
            hazard_params: dict[str, float] = {}
            for key in required:
                val = getattr(args, f"frailty_{key}{trait}", None)
                if val is None:
                    raise ValueError(
                        f"--frailty-{key}{trait} is required for --frailty-distribution{trait}={distribution}"
                    )
                hazard_params[key] = val
            return cls(
                distribution=distribution,
                hazard_params=hazard_params,
                beta=getattr(args, f"beta{trait}"),
                beta_sex=getattr(args, f"beta_sex{trait}", 0.0),
            )

    @classmethod
    def cli_flag_attrs(cls, trait: int) -> set[str]:
        attrs = {f"frailty_distribution{trait}"}
        attrs.update(f"frailty_{root}{trait}" for root in HAZARD_FLAG_ROOTS)
        return attrs

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
