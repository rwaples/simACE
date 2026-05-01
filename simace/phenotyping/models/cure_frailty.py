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
        distribution:       baseline hazard for cases (see ``simace.phenotyping.hazards``).
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

    def __post_init__(self) -> None:
        validate_hazard_params(self.distribution, self.hazard_params, model_name="cure_frailty")
        check_finite_beta(self.beta)
        if self.standardize_hazard is not None:
            coerce_standardize_mode(self.standardize_hazard)

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
                raise ValueError(f"phenotype_params{trait_num} for model 'cure_frailty' must include 'prevalence' key")
            prevalence = phenotype_params.pop("prevalence")
            standardize_hazard = phenotype_params.pop("standardize_hazard", None)
            return cls(
                distribution=distribution,
                hazard_params=phenotype_params,
                prevalence=prevalence,
                beta=params[f"beta{trait_num}"],
                beta_sex=params.get(f"beta_sex{trait_num}", 0.0),
                standardize_hazard=standardize_hazard,
            )

    @classmethod
    def add_cli_args(cls, parser: argparse.ArgumentParser, trait: int) -> None:
        group = add_hazard_cli_args(parser, trait, name="cure-frailty")
        group.add_argument(f"--cure-frailty-prevalence{trait}", type=float, default=None)
        add_standardize_hazard_cli_arg(group, trait, name="cure-frailty")

    @classmethod
    def from_cli(cls, args: argparse.Namespace, trait: int) -> Self:
        check_no_foreign_flags(cls, args, trait)
        with wrap_trait_error(trait):
            distribution, hazard_params = parse_hazard_cli(args, trait, name="cure-frailty")
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
                standardize_hazard=getattr(args, standardize_hazard_cli_attr(trait, name="cure-frailty")),
            )

    @classmethod
    def cli_flag_attrs(cls, trait: int) -> set[str]:
        return hazard_cli_flag_attrs(trait, name="cure-frailty") | {
            f"cure_frailty_prevalence{trait}",
            standardize_hazard_cli_attr(trait, name="cure-frailty"),
        }

    def to_params_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "distribution": self.distribution,
            "prevalence": self.prevalence,
            **self.hazard_params,
        }
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
        # Threshold step: uses the global ``standardize`` mode for liability.
        mode_thr = coerce_standardize_mode(standardize)
        L = standardize_liability(liability, mode_thr, generation)
        prevalence = resolve_prevalence(self.prevalence, sex, generation)
        threshold = ndtri(1.0 - np.asarray(prevalence))
        is_case = threshold < L

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
        if self.beta_sex != 0.0 and sex is not None:
            neg_log_u_full[is_case] = neg_log_u_full[is_case] / np.exp(self.beta_sex * sex[is_case])

        for mask in iter_generation_groups(mode_haz, generation):
            cell = mask & is_case
            if not cell.any():
                continue
            m = float(mean_arr[cell][0])
            b = float(beta_arr[cell][0])
            # Pass RAW liability (not standardized L) so the kernel's
            # ``z = exp(scaled_beta * (L - mean))`` evaluates to
            # ``exp(beta * z_score(L_raw))`` as intended.  Passing L (the
            # threshold-standardized liability) here would double-shift,
            # introducing a constant ``-beta * mean / std`` offset in the
            # log-hazard; silent under default A+C+E=1 configs but real
            # under non-zero-mean or non-unit-variance liability.
            t[cell] = compute_event_times(
                neg_log_u_full[cell],
                liability[cell],
                m,
                b,
                self.distribution,
                self.hazard_params,
            )
        return t
