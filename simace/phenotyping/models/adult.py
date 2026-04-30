"""ADuLT phenotype model.

Two sub-methods, selected by ``method``:

* ``ltm`` — liability threshold model: case status from raw liability vs.
  threshold; case onset age via logistic-CIP transform of the case CIR
  computed from a probit scaling of liability + sex.
* ``cox`` — Weibull(shape=2) proportional hazards: case status by
  rank/(N+1) capped at K; case onset age via logistic CIP inverse.

Both share the CIP shape parameters ``cip_x0`` and ``cip_k``. Reference:
Pedersen et al., Nat Commun 2023 (ADuLT).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, Self

import numpy as np
from scipy.special import erfc, ndtri

from simace.phenotyping.models._base import (
    PhenotypeModel,
    check_no_foreign_flags,
    wrap_trait_error,
)
from simace.phenotyping.models._prevalence import resolve_prevalence

if TYPE_CHECKING:
    import argparse

__all__ = ["AdultModel"]


_ADULT_METHODS: frozenset[str] = frozenset({"ltm", "cox"})


@dataclass(frozen=True)
class AdultModel(PhenotypeModel):
    """ADuLT phenotype model.

    Parameters:
        method:     ``"ltm"`` or ``"cox"``.
        prevalence: case fraction K (scalar, per-generation, or sex-specific).
        cip_x0:     logistic CIP midpoint age (default 50.0).
        cip_k:      logistic CIP growth rate (default 0.2).
        beta:       liability scaling factor on the probit (ltm) or log-hazard
                    (cox) scale (1.0 = no scaling).
        beta_sex:   sex coefficient (0.0 = no effect).
    """

    method: str
    prevalence: Any
    cip_x0: float = 50.0
    cip_k: float = 0.2
    beta: float = 1.0
    beta_sex: float = 0.0

    name: ClassVar[str] = "adult"

    def __post_init__(self) -> None:
        if self.method not in _ADULT_METHODS:
            raise ValueError(f"unknown adult method {self.method!r}; valid: {sorted(_ADULT_METHODS)}")
        if not np.isfinite(self.beta):
            raise ValueError(f"beta must be finite, got {self.beta}")

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, params: dict[str, Any], trait_num: int) -> Self:
        with wrap_trait_error(trait_num):
            phenotype_params = dict(params.get(f"phenotype_params{trait_num}", {}))
            method = phenotype_params.get("method")
            if method is None:
                raise ValueError(
                    f"phenotype_params{trait_num} for model 'adult' must include "
                    f"'method' key (one of {sorted(_ADULT_METHODS)})"
                )
            if "prevalence" not in phenotype_params:
                raise ValueError(
                    f"phenotype_params{trait_num} for model 'adult' must include 'prevalence' key"
                )
            return cls(
                method=method,
                prevalence=phenotype_params["prevalence"],
                cip_x0=phenotype_params.get("cip_x0", 50.0),
                cip_k=phenotype_params.get("cip_k", 0.2),
                beta=params[f"beta{trait_num}"],
                beta_sex=params.get(f"beta_sex{trait_num}", 0.0),
            )

    @classmethod
    def add_cli_args(cls, parser: argparse.ArgumentParser, trait: int) -> None:
        group = parser.add_argument_group(f"Trait {trait} — adult")
        group.add_argument(
            f"--adult-method{trait}",
            default=None,
            choices=sorted(_ADULT_METHODS),
            help=f"Adult sub-method for trait {trait}",
        )
        group.add_argument(f"--adult-cip-x0-{trait}", type=float, default=None)
        group.add_argument(f"--adult-cip-k-{trait}", type=float, default=None)
        group.add_argument(f"--adult-prevalence{trait}", type=float, default=None)

    @classmethod
    def from_cli(cls, args: argparse.Namespace, trait: int) -> Self:
        check_no_foreign_flags(cls, args, trait)
        with wrap_trait_error(trait):
            method = getattr(args, f"adult_method{trait}")
            if method is None:
                raise ValueError(f"--adult-method{trait} is required when --phenotype-model{trait}=adult")
            prevalence = getattr(args, f"adult_prevalence{trait}")
            if prevalence is None:
                raise ValueError(f"--adult-prevalence{trait} is required when --phenotype-model{trait}=adult")
            cip_x0 = getattr(args, f"adult_cip_x0_{trait}")
            cip_k = getattr(args, f"adult_cip_k_{trait}")
            return cls(
                method=method,
                prevalence=prevalence,
                cip_x0=cip_x0 if cip_x0 is not None else 50.0,
                cip_k=cip_k if cip_k is not None else 0.2,
                beta=getattr(args, f"beta{trait}"),
                beta_sex=getattr(args, f"beta_sex{trait}", 0.0),
            )

    @classmethod
    def cli_flag_attrs(cls, trait: int) -> set[str]:
        return {
            f"adult_method{trait}",
            f"adult_cip_x0_{trait}",
            f"adult_cip_k_{trait}",
            f"adult_prevalence{trait}",
        }

    def to_params_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "cip_x0": self.cip_x0,
            "cip_k": self.cip_k,
            "prevalence": self.prevalence,
        }

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
        prevalence = resolve_prevalence(self.prevalence, sex, generation)
        if self.method == "ltm":
            return self._simulate_ltm(liability, prevalence, sex, standardize)
        return self._simulate_cox(liability, prevalence, sex, seed, standardize)

    def _simulate_ltm(
        self,
        liability: np.ndarray,
        prevalence: float | np.ndarray,
        sex: np.ndarray | None,
        standardize: bool,
    ) -> np.ndarray:
        L = liability
        if standardize:
            std = np.std(L)
            if std > 0:
                L = (L - L.mean()) / std

        threshold = ndtri(1.0 - np.asarray(prevalence))
        is_case = threshold < L

        t = np.full(len(L), 1e6)
        n_cases = is_case.sum()
        if n_cases > 0:
            prev_case = prevalence[is_case] if isinstance(prevalence, np.ndarray) else prevalence
            L_eff = self.beta * L[is_case]
            if self.beta_sex != 0.0 and sex is not None:
                L_eff = L_eff + self.beta_sex * sex[is_case]
            cir = 0.5 * erfc(L_eff / np.sqrt(2.0))
            valid = cir < prev_case
            cir = np.clip(cir, 1e-10, np.asarray(prev_case) - 1e-10)
            onset = self.cip_x0 + (1.0 / self.cip_k) * np.log(cir / (prev_case - cir))
            onset[~valid] = 1e6
            t[is_case] = onset

        np.clip(t, 0.01, 1e6, out=t)
        return t

    def _simulate_cox(
        self,
        liability: np.ndarray,
        prevalence: float | np.ndarray,
        sex: np.ndarray | None,
        seed: int,
        standardize: bool,
    ) -> np.ndarray:
        rng = np.random.default_rng(seed)
        L = liability
        if standardize:
            std = np.std(L)
            if std > 0:
                L = (L - L.mean()) / std

        n = len(L)
        neg_log_u = rng.exponential(size=n)
        if self.beta_sex != 0.0 and sex is not None:
            neg_log_u = neg_log_u / np.exp(self.beta_sex * sex)
        t_raw = np.sqrt(neg_log_u / np.exp(self.beta * L))

        t = np.full(n, 1e6)

        if isinstance(prevalence, np.ndarray):
            for grp_prev in np.unique(prevalence):
                mask = prevalence == grp_prev
                idx = np.where(mask)[0]
                n_grp = mask.sum()
                if n_grp == 0:
                    continue
                grp_order = np.argsort(t_raw[mask])
                cip = (np.arange(1, n_grp + 1)) / (n_grp + 1)
                is_case = cip < grp_prev
                case_cip = cip[is_case]
                case_age = self.cip_x0 + (1.0 / self.cip_k) * np.log(case_cip / (grp_prev - case_cip))
                t[idx[grp_order[is_case]]] = case_age
        else:
            order = np.argsort(t_raw)
            cip = (np.arange(1, n + 1)) / (n + 1)
            is_case = cip < prevalence
            case_cip = cip[is_case]
            case_age = self.cip_x0 + (1.0 / self.cip_k) * np.log(case_cip / (prevalence - case_cip))
            t[order[is_case]] = case_age

        np.clip(t, 0.01, 1e6, out=t)
        return t
