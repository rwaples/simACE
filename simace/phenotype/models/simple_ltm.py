"""Simple liability-threshold phenotype model.

Liability above a probit threshold sets case status (WHO has the disease) at a
target prevalence ``K``; a small ``onset`` sub-model assigns an age-of-onset
(WHEN) to cases.  Unlike ``cure_frailty`` the onset is independent of liability:

* ``onset.kind = "fixed"`` — every case onsets at the same ``age``.
* ``onset.kind = "normal"`` — case onset ~ ``Normal(mean, sd)``.

Controls are censored at ``1e6``.  Onset times flow through the standard censor
stage like every other model, so the *observed* affected rate after age-window
and death censoring is below ``K``.

This is a threshold-on-liability model: it consumes the global ``standardize``
flag directly (like ``adult.ltm``) and does not accept ``standardize_hazard``.
``beta`` / ``beta_sex`` are parsed for uniformity with the other models but are
unused — neither case status nor onset depends on them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar, Self

import numpy as np

from simace.phenotype.hazards import coerce_standardize_mode
from simace.phenotype.models._base import (
    PhenotypeModel,
    check_finite_beta,
    check_no_foreign_flags,
    wrap_trait_error,
)
from simace.phenotype.models._prevalence import case_status_from_liability

if TYPE_CHECKING:
    import argparse

    from simace.phenotype.hazards import StandardizeMode

__all__ = ["SimpleLtmModel"]


_ONSET_KINDS: frozenset[str] = frozenset({"fixed", "normal"})


def _validate_onset(onset: dict[str, Any]) -> None:
    """Validate the ``onset`` sub-model dict (discriminated on ``kind``)."""
    kind = onset.get("kind")
    if kind not in _ONSET_KINDS:
        raise ValueError(f"simple_ltm onset.kind must be one of {sorted(_ONSET_KINDS)}; got {kind!r}")
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

    def __post_init__(self) -> None:
        check_finite_beta(self.beta)
        _validate_onset(self.onset)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, params: dict[str, Any], trait_num: int) -> Self:
        with wrap_trait_error(trait_num):
            phenotype_params = dict(params.get(f"phenotype_params{trait_num}", {}))
            if "prevalence" not in phenotype_params:
                raise ValueError(f"phenotype_params{trait_num} for model 'simple_ltm' must include 'prevalence' key")
            if "onset" not in phenotype_params:
                raise ValueError(
                    f"phenotype_params{trait_num} for model 'simple_ltm' must include 'onset' key "
                    f"(e.g. {{'kind': 'fixed', 'age': 30}} or {{'kind': 'normal', 'mean': 30, 'sd': 8}})"
                )
            return cls(
                prevalence=phenotype_params["prevalence"],
                onset=dict(phenotype_params["onset"]),
                beta=params[f"beta{trait_num}"],
                beta_sex=params.get(f"beta_sex{trait_num}", 0.0),
            )

    @classmethod
    def add_cli_args(cls, parser: argparse.ArgumentParser, trait: int) -> None:
        group = parser.add_argument_group(f"Trait {trait} — simple_ltm")
        group.add_argument(f"--simple-ltm-prevalence{trait}", type=float, default=None)
        group.add_argument(
            f"--simple-ltm-onset-kind-{trait}",
            default=None,
            choices=sorted(_ONSET_KINDS),
            help=f"Onset sub-model for trait {trait}",
        )
        group.add_argument(f"--simple-ltm-onset-age-{trait}", type=float, default=None, help="fixed onset age")
        group.add_argument(f"--simple-ltm-onset-mean-{trait}", type=float, default=None, help="normal onset mean")
        group.add_argument(f"--simple-ltm-onset-sd-{trait}", type=float, default=None, help="normal onset sd")

    @classmethod
    def from_cli(cls, args: argparse.Namespace, trait: int) -> Self:
        check_no_foreign_flags(cls, args, trait)
        with wrap_trait_error(trait):
            prevalence = getattr(args, f"simple_ltm_prevalence{trait}")
            if prevalence is None:
                raise ValueError(f"--simple-ltm-prevalence{trait} is required when --phenotype-model{trait}=simple_ltm")
            kind = getattr(args, f"simple_ltm_onset_kind_{trait}")
            if kind is None:
                raise ValueError(
                    f"--simple-ltm-onset-kind-{trait} is required when --phenotype-model{trait}=simple_ltm"
                )
            if kind == "fixed":
                age = getattr(args, f"simple_ltm_onset_age_{trait}")
                if age is None:
                    raise ValueError(f"--simple-ltm-onset-age-{trait} is required for onset kind 'fixed'")
                onset: dict[str, Any] = {"kind": "fixed", "age": age}
            else:  # normal
                mean = getattr(args, f"simple_ltm_onset_mean_{trait}")
                sd = getattr(args, f"simple_ltm_onset_sd_{trait}")
                if mean is None or sd is None:
                    raise ValueError(
                        f"--simple-ltm-onset-mean-{trait} and --simple-ltm-onset-sd-{trait} "
                        f"are required for onset kind 'normal'"
                    )
                onset = {"kind": "normal", "mean": mean, "sd": sd}
            return cls(
                prevalence=prevalence,
                onset=onset,
                beta=getattr(args, f"beta{trait}"),
                beta_sex=getattr(args, f"beta_sex{trait}", 0.0),
            )

    @classmethod
    def cli_flag_attrs(cls, trait: int) -> set[str]:
        return {
            f"simple_ltm_prevalence{trait}",
            f"simple_ltm_onset_kind_{trait}",
            f"simple_ltm_onset_age_{trait}",
            f"simple_ltm_onset_mean_{trait}",
            f"simple_ltm_onset_sd_{trait}",
        }

    def to_params_dict(self) -> dict[str, Any]:
        return {"prevalence": self.prevalence, "onset": dict(self.onset)}

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
        mode = coerce_standardize_mode(standardize)
        is_case = case_status_from_liability(liability, self.prevalence, sex, generation, mode)

        t = np.full(len(liability), 1e6)
        n_cases = int(is_case.sum())
        if n_cases > 0:
            rng = np.random.default_rng(seed)
            t[is_case] = _draw_onset(self.onset, n_cases, rng)

        np.clip(t, 0.01, 1e6, out=t)
        return t
