"""First-passage time phenotype model.

A latent health process Y(t) = y0 + drift*t + W(t) starts at
y0 = sqrt(shape) * exp(-scaled_beta*(L - mean) - beta_sex*sex). Disease onset
is the first time Y(t) ≤ 0. When drift < 0 the process hits the boundary
in finite time (everyone eventually onsets); when drift > 0 an emergent
cure fraction P(never hit) = 1 - exp(-2*y0*drift) arises.

References:
    Lee & Whitmore 2006 (threshold regression).
    Aalen & Gjessing 2001 (first-passage models in survival).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, Self

import numpy as np
from numba import njit, prange

from simace.phenotyping.hazards import standardize_beta
from simace.phenotyping.models._base import (
    PhenotypeModel,
    check_finite_beta,
    check_no_foreign_flags,
    wrap_trait_error,
)

if TYPE_CHECKING:
    import argparse

__all__ = ["FirstPassageModel"]


# ---------------------------------------------------------------------------
# Numba kernels — fused y0 + Michael-Schucany-Haas inverse-Gaussian sampler
# ---------------------------------------------------------------------------


@njit(cache=True)
def _msh_sample(normal, uniform, mu, lam):
    """Michael-Schucany-Haas (1976) inverse Gaussian sampler, clamped to [1e-10, 1e6]."""
    y = normal * normal
    mu2 = mu * mu
    half_mu_over_lam = mu / (2.0 * lam)
    x = mu + half_mu_over_lam * (mu * y - np.sqrt(4.0 * mu * lam * y + mu2 * y * y))
    if uniform <= mu / (mu + x):
        return min(max(x, 1e-10), 1e6)
    return min(max(mu2 / x, 1e-10), 1e6)


@njit(parallel=True, cache=True)
def _nb_fpt(normals, uniforms, liability, mean, scaled_beta, sex, beta_sex, y0_base, inv_drift):
    """Fused FPT kernel for drift < 0 (everyone hits)."""
    n = len(normals)
    t = np.empty(n)
    for i in prange(n):
        y0 = y0_base * np.exp(-scaled_beta * (liability[i] - mean) - beta_sex * sex[i])
        if y0 < 1e-300:
            t[i] = 1e-10
        elif y0 > 1e150:
            t[i] = 1e6
        else:
            t[i] = _msh_sample(normals[i], uniforms[i], y0 * inv_drift, y0 * y0)
    return t


@njit(parallel=True, cache=True)
def _nb_fpt_cure(normals, uniforms, cure_draws, liability, mean, scaled_beta, sex, beta_sex, y0_base, drift, inv_drift):
    """Fused FPT kernel for drift > 0 (emergent cure fraction)."""
    n = len(normals)
    t = np.empty(n)
    for i in prange(n):
        y0 = y0_base * np.exp(-scaled_beta * (liability[i] - mean) - beta_sex * sex[i])
        if y0 < 1e-300:
            t[i] = 1e-10
        else:
            p_hit = np.exp(-2.0 * y0 * drift)
            if cure_draws[i] >= p_hit:
                t[i] = 1e6
            else:
                t[i] = _msh_sample(normals[i], uniforms[i], y0 * inv_drift, y0 * y0)
    return t


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FirstPassageModel(PhenotypeModel):
    """First-passage-time phenotype model.

    Parameters:
        drift:    drift rate μ; must be non-zero. Negative drift → toward
                  boundary (everyone onsets). Positive drift → emergent
                  cure fraction.
        shape:    y0² (initial distance from boundary, squared).
        beta:     coefficient on liability for log(y0); positive β → worse.
        beta_sex: coefficient on sex for log(y0) (0 = no effect).
    """

    drift: float
    shape: float
    beta: float = 1.0
    beta_sex: float = 0.0

    name: ClassVar[str] = "first_passage"

    def __post_init__(self) -> None:
        check_finite_beta(self.beta)
        if self.drift == 0.0:
            raise ValueError("first_passage drift must be non-zero")

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, params: dict[str, Any], trait_num: int) -> Self:
        with wrap_trait_error(trait_num):
            phenotype_params = params.get(f"phenotype_params{trait_num}", {})
            try:
                drift = phenotype_params["drift"]
                shape = phenotype_params["shape"]
            except KeyError as e:
                raise ValueError(
                    f"phenotype_params{trait_num} for model 'first_passage' missing "
                    f"required key {e.args[0]!r}; needs 'drift' and 'shape'"
                ) from e
            return cls(
                drift=drift,
                shape=shape,
                beta=params[f"beta{trait_num}"],
                beta_sex=params.get(f"beta_sex{trait_num}", 0.0),
            )

    @classmethod
    def add_cli_args(cls, parser: argparse.ArgumentParser, trait: int) -> None:
        group = parser.add_argument_group(f"Trait {trait} — first_passage")
        group.add_argument(f"--first-passage-drift{trait}", type=float, default=None)
        group.add_argument(f"--first-passage-shape{trait}", type=float, default=None)

    @classmethod
    def from_cli(cls, args: argparse.Namespace, trait: int) -> Self:
        check_no_foreign_flags(cls, args, trait)
        with wrap_trait_error(trait):
            drift = getattr(args, f"first_passage_drift{trait}")
            if drift is None:
                raise ValueError(
                    f"--first-passage-drift{trait} is required when --phenotype-model{trait}=first_passage"
                )
            shape = getattr(args, f"first_passage_shape{trait}")
            if shape is None:
                raise ValueError(
                    f"--first-passage-shape{trait} is required when --phenotype-model{trait}=first_passage"
                )
            return cls(
                drift=drift,
                shape=shape,
                beta=getattr(args, f"beta{trait}"),
                beta_sex=getattr(args, f"beta_sex{trait}", 0.0),
            )

    @classmethod
    def cli_flag_attrs(cls, trait: int) -> set[str]:
        return {f"first_passage_drift{trait}", f"first_passage_shape{trait}"}

    def to_params_dict(self) -> dict[str, Any]:
        return {"drift": self.drift, "shape": self.shape}

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
        rng = np.random.default_rng(seed)
        mean, scaled_beta = standardize_beta(liability, self.beta, standardize)

        y0_base = np.sqrt(self.shape)
        normals = rng.standard_normal(n)
        uniforms = rng.random(n)
        sex_arr = sex if (sex is not None and self.beta_sex != 0.0) else np.zeros(n)
        sex_beta = self.beta_sex if sex is not None else 0.0
        inv_drift = 1.0 / abs(self.drift)

        if self.drift < 0:
            return _nb_fpt(normals, uniforms, liability, mean, scaled_beta, sex_arr, sex_beta, y0_base, inv_drift)
        cure_draws = rng.random(n)
        return _nb_fpt_cure(
            normals,
            uniforms,
            cure_draws,
            liability,
            mean,
            scaled_beta,
            sex_arr,
            sex_beta,
            y0_base,
            self.drift,
            inv_drift,
        )
