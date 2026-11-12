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

from simace.phenotype.hazards import (
    iter_generation_groups,
    resolve_hazard_mode,
    standardize_beta,
)
from simace.phenotype.models._base import (
    PhenotypeModel,
    check_finite_beta,
    validate_standardize_hazard,
)

if TYPE_CHECKING:
    from simace.phenotype.hazards import StandardizeMode

__all__ = ["FirstPassageModel"]


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
    standardize_hazard: StandardizeMode | None = None

    name: ClassVar[str] = "first_passage"
    config_keys: ClassVar[frozenset[str]] = frozenset({"drift", "shape", "standardize_hazard"})

    def __post_init__(self) -> None:
        check_finite_beta(self.beta)
        if self.drift == 0.0:
            raise ValueError("first_passage drift must be non-zero")
        if not self.shape > 0.0:
            raise ValueError(f"first_passage shape must be > 0, got {self.shape!r}")
        validate_standardize_hazard(self.standardize_hazard)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def _from_config(cls, phenotype_params: dict[str, Any], trait_num: int, *, beta: float, beta_sex: float) -> Self:
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
            beta=beta,
            beta_sex=beta_sex,
            standardize_hazard=phenotype_params.get("standardize_hazard"),
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
        from simace.phenotype._kernels import _nb_fpt, _nb_fpt_cure

        mode_haz = resolve_hazard_mode(standardize, self.standardize_hazard)
        n = len(liability)
        rng = np.random.default_rng(seed)
        mean_arr, beta_arr = standardize_beta(liability, self.beta, mode_haz, generation)

        y0_base = np.sqrt(self.shape)
        normals = rng.standard_normal(n)
        uniforms = rng.random(n)
        sex_beta = self.beta_sex
        inv_drift = 1.0 / abs(self.drift)
        cure_draws = rng.random(n) if self.drift > 0 else None

        t = np.empty(n)
        for mask in iter_generation_groups(mode_haz, generation):
            m = float(mean_arr[mask][0])
            b = float(beta_arr[mask][0])
            if self.drift < 0:
                t[mask] = _nb_fpt(
                    normals[mask],
                    uniforms[mask],
                    liability[mask],
                    m,
                    b,
                    sex[mask],
                    sex_beta,
                    y0_base,
                    inv_drift,
                )
            else:
                # else of `drift < 0` and drift != 0 (1/abs(drift) above) => drift > 0,
                # which is exactly when cure_draws was allocated.
                assert cure_draws is not None
                t[mask] = _nb_fpt_cure(
                    normals[mask],
                    uniforms[mask],
                    cure_draws[mask],
                    liability[mask],
                    m,
                    b,
                    sex[mask],
                    sex_beta,
                    y0_base,
                    self.drift,
                    inv_drift,
                )
        return t
