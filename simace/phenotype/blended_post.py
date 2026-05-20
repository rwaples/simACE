"""Post-hoc blended-diagnosis transform.

Re-derives trait-1 case status as a per-generation blend of the simulated
liabilities L1 and L2. This models the empirical "Narrative B" hypothesis
for the inverted-U in ADHD pedigree h² over birth cohorts: diagnostic
criteria broaden over time so that a fraction α_g of trait-1 cases in
cohort *g* are actually loaded for trait 2's genetics (e.g. ASD-loaded
individuals diagnosed with ADHD under expanded DSM criteria).

The diagnostic liability is

    L_diag(g) = (1 − α_g) · L1 + α_g · L2

per-generation standardized, and the case-status threshold corresponds
to per-generation prevalence K_g.

The transform takes the standard simACE `trait.parquet` (which has
A1/C1/E1 + L1/L2 columns plus the standard `affected1`/`t_observed1`
fields) and returns a new DataFrame with:

  - `affected1`/`t_observed1`/`age_censored1`/`death_censored1`
    overwritten using the blended diagnosis;
  - new audit columns `A_blend`, `C_blend`, `E_blend`, `liability_blend`
    for the per-row blended ACE components and liability;
  - all other columns (including the original A1/C1/E1/L1) preserved.

Downstream consumers (the temporal plot script) use the blend columns
to compute per-cohort and pooled true h² of the diagnosed phenotype:

    h²_diag(g) = Var(A_blend[gen == g]) / Var(liability_blend[gen == g])

The original `trait.parquet` is preserved as an audit trail; this
transform is invoked by `fitACE/workflow/rules/blended_phenotype.smk`
and writes its output to `phenotype.blended.parquet` alongside.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

__all__ = ["blended_diagnosis"]

import numpy as np
from scipy.special import erfc, ndtri

from simace.phenotype.hazards import standardize_liability
from simace.phenotype.models._prevalence import prevalence_to_array

if TYPE_CHECKING:
    import pandas as pd

#: Right-censoring age for late-onset cases. Matches the simACE default
#: `censoring.max_age`.
MAX_AGE = 80.0
#: ADuLT/LTM cumulative-incidence shape parameters. The default values
#: track those of the source `adult/ltm` phenotype model — the temporal
#: scenarios all use these defaults, so we hardcode them here. If a
#: scenario uses non-default values they should be passed in.
DEFAULT_CIP_X0 = 50.0
DEFAULT_CIP_K = 0.15


def _compute_onset(L_eff: np.ndarray, K: np.ndarray, cip_x0: float, cip_k: float) -> np.ndarray:
    """ADuLT/LTM onset-age inversion. Clipped to ``[0.01, 1e6]``."""
    cir = 0.5 * erfc(L_eff / np.sqrt(2.0))
    cir = np.clip(cir, 1e-10, K - 1e-10)
    onset = cip_x0 + (1.0 / cip_k) * np.log(cir / (K - cir))
    np.clip(onset, 0.01, 1e6, out=onset)
    return onset


def blended_diagnosis(
    phenotype: pd.DataFrame,
    *,
    alpha_by_gen: dict[int, float],
    K_by_gen: dict[int, float],
    cip_x0: float = DEFAULT_CIP_X0,
    cip_k: float = DEFAULT_CIP_K,
) -> pd.DataFrame:
    """Return a copy of `phenotype` with trait-1 case status redefined.

    Args:
        phenotype: standard simACE phenotype DataFrame with at least
            ``generation``, ``A1``, ``C1``, ``E1``, ``A2``, ``C2``, ``E2``,
            ``liability1``, ``liability2``, ``death_age``, and the existing
            trait-1 status columns (``affected1``, ``t_observed1``,
            ``age_censored1``, ``death_censored1``).
        alpha_by_gen: per-output-generation weight on trait-2 liability.
            Must include every distinct value in ``phenotype["generation"]``.
        K_by_gen: per-output-generation diagnostic prevalence. Same key
            constraint as ``alpha_by_gen``.
        cip_x0: logistic CIP shape parameter (location) for the onset-age
            inversion. Defaults to the ``adult/ltm`` default.
        cip_k: logistic CIP shape parameter (slope) for the onset-age
            inversion. Defaults to the ``adult/ltm`` default.

    Returns:
        Copy of `phenotype` with overwritten trait-1 status columns and
        new ``A_blend`` / ``C_blend`` / ``E_blend`` / ``liability_blend``
        columns.
    """
    required = {"generation", "A1", "C1", "E1", "A2", "C2", "E2", "liability1", "liability2", "death_age"}
    missing = required - set(phenotype.columns)
    if missing:
        raise ValueError(f"phenotype is missing required columns: {sorted(missing)}")

    pheno = phenotype.copy()
    gen = pheno["generation"].to_numpy()

    # Reuse the prevalence resolver (raises ValueError on missing keys); it
    # accepts arbitrary per-gen dicts and produces a per-individual array.
    alpha = np.asarray(prevalence_to_array(alpha_by_gen, gen), dtype=np.float64)
    K = np.asarray(prevalence_to_array(K_by_gen, gen), dtype=np.float64)

    A1 = pheno["A1"].to_numpy()
    C1 = pheno["C1"].to_numpy()
    E1 = pheno["E1"].to_numpy()
    A2 = pheno["A2"].to_numpy()
    C2 = pheno["C2"].to_numpy()
    E2 = pheno["E2"].to_numpy()
    L1 = pheno["liability1"].to_numpy()
    L2 = pheno["liability2"].to_numpy()
    death_age = pheno["death_age"].to_numpy()

    # Per-row blended components (audit columns; not standardized).
    A_blend = (1.0 - alpha) * A1 + alpha * A2
    C_blend = (1.0 - alpha) * C1 + alpha * C2
    E_blend = (1.0 - alpha) * E1 + alpha * E2
    L_blend = (1.0 - alpha) * L1 + alpha * L2

    # Per-generation-standardised blend → threshold by per-gen K.
    L_blend_std = standardize_liability(L_blend, mode="per_generation", generation=gen)
    threshold = ndtri(1.0 - K)
    is_case = L_blend_std > threshold

    # Onset ages for cases via the ADuLT/LTM inverse-CIF formula. Non-cases
    # and late-onset cases keep the latent-onset placeholder of MAX_AGE.
    latent_onset = np.full(len(pheno), MAX_AGE, dtype=np.float64)
    if is_case.any():
        latent_onset[is_case] = _compute_onset(L_blend_std[is_case], K[is_case], cip_x0, cip_k)
    age_right_censored = is_case & (latent_onset > MAX_AGE)
    follow_up = np.minimum(latent_onset, MAX_AGE)

    # Apply competing-risk death censoring uniformly across cases and
    # non-cases, matching the canonical censoring/censor.py pipeline.
    death_censored = death_age < follow_up
    onset = np.where(death_censored, death_age, follow_up)
    affected = is_case & ~death_censored & ~age_right_censored

    pheno["affected1"] = affected
    pheno["t_observed1"] = onset
    pheno["age_censored1"] = ~affected & ~death_censored
    pheno["death_censored1"] = death_censored

    pheno["A_blend"] = A_blend.astype(np.float32)
    pheno["C_blend"] = C_blend.astype(np.float32)
    pheno["E_blend"] = E_blend.astype(np.float32)
    pheno["liability_blend"] = L_blend.astype(np.float32)

    return pheno
