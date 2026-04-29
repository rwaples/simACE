"""Bimodal phenotype models (prototype).

Three models that produce bimodal age-of-onset distributions:
  1. mixture_cip:          Two logistic CIP components, shared β
  2. mixture_cure_frailty: Two cure-frailty hazards, shared β
  3. two_threshold:        Two liability thresholds with separate CIP per stratum
"""

from __future__ import annotations

__all__ = [
    "phenotype_mixture_cip",
    "phenotype_mixture_cure_frailty",
    "phenotype_two_threshold",
]

import numpy as np

from simace.core._numba_utils import _ndtri_approx
from simace.phenotyping.hazards import BASELINE_HAZARDS, standardize_beta


def phenotype_mixture_cip(
    liability: np.ndarray,
    prevalence: float | np.ndarray,
    beta: float = 1.0,
    pi: float = 0.5,
    cip_x0_1: float = 10.0,
    cip_k_1: float = 0.3,
    cip_x0_2: float = 40.0,
    cip_k_2: float = 0.2,
    seed: int = 42,
    standardize: bool = True,
    sex: np.ndarray | None = None,
    beta_sex: float = 0.0,
) -> np.ndarray:
    """Mixture of two logistic CIP curves with shared liability threshold.

    Each case is randomly assigned to component 1 (probability π) or
    component 2 (probability 1−π).  Both components share the same
    prevalence K and liability scaling (β, β_sex), but have different
    CIP midpoints (x₀) and steepness (k), producing bimodal onset.

    Args:
        liability:   quantitative liability, shape (n,)
        prevalence:  population prevalence K; scalar or per-individual array
        beta:        probit scaling factor for liability
        pi:          mixing weight for component 1 (0–1)
        cip_x0_1:    logistic CIP midpoint for component 1 (early)
        cip_k_1:     logistic CIP growth rate for component 1
        cip_x0_2:    logistic CIP midpoint for component 2 (late)
        cip_k_2:     logistic CIP growth rate for component 2
        seed:        random seed for component assignment
        standardize: if True, standardize liability to N(0,1)
        sex:         binary sex covariate (0/1), shape (n,), or None
        beta_sex:    probit-scale coefficient for sex

    Returns:
        Array of simulated event times, shape (n,)
    """
    L = liability.copy()
    if standardize:
        std = np.std(L)
        if std > 0:
            L = (L - L.mean()) / std

    _ndtri_vec = np.vectorize(_ndtri_approx)
    threshold = _ndtri_vec(1.0 - prevalence)
    is_case = threshold < L

    t = np.full(len(L), 1e6)
    n_cases = is_case.sum()
    if n_cases > 0:
        prev_case = prevalence[is_case] if isinstance(prevalence, np.ndarray) else prevalence
        L_eff = beta * L[is_case]
        if beta_sex != 0.0 and sex is not None:
            L_eff = L_eff + beta_sex * sex[is_case]
        from scipy.special import erfc

        cir = 0.5 * erfc(L_eff / np.sqrt(2.0))
        valid = cir < prev_case
        cir_clipped = np.clip(cir, 1e-10, np.asarray(prev_case) - 1e-10)

        # Component assignment
        rng = np.random.default_rng(seed)
        comp1 = rng.random(n_cases) < pi

        # Invert through each component's logistic CIP
        onset_1 = cip_x0_1 + (1.0 / cip_k_1) * np.log(cir_clipped / (prev_case - cir_clipped))
        onset_2 = cip_x0_2 + (1.0 / cip_k_2) * np.log(cir_clipped / (prev_case - cir_clipped))

        onset = np.where(comp1, onset_1, onset_2)
        onset[~valid] = 1e6
        t[is_case] = onset

    np.clip(t, 0.01, 1e6, out=t)
    return t


def phenotype_mixture_cure_frailty(
    liability: np.ndarray,
    prevalence: float | np.ndarray,
    beta: float,
    pi: float,
    baseline: str,
    hazard_params_1: dict[str, float],
    hazard_params_2: dict[str, float],
    seed: int,
    standardize: bool = True,
    sex: np.ndarray | None = None,
    beta_sex: float = 0.0,
) -> np.ndarray:
    """Mixture of two cure-frailty components with shared threshold.

    Cases (L > threshold) are randomly assigned to component 1 (prob π)
    or component 2 (prob 1−π).  Each component has its own baseline hazard
    parameters, producing different onset timing distributions.

    Args:
        liability:       quantitative liability, shape (n,)
        prevalence:      population prevalence K
        beta:            effect of liability on log-hazard among cases
        pi:              mixing weight for component 1
        baseline:        baseline hazard model name (shared)
        hazard_params_1: hazard parameters for component 1
        hazard_params_2: hazard parameters for component 2
        seed:            random seed
        standardize:     if True, standardize liability to N(0,1)
        sex:             binary sex covariate (0/1), shape (n,), or None
        beta_sex:        effect of sex on log-hazard

    Returns:
        Array of simulated event times, shape (n,)
    """
    n = len(liability)
    mean, scaled_beta = standardize_beta(liability, beta, standardize)

    if standardize:
        std = np.std(liability)
        L = (liability - liability.mean()) / std if std > 0 else liability - liability.mean()
    else:
        L = liability

    _ndtri_vec = np.vectorize(_ndtri_approx)
    threshold = _ndtri_vec(1.0 - prevalence)
    is_case = threshold < L

    t = np.full(n, 1e6)
    n_cases = is_case.sum()
    if n_cases > 0:
        rng = np.random.default_rng(seed)
        neg_log_u = rng.exponential(size=n_cases)
        comp1 = rng.random(n_cases) < pi

        if beta_sex != 0.0 and sex is not None:
            neg_log_u = neg_log_u / np.exp(beta_sex * sex[is_case])

        L_cases = L[is_case]

        # Component 1
        t1 = BASELINE_HAZARDS[baseline](neg_log_u, L_cases, mean, scaled_beta, hazard_params_1)
        # Component 2
        t2 = BASELINE_HAZARDS[baseline](neg_log_u, L_cases, mean, scaled_beta, hazard_params_2)

        t[is_case] = np.where(comp1, t1, t2)

    return t


def phenotype_two_threshold(
    liability: np.ndarray,
    prevalence_early: float | np.ndarray,
    prevalence_late: float | np.ndarray,
    beta: float = 1.0,
    cip_x0_1: float = 10.0,
    cip_k_1: float = 0.3,
    cip_x0_2: float = 40.0,
    cip_k_2: float = 0.2,
    seed: int = 42,
    standardize: bool = True,
    sex: np.ndarray | None = None,
    beta_sex: float = 0.0,
) -> np.ndarray:
    """Two-threshold liability model with separate CIP per stratum.

    High-liability individuals (L > τ₁) are early-onset cases mapped
    through CIP₁.  Moderate-liability individuals (τ₂ < L ≤ τ₁) are
    late-onset cases mapped through CIP₂.  This preserves a single
    liability dimension while producing bimodal age-of-onset.

    Args:
        liability:        quantitative liability, shape (n,)
        prevalence_early: K_early (fraction with L > τ₁)
        prevalence_late:  K_late (fraction with τ₂ < L ≤ τ₁);
                          total prevalence = K_early + K_late
        beta:             probit scaling factor for liability
        cip_x0_1:        logistic CIP midpoint for early component
        cip_k_1:         logistic CIP growth rate for early component
        cip_x0_2:        logistic CIP midpoint for late component
        cip_k_2:         logistic CIP growth rate for late component
        seed:             unused (deterministic model)
        standardize:      if True, standardize liability to N(0,1)
        sex:              binary sex covariate (0/1), shape (n,), or None
        beta_sex:         probit-scale coefficient for sex

    Returns:
        Array of simulated event times, shape (n,)
    """
    L = liability.copy()
    if standardize:
        std = np.std(L)
        if std > 0:
            L = (L - L.mean()) / std

    _ndtri_vec = np.vectorize(_ndtri_approx)
    K_total = np.asarray(prevalence_early) + np.asarray(prevalence_late)
    tau1 = _ndtri_vec(1.0 - prevalence_early)  # early threshold (higher)
    tau2 = _ndtri_vec(1.0 - K_total)  # late threshold (lower)

    early = tau1 < L
    late = (tau2 < L) & (~early)

    t = np.full(len(L), 1e6)

    from scipy.special import erfc

    for mask, prev_param, x0, k in [
        (early, prevalence_early, cip_x0_1, cip_k_1),
        (late, prevalence_late, cip_x0_2, cip_k_2),
    ]:
        n_grp = mask.sum()
        if n_grp == 0:
            continue
        prev_grp = prev_param[mask] if isinstance(prev_param, np.ndarray) else prev_param
        L_eff = beta * L[mask]
        if beta_sex != 0.0 and sex is not None:
            L_eff = L_eff + beta_sex * sex[mask]

        cir = 0.5 * erfc(L_eff / np.sqrt(2.0))
        valid = cir < prev_grp
        cir = np.clip(cir, 1e-10, np.asarray(prev_grp) - 1e-10)
        onset = x0 + (1.0 / k) * np.log(cir / (prev_grp - cir))
        onset[~valid] = 1e6
        t[mask] = onset

    np.clip(t, 0.01, 1e6, out=t)
    return t
