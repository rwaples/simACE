"""
Phenotype simulation for two correlated traits.

Per-trait phenotype model selection via phenotype_model1/phenotype_model2:
  Frailty models: "weibull", "exponential", "gompertz", "lognormal", "loglogistic", "gamma"
  ADuLT models:   "adult_ltm", "adult_cox"
  Cure models:    "cure_frailty"
  FPT models:     "first_passage"

Frailty models convert liability to raw event times (age-at-onset) using a proportional
hazards frailty model with pluggable baseline hazard.

Model (per trait):
    L        = A + C + E          (liability from pedigree)
    z        = exp(beta * L)      (frailty / hazard multiplier)
    S(t | z) = exp(-H0(t) * z)   (conditional survival)
    t        = H0^{-1}(-log(U) / z)  where U ~ Uniform(0, 1]

Baseline hazard models supported (via compute_hazard_terms):
    "weibull"     : {"scale": s, "rho": rho}
    "exponential" : {"rate": lam}  |  {"scale": s}
    "gompertz"    : {"rate": b, "gamma": g}
    "lognormal"   : {"mu": mu, "sigma": sigma}
    "loglogistic" : {"scale": alpha, "shape": k}
    "gamma"       : {"shape": k, "scale": theta}

Inversion strategy:
    Each model has an analytic/vectorized inverse — no per-individual loop.
    Weibull      → t = scale * (target / z)^(1/rho)
    Exponential  → t = target / (z * rate)
    Gompertz     → t = log1p(target * g / (z * b)) / g
    Lognormal    → t = exp(mu + sigma * norm.isf(exp(-target/z)))
    Loglogistic  → t = alpha * expm1(target/z)^(1/k)
    Gamma        → t = gamma_dist.isf(exp(-target/z), k, scale=theta)
    Unknown      → KeyError (all supported models have analytic inverses)
"""

from __future__ import annotations

import argparse
import logging
import time
from typing import Any

import numpy as np
import pandas as pd
from numba import njit, prange
from scipy.stats import gamma as gamma_dist

from sim_ace.core._numba_utils import _ndtri_approx
from sim_ace.core.compute_hazard_terms import compute_hazard_terms
from sim_ace.core.utils import save_parquet

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Numba kernels — fuse frailty computation + inversion in a single pass
# ---------------------------------------------------------------------------


@njit(parallel=True, cache=True)
def _nb_weibull(neg_log_u, liability, mean, scaled_beta, scale, inv_rho):
    n = len(neg_log_u)
    t = np.empty(n)
    for i in prange(n):
        z = np.exp(scaled_beta * (liability[i] - mean))
        t[i] = scale * np.exp(np.log(neg_log_u[i] / z) * inv_rho)
    return t


@njit(parallel=True, cache=True)
def _nb_exponential(neg_log_u, liability, mean, scaled_beta, inv_rate):
    n = len(neg_log_u)
    t = np.empty(n)
    for i in prange(n):
        z = np.exp(scaled_beta * (liability[i] - mean))
        val = neg_log_u[i] * inv_rate / z
        t[i] = min(max(val, 1e-10), 1e6)
    return t


@njit(parallel=True, cache=True)
def _nb_gompertz(neg_log_u, liability, mean, scaled_beta, g_over_b, inv_g):
    n = len(neg_log_u)
    t = np.empty(n)
    for i in prange(n):
        z = np.exp(scaled_beta * (liability[i] - mean))
        target = neg_log_u[i] / z
        val = np.log1p(target * g_over_b) * inv_g
        t[i] = min(max(val, 1e-10), 1e6)
    return t


@njit(parallel=True, cache=True)
def _nb_lognormal(neg_log_u, liability, mean, scaled_beta, mu, sigma):
    n = len(neg_log_u)
    t = np.empty(n)
    for i in prange(n):
        z = np.exp(scaled_beta * (liability[i] - mean))
        target = neg_log_u[i] / z
        surv = np.exp(-target)
        if surv <= 0.0:
            t[i] = 1e6
        else:
            # norm.isf(surv) = -ndtri(surv) for symmetric normal
            val = np.exp(mu - sigma * _ndtri_approx(surv))
            t[i] = min(max(val, 1e-10), 1e6)
    return t


@njit(parallel=True, cache=True)
def _nb_loglogistic(neg_log_u, liability, mean, scaled_beta, alpha, inv_k):
    n = len(neg_log_u)
    t = np.empty(n)
    for i in prange(n):
        z = np.exp(scaled_beta * (liability[i] - mean))
        target = neg_log_u[i] / z
        val = alpha * np.exp(np.log(np.expm1(target)) * inv_k)
        t[i] = min(max(val, 1e-10), 1e6)
    return t


# ---------------------------------------------------------------------------
# Python wrappers — unpack params dict, call numba kernel
# ---------------------------------------------------------------------------


def _invert_weibull(neg_log_u, liability, mean, scaled_beta, params):
    return _nb_weibull(neg_log_u, liability, mean, scaled_beta, params["scale"], 1.0 / params["rho"])


def _invert_exponential(neg_log_u, liability, mean, scaled_beta, params):
    rate = params["rate"] if "rate" in params else 1.0 / params["scale"]
    return _nb_exponential(neg_log_u, liability, mean, scaled_beta, 1.0 / rate)


def _invert_gompertz(neg_log_u, liability, mean, scaled_beta, params):
    b, g = params["rate"], params["gamma"]
    return _nb_gompertz(neg_log_u, liability, mean, scaled_beta, g / b, 1.0 / g)


def _invert_lognormal(neg_log_u, liability, mean, scaled_beta, params):
    return _nb_lognormal(neg_log_u, liability, mean, scaled_beta, params["mu"], params["sigma"])


def _invert_loglogistic(neg_log_u, liability, mean, scaled_beta, params):
    return _nb_loglogistic(neg_log_u, liability, mean, scaled_beta, params["scale"], 1.0 / params["shape"])


def _invert_gamma(neg_log_u, liability, mean, scaled_beta, params):
    """Gamma inverse — scipy iterative solver, not numba-fusible."""
    frailty = np.exp(scaled_beta * (liability - mean))
    target = neg_log_u / frailty
    t = gamma_dist.isf(np.exp(-target), params["shape"], scale=params["scale"])
    np.clip(t, 1e-10, 1e6, out=t)
    return t


# Dispatch table: model name → inversion function
_INVERTERS = {
    "weibull": _invert_weibull,
    "exponential": _invert_exponential,
    "gompertz": _invert_gompertz,
    "lognormal": _invert_lognormal,
    "loglogistic": _invert_loglogistic,
    "gamma": _invert_gamma,
}


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _standardize_beta(liability: np.ndarray, beta: float, standardize: bool) -> tuple[float, float]:
    """Compute liability mean and scaled beta for frailty/FPT models.

    Returns (mean, scaled_beta) where scaled_beta = beta / std(liability)
    when standardize is True, or (0.0, beta) when False.
    """
    if standardize:
        std = np.std(liability)
        mean = liability.mean()
        scaled_beta = beta / std if std > 0 else 0.0
    else:
        mean = 0.0
        scaled_beta = beta
    return mean, scaled_beta


# ---------------------------------------------------------------------------
# Core simulation
# ---------------------------------------------------------------------------


def simulate_phenotype(
    liability: np.ndarray,
    beta: float,
    hazard_model: str,
    hazard_params: dict[str, float],
    seed: int,
    standardize: bool = True,
    sex: np.ndarray | None = None,
    beta_sex: float = 0.0,
) -> np.ndarray:
    """Convert liability array to simulated time-to-onset.

    Args:
        liability:     quantitative liability, shape (n,)
        beta:          effect of liability on log-hazard
        hazard_model:  baseline hazard model name
        hazard_params: model parameter dict (see module docstring)
        seed:          random seed
        standardize:   if True, standardize liability to mean 0 / std 1
        sex:           sex covariate (0=female, 1=male), shape (n,)
        beta_sex:      effect of sex on log-hazard (0.0 = no effect)

    Returns:
        Array of simulated event times, shape (n,)

    Raises:
        ValueError: if beta is non-finite or hazard_params is invalid
    """
    if not np.isfinite(beta):
        raise ValueError(f"beta must be finite, got {beta}")

    # Validate model/params eagerly via a single dry-run call
    compute_hazard_terms(hazard_model, np.array([1.0]), hazard_params)

    rng = np.random.default_rng(seed)
    mean, scaled_beta = _standardize_beta(liability, beta, standardize)

    neg_log_u = rng.exponential(size=len(liability))  # -log(U), U ~ (0,1]

    if beta_sex != 0.0 and sex is not None:
        neg_log_u = neg_log_u / np.exp(beta_sex * sex)

    return _INVERTERS[hazard_model](neg_log_u, liability, mean, scaled_beta, hazard_params)


# ---------------------------------------------------------------------------
# ADuLT phenotype models (Pedersen et al., Nat Commun 2023)
# ---------------------------------------------------------------------------


def phenotype_adult_ltm(
    liability: np.ndarray,
    prevalence: float | np.ndarray,
    beta: float = 1.0,
    cip_x0: float = 50.0,
    cip_k: float = 0.2,
    seed: int = 42,
    standardize: bool = True,
    sex: np.ndarray | None = None,
    beta_sex: float = 0.0,
) -> np.ndarray:
    """ADuLT liability threshold model: age-of-onset via logistic CIP.

    Cases (L > threshold): age = x₀ + (1/k)·log(Φ(−βL − β_sex·sex) / (K − Φ(−βL − β_sex·sex)))
    Controls (L ≤ threshold): t = 1e6 (censored downstream)

    Case/control status is determined from raw (unscaled) liability so that
    prevalence is preserved exactly.  Beta and sex enter the CIR→age mapping
    via probit scaling: L_eff = β·L + β_sex·sex is passed through norm.sf()
    to compute cumulative incidence rate (CIR), which maps to onset age.

    Note on beta semantics: in the frailty/cure_frailty models, beta enters
    the log-hazard as exp(β·L) — an exponential dose-response.  Here beta
    scales L on the probit scale (inside Φ), which is a different functional
    form.  The qualitative behavior is consistent (β=1 is baseline, β=0
    removes liability's effect on timing, β>1 strengthens it), but the same
    numeric beta value will produce different effect magnitudes across model
    types.  An alternative "log-odds scaling" approach (β multiplying the
    logit of CIR) would make beta more comparable, but probit scaling is
    the more natural parameterization for a threshold/CIP model.

    Args:
        liability:   quantitative liability, shape (n,)
        prevalence:  population prevalence K; scalar (uniform) or array
                     of shape (n,) for per-group prevalence (by sex,
                     generation, or sex×generation)
        beta:        probit scaling factor for liability (1.0 = no scaling);
                     scales L inside Φ(·), not a log-hazard coefficient
        cip_x0:      logistic CIP midpoint age
        cip_k:       logistic CIP growth rate
        seed:        unused (deterministic model)
        standardize: if True, standardize liability to N(0,1)
        sex:         binary sex covariate (0/1), shape (n,), or None
        beta_sex:    probit-scale coefficient for sex (0.0 = no effect);
                     positive β_sex → males (sex=1) onset earlier

    Returns:
        Array of simulated event times, shape (n,)
    """
    L = liability.copy()
    if standardize:
        std = np.std(L)
        if std > 0:
            L = (L - L.mean()) / std

    # ndtri is scalar; np.vectorize handles array prevalence transparently
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
        cir = np.clip(cir, 1e-10, prev_case - 1e-10)
        t[is_case] = cip_x0 + (1.0 / cip_k) * np.log(cir / (prev_case - cir))

    np.clip(t, 0.01, 1e6, out=t)
    return t


def phenotype_adult_cox(
    liability: np.ndarray,
    prevalence: float | np.ndarray,
    beta: float = 1.0,
    cip_x0: float = 50.0,
    cip_k: float = 0.2,
    seed: int = 42,
    standardize: bool = True,
    sex: np.ndarray | None = None,
    beta_sex: float = 0.0,
) -> np.ndarray:
    """ADuLT proportional hazards model: Weibull(shape=2) + CIP→age mapping.

    Raw time: t̃ = √(Exp(1)·exp(β_sex·sex) / exp(β·L))
    Sorted by t_raw, running CIP = rank/(N+1) capped at K (prevalence).
    Cases (CIP < K): age = x₀ + (1/k)·log(CIP / (K − CIP))
    Controls (CIP ≥ K): t = 1e6 (censored downstream)

    Args:
        liability:    quantitative liability, shape (n,)
        prevalence:   population prevalence K (determines case fraction);
                      scalar (uniform) or array of shape (n,) for
                      per-group prevalence (by sex, generation, or
                      sex×generation)
        beta:         liability scaling factor (1.0 = no scaling)
        cip_x0:       logistic CIP midpoint age
        cip_k:        logistic CIP growth rate
        seed:         random seed
        standardize:  if True, standardize liability to N(0,1)
        sex:          binary sex covariate (0/1), shape (n,), or None
        beta_sex:     log-hazard coefficient for sex (0.0 = no effect)

    Returns:
        Array of simulated event times, shape (n,)
    """
    rng = np.random.default_rng(seed)

    L = liability.copy()
    if standardize:
        std = np.std(L)
        if std > 0:
            L = (L - L.mean()) / std

    n = len(L)
    neg_log_u = rng.exponential(size=n)
    if beta_sex != 0.0 and sex is not None:
        neg_log_u = neg_log_u / np.exp(beta_sex * sex)
    t_raw = np.sqrt(neg_log_u / np.exp(beta * L))

    t = np.full(n, 1e6)

    if isinstance(prevalence, np.ndarray):
        # Per-group ranking: group by unique prevalence values to achieve
        # exact case rates for each group (sex, generation, or sex×generation)
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
            case_age = cip_x0 + (1.0 / cip_k) * np.log(case_cip / (grp_prev - case_cip))
            t[idx[grp_order[is_case]]] = case_age
    else:
        # Sort by raw time; assign running CIP capped at prevalence
        order = np.argsort(t_raw)
        cip = (np.arange(1, n + 1)) / (n + 1)  # ranks in (0, 1)
        is_case = cip < prevalence
        # Map case CIP to age via logistic CIP inverse: age = x₀ + (1/k)·log(CIP/(K−CIP))
        case_cip = cip[is_case]
        case_age = cip_x0 + (1.0 / cip_k) * np.log(case_cip / (prevalence - case_cip))
        t[order[is_case]] = case_age

    np.clip(t, 0.01, 1e6, out=t)
    return t


# ---------------------------------------------------------------------------
# Mixture cure frailty model (Berkson & Gage 1952, Farewell 1982)
# ---------------------------------------------------------------------------


def phenotype_cure_frailty(
    liability: np.ndarray,
    prevalence: float | np.ndarray,
    beta: float,
    baseline: str,
    hazard_params: dict[str, float],
    seed: int,
    standardize: bool = True,
    sex: np.ndarray | None = None,
    beta_sex: float = 0.0,
) -> np.ndarray:
    """Mixture cure frailty model: threshold WHO, frailty WHEN.

    Liability threshold determines case status (WHO gets the disorder),
    then a proportional hazards frailty model determines age-of-onset
    (WHEN among cases). Controls are censored at 1e6.

    Args:
        liability:     quantitative liability, shape (n,)
        prevalence:    population prevalence K (case fraction); scalar
                       (uniform) or array of shape (n,) for per-group
                       prevalence (by sex, generation, or sex×generation)
        beta:          effect of liability on log-hazard among cases
        baseline:      baseline hazard model name (e.g. "weibull", "gompertz")
        hazard_params: model parameter dict for the baseline hazard
        seed:          random seed
        standardize:   if True, standardize liability to N(0,1)
        sex:           sex covariate (0=female, 1=male), shape (n,)
        beta_sex:      effect of sex on log-hazard (0.0 = no effect)

    Returns:
        Array of simulated event times, shape (n,)
    """
    n = len(liability)
    mean, scaled_beta = _standardize_beta(liability, beta, standardize)

    # Standardize liability for thresholding (cure_frailty needs N(0,1) scale)
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

        if beta_sex != 0.0 and sex is not None:
            neg_log_u = neg_log_u / np.exp(beta_sex * sex[is_case])

        t[is_case] = _INVERTERS[baseline](
            neg_log_u,
            L[is_case],
            mean,
            scaled_beta,
            hazard_params,
        )

    return t


# ---------------------------------------------------------------------------
# First-passage time model (Lee & Whitmore 2006, Aalen & Gjessing 2001)
# ---------------------------------------------------------------------------


def phenotype_first_passage(
    liability: np.ndarray,
    beta: float,
    drift: float,
    shape: float,
    seed: int,
    standardize: bool = True,
    sex: np.ndarray | None = None,
    beta_sex: float = 0.0,
) -> np.ndarray:
    """First-passage time model: Wiener process hitting a boundary.

    Latent health process Y(t) = y0 + drift*t + W(t) starts at y0 = sqrt(shape)
    and disease onset occurs at the first time Y(t) <= 0.  Liability scales the
    initial distance y0: higher liability → closer to boundary → earlier onset.

    When drift < 0 (toward boundary), everyone eventually hits and event times
    follow an inverse Gaussian distribution.  When drift > 0 (away from boundary),
    an emergent cure fraction arises: P(never hit) = 1 - exp(-2*y0*drift).

    Args:
        liability:   quantitative liability, shape (n,)
        beta:        effect of liability on log(y0); positive β → worse outcome
        drift:       drift rate μ; negative = toward boundary, positive = away
        shape:       y0² (initial distance squared); controls spread
        seed:        random seed
        standardize: if True, standardize liability to mean 0 / std 1
        sex:         sex covariate (0=female, 1=male), shape (n,)
        beta_sex:    effect of sex on log(y0) (0.0 = no effect)

    Returns:
        Array of simulated event times, shape (n,)

    Raises:
        ValueError: if beta is non-finite or drift is zero
    """
    if not np.isfinite(beta):
        raise ValueError(f"beta must be finite, got {beta}")
    if drift == 0.0:
        raise ValueError("first_passage drift must be non-zero")

    n = len(liability)
    rng = np.random.default_rng(seed)
    mean, scaled_beta = _standardize_beta(liability, beta, standardize)

    # Per-individual initial distance: y0_i = sqrt(shape) * exp(-beta*L - beta_sex*sex)
    y0_base = np.sqrt(shape)
    adjustment = np.exp(-scaled_beta * (liability - mean))
    if beta_sex != 0.0 and sex is not None:
        adjustment = adjustment * np.exp(-beta_sex * sex)
    y0 = y0_base * adjustment

    abs_drift = abs(drift)

    if drift < 0:
        # Everyone hits — standard inverse Gaussian
        t = rng.wald(mean=y0 / abs_drift, scale=y0**2)
    else:
        # Emergent cure fraction: P(ever hit) = exp(-2*y0*drift)
        p_hit = np.exp(-2.0 * y0 * drift)
        hits = rng.random(size=n) < p_hit
        t = np.full(n, 1e6)  # default: censored (never hit)
        n_hits = hits.sum()
        if n_hits > 0:
            mean_ig = y0[hits] / drift
            shape_ig = y0[hits] ** 2
            t[hits] = rng.wald(mean=mean_ig, scale=shape_ig)

    return np.clip(t, 1e-10, 1e6)


_FRAILTY_MODELS = {"weibull", "exponential", "gompertz", "lognormal", "loglogistic", "gamma"}
_ADULT_MODELS = {"adult_ltm", "adult_cox"}
_CURE_MODELS = {"cure_frailty"}
_FPT_MODELS = {"first_passage"}


def _prevalence_to_array(prev, generation):
    """Expand a scalar or per-generation dict prevalence to a per-individual array.

    Returns the scalar unchanged if *prev* is not a dict.
    """
    if isinstance(prev, dict):
        arr = np.empty(len(generation))
        for gen in np.unique(generation):
            mask = generation == gen
            gen_key = int(gen)
            if gen_key not in prev:
                raise ValueError(f"prevalence dict missing generation {gen_key}; dict has keys {sorted(prev.keys())}")
            arr[mask] = prev[gen_key]
        return arr
    return prev


def _resolve_prevalence(params, trait_num, sex, generation):
    """Resolve prevalence to scalar or per-individual array.

    Supports three formats for ``prevalence{N}``:
      - scalar float: same prevalence for everyone
      - per-generation dict (int keys): different prevalence per generation
      - sex-specific dict (``{"female": f, "male": m}``): different
        prevalence per sex, where each sex value may itself be a scalar
        or per-generation dict

    Returns a scalar (when uniform) or a per-individual array.
    """
    prev = params[f"prevalence{trait_num}"]
    if isinstance(prev, dict) and "female" in prev and "male" in prev:
        f_prev = _prevalence_to_array(prev["female"], generation)
        m_prev = _prevalence_to_array(prev["male"], generation)
        return np.where(sex == 1, m_prev, f_prev)
    return _prevalence_to_array(prev, generation)


_ALL_PHENOTYPE_MODELS = sorted(_FRAILTY_MODELS | _ADULT_MODELS | _CURE_MODELS | _FPT_MODELS)

# Parameter names required per model
_MODEL_PARAMS: dict[str, list[str]] = {
    "weibull": ["scale", "rho"],
    "exponential": ["rate"],
    "gompertz": ["rate", "gamma"],
    "lognormal": ["mu", "sigma"],
    "loglogistic": ["scale", "shape"],
    "gamma": ["shape", "scale"],
    "first_passage": ["drift", "shape"],
}
_ADULT_PARAMS: dict[str, list[str]] = {
    "adult_ltm": ["cip_x0", "cip_k"],
    "adult_cox": ["cip_x0", "cip_k"],
}


def _validate_phenotype_params(
    model: str,
    phenotype_params: dict,
    trait_num: int,
) -> None:
    """Validate that phenotype_params contains the required keys for the model.

    Raises:
        ValueError: if required keys are missing or unexpected keys are present
    """
    if model in _FRAILTY_MODELS or model in _FPT_MODELS:
        required = set(_MODEL_PARAMS[model])
    elif model in _ADULT_MODELS:
        required = set(_ADULT_PARAMS[model])
    elif model in _CURE_MODELS:
        if "baseline" not in phenotype_params:
            raise ValueError(
                f"phenotype_params{trait_num} for model {model!r} must include "
                f"'baseline' key specifying the baseline hazard model"
            )
        baseline = phenotype_params["baseline"]
        if baseline not in _FRAILTY_MODELS:
            raise ValueError(
                f"phenotype_params{trait_num} baseline={baseline!r} is not a valid "
                f"frailty model; valid baselines: {sorted(_FRAILTY_MODELS)}"
            )
        required = set(_MODEL_PARAMS[baseline]) | {"baseline"}
    else:
        raise ValueError(f"Unknown phenotype_model{trait_num}={model!r}; valid models: {_ALL_PHENOTYPE_MODELS}")

    provided = set(phenotype_params.keys())
    missing = required - provided
    if missing:
        raise ValueError(f"phenotype_params{trait_num} missing required keys for model {model!r}: {sorted(missing)}")
    extra = provided - required
    if extra:
        logger.warning(
            "phenotype_params%d has unexpected keys for model %r: %s (ignored)",
            trait_num,
            model,
            sorted(extra),
        )


def _simulate_one_trait(
    pedigree: pd.DataFrame,
    params: dict[str, Any],
    trait_num: int,
    seed_offset: int,
) -> np.ndarray:
    """Dispatch a single trait to the appropriate phenotype model.

    Args:
        pedigree:    DataFrame with liability columns and sex
        params:      simulation parameter dict
        trait_num:   1 or 2
        seed_offset: offset added to base seed

    Returns:
        Array of simulated event times, shape (n,)
    """
    model = params[f"phenotype_model{trait_num}"]
    phenotype_params = params.get(f"phenotype_params{trait_num}", {})
    seed = params["seed"] + seed_offset

    _validate_phenotype_params(model, phenotype_params, trait_num)

    if model in _ADULT_MODELS:
        sex = pedigree["sex"].values if "sex" in pedigree.columns else None
        generation = pedigree["generation"].values
        prevalence = _resolve_prevalence(params, trait_num, sex, generation)
        func = phenotype_adult_ltm if model == "adult_ltm" else phenotype_adult_cox
        return func(
            liability=pedigree[f"liability{trait_num}"].values,
            prevalence=prevalence,
            beta=params[f"beta{trait_num}"],
            cip_x0=phenotype_params.get("cip_x0", 50.0),
            cip_k=phenotype_params.get("cip_k", 0.2),
            seed=seed,
            standardize=params["standardize"],
            sex=sex,
            beta_sex=params.get(f"beta_sex{trait_num}", 0.0),
        )

    if model in _CURE_MODELS:
        sex = pedigree["sex"].values
        generation = pedigree["generation"].values
        prevalence = _resolve_prevalence(params, trait_num, sex, generation)
        hazard_params = {k: v for k, v in phenotype_params.items() if k != "baseline"}
        return phenotype_cure_frailty(
            liability=pedigree[f"liability{trait_num}"].values,
            prevalence=prevalence,
            beta=params[f"beta{trait_num}"],
            baseline=phenotype_params["baseline"],
            hazard_params=hazard_params,
            seed=seed,
            standardize=params["standardize"],
            sex=pedigree["sex"].values,
            beta_sex=params.get(f"beta_sex{trait_num}", 0.0),
        )

    if model in _FPT_MODELS:
        return phenotype_first_passage(
            liability=pedigree[f"liability{trait_num}"].values,
            beta=params[f"beta{trait_num}"],
            drift=phenotype_params["drift"],
            shape=phenotype_params["shape"],
            seed=seed,
            standardize=params["standardize"],
            sex=pedigree["sex"].values,
            beta_sex=params.get(f"beta_sex{trait_num}", 0.0),
        )

    # Frailty model
    return simulate_phenotype(
        liability=pedigree[f"liability{trait_num}"].values,
        beta=params[f"beta{trait_num}"],
        hazard_model=model,
        hazard_params=phenotype_params,
        seed=seed,
        standardize=params["standardize"],
        sex=pedigree["sex"].values,
        beta_sex=params.get(f"beta_sex{trait_num}", 0.0),
    )


def run_phenotype(pedigree: pd.DataFrame, params: dict[str, Any]) -> pd.DataFrame:
    """Simulate phenotype from pedigree and parameter dict.

    Expected keys in params:
        G_pheno, seed, standardize,
        phenotype_model1, phenotype_params1, beta1,
        phenotype_model2, phenotype_params2, beta2

    Args:
        pedigree: DataFrame with liability1, liability2, generation, …
        params:   simulation parameter dict (see above)

    Returns:
        phenotype DataFrame with columns t1, t2 (raw event times)
    """
    logger.info("Running phenotype simulation for %d individuals", len(pedigree))
    t0 = time.perf_counter()

    max_gen = pedigree["generation"].max()
    min_gen = max_gen - params["G_pheno"] + 1
    if min_gen < 0:
        raise ValueError(f"G_pheno ({params['G_pheno']}) exceeds available generations ({max_gen + 1})")
    pedigree = pedigree[pedigree["generation"] >= min_gen].reset_index(drop=True)

    t1 = _simulate_one_trait(pedigree, params, trait_num=1, seed_offset=0)
    t2 = _simulate_one_trait(pedigree, params, trait_num=2, seed_offset=100)

    phenotype = pd.DataFrame(
        {
            "id": pedigree["id"].values,
            "generation": pedigree["generation"].values,
            "sex": pedigree["sex"].values,
            "household_id": pedigree["household_id"].values,
            "mother": pedigree["mother"].values,
            "father": pedigree["father"].values,
            "twin": pedigree["twin"].values,
            "A1": pedigree["A1"].values,
            "C1": pedigree["C1"].values,
            "E1": pedigree["E1"].values,
            "liability1": pedigree["liability1"].values,
            "A2": pedigree["A2"].values,
            "C2": pedigree["C2"].values,
            "E2": pedigree["E2"].values,
            "liability2": pedigree["liability2"].values,
            "t1": t1,
            "t2": t2,
        }
    )

    logger.info(
        "Phenotype simulation complete in %.1fs: %d individuals",
        time.perf_counter() - t0,
        len(phenotype),
    )
    return phenotype


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def cli() -> None:
    """Command-line interface for phenotype simulation."""
    from sim_ace.core.cli_base import add_logging_args, init_logging

    parser = argparse.ArgumentParser(
        description="Simulate frailty phenotype with pluggable baseline hazard",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_logging_args(parser)
    parser.add_argument("--pedigree", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--G-pheno", type=int, default=3)
    parser.add_argument("--standardize", action="store_true", default=True)
    parser.add_argument("--prevalence1", type=float, default=0.10)
    parser.add_argument("--prevalence2", type=float, default=0.20)

    for k in (1, 2):
        g = parser.add_argument_group(f"Trait {k}")
        g.add_argument(f"--phenotype-model{k}", choices=_ALL_PHENOTYPE_MODELS, default="weibull")
        g.add_argument(f"--beta{k}", type=float, default=1.0)
        g.add_argument(f"--beta-sex{k}", type=float, default=0.0)
        # Frailty model parameters (user only needs to supply what their
        # chosen model requires).
        g.add_argument(f"--scale{k}", type=float, default=None)
        g.add_argument(f"--rho{k}", type=float, default=None)
        g.add_argument(f"--rate{k}", type=float, default=None)
        g.add_argument(f"--gamma{k}", type=float, default=None)
        g.add_argument(f"--mu{k}", type=float, default=None)
        g.add_argument(f"--sigma{k}", type=float, default=None)
        g.add_argument(f"--shape{k}", type=float, default=None)
        # FPT model parameters
        g.add_argument(f"--drift{k}", type=float, default=None)
        # ADuLT CIP parameters
        g.add_argument(f"--cip-x0-{k}", type=float, default=50.0)
        g.add_argument(f"--cip-k-{k}", type=float, default=0.2)

    args = parser.parse_args()
    init_logging(args)

    pm1 = args.phenotype_model1
    pm2 = args.phenotype_model2
    params = {
        "G_pheno": args.G_pheno,
        "seed": args.seed,
        "standardize": args.standardize,
        "phenotype_model1": pm1,
        "phenotype_model2": pm2,
        "prevalence1": args.prevalence1,
        "prevalence2": args.prevalence2,
        "beta1": args.beta1,
        "beta_sex1": args.beta_sex1,
        "phenotype_params1": _build_phenotype_params(args, trait=1),
        "beta2": args.beta2,
        "beta_sex2": args.beta_sex2,
        "phenotype_params2": _build_phenotype_params(args, trait=2),
    }

    pedigree = pd.read_parquet(args.pedigree)
    phenotype = run_phenotype(pedigree, params)
    save_parquet(phenotype, args.output)


def _build_phenotype_params(args: argparse.Namespace, trait: int) -> dict[str, float]:
    """Collect model-specific CLI floats into a phenotype_params dict."""
    model = getattr(args, f"phenotype_model{trait}")

    if model in _ADULT_MODELS:
        return {
            "cip_x0": getattr(args, f"cip_x0_{trait}"),
            "cip_k": getattr(args, f"cip_k_{trait}"),
        }

    # Frailty or FPT model
    required = _MODEL_PARAMS[model]
    params: dict[str, float] = {}
    for key in required:
        val = getattr(args, f"{key}{trait}", None)
        if val is None:
            raise ValueError(f"--{key}{trait} is required when --phenotype-model{trait}={model}")
        params[key] = val
    return params
