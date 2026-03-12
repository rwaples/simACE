"""
Phenotype simulation for two correlated traits.

Supports three phenotype models (set via phenotype_model parameter):
  "frailty"    — Proportional hazards frailty model with pluggable baseline hazard (default)
  "adult_ltm"  — ADuLT liability threshold model (Pedersen et al., Nat Commun 2023)
  "adult_cox"  — ADuLT proportional hazards model (Weibull shape=2 + CIP→age mapping)

Frailty model converts liability to raw event times (age-at-onset) using a proportional
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

from scipy.stats import norm

from sim_ace.utils import save_parquet
from sim_ace.compute_hazard_terms import compute_hazard_terms

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Numba kernels — fuse frailty computation + inversion in a single pass
# ---------------------------------------------------------------------------

@njit(cache=True)
def _ndtri_approx(p):
    """Acklam rational approximation for the normal quantile (~1e-9 accuracy)."""
    a0, a1, a2 = -3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02
    a3, a4, a5 = 1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00
    b0, b1, b2 = -5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02
    b3, b4     = 6.680131188771972e+01, -1.328068155288572e+01
    c0, c1, c2 = -7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00
    c3, c4, c5 = -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00
    d0, d1, d2, d3 = 7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00, 3.754408661907416e+00
    p_low = 0.02425
    if p < p_low:
        q = np.sqrt(-2.0 * np.log(p))
        num = (((((c0*q + c1)*q + c2)*q + c3)*q + c4)*q + c5)
        den = ((((d0*q + d1)*q + d2)*q + d3)*q + 1.0)
        return num / den
    elif p <= 1.0 - p_low:
        q = p - 0.5
        r = q * q
        num = (((((a0*r + a1)*r + a2)*r + a3)*r + a4)*r + a5) * q
        den = (((((b0*r + b1)*r + b2)*r + b3)*r + b4)*r + 1.0)
        return num / den
    else:
        q = np.sqrt(-2.0 * np.log(1.0 - p))
        num = (((((c0*q + c1)*q + c2)*q + c3)*q + c4)*q + c5)
        den = ((((d0*q + d1)*q + d2)*q + d3)*q + 1.0)
        return -(num / den)


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
    return _nb_weibull(neg_log_u, liability, mean, scaled_beta,
                       params["scale"], 1.0 / params["rho"])


def _invert_exponential(neg_log_u, liability, mean, scaled_beta, params):
    rate = params["rate"] if "rate" in params else 1.0 / params["scale"]
    return _nb_exponential(neg_log_u, liability, mean, scaled_beta, 1.0 / rate)


def _invert_gompertz(neg_log_u, liability, mean, scaled_beta, params):
    b, g = params["rate"], params["gamma"]
    return _nb_gompertz(neg_log_u, liability, mean, scaled_beta, g / b, 1.0 / g)


def _invert_lognormal(neg_log_u, liability, mean, scaled_beta, params):
    return _nb_lognormal(neg_log_u, liability, mean, scaled_beta,
                         params["mu"], params["sigma"])


def _invert_loglogistic(neg_log_u, liability, mean, scaled_beta, params):
    return _nb_loglogistic(neg_log_u, liability, mean, scaled_beta,
                           params["scale"], 1.0 / params["shape"])


def _invert_gamma(neg_log_u, liability, mean, scaled_beta, params):
    """Gamma inverse — scipy iterative solver, not numba-fusible."""
    frailty = np.exp(scaled_beta * (liability - mean))
    target = neg_log_u / frailty
    t = gamma_dist.isf(np.exp(-target), params["shape"], scale=params["scale"])
    np.clip(t, 1e-10, 1e6, out=t)
    return t


# Dispatch table: model name → inversion function
_INVERTERS = {
    "weibull":     _invert_weibull,
    "exponential": _invert_exponential,
    "gompertz":    _invert_gompertz,
    "lognormal":   _invert_lognormal,
    "loglogistic": _invert_loglogistic,
    "gamma":       _invert_gamma,
}


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

    if standardize:
        std = np.std(liability)
        mean = liability.mean()
        scaled_beta = beta / std if std > 0 else 0.0
    else:
        mean = 0.0
        scaled_beta = beta

    neg_log_u = rng.exponential(size=len(liability))  # -log(U), U ~ (0,1]

    if beta_sex != 0.0 and sex is not None:
        neg_log_u = neg_log_u / np.exp(beta_sex * sex)

    return _INVERTERS[hazard_model](neg_log_u, liability, mean, scaled_beta,
                                    hazard_params)


# ---------------------------------------------------------------------------
# ADuLT phenotype models (Pedersen et al., Nat Commun 2023)
# ---------------------------------------------------------------------------

def phenotype_adult_ltm(
    liability: np.ndarray,
    prevalence: float,
    cip_x0: float = 50.0,
    cip_k: float = 0.2,
    seed: int = 42,
    standardize: bool = True,
) -> np.ndarray:
    """ADuLT liability threshold model: age-of-onset via logistic CIP.

    Cases (L > threshold): age = x₀ + (1/k)·log(Φ(−L) / (K − Φ(−L)))
    Controls (L ≤ threshold): t = 1e6 (censored downstream)

    Age is a deterministic function of liability — higher liability
    maps to younger onset age within cases.

    Args:
        liability:   quantitative liability, shape (n,)
        prevalence:  population prevalence K
        cip_x0:      logistic CIP midpoint age
        cip_k:       logistic CIP growth rate
        seed:        unused (kept for API compatibility)
        standardize: if True, standardize liability to N(0,1)

    Returns:
        Array of simulated event times, shape (n,)
    """
    L = liability.copy()
    if standardize:
        std = np.std(L)
        if std > 0:
            L = (L - L.mean()) / std

    threshold = norm.ppf(1.0 - prevalence)
    is_case = L > threshold

    t = np.full(len(L), 1e6)
    n_cases = is_case.sum()
    if n_cases > 0:
        # CIP inverse: cir = Φ(−L) = 1−Φ(L), age = x₀ + (1/k)·log(cir/(K−cir))
        cir = norm.sf(L[is_case])
        t[is_case] = cip_x0 + (1.0 / cip_k) * np.log(cir / (prevalence - cir))

    np.clip(t, 0.01, 1e6, out=t)
    return t


def phenotype_adult_cox(
    liability: np.ndarray,
    prevalence: float,
    cip_x0: float = 50.0,
    cip_k: float = 0.2,
    seed: int = 42,
    standardize: bool = True,
) -> np.ndarray:
    """ADuLT proportional hazards model: Weibull(shape=2) + CIP→age mapping.

    Raw time: t̃ = √(−log(U) / exp(L))
    Sorted by t_raw, running CIP = rank/(N+1) capped at K (prevalence).
    Cases (CIP < K): age = x₀ + (1/k)·log(CIP / (K − CIP))
    Controls (CIP ≥ K): t = 1e6 (censored downstream)

    Args:
        liability:    quantitative liability, shape (n,)
        prevalence:   population prevalence K (determines case fraction)
        cip_x0:       logistic CIP midpoint age
        cip_k:        logistic CIP growth rate
        seed:         random seed
        standardize:  if True, standardize liability to N(0,1)

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
    neg_log_u = -np.log(rng.uniform(size=n))
    t_raw = np.sqrt(neg_log_u / np.exp(L))

    # Sort by raw time; assign running CIP capped at prevalence
    order = np.argsort(t_raw)
    cip = (np.arange(1, n + 1)) / (n + 1)  # ranks in (0, 1)

    t = np.full(n, 1e6)
    is_case = cip < prevalence
    # Map case CIP to age via logistic CIP inverse: age = x₀ + (1/k)·log(CIP/(K−CIP))
    case_cip = cip[is_case]
    case_age = cip_x0 + (1.0 / cip_k) * np.log(case_cip / (prevalence - case_cip))
    t[order[is_case]] = case_age

    np.clip(t, 0.01, 1e6, out=t)
    return t


def run_phenotype(pedigree: pd.DataFrame, params: dict[str, Any]) -> pd.DataFrame:
    """Simulate phenotype from pedigree and parameter dict.

    Expected keys in params:
        G_pheno, seed, standardize,
        beta1, hazard_model1, hazard_params1,
        beta2, hazard_model2, hazard_params2

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
        raise ValueError(
            f"G_pheno ({params['G_pheno']}) exceeds available generations ({max_gen + 1})"
        )
    pedigree = pedigree[pedigree["generation"] >= min_gen].reset_index(drop=True)

    sex_vals = pedigree["sex"].values
    phenotype_model = params.get("phenotype_model", "frailty")

    if phenotype_model == "adult_ltm":
        t1 = phenotype_adult_ltm(
            liability  = pedigree["liability1"].values,
            prevalence = params["prevalence1"],
            cip_x0     = params.get("cip_x0", 50.0),
            cip_k      = params.get("cip_k", 0.2),
            seed       = params["seed"],
            standardize = params["standardize"],
        )
        t2 = phenotype_adult_ltm(
            liability  = pedigree["liability2"].values,
            prevalence = params["prevalence2"],
            cip_x0     = params.get("cip_x0", 50.0),
            cip_k      = params.get("cip_k", 0.2),
            seed       = params["seed"] + 100,
            standardize = params["standardize"],
        )
    elif phenotype_model == "adult_cox":
        t1 = phenotype_adult_cox(
            liability   = pedigree["liability1"].values,
            prevalence  = params["prevalence1"],
            cip_x0      = params.get("cip_x0", 50.0),
            cip_k       = params.get("cip_k", 0.2),
            seed        = params["seed"],
            standardize = params["standardize"],
        )
        t2 = phenotype_adult_cox(
            liability   = pedigree["liability2"].values,
            prevalence  = params["prevalence2"],
            cip_x0      = params.get("cip_x0", 50.0),
            cip_k       = params.get("cip_k", 0.2),
            seed        = params["seed"] + 100,
            standardize = params["standardize"],
        )
    else:
        # Default: frailty model
        t1 = simulate_phenotype(
            liability     = pedigree["liability1"].values,
            beta          = params["beta1"],
            hazard_model  = params["hazard_model1"],
            hazard_params = params["hazard_params1"],
            seed          = params["seed"],
            standardize   = params["standardize"],
            sex           = sex_vals,
            beta_sex      = params.get("beta_sex1", 0.0),
        )
        t2 = simulate_phenotype(
            liability     = pedigree["liability2"].values,
            beta          = params["beta2"],
            hazard_model  = params["hazard_model2"],
            hazard_params = params["hazard_params2"],
            seed          = params["seed"] + 100,
            standardize   = params["standardize"],
            sex           = sex_vals,
            beta_sex      = params.get("beta_sex2", 0.0),
        )

    phenotype = pd.DataFrame({
        "id":           pedigree["id"].values,
        "generation":   pedigree["generation"].values,
        "sex":          pedigree["sex"].values,
        "household_id": pedigree["household_id"].values,
        "mother":       pedigree["mother"].values,
        "father":       pedigree["father"].values,
        "twin":         pedigree["twin"].values,
        "A1":           pedigree["A1"].values,
        "C1":           pedigree["C1"].values,
        "E1":           pedigree["E1"].values,
        "liability1":   pedigree["liability1"].values,
        "A2":           pedigree["A2"].values,
        "C2":           pedigree["C2"].values,
        "E2":           pedigree["E2"].values,
        "liability2":   pedigree["liability2"].values,
        "t1":           t1,
        "t2":           t2,
    })

    logger.info(
        "Phenotype simulation complete in %.1fs: %d individuals",
        time.perf_counter() - t0, len(phenotype),
    )
    return phenotype


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

_MODELS = ["weibull", "exponential", "gompertz", "lognormal", "loglogistic", "gamma"]
_PHENOTYPE_MODELS = ["frailty", "adult_ltm", "adult_cox"]

# Parameter names required per model
_MODEL_PARAMS: dict[str, list[str]] = {
    "weibull":     ["scale", "rho"],
    "exponential": ["rate"],
    "gompertz":    ["rate", "gamma"],
    "lognormal":   ["mu", "sigma"],
    "loglogistic": ["scale", "shape"],
    "gamma":       ["shape", "scale"],
}


def cli() -> None:
    """Command-line interface for phenotype simulation."""
    from sim_ace.cli_base import add_logging_args, init_logging

    parser = argparse.ArgumentParser(
        description="Simulate frailty phenotype with pluggable baseline hazard",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_logging_args(parser)
    parser.add_argument("--pedigree",    required=True)
    parser.add_argument("--output",      required=True)
    parser.add_argument("--seed",        type=int,  default=42)
    parser.add_argument("--G-pheno",     type=int,  default=3)
    parser.add_argument("--standardize", action="store_true", default=True)
    parser.add_argument("--phenotype-model", choices=_PHENOTYPE_MODELS, default="frailty")

    # ADuLT shared CIP parameters
    parser.add_argument("--cip-x0",       type=float, default=50.0)
    parser.add_argument("--cip-k",        type=float, default=0.2)
    parser.add_argument("--prevalence1",  type=float, default=0.10)
    parser.add_argument("--prevalence2",  type=float, default=0.20)

    for k in (1, 2):
        g = parser.add_argument_group(f"Trait {k}")
        g.add_argument(f"--beta{k}",         type=float, default=1.0)
        g.add_argument(f"--beta-sex{k}",     type=float, default=0.0)
        g.add_argument(f"--hazard-model{k}",  choices=_MODELS, default="weibull")
        # One flag per model parameter; user only needs to supply what their
        # chosen model requires.
        g.add_argument(f"--scale{k}",  type=float, default=None)
        g.add_argument(f"--rho{k}",    type=float, default=None)
        g.add_argument(f"--rate{k}",   type=float, default=None)
        g.add_argument(f"--gamma{k}",  type=float, default=None)
        g.add_argument(f"--mu{k}",     type=float, default=None)
        g.add_argument(f"--sigma{k}",  type=float, default=None)
        g.add_argument(f"--shape{k}",  type=float, default=None)

    args = parser.parse_args()
    init_logging(args)

    phenotype_model = getattr(args, "phenotype_model")
    params = {
        "G_pheno":         args.G_pheno,
        "seed":            args.seed,
        "standardize":     args.standardize,
        "phenotype_model": phenotype_model,
        "cip_x0":          getattr(args, "cip_x0"),
        "cip_k":           getattr(args, "cip_k"),
        "prevalence1":     args.prevalence1,
        "prevalence2":     args.prevalence2,
        "beta1":           args.beta1,
        "beta_sex1":       getattr(args, "beta_sex1"),
        "hazard_model1":   getattr(args, "hazard_model1"),
        "hazard_params1":  _build_hazard_params(args, trait=1) if phenotype_model == "frailty" else {},
        "beta2":           args.beta2,
        "beta_sex2":       getattr(args, "beta_sex2"),
        "hazard_model2":   getattr(args, "hazard_model2"),
        "hazard_params2":  _build_hazard_params(args, trait=2) if phenotype_model == "frailty" else {},
    }

    pedigree  = pd.read_parquet(args.pedigree)
    phenotype = run_phenotype(pedigree, params)
    save_parquet(phenotype, args.output)


def _build_hazard_params(args: argparse.Namespace, trait: int) -> dict[str, float]:
    """Collect model-specific CLI floats into a hazard_params dict."""
    model    = getattr(args, f"hazard_model{trait}")
    required = _MODEL_PARAMS[model]
    params: dict[str, float] = {}
    for key in required:
        val = getattr(args, f"{key}{trait}", None)
        if val is None:
            raise ValueError(
                f"--{key}{trait} is required when --hazard-model{trait}={model}"
            )
        params[key] = val
    return params