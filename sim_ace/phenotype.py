"""
Frailty model phenotype simulation for two correlated traits.

Converts liability to raw event times (age-at-onset) using a proportional
hazards frailty model with pluggable baseline hazard.

Model (per trait):
    L        = A + C + E          (liability from pedigree)
    z        = exp(beta * L)      (frailty / hazard multiplier)
    S(t | z) = exp(-H0(t) * z)   (conditional survival)
    t        = H0^{-1}(-log(U) / z)  where U ~ Uniform(0, 1]

Baseline hazard models supported (via weibull_frailty.compute_hazard_terms):
    "weibull"     : {"scale": s, "rho": rho}
    "exponential" : {"rate": lam}  |  {"scale": s}
    "gompertz"    : {"rate": b, "gamma": g}
    "lognormal"   : {"mu": mu, "sigma": sigma}
    "loglogistic" : {"scale": alpha, "shape": k}
    "gamma"       : {"shape": k, "scale": theta}

Inversion strategy:
    Weibull   → analytic:   t = scale * (-log U / z)^(1/rho)
    All other → numerical:  scipy.optimize.brentq on H0(t)*z + log U = 0
"""

from __future__ import annotations

import argparse
import logging
import time
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from sim_ace.utils import save_parquet
from sim_ace.compute_hazard_terms import compute_hazard_terms

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Inversion helpers
# ---------------------------------------------------------------------------

def _invert_weibull(
    neg_log_u: np.ndarray,
    frailty: np.ndarray,
    params: dict[str, float],
) -> np.ndarray:
    """Analytic inverse CDF for Weibull: t = scale * ((-log U)/z)^(1/rho)."""
    return params["scale"] * (neg_log_u / frailty) ** (1.0 / params["rho"])


def _invert_generic(
    neg_log_u: np.ndarray,
    frailty: np.ndarray,
    hazard_model: str,
    params: dict[str, float],
    t_max: float = 1e6,
) -> np.ndarray:
    """Numerical inverse CDF via Brent's method for non-Weibull baselines.

    Solves H0(t) = (-log U) / z for each individual.
    If H0(t_max) < target (extremely low frailty), clips to t_max.
    """
    n = len(neg_log_u)
    t_out = np.empty(n, dtype=np.float64)

    # Evaluate H0 at the bracket ceiling once
    _, H_max_arr = compute_hazard_terms(hazard_model, np.array([t_max]), params)
    H_max = float(H_max_arr[0])

    for i in range(n):
        target = neg_log_u[i] / frailty[i]

        if H_max < target:
            t_out[i] = t_max
            continue

        def f(t: float) -> float:
            _, H = compute_hazard_terms(hazard_model, np.array([t]), params)
            return float(H[0]) - target

        try:
            t_out[i] = brentq(f, 1e-10, t_max, xtol=1e-6, maxiter=200)
        except ValueError:
            t_out[i] = t_max

    return t_out


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
) -> np.ndarray:
    """Convert liability array to simulated time-to-onset.

    Args:
        liability:     quantitative liability, shape (n,)
        beta:          effect of liability on log-hazard
        hazard_model:  baseline hazard model name
        hazard_params: model parameter dict (see module docstring)
        seed:          random seed
        standardize:   if True, standardize liability to mean 0 / std 1

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
        liability = (liability - liability.mean()) / std if std > 0 else liability - liability.mean()

    frailty   = np.exp(beta * liability)
    u         = 1.0 - rng.uniform(size=len(liability))   # sample from (0, 1]
    neg_log_u = -np.log(u)

    if hazard_model == "weibull":
        return _invert_weibull(neg_log_u, frailty, hazard_params)
    return _invert_generic(neg_log_u, frailty, hazard_model, hazard_params)


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

    # ----------------------------------------------------
    # Application of gene-environment interaction
    # ----------------------------------------------------
    gxe = params.get("gxe", {})
    model = gxe.get("model", "none")

    if model == "observed":
        G = pedigree[gxe["g_var"]].values
        E = pedigree[gxe["e_var"]].values
        pedigree["liability1"] = pedigree["liability1"] + gxe["gamma"] * (G * E)
        pedigree["liability2"] = pedigree["liability2"] + gxe["gamma"] * (G * E)

    elif model == "moderator":
        M = pedigree[gxe["m_var"]].values
        pedigree["liability1"] = pedigree["liability1"] * (1.0 + gxe["alpha"] * M)
        pedigree["liability2"] = pedigree["liability2"] * (1.0 + gxe["alpha"] * M)
    # ----------------------------------------------------

    t1 = simulate_phenotype(
        liability     = pedigree["liability1"].values,
        beta          = params["beta1"],
        hazard_model  = params["hazard_model1"],
        hazard_params = params["hazard_params1"],
        seed          = params["seed"],
        standardize   = params["standardize"],
    )
    t2 = simulate_phenotype(
        liability     = pedigree["liability2"].values,
        beta          = params["beta2"],
        hazard_model  = params["hazard_model2"],
        hazard_params = params["hazard_params2"],
        seed          = params["seed"] + 100,
        standardize   = params["standardize"],
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

    for k in (1, 2):
        g = parser.add_argument_group(f"Trait {k}")
        g.add_argument(f"--beta{k}",         type=float, required=True)
        g.add_argument(f"--hazard-model{k}",  choices=_MODELS, required=True)
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

    params = {
        "G_pheno":      args.G_pheno,
        "seed":         args.seed,
        "standardize":  args.standardize,
        "beta1":        args.beta1,
        "hazard_model1": getattr(args, "hazard_model1"),
        "hazard_params1": _build_hazard_params(args, trait=1),
        "beta2":        args.beta2,
        "hazard_model2": getattr(args, "hazard_model2"),
        "hazard_params2": _build_hazard_params(args, trait=2),
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