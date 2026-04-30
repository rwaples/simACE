"""Baseline hazard registry for parametric survival models.

Each baseline hazard supplies a vectorized inverter that converts -log(U)
draws (Exp(1)) and a liability array into event times under the
proportional-hazards frailty model:

    L            = additive genetic + shared + unique liability
    z            = exp(scaled_beta * (L - mean))
    S(t | z)     = exp(-H0(t) * z)
    t            = H0^{-1}(-log(U) / z)

``compute_event_times`` is the public dispatch over the registry. The numba
kernels and per-distribution wrappers stay module-private; consumers should
go through ``compute_event_times`` (or, for tooling that needs the dispatch
table itself, ``BASELINE_HAZARDS``).

Supported distributions and required ``params`` keys:
    "weibull"     : {"scale": s, "rho": rho}
    "exponential" : {"rate": lam}  |  {"scale": s}
    "gompertz"    : {"rate": b, "gamma": g}
    "lognormal"   : {"mu": mu, "sigma": sigma}
    "loglogistic" : {"scale": alpha, "shape": k}
    "gamma"       : {"shape": k, "scale": theta}
"""

from __future__ import annotations

__all__ = [
    "BASELINE_HAZARDS",
    "BASELINE_PARAMS",
    "HAZARD_FLAG_ROOTS",
    "add_hazard_cli_args",
    "compute_event_times",
    "hazard_cli_flag_attrs",
    "parse_hazard_cli",
    "standardize_beta",
    "validate_hazard_params",
]

from typing import TYPE_CHECKING

import numpy as np
from numba import njit, prange
from scipy.stats import gamma as gamma_dist

from simace.core._numba_utils import _ndtri_approx

if TYPE_CHECKING:
    import argparse

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


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


BASELINE_HAZARDS = {
    "weibull": _invert_weibull,
    "exponential": _invert_exponential,
    "gompertz": _invert_gompertz,
    "lognormal": _invert_lognormal,
    "loglogistic": _invert_loglogistic,
    "gamma": _invert_gamma,
}

# Required parameter keys per distribution. Exponential accepts either
# "rate" or "scale" — only "rate" is listed as canonical; callers may
# substitute "scale" and the wrapper converts.
BASELINE_PARAMS: dict[str, list[str]] = {
    "weibull": ["scale", "rho"],
    "exponential": ["rate"],
    "gompertz": ["rate", "gamma"],
    "lognormal": ["mu", "sigma"],
    "loglogistic": ["scale", "shape"],
    "gamma": ["shape", "scale"],
}


# Union of every key any baseline distribution requires, plus exponential's
# alternate ``scale``.  Shared by FrailtyModel and CureFrailtyModel CLI plumbing.
HAZARD_FLAG_ROOTS: tuple[str, ...] = tuple(
    sorted({k for params in BASELINE_PARAMS.values() for k in params} | {"scale"})
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def validate_hazard_params(
    distribution: str,
    hazard_params: dict[str, float],
    model_name: str,
) -> None:
    """Validate distribution name and required ``hazard_params`` keys.

    Exponential accepts either ``rate`` or ``scale`` as the canonical key;
    others must contain every key listed in ``BASELINE_PARAMS[distribution]``.
    """
    if distribution not in BASELINE_HAZARDS:
        raise ValueError(
            f"unknown {model_name} distribution {distribution!r}; valid: {sorted(BASELINE_HAZARDS)}"
        )
    required = set(BASELINE_PARAMS[distribution])
    if distribution == "exponential" and "scale" in hazard_params:
        required = (required - {"rate"}) | {"scale"}
    missing = required - set(hazard_params)
    if missing:
        raise ValueError(
            f"{model_name} distribution {distribution!r} missing required hazard params: {sorted(missing)}"
        )


def add_hazard_cli_args(
    parser: argparse.ArgumentParser,
    trait: int,
    *,
    kebab_prefix: str,
    model_label: str,
) -> argparse._ArgumentGroup:
    """Register ``--{kebab_prefix}-distribution{trait}`` + one flag per HAZARD_FLAG_ROOTS.

    Returns the argument group so callers can attach model-specific flags
    (e.g. cure_frailty's ``--cure-frailty-prevalence{trait}``).
    """
    group = parser.add_argument_group(f"Trait {trait} — {model_label}")
    group.add_argument(
        f"--{kebab_prefix}-distribution{trait}",
        default=None,
        choices=sorted(BASELINE_HAZARDS),
        help=f"Baseline hazard for trait {trait} when phenotype-model{trait}={model_label}",
    )
    for flag_root in HAZARD_FLAG_ROOTS:
        group.add_argument(f"--{kebab_prefix}-{flag_root}{trait}", type=float, default=None)
    return group


def parse_hazard_cli(
    args: argparse.Namespace,
    trait: int,
    *,
    attr_prefix: str,
    kebab_prefix: str,
) -> tuple[str, dict[str, float]]:
    """Pull hazard-distribution + per-param values from argparse ``Namespace``.

    Returns ``(distribution, hazard_params)``.  Raises if the distribution flag
    is missing or any required per-distribution param flag is unset.
    """
    distribution = getattr(args, f"{attr_prefix}_distribution{trait}")
    if distribution is None:
        raise ValueError(
            f"--{kebab_prefix}-distribution{trait} is required when --phenotype-model{trait}={attr_prefix}"
        )
    hazard_params: dict[str, float] = {}
    for key in BASELINE_PARAMS[distribution]:
        val = getattr(args, f"{attr_prefix}_{key}{trait}", None)
        if val is None:
            raise ValueError(
                f"--{kebab_prefix}-{key}{trait} is required for --{kebab_prefix}-distribution{trait}={distribution}"
            )
        hazard_params[key] = val
    return distribution, hazard_params


def hazard_cli_flag_attrs(trait: int, *, attr_prefix: str) -> set[str]:
    """Return the argparse attrs registered by ``add_hazard_cli_args``."""
    attrs = {f"{attr_prefix}_distribution{trait}"}
    attrs.update(f"{attr_prefix}_{root}{trait}" for root in HAZARD_FLAG_ROOTS)
    return attrs


def standardize_beta(liability: np.ndarray, beta: float, standardize: bool) -> tuple[float, float]:
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


def compute_event_times(
    neg_log_u: np.ndarray,
    liability: np.ndarray,
    mean: float,
    scaled_beta: float,
    distribution: str,
    params: dict[str, float],
) -> np.ndarray:
    """Convert -log(U) draws to event times under the named baseline hazard.

    Args:
        neg_log_u:    -log(U) draws, U ~ Uniform(0, 1], shape (n,).
        liability:    quantitative liability, shape (n,).
        mean:         liability mean (used when standardize=True; pass 0.0 otherwise).
        scaled_beta:  liability coefficient on log-hazard (already divided by std
                      if standardize=True).
        distribution: baseline hazard name; one of ``BASELINE_HAZARDS`` keys.
        params:       distribution-specific parameter dict; see ``BASELINE_PARAMS``.

    Returns:
        Event-time array, shape (n,), clamped to ``[1e-10, 1e6]``.

    Raises:
        ValueError: unknown ``distribution`` name.
        KeyError:   missing required parameter for the selected distribution.
    """
    if distribution not in BASELINE_HAZARDS:
        raise ValueError(f"Unknown baseline hazard {distribution!r}; valid: {sorted(BASELINE_HAZARDS)}")
    return BASELINE_HAZARDS[distribution](neg_log_u, liability, mean, scaled_beta, params)
