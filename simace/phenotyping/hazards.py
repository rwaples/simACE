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
    "StandardizeMode",
    "add_hazard_cli_args",
    "add_standardize_hazard_cli_arg",
    "coerce_standardize_mode",
    "compute_event_times",
    "hazard_cli_flag_attrs",
    "iter_generation_groups",
    "parse_hazard_cli",
    "resolve_hazard_mode",
    "standardize_beta",
    "standardize_hazard_cli_attr",
    "standardize_liability",
    "validate_hazard_params",
]

from typing import TYPE_CHECKING, Literal

import numpy as np
from numba import njit, prange
from scipy.stats import gamma as gamma_dist

from simace.core._numba_utils import _ndtri_approx

if TYPE_CHECKING:
    import argparse
    from collections.abc import Iterator

StandardizeMode = Literal["none", "global", "per_generation"]
_VALID_STD_MODES: frozenset[str] = frozenset({"none", "global", "per_generation"})

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
# alternate ``scale``.
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
        raise ValueError(f"unknown {model_name} distribution {distribution!r}; valid: {sorted(BASELINE_HAZARDS)}")
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
    name: str,
) -> argparse._ArgumentGroup:
    """Register ``--{name}-distribution{trait}`` + one flag per HAZARD_FLAG_ROOTS.

    ``name`` is the kebab-form model name (e.g. ``"frailty"``, ``"cure-frailty"``).
    Returns the argument group so callers can attach model-specific flags
    (e.g. cure_frailty's ``--cure-frailty-prevalence{trait}``).
    """
    attr_name = name.replace("-", "_")
    group = parser.add_argument_group(f"Trait {trait} — {attr_name}")
    group.add_argument(
        f"--{name}-distribution{trait}",
        default=None,
        choices=sorted(BASELINE_HAZARDS),
        help=f"Baseline hazard for trait {trait} when phenotype-model{trait}={attr_name}",
    )
    for flag_root in HAZARD_FLAG_ROOTS:
        group.add_argument(f"--{name}-{flag_root}{trait}", type=float, default=None)
    return group


def parse_hazard_cli(
    args: argparse.Namespace,
    trait: int,
    *,
    name: str,
) -> tuple[str, dict[str, float]]:
    """Pull hazard-distribution + per-param values from argparse ``Namespace``.

    ``name`` is the kebab-form model name; the snake form is derived for attribute lookup.
    Returns ``(distribution, hazard_params)``.  Raises if the distribution flag is missing
    or any required per-distribution param flag is unset.
    """
    attr_name = name.replace("-", "_")
    distribution = getattr(args, f"{attr_name}_distribution{trait}")
    if distribution is None:
        raise ValueError(f"--{name}-distribution{trait} is required when --phenotype-model{trait}={attr_name}")
    hazard_params: dict[str, float] = {}
    for key in BASELINE_PARAMS[distribution]:
        val = getattr(args, f"{attr_name}_{key}{trait}", None)
        if val is None:
            raise ValueError(f"--{name}-{key}{trait} is required for --{name}-distribution{trait}={distribution}")
        hazard_params[key] = val
    return distribution, hazard_params


def hazard_cli_flag_attrs(trait: int, *, name: str) -> set[str]:
    """Return the argparse attrs registered by ``add_hazard_cli_args``."""
    attr_name = name.replace("-", "_")
    attrs = {f"{attr_name}_distribution{trait}"}
    attrs.update(f"{attr_name}_{root}{trait}" for root in HAZARD_FLAG_ROOTS)
    return attrs


def add_standardize_hazard_cli_arg(
    parser_or_group: argparse.ArgumentParser | argparse._ArgumentGroup,
    trait: int,
    *,
    name: str,
) -> None:
    """Register ``--{name}-standardize-hazard{trait}`` flag.

    The flag is namespaced per model so foreign-flag detection
    (:func:`check_no_foreign_flags`) cleanly distinguishes overrides
    intended for one model from those intended for another.
    """
    parser_or_group.add_argument(
        f"--{name}-standardize-hazard{trait}",
        default=None,
        choices=["none", "global", "per_generation"],
        help=(
            f"Per-trait override of the hazard-step standardization mode for trait {trait} "
            f"when phenotype-model{trait}={name.replace('-', '_')} (default: inherit from --standardize)"
        ),
    )


def standardize_hazard_cli_attr(trait: int, *, name: str) -> str:
    """Return the argparse attribute name for the standardize_hazard CLI flag."""
    attr_name = name.replace("-", "_")
    return f"{attr_name}_standardize_hazard{trait}"


def coerce_standardize_mode(value: object) -> StandardizeMode:
    """Resolve a user-supplied standardize value to one of the canonical modes.

    Accepts the legacy bool form (``True`` → ``"global"``, ``False`` → ``"none"``)
    or one of the three string modes. Raises ``ValueError`` otherwise.
    """
    if isinstance(value, bool):
        return "global" if value else "none"
    if isinstance(value, str) and value in _VALID_STD_MODES:
        return value  # type: ignore[return-value]
    raise ValueError(f"standardize must be one of {sorted(_VALID_STD_MODES)} or bool; got {value!r}")


def resolve_hazard_mode(
    standardize: StandardizeMode | bool,
    standardize_hazard: StandardizeMode | bool | None,
) -> StandardizeMode:
    """Pick the mode used for hazard-step beta/L scaling.

    ``standardize_hazard`` is the per-trait override stored inside
    ``phenotype_params{N}``. ``None`` means "inherit from the global
    ``standardize`` flag". Bools accepted on either argument for the same
    legacy reason as ``coerce_standardize_mode``.
    """
    if standardize_hazard is None:
        return coerce_standardize_mode(standardize)
    return coerce_standardize_mode(standardize_hazard)


def standardize_liability(
    liability: np.ndarray,
    mode: StandardizeMode | bool,
    generation: np.ndarray | None = None,
) -> np.ndarray:
    """Return liability transformed per ``mode``.

    Modes:
      * ``"none"`` — returned unchanged.
      * ``"global"`` — ``(L - L.mean()) / L.std()`` across all rows.
      * ``"per_generation"`` — same z-score applied within each unique
        ``generation`` value; ``generation`` must be supplied.

    A degenerate group (singleton or all-equal liability) gets
    ``L - mean`` rather than NaN; the caller's downstream comparison
    against an N(0,1) threshold then reduces to the centered raw value.
    """
    mode = coerce_standardize_mode(mode)
    if mode == "none":
        return liability
    if mode == "global":
        std = float(liability.std())
        mean = float(liability.mean())
        if std < 1e-12:
            return liability - mean
        return (liability - mean) / std
    if generation is None:
        raise ValueError("standardize_liability: generation is required for mode='per_generation'")
    out = np.asarray(liability, dtype=np.float64).copy()
    for g in np.unique(generation):
        mask = generation == g
        sub = liability[mask]
        m = float(sub.mean())
        s = float(sub.std())
        if s < 1e-12:
            out[mask] = sub - m
        else:
            out[mask] = (sub - m) / s
    return out


def standardize_beta(
    liability: np.ndarray,
    beta: float,
    mode: StandardizeMode | bool,
    generation: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return per-individual ``(mean, scaled_beta)`` arrays of length ``n``.

    Modes:
      * ``"none"`` — ``mean[i] = 0``, ``scaled_beta[i] = beta`` everywhere.
      * ``"global"`` — ``mean[i] = L.mean()``, ``scaled_beta[i] = beta/L.std()``
        broadcast to all rows (``scaled_beta = 0`` when std is degenerate).
      * ``"per_generation"`` — per individual filled from their generation's
        mean and ``beta / std_g``; ``generation`` must be supplied.

    Returning arrays (not scalars) lets callers slice ``mean[mask][0]`` /
    ``scaled_beta[mask][0]`` inside an ``iter_generation_groups`` loop and
    pass the per-group scalars to ``compute_event_times`` without branching.
    """
    mode = coerce_standardize_mode(mode)
    n = len(liability)
    if mode == "none":
        return np.zeros(n, dtype=np.float64), np.full(n, beta, dtype=np.float64)
    if mode == "global":
        std = float(liability.std())
        mean = float(liability.mean())
        scaled = 0.0 if std < 1e-12 else beta / std
        return np.full(n, mean, dtype=np.float64), np.full(n, scaled, dtype=np.float64)
    if generation is None:
        raise ValueError("standardize_beta: generation is required for mode='per_generation'")
    mean_arr = np.zeros(n, dtype=np.float64)
    beta_arr = np.zeros(n, dtype=np.float64)
    for g in np.unique(generation):
        mask = generation == g
        sub = liability[mask]
        m = float(sub.mean())
        s = float(sub.std())
        mean_arr[mask] = m
        beta_arr[mask] = 0.0 if s < 1e-12 else beta / s
    return mean_arr, beta_arr


def iter_generation_groups(
    mode: StandardizeMode | bool,
    generation: np.ndarray,
) -> Iterator[np.ndarray]:
    """Yield boolean masks for the per-group loop pattern in model ``simulate()``.

    Under ``"none"`` or ``"global"`` yields a single full-coverage mask, so
    the caller's loop runs once over all rows. Under ``"per_generation"``
    yields one mask per unique value of ``generation``.
    """
    mode = coerce_standardize_mode(mode)
    if mode != "per_generation":
        yield np.ones(len(generation), dtype=bool)
        return
    for g in np.unique(generation):
        yield generation == g


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
