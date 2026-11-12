"""Baseline hazard registry for parametric survival models.

Each baseline hazard supplies a vectorized inverter that converts -log(U)
draws (Exp(1)) and a liability array into event times under the
proportional-hazards frailty model:

    L            = additive genetic + shared + unique liability
    z            = exp(scaled_beta * (L - mean))
    S(t | z)     = exp(-H0(t) * z)
    t            = H0^{-1}(-log(U) / z)

``compute_event_times`` is the public dispatch over the registry. The numba
kernels (in :mod:`simace.phenotype._kernels`) and per-distribution wrappers
stay private; consumers should go through ``compute_event_times`` (or, for
tooling that needs the dispatch table itself, ``BASELINE_HAZARDS``).

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
    "HAZARD_KEYS",
    "STANDARDIZE_CHOICES",
    "StandardizeMode",
    "coerce_standardize_mode",
    "compute_event_times",
    "hazard_config_keys",
    "iter_generation_groups",
    "resolve_hazard_mode",
    "standardize_beta",
    "standardize_liability",
    "true_lifetime_prevalence_weibull",
    "validate_hazard_params",
]

from typing import TYPE_CHECKING

import numpy as np

from simace.core.phenotype_keys import BASELINE_PARAMS, HAZARD_ALIASES, exponential_rate
from simace.core.standardize import STANDARDIZE_CHOICES, StandardizeMode, coerce_standardize_mode

if TYPE_CHECKING:
    from collections.abc import Iterator

# ---------------------------------------------------------------------------
# Python wrappers — unpack params dict, call numba kernel. The kernel module is
# imported per call so that importing this module does not load numba.
# ---------------------------------------------------------------------------


def _invert_weibull(neg_log_u, liability, mean, scaled_beta, params):
    from simace.phenotype._kernels import _nb_weibull

    return _nb_weibull(neg_log_u, liability, mean, scaled_beta, params["scale"], 1.0 / params["rho"])


def _invert_exponential(neg_log_u, liability, mean, scaled_beta, params):
    from simace.phenotype._kernels import _nb_exponential

    return _nb_exponential(neg_log_u, liability, mean, scaled_beta, 1.0 / exponential_rate(params))


def _invert_gompertz(neg_log_u, liability, mean, scaled_beta, params):
    from simace.phenotype._kernels import _nb_gompertz

    b, g = params["rate"], params["gamma"]
    return _nb_gompertz(neg_log_u, liability, mean, scaled_beta, g / b, 1.0 / g)


def _invert_lognormal(neg_log_u, liability, mean, scaled_beta, params):
    from simace.phenotype._kernels import _nb_lognormal

    return _nb_lognormal(neg_log_u, liability, mean, scaled_beta, params["mu"], params["sigma"])


def _invert_loglogistic(neg_log_u, liability, mean, scaled_beta, params):
    from simace.phenotype._kernels import _nb_loglogistic

    return _nb_loglogistic(neg_log_u, liability, mean, scaled_beta, params["scale"], 1.0 / params["shape"])


def _invert_gamma(neg_log_u, liability, mean, scaled_beta, params):
    """Gamma inverse — scipy iterative solver, not numba-fusible."""
    # Imported here, not at module level: scipy.stats costs ~0.5 s to import and
    # this is the only caller, so keep it off the per-job startup path.
    from scipy.stats import gamma as gamma_dist

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

# Union of every key any baseline distribution requires or accepts as an alternate.
HAZARD_KEYS: tuple[str, ...] = tuple(
    sorted(
        {k for params in BASELINE_PARAMS.values() for k in params}
        | {alt for aliases in HAZARD_ALIASES.values() for alt in aliases.values()}
    )
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def hazard_config_keys(distribution: str | None) -> frozenset[str]:
    """The ``phenotype_params`` hazard keys ``distribution`` accepts, alternates included.

    An unknown distribution accepts every hazard key, so the error names the
    distribution (``validate_hazard_params``), not its parameters.
    """
    if distribution not in BASELINE_PARAMS:
        return frozenset(HAZARD_KEYS)
    return frozenset(BASELINE_PARAMS[distribution]) | frozenset(HAZARD_ALIASES.get(distribution, {}).values())


#: Hazard parameters that may take any sign; every other one must be positive.
_SIGNED_HAZARD_PARAMS: frozenset[tuple[str, str]] = frozenset({("lognormal", "mu")})


def validate_hazard_params(
    distribution: str,
    hazard_params: dict[str, float],
    model_name: str,
) -> None:
    """Validate distribution name and required ``hazard_params`` keys.

    Every key in ``BASELINE_PARAMS[distribution]`` must be present, or its
    alternate in ``HAZARD_ALIASES``, and every value but lognormal ``mu``
    must be positive.
    """
    if distribution not in BASELINE_HAZARDS:
        raise ValueError(f"unknown {model_name} distribution {distribution!r}; valid: {sorted(BASELINE_HAZARDS)}")
    aliases = HAZARD_ALIASES.get(distribution, {})
    missing = {
        k for k in BASELINE_PARAMS[distribution] if k not in hazard_params and aliases.get(k) not in hazard_params
    }
    if missing:
        raise ValueError(
            f"{model_name} distribution {distribution!r} missing required hazard params: {sorted(missing)}"
        )
    for key, value in hazard_params.items():
        if (distribution, key) not in _SIGNED_HAZARD_PARAMS and not value > 0.0:
            raise ValueError(f"{model_name} distribution {distribution!r}: {key} must be > 0, got {value!r}")


def true_lifetime_prevalence_weibull(
    scale: float,
    rho: float,
    beta: float,
    max_age: float,
    n_quad: int = 64,
) -> float:
    """Marginal lifetime prevalence K of the Weibull-frailty model at ``max_age``.

    Under the simACE generative model (:func:`_nb_weibull`) an individual's onset
    time is ``T = scale * (E / z) ** (1 / rho)`` with ``E ~ Exp(1)`` and frailty
    ``z = exp(scaled_beta * (L - mean))``.  When the liability is standardized
    (``standardize`` in ``{"global", "per_generation"}``) it is N(0, 1) and
    ``scaled_beta = beta`` (see :func:`standardize_beta`), so the cumulative hazard
    is ``H0(t) = (t / scale) ** rho`` and

        K = 1 - E_{L ~ N(0, 1)}[ exp(-H0(max_age) * exp(beta * L)) ],

    evaluated here by Gauss-Hermite quadrature.

    This is the *uncensored* population lifetime prevalence at ``max_age``; it does
    not apply the age/death censoring the simulator later imposes, so the observed
    sample prevalence is lower.

    Precondition: an N(0, 1)-standardized liability (``standardize`` != ``"none"``);
    the caller is responsible for that.
    """
    from numpy.polynomial.hermite import hermgauss

    x, w = hermgauss(n_quad)
    liab = np.sqrt(2.0) * x  # nodes on the N(0, 1) scale
    wt = w / np.sqrt(np.pi)  # normalised weights (sum to 1)
    h0 = (float(max_age) / float(scale)) ** float(rho)
    integrand = np.exp(-h0 * np.exp(float(beta) * liab))
    return float(1.0 - (integrand * wt).sum())


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
