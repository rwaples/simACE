"""The ``phenotype_params{N}`` keys each phenotype model accepts.

Config loading checks scenarios against these lists without importing the
models, which load numba (~0.3 s on every ``simace ls``). The models import
the same lists, so there is one copy.
"""

from __future__ import annotations

__all__ = [
    "ADULT_METHODS",
    "BASELINE_PARAMS",
    "HAZARD_ALIASES",
    "MODEL_FAMILIES",
    "ONSET_KINDS",
    "check_phenotype_params",
    "exponential_rate",
]

from typing import Any

MODEL_FAMILIES: frozenset[str] = frozenset({"frailty", "cure_frailty", "adult", "first_passage", "simple_ltm"})

# Required hazard parameter keys per baseline distribution.
BASELINE_PARAMS: dict[str, list[str]] = {
    "weibull": ["scale", "rho"],
    "exponential": ["rate"],
    "gompertz": ["rate", "gamma"],
    "lognormal": ["mu", "sigma"],
    "loglogistic": ["scale", "shape"],
    "gamma": ["shape", "scale"],
}

# Alternate keys a distribution accepts in place of a required one, as
# {distribution: {required key: alternate}}. Where both are given, the required
# key wins. Converting an alternate is the consumer's job (``exponential_rate``).
HAZARD_ALIASES: dict[str, dict[str, str]] = {"exponential": {"rate": "scale"}}

ADULT_METHODS: frozenset[str] = frozenset({"ltm", "cox"})
ONSET_KINDS: frozenset[str] = frozenset({"fixed", "normal"})

_MODEL_KEYS: dict[str, frozenset[str]] = {
    "frailty": frozenset({"distribution", "standardize_hazard"}),
    "cure_frailty": frozenset({"distribution", "standardize_hazard", "prevalence"}),
    "adult": frozenset({"method", "prevalence", "cip_x0", "cip_k", "standardize_hazard"}),
    "first_passage": frozenset({"drift", "shape", "standardize_hazard"}),
    "simple_ltm": frozenset({"prevalence", "onset"}),
}


def _hazard_keys(distribution: str) -> frozenset[str]:
    return frozenset(BASELINE_PARAMS[distribution]) | frozenset(HAZARD_ALIASES.get(distribution, {}).values())


def exponential_rate(params: dict[str, float]) -> float:
    """The exponential rate from ``params``: ``rate``, else ``1 / scale``."""
    if "rate" in params:
        return params["rate"]
    if "scale" in params:
        return 1.0 / params["scale"]
    raise ValueError("exponential: need 'rate' or 'scale'")


def check_phenotype_params(model: str, phenotype_params: dict[str, Any], where: str) -> None:
    """Raise ``ValueError`` naming every key ``model`` does not accept.

    ``frailty`` and ``cure_frailty`` also accept their distribution's hazard
    parameters. An unknown distribution is left for the caller's distribution
    check to report.
    """
    allowed = _MODEL_KEYS[model]
    if model in ("frailty", "cure_frailty"):
        distribution = phenotype_params.get("distribution")
        if distribution not in BASELINE_PARAMS:
            return
        allowed = allowed | _hazard_keys(distribution)
    unknown = set(phenotype_params) - allowed
    if unknown:
        raise ValueError(f"{where} for model {model!r} has unknown key(s) {sorted(unknown)}; valid: {sorted(allowed)}")
