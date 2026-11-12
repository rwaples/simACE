"""Parameter vocabularies shared by the phenotype models and their consumers.

Baseline hazard parameter names, the ``adult`` methods, and the ``simple_ltm``
onset kinds. Each model checks its own ``phenotype_params`` keys (see
``PhenotypeModel.from_config``).
"""

from __future__ import annotations

__all__ = [
    "ADULT_METHODS",
    "BASELINE_PARAMS",
    "HAZARD_ALIASES",
    "ONSET_KINDS",
    "exponential_rate",
]

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


def exponential_rate(params: dict[str, float]) -> float:
    """The exponential rate from ``params``: ``rate``, else ``1 / scale``."""
    if "rate" in params:
        return params["rate"]
    if "scale" in params:
        return 1.0 / params["scale"]
    raise ValueError("exponential: need 'rate' or 'scale'")
