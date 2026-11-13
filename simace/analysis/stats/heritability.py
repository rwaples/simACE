"""Heritability-scale conversions: GE-covariance and SNP-like h² from variance primitives, and the Dempster-Lerner factor."""

from __future__ import annotations

__all__ = ["GeH2Metrics", "dempster_lerner_factor", "ge_h2_metrics"]

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.stats import norm


def _finite_float(value: Any) -> float:
    """Return value as float, or NaN for missing/non-numeric values."""
    if value is None:
        return float("nan")
    try:
        return float(value)
    except TypeError, ValueError:
        return float("nan")


@dataclass(frozen=True)
class GeH2Metrics:
    """One generation's gene-environment covariance and h² views of a trait (NaN where undefined).

    ``ge_cov_fraction`` is ``2 Cov(A, C+E) / Var(L)``, ``h2_realized_A`` is
    ``Var(A) / Var(L)``, ``h2_snp_like`` is ``(Var(A) + Cov)² / (Var(A) Var(L))``,
    ``var_non_genetic`` is ``Var(C+E)``, and ``null_sd`` is the sampling SD of
    ``ge_cov_fraction`` when the true covariance is zero.
    """

    ge_cov_fraction: float
    h2_realized_A: float
    h2_snp_like: float
    var_non_genetic: float
    null_sd: float


_NAN_METRICS = GeH2Metrics(*(float("nan"),) * 5)


def ge_h2_metrics(
    var_a: Any,
    var_liability: Any,
    cov_a_non_genetic: Any,
    n: Any = None,
) -> GeH2Metrics:
    """Derive the GE-covariance and SNP-like h² metrics from a generation's variance primitives.

    Inputs are per-generation primitives computed with population denominators
    (ddof=0). ``var_non_genetic`` is derived from
    ``Var(L) = Var(A) + Var(C+E) + 2 Cov(A,C+E)``, so only the primitives the
    report stores are needed.
    """
    a = _finite_float(var_a)
    var_l = _finite_float(var_liability)
    cov = _finite_float(cov_a_non_genetic)
    n_float = _finite_float(n)

    finite_core = np.isfinite([a, var_l, cov]).all()
    if not finite_core:
        return _NAN_METRICS

    var_u = var_l - a - 2.0 * cov
    if var_u < 0 and np.isclose(var_u, 0.0, atol=1e-12):
        var_u = 0.0

    ge_cov_fraction = 2.0 * cov / var_l if var_l != 0 else float("nan")
    h2_realized = a / var_l if var_l != 0 else float("nan")
    h2_snp_like = ((a + cov) ** 2) / (a * var_l) if a > 0 and var_l > 0 else float("nan")

    null_sd = float("nan")
    if n_float > 1 and a >= 0 and var_u >= 0 and var_l > 0:
        null_sd = abs(2.0 * np.sqrt(a * var_u) / var_l) / np.sqrt(n_float - 1.0)

    return GeH2Metrics(
        ge_cov_fraction=float(ge_cov_fraction),
        h2_realized_A=float(h2_realized),
        h2_snp_like=float(h2_snp_like),
        var_non_genetic=float(var_u),
        null_sd=float(null_sd),
    )


def dempster_lerner_factor(K: float) -> float:
    """K(1−K) / z(K)² where z(K) = φ(Φ⁻¹(1−K)).

    Converts observed-scale h² to liability-scale h² under LTM.
    """
    K = float(np.clip(K, 1e-3, 1.0 - 1e-3))
    z = float(norm.pdf(norm.ppf(1.0 - K)))
    if z <= 0:
        return float("nan")
    return K * (1.0 - K) / (z * z)
