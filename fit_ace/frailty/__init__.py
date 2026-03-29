"""Frailty (survival) correlation estimation."""

from fit_ace.frailty.frailty_stats import (
    compute_frailty_correlations,
    compute_frailty_cross_trait_corr,
)

__all__ = [
    "compute_frailty_correlations",
    "compute_frailty_cross_trait_corr",
]
