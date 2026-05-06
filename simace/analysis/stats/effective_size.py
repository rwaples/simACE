"""Per-rep effective population size (Ne) summary for the stats runner.

Thin wrapper that invokes :func:`pedigree_graph.compute_all_ne` and
attaches scenario-level theoretical expectations from the rep's
``params.yaml`` when the configuration matches the canonical
random-mating, balanced-sex, ZTP-family regime.

The result is YAML-ready: each estimator key holds the dataclass
``to_dict()`` payload plus an ``expected`` field (``None`` when the
config has non-standard knobs such as assortative mating).
"""

from __future__ import annotations

__all__ = ["compute_effective_size", "theoretical_expectations"]

from typing import TYPE_CHECKING, Any

from pedigree_graph import PedigreeGraph, compute_all_ne

if TYPE_CHECKING:
    import pandas as pd


_NE_KEYS = (
    "ne_inbreeding",
    "ne_coancestry",
    "ne_variance_family_size",
    "ne_sex_ratio",
    "ne_individual_delta_f",
    "ne_long_term_contributions",
    "ne_hill_overlapping",
    "ne_caballero_toro",
)


def theoretical_expectations(config: dict[str, Any] | None) -> dict[str, float | None]:
    """Closed-form Ne expectations under standard random-mating assumptions.

    Returns a per-estimator dict.  Every entry is ``None`` when ``config``
    is missing, ``N`` is unknown, or the configuration includes a
    non-standard knob (currently: nonzero ``assort1`` / ``assort2``).

    Under random mating with 50/50 sex and a ZTP family-size
    distribution, all eight estimators converge to the cohort size ``N``
    *except* the long-term-contribution estimator: with the
    ``Ne = 1 / (2·Σ_f c_f²)`` form of Wray & Thompson 1990 used here, the
    analytic value for a symmetric Wright–Fisher pedigree is ``N / 2``.
    """
    if config is None:
        return dict.fromkeys(_NE_KEYS)

    assort1 = float(config.get("assort1") or 0.0)
    assort2 = float(config.get("assort2") or 0.0)
    if assort1 != 0.0 or assort2 != 0.0:
        return dict.fromkeys(_NE_KEYS)

    N = config.get("N")
    if N is None:
        return dict.fromkeys(_NE_KEYS)

    n = float(N)
    return {
        "ne_inbreeding": n,
        "ne_coancestry": n,
        "ne_variance_family_size": n,
        "ne_sex_ratio": n,
        "ne_individual_delta_f": n,
        "ne_long_term_contributions": n / 2.0,
        "ne_hill_overlapping": n,
        "ne_caballero_toro": n,
    }


def compute_effective_size(
    pedigree: pd.DataFrame | PedigreeGraph,
    config: dict[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    """Run all eight Ne estimators on ``pedigree`` and serialize to dicts.

    Args:
        pedigree: Either a pandas DataFrame with the standard pedigree
            columns or an already-built :class:`PedigreeGraph`.  Passing
            a graph avoids rebuilding it when the runner has already
            constructed one for relationship extraction.
        config: Per-rep params (e.g. loaded from ``params.yaml``).
            Used solely to derive theoretical expectations.

    Returns:
        Dict keyed on estimator name; each value is the matching
        dataclass's ``to_dict()`` payload merged with an ``expected``
        field (``float`` or ``None``).
    """
    pg = pedigree if isinstance(pedigree, PedigreeGraph) else PedigreeGraph(pedigree)
    raw = compute_all_ne(pg)
    expected = theoretical_expectations(config)
    out: dict[str, dict[str, Any]] = {}
    for name, result in raw.items():
        d = result.to_dict()
        d["expected"] = expected.get(name)
        out[name] = d
    return out
