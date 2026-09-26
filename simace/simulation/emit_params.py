"""Echo scenario parameters to a YAML sidecar.

``params.yaml`` records a replicate's scenario provenance: an echo of the
resolved scenario config with no computation. Analyze, effective-size, and
the scenario atlas read it, and so does fitACE, so every key written here is
part of the fitACE compatibility contract: keys may be added, never renamed
or removed.

``E1`` / ``E2`` are guaranteed non-null by the config-load validator
(:func:`simace.config._validate_pedigree_config`). ``assort_matrix`` is
included only when not ``None``.
"""

from __future__ import annotations

__all__ = ["emit_params"]

from typing import Any

import simace
from simace.core.relationships import DEFAULT_MAX_DEGREE


def emit_params(
    *,
    seed: int,
    rep: int,
    A1: float,
    C1: float,
    E1: float,
    A2: float,
    C2: float,
    E2: float,
    rA: float,
    rC: float,
    rE: float,
    N: int,
    G_ped: int,
    G_sim: int | None,
    G_pheno: int,
    mating_model: str,
    mating_lambda: float,
    p_mztwin: float,
    assort1: float,
    assort2: float,
    max_degree: int = DEFAULT_MAX_DEGREE,
    skip_ne_coancestry: bool = True,
    assort_matrix: list[list[float]] | None = None,
) -> dict[str, Any]:
    """Build the params.yaml dict for a single replicate.

    Args:
        seed: per-replicate seed (already offset by rep upstream).
        rep: replicate number (1-based).
        A1: trait-1 additive-genetic variance.
        C1: trait-1 shared-environment variance.
        E1: trait-1 unique-environment variance.
        A2: trait-2 additive-genetic variance.
        C2: trait-2 shared-environment variance.
        E2: trait-2 unique-environment variance.
        rA: cross-trait genetic correlation.
        rC: cross-trait shared-environment correlation.
        rE: cross-trait unique-environment correlation.
        N: founder population size.
        G_ped: pedigree generations.
        G_sim: simulation generations including burn-in.
        G_pheno: phenotyped generations.
        mating_model: ``"standard"`` or ``"wright_fisher"``.  Recorded as
            scenario provenance; downstream consumers branch on this.
        mating_lambda: ZTP mating count parameter.
        p_mztwin: MZ twin probability.
        assort1: trait-1 assortative-mating correlation.
        assort2: trait-2 assortative-mating correlation.
        max_degree: Maximum relationship degree extracted by Analyze.
        skip_ne_coancestry: Whether the effective-size stage skips the
            coancestry-rate estimator.
        assort_matrix: optional 2x2 correlation matrix; included in the
            dict only when not ``None``.

    Returns:
        Dict to be serialized to ``params.yaml`` via :func:`dump_yaml`.
        Always carries ``simace_version`` (the installed ``simace``
        distribution version) for lockstep-family provenance.
    """
    out: dict[str, Any] = {
        "seed": seed,
        "rep": rep,
        "A1": A1,
        "C1": C1,
        "E1": E1,
        "A2": A2,
        "C2": C2,
        "E2": E2,
        "rA": rA,
        "rC": rC,
        "rE": rE,
        "N": N,
        "G_ped": G_ped,
        "G_sim": G_sim,
        "G_pheno": G_pheno,
        "mating_model": mating_model,
        "mating_lambda": mating_lambda,
        "p_mztwin": p_mztwin,
        "assort1": assort1,
        "assort2": assort2,
        "max_degree": max_degree,
        "skip_ne_coancestry": skip_ne_coancestry,
        "simace_version": simace.__version__,
    }
    if assort_matrix is not None:
        out["assort_matrix"] = assort_matrix
    return out
