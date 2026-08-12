"""Assortative-mating additive-variance equilibrium validation.

Under phenotypic assortative mating the additive genetic variance inflates
across generations to the Bulmer equilibrium ``a²`` — identical to the
assortative-mating-only result of Herzig et al. (2026, *Theor. Popul. Biol.*
170:26–35, doi:10.1016/j.tpb.2026.06.003). This module checks that the simulated
``Var(A)`` at the final recorded generation matches the value the infinitesimal
recursion predicts after ``G_sim`` reproduce steps (see
:mod:`simace.simulation.am_equilibrium` for the theory).

The final recorded generation has undergone exactly ``G_sim`` reproduce steps
from the founders, so the recursion is evaluated at ``G_sim`` regardless of any
burn-in. The asymptotic equilibrium ``a²`` is reported for context but the
assertion is against the recursion value at the actual ``G_sim`` (valid whether
or not the run has converged).
"""

from typing import Any

import numpy as np
import pandas as pd

from simace.simulation.am_equilibrium import am_equilibrium_variance, am_variance_trajectory

from ._common import _result
from .am_relatedness import am_relatedness_mode


def validate_am_equilibrium(df: pd.DataFrame, params: dict[str, Any]) -> dict[str, Any]:
    """Validate that ``Var(A)`` reaches the AM-inflated equilibrium per trait.

    Emits no checks when assortative mating is inactive (Wright-Fisher, or both
    ``assort1`` and ``assort2`` zero). For an assorting trait, asserts the
    observed final-generation ``Var(A)`` against the infinitesimal recursion
    prediction; reports the closed-form equilibrium ``a²`` alongside.

    Skips (passing, with a reason) the per-trait check when the equilibrium is
    ill-defined or not modelled by the univariate recursion: per-generation
    (dict-valued) AM / C / E, or both-trait assortment (cross-trait paths).

    Args:
        df: Pedigree DataFrame with ``id`` and ``A1``/``A2`` columns.
        params: Scenario parameters; uses ``assort1``/``assort2``, ``A{t}``,
            ``C{t}``, ``E{t}``, ``N``, ``G_ped``, ``G_sim``, ``mating_model``.

    Returns:
        Dict of check-name to result dicts (possibly empty).
    """
    results: dict[str, Any] = {}

    if params.get("mating_model", "standard") != "standard":
        return results  # Wright-Fisher has no assortative mating

    assort1 = params.get("assort1", 0.0)
    assort2 = params.get("assort2", 0.0)
    if not assort1 and not assort2:
        return results  # no AM configured -> nothing to validate

    N = params.get("N")
    G_ped = params.get("G_ped")
    if N is None or G_ped is None:
        return results
    G_sim = params.get("G_sim") or G_ped

    gen_labels = df["id"].values // int(N)
    last_mask = gen_labels == (int(G_ped) - 1)
    n_last = int(last_mask.sum())

    for t in (1, 2):
        mode = am_relatedness_mode(params, t)
        if mode == "none":
            continue
        if mode == "bivariate":
            results[f"am_equilibrium_A{t}"] = _result(
                True,
                f"Both-trait AM active; AM Var(A{t}) equilibrium not validated "
                f"(cross-trait paths not in the univariate recursion).",
            )
            continue

        assort_t = params.get(f"assort{t}", 0.0)
        A_base = params.get(f"A{t}")
        C = params.get(f"C{t}")
        E = params.get(f"E{t}")

        if isinstance(assort_t, dict) or isinstance(C, dict) or isinstance(E, dict):
            results[f"am_equilibrium_A{t}"] = _result(
                True,
                f"Per-generation AM/C/E (dict-valued); AM equilibrium ill-defined, skipping trait {t}.",
            )
            continue

        if A_base is None or float(A_base) <= 0.0 or n_last < 2:
            continue

        A_base = float(A_base)
        V_env = float(C or 0.0) + float(E or 0.0)
        r_ho = float(assort_t)

        V_pred = float(am_variance_trajectory(A_base, V_env, r_ho, int(G_sim))[-1])
        a2 = float(am_equilibrium_variance(A_base, V_env, r_ho))
        obs = float(np.var(df[f"A{t}"].values[last_mask]))

        # SE of a sample variance for ~Gaussian A: V·sqrt(2/(n-1)). The 5x
        # multiplier (floored at 0.03) absorbs the slight non-normality of A
        # under AM and accumulated process noise along the trajectory; a real
        # transmission bug shifts Var(A) far beyond this.
        se = V_pred * np.sqrt(2.0 / max(n_last - 1, 1))
        tol = max(0.03, 5.0 * se)
        ok = abs(obs - V_pred) < tol

        results[f"am_equilibrium_A{t}"] = _result(
            ok,
            f"AM Var(A{t}) final generation: {obs:.4f} (predicted {V_pred:.4f} "
            f"after {int(G_sim)} gens; equilibrium a²={a2:.4f}; tol {tol:.4f})",
            expected=V_pred,
            observed=obs,
            equilibrium=a2,
            baseline=A_base,
            mate_corr=r_ho,
            n_steps=int(G_sim),
            n_individuals=n_last,
        )

    return results
