"""Half-sibling structure and variance-component correlation checks."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from pedigree_graph import RELATIONSHIPS

from ._common import (
    _MIN_PAIRS_FOR_CORR,
    SIBLING_CATEGORIES,
    _corr_tolerance,
    _info,
    _result,
    category_cell,
    pair_correlation,
)
from .am_relatedness import resolve_expected_a_corr

if TYPE_CHECKING:
    import pandas as pd
    import polars as pl
    from pedigree_graph import RelationshipMoments

    from simace.core.pedigree_arrays import PedigreeArrays


def household_sibling_counts(df: pd.DataFrame | pl.DataFrame) -> dict[str, int]:
    """Count offspring with a maternal sibling and with a maternal half-sib, from household sizes.

    A household is one mother's offspring (CONTEXT.md), so the counts are
    O(N) from the ``mother``, ``father`` and ``twin`` columns:

    - ``n_offspring_with_sibs``: offspring with a known mother whose
      household holds another offspring besides themselves and their own
      co-twin. Only the individual's own co-twin is excluded.
    - ``n_offspring_with_maternal_half_sib``: offspring with a known mother
      whose household holds an offspring with a different father. An
      unknown father differs from every other father, including another
      unknown one, as the engine classifies such pairs as ``MHS``.

    Offspring with an unknown mother take part in neither count.
    """
    mothers = df["mother"].to_numpy().astype(np.int64)
    fathers = df["father"].to_numpy().astype(np.int64)
    twins = df["twin"].to_numpy().astype(np.int64)
    known = mothers != -1
    mothers, fathers, twins = mothers[known], fathers[known], twins[known]
    if mothers.size == 0:
        return {"n_offspring_with_sibs": 0, "n_offspring_with_maternal_half_sib": 0}

    households, household_of = np.unique(mothers, return_inverse=True)
    size = np.bincount(household_of, minlength=len(households))
    others = size[household_of] - 1 - (twins != -1)

    father_known = fathers != -1
    base = np.int64(max(int(fathers.max()), 0) + 1)
    known_pairs = np.unique(household_of[father_known] * base + fathers[father_known])
    n_fathers = np.bincount((known_pairs // base).astype(np.intp), minlength=len(households))
    n_unknown = np.bincount(household_of[~father_known], minlength=len(households))
    other_father = np.where(
        father_known,
        (n_fathers[household_of] >= 2) | (n_unknown[household_of] >= 1),
        others >= 1,
    )
    return {
        "n_offspring_with_sibs": int((others >= 1).sum()),
        "n_offspring_with_maternal_half_sib": int(other_father.sum()),
    }


def _validate_half_sib_correlations(
    df: pd.DataFrame | pl.DataFrame,
    ped: PedigreeArrays,
    sibling_moments: RelationshipMoments,
    A_params: dict[int, float],
    params: dict[str, Any],
    results: dict[str, Any],
) -> None:
    """Compute half-sib correlations for A, liability, and shared C.

    Pooling rule:
    - **A correlation** uses MHS ∪ PHS — both share kinship 0.25 for the
      additive component, so pooling is a sample-size win. Expected: 0.25
      (kinship) under random mating; under single-trait assortative mating it
      inflates to ``(1 + 2·mu_A + mu_A·r_ho)/4`` (see :mod:`.am_relatedness`).
      Both-trait AM skips the scored check.
    - **Liability and shared-C correlations** use PHS only. Maternal
      half-sibs share households, so MHS liability corr = 0.25·A + 1·C and
      MHS shared_C ≠ 0; PHS gives the clean expected formulas (0.25·A and 0).

    Every correlation is exact over all pairs (ADR 0020); ``n_pairs`` is the
    true pair count and the tolerance is evaluated with it.
    """
    pooled = category_cell(sibling_moments, "MHS", "PHS")
    phs = category_cell(sibling_moments, "PHS")
    n_pooled = int(pooled.counts)
    n_phs = int(phs.counts)

    if n_pooled >= _MIN_PAIRS_FOR_CORR:
        for t in [1, 2]:
            col = f"A{t}"
            obs = pair_correlation(pooled, col)
            # Half-sib A correlation: 2*kinship under random mating, AM-inflated
            # to (1 + 2*mu_A + mu_A*r_ho)/4 under single-trait assortment.
            expected_a, skip, info = resolve_expected_a_corr(
                df, ped, params, t, "HS", 2.0 * RELATIONSHIPS["MHS"].nominal_kinship
            )
            if expected_a is None:
                # Reported, not asserted: no single-trait formula under {skip}.
                results[f"half_sib_{col}_correlation"] = _info(
                    f"Half-sib (pooled MHS+PHS) {col} correlation: {obs:.4f} (not asserted — {skip})",
                    observed=float(obs),
                    n_pairs=n_pooled,
                )
                continue
            tol = _corr_tolerance(expected_a, n_pooled)
            ok = (A_params[t] == 0) if np.isnan(obs) else (abs(obs - expected_a) < tol)
            results[f"half_sib_{col}_correlation"] = _result(
                ok,
                f"Half-sib (pooled MHS+PHS) {col} correlation: {obs:.4f} (expected: {expected_a:.4f}, tol: {tol:.4f})",
                expected=float(expected_a),
                observed=float(obs),
                n_pairs=n_pooled,
                **info,
            )
    else:
        for t in [1, 2]:
            results[f"half_sib_A{t}_correlation"] = _result(
                True, f"Not enough pooled half-sib pairs ({n_pooled}) for A{t} correlation"
            )

    if n_phs >= _MIN_PAIRS_FOR_CORR:
        for t in [1, 2]:
            phs_pheno = pair_correlation(phs, f"P{t}")
            results[f"half_sib_liability{t}_correlation"] = _info(
                f"PHS liability{t} correlation: {phs_pheno:.4f} (expected ~0.25·A{t})",
                observed=float(phs_pheno),
                n_pairs=n_phs,
            )

            obs_c = pair_correlation(phs, f"C{t}")
            tol = _corr_tolerance(0.0, n_phs)
            ok_c = True if np.isnan(obs_c) else abs(obs_c) < tol
            results[f"half_sib_shared_C{t}"] = _result(
                ok_c,
                f"PHS shared C{t} correlation: {obs_c:.4f} (expected: ~0, tol: {tol:.4f})",
                expected=0.0,
                observed=float(obs_c),
                n_pairs=n_phs,
            )
    else:
        for t in [1, 2]:
            # Liability correlation is informational in the enough-data branch
            # above, so its insufficient-data fallback is informational too —
            # not a trivially-passing scored check.
            results[f"half_sib_liability{t}_correlation"] = _info(
                f"Not enough PHS pairs ({n_phs}) for liability{t} correlation"
            )
            results[f"half_sib_shared_C{t}"] = _result(True, f"Not enough PHS pairs ({n_phs}) for C{t} correlation")


def validate_half_sibs(
    df: pd.DataFrame | pl.DataFrame,
    params: dict[str, Any],
    ped: PedigreeArrays,
    sibling_moments: RelationshipMoments,
) -> dict[str, Any]:
    """Validate half-sibling structure under the mating-pair model.

    Reports observed counts and proportions of full-sib, maternal half-sib,
    and paternal half-sib pairs as informational checks. With a
    zero-truncated Poisson mating model, both maternal and paternal
    half-sibs arise naturally when individuals have multiple partners.

    Also computes half-sib variance-component correlations (A, liability,
    shared C) — see ``_validate_half_sib_correlations`` for pooling rules.

    Args:
        df: Pedigree DataFrame with columns id, mother, father, twin.
        params: Scenario parameters; requires keys ``mating_lambda``, ``A1``,
            ``A2``.
        ped: The same pedigree as id-addressable arrays; supplies the
            variance-component arrays for the correlation checks.
        sibling_moments: Relationship moments of ``df`` over ``FS``, ``MHS``
            and ``PHS`` with the ``A{t}``, ``C{t}`` and ``P{t}`` columns
            (:func:`~simace.analysis.validate._common.sibling_moments`).

    Returns:
        Dict of check-name to result dicts.
    """
    results: dict[str, Any] = {}

    n_full, n_mat, n_pat = (int(category_cell(sibling_moments, code).counts) for code in SIBLING_CATEGORIES)

    # Report sibling structure (informational — no closed-form expected value)
    total_maternal_pairs = n_full + n_mat
    if total_maternal_pairs > 0:
        observed_half_sib_prop = n_mat / total_maternal_pairs
        # Range check: at lambda=0.5, most people have 1 partner, so half-sibs
        # should be present but not dominant. Wide tolerance for any lambda.
        results["half_sib_pair_proportion"] = _info(
            f"Maternal half-sib pair proportion: {observed_half_sib_prop:.4f} "
            f"(full={n_full}, mat_hs={n_mat}, pat_hs={n_pat})",
            observed=float(observed_half_sib_prop),
            n_full_sib_pairs=n_full,
            n_maternal_half_sib_pairs=n_mat,
            n_paternal_half_sib_pairs=n_pat,
        )
    else:
        results["half_sib_pair_proportion"] = _info("No maternal sibling pairs to check")

    # Offspring with maternal half-sib (informational)
    household = household_sibling_counts(df)
    n_offspring_with_sibs = household["n_offspring_with_sibs"]
    n_offspring_with_hs = household["n_offspring_with_maternal_half_sib"]
    if n_offspring_with_sibs > 0:
        observed_frac = n_offspring_with_hs / n_offspring_with_sibs
        results["offspring_with_half_sib"] = _info(
            f"Offspring with maternal half-sib: {observed_frac:.4f} ({n_offspring_with_hs}/{n_offspring_with_sibs})",
            observed=float(observed_frac),
            n_offspring_with_half_sib=int(n_offspring_with_hs),
            n_offspring_with_sibs=int(n_offspring_with_sibs),
        )
    else:
        results["offspring_with_half_sib"] = _info("No offspring with maternal siblings to check")

    A_params = {1: params["A1"], 2: params["A2"]}
    _validate_half_sib_correlations(df, ped, sibling_moments, A_params, params, results)

    return results
