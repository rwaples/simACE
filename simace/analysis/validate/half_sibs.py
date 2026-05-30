"""Half-sibling structure and variance-component correlation checks."""

from typing import Any

import numpy as np
import pandas as pd

from simace.core.numerics import safe_corrcoef

from ._common import (
    _DEFAULT_RNG_SEED,
    _MIN_PAIRS_FOR_CORR,
    _corr_tolerance,
    _extract_comp_vals,
    _result,
    _subsample_pairs,
)


def _sib_counts_from_pairs(
    sibling_pairs: dict[str, tuple[np.ndarray, np.ndarray]],
) -> dict[str, int]:
    """Derive sibling counts from pre-extracted pair arrays."""
    full = sibling_pairs["FS"]
    mat = sibling_pairs["MHS"]
    pat = sibling_pairs["PHS"]
    n_full = len(full[0])
    n_mat = len(mat[0])
    n_pat = len(pat[0])

    # Individuals with any maternal sibling (full or half)
    maternal_parts: list[np.ndarray] = []
    if n_full > 0:
        maternal_parts.extend([full[0], full[1]])
    if n_mat > 0:
        maternal_parts.extend([mat[0], mat[1]])
    n_with_sibs = len(np.unique(np.concatenate(maternal_parts))) if maternal_parts else 0

    # Individuals with a maternal half-sib
    if n_mat > 0:
        n_with_mat_hs = len(np.unique(np.concatenate([mat[0], mat[1]])))
    else:
        n_with_mat_hs = 0

    return {
        "n_full_sib_pairs": n_full,
        "n_maternal_half_sib_pairs": n_mat,
        "n_paternal_half_sib_pairs": n_pat,
        "n_offspring_with_sibs": n_with_sibs,
        "n_offspring_with_maternal_half_sib": n_with_mat_hs,
    }


def _validate_half_sib_correlations(
    sibling_pairs: dict[str, tuple[np.ndarray, np.ndarray]],
    comp_vals: dict[str, np.ndarray],
    A_params: dict[int, float],
    rng: np.random.Generator,
    results: dict[str, Any],
) -> None:
    """Compute half-sib correlations for A, liability, and shared C.

    Pooling rule:
    - **A correlation** uses MHS ∪ PHS — both share kinship 0.25 for the
      additive component, so pooling is a sample-size win. Expected: 0.25
      (kinship), regardless of A's variance share.
    - **Liability and shared-C correlations** use PHS only. Maternal
      half-sibs share households, so MHS liability corr = 0.25·A + 1·C and
      MHS shared_C ≠ 0; PHS gives the clean expected formulas (0.25·A and 0).
    """
    pooled_idx1 = np.concatenate([sibling_pairs["MHS"][0], sibling_pairs["PHS"][0]])
    pooled_idx2 = np.concatenate([sibling_pairs["MHS"][1], sibling_pairs["PHS"][1]])
    pooled_idx1, pooled_idx2, n_pooled = _subsample_pairs(pooled_idx1, pooled_idx2, rng)
    phs_idx1, phs_idx2, n_phs = _subsample_pairs(sibling_pairs["PHS"][0], sibling_pairs["PHS"][1], rng)

    expected_a = 0.25
    if n_pooled >= _MIN_PAIRS_FOR_CORR:
        for t in [1, 2]:
            col = f"A{t}"
            obs = safe_corrcoef(comp_vals[col][pooled_idx1], comp_vals[col][pooled_idx2])
            tol = _corr_tolerance(expected_a, n_pooled)
            ok = (A_params[t] == 0) if np.isnan(obs) else (abs(obs - expected_a) < tol)
            results[f"half_sib_{col}_correlation"] = _result(
                ok,
                f"Half-sib (pooled MHS+PHS) {col} correlation: {obs:.4f} (expected: {expected_a}, tol: {tol:.4f})",
                expected=expected_a,
                observed=float(obs),
                n_pairs=n_pooled,
            )
    else:
        for t in [1, 2]:
            results[f"half_sib_A{t}_correlation"] = _result(
                True, f"Not enough pooled half-sib pairs ({n_pooled}) for A{t} correlation"
            )

    if n_phs >= _MIN_PAIRS_FOR_CORR:
        for t in [1, 2]:
            P1 = comp_vals[f"A{t}"][phs_idx1] + comp_vals[f"C{t}"][phs_idx1] + comp_vals[f"E{t}"][phs_idx1]
            P2 = comp_vals[f"A{t}"][phs_idx2] + comp_vals[f"C{t}"][phs_idx2] + comp_vals[f"E{t}"][phs_idx2]
            phs_pheno = safe_corrcoef(P1, P2)
            results[f"half_sib_liability{t}_correlation"] = {
                "observed": float(phs_pheno),
                "details": f"PHS liability{t} correlation: {phs_pheno:.4f} (expected ~0.25·A{t})",
                "n_pairs": n_phs,
            }

            c_col = f"C{t}"
            obs_c = safe_corrcoef(comp_vals[c_col][phs_idx1], comp_vals[c_col][phs_idx2])
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
            results[f"half_sib_liability{t}_correlation"] = _result(
                True, f"Not enough PHS pairs ({n_phs}) for liability{t} correlation"
            )
            results[f"half_sib_shared_C{t}"] = _result(True, f"Not enough PHS pairs ({n_phs}) for C{t} correlation")


def validate_half_sibs(
    df: pd.DataFrame,
    params: dict[str, Any],
    df_indexed: pd.DataFrame,
    sibling_pairs: dict[str, tuple[np.ndarray, np.ndarray]],
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
            ``A2``, ``seed``.
        df_indexed: Pedigree DataFrame indexed by ``id``; supplies the
            variance-component arrays for the correlation checks.
        sibling_pairs: Dict with keys ``FS``, ``MHS``, ``PHS`` mapping to
            ``(idx1, idx2)`` row-index arrays.

    Returns:
        Dict of check-name to result dicts.
    """
    results: dict[str, Any] = {}

    sib_info = _sib_counts_from_pairs(sibling_pairs)

    # Report sibling structure (informational — no closed-form expected value)
    total_maternal_pairs = sib_info["n_full_sib_pairs"] + sib_info["n_maternal_half_sib_pairs"]
    if total_maternal_pairs > 0:
        observed_half_sib_prop = sib_info["n_maternal_half_sib_pairs"] / total_maternal_pairs
        # Range check: at lambda=0.5, most people have 1 partner, so half-sibs
        # should be present but not dominant. Wide tolerance for any lambda.
        results["half_sib_pair_proportion"] = _result(
            True,
            f"Maternal half-sib pair proportion: {observed_half_sib_prop:.4f} "
            f"(full={sib_info['n_full_sib_pairs']}, mat_hs={sib_info['n_maternal_half_sib_pairs']}, "
            f"pat_hs={sib_info['n_paternal_half_sib_pairs']})",
            observed=float(observed_half_sib_prop),
            n_full_sib_pairs=int(sib_info["n_full_sib_pairs"]),
            n_maternal_half_sib_pairs=int(sib_info["n_maternal_half_sib_pairs"]),
            n_paternal_half_sib_pairs=int(sib_info["n_paternal_half_sib_pairs"]),
        )
    else:
        results["half_sib_pair_proportion"] = _result(True, "No maternal sibling pairs to check")

    # Offspring with maternal half-sib (informational)
    n_offspring_with_sibs = sib_info["n_offspring_with_sibs"]
    n_offspring_with_hs = sib_info["n_offspring_with_maternal_half_sib"]
    if n_offspring_with_sibs > 0:
        observed_frac = n_offspring_with_hs / n_offspring_with_sibs
        results["offspring_with_half_sib"] = _result(
            True,
            f"Offspring with maternal half-sib: {observed_frac:.4f} ({n_offspring_with_hs}/{n_offspring_with_sibs})",
            observed=float(observed_frac),
            n_offspring_with_half_sib=int(n_offspring_with_hs),
            n_offspring_with_sibs=int(n_offspring_with_sibs),
        )
    else:
        results["offspring_with_half_sib"] = _result(True, "No non-twin offspring with siblings to check")

    comp_vals = _extract_comp_vals(df_indexed)
    A_params = {1: params["A1"], 2: params["A2"]}
    rng = np.random.default_rng(params.get("seed", _DEFAULT_RNG_SEED))
    _validate_half_sib_correlations(sibling_pairs, comp_vals, A_params, rng, results)

    return results
