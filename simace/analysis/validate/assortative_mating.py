"""Mate-correlation (assortative-mating) validation."""

from typing import Any

import numpy as np
import pandas as pd

from simace.core.numerics import safe_corrcoef
from simace.core.pedigree_arrays import PedigreeArrays

from ._common import _corr_se, _info, _result


def validate_assortative_mating(df: pd.DataFrame, params: dict[str, Any], ped: PedigreeArrays) -> dict[str, Any]:
    """Validate mate correlation on liability when assortative mating is configured.

    Extracts unique mating pairs from non-founders, computes Pearson
    correlation of mother and father liability for each trait, and checks
    against the configured ``assort1`` / ``assort2`` parameters.

    Args:
        df: Pedigree DataFrame.
        params: Scenario parameters; uses keys ``assort1``, ``assort2``.
        ped: The same pedigree as id-addressable arrays.

    Returns:
        Dict of check-name to result dicts.
    """
    results: dict[str, Any] = {}
    assort1 = params.get("assort1", 0.0)
    assort2 = params.get("assort2", 0.0)

    # Per-gen AM (dict) is not currently validated — observed pairs span
    # multiple generations with potentially different per-gen AM strengths,
    # so a single-value tolerance check is ill-defined. Skip with a
    # passing result. A future improvement: stratify pairs by generation
    # and validate each cohort against its per-gen target.
    if isinstance(assort1, dict) or isinstance(assort2, dict):
        msg = (
            "Per-generation assort1/assort2 (dict-valued); skipping pooled "
            "mate-correlation check (would conflate cohort-varying targets)."
        )
        results["mate_corr_liability1"] = _result(True, msg)
        results["mate_corr_liability2"] = _result(True, msg)
        return results

    non_founders = df[df["mother"] != -1]
    if len(non_founders) == 0:
        results["mate_corr_liability1"] = _result(True, "No non-founders to check")
        results["mate_corr_liability2"] = _result(True, "No non-founders to check")
        return results

    # Extract unique mating pairs
    pairs = non_founders[["mother", "father"]].drop_duplicates()
    mother_ids = pairs["mother"].values
    father_ids = pairs["father"].values
    n_pairs = len(pairs)

    for t, expected in [(1, assort1), (2, assort2)]:
        m_liab = ped.gather(f"liability{t}", mother_ids)
        f_liab = ped.gather(f"liability{t}", father_ids)
        obs = safe_corrcoef(m_liab, f_liab)

        if np.isnan(obs):
            results[f"mate_corr_liability{t}"] = _result(
                True,
                f"Cannot compute mate correlation for trait {t} (zero variance)",
                expected=float(expected),
                observed=float(obs),
            )
            continue

        se = _corr_se(expected, n_pairs)
        tol = max(0.1, 3 * se)
        ok = abs(obs - expected) < tol
        results[f"mate_corr_liability{t}"] = _result(
            ok,
            f"Mate correlation liability{t}: {obs:.4f} (expected: {expected}, tol: {tol:.4f})",
            expected=float(expected),
            observed=float(obs),
            n_pairs=n_pairs,
        )

    # Genetic (A-component) mate correlation mu_A — informational; consumed by
    # the AM-corrected relative-correlation reference lines (see am_relatedness
    # and plot_A_correlations). Reduces to ~0 with no assortment.
    for t in [1, 2]:
        m_a = ped.gather(f"A{t}", mother_ids)
        f_a = ped.gather(f"A{t}", father_ids)
        obs_a = safe_corrcoef(m_a, f_a)
        results[f"mate_corr_A{t}"] = _info(
            f"Genetic mate correlation A{t} (mu_A): {obs_a:.4f}",
            observed=None if np.isnan(obs_a) else float(obs_a),
            n_pairs=n_pairs,
        )

    # Cross-trait validation (only when both traits assort)
    if assort1 != 0 and assort2 != 0:
        am = params.get("assort_matrix")
        if am is not None:
            c_expected = float(np.asarray(am)[0, 1])
        else:
            rho_w = params.get("rA", 0) * np.sqrt(params.get("A1", 0) * params.get("A2", 0)) + params.get(
                "rC", 0
            ) * np.sqrt(params.get("C1", 0) * params.get("C2", 0))
            c_expected = rho_w * np.sqrt(abs(assort1 * assort2)) * np.sign(assort1 * assort2)

        for label, fi, mi in [("cross_12", 1, 2), ("cross_21", 2, 1)]:
            m_liab = ped.gather(f"liability{fi}", mother_ids)
            f_liab = ped.gather(f"liability{mi}", father_ids)
            obs = safe_corrcoef(m_liab, f_liab)

            if np.isnan(obs):
                results[f"mate_corr_{label}"] = _result(
                    True,
                    f"Cannot compute mate correlation {label} (zero variance)",
                    expected=float(c_expected),
                    observed=float(obs),
                )
                continue

            se = _corr_se(c_expected, n_pairs)
            tol = max(0.1, 3 * se)
            ok = abs(obs - c_expected) < tol
            results[f"mate_corr_{label}"] = _result(
                ok,
                f"Mate correlation {label}: {obs:.4f} (expected: {c_expected:.4f}, tol: {tol:.4f})",
                expected=float(c_expected),
                observed=float(obs),
                n_pairs=n_pairs,
            )

    return results
