"""Frailty (survival) correlation estimation — pluggable baseline hazard.

Moved from sim_ace.analysis.stats to fit_ace.frailty.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import pandas as pd


def compute_frailty_correlations(
    df: pd.DataFrame,
    frailty_params: dict[str, dict[str, Any]],
    pairs: dict[str, tuple[np.ndarray, np.ndarray]],
    n_quad: int = 20,
    seed: int = 42,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Compute pairwise frailty liability correlations for all relationship types.

    Args:
        df: phenotype DataFrame
        frailty_params: {
            "trait1": {"beta": float, "hazard_model": str, "hazard_params": dict},
            "trait2": {...}
          }
        pairs: pre-extracted relationship pair indices
        n_quad: quadrature nodes per dimension
        seed: random seed for pair subsampling

    Returns:
        (censored, uncensored) dicts keyed by "trait1"/"trait2"
    """
    from fit_ace.frailty.survival_stats import compute_pair_corr

    result: dict[str, Any] = {}
    result_uncensored: dict[str, Any] = {}
    for trait_num in [1, 2]:
        key = f"trait{trait_num}"
        params = frailty_params.get(key, {})
        if not params:
            result[key] = {}
            result_uncensored[key] = {}
            continue
        common = dict(
            df=df,
            trait_num=trait_num,
            beta=params["beta"],
            pairs=pairs,
            n_quad=n_quad,
            seed=seed,
            hazard_model=params["hazard_model"],
            hazard_params=params["hazard_params"],
        )
        result[key] = compute_pair_corr(**common)
        result_uncensored[key] = compute_pair_corr(**common, use_raw=True)
    return result, result_uncensored


def compute_frailty_cross_trait_corr(
    df: pd.DataFrame,
    frailty_params: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Compute cross-trait frailty liability correlation.

    Each individual is self-paired across traits. Returns
    (censored, uncensored, stratified) dicts with keys {r, se, n}.
    The stratified estimate uses inverse-variance weighting over generations.
    """
    from fit_ace.frailty.survival_stats import cross_trait_corr_se

    p1 = frailty_params.get("trait1", {})
    p2 = frailty_params.get("trait2", {})
    empty = {"r": None, "se": None, "n": 0}
    if not p1 or not p2:
        return empty, empty, empty

    def _run(t1, d1, t2, d2):
        return cross_trait_corr_se(
            t1,
            d1,
            t2,
            d2,
            beta1=p1["beta"],
            beta2=p2["beta"],
            hazard_model_1=p1["hazard_model"],
            hazard_params_1=p1["hazard_params"],
            hazard_model_2=p2["hazard_model"],
            hazard_params_2=p2["hazard_params"],
        )

    n = len(df)
    ones = np.ones(n, dtype=np.float64)

    r_cens, se_cens = _run(
        df["t_observed1"].values.astype(np.float64),
        df["affected1"].values.astype(np.float64),
        df["t_observed2"].values.astype(np.float64),
        df["affected2"].values.astype(np.float64),
    )
    r_uncens, se_uncens = _run(
        df["t1"].values.astype(np.float64),
        ones,
        df["t2"].values.astype(np.float64),
        ones,
    )

    # Generation-stratified with inverse-variance weighting
    r_strat = se_strat = np.nan
    gen_details: dict[str, Any] = {}
    if "generation" in df.columns:
        gen_rs, gen_ses = [], []
        for gen in sorted(df["generation"].unique()):
            g = df[df["generation"] == gen]
            r_g, se_g = _run(
                g["t_observed1"].values.astype(np.float64),
                g["affected1"].values.astype(np.float64),
                g["t_observed2"].values.astype(np.float64),
                g["affected2"].values.astype(np.float64),
            )
            gen_details[f"gen{gen}"] = {
                "r": float(r_g) if not np.isnan(r_g) else None,
                "se": float(se_g) if not np.isnan(se_g) else None,
                "n": len(g),
            }
            if not np.isnan(r_g) and not np.isnan(se_g) and se_g > 0:
                gen_rs.append(r_g)
                gen_ses.append(se_g)
        if gen_rs:
            rs = np.array(gen_rs)
            w = 1.0 / np.array(gen_ses) ** 2
            r_strat = float(np.sum(w * rs) / np.sum(w))
            se_strat = float(1.0 / np.sqrt(np.sum(w)))

    def _fmt(r, se):
        return {
            "r": float(r) if not np.isnan(r) else None,
            "se": float(se) if not np.isnan(se) else None,
            "n": n,
        }

    return (
        _fmt(r_cens, se_cens),
        _fmt(r_uncens, se_uncens),
        {**_fmt(r_strat, se_strat), "per_generation": gen_details},
    )
