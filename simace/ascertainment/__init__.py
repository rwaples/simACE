"""Unified ascertainment stage: random dropout + case-weighted N_sample draw.

Replaces the legacy two-stage design (pre-phenotype pedigree dropout +
post-censor subsampling). Per ADR 0001: ascertainment writes the canonical
post-stage ``pedigree.parquet`` and ``trait.parquet`` outputs that both
simACE-stats and fitACE consume.
"""

__all__ = ["run_ascertainment"]

import argparse
import logging
import time

import numpy as np
import pandas as pd

from simace.core.parquet import save_parquet
from simace.core.pedigree_filter import filter_pedigree_to_observed

logger = logging.getLogger(__name__)


def _sever_dangling_links(df: pd.DataFrame, valid_ids: np.ndarray) -> pd.DataFrame:
    """Rewrite mother/father/twin references pointing outside ``valid_ids`` to -1."""
    result = df.copy()
    for col in ("mother", "father", "twin"):
        if col not in result.columns:
            continue
        vals = result[col].to_numpy()
        in_valid = np.isin(vals, valid_ids)
        dangling = ~in_valid & (vals >= 0)
        if dangling.any():
            result.loc[result.index[dangling], col] = -1
    return result


def run_ascertainment(
    pedigree: pd.DataFrame,
    trait: pd.DataFrame,
    trait_simple_ltm: pd.DataFrame,
    *,
    dropout_rate: float = 0.0,
    case_ascertainment_ratio: float = 1.0,
    N_sample: int = 0,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Apply dropout + case-weighted sampling and return the ascertained subset.

    Two explicit steps, applied to IDs not weights, so ``dropout_rate``
    has effect even when ``N_sample > 0``:

    1. Uniform pedigree dropout: ``round(N_total * dropout_rate)`` individuals
       are removed uniformly at random from the full pedigree.
    2. Case-weighted draw from the post-dropout trait population:
       ``N_sample`` IDs are drawn with weights ``case_ascertainment_ratio``
       for cases vs ``1`` for controls. Pass-through when ``N_sample <= 0``
       or ``N_sample >= len(post-dropout trait)``.

    The same sampled IDs are applied to both trait branches, guaranteeing
    identical ``id`` columns in ``trait.parquet`` and ``trait.simple_ltm.parquet``.

    The pedigree output is the ancestor closure of the sampled IDs within
    the post-dropout pedigree, with all dangling parent/twin references
    rewritten to ``-1`` against the final closure ID set. Parent links are
    safe by construction (closure follows them); twin links may dangle and
    require the explicit fixup.

    Args:
        pedigree: Full pre-ascertainment pedigree (post-burn-in, pre-dropout).
        trait: Per-individual censored trait observations (G_pheno generations).
        trait_simple_ltm: Parallel simple-LTM trait observations.
        dropout_rate: Fraction of pedigree to remove uniformly at random.
        case_ascertainment_ratio: Sampling weight for cases relative to controls.
        N_sample: Target sample size; ``<=0`` or ``>= post-dropout pool`` passes everything through.
        seed: RNG seed.

    Returns:
        Tuple of (pedigree_ascertained, trait_ascertained, trait_simple_ltm_ascertained).
    """
    n_total = len(pedigree)
    rate = float(dropout_rate)
    ratio = float(case_ascertainment_ratio)
    n_sample = int(N_sample)
    seed = int(seed)

    if rate < 0 or rate >= 1:
        raise ValueError(f"dropout_rate must be in [0, 1), got {rate}")
    if ratio < 0:
        raise ValueError(f"case_ascertainment_ratio must be >= 0, got {ratio}")

    rng = np.random.default_rng(seed)
    t0 = time.perf_counter()

    # Step 1: uniform dropout from the full pedigree.
    n_drop = round(n_total * rate)
    if n_drop > 0:
        if n_drop >= n_total:
            raise ValueError(f"dropout_rate {rate} would remove all {n_total} individuals")
        drop_idx = rng.choice(n_total, n_drop, replace=False)
        keep_mask = np.ones(n_total, dtype=bool)
        keep_mask[drop_idx] = False
        ped_post_dropout = pedigree.loc[keep_mask].reset_index(drop=True)
    else:
        ped_post_dropout = pedigree
    ped_post_dropout_ids = ped_post_dropout["id"].to_numpy()

    # Step 2: intersect trait branches with dropout survivors.
    trait_post_dropout = trait[trait["id"].isin(ped_post_dropout_ids)].reset_index(drop=True)
    simple_ltm_post_dropout = trait_simple_ltm[trait_simple_ltm["id"].isin(ped_post_dropout_ids)].reset_index(drop=True)

    # Step 3: case-weighted draw from the main trait branch.
    n_pool = len(trait_post_dropout)
    if n_pool == 0:
        sampled_ids = np.empty(0, dtype=trait_post_dropout["id"].dtype if "id" in trait_post_dropout.columns else np.int64)
        case_summary = "empty pool"
    elif n_sample <= 0 or n_sample >= n_pool:
        if n_sample > 0:
            logger.info(
                "Ascertainment: N_sample=%d >= post-dropout pool of %d; passing all through",
                n_sample,
                n_pool,
            )
        sampled_ids = trait_post_dropout["id"].to_numpy()
        case_summary = f"pass-through (N_sample={n_sample}, pool={n_pool})"
    else:
        is_case = trait_post_dropout["affected1"].to_numpy()
        n_cases = int(is_case.sum())
        n_controls = n_pool - n_cases

        if ratio == 1.0 or n_cases == 0 or n_cases == n_pool:
            if ratio != 1.0:
                logger.warning(
                    "case_ascertainment_ratio=%.2f ignored (degenerate: n_cases=%d, n_pool=%d)",
                    ratio,
                    n_cases,
                    n_pool,
                )
            sample_idx = rng.choice(n_pool, n_sample, replace=False)
        elif ratio == 0:
            actual_n = min(n_sample, n_controls)
            if actual_n < n_sample:
                logger.warning(
                    "case_ascertainment_ratio=0: clamping N_sample from %d to %d (only %d controls)",
                    n_sample,
                    actual_n,
                    n_controls,
                )
            control_indices = np.where(~is_case)[0]
            sample_idx = rng.choice(control_indices, actual_n, replace=False)
        else:
            weights = np.where(is_case, ratio, 1.0)
            probabilities = weights / weights.sum()
            sample_idx = rng.choice(n_pool, n_sample, replace=False, p=probabilities)

        sampled_ids = trait_post_dropout["id"].to_numpy()[np.sort(sample_idx)]
        case_summary = f"ratio={ratio}, n_cases={n_cases}/{n_pool}"

    # Step 4: filter both trait branches to sampled IDs (identical IDs guaranteed).
    trait_out = trait_post_dropout[trait_post_dropout["id"].isin(sampled_ids)].reset_index(drop=True)
    simple_ltm_out = simple_ltm_post_dropout[simple_ltm_post_dropout["id"].isin(sampled_ids)].reset_index(drop=True)

    # Step 5: pedigree = ancestor closure of sampled IDs within post-dropout pedigree.
    if len(sampled_ids) == 0:
        ped_closure = ped_post_dropout.iloc[0:0].copy()
    else:
        ped_closure = filter_pedigree_to_observed(ped_post_dropout, sampled_ids)

    # Step 6: explicit fixup — twin (and any other) refs pointing outside the closure → -1.
    closure_ids = ped_closure["id"].to_numpy()
    ped_out = _sever_dangling_links(ped_closure, closure_ids).reset_index(drop=True)

    elapsed = time.perf_counter() - t0
    logger.info(
        "Ascertainment: ped %d → %d, trait %d → %d, simple_ltm %d → %d (dropout=%.3f, %s) in %.2fs (seed=%d)",
        n_total,
        len(ped_out),
        len(trait),
        len(trait_out),
        len(trait_simple_ltm),
        len(simple_ltm_out),
        rate,
        case_summary,
        elapsed,
        seed,
    )
    return ped_out, trait_out, simple_ltm_out


def cli() -> None:
    """Command-line entry point for the ascertainment stage."""
    from simace.core.cli_base import add_logging_args, init_logging

    parser = argparse.ArgumentParser(description="Unified ascertainment: dropout + case-weighted N_sample draw")
    add_logging_args(parser)
    parser.add_argument("--pedigree", required=True, help="Input pre-ascertainment pedigree parquet")
    parser.add_argument("--trait", required=True, help="Input post-censor trait parquet")
    parser.add_argument("--trait-simple-ltm", required=True, help="Input simple-LTM trait parquet")
    parser.add_argument("--out-pedigree", required=True, help="Output ascertained pedigree parquet")
    parser.add_argument("--out-trait", required=True, help="Output ascertained trait parquet")
    parser.add_argument("--out-trait-simple-ltm", required=True, help="Output ascertained simple-LTM trait parquet")
    parser.add_argument("--dropout-rate", type=float, default=0.0, help="Fraction of pedigree to drop uniformly")
    parser.add_argument("--case-ascertainment-ratio", type=float, default=1.0, help="Case weight vs controls")
    parser.add_argument("--N-sample", type=int, default=0, help="Target sample size (0 = pass-through)")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed")

    args = parser.parse_args()
    init_logging(args)

    ped = pd.read_parquet(args.pedigree)
    trait = pd.read_parquet(args.trait)
    simple_ltm = pd.read_parquet(args.trait_simple_ltm)
    ped_out, trait_out, simple_ltm_out = run_ascertainment(
        ped,
        trait,
        simple_ltm,
        dropout_rate=args.dropout_rate,
        case_ascertainment_ratio=args.case_ascertainment_ratio,
        N_sample=args.N_sample,
        seed=args.seed,
    )
    save_parquet(ped_out, args.out_pedigree)
    save_parquet(trait_out, args.out_trait)
    save_parquet(simple_ltm_out, args.out_trait_simple_ltm)
