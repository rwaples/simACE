"""Ascertainment implementation: dropout, case-weighted draw, pedigree closure, CLI.

See the package docstring in :mod:`simace.ascertainment` for the stage's role
and ADR 0001 context. ``cli`` is the ``simace-ascertain`` entry point.
"""

from __future__ import annotations

import argparse
import logging
import shutil
import time
from pathlib import Path

import numpy as np
import polars as pl

from simace.core.parquet import load_parquet, save_parquet
from simace.core.pedigree_filter import filter_pedigree_to_observed

logger = logging.getLogger(__name__)


def _sever_dangling_links(df: pl.DataFrame, valid_ids: np.ndarray) -> pl.DataFrame:
    """Rewrite mother/father/twin references pointing outside ``valid_ids`` to -1."""
    result = df
    for col in ("mother", "father", "twin"):
        if col not in result.columns:
            continue
        vals = result[col].to_numpy()
        in_valid = np.isin(vals, valid_ids)
        dangling = ~in_valid & (vals >= 0)
        if dangling.any():
            fixed = vals.copy()
            fixed[dangling] = -1
            result = result.with_columns(pl.Series(col, fixed))
    return result


def _apply_dropout(pedigree: pl.DataFrame, rate: float, rng: np.random.Generator) -> pl.DataFrame:
    """Remove a uniform random subset of the full pedigree.

    Row-position NumPy selection — the fixed-seed kept set is unchanged by the
    polars migration (ADR 0015 decision 14).
    """
    n_total = len(pedigree)
    n_drop = round(n_total * rate)
    if n_drop <= 0:
        return pedigree
    if n_drop >= n_total:
        raise ValueError(f"dropout_rate {rate} would remove all {n_total} individuals")

    drop_idx = rng.choice(n_total, n_drop, replace=False)
    keep_mask = np.ones(n_total, dtype=bool)
    keep_mask[drop_idx] = False
    return pedigree.filter(pl.Series(keep_mask))


def _filter_to_ids(df: pl.DataFrame, ids: np.ndarray) -> pl.DataFrame:
    """Filter a trait-like DataFrame to an ID set, preserving row order."""
    return df.filter(pl.Series(np.isin(df["id"].to_numpy(), ids)))


def _read_id_column(path: str | Path) -> pl.Series:
    """Read only the ``id`` column from a parquet file."""
    return load_parquet(path, columns=["id"])["id"]


def _same_id_sequence(left: pl.Series, right: pl.Series) -> bool:
    """Return True when two id columns have identical length, order, and values."""
    if len(left) != len(right):
        return False
    return bool(np.array_equal(left.to_numpy(), right.to_numpy()))


def _copy_file(src: str | Path, dst: str | Path) -> None:
    """Copy one file, creating the output directory and tolerating same-file calls."""
    src_path = Path(src)
    dst_path = Path(dst)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        if src_path.samefile(dst_path):
            return
    except FileNotFoundError:
        pass
    shutil.copy2(src_path, dst_path)


def copy_passthrough_if_possible(
    pedigree_path: str | Path,
    trait_path: str | Path,
    out_pedigree_path: str | Path,
    out_trait_path: str | Path,
    *,
    dropout_rate: float | None = 0.0,
    N_sample: int | None = 0,
) -> bool:
    """Fast-path no-op ascertainment by copying parquet inputs to outputs.

    The DataFrame API deliberately preserves the semantic ancestor-closure
    step even when ``dropout_rate=0`` and ``N_sample`` passes all trait rows.
    The file-level Snakemake/CLI path can skip the expensive pandas
    decode/filter/re-encode cycle only when the phenotype and pedigree files
    already contain the exact same ordered ID set, making the closure equal to
    the input pedigree.

    Returns ``True`` when both outputs were copied and the caller can skip
    regular ascertainment. Returns ``False`` when regular ascertainment must
    run to preserve semantics.
    """
    rate = float(dropout_rate or 0.0)
    n_sample = int(N_sample or 0)
    if rate != 0.0:
        return False

    trait_ids = _read_id_column(trait_path)
    if n_sample > 0 and n_sample < len(trait_ids):
        return False

    pedigree_ids = _read_id_column(pedigree_path)
    if not _same_id_sequence(pedigree_ids, trait_ids):
        return False

    _copy_file(pedigree_path, out_pedigree_path)
    _copy_file(trait_path, out_trait_path)
    logger.info(
        "Ascertainment pass-through: copied %d rows unchanged (dropout=%.3f, N_sample=%d)",
        len(trait_ids),
        rate,
        n_sample,
    )
    return True


def _sample_trait_ids(
    trait_post_dropout: pl.DataFrame,
    *,
    case_ascertainment_ratio: float,
    N_sample: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, str]:
    """Draw the shared post-ascertainment trait ID set."""
    n_pool = len(trait_post_dropout)
    if n_pool == 0:
        dtype = trait_post_dropout["id"].to_numpy().dtype if "id" in trait_post_dropout.columns else np.int64
        return np.empty(0, dtype=dtype), "empty pool"

    if N_sample <= 0 or N_sample >= n_pool:
        if N_sample > 0:
            logger.info(
                "Ascertainment: N_sample=%d >= post-dropout pool of %d; passing all through",
                N_sample,
                n_pool,
            )
        return trait_post_dropout["id"].to_numpy(), f"pass-through (N_sample={N_sample}, pool={n_pool})"

    is_case = trait_post_dropout["affected1"].to_numpy()
    n_cases = int(is_case.sum())
    n_controls = n_pool - n_cases

    # Zero case weight is checked before the degenerate branch: an all-case pool
    # is degenerate for every *other* ratio, but for ratio=0 it has a defined
    # answer -- no eligible controls, so nothing is drawn. Checking degeneracy
    # first would silently draw cases under a zero case weight.
    if case_ascertainment_ratio == 0:
        actual_n = min(N_sample, n_controls)
        if actual_n < N_sample:
            logger.warning(
                "case_ascertainment_ratio=0: clamping N_sample from %d to %d (only %d controls)",
                N_sample,
                actual_n,
                n_controls,
            )
        control_indices = np.where(~is_case)[0]
        sample_idx = rng.choice(control_indices, actual_n, replace=False)
    elif case_ascertainment_ratio == 1.0 or n_cases == 0 or n_cases == n_pool:
        if case_ascertainment_ratio != 1.0:
            logger.warning(
                "case_ascertainment_ratio=%.2f ignored (degenerate: n_cases=%d, n_pool=%d)",
                case_ascertainment_ratio,
                n_cases,
                n_pool,
            )
        sample_idx = rng.choice(n_pool, N_sample, replace=False)
    else:
        weights = np.where(is_case, case_ascertainment_ratio, 1.0)
        probabilities = weights / weights.sum()
        sample_idx = rng.choice(n_pool, N_sample, replace=False, p=probabilities)

    sampled_ids = trait_post_dropout["id"].to_numpy()[np.sort(sample_idx)]
    return sampled_ids, f"ratio={case_ascertainment_ratio}, n_cases={n_cases}/{n_pool}"


def _pedigree_closure_for_ids(pedigree: pl.DataFrame, sampled_ids: np.ndarray) -> pl.DataFrame:
    """Filter pedigree to sampled IDs plus ancestors, then sever dangling links."""
    ped_closure = pedigree.head(0) if len(sampled_ids) == 0 else filter_pedigree_to_observed(pedigree, sampled_ids)
    return _sever_dangling_links(ped_closure, ped_closure["id"].to_numpy())


def run_ascertainment(
    pedigree: pl.DataFrame,
    trait: pl.DataFrame,
    *,
    dropout_rate: float = 0.0,
    case_ascertainment_ratio: float = 1.0,
    N_sample: int = 0,
    seed: int = 42,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Apply dropout + case-weighted sampling and return the ascertained subset.

    All random selection runs on NumPy row positions, so the fixed-seed
    sampled IDs are unchanged by the polars migration (ADR 0015 decision 14).

    Two explicit steps, applied to IDs not weights, so ``dropout_rate``
    has effect even when ``N_sample > 0``:

    1. Uniform pedigree dropout: ``round(N_total * dropout_rate)`` individuals
       are removed uniformly at random from the full pedigree.
    2. Case-weighted draw from the post-dropout trait population:
       ``N_sample`` IDs are drawn with weights ``case_ascertainment_ratio``
       for cases vs ``1`` for controls. Pass-through when ``N_sample <= 0``
       or ``N_sample >= len(post-dropout trait)``.

    The pedigree output is the ancestor closure of the sampled IDs within
    the post-dropout pedigree, with all dangling parent/twin references
    rewritten to ``-1`` against the final closure ID set. Twin links dangle
    whenever a twin partner goes unsampled. Parent links are safe by
    construction *only* at ``dropout_rate=0``, where the closure follows every
    parent pointer: dropout removes individuals **before** the closure is
    built, so an ancestor can be absent and unrecoverable. Each parent is
    severed independently, so a row may carry one real parent and one ``-1``
    (measured at ``dropout_rate=0.2`` on a small pedigree: ~140 such rows).

    Args:
        pedigree: Full pre-ascertainment pedigree (post-burn-in, pre-dropout).
        trait: Per-individual censored trait observations (G_pheno generations).
        dropout_rate: Fraction of pedigree to remove uniformly at random.
        case_ascertainment_ratio: Sampling weight for cases relative to controls.
        N_sample: Target sample size; ``<=0`` or ``>= post-dropout pool`` passes everything through.
        seed: RNG seed.

    Returns:
        Tuple of (pedigree_ascertained, trait_ascertained).
    """
    from simace.core.trait_schema import _require_polars

    _require_polars(pedigree, where="run_ascertainment: pedigree")
    _require_polars(trait, where="run_ascertainment: trait")

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

    ped_post_dropout = _apply_dropout(pedigree, rate, rng)
    ped_post_dropout_ids = ped_post_dropout["id"].to_numpy()

    trait_post_dropout = _filter_to_ids(trait, ped_post_dropout_ids)
    sampled_ids, case_summary = _sample_trait_ids(
        trait_post_dropout,
        case_ascertainment_ratio=ratio,
        N_sample=n_sample,
        rng=rng,
    )

    trait_out = _filter_to_ids(trait_post_dropout, sampled_ids)
    ped_out = _pedigree_closure_for_ids(ped_post_dropout, sampled_ids)

    elapsed = time.perf_counter() - t0
    logger.info(
        "Ascertainment: ped %d → %d, trait %d → %d (dropout=%.3f, %s) in %.2fs (seed=%d)",
        n_total,
        len(ped_out),
        len(trait),
        len(trait_out),
        rate,
        case_summary,
        elapsed,
        seed,
    )
    return ped_out, trait_out


def cli() -> None:
    """Command-line entry point for the ascertainment stage."""
    from simace.core.cli_base import add_logging_args, add_version_arg, init_logging

    parser = argparse.ArgumentParser(description="Unified ascertainment: dropout + case-weighted N_sample draw")
    add_logging_args(parser)
    add_version_arg(parser, "simace")
    parser.add_argument("--pedigree", required=True, help="Input pre-ascertainment pedigree parquet")
    parser.add_argument("--trait", required=True, help="Input post-censor trait parquet")
    parser.add_argument("--out-pedigree", required=True, help="Output ascertained pedigree parquet")
    parser.add_argument("--out-trait", required=True, help="Output ascertained trait parquet")
    parser.add_argument("--dropout-rate", type=float, default=0.0, help="Fraction of pedigree to drop uniformly")
    parser.add_argument("--case-ascertainment-ratio", type=float, default=1.0, help="Case weight vs controls")
    parser.add_argument("--N-sample", type=int, default=0, help="Target sample size (0 = pass-through)")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed")

    args = parser.parse_args()
    init_logging(args)

    if copy_passthrough_if_possible(
        args.pedigree,
        args.trait,
        args.out_pedigree,
        args.out_trait,
        dropout_rate=args.dropout_rate,
        N_sample=args.N_sample,
    ):
        return

    ped = load_parquet(args.pedigree)
    trait = load_parquet(args.trait)
    ped_out, trait_out = run_ascertainment(
        ped,
        trait,
        dropout_rate=args.dropout_rate,
        case_ascertainment_ratio=args.case_ascertainment_ratio,
        N_sample=args.N_sample,
        seed=args.seed,
    )
    save_parquet(ped_out, args.out_pedigree)
    save_parquet(trait_out, args.out_trait)
