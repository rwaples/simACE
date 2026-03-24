"""Compute Falconer h² from simple LTM binary phenotype using tetrachoric correlations.

For each EPIMIGHT relationship kind, computes:
  1. Tetrachoric correlation from binary affected status
  2. Falconer h² = r_tetrachoric / (2 * kinship)
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from sim_ace.pedigree_graph import extract_relationship_pairs
from sim_ace.stats import tetrachoric_corr_se

logger = logging.getLogger(__name__)

# EPIMIGHT kind → list of ACE pair type names (mirrors epimight/create_parquet.py)
KIND_TO_PAIRS: dict[str, list[str]] = {
    "PO": ["Mother-offspring", "Father-offspring"],
    "FS": ["Full sib", "MZ twin"],
    "HS": ["Maternal half sib", "Paternal half sib"],
    "mHS": ["Maternal half sib"],
    "pHS": ["Paternal half sib"],
    "1C": ["1st cousin"],
    "Av": ["Avuncular"],
    "1G": ["Grandparent-grandchild"],
}

# Kinship coefficient (f) for each EPIMIGHT kind.
# The Falconer formula is h² = r_tetrachoric / (2f).
KINSHIP: dict[str, float] = {
    "PO": 0.25,
    "FS": 0.25,
    "HS": 0.125,
    "mHS": 0.125,
    "pHS": 0.125,
    "1C": 0.0625,
    "Av": 0.125,
    "1G": 0.125,
}

MIN_PAIRS = 50


def compute_ltm_falconer(
    df: pd.DataFrame,
    kinds: list[str],
    trait_num: int = 1,
    seed: int = 42,
    pedigree: pd.DataFrame | None = None,
) -> dict[str, dict[str, Any]]:
    """Compute Falconer h² per EPIMIGHT kind from simple LTM phenotype.

    Parameters
    ----------
    df : DataFrame
        Simple LTM phenotype with ``affected{trait_num}`` column plus
        id/mother/father/twin/sex/generation columns.
    kinds : list[str]
        EPIMIGHT relationship kinds to compute (e.g. ``["PO", "FS", "HS"]``).
    trait_num : int
        Which trait to analyse (1 or 2).
    seed : int
        Random seed for pair extraction.
    pedigree : DataFrame, optional
        Full pedigree for complete multi-hop pair extraction.

    Returns
    -------
    dict
        Keyed by kind, each value a dict with keys:
        ``r_tetrachoric``, ``se_r``, ``h2_falconer``, ``se_h2``,
        ``n_pairs``, ``kinship``.
    """
    pairs = extract_relationship_pairs(df, seed=seed, full_pedigree=pedigree)
    affected = df[f"affected{trait_num}"].values.astype(bool)

    results: dict[str, dict[str, Any]] = {}
    for kind in kinds:
        pair_types = KIND_TO_PAIRS.get(kind, [])
        all_idx1: list[np.ndarray] = []
        all_idx2: list[np.ndarray] = []
        for pt in pair_types:
            if pt in pairs:
                idx1, idx2 = pairs[pt]
                all_idx1.append(idx1)
                all_idx2.append(idx2)

        if not all_idx1:
            results[kind] = _empty_result(kind)
            continue

        merged_idx1 = np.concatenate(all_idx1)
        merged_idx2 = np.concatenate(all_idx2)
        n_pairs = len(merged_idx1)

        if n_pairs < MIN_PAIRS:
            logger.warning("Kind %s: only %d pairs (< %d), skipping", kind, n_pairs, MIN_PAIRS)
            results[kind] = _empty_result(kind, n_pairs)
            continue

        r, se_r = tetrachoric_corr_se(affected[merged_idx1], affected[merged_idx2])
        k = KINSHIP[kind]
        h2 = r / (2 * k) if not np.isnan(r) else np.nan
        se_h2 = se_r / (2 * k) if not np.isnan(se_r) else np.nan

        logger.info("Kind %s: n_pairs=%d, r_tet=%.4f (SE %.4f), h2=%.4f", kind, n_pairs, r, se_r, h2)
        results[kind] = {
            "r_tetrachoric": _safe_float(r),
            "se_r": _safe_float(se_r),
            "h2_falconer": _safe_float(h2),
            "se_h2": _safe_float(se_h2),
            "n_pairs": int(n_pairs),
            "kinship": float(k),
        }

    return results


def _empty_result(kind: str, n_pairs: int = 0) -> dict[str, Any]:
    return {
        "r_tetrachoric": None,
        "se_r": None,
        "h2_falconer": None,
        "se_h2": None,
        "n_pairs": n_pairs,
        "kinship": KINSHIP.get(kind, 0.0),
    }


def _safe_float(x: float) -> float | None:
    return float(x) if not np.isnan(x) else None


# ---------------------------------------------------------------------------
# Main entry point (for both Snakemake wrapper and CLI)
# ---------------------------------------------------------------------------


def main(
    simple_ltm_path: str,
    pedigree_path: str,
    output_path: str,
    kinds: list[str] | None = None,
    trait_num: int = 1,
    seed: int = 42,
) -> None:
    """Compute LTM Falconer h² and write to JSON."""
    if kinds is None:
        kinds = list(KIND_TO_PAIRS.keys())

    logger.info("Loading simple LTM phenotype: %s", simple_ltm_path)
    df = pd.read_parquet(simple_ltm_path)

    pedigree = None
    if pedigree_path:
        logger.info("Loading full pedigree: %s", pedigree_path)
        pedigree = pd.read_parquet(pedigree_path)

    results = compute_ltm_falconer(df, kinds, trait_num=trait_num, seed=seed, pedigree=pedigree)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Wrote LTM Falconer results to %s", output_path)


def cli() -> None:
    """CLI entry point."""
    from sim_ace import setup_logging

    parser = argparse.ArgumentParser(description="Compute Falconer h² from simple LTM phenotype")
    parser.add_argument("--simple-ltm", required=True, help="Path to phenotype.simple_ltm.parquet")
    parser.add_argument("--pedigree", required=True, help="Path to pedigree.parquet")
    parser.add_argument("--output", required=True, help="Output JSON path")
    parser.add_argument("--kinds", nargs="+", default=None, help="EPIMIGHT kinds to compute")
    parser.add_argument("--trait", type=int, default=1, help="Trait number (1 or 2)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--log-file", default=None, help="Log file")
    args = parser.parse_args()

    setup_logging(log_file=args.log_file)
    main(args.simple_ltm, args.pedigree, args.output, kinds=args.kinds, trait_num=args.trait, seed=args.seed)


if __name__ == "__main__":
    cli()
