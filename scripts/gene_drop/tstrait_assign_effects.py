r"""Assign causal sites and effect sizes for tstrait.

Reads the shared founder-AF site catalog (from tstrait_site_catalog_concat.py)
and writes a parquet of causal sites in the schema tstrait expects:

    site_id, effect_size, causal_allele, trait_id, CHR, AF

Steps:
  (a) Validate `num_causal` xor `frac_causal` (exactly one must be set; the
      other must be `null`).
  (b) MAF filter: keep sites with `min(AF, 1-AF) > maf_threshold`. A
      threshold of 0 disables the filter.
  (c) `n_causal = num_causal` if absolute, else `round(frac_causal * n_eligible)`;
      assert it does not exceed the eligible pool.
  (d) Uniform sample without replacement using `np.random.default_rng(seed)`.
  (e) Raw beta ~ Normal(`effect_mean`, sqrt(`effect_var`)).
  (f) Multiply raw beta by `[2 p (1-p)]^alpha` (so alpha=0 keeps it raw,
      alpha=-0.5 mimics the LDAK-thin / Speed et al. parameterisation).
  (g) Sort the output by `(CHR, site_id)` so tstrait sees site_id ascending
      within each chromosome (its slicing per-chrom relies on that ordering).

The script runs in one of two modes. Per-rep: ``--seed`` is the scenario seed
``+ (rep - 1) + 10000`` and outputs are rep-scoped. Shared
(``tstrait_share_architecture: true``): ``--seed`` is the scenario seed
``+ 10000`` and outputs are scenario-scoped. Needs the conda env in
``scripts/gene_drop/envs/tskit.yaml``. Per-rep example::

    python scripts/gene_drop/tstrait_assign_effects.py \
        --catalog {output_dir}/site_catalog.parquet \
        --effects results/{folder}/{scenario}/rep{rep}/causal_effects.parquet \
        --meta results/{folder}/{scenario}/rep{rep}/causal_effects_meta.json \
        --num-causal 1000 --maf-threshold 0.01 --alpha -0.5 \
        --effect-mean 0.0 --effect-var 1.0 --trait-id 0 \
        --seed {seed + rep - 1 + 10000} \
        --log logs/{folder}/{scenario}/rep{rep}/tstrait_assign_effects.log
"""

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s")
log = logging.getLogger("tstrait_assign_effects")


def _resolve_n_causal(
    num_causal: int | None,
    frac_causal: float | None,
    n_eligible: int,
) -> int:
    if num_causal is not None and frac_causal is not None:
        raise ValueError("tstrait: set exactly one of `num_causal` or `frac_causal`; both are set")
    if num_causal is None and frac_causal is None:
        raise ValueError("tstrait: set exactly one of `num_causal` or `frac_causal`; both are null")
    if num_causal is not None:
        n = int(num_causal)
    else:
        if not 0.0 < float(frac_causal) <= 1.0:
            raise ValueError(f"tstrait: frac_causal must be in (0, 1]; got {frac_causal}")
        n = round(float(frac_causal) * n_eligible)
    if n <= 0:
        raise ValueError(f"tstrait: resolved n_causal={n} is non-positive")
    if n > n_eligible:
        raise ValueError(f"tstrait: n_causal={n} exceeds n_eligible={n_eligible} after MAF filter")
    return n


def assign_effects(
    catalog: pd.DataFrame,
    *,
    num_causal: int | None,
    frac_causal: float | None,
    maf_threshold: float,
    alpha: float,
    effect_mean: float,
    effect_var: float,
    trait_id: int,
    seed: int,
) -> tuple[pd.DataFrame, dict]:
    """Pick causal sites + draw effect sizes. Pure function for testability."""
    if effect_var < 0:
        raise ValueError(f"tstrait: effect_var must be >= 0; got {effect_var}")
    if not 0.0 <= float(maf_threshold) < 0.5:
        raise ValueError(f"tstrait: maf_threshold must be in [0, 0.5); got {maf_threshold}")

    n_total = len(catalog)
    af = catalog["AF"].to_numpy()
    maf = np.minimum(af, 1.0 - af)
    eligible_mask = maf > float(maf_threshold)
    eligible_idx = np.flatnonzero(eligible_mask)
    n_eligible = int(eligible_idx.size)
    if n_eligible == 0:
        raise ValueError(
            f"tstrait: 0 sites pass MAF filter (maf_threshold={maf_threshold}); catalog has {n_total} sites"
        )

    n_causal = _resolve_n_causal(num_causal, frac_causal, n_eligible)

    rng = np.random.default_rng(int(seed))
    chosen_idx = rng.choice(eligible_idx, size=n_causal, replace=False)
    chosen = catalog.iloc[chosen_idx].copy()

    raw_beta = rng.normal(loc=float(effect_mean), scale=float(np.sqrt(effect_var)), size=n_causal)
    p = chosen["AF"].to_numpy()
    af_factor = np.power(2.0 * p * (1.0 - p), float(alpha)) if alpha != 0.0 else np.ones(n_causal)
    effect_size = raw_beta * af_factor

    out = pd.DataFrame(
        {
            "site_id": chosen["site_id"].to_numpy(),
            "POS": chosen["POS"].to_numpy().astype(np.int64),
            "effect_size": effect_size.astype(np.float64),
            "causal_allele": chosen["causal_allele"].to_numpy(),
            "trait_id": np.full(n_causal, int(trait_id), dtype=np.int32),
            "CHR": chosen["CHR"].to_numpy().astype(np.int8),
            "AF": p.astype(np.float64),
        }
    )
    out = out.sort_values(["CHR", "site_id"], kind="stable").reset_index(drop=True)

    meta = {
        "n_total_catalog_sites": int(n_total),
        "n_eligible_after_maf": n_eligible,
        "n_causal": int(n_causal),
        "num_causal": None if num_causal is None else int(num_causal),
        "frac_causal": None if frac_causal is None else float(frac_causal),
        "maf_threshold": float(maf_threshold),
        "alpha": float(alpha),
        "effect_mean": float(effect_mean),
        "effect_var": float(effect_var),
        "trait_id": int(trait_id),
        "seed": int(seed),
        "var_raw_beta": float(np.var(raw_beta, ddof=0)),
        "var_effect_size": float(np.var(effect_size, ddof=0)),
        "mean_af_factor": float(af_factor.mean()),
        "per_chromosome": (out.groupby("CHR")["site_id"].count().rename("n_causal").reset_index().to_dict("records")),
    }
    return out, meta


def _route_logging_to_file(log_path: str | None) -> None:
    """Add a FileHandler writing to ``log_path`` when one is given."""
    if not log_path:
        return
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(log_path, mode="w")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s | %(message)s"))
    logging.getLogger().addHandler(fh)


def main(argv: list[str] | None = None) -> None:
    """Read catalog, pick causal sites, dump effects + meta."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--effects", required=True)
    parser.add_argument("--meta", required=True)
    parser.add_argument("--num-causal", type=int, default=None)
    parser.add_argument("--frac-causal", type=float, default=None)
    parser.add_argument("--maf-threshold", type=float, required=True)
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--effect-mean", type=float, required=True)
    parser.add_argument("--effect-var", type=float, required=True)
    parser.add_argument("--trait-id", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--log", default=None)
    args = parser.parse_args(argv)

    _route_logging_to_file(args.log)
    catalog_path = Path(args.catalog)
    out_effects = Path(args.effects)
    out_meta = Path(args.meta)
    out_effects.parent.mkdir(parents=True, exist_ok=True)

    log.info("loading catalog %s", catalog_path)
    t = time.perf_counter()
    catalog = pd.read_parquet(catalog_path)
    log.info("  catalog: %d rows (load %.1fs)", len(catalog), time.perf_counter() - t)

    p = args
    log.info(
        "params: num_causal=%s, frac_causal=%s, maf=%.4f, alpha=%.3f, "
        "effect_mean=%.3f, effect_var=%.3f, trait_id=%d, seed=%d",
        p.num_causal,
        p.frac_causal,
        float(p.maf_threshold),
        float(p.alpha),
        float(p.effect_mean),
        float(p.effect_var),
        int(p.trait_id),
        int(p.seed),
    )

    effects, meta = assign_effects(
        catalog,
        num_causal=p.num_causal,
        frac_causal=p.frac_causal,
        maf_threshold=p.maf_threshold,
        alpha=p.alpha,
        effect_mean=p.effect_mean,
        effect_var=p.effect_var,
        trait_id=p.trait_id,
        seed=p.seed,
    )
    log.info(
        "selected %d causal sites; var(raw beta)=%.4f, var(effect)=%.4f, mean AF factor=%.4f",
        meta["n_causal"],
        meta["var_raw_beta"],
        meta["var_effect_size"],
        meta["mean_af_factor"],
    )

    effects.to_parquet(out_effects, index=False, compression="zstd")
    out_meta.write_text(json.dumps(meta, indent=2, default=float))
    log.info("wrote %s (%d rows) and %s", out_effects, len(effects), out_meta)


if __name__ == "__main__":
    main()
