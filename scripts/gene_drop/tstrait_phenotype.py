r"""Aggregate per-chromosome GVs and add environmental noise.

Sums the 22 per-chrom genetic-value parquets produced by tstrait_gv_chrom.py
into a genome-wide GV per (trait_id, individual_id), then calls
`tstrait.sim_env(genetic_df, h2=h2)` to draw the environmental noise that
hits the requested heritability and produce the final phenotype column.

Outputs:
  - `tstrait_phenotype.parquet`: trait_id, individual_id, genetic_value,
    environmental_noise, phenotype
  - `tstrait_phenotype_meta.json`: realized h^2, Var(GV), Var(E), n_causal
    seen across all chroms, plus the params echoed

``--h2`` is trait 1's ``A1 / (A1 + C1 + E1)``; ``--seed`` is the scenario seed
``+ (rep - 1) + 20000``. Needs the conda env in
``scripts/gene_drop/envs/tskit.yaml``. Example::

    python scripts/gene_drop/tstrait_phenotype.py \
        --gv results/{folder}/{scenario}/rep{rep}/gv_chrom_{1..22}.parquet \
        --meta results/{folder}/{scenario}/rep{rep}/gv_chrom_{1..22}_meta.json \
        --effects results/{folder}/{scenario}/rep{rep}/causal_effects.parquet \
        --effects-meta results/{folder}/{scenario}/rep{rep}/causal_effects_meta.json \
        --phenotype results/{folder}/{scenario}/rep{rep}/tstrait_phenotype.parquet \
        --out-meta results/{folder}/{scenario}/rep{rep}/tstrait_phenotype_meta.json \
        --h2 0.5 --trait-id 0 --seed {seed + rep - 1 + 20000} \
        --log logs/{folder}/{scenario}/rep{rep}/tstrait_phenotype.log
"""

import argparse
import json
import logging
import re
import time
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import tstrait

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s")
log = logging.getLogger("tstrait_phenotype")


def _chrom_key(p: Path) -> int:
    m = re.search(r"chrom_?([0-9]+)", p.stem, re.IGNORECASE)
    if not m:
        raise ValueError(f"could not parse chromosome number from {p.name}")
    return int(m.group(1))


_ECHOED_PARAM_KEYS = (
    "num_causal",
    "frac_causal",
    "maf_threshold",
    "alpha",
    "effect_mean",
    "effect_var",
)


def _route_logging_to_file(log_path: str | None) -> None:
    """Add a FileHandler writing to ``log_path`` when one is given."""
    if not log_path:
        return
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(log_path, mode="w")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s | %(message)s"))
    logging.getLogger().addHandler(fh)


def main(argv: list[str] | None = None) -> None:
    """Sum per-chrom GVs + sim_env + meta."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--gv", nargs="+", required=True)
    parser.add_argument("--meta", nargs="+", required=True)
    parser.add_argument("--effects", required=True)
    parser.add_argument("--effects-meta", required=True)
    parser.add_argument("--phenotype", required=True)
    parser.add_argument("--out-meta", required=True)
    parser.add_argument("--h2", type=float, required=True)
    parser.add_argument("--trait-id", type=int, required=True)
    parser.add_argument("--share-architecture", action="store_true")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--log", default=None)
    args = parser.parse_args(argv)

    _route_logging_to_file(args.log)
    gv_paths = sorted([Path(p) for p in args.gv], key=_chrom_key)
    meta_paths = sorted([Path(p) for p in args.meta], key=_chrom_key)
    effects_path = Path(args.effects)
    effects_meta_path = Path(args.effects_meta)
    out_pheno = Path(args.phenotype)
    out_meta_path = Path(args.out_meta)
    out_pheno.parent.mkdir(parents=True, exist_ok=True)

    p = args
    h2 = float(p.h2)
    seed = int(p.seed)
    trait_id = int(p.trait_id)
    share_architecture = bool(p.share_architecture)

    log.info("loading %d per-chrom GV files", len(gv_paths))
    t = time.perf_counter()
    dfs = [pd.read_parquet(q) for q in gv_paths]
    n_inds_per_chrom = {q.name: len(df) for q, df in zip(gv_paths, dfs, strict=True)}
    log.info("  loaded in %.1fs; per-chrom rows: %s", time.perf_counter() - t, n_inds_per_chrom)

    cat = pd.concat(dfs, ignore_index=True)
    summed = cat.groupby(["trait_id", "individual_id"], as_index=False, sort=True)["genetic_value"].sum()
    log.info(
        "summed: %d (trait_id, individual_id) rows; var(GV)=%.4f, mean(GV)=%.4f",
        len(summed),
        float(summed["genetic_value"].var(ddof=0)),
        float(summed["genetic_value"].mean()),
    )

    log.info("running tstrait.sim_env(h2=%.3f, seed=%d)", h2, seed)
    pheno = tstrait.sim_env(summed, h2=h2, random_seed=seed)
    pheno.to_parquet(out_pheno, index=False, compression="zstd")
    log.info("wrote %s (%d rows)", out_pheno, len(pheno))

    var_gv = float(pheno["genetic_value"].var(ddof=0))
    var_env = float(pheno["environmental_noise"].var(ddof=0))
    realized_h2 = var_gv / (var_gv + var_env) if (var_gv + var_env) > 0 else float("nan")

    per_chrom_meta = [json.loads(q.read_text()) for q in meta_paths]
    n_causal_used = int(sum(m["n_causal_on_chrom"] for m in per_chrom_meta))

    # Echo the assignment-time params from the assign_effects meta so they
    # cannot drift from what was actually used to draw the causal sites.
    effects_meta = json.loads(effects_meta_path.read_text())
    echoed_params = {k: effects_meta[k] for k in _ECHOED_PARAM_KEYS}

    n_causal_in_effects_file = pq.ParquetFile(effects_path).metadata.num_rows
    out_meta = {
        "trait_id": trait_id,
        "h2_target": h2,
        "h2_realized": realized_h2,
        "var_gv": var_gv,
        "var_env": var_env,
        "var_phenotype": float(pheno["phenotype"].var(ddof=0)),
        "mean_gv": float(pheno["genetic_value"].mean()),
        "mean_phenotype": float(pheno["phenotype"].mean()),
        "n_individuals": len(pheno),
        "n_causal_used": n_causal_used,
        "n_causal_in_effects_file": n_causal_in_effects_file,
        "seed": seed,
        "params": {
            **echoed_params,
            "h2": h2,
            "trait_id": trait_id,
            "share_architecture": share_architecture,
        },
        "per_chromosome": per_chrom_meta,
    }
    out_meta_path.write_text(json.dumps(out_meta, indent=2, default=float))
    log.info(
        "h2: target=%.3f, realized=%.3f (var_gv=%.4f, var_env=%.4f); n_causal_used=%d",
        h2,
        realized_h2,
        var_gv,
        var_env,
        n_causal_used,
    )


if __name__ == "__main__":
    main()
