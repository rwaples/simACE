"""Snakemake wrapper: overwrite pedigree A1 with rescaled tstrait GV.

Produces a sibling `pedigree.full.tstrait.parquet` where the A1 column for
sample-tagged individuals (the latest G_pheno generations) is replaced with
the tstrait genome-wide genetic value, centered to mean 0 and rescaled so
its sample variance matches the configured A1 variance. Non-sample
individuals (older ancestors not in the genotyped sample set) keep their
original parametric A1.

`liability1 = A1 + C1 + E1` is recomputed after the overwrite so it stays
consistent. C1 and E1 come from the original pedigree unchanged — the goal
is to use simACE's standard parametric C/E components alongside a realistic
A from the genotypes.

This makes the tstrait branch comparable to standard simACE scenarios:
downstream phenotype models (frailty, threshold, etc.) can be run on the
augmented pedigree and use the simACE-style A+C+E variance composition.
"""

import json
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s")
log = logging.getLogger("tstrait_augment_pedigree")


def _route_logging_to_snakemake_log() -> None:
    """Add a FileHandler that writes to snakemake.log[0] (see genotype_drop_chrom.py)."""
    log_path = snakemake.log[0] if snakemake.log else None
    if not log_path:
        return
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(log_path, mode="w")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s | %(message)s"))
    logging.getLogger().addHandler(fh)


def _chrom_key(p: Path) -> int:
    m = re.search(r"chrom_?([0-9]+)", p.stem, re.IGNORECASE)
    if not m:
        raise ValueError(f"could not parse chromosome from {p.name}")
    return int(m.group(1))


def rescale_gv(gv: np.ndarray, target_var: float) -> tuple[np.ndarray, dict]:
    """Center GV at 0 and rescale so its sample variance == target_var exactly.

    Sample variance uses ddof=0 (population variance) so the post-rescale
    `np.var(out, ddof=0) == target_var` holds to floating-point precision.
    """
    if target_var < 0:
        raise ValueError(f"target_var must be >= 0; got {target_var}")
    realized_mean = float(gv.mean())
    realized_var = float(gv.var(ddof=0))
    if realized_var == 0:
        raise ValueError("realized GV variance is 0; cannot rescale")
    scale = float(np.sqrt(target_var / realized_var))
    out = (gv - realized_mean) * scale
    info = {
        "realized_mean": realized_mean,
        "realized_var": realized_var,
        "target_var": float(target_var),
        "scale_factor": scale,
        "rescaled_var": float(out.var(ddof=0)),
        "rescaled_mean": float(out.mean()),
    }
    return out, info


def main() -> None:
    """Snakemake entry: rescale + overwrite A1 + recompute liability1."""
    _route_logging_to_snakemake_log()
    pedigree_path = Path(snakemake.input.pedigree)
    gv_paths = sorted([Path(p) for p in snakemake.input.gv], key=_chrom_key)
    out_pedigree = Path(snakemake.output.pedigree)
    out_meta = Path(snakemake.output.meta)
    target_var_a = float(snakemake.params.target_var_A)
    out_pedigree.parent.mkdir(parents=True, exist_ok=True)

    log.info("loading pedigree %s", pedigree_path)
    ped = pd.read_parquet(pedigree_path)
    log.info("  pedigree: %d rows, columns: %s", len(ped), list(ped.columns))

    log.info("summing per-chrom GVs across %d files", len(gv_paths))
    gv_dfs = [pd.read_parquet(p) for p in gv_paths]
    gvs = pd.concat(gv_dfs, ignore_index=True).groupby("individual_id", as_index=False)["genetic_value"].sum()
    log.info("  GV summed: %d unique individuals", len(gvs))

    rescaled_arr, rescale_info = rescale_gv(gvs["genetic_value"].to_numpy(), target_var_a)
    log.info(
        "  rescale: realized_var=%.4f -> target_var=%.4f (scale=%.4f)",
        rescale_info["realized_var"],
        rescale_info["target_var"],
        rescale_info["scale_factor"],
    )

    gv_map = pd.Series(rescaled_arr, index=gvs["individual_id"].to_numpy())
    mask = ped["id"].isin(gv_map.index)
    n_overwritten = int(mask.sum())
    n_in_gv_not_in_ped = int((~gv_map.index.isin(ped["id"])).sum())
    if n_in_gv_not_in_ped:
        raise ValueError(
            f"{n_in_gv_not_in_ped} GV individual_ids not present in pedigree.id — "
            "drop+graft and pedigree are out of sync"
        )

    a1_orig = ped["A1"].to_numpy().copy()
    a1_new = a1_orig.copy()
    a1_new[mask.to_numpy()] = gv_map.reindex(ped.loc[mask, "id"].to_numpy()).to_numpy()

    ped_out = ped.copy()
    ped_out["A1"] = a1_new.astype(np.float32)
    if "liability1" in ped_out.columns:
        ped_out["liability1"] = (
            ped_out["A1"].astype(np.float64) + ped_out["C1"].astype(np.float64) + ped_out["E1"].astype(np.float64)
        )

    log.info("writing %s", out_pedigree)
    ped_out.to_parquet(out_pedigree, index=False, compression="zstd")

    sample_mask = mask.to_numpy()
    meta = {
        "n_pedigree_rows": len(ped),
        "n_inds_overwritten": n_overwritten,
        "n_inds_kept_parametric": int((~mask).sum()),
        "rescale": rescale_info,
        "var_A1_after_in_overwritten": float(np.var(a1_new[sample_mask], ddof=0)),
        "mean_A1_after_in_overwritten": float(np.mean(a1_new[sample_mask])),
        "var_A1_orig_in_overwritten": float(np.var(a1_orig[sample_mask], ddof=0)),
        "var_A1_orig_in_kept": (float(np.var(a1_orig[~sample_mask], ddof=0)) if (~sample_mask).any() else None),
    }
    out_meta.write_text(json.dumps(meta, indent=2, default=float))
    log.info(
        "overwrote A1 for %d / %d individuals; var(A1_new) in overwritten = %.4f",
        n_overwritten,
        len(ped),
        meta["var_A1_after_in_overwritten"],
    )


if __name__ == "__main__":
    main()
