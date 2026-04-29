"""Snakemake wrapper: per-chromosome tstrait site catalog.

Reads one canonicalized per-chrom .trees (from `tskit_preprocess`) and writes
a parquet catalog with one row per site:

    CHR, site_id, POS, AC, AF, causal_allele

`AC` is computed by walking the tree carrying each site and asking how many
samples descend from the (single) mutation node — far cheaper than iterating
genotypes via `ts.variants()` on a 35,951-sample / ~1M-site chromosome.

`causal_allele` is the mutation's `derived_state`. For SimHumanity / SLiM
ancestry this is the integer SLiM mutation_id encoded as a string (e.g.
`'113527630'`), not ACGT — and `ancestral_state` is `''`. tstrait matches
`causal_allele` against `variant.alleles` element-wise, so each site's unique
`('', '<mutation_id>')` tuple lets it find the correct dosage column.

Assumes one mutation per site (enforced by the canonicalize step's
`delete_sites` filter on `mut_count != 1`). Run inside the workflow's tskit
conda env via `--use-conda`.
"""

import logging
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import tskit

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s")
log = logging.getLogger("tstrait_site_catalog_chrom")


def _chrom_number(p: Path) -> int:
    m = re.search(r"chromosome[_\-]?([0-9]+)", p.stem, re.IGNORECASE)
    if not m:
        raise ValueError(f"could not parse chromosome number from {p.name}")
    return int(m.group(1))


def build_catalog(ts: tskit.TreeSequence, chrom_n: int) -> pd.DataFrame:
    """Vectorized AC/AF + causal_allele extraction for one canonicalized chrom.

    POS rounding. tskit stores `sites.position` as float64. SimHumanity /
    SLiM positions are integer base pairs (29.0, 32.0, ...), so casting to
    int64 truncates the trailing .0 with no loss. If this catalog is ever
    fed non-integer positions, `.astype(np.int64)` will silently round
    *toward zero* — callers that care about sub-bp positions should widen
    POS back to float and remove this cast.
    """
    n_samples = ts.num_samples
    n_sites = ts.num_sites
    if n_sites == 0:
        return pd.DataFrame(
            {
                "CHR": pd.Series([], dtype=np.int8),
                "site_id": pd.Series([], dtype=np.int64),
                "POS": pd.Series([], dtype=np.int64),
                "AC": pd.Series([], dtype=np.int32),
                "AF": pd.Series([], dtype=np.float64),
                "causal_allele": pd.Series([], dtype=object),
            }
        )

    site_table = ts.tables.sites
    mut_table = ts.tables.mutations
    mut_counts = np.bincount(mut_table.site, minlength=n_sites)
    if not np.all(mut_counts == 1):
        bad = int(np.sum(mut_counts != 1))
        raise ValueError(
            f"chr{chrom_n}: {bad} sites do not have exactly one mutation; "
            "canonicalize step should have filtered these"
        )

    site_to_mut = np.empty(n_sites, dtype=np.int64)
    site_to_mut[mut_table.site] = np.arange(mut_table.num_rows)

    pos = site_table.position.astype(np.int64)
    ac = np.zeros(n_sites, dtype=np.int64)
    for tree in ts.trees():
        if tree.num_sites == 0:
            continue
        for site in tree.sites():
            mut = site.mutations[0]
            ac[site.id] = tree.num_samples(mut.node)

    af = ac.astype(np.float64) / float(n_samples)

    # Vectorized derived_state decode via PyArrow LargeStringArray over the
    # raw (offsets, buffer) pair. ~10x faster than a Python loop on 1M muts.
    ds_offset_i64 = np.ascontiguousarray(mut_table.derived_state_offset, dtype=np.int64)
    ds_buf_bytes = bytes(mut_table.derived_state)
    arr = pa.LargeStringArray.from_buffers(
        length=mut_table.num_rows,
        value_offsets=pa.py_buffer(ds_offset_i64),
        data=pa.py_buffer(ds_buf_bytes),
    )
    derived_states = arr.to_numpy(zero_copy_only=False)
    causal_allele = derived_states[site_to_mut]

    return pd.DataFrame(
        {
            "CHR": np.full(n_sites, chrom_n, dtype=np.int8),
            "site_id": np.arange(n_sites, dtype=np.int64),
            "POS": pos,
            "AC": ac.astype(np.int32),
            "AF": af,
            "causal_allele": causal_allele,
        }
    )


def _route_logging_to_snakemake_log() -> None:
    """Route logging to snakemake.log[0].

    Snakemake's `script:` runner exposes the rule's log path via
    snakemake.log[0] but doesn't auto-redirect Python stderr to it.
    """
    log_path = snakemake.log[0] if snakemake.log else None
    if not log_path:
        return
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(log_path, mode="w")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s | %(message)s"))
    logging.getLogger().addHandler(fh)


def main() -> None:
    """Snakemake entry: load canon chrom, build catalog, dump parquet."""
    _route_logging_to_snakemake_log()
    trees_path = Path(snakemake.input.trees)
    out_path = Path(snakemake.output.catalog)
    chrom_n = int(snakemake.wildcards.n)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    log.info("loading %s (chr%d)", trees_path, chrom_n)
    t = time.perf_counter()
    ts = tskit.load(str(trees_path))
    log.info(
        "  ts: %d samples, %d sites, %d muts (load %.1fs)",
        ts.num_samples,
        ts.num_sites,
        ts.num_mutations,
        time.perf_counter() - t,
    )

    t = time.perf_counter()
    df = build_catalog(ts, chrom_n)
    log.info("  catalog: %d rows (build %.1fs)", len(df), time.perf_counter() - t)
    if len(df):
        log.info(
            "  AF: min=%.4f, mean=%.4f, max=%.4f; AC: min=%d, max=%d",
            float(df["AF"].min()),
            float(df["AF"].mean()),
            float(df["AF"].max()),
            int(df["AC"].min()),
            int(df["AC"].max()),
        )

    df.to_parquet(out_path, index=False, compression="zstd")
    log.info("wrote %s (%.1f MB)", out_path, out_path.stat().st_size / 1e6)


if __name__ == "__main__":
    main()
