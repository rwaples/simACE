"""Snakemake wrapper: concat per-chromosome tstrait site catalogs.

Reads N per-chrom catalog parquet files (one row per site with
CHR/site_id/POS/AC/AF/causal_allele), concatenates them sorted by
chromosome number, and writes a single multi-chrom parquet to
`<preprocessed>/site_catalog.parquet`. Also emits a small JSON summary.

`site_id` stays per-chromosome local — that's the index space tstrait uses
when running on a single-chrom .trees, so we keep it as-is for downstream
slicing per chromosome. The (CHR, site_id) pair is the cross-chrom unique
key.
"""

import json
import logging
import re
import time
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s")
log = logging.getLogger("tstrait_site_catalog_concat")


def _chrom_key(p: Path) -> int:
    m = re.search(r"chrom_?([0-9]+)", p.stem, re.IGNORECASE)
    if not m:
        raise ValueError(f"could not parse chromosome number from {p.name}")
    return int(m.group(1))


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
    """Snakemake entry: concat per-chrom catalogs into one parquet + summary."""
    _route_logging_to_snakemake_log()
    in_paths = sorted([Path(p) for p in snakemake.input.catalogs], key=_chrom_key)
    out_catalog = Path(snakemake.output.catalog)
    out_summary = Path(snakemake.output.summary)
    out_catalog.parent.mkdir(parents=True, exist_ok=True)

    log.info("concatenating %d per-chrom catalogs", len(in_paths))
    t = time.perf_counter()
    dfs: list[pd.DataFrame] = []
    per_chrom: list[dict] = []
    for p in in_paths:
        df = pd.read_parquet(p)
        n = len(df)
        chrom_n = int(df["CHR"].iloc[0]) if n else _chrom_key(p)
        per_chrom.append(
            {
                "chrom": chrom_n,
                "path": str(p),
                "n_sites": n,
                "af_mean": float(df["AF"].mean()) if n else None,
            }
        )
        dfs.append(df)

    catalog = pd.concat(dfs, ignore_index=True)
    catalog = catalog.sort_values(["CHR", "site_id"], kind="stable").reset_index(drop=True)
    log.info("  concatenated %d rows in %.1fs", len(catalog), time.perf_counter() - t)

    catalog.to_parquet(out_catalog, index=False, compression="zstd")
    log.info("wrote %s (%.1f MB)", out_catalog, out_catalog.stat().st_size / 1e6)

    summary = {
        "n_chromosomes": len(in_paths),
        "n_sites_total": len(catalog),
        "af_mean": float(catalog["AF"].mean()) if len(catalog) else None,
        "af_min": float(catalog["AF"].min()) if len(catalog) else None,
        "af_max": float(catalog["AF"].max()) if len(catalog) else None,
        "per_chromosome": per_chrom,
    }
    out_summary.write_text(json.dumps(summary, indent=2, default=float))
    log.info("wrote summary %s", out_summary)


if __name__ == "__main__":
    main()
