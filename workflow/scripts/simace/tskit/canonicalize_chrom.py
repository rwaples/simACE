"""Snakemake wrapper: canonicalize one per-chromosome .trees file.

Self-contained: inlines the population filter and canonicalize logic so the
script does not depend on the gitignored `external/tskit/` checkout. Run
inside the workflow's tskit conda env via `--use-conda`.
"""

import json
import logging
import re
import time
from pathlib import Path

import numpy as np
import tskit

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s"
)
log = logging.getLogger("canonicalize_chrom")


def eligible_individuals(ts: tskit.TreeSequence, pop_name: str | None) -> np.ndarray:
    """Individual IDs eligible to serve as founders.

    If pop_name is None, all individuals with >=1 sample node are eligible.
    Otherwise restrict to individuals whose sample nodes lie in the named pop.
    """
    if pop_name is None:
        sample_inds = ts.tables.nodes.individual[ts.samples()]
        return np.unique(sample_inds[sample_inds != tskit.NULL])

    pop_id = None
    for pop in ts.populations():
        md = pop.metadata if isinstance(pop.metadata, dict) else {}
        if md.get("name") == pop_name:
            pop_id = pop.id
            break
    if pop_id is None:
        raise ValueError(f"population {pop_name!r} not found in tree sequence")

    samples = ts.samples()
    node_pop = ts.tables.nodes.population[samples]
    node_ind = ts.tables.nodes.individual[samples]
    in_pop = node_pop == pop_id
    return np.unique(node_ind[in_pop & (node_ind != tskit.NULL)])


def canonicalize(
    ts: tskit.TreeSequence, pop_name: str | None
) -> tuple[tskit.TreeSequence, dict]:
    """Sort eligible individuals by ID and simplify to a canonical sample-node order.

    After this, sample node `2*j` and `2*j+1` always belong to the j-th
    eligible individual on every chromosome processed with the same pop_name.
    """
    pre = {
        "seqlen": float(ts.sequence_length),
        "num_samples_pre": int(ts.num_samples),
        "num_individuals_pre": int(ts.num_individuals),
        "num_nodes_pre": int(ts.num_nodes),
        "num_edges_pre": int(ts.num_edges),
        "num_sites_pre": int(ts.num_sites),
        "num_mutations_pre": int(ts.num_mutations),
    }
    eligible = eligible_individuals(ts, pop_name).astype(np.int64)
    eligible.sort()

    sample_nodes: list[int] = []
    for ind_id in eligible:
        sample_nodes.extend(int(n) for n in ts.individual(int(ind_id)).nodes)
    sample_node_array = np.asarray(sample_nodes, dtype=np.int32)

    ts_c = ts.simplify(
        samples=sample_node_array,
        filter_individuals=False,
        filter_populations=False,
        filter_sites=True,
        update_sample_flags=True,
        reduce_to_site_topology=True,
        record_provenance=False,
    )
    n_sites_simplify = int(ts_c.num_sites)
    n_muts_simplify = int(ts_c.num_mutations)

    # Drop any site that doesn't have exactly one mutation (recurrent / back
    # mutations, multi-allelic sites). Leaves a clean 1-mutation-per-site
    # biallelic SNP set for downstream graft/genotype work.
    mut_counts = np.bincount(ts_c.tables.mutations.site, minlength=ts_c.num_sites)
    sites_to_drop = np.where(mut_counts != 1)[0]
    if sites_to_drop.size:
        ts_c = ts_c.delete_sites(sites_to_drop)

    post = {
        "n_eligible_individuals": len(eligible),
        "eligible_individual_ids_first10": [int(x) for x in eligible[:10]],
        "num_samples_post": int(ts_c.num_samples),
        "num_individuals_post": int(ts_c.num_individuals),
        "num_nodes_post": int(ts_c.num_nodes),
        "num_edges_post": int(ts_c.num_edges),
        "num_sites_post_simplify": n_sites_simplify,
        "num_mutations_post_simplify": n_muts_simplify,
        "num_sites_dropped_multimut": int(sites_to_drop.size),
        "num_sites_post": int(ts_c.num_sites),
        "num_mutations_post": int(ts_c.num_mutations),
    }
    return ts_c, {**pre, **post}


def _chrom_number(src: Path) -> int:
    m = re.search(r"chromosome[_\-]?([0-9]+)", src.stem, re.IGNORECASE)
    if not m:
        raise ValueError(f"could not parse chromosome number from {src.name}")
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
    """Snakemake entry: load src, canonicalize, dump canon + per-chrom stats."""
    _route_logging_to_snakemake_log()
    src = Path(snakemake.input.src)
    out_t = Path(snakemake.output.canon)
    out_s = Path(snakemake.output.stats)
    pop = snakemake.params.pop_name

    n = _chrom_number(src)
    if not 1 <= n <= 22:
        raise ValueError(
            f"non-autosomal chromosome {n} (from {src.name}); "
            "tskit_preprocess is autosomes-only — restrict config['tskit_preprocess']['chroms'] to 1..22",
        )

    out_t.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    log.info("loading %s", src)
    ts = tskit.load(str(src))
    src_size = src.stat().st_size
    log.info(
        "loaded: %d samples, %d individuals, %d sites, %d muts (%.1f MB on disk)",
        ts.num_samples,
        ts.num_individuals,
        ts.num_sites,
        ts.num_mutations,
        src_size / 1e6,
    )

    ts_c, stats = canonicalize(ts, pop)
    ts_c.dump(str(out_t))
    dst_size = out_t.stat().st_size

    stats["name"] = src.stem
    stats["src"] = str(src)
    stats["dst"] = str(out_t)
    stats["src_size_bytes"] = int(src_size)
    stats["dst_size_bytes"] = int(dst_size)
    stats["elapsed_seconds"] = time.perf_counter() - t0
    with out_s.open("w") as fh:
        json.dump(stats, fh, indent=2, default=float)

    log.info(
        "done %s in %.1fs: pre nodes=%d edges=%d sites=%d muts=%d -> "
        "post nodes=%d edges=%d sites=%d muts=%d (%.0f MB -> %.0f MB)",
        src.name,
        stats["elapsed_seconds"],
        stats["num_nodes_pre"],
        stats["num_edges_pre"],
        stats["num_sites_pre"],
        stats["num_mutations_pre"],
        stats["num_nodes_post"],
        stats["num_edges_post"],
        stats["num_sites_post"],
        stats["num_mutations_post"],
        src_size / 1e6,
        dst_size / 1e6,
    )


if __name__ == "__main__":
    main()
