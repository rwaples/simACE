"""Snakemake wrapper: streaming concat of canonicalized per-chrom .trees files.

Self-contained: inlines `stream_concat`, `natural_sort_key`, and the
cross-chrom consistency check so the script does not depend on the gitignored
`external/tskit/` checkout. Memory footprint stays at ~2 tree sequences
regardless of chromosome count.

Also writes a sidecar `<out_trees>.fingerprint` (BLAKE2b over the concatenated
key table column buffers) for the verify rule's corruption check.
"""

import hashlib
import json
import logging
import re
import time
from pathlib import Path

import numpy as np
import tskit

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s")
log = logging.getLogger("concat_chroms")


def natural_sort_key(p: Path) -> tuple:
    """Sort chromosome filenames in human-natural order: 1..22, X, Y, MT."""
    m = re.search(r"chromosome[_\-]?([0-9]+|X|Y|MT|M)", p.stem, re.IGNORECASE)
    if not m:
        return (99, 99, p.name)
    tag = m.group(1).upper()
    if tag.isdigit():
        return (0, int(tag), p.name)
    order = {"X": 23, "Y": 24, "M": 25, "MT": 25}
    return (0, order.get(tag, 99), p.name)


def stream_concat(canonical_paths: list[Path], out_path: Path) -> dict:
    """Concatenate canonicalized per-chrom .trees into a single multi-chrom .trees.

    Loads one chrom at a time. Memory stays at ~2 tree sequences (running
    concat tables + the chrom currently being appended). Returns summary stats.
    """
    if not canonical_paths:
        raise ValueError("canonical_paths is empty")

    ts0 = tskit.load(str(canonical_paths[0]))
    base_samples = np.asarray(ts0.samples())
    n_initial = len(base_samples)
    expected = np.arange(n_initial, dtype=base_samples.dtype)
    if not np.array_equal(base_samples, expected):
        raise RuntimeError(f"canonicalized chrom 0 still has non-trivial sample IDs: {base_samples[:5]}...")

    tables = ts0.dump_tables()
    cum_offset = float(ts0.sequence_length)
    log.info(
        "  appended %s: cum_seqlen=%.1f Mb, tables nodes=%d edges=%d sites=%d muts=%d",
        canonical_paths[0].name,
        cum_offset / 1e6,
        tables.nodes.num_rows,
        tables.edges.num_rows,
        tables.sites.num_rows,
        tables.mutations.num_rows,
    )
    del ts0

    for p in canonical_paths[1:]:
        ts = tskit.load(str(p))
        s = np.asarray(ts.samples())
        if len(s) != n_initial or not np.array_equal(s, expected):
            raise RuntimeError(f"sample-node mismatch in {p.name}: expected 0..{n_initial - 1}, got {s[:5]}...")
        node_offset = tables.nodes.num_rows - n_initial
        site_id_offset = tables.sites.num_rows
        mut_id_offset = tables.mutations.num_rows

        sn, se, ss, sm = ts.tables.nodes, ts.tables.edges, ts.tables.sites, ts.tables.mutations

        if ts.num_nodes > n_initial:
            nm_off = sn.metadata_offset
            tables.nodes.append_columns(
                flags=sn.flags[n_initial:],
                time=sn.time[n_initial:],
                population=sn.population[n_initial:],
                individual=sn.individual[n_initial:],
                metadata=sn.metadata[nm_off[n_initial] :],
                metadata_offset=nm_off[n_initial:] - nm_off[n_initial],
            )

        new_parent = np.where(se.parent < n_initial, se.parent, se.parent + node_offset)
        new_child = np.where(se.child < n_initial, se.child, se.child + node_offset)
        tables.edges.append_columns(
            left=se.left + cum_offset,
            right=se.right + cum_offset,
            parent=new_parent,
            child=new_child,
            metadata=se.metadata,
            metadata_offset=se.metadata_offset,
        )

        tables.sites.append_columns(
            position=ss.position + cum_offset,
            ancestral_state=ss.ancestral_state,
            ancestral_state_offset=ss.ancestral_state_offset,
            metadata=ss.metadata,
            metadata_offset=ss.metadata_offset,
        )

        new_mut_node = np.where(sm.node < n_initial, sm.node, sm.node + node_offset)
        new_mut_parent = np.where(sm.parent == tskit.NULL, tskit.NULL, sm.parent + mut_id_offset)
        tables.mutations.append_columns(
            site=sm.site + site_id_offset,
            node=new_mut_node,
            derived_state=sm.derived_state,
            derived_state_offset=sm.derived_state_offset,
            parent=new_mut_parent,
            metadata=sm.metadata,
            metadata_offset=sm.metadata_offset,
            time=sm.time,
        )
        cum_offset += float(ts.sequence_length)
        log.info(
            "  appended %s: cum_seqlen=%.1f Mb, tables nodes=%d edges=%d sites=%d muts=%d",
            p.name,
            cum_offset / 1e6,
            tables.nodes.num_rows,
            tables.edges.num_rows,
            tables.sites.num_rows,
            tables.mutations.num_rows,
        )
        del ts

    tables.sequence_length = cum_offset
    tables.sort()
    ts_all = tables.tree_sequence()
    ts_all.dump(str(out_path))

    return ts_all, {
        "total_seqlen": cum_offset,
        "num_samples": int(ts_all.num_samples),
        "num_individuals": int(ts_all.num_individuals),
        "num_nodes": int(ts_all.num_nodes),
        "num_edges": int(ts_all.num_edges),
        "num_sites": int(ts_all.num_sites),
        "num_mutations": int(ts_all.num_mutations),
        "num_trees": int(ts_all.num_trees),
        "out_size_bytes": int(out_path.stat().st_size),
    }


# Columns hashed for the sidecar fingerprint. Choice: every numeric column on
# the four most load-bearing tables (nodes/edges/sites/mutations) plus the
# individual table. Catches any structural drift; skips per-row metadata
# blobs so the hash is portable across tskit versions that may pad metadata.
_HASH_TABLES = (
    ("nodes", ("flags", "time", "population", "individual")),
    ("edges", ("left", "right", "parent", "child")),
    ("sites", ("position",)),
    ("mutations", ("site", "node", "parent", "time")),
    ("individuals", ("flags",)),
)


def fingerprint_tables(ts: tskit.TreeSequence) -> str:
    """BLAKE2b-128 hash over key table columns.

    Stable across re-runs of the same input data; sensitive to any structural
    change in nodes/edges/sites/mutations/individuals.
    """
    h = hashlib.blake2b(digest_size=16)
    tables = ts.tables
    for table_name, cols in _HASH_TABLES:
        table = getattr(tables, table_name)
        h.update(table_name.encode())
        h.update(int(table.num_rows).to_bytes(8, "little"))
        for col in cols:
            arr = np.ascontiguousarray(getattr(table, col))
            h.update(col.encode())
            h.update(arr.tobytes())
    h.update(b"sequence_length")
    h.update(np.float64(ts.sequence_length).tobytes())
    return h.hexdigest()


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
    """Snakemake entry: cross-chrom check + concat + summary + sidecar fingerprint."""
    _route_logging_to_snakemake_log()
    canon_paths = sorted([Path(p) for p in snakemake.input.canon], key=natural_sort_key)
    stat_paths = sorted([Path(p) for p in snakemake.input.stats], key=natural_sort_key)
    out_trees = Path(snakemake.output.trees)
    out_summary = Path(snakemake.output.summary)
    out_fp = Path(snakemake.output.fingerprint)
    out_trees.parent.mkdir(parents=True, exist_ok=True)

    per_chrom = [json.loads(p.read_text()) for p in stat_paths]
    if not per_chrom:
        raise RuntimeError("no per-chrom stats inputs")

    base = per_chrom[0]
    for r in per_chrom[1:]:
        if (
            r["n_eligible_individuals"] != base["n_eligible_individuals"]
            or r["eligible_individual_ids_first10"] != base["eligible_individual_ids_first10"]
        ):
            raise RuntimeError(
                f"eligible-individual mismatch between {r['name']} and {base['name']}: "
                f"n={r['n_eligible_individuals']} vs {base['n_eligible_individuals']}, "
                f"first10={r['eligible_individual_ids_first10']} vs {base['eligible_individual_ids_first10']}"
            )

    log.info("concatenating %d chromosomes -> %s", len(canon_paths), out_trees)
    t0 = time.perf_counter()
    ts_all, concat_stats = stream_concat(canon_paths, out_trees)
    elapsed_concat = time.perf_counter() - t0
    log.info(
        "  concat in %.1fs: total seqlen=%.1f Mb, sites=%d, muts=%d, file=%.1f MB",
        elapsed_concat,
        concat_stats["total_seqlen"] / 1e6,
        concat_stats["num_sites"],
        concat_stats["num_mutations"],
        concat_stats["out_size_bytes"] / 1e6,
    )

    log.info("computing sidecar fingerprint")
    t = time.perf_counter()
    fp_hex = fingerprint_tables(ts_all)
    out_fp.write_text(fp_hex + "\n")
    log.info("  fingerprint blake2b-128 = %s (%.1fs)", fp_hex, time.perf_counter() - t)
    del ts_all

    chrom_offsets = []
    cum = 0.0
    for r in per_chrom:
        chrom_offsets.append({"name": r["name"], "offset": cum, "length": r["seqlen"]})
        cum += r["seqlen"]

    summary = {
        "args": {
            "source_dir": str(snakemake.params.source_dir),
            "out_trees": str(out_trees),
            "pop": snakemake.params.pop_name,
            "chroms": list(snakemake.params.chroms),
        },
        "n_chromosomes": len(canon_paths),
        "consistency": {
            "eligible_individuals_consistent": True,
            "n_eligible_individuals": int(base["n_eligible_individuals"]),
        },
        "totals": concat_stats,
        "fingerprint_blake2b_128": fp_hex,
        "chromosomes_offset_table": chrom_offsets,
        "per_chromosome": per_chrom,
        "elapsed_seconds": time.perf_counter() - t0,
    }
    out_summary.write_text(json.dumps(summary, indent=2, default=float))
    log.info("saved summary -> %s", out_summary)


if __name__ == "__main__":
    main()
