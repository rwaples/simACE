"""Snakemake wrapper: per-chromosome genetic value via a numba-jitted kernel.

Loads one grafted per-chrom .trees (from `simulate_genotypes_chrom`) plus the
shared causal-effects parquet, slices the effects to this chromosome, and
computes per-individual GV with a custom single-pass numba kernel built on
`tskit.jit.numba.NumbaTreeSequence`. Writes a per-chrom GV parquet plus a
small meta JSON so each chrom's contribution is auditable in isolation.

Why not `tstrait.genetic_value`? tstrait restarts a tree-traversal per causal
site and runs an O(num_nodes) accumulate per call — total cost ~`n_causal *
num_nodes`. For our scenario (n_causal up to 1M, num_nodes ~1.1M per grafted
chrom), that's days of compute. The numba kernel iterates the trees once,
maintains left_child/right_sib incrementally as edges enter/exit, and DFS's
each causal site's mutation node into per-individual GV — total cost is
roughly `total_edges + sum(descendants_per_causal)`. ~5x faster at n=1k,
similar at n=50k, and finishes ~1M causal sites in seconds rather than hours.

Equivalence: byte-equivalent to tstrait up to float-summation order
(verified at n=1k,10k,50k with max abs diff < 3e-12).

Site-id remap. The catalog uses canonical (founder-only) site_ids; the
grafted .trees has its own consecutive site_ids that drop the ~15-20% of
sites which were lost during the pedigree drop (founder allele not
inherited by any present-day sample). We map canonical site_id -> grafted
site_id by position (positions are preserved through drop+graft). Causal
sites whose position isn't in the grafted ts are dropped with a warning.

Empty-chrom guard: if the assigned causal-site set has no sites on this
chromosome (possible at low `num_causal` or `frac_causal`, OR if every
causal site got dropped during the position remap), we synthesize a zero-GV
row for every unique present-day individual derived from `ts.samples()`.
The aggregator can then sum freely without special-casing missing chromosomes.

Genetic value is additive across chromosomes for a fixed causal-effect set,
so summing the per-chrom GVs equals the one-shot whole-genome GV — modulo
numerical floating-point order, which the test fixture verifies.

Assumes one mutation per site (canonicalize step's `mut_count != 1` filter)
so the per-site mutation node is unambiguous and equals the carrier of the
catalog's `causal_allele` (assigned from that mutation's `derived_state`).
"""

import json
import logging
import time
from pathlib import Path

import numba
import numpy as np
import pandas as pd
import tskit
import tskit.jit.numba as tjn

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s")
log = logging.getLogger("tstrait_gv_chrom")


def _individuals_from_samples(ts: tskit.TreeSequence) -> np.ndarray:
    """Unique individual IDs covered by the tree sequence's sample nodes."""
    sample_inds = ts.tables.nodes.individual[ts.samples()]
    sample_inds = sample_inds[sample_inds != tskit.NULL]
    return np.unique(sample_inds)


@numba.njit(cache=True, nogil=True)
def _gv_kernel(
    edges_left,
    edges_right,
    edges_parent,
    edges_child,
    indexes_edge_insertion_order,
    indexes_edge_removal_order,
    breakpoints,
    nodes_individual,
    causal_position,
    causal_mut_node,
    causal_effect_size,
    num_nodes,
    num_individuals,
):
    """Single-pass per-individual GV.

    Iterates trees in genome order, maintaining `left_child` / `right_sib`
    arrays incrementally as edges enter (in_range) and exit (out_range). For
    each causal site within the current tree, DFS from the mutation node and
    add `effect_size` into `gv[nodes_individual[descendant]]`. Returns a
    length-`num_individuals` GV array (caller filters to sample inds).

    `nogil=True` lets multiple chromosomes run in parallel via Python threads.
    Early-terminates after the last causal site so genomes with causal sites
    concentrated at the start don't pay the trailing edge-maintenance cost.
    """
    NULL = -1
    left_child = np.full(num_nodes, NULL, dtype=np.int32)
    right_sib = np.full(num_nodes, NULL, dtype=np.int32)
    parent = np.full(num_nodes, NULL, dtype=np.int32)
    stack = np.empty(num_nodes, dtype=np.int32)
    gv = np.zeros(num_individuals, dtype=np.float64)

    n_causal = causal_position.shape[0]
    if n_causal == 0:
        return gv
    causal_cursor = 0
    M = edges_parent.shape[0]
    NT = breakpoints.shape[0] - 1
    in_idx = 0
    out_idx = 0
    cur_left = 0.0

    for tree_idx in range(NT):
        # Drop edges whose right == cur_left
        while out_idx < M:
            e = indexes_edge_removal_order[out_idx]
            if edges_right[e] != cur_left:
                break
            p_node = edges_parent[e]
            c_node = edges_child[e]
            prev = NULL
            cur = left_child[p_node]
            while cur != NULL and cur != c_node:
                prev = cur
                cur = right_sib[cur]
            if cur == c_node:
                if prev == NULL:
                    left_child[p_node] = right_sib[c_node]
                else:
                    right_sib[prev] = right_sib[c_node]
            right_sib[c_node] = NULL
            parent[c_node] = NULL
            out_idx += 1

        # Insert edges whose left == cur_left
        while in_idx < M:
            e = indexes_edge_insertion_order[in_idx]
            if edges_left[e] != cur_left:
                break
            p_node = edges_parent[e]
            c_node = edges_child[e]
            right_sib[c_node] = left_child[p_node]
            left_child[p_node] = c_node
            parent[c_node] = p_node
            in_idx += 1

        cur_right = breakpoints[tree_idx + 1]

        # Process causal sites within [cur_left, cur_right)
        while causal_cursor < n_causal and causal_position[causal_cursor] < cur_right:
            mut_node = causal_mut_node[causal_cursor]
            eff = causal_effect_size[causal_cursor]
            stack[0] = mut_node
            sp = 1
            while sp > 0:
                sp -= 1
                node = stack[sp]
                ind = nodes_individual[node]
                if ind != NULL:
                    gv[ind] += eff
                c = left_child[node]
                while c != NULL:
                    stack[sp] = c
                    sp += 1
                    c = right_sib[c]
            causal_cursor += 1

        # Early-terminate once all causal sites processed; trailing trees
        # would only do edge maintenance with no DFS payoff.
        if causal_cursor >= n_causal:
            return gv

        cur_left = cur_right

    return gv


def compute_gv(ts: tskit.TreeSequence, chrom_effects: pd.DataFrame, trait_id: int) -> tuple[pd.DataFrame, float]:
    """Run the numba kernel and return (sample-filtered GV df, elapsed seconds).

    Assumes `chrom_effects` is sorted by `site_id` (== sorted by position
    after remap), one mutation per site, and `causal_allele` matches that
    mutation's `derived_state` (true by construction in this pipeline).
    """
    site_ids = chrom_effects["site_id"].to_numpy(dtype=np.int64)
    # Each site has 1 mutation (canonicalize filter). Build site -> mut_node directly.
    mut_table = ts.tables.mutations
    site_to_mut = np.empty(ts.num_sites, dtype=np.int64)
    site_to_mut[mut_table.site] = np.arange(mut_table.num_rows)
    causal_mut_node = mut_table.node[site_to_mut[site_ids]].astype(np.int32)
    causal_position = ts.tables.sites.position[site_ids]
    causal_effect = chrom_effects["effect_size"].to_numpy(dtype=np.float64)

    nts = tjn.jitwrap(ts)
    t = time.perf_counter()
    gv = _gv_kernel(
        edges_left=nts.edges_left,
        edges_right=nts.edges_right,
        edges_parent=nts.edges_parent,
        edges_child=nts.edges_child,
        indexes_edge_insertion_order=nts.indexes_edge_insertion_order,
        indexes_edge_removal_order=nts.indexes_edge_removal_order,
        breakpoints=nts.breakpoints,
        nodes_individual=nts.nodes_individual,
        causal_position=causal_position,
        causal_mut_node=causal_mut_node,
        causal_effect_size=causal_effect,
        num_nodes=ts.num_nodes,
        num_individuals=ts.num_individuals,
    )
    elapsed = time.perf_counter() - t

    sample_inds = _individuals_from_samples(ts)
    df = pd.DataFrame(
        {
            "trait_id": np.full(len(sample_inds), trait_id, dtype=np.int64),
            "individual_id": sample_inds.astype(np.int64),
            "genetic_value": gv[sample_inds],
        }
    )
    return df, elapsed


def remap_effects_by_position(chrom_effects: pd.DataFrame, ts: tskit.TreeSequence) -> tuple[pd.DataFrame, int]:
    """Map canonical site_ids in chrom_effects -> grafted site_ids by position.

    Returns ``(remapped, n_dropped)``. Causal sites whose POS is not present
    in ``ts`` are dropped from the output and counted in ``n_dropped``.
    The returned DataFrame is sorted by the new site_id and reset-indexed.
    """
    if len(chrom_effects) == 0:
        return chrom_effects.copy(), 0
    graft_pos = np.asarray(ts.tables.sites.position).astype(np.int64)
    pos_to_site = pd.Series(np.arange(len(graft_pos), dtype=np.int64), index=graft_pos)
    canonical_pos = chrom_effects["POS"].to_numpy().astype(np.int64)
    new_site_ids = pos_to_site.reindex(canonical_pos).to_numpy()
    survived = ~np.isnan(new_site_ids)
    n_dropped = int((~survived).sum())
    out = chrom_effects.loc[survived].copy()
    out["site_id"] = new_site_ids[survived].astype(np.int64)
    out = out.sort_values("site_id", kind="stable").reset_index(drop=True)
    return out, n_dropped


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
    """Snakemake entry: per-chrom GV with empty-chrom guard."""
    _route_logging_to_snakemake_log()
    trees_path = Path(snakemake.input.trees)
    effects_path = Path(snakemake.input.effects)
    out_gv = Path(snakemake.output.gv)
    out_meta = Path(snakemake.output.meta)
    chrom_n = int(snakemake.wildcards.n)
    trait_id = int(snakemake.params.trait_id)
    out_gv.parent.mkdir(parents=True, exist_ok=True)

    log.info("loading trees %s (chr%d)", trees_path, chrom_n)
    t = time.perf_counter()
    ts = tskit.load(str(trees_path))
    log.info(
        "  ts: %d samples, %d sites, %d muts (load %.1fs)",
        ts.num_samples,
        ts.num_sites,
        ts.num_mutations,
        time.perf_counter() - t,
    )

    log.info("loading effects %s", effects_path)
    effects = pd.read_parquet(effects_path)
    chrom_effects = effects[effects["CHR"] == chrom_n].copy()
    log.info("  %d / %d causal sites on chr%d (pre-remap)", len(chrom_effects), len(effects), chrom_n)

    chrom_effects, n_dropped_remap = remap_effects_by_position(chrom_effects, ts)
    if n_dropped_remap:
        log.warning(
            "  %d causal sites on chr%d not in grafted ts (lost during drop) — dropping",
            n_dropped_remap,
            chrom_n,
        )

    if len(chrom_effects) == 0:
        log.warning("no usable causal sites on chr%d — emitting zero GVs", chrom_n)
        sample_inds = _individuals_from_samples(ts)
        gv = pd.DataFrame(
            {
                "trait_id": np.full(len(sample_inds), trait_id, dtype=np.int64),
                "individual_id": sample_inds.astype(np.int64),
                "genetic_value": np.zeros(len(sample_inds), dtype=np.float64),
            }
        )
    else:
        gv, elapsed = compute_gv(ts, chrom_effects, trait_id)
        log.info("  computed GV in %.1fs (%d sample inds)", elapsed, len(gv))

    gv.to_parquet(out_gv, index=False, compression="zstd")

    sum_effect_squared = float(np.sum(chrom_effects["effect_size"].to_numpy() ** 2)) if len(chrom_effects) else 0.0
    meta = {
        "chrom": chrom_n,
        "n_causal_on_chrom": len(chrom_effects),
        "n_dropped_in_remap": n_dropped_remap,
        "n_individuals": len(gv),
        "mean_gv": float(gv["genetic_value"].mean()),
        "var_gv": float(gv["genetic_value"].var(ddof=0)),
        "sum_effect_squared": sum_effect_squared,
        "trait_id": trait_id,
    }
    out_meta.write_text(json.dumps(meta, indent=2, default=float))
    log.info(
        "wrote %s (mean=%.4f, var=%.4f) and %s",
        out_gv,
        meta["mean_gv"],
        meta["var_gv"],
        out_meta,
    )


if __name__ == "__main__":
    main()
