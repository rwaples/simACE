"""Per-site allele counts via an incremental edge-diff sweep.

Replaces the per-mutation ``tree.num_samples(mut.node)`` Python call with a
numba kernel that maintains ``num_samples[node]`` incrementally as edges
enter/leave each tree.  Assumes one mutation per site (canonicalized data;
``tstrait_site_catalog_chrom`` enforces with ``mut_counts == 1``).
"""

import numba
import numpy as np
import tskit
import tskit.jit.numba as tjn


@numba.njit(cache=True)
def _allele_counts_numba(
    edges_left,
    edges_right,
    edges_parent,
    edges_child,
    indexes_edge_insertion_order,
    indexes_edge_removal_order,
    breakpoints,
    site_position,
    mut_site,
    mut_node,
    samples_mask,
    num_nodes,
):
    NULL = -1
    parent = np.full(num_nodes, NULL, dtype=np.int32)
    num_samples = samples_mask.astype(np.int32)

    n_sites = site_position.shape[0]
    n_muts = mut_site.shape[0]
    M = edges_parent.shape[0]
    NT = breakpoints.shape[0] - 1
    in_idx = 0
    out_idx = 0
    mut_idx = 0
    cur_left = 0.0

    ac = np.zeros(n_sites, dtype=np.int32)

    for tree_idx in range(NT):
        while out_idx < M:
            e = indexes_edge_removal_order[out_idx]
            if edges_right[e] != cur_left:
                break
            p = edges_parent[e]
            c = edges_child[e]
            delta = num_samples[c]
            u = p
            while u != NULL:
                num_samples[u] -= delta
                u = parent[u]
            parent[c] = NULL
            out_idx += 1

        while in_idx < M:
            e = indexes_edge_insertion_order[in_idx]
            if edges_left[e] != cur_left:
                break
            p = edges_parent[e]
            c = edges_child[e]
            parent[c] = p
            delta = num_samples[c]
            u = p
            while u != NULL:
                num_samples[u] += delta
                u = parent[u]
            in_idx += 1

        cur_right = breakpoints[tree_idx + 1]

        while mut_idx < n_muts and site_position[mut_site[mut_idx]] < cur_right:
            ac[mut_site[mut_idx]] = num_samples[mut_node[mut_idx]]
            mut_idx += 1

        cur_left = cur_right

    return ac


def allele_counts(ts: tskit.TreeSequence) -> np.ndarray:
    """Per-site sample count of the (unique) derived allele."""
    samples_mask = np.zeros(ts.num_nodes, dtype=np.int32)
    samples_mask[ts.samples()] = 1
    nts = tjn.jitwrap(ts)
    return _allele_counts_numba(
        edges_left=nts.edges_left,
        edges_right=nts.edges_right,
        edges_parent=nts.edges_parent,
        edges_child=nts.edges_child,
        indexes_edge_insertion_order=nts.indexes_edge_insertion_order,
        indexes_edge_removal_order=nts.indexes_edge_removal_order,
        breakpoints=nts.breakpoints,
        site_position=ts.tables.sites.position,
        mut_site=ts.tables.mutations.site,
        mut_node=ts.tables.mutations.node,
        samples_mask=samples_mask,
        num_nodes=ts.num_nodes,
    )
