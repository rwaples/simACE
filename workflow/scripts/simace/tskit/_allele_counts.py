"""Per-site allele counts via an incremental edge-diff sweep.

The reference implementation walks every tree in the tree sequence and
calls ``tree.num_samples(mut.node)`` for each mutation. tskit caches that
count in O(1), but for ~1M mutations × 800K trees the per-call Python
overhead dominates (~14.6s on a 71K-sample / 2.7M-site real chromosome,
multiplied across 22 autosomes is ~5 min wall-clock per preprocess).

This module provides a numba kernel that maintains
``num_samples_per_node`` incrementally as edges are inserted/removed at
tree boundaries — i.e. it inlines what tskit's C code does internally,
but in numba so each per-mutation lookup becomes a numpy array index
rather than a crossed-language method call. Assumes:

- 1 mutation per site (canonicalized data; the calling script enforces
  this with an assertion).
- Mutations sorted by (site, time) — a tskit invariant.
"""

import numba
import numpy as np

NULL = -1


@numba.njit(cache=True)
def _allele_counts_numba(
    edges_parent: np.ndarray,
    edges_child: np.ndarray,
    edges_left: np.ndarray,
    edges_right: np.ndarray,
    insertion_order: np.ndarray,
    removal_order: np.ndarray,
    site_position: np.ndarray,
    mut_site: np.ndarray,
    mut_node: np.ndarray,
    samples_mask: np.ndarray,
    sequence_length: float,
) -> np.ndarray:
    n_nodes = samples_mask.shape[0]
    parent = np.full(n_nodes, NULL, dtype=np.int32)
    num_samples = samples_mask.copy().astype(np.int64)

    n_sites = site_position.shape[0]
    n_muts = mut_site.shape[0]
    n_edges = edges_parent.shape[0]

    ac = np.zeros(n_sites, dtype=np.int64)

    in_idx = 0
    out_idx = 0
    mut_idx = 0
    left = 0.0

    while left < sequence_length:
        # Apply removals: edges whose right endpoint is the current left.
        while out_idx < n_edges and edges_right[removal_order[out_idx]] == left:
            e = removal_order[out_idx]
            p = edges_parent[e]
            c = edges_child[e]
            delta = num_samples[c]
            u = p
            while u != NULL:
                num_samples[u] -= delta
                u = parent[u]
            parent[c] = NULL
            out_idx += 1

        # Apply insertions: edges whose left endpoint is the current left.
        while in_idx < n_edges and edges_left[insertion_order[in_idx]] == left:
            e = insertion_order[in_idx]
            p = edges_parent[e]
            c = edges_child[e]
            parent[c] = p
            delta = num_samples[c]
            u = p
            while u != NULL:
                num_samples[u] += delta
                u = parent[u]
            in_idx += 1

        # Determine the right boundary of the current tree.
        right = sequence_length
        if in_idx < n_edges:
            cand = edges_left[insertion_order[in_idx]]
            if cand < right:
                right = cand
        if out_idx < n_edges:
            cand = edges_right[removal_order[out_idx]]
            if cand < right:
                right = cand

        # Sweep mutations whose site_position falls in [left, right).
        while mut_idx < n_muts:
            s = mut_site[mut_idx]
            pos = site_position[s]
            if pos >= right:
                break
            ac[s] = num_samples[mut_node[mut_idx]]
            mut_idx += 1

        left = right

    return ac


def allele_counts(ts) -> np.ndarray:
    """Return per-site sample count of the (unique) derived allele.

    Equivalent to::

        for tree in ts.trees():
            for site in tree.sites():
                mut = site.mutations[0]
                ac[site.id] = tree.num_samples(mut.node)

    on canonicalized data (one mutation per site), but ~10× faster on the
    71K-sample / 2.7M-site SimHumanity chromosomes.
    """
    samples_mask = np.zeros(ts.num_nodes, dtype=np.int64)
    samples_mask[ts.samples()] = 1
    tables = ts.tables
    return _allele_counts_numba(
        tables.edges.parent,
        tables.edges.child,
        tables.edges.left,
        tables.edges.right,
        tables.indexes.edge_insertion_order,
        tables.indexes.edge_removal_order,
        tables.sites.position,
        tables.mutations.site,
        tables.mutations.node,
        samples_mask,
        ts.sequence_length,
    )
